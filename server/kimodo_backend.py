"""The Kimodo model server: inference on the shared fxmotion server.

Runs in the kimodo:1.0 image (docker-compose.bridge.yaml), which provides
torch and the kimodo package. Everything that needs them is imported inside a
method, so this module imports, and the MOCK_MODE server runs, without them.
The pure conversions live in kimodo_adapter.py.

    uvicorn kimodo_backend:make_app --factory --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

import fxmotion_server as fs
import kimodo_adapter as ka

log = logging.getLogger("kimodo_backend")
logging.basicConfig(level=logging.INFO)

MODELS = ("Kimodo-SOMA-RP-v1.1", "Kimodo-SOMA-SEED-v1.1", "Kimodo-SOMA-RP-v1")


def _build_initial_motion(cf, model):
    """Motion features for `continue_from`, so the first segment continues
    rather than starts.

    The NPZ carries 77-joint local rotations; the model works on its own smaller
    skeleton, and `from_SOMASkeleton77` is the exact inverse of the conversion
    used on output, so the round-trip is lossless. Root positions are skeleton
    independent.
    """
    if not cf:
        return None
    import torch

    skeleton = model.skeleton
    device = skeleton.device
    lr = torch.tensor(cf["local_rot_mats"], dtype=torch.float32, device=device)
    rp = torch.tensor(cf["root_positions"], dtype=torch.float32, device=device)
    if lr.ndim != 4 or rp.ndim != 2 or lr.shape[0] != rp.shape[0]:
        raise ValueError(
            "continue_from needs local_rot_mats [n,J,3,3] and root_positions [n,3]"
        )
    if lr.shape[1] != skeleton.nbjoints:
        lr = skeleton.from_SOMASkeleton77(lr)
    feats = model.motion_rep(lr[None], rp[None], to_normalize=False)
    log.info("[CONT] seeding %d frames of motion", feats.shape[1])
    return feats


def _build_constraints(constraints, model) -> list:
    """Turn the request's constraint dicts into Kimodo constraint objects.

    Standard Kimodo dicts (root2d / fullbody / end-effector with local
    axis-angle) go through load_constraints_lst. The bridge also accepts two
    "global" dict types authored from posed Houdini geometry, the
    `fullbody-global` and `ee-global` types. They carry global joint
    positions and rotation matrices, built via the constraint constructors
    (the path Kimodo's own demo uses), avoiding any local / rest-pose convention
    round-trip on the client side."""
    if not constraints:
        return []
    import torch
    from kimodo.constraints import (
        EndEffectorConstraintSet,
        FullBodyConstraintSet,
        load_constraints_lst,
    )

    skeleton = model.skeleton
    device = skeleton.device
    # The HDA sends SOMA77 (77-joint) global data, but a SOMA-RP model
    # constrains on its smaller model skeleton (e.g. SOMASkeleton30). Map 77 ->
    # the model joint set/ order the same way Kimodo's demo does (get_skel_slice
    # against the 77 skeleton).
    src77 = getattr(skeleton, "somaskel77", None)
    skel_slice = (
        skeleton.get_skel_slice(src77)
        if src77 is not None
        and getattr(src77, "nbjoints", None) != skeleton.nbjoints
        else None
    )
    std, extra = [], []
    for c in constraints:
        t = c.get("type")
        if t in ("fullbody-global", "ee-global"):
            pos = torch.tensor(
                c["global_joints_positions"], dtype=torch.float32, device=device
            )
            rot = torch.tensor(
                c["global_joints_rots"], dtype=torch.float32, device=device
            )
            if skel_slice is not None and pos.shape[1] != skeleton.nbjoints:
                pos, rot = pos[:, skel_slice], rot[:, skel_slice]
            fi = torch.tensor(c["frame_indices"])
            sr = c.get("smooth_root_2d")
            sr = (
                torch.tensor(sr, dtype=torch.float32, device=device)
                if sr
                else None
            )
            if t == "fullbody-global":
                extra.append(
                    FullBodyConstraintSet(
                        skeleton, fi, pos, rot, smooth_root_2d=sr
                    )
                )
            else:
                extra.append(
                    EndEffectorConstraintSet(
                        skeleton, fi, pos, rot, sr, joint_names=c["joint_names"]
                    )
                )
        else:
            std.append(c)
    return (load_constraints_lst(std, skeleton) if std else []) + extra


class _Sampler:
    """Stand-in for tqdm: Kimodo wraps one denoising loop per segment with
    progress_bar(indices). A segment costs roughly 30 s of text encoding (CPU,
    reports nothing), ~8 s of denoising (GPU) and a few seconds of
    post-processing, so the fraction is (finished loops + position in the
    current loop) / segments, and the encode phase is announced with an
    estimate the server's display creeps across. The first measured encode
    replaces the estimate for every later job."""

    def __init__(self, backend, progress, expected: int):
        self.backend, self.progress = backend, progress
        self.expected, self.done = max(1, expected), 0
        self._encoding_since = None
        self._encode()

    def _encode(self) -> None:
        self._encoding_since = time.monotonic()
        self.progress.set(
            self.done / self.expected,
            "encoding text",
            est_s=self.backend.encode_est_s,
            span=1.0 / self.expected,
        )

    def __call__(self, iterable, **_):
        if self._encoding_since is not None:
            took = time.monotonic() - self._encoding_since
            self.backend.encode_est_s = max(0.1, took)
            self._encoding_since = None
        items = list(iterable)
        n = max(1, len(items))
        phase = "denoising segment %d/%d" % (self.done + 1, self.expected)
        for i, item in enumerate(items):
            self.progress.set((self.done + i / n) / self.expected, phase)
            yield item
        self.done += 1
        if self.done >= self.expected:
            self.progress.set(1.0, "post-processing")
        else:
            self._encode()


def _segments(inp: ka.KimodoInputs) -> list:
    out, start = [], 0
    for text, n in zip(inp.texts, inp.num_frames, strict=True):
        out.append({"prompt": text, "start": start, "end": start + n - 1})
        start += n
    return out


class KimodoBackend(fs.Backend):
    name = "kimodo"
    skeleton = "soma77"
    fps = ka.FPS
    models = MODELS
    capabilities = {
        "text": True,
        "styles": [],
        "root_path": True,
        "keyframes": True,
        "effectors": True,
        "continue": True,
        "live": False,
    }

    def __init__(self, default_model="", mock_clip=None, encode_est_s=30.0):
        self.default_model = default_model or MODELS[0]
        self.mock_clip = Path(mock_clip) if mock_clip else None
        self.encode_est_s = float(encode_est_s)
        self._model = None

    def validate(self, req) -> None:
        ka.kimodo_inputs(req.model_dump())  # AdapterError is a ValueError

    def load(self, model: str) -> None:
        import torch
        from kimodo import load_model
        from kimodo.model.registry import resolve_model_name

        key = resolve_model_name(model or "", default_family="Kimodo")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("[RESIDENT] loading model %s on %s ...", key, device)
        self._model, resolved = load_model(
            key,
            device=device,
            default_family="Kimodo",
            return_resolved_name=True,
        )
        log.info("[RESIDENT] model ready: %s", resolved)

    def unload(self) -> None:
        self._model = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, req, progress) -> dict:
        inp = ka.kimodo_inputs(req.model_dump())
        model = self._model
        if req.seed is not None:
            import torch

            torch.manual_seed(int(req.seed))
        constraint_lst = _build_constraints(inp.constraints, model)
        initial_motion = _build_initial_motion(inp.continue_from, model)
        sampler = _Sampler(self, progress, len(inp.texts))
        # Kimodo's multi-prompt path does not forward progress_bar to the
        # sampling loop (kimodo_model._multiprompt calls self._generate
        # without it), so inject it; inference is serialised by the server,
        # so patching the resident model is safe.
        orig = model._generate

        def _with_progress(*a, **k):
            k.setdefault("progress_bar", sampler)
            return orig(*a, **k)

        model._generate = _with_progress
        try:
            output = model(
                inp.texts,
                inp.num_frames,
                num_denoising_steps=100,
                num_samples=1,
                multi_prompt=True,
                num_transition_frames=inp.transition_frames,
                initial_motion=initial_motion,
                post_processing=True,
                constraint_lst=constraint_lst,
                return_numpy=True,
                progress_bar=sampler,
            )
        finally:
            del model._generate  # back to the class method
        n = int(output["posed_joints"].shape[0])
        single = {
            k: (
                v[0]
                if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n
                else v
            )
            for k, v in output.items()
        }
        return ka.to_clip(
            single,
            inp.canon,
            segments=_segments(inp),
            source={"model": req.model or self.default_model, "seed": req.seed},
        )

    def mock(self, req) -> dict:
        inp = ka.kimodo_inputs(req.model_dump())
        if self.mock_clip is None or not self.mock_clip.exists():
            raise FileNotFoundError(
                "MOCK_MODE serves a Kimodo NPZ, none found at %s"
                % self.mock_clip
            )
        with np.load(self.mock_clip) as z:
            npz = {k: z[k] for k in z.files}
        return ka.to_clip(
            npz,
            inp.canon,
            segments=_segments(inp),
            source={"model": "mock", "seed": req.seed},
        )


def make_app():
    out = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
    backend = KimodoBackend(
        default_model=os.environ.get("KIMODO_MODEL", ""),
        mock_clip=os.environ.get(
            "FXMOTION_MOCK_CLIP", str(out / "dev_reference.npz")
        ),
        encode_est_s=float(os.environ.get("KIMODO_ENCODE_EST_S", "30")),
    )
    return fs.create_app(
        backend,
        output_dir=out,
        mock=os.environ.get("MOCK_MODE", "1") == "1",
        idle_unload_s=float(os.environ.get("FXMOTION_IDLE_UNLOAD_S", "0")),
        preload=True,
    )
