"""The ARDY model server: ARDY Core on the shared fxmotion server.

Runs natively in ARDY's own venv (scripts/run_ardy_server.ps1). The text
encoder is the Kimodo text-encoder container's (same LLM2Vec model and API,
TEXT_ENCODER_MODE=api), so the two models share one copy. Every torch and
ardy import is inside a method: the module imports, and the MOCK_MODE server
runs, without them.

The timeline is generated one horizon at a time, like ARDY's interactive
demo: each chunk sees the last few seconds of motion as
init_history_sequence (HISTORY_S) and the prompt of the segment it falls in. A whole
clip as history would outgrow ARDY's trained 10 s window and jitter (ARDY
refuses crop_history_length together with an explicit history).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import diffusion_adapter as da
import numpy as np
import requests

import fxmotion_server as fs

log = logging.getLogger("ardy_backend")
logging.basicConfig(level=logging.INFO)

ARDY = da.ModelSpec(name="ardy", skeleton="ardy_core", fps=20.0)
MODELS = ("ARDY-Core-RP-20FPS-Horizon40", "ARDY-Core-RP-20FPS-Horizon8")
# ARDY's trained context; past it, ARDY's own code warns the motion degrades
WINDOW_S = 10.0


# History each chunk sees by default. Measured on walk / turn / wave
# timelines: the full 8 s ARDY allows follows a late prompt change on 1 seed
# in 4, 4 s on 3 in 4 (seams up to about 2x the median step), 2 s on 4 in 4
# but with harder seams. Requests override it with options.history_s.
HISTORY_S = 4.0


def history_frames(
    fps: float, horizon: int, token: int, seconds=HISTORY_S
) -> int:
    """History frames for `seconds`, in whole tokens, at least one token and
    at most what fits the trained window next to one horizon (the bound of
    scripts/generate.py's _default_history_frames)."""
    window = (int(WINDOW_S * fps) // token) * token
    most = ((window - horizon) // token) * token
    want = (int(round(float(seconds) * fps)) // token) * token
    return min(most, max(token, want))


def chunks(num_frames, horizon: int, token: int) -> list:
    """[(segment index, first frame, frame count)]: each segment cut into
    pieces of at most one horizon. ARDY reads a history in whole tokens, so
    the segment boundaries inside the clip move to the nearest token (a
    segment shorter than half a token disappears) and every chunk but the
    last is a whole number of tokens."""
    total = sum(num_frames)
    ends, cum = [], 0
    for n in num_frames[:-1]:
        cum += n
        ends.append(min(total, (cum + token // 2) // token * token))
    ends.append(total)
    out, start = [], 0
    for k, end in enumerate(ends):
        for s in range(start, end, horizon):
            out.append((k, s, min(horizon, end - s)))
        start = max(start, end)
    return out


# (text, constraint) guidance, scripts/generate.py's default. A bare float
# means text guidance only: ARDY then ignores root paths and pose keys.
CFG_WEIGHT = (2.0, 2.0)

# Each chunk also generates up to one more horizon, so it sees the pose keys
# and waypoints just past it (ARDY's own calls see the whole clip's); the
# extra frames are dropped and generated again by the next chunk.
LOOKAHEAD = True


def lookahead(
    history: int, n: int, horizon: int, window: int, remaining: int
) -> int:
    """Frames generated past a chunk: one horizon, within the trained window
    and the clip."""
    return max(0, min(horizon, window - history - n, remaining))


def encode_prompts(encode, texts, url) -> dict:
    """{prompt: encode([prompt])}, once per distinct prompt. A failure while
    the text encoder is gone is re-raised naming it (the container can stop
    after the model loaded)."""
    try:
        return {t: encode([t]) for t in dict.fromkeys(texts)}
    except Exception:
        check_text_encoder(url)
        raise


def check_text_encoder(url: str, timeout: float = 3.0) -> None:
    """Raise a RuntimeError naming the URL when the text-encoder service
    (the Kimodo text-encoder container) does not answer."""
    try:
        requests.get(
            url.rstrip("/") + "/config", timeout=timeout
        ).raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            "ardy: text encoder unreachable at %s; start the Kimodo "
            "text-encoder container (docker compose -f "
            "docker-compose.bridge.yaml up text-encoder -d): %s" % (url, e)
        ) from e


def effector_joints(names) -> list:
    """An end-effector set's joints, with the Hips ARDY needs on every
    position-constrained frame (as its own LeftHandConstraintSet)."""
    names = list(names)
    return names if "Hips" in names else names + ["Hips"]


def _constraint_objects(dicts, skeleton) -> list:
    """Model-space constraint dicts (the shared adapter's output) as ARDY
    constraint sets. root2d and ARDY's own dicts go through
    load_constraints_lst; the global types through the constructors, as
    Kimodo's backend does."""
    import torch
    from ardy.constraints import (
        EndEffectorConstraintSet,
        FullBodyConstraintSet,
        load_constraints_lst,
    )

    device = skeleton.device
    std, out = [], []
    for c in dicts:
        kind = c.get("type")
        if kind in ("fullbody-global", "ee-global"):
            fi = torch.tensor(c["frame_indices"])
            pos = torch.tensor(
                c["global_joints_positions"], dtype=torch.float32, device=device
            )
            rot = torch.tensor(
                c["global_joints_rots"], dtype=torch.float32, device=device
            )
            root = c.get("smooth_root_2d")
            root = (
                torch.tensor(root, dtype=torch.float32, device=device)
                if root
                else None
            )
            if kind == "fullbody-global":
                out.append(
                    FullBodyConstraintSet(skeleton, fi, pos, rot, root_2d=root)
                )
            else:
                out.append(
                    EndEffectorConstraintSet(
                        skeleton,
                        fi,
                        pos,
                        rot,
                        root,
                        joint_names=effector_joints(c["joint_names"]),
                    )
                )
        else:
            std.append(c)
    return (load_constraints_lst(std, skeleton) if std else []) + out


class ArdyBackend(fs.Backend):
    name = "ardy"
    skeleton = "ardy_core"
    fps = ARDY.fps
    models = MODELS
    capabilities = {
        "text": True,
        "styles": [],
        "root_path": True,
        "keyframes": True,
        "effectors": True,
        "continue": False,
        "live": False,
    }

    def __init__(
        self,
        default_model="",
        mock_clip=None,
        text_encoder_url="http://127.0.0.1:9550/",
    ):
        self.default_model = default_model or MODELS[0]
        self.mock_clip = Path(mock_clip) if mock_clip else None
        self.text_encoder_url = text_encoder_url
        self._model = None
        self._device = None

    def validate(self, req) -> None:
        da.model_inputs(req.model_dump(), ARDY)  # AdapterError is a ValueError

    def load(self, model: str) -> None:
        import torch
        from ardy.model.load_model import load_model
        from ardy.model.registry import resolve_model_name

        check_text_encoder(self.text_encoder_url)
        os.environ.setdefault("TEXT_ENCODER_MODE", "api")
        os.environ.setdefault("TEXT_ENCODER_URL", self.text_encoder_url)
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        name = resolve_model_name(model or self.default_model)
        log.info("[RESIDENT] loading %s on %s ...", name, self._device)
        self._model = load_model(name, device=self._device)
        log.info("[RESIDENT] model ready: %s", name)

    def unload(self) -> None:
        self._model = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, req, progress) -> dict:
        import torch
        from ardy.motion_rep.tools import length_to_mask
        from ardy.postprocess import post_process_motion
        from ardy.tools import seed_everything, to_numpy

        inp = da.model_inputs(req.model_dump(), ARDY)
        model, device = self._model, self._device
        if req.seed is not None:
            seed_everything(int(req.seed))
        total = sum(inp.num_frames)
        constraints = _constraint_objects(inp.constraints, model.skeleton)
        observed = mask = None
        if constraints:
            observed, mask = (
                model.motion_rep.create_conditions_from_constraints_batched(
                    constraints,
                    torch.tensor([total], device=device),
                    to_normalize=True,
                    device=device,
                )
            )
        steps = int(model.diffusion.num_base_steps)
        horizon = int(model.gen_horizon_len)
        token = int(model.num_frames_per_token)
        keep = history_frames(
            ARDY.fps, horizon, token, req.options.get("history_s", HISTORY_S)
        )
        window_frames = (int(WINDOW_S * ARDY.fps) // token) * token
        texts = encode_prompts(
            model._encode_text, inp.texts, self.text_encoder_url
        )
        plan = chunks(inp.num_frames, horizon, token)
        motion = None
        for i, (k, start, n) in enumerate(plan):
            progress.set(
                0.9 * i / len(plan), "segment %d/%d" % (k + 1, len(inp.texts))
            )
            hist = None if motion is None else motion[:, -keep:]
            h = 0 if hist is None else hist.shape[1]
            la = (
                lookahead(h, n, horizon, window_frames, total - start - n)
                if LOOKAHEAD
                else 0
            )
            window = slice(start - h, start + n + la)
            feat, pad = texts[inp.texts[k]]
            with torch.no_grad():
                out = model(
                    [inp.texts[k]],
                    h + n + la,
                    num_denoising_steps=steps,
                    cfg_weight=CFG_WEIGHT,
                    pad_mask=length_to_mask(
                        torch.tensor([h + n + la], device=device)
                    ),
                    first_heading_angle=(
                        torch.zeros(1, device=device) if hist is None else None
                    ),
                    motion_mask=None if mask is None else mask[:, window],
                    observed_motion=None
                    if observed is None
                    else observed[:, window],
                    text_feat=feat,
                    text_pad_mask=pad,
                    init_history_sequence=hist,
                )
            new = out[:, h : h + n]
            motion = new if motion is None else torch.cat([motion, new], dim=1)
        progress.set(0.95, "post-processing")
        with torch.no_grad():
            out = model.motion_rep.inverse(motion, is_normalized=True)
            out.update(
                post_process_motion(
                    out["local_rot_mats"],
                    out["root_positions"],
                    out["foot_contacts"],
                    model.skeleton,
                    constraint_lst=constraints or None,
                )
            )
        out = to_numpy(out)
        single = {
            k: (
                v[0]
                if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == 1
                else v
            )
            for k, v in out.items()
        }
        return da.to_clip(
            single,
            inp.canon,
            ARDY,
            segments=da.segment_ranges(inp),
            source={"model": req.model or self.default_model, "seed": req.seed},
        )

    def mock(self, req) -> dict:
        inp = da.model_inputs(req.model_dump(), ARDY)
        if self.mock_clip is None or not self.mock_clip.exists():
            raise FileNotFoundError(
                "MOCK_MODE serves an ARDY NPZ, none found at %s"
                % self.mock_clip
            )
        with np.load(self.mock_clip) as z:
            npz = {k: z[k] for k in z.files}
        return da.to_clip(
            npz,
            inp.canon,
            ARDY,
            segments=da.segment_ranges(inp),
            source={"model": "mock", "seed": req.seed},
        )


def make_app():
    default_out = Path(tempfile.gettempdir()) / "fxmotion" / "ardy"
    out = Path(os.environ.get("OUTPUT_DIR", str(default_out)))
    backend = ArdyBackend(
        default_model=os.environ.get("ARDY_MODEL", ""),
        mock_clip=os.environ.get("FXMOTION_MOCK_CLIP") or None,
        text_encoder_url=os.environ.get(
            "TEXT_ENCODER_URL", "http://127.0.0.1:9550/"
        ),
    )
    return fs.create_app(
        backend,
        output_dir=out,
        mock=os.environ.get("MOCK_MODE", "0") == "1",
        idle_unload_s=float(os.environ.get("FXMOTION_IDLE_UNLOAD_S", "0")),
        preload=True,
    )
