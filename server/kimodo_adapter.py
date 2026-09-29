"""Kimodo <-> fxmotion: the pure half of the Kimodo server.

No torch and no kimodo import, so the conversion is tested offline against a
real Kimodo NPZ. The inference half is kimodo_backend.py.

Two spaces meet here:
  model space  where Kimodo generates: root at XZ (0, 0) facing +Z on the
               first sample; rotations column-vector, world-axis-aligned at
               rest.
  Houdini      where the request and the clip live: the root path as drawn;
               rotations row-vector and bone-aligned, ready for KineFX.
Canon moves positions and rotations between the two frames; to_clip and
keyframe_constraints convert the rotation convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fxmotion import clipformat, skeletons

FPS = 30.0

# Every per-sample array in a Kimodo NPZ, kept as native_* for Regenerate.
NATIVE_KEYS = (
    "local_rot_mats",
    "global_rot_mats",
    "posed_joints",
    "root_positions",
    "smooth_root_pos",
    "foot_contacts",
    "global_root_heading",
)

# Joints behind each foot_contacts channel. A SOMA77 NPZ carries 6 channels
# (ToeEnd duplicates ToeBase); NPZs from the 4-channel internal representation
# carry only the detected pair.
FOOT_CHANNELS = {
    6: (
        "LeftFoot",
        "LeftToeBase",
        "LeftToeEnd",
        "RightFoot",
        "RightToeBase",
        "RightToeEnd",
    ),
    4: ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"),
}


class AdapterError(ValueError):
    """A request Kimodo cannot run; the server answers 422 with it."""


def yaw(ang: float) -> np.ndarray:
    """Rotation about +Y by `ang`, column-vector (the 1.1 cook's matrix)."""
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass(frozen=True)
class Canon:
    """Where model space sits in Houdini: origin (ox, 0, oz), heading ang."""

    ox: float = 0.0
    oz: float = 0.0
    ang: float = 0.0

    @classmethod
    def from_xz(cls, xz) -> Canon:
        """Origin on the first point, heading toward the first point that
        is not on top of it (+Z if there is none)."""
        ox, oz = (float(v) for v in xz[0])
        dx, dz = next(
            (
                (x - ox, z - oz)
                for x, z in xz[1:]
                if abs(x - ox) + abs(z - oz) > 1e-4
            ),
            (0.0, 1.0),
        )
        return cls(ox, oz, math.atan2(dx, dz))

    def _origin(self) -> np.ndarray:
        return np.array([self.ox, 0.0, self.oz])

    def pos_to_model(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return (p - self._origin()) @ yaw(self.ang)

    def pos_to_world(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return p @ yaw(self.ang).T + self._origin()

    def rot_to_model(self, r) -> np.ndarray:
        return yaw(self.ang).T @ np.asarray(r, dtype=np.float64)

    def rot_to_world(self, r) -> np.ndarray:
        return yaw(self.ang) @ np.asarray(r, dtype=np.float64)

    def as_list(self) -> list:
        return [self.ox, self.oz, self.ang]


def sample(time_s) -> int:
    """Clip sample for a time in seconds."""
    return int(round(float(time_s) * FPS))


def _tpose() -> np.ndarray:
    """Column-vector T-pose offsets per joint (1.1's TPOSE_ROTS)."""
    rest = skeletons.get("soma77").rest_rot
    return np.swapaxes(rest.astype(np.float64), -1, -2)


def to_clip(npz, canon: Canon, *, segments=None, source=None) -> dict:
    """A Kimodo output (model space) as an fxmotion clip (Houdini space)."""
    skel = skeletons.get("soma77")
    pos = canon.pos_to_world(npz["posed_joints"])
    grot = canon.rot_to_world(npz["global_rot_mats"])
    # right-multiplying the T-pose offsets makes each frame bone-aligned,
    # matching the T-Pose output; the transpose is KineFX's row-vector form
    world_rot = np.swapaxes(grot @ _tpose(), -1, -2)
    contacts = None
    fc = npz.get("foot_contacts")
    if fc is not None and fc.shape[-1] in FOOT_CHANNELS:
        contacts = np.zeros(pos.shape[:2], dtype=np.int8)
        for ch, name in enumerate(FOOT_CHANNELS[fc.shape[-1]]):
            contacts[:, skel.index(name)] = fc[:, ch]
    return clipformat.make(
        "soma77",
        skel.joint_names,
        skel.parents,
        FPS,
        pos,
        world_rot,
        skel.rest_pos,
        skel.rest_rot,
        contacts=contacts,
        segments=segments,
        source=dict(source or {}, backend="kimodo", canon=canon.as_list()),
        native={k: npz[k] for k in NATIVE_KEYS if k in npz},
    )


def keyframe_constraints(keyframes, canon: Canon) -> list:
    """Request keyframes (Houdini space, KineFX rotations) as Kimodo
    fullbody-global / ee-global dicts in model space. Keys sharing the same
    `joints` become one constraint, like 1.1's one per track."""
    skel = skeletons.get("soma77")
    count = len(skel.joint_names)
    tp_t = np.swapaxes(_tpose(), -1, -2)
    hips = skel.index("Hips")
    groups: dict = {}
    for kf in keyframes:
        pos = np.asarray(kf["world_pos"], dtype=np.float64)
        rot = np.asarray(kf["world_rot"], dtype=np.float64)
        if pos.shape != (count, 3) or rot.shape != (count, 3, 3):
            raise AdapterError(
                "kimodo: a keyframe needs world_pos (%d, 3) and world_rot "
                "(%d, 3, 3) for every SOMA77 joint, got %s and %s"
                % (count, count, pos.shape, rot.shape)
            )
        joints = tuple(kf.get("joints") or ())
        unknown = [j for j in joints if j not in skel.joint_names]
        if unknown:
            raise AdapterError(
                "kimodo: unknown joints %s" % ", ".join(unknown)
            )
        # KineFX transform R = (grot @ tp).T, so grot = R.T @ tp.T
        grot = np.swapaxes(rot, -1, -2) @ tp_t
        groups.setdefault(joints, []).append(
            (
                sample(kf["time_s"]),
                canon.pos_to_model(pos),
                canon.rot_to_model(grot),
            )
        )
    out = []
    for joints, keys in groups.items():
        keys.sort(key=lambda k: k[0])
        c = {
            "frame_indices": [f for f, _, _ in keys],
            "global_joints_positions": [p.tolist() for _, p, _ in keys],
            "global_joints_rots": [r.tolist() for _, _, r in keys],
            "smooth_root_2d": [
                [float(p[hips, 0]), float(p[hips, 2])] for _, p, _ in keys
            ],
        }
        if joints:
            c.update(type="ee-global", joint_names=list(joints))
        else:
            c["type"] = "fullbody-global"
        out.append(c)
    return out


def _path_items(points, duration_s):
    """[(sample, [x, z])], deduplicated by sample (first wins), sorted."""
    timed = [p.get("time_s") is not None for p in points]
    if any(timed) and not all(timed):
        raise AdapterError(
            "kimodo: root_path points must all have time_s or none"
        )
    xz = [[float(p["pos"][0]), float(p["pos"][2])] for p in points]
    if all(timed):
        frames = [sample(p["time_s"]) for p in points]
    elif len(xz) > 1:
        # spread evenly over the clip, as 1.1 did
        last = max(2, int(duration_s * FPS)) - 1
        frames = [
            int(round(i * last / (len(xz) - 1))) for i in range(len(xz))
        ]
    else:
        frames = [0]
    by_frame: dict = {}
    for f, c in zip(frames, xz):
        by_frame.setdefault(f, c)
    return sorted(by_frame.items())


@dataclass
class KimodoInputs:
    texts: list
    durations_s: list
    num_frames: list
    constraints: list
    canon: Canon
    transition_frames: int
    continue_from: dict | None


def kimodo_inputs(req: dict) -> KimodoInputs:
    """A GenerateRequest (as a dict) turned into what Kimodo's model call
    takes. Raises AdapterError for anything Kimodo cannot run."""
    texts, durs = [], []
    for i, seg in enumerate(req.get("segments") or [], start=1):
        text = (seg.get("prompt") or "").strip()
        dur = float(seg.get("duration_s") or 0.0)
        if not text:
            raise AdapterError("kimodo: segment %d has no prompt" % i)
        if dur <= 0:
            raise AdapterError("kimodo: segment %d needs duration_s > 0" % i)
        texts.append(text)
        durs.append(dur)
    if not texts:
        raise AdapterError("kimodo: at least one segment is required")
    opts = req.get("options") or {}
    root_path = req.get("root_path") or []
    items = _path_items(root_path, sum(durs)) if root_path else []
    if opts.get("canon") is not None:
        canon = Canon(*(float(v) for v in opts["canon"]))
    elif items:
        canon = Canon.from_xz([c for _, c in items])
    else:
        canon = Canon()
    native = opts.get("native_constraints") or []
    if not isinstance(native, list):
        raise AdapterError("kimodo: native_constraints must be a list")
    constraints = list(native)
    if items:
        constraints.append(
            {
                "type": "root2d",
                "frame_indices": [f for f, _ in items],
                "smooth_root_2d": [
                    canon.pos_to_model([x, 0.0, z])[[0, 2]].tolist()
                    for _, (x, z) in items
                ],
            }
        )
    constraints += keyframe_constraints(req.get("keyframes") or [], canon)
    cont = None
    cf = req.get("continue_from")
    if cf:
        try:
            cont = {
                "local_rot_mats": cf["native_local_rot_mats"],
                "root_positions": cf["native_root_positions"],
            }
        except KeyError as e:
            raise AdapterError(
                "kimodo: continue_from needs native_local_rot_mats and "
                "native_root_positions (%s missing)" % e
            ) from e
    return KimodoInputs(
        texts=texts,
        durations_s=durs,
        num_frames=[max(1, int(d * FPS)) for d in durs],
        constraints=constraints,
        canon=canon,
        transition_frames=max(1, int(opts.get("transition_frames", 5))),
        continue_from=cont,
    )
