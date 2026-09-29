"""The fxmotion.clip/1 format: the one shape every model server returns.

Spec: superpowers/specs/2026-09-29-fxhoudinimotion-design.md, section 3.
Pure numpy and no hou, because both sides use it: a server writes clips with
make() and save(), Houdini reads them with load().

Conventions: positions in Houdini space (Y up, metres), rotations in KineFX's
row-vector form and bone-aligned, joints in parent-first order.
"""

from __future__ import annotations

import json

import numpy as np

FORMAT = "fxmotion.clip/1"

REQUIRED = (
    "format",
    "skeleton",
    "joint_names",
    "parents",
    "fps",
    "world_pos",
    "world_rot",
    "rest_pos",
    "rest_rot",
)

# Model rotations are float32 and drift off orthonormal as they compose down
# the chain: 4.5e-3 measured on a 50-sample Kimodo clip. A real mistake, such
# as a matrix carrying a scale, is far larger.
ORTHO_TOL = 1e-2


class ClipError(ValueError):
    """Not a valid fxmotion.clip/1; the message names the offending key."""


def make(
    skeleton,
    joint_names,
    parents,
    fps,
    world_pos,
    world_rot,
    rest_pos,
    rest_rot,
    *,
    contacts=None,
    segments=None,
    source=None,
    native=None,
) -> dict:
    """Assemble and validate a clip. `native` maps names to arrays stored
    as native_<name>: the model's raw data, for its own features."""
    clip = {
        "format": np.array(FORMAT),
        "skeleton": np.array(str(skeleton)),
        "joint_names": np.array([str(n) for n in joint_names]),
        "parents": np.asarray(parents, dtype=np.int32),
        "fps": np.float32(fps),
        "world_pos": np.asarray(world_pos, dtype=np.float32),
        "world_rot": np.asarray(world_rot, dtype=np.float32),
        "rest_pos": np.asarray(rest_pos, dtype=np.float32),
        "rest_rot": np.asarray(rest_rot, dtype=np.float32),
    }
    if contacts is not None:
        clip["contacts"] = np.asarray(contacts, dtype=np.int8)
    if segments is not None:
        clip["segments"] = np.array(json.dumps(segments))
    if source is not None:
        clip["source"] = np.array(json.dumps(source))
    for name, value in (native or {}).items():
        clip["native_" + name] = np.asarray(value)
    validate(clip)
    return clip


def _shape(clip, key, want) -> None:
    if clip[key].shape != want:
        raise ClipError(
            "%s has shape %s, expected %s" % (key, clip[key].shape, want)
        )


def validate(clip) -> None:
    missing = [k for k in REQUIRED if k not in clip]
    if missing:
        raise ClipError(
            "not an %s clip: missing %s" % (FORMAT, ", ".join(missing))
        )
    if str(clip["format"]) != FORMAT:
        raise ClipError(
            "format is %r, expected %r" % (str(clip["format"]), FORMAT)
        )
    names, parents = clip["joint_names"], clip["parents"]
    count = len(names)
    if parents.shape != (count,):
        raise ClipError(
            "parents has shape %s for %d joints" % (parents.shape, count)
        )
    roots = int((parents < 0).sum())
    if roots != 1:
        raise ClipError("parents must have exactly one root, found %d" % roots)
    for i, p in enumerate(parents):
        if p >= i:
            raise ClipError(
                "joint %d (%s) comes before its parent %d; joints must be "
                "in parent-first order" % (i, names[i], p)
            )
    if not float(clip["fps"]) > 0:
        raise ClipError("fps must be positive, got %r" % float(clip["fps"]))
    pos = clip["world_pos"]
    if pos.ndim != 3 or pos.shape[1:] != (count, 3) or pos.shape[0] < 1:
        raise ClipError(
            "world_pos has shape %s, expected (T, %d, 3)" % (pos.shape, count)
        )
    frames = pos.shape[0]
    _shape(clip, "world_rot", (frames, count, 3, 3))
    _shape(clip, "rest_pos", (count, 3))
    _shape(clip, "rest_rot", (count, 3, 3))
    if "contacts" in clip:
        _shape(clip, "contacts", (frames, count))
    for key in ("world_pos", "world_rot", "rest_pos", "rest_rot"):
        if not np.isfinite(clip[key]).all():
            raise ClipError("%s holds NaN or inf" % key)
    for key in ("world_rot", "rest_rot"):
        r = clip[key].astype(np.float64)
        err = float(np.abs(r @ np.swapaxes(r, -1, -2) - np.eye(3)).max())
        if err > ORTHO_TOL:
            raise ClipError("%s is not orthonormal (error %.3g)" % (key, err))
    for key in ("segments", "source"):
        if key in clip:
            try:
                json.loads(str(clip[key]))
            except ValueError as e:
                raise ClipError("%s is not JSON: %s" % (key, e)) from e


def meta(clip, key):
    """The JSON stored under `key` (segments, source), or None."""
    return json.loads(str(clip[key])) if key in clip else None


def frame_count(clip) -> int:
    return int(clip["world_pos"].shape[0])


def save(path, clip) -> None:
    """Validate and write. Give `path` an .npz suffix: numpy appends one
    otherwise."""
    validate(clip)
    np.savez_compressed(path, **clip)


def load(path) -> dict:
    """Every array of a clip file, validated."""
    with np.load(path, allow_pickle=False) as z:
        clip = {k: z[k] for k in z.files}
    validate(clip)
    return clip
