"""A clip on disk -> KineFX skeleton geometry.

The math is pure and tested offline against the 1.1 cook; build() and
build_rest() take a hou.Geometry and run inside the generator SOPs. They set
attributes in batches (one HOM call per attribute, not per point).
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import clipformat

_CACHE: dict = {}
_CACHE_SIZE = 8  # generator nodes in one scene, each with its own clip


def load(path) -> dict:
    """Every array of a clip, read once per file version: the animated SOP
    cooks on every frame change and an NpzFile re-decompresses per lookup."""
    key = (str(path), os.path.getmtime(path))
    hit = _CACHE.get(key)
    if hit is None:
        if len(_CACHE) >= _CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))  # oldest first
        hit = _CACHE[key] = clipformat.load(path)
    return hit


def sample_index(
    scene_frame, start_frame, scene_fps, clip_fps, retime, frames
) -> int:
    """Which clip sample a scene frame shows. The clip starts on Start
    Frame; with Retime on, scene frames map onto samples by real time, so a
    3 s clip lasts 3 s at any $FPS. The first sample holds before the start,
    the last after the end. Nearest sample, no blending."""
    f = float(scene_frame) - float(start_frame)
    if retime:
        f = f * float(clip_fps) / float(scene_fps)
    return max(0, min(int(round(f)), int(frames) - 1))


def joint_paths(names, parents) -> list:
    out = []
    for name, parent in zip(names, parents, strict=True):
        out.append("/" + name if parent < 0 else out[parent] + "/" + name)
    return out


def joint_frames(clip, index):
    """World positions (J, 3), KineFX transforms (J, 9) and local 4x4s
    (J, 16) at one sample. Row-vector: local = world @ inverse(parent)."""
    pos = clip["world_pos"][index].astype(np.float64)
    rot = clip["world_rot"][index].astype(np.float64)
    count = len(pos)
    world = np.zeros((count, 4, 4))
    world[:, :3, :3] = rot
    world[:, 3, :3] = pos
    world[:, 3, 3] = 1.0
    parents = np.asarray(clip["parents"])
    local = world.copy()
    has = parents >= 0
    local[has] = world[has] @ np.linalg.inv(world[parents[has]])
    return pos, rot.reshape(count, 9), local.reshape(count, 16)


def _skeleton_points(geo, pos, names, parents) -> None:
    import hou

    geo.addAttrib(hou.attribType.Point, "name", "")
    geo.addAttrib(hou.attribType.Point, "path", "")
    geo.addAttrib(hou.attribType.Point, "parent_id", -1)
    geo.createPoints([hou.Vector3(p) for p in np.asarray(pos).tolist()])
    geo.setPointStringAttribValues("name", tuple(names))
    geo.setPointStringAttribValues("path", tuple(joint_paths(names, parents)))
    geo.setPointIntAttribValues("parent_id", tuple(parents))
    bones = tuple((p, i) for i, p in enumerate(parents) if p >= 0)
    geo.createPolygons(bones, False)  # is_closed only works positionally


def _detail(geo, name, value) -> None:
    import hou

    default = "" if isinstance(value, str) else 0.0
    geo.addAttrib(hou.attribType.Global, name, default)
    geo.setGlobalAttribValue(name, value)


def build(geo, clip, index) -> None:
    """The animated skeleton at one sample, plus the fxmotion_* details."""
    import hou

    pos, xform, local = joint_frames(clip, index)
    names = [str(n) for n in clip["joint_names"]]
    parents = [int(p) for p in clip["parents"]]
    _skeleton_points(geo, pos, names, parents)
    # HOM wants Python floats, not numpy scalars: hence tolist() throughout
    geo.addAttrib(
        hou.attribType.Point, "transform", tuple(np.eye(3).flatten().tolist())
    )
    geo.addAttrib(
        hou.attribType.Point,
        "localtransform",
        tuple(np.eye(4).flatten().tolist()),
    )
    geo.setPointFloatAttribValues("transform", tuple(xform.flatten().tolist()))
    geo.setPointFloatAttribValues(
        "localtransform", tuple(local.flatten().tolist())
    )
    if "contacts" in clip:
        geo.addAttrib(hou.attribType.Point, "contact", 0)
        geo.setPointIntAttribValues(
            "contact", tuple(int(v) for v in clip["contacts"][index])
        )
    _detail(geo, "fxmotion_skeleton", str(clip["skeleton"]))
    _detail(geo, "fxmotion_fps", float(clip["fps"]))
    source = clipformat.meta(clip, "source") or {}
    _detail(geo, "fxmotion_source", json.dumps(source, sort_keys=True))


def build_rest(geo, skeleton) -> None:
    """The skeleton's rest pose (the T-Pose output)."""
    import hou

    names, parents = list(skeleton.joint_names), list(skeleton.parents)
    _skeleton_points(geo, skeleton.rest_pos, names, parents)
    geo.addAttrib(
        hou.attribType.Point, "transform", tuple(np.eye(3).flatten().tolist())
    )
    geo.setPointFloatAttribValues(
        "transform", tuple(skeleton.rest_rot.flatten().tolist())
    )
    _detail(geo, "fxmotion_skeleton", skeleton.name)
