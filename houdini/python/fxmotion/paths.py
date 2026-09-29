"""Root-path and timing helpers for the generator nodes. Pure."""

from __future__ import annotations

import math


def thin(xz, count) -> list:
    """Indices of about `count` points spread by arc length, first and last
    always kept. A waypoint every frame or two pins the pelvis to a
    constant-speed glide; a handful of anchors lets the model put its own
    stride rhythm back between them. count < 2 keeps every point."""
    if count < 2 or len(xz) <= count:
        return list(range(len(xz)))
    cum = [0.0]
    # pairs each point with the next, so the second list is one shorter
    for (ax, az), (bx, bz) in zip(xz, xz[1:], strict=False):
        cum.append(cum[-1] + math.hypot(bx - ax, bz - az))
    total = cum[-1] or 1.0
    keep, j = [], 0
    for k in range(count):
        target = total * k / (count - 1)
        while j < len(cum) - 1 and cum[j + 1] < target:
            j += 1
        keep.append(j if k < count - 1 else len(xz) - 1)
    return sorted(set(keep))


def scene_seconds(
    scene_frame, start_frame, scene_fps, clip_fps, retime
) -> float:
    """Clip time, in seconds, that a scene frame lands on: what a request
    speaks. Retime on maps scene frames to real time; off, one clip sample
    per scene frame, so the model's own rate converts."""
    f = float(scene_frame) - float(start_frame)
    return max(0.0, f / float(scene_fps) if retime else f / float(clip_fps))
