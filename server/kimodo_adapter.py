"""Kimodo on the shared diffusion adapter (diffusion_adapter.py). Keeps the
phase 1 names, which kimodo_backend and the tests use."""

from __future__ import annotations

import diffusion_adapter as da
from diffusion_adapter import (  # noqa: F401  (phase 1 names)
    FOOT_CHANNELS,
    NATIVE_KEYS,
    AdapterError,
    Canon,
    yaw,
)

KIMODO = da.ModelSpec(name="kimodo", skeleton="soma77", fps=30.0)
FPS = KIMODO.fps
KimodoInputs = da.ModelInputs


def sample(time_s) -> int:
    return da.sample(time_s, KIMODO)


def to_clip(npz, canon, *, segments=None, source=None) -> dict:
    return da.to_clip(npz, canon, KIMODO, segments=segments, source=source)


def keyframe_constraints(
    keyframes, canon, frames=None, duration_s=None
) -> list:
    return da.keyframe_constraints(keyframes, canon, KIMODO, frames, duration_s)


def kimodo_inputs(req: dict) -> da.ModelInputs:
    return da.model_inputs(req, KIMODO)
