"""The ARDY Motion node: generator.py on ARDY's Core skeleton (ardy_core)."""

from __future__ import annotations

from . import generator
from .common import cancel, test_connection  # noqa: F401  (HDA callbacks)
from .generator import open_timeline, regenerate, sync_segments  # noqa: F401
from .timeline_parms import (  # noqa: F401  (HDA callbacks)
    on_created,
    refresh_starts,
    split_segment,
)

SKELETON = "ardy_core"


def generate(node) -> None:
    generator.generate(node, SKELETON)


def build_payload(node, clip_fps) -> dict:
    return generator.build_payload(node, clip_fps, SKELETON)


def make_pose_rig(node) -> None:
    generator.make_pose_rig(node, SKELETON)
