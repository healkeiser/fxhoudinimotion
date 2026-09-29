"""Skeletons the models generate on: joints, rest pose and rest geometry.

Rotations are stored in Houdini's convention (row-vector, bone-aligned), the
same as a clip's world_rot, so a node can write them straight into the
`transform` attribute. A server that needs the column-vector form transposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True, eq=False)
class Skeleton:
    name: str
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    rest_pos: np.ndarray  # (J, 3), Houdini space
    rest_rot: np.ndarray  # (J, 3, 3), Houdini row-vector
    skin: Path | None = None  # rest geometry with boneCapture (output 0)
    capture_pose: Path | None = (
        None  # the skeleton the skin is bound to (output 1)
    )

    def index(self, name: str) -> int:
        return self.joint_names.index(name)


def _soma77() -> Skeleton:
    from . import soma77 as d

    # TPOSE_ROTS are column-vector; KineFX wants row-vector, hence the transpose
    tp = np.asarray(d.TPOSE_ROTS, dtype=np.float64).reshape(-1, 3, 3)
    return Skeleton(
        name="soma77",
        joint_names=tuple(d.SOMA77_JOINTS),
        parents=tuple(int(p) for p in d.SOMA77_PARENTS),
        rest_pos=np.asarray(d.NEUTRAL_JOINTS, dtype=np.float64),
        rest_rot=np.transpose(tp, (0, 2, 1)),
        skin=DATA / "soma77_skin.bgeo.sc",
        capture_pose=DATA / "soma77_apose.bgeo.sc",
    )


_BUILDERS = {"soma77": _soma77}
_CACHE: dict[str, Skeleton] = {}


def names() -> tuple[str, ...]:
    return tuple(_BUILDERS)


def get(name: str) -> Skeleton:
    if name not in _BUILDERS:
        raise KeyError(
            "unknown skeleton %r; known: %s" % (name, ", ".join(_BUILDERS))
        )
    if name not in _CACHE:
        _CACHE[name] = _BUILDERS[name]()
    return _CACHE[name]
