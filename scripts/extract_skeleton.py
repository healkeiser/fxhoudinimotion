"""Write fxmotion's copy of an ARDY skeleton table. Run with ARDY's venv:

    <ardy>/.venv/Scripts/python.exe scripts/extract_skeleton.py ardy_core

The rest pose comes from ARDY's joints.p (a T-pose, hips at the origin).
"""

import sys
from pathlib import Path

import ardy
import torch
from ardy.skeleton.definitions import CoreSkeleton27

SKELETONS = {"ardy_core": CoreSkeleton27}

name = sys.argv[1]
cls = SKELETONS[name]
assets = (
    Path(ardy.__file__).resolve().parent / "assets" / "skeletons" / cls.name
)
rest = torch.load(assets / "joints.p", weights_only=False).reshape(-1, 3)
names = [n for n, _ in cls.bone_order_names_with_parents]
parents = [
    -1 if p is None else names.index(p)
    for _, p in cls.bone_order_names_with_parents
]
assert len(rest) == len(names), (len(rest), len(names))
out = (
    Path(__file__).resolve().parents[1]
    / "houdini"
    / "python"
    / "fxmotion"
    / "skeletons"
    / ("%s.py" % name)
)
lines = [
    '"""%s skeleton data, extracted from NVIDIA ARDY (Apache-2.0) by' % name,
    "scripts/extract_skeleton.py. Rest pose: a T-pose, hips at the origin,",
    'joint frames world-axis-aligned."""',
    "",
    "JOINTS = %r" % names,
    "PARENTS = %r" % parents,
    "REST_POS = [",
]
lines += ["    (%.6f, %.6f, %.6f)," % tuple(float(v) for v in p) for p in rest]
lines += ["]", ""]
out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
print("wrote", out)
