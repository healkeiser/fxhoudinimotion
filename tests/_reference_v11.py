"""The vb::kimodo_motion::1.1 animated cook, frozen as a reference.

Copied from _COOK_SCRIPT in scripts/create_hda.py as of commit b5e3180 (the
last change to the 1.1 build). 2.0 must reproduce these numbers; do not edit
this file to make a test pass.
"""

import math

import numpy as np

from fxmotion.skeletons.soma77 import SOMA77_PARENTS, TPOSE_ROTS


def cook(npz, frame, ox=0.0, oz=0.0, ang=0.0):
    pos = npz["posed_joints"][frame].copy().astype(np.float64)
    grot = npz["global_rot_mats"][frame].astype(np.float64)
    if ang or ox or oz:
        c, s_ = np.cos(ang), np.sin(ang)
        R = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
        pos = pos @ R.T + np.array([ox, 0.0, oz])
        grot = np.einsum("ij,njk->nik", R, grot)
    tp = np.asarray(TPOSE_ROTS, dtype=float).reshape(-1, 3, 3)
    world_rot = [grot[i] @ tp[i] for i in range(len(SOMA77_PARENTS))]
    world_m = []
    for i in range(len(SOMA77_PARENTS)):
        m = np.identity(4)
        m[:3, :3] = world_rot[i].T
        m[3, :3] = pos[i]
        world_m.append(m)
    transforms, locals_ = [], []
    for i, parent in enumerate(SOMA77_PARENTS):
        local_m = (
            world_m[i]
            if parent < 0
            else world_m[i] @ np.linalg.inv(world_m[parent])
        )
        transforms.append(world_rot[i].T.flatten())
        locals_.append(local_m.flatten())
    return pos, np.array(transforms), np.array(locals_)


def canon_xz(xz):
    """1.1's Generate-callback canonicalisation of a root path."""
    ox, oz = xz[0]
    dx, dz = next(
        (
            (x - ox, z - oz)
            for x, z in xz[1:]
            if abs(x - ox) + abs(z - oz) > 1e-4
        ),
        (0.0, 1.0),
    )
    ang = math.atan2(dx, dz)
    ca, sa = math.cos(-ang), math.sin(-ang)
    pts = [
        [(x - ox) * ca + (z - oz) * sa, -(x - ox) * sa + (z - oz) * ca]
        for x, z in xz
    ]
    return pts, (ox, oz, ang)
