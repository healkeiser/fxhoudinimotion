"""Small valid clips for tests that do not care about the skeleton."""

import numpy as np

from fxmotion import clipformat


def tiny(frames=3, **overrides):
    kw = {
        "skeleton": "test",
        "joint_names": ["root", "tip"],
        "parents": [-1, 0],
        "fps": 30.0,
        "world_pos": np.zeros((frames, 2, 3)),
        "world_rot": np.tile(np.eye(3), (frames, 2, 1, 1)),
        "rest_pos": np.zeros((2, 3)),
        "rest_rot": np.tile(np.eye(3), (2, 1, 1)),
    }
    kw.update(overrides)
    return clipformat.make(**kw)
