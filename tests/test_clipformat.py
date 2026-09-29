"""fxmotion.clip/1: what every server writes and every node reads."""

from pathlib import Path

import numpy as np
import pytest
from cliptools import tiny

from fxmotion import clipformat

FIXTURE = Path(__file__).parent / "fixtures" / "kimodo_stop.npz"


def test_round_trip_keeps_arrays_and_metadata(tmp_path):
    clip = tiny(
        4,
        contacts=np.ones((4, 2)),
        segments=[{"prompt": "walk", "start": 0, "end": 3}],
        source={"backend": "test"},
        native={"extra": np.arange(4)},
    )
    path = tmp_path / "c.npz"
    clipformat.save(path, clip)
    back = clipformat.load(path)
    assert set(back) == set(clip)
    assert np.array_equal(back["world_rot"], clip["world_rot"])
    assert back["contacts"].dtype == np.int8
    assert clipformat.meta(back, "segments")[0]["prompt"] == "walk"
    assert clipformat.meta(back, "source") == {"backend": "test"}
    assert clipformat.meta(back, "missing") is None
    assert clipformat.frame_count(back) == 4
    assert np.array_equal(back["native_extra"], np.arange(4))


def test_a_kimodo_native_npz_is_rejected_by_name():
    with pytest.raises(clipformat.ClipError, match="missing format, skeleton"):
        clipformat.load(FIXTURE)


def test_wrong_format_string():
    clip = tiny()
    clip["format"] = np.array("fxmotion.clip/0")
    with pytest.raises(
        clipformat.ClipError, match="expected 'fxmotion.clip/1'"
    ):
        clipformat.validate(clip)


def test_exactly_one_root():
    with pytest.raises(clipformat.ClipError, match="exactly one root"):
        tiny(parents=[-1, -1])


def test_parent_first_order():
    with pytest.raises(clipformat.ClipError, match="parent-first"):
        tiny(
            joint_names=["a", "b", "c"],
            parents=[-1, 2, 0],
            world_pos=np.zeros((3, 3, 3)),
            world_rot=np.tile(np.eye(3), (3, 3, 1, 1)),
            rest_pos=np.zeros((3, 3)),
            rest_rot=np.tile(np.eye(3), (3, 1, 1)),
        )


def test_shapes_must_agree():
    with pytest.raises(clipformat.ClipError, match="world_rot has shape"):
        tiny(world_rot=np.tile(np.eye(3), (2, 2, 1, 1)))  # 2 frames, not 3
    with pytest.raises(clipformat.ClipError, match="contacts has shape"):
        tiny(contacts=np.zeros((3, 5)))


def test_nan_is_rejected():
    pos = np.zeros((3, 2, 3))
    pos[1, 1, 2] = np.nan
    with pytest.raises(clipformat.ClipError, match="NaN"):
        tiny(world_pos=pos)


def test_scaled_rotation_is_rejected_but_float32_drift_is_not():
    rot = np.tile(np.eye(3), (3, 2, 1, 1))
    with pytest.raises(clipformat.ClipError, match="not orthonormal"):
        tiny(world_rot=rot * 1.1)
    drift = rot.copy()
    drift[..., 0, 1] = 4.5e-3  # the drift measured on a real Kimodo clip
    tiny(world_rot=drift)


def test_fps_must_be_positive():
    with pytest.raises(clipformat.ClipError, match="fps"):
        tiny(fps=0)
