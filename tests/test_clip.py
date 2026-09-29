"""Clip -> KineFX math, against the frozen 1.1 cook."""

import os
from pathlib import Path

import _reference_v11 as v11
import numpy as np
import pytest
from cliptools import tiny

import kimodo_adapter as ka
from fxmotion import clip, clipformat

FIXTURE = Path(__file__).parent / "fixtures" / "kimodo_stop.npz"


def _npz():
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize("canon", (ka.Canon(), ka.Canon(1.5, -2.0, 0.7)))
def test_joint_frames_match_the_v11_cook(canon):
    npz = _npz()
    c = ka.to_clip(npz, canon)
    for frame in (0, 17, 49):
        pos, xform, local = clip.joint_frames(c, frame)
        rpos, rxform, rlocal = v11.cook(
            npz, frame, canon.ox, canon.oz, canon.ang
        )
        assert np.abs(pos - rpos).max() < 1e-5
        assert np.abs(xform - rxform).max() < 1e-5
        assert np.abs(local - rlocal).max() < 1e-4


def test_sample_index_matches_v11():
    # 1.1: f = frame - start; with retime f *= clip_fps / scene_fps; round; clamp
    assert clip.sample_index(1, 1, 24, 30, True, 50) == 0
    assert clip.sample_index(25, 1, 24, 30, True, 50) == 30
    assert clip.sample_index(25, 1, 24, 30, False, 50) == 24
    assert clip.sample_index(1001, 1001, 25, 30, True, 50) == 0
    assert clip.sample_index(1011, 1001, 25, 30, True, 50) == 12
    assert clip.sample_index(-5, 1, 24, 30, True, 50) == 0
    assert clip.sample_index(500, 1, 24, 30, True, 50) == 49


def test_joint_paths():
    paths = clip.joint_paths(["Hips", "Spine1", "LeftUpLeg"], [-1, 0, 0])
    assert paths == ["/Hips", "/Hips/Spine1", "/Hips/LeftUpLeg"]


def test_load_caches_by_file_version(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    clipformat.save(a, tiny(3))
    clipformat.save(b, tiny(4))
    first = clip.load(a)
    assert clip.load(b) is not first
    assert clip.load(a) is first  # two nodes alternating stay cached
    clipformat.save(a, tiny(5))
    st = a.stat()
    os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert clipformat.frame_count(clip.load(a)) == 5


def test_load_names_what_is_wrong_with_a_native_npz():
    with pytest.raises(clipformat.ClipError, match="missing format"):
        clip.load(FIXTURE)
