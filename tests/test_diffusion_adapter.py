"""The adapter with ARDY's spec: a real ARDY Core clip (20 fps, 27 joints)."""

from pathlib import Path

import diffusion_adapter as da
import numpy as np
import pytest

from fxmotion import clipformat

ARDY = da.ModelSpec("ardy", "ardy_core", 20.0)
FIXTURE = Path(__file__).parent / "fixtures" / "ardy_path.npz"


def _npz():
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


def _req(**over):
    req = {
        "segments": [{"prompt": "a person walks", "duration_s": 5.0}],
        "root_path": None,
        "keyframes": None,
        "continue_from": None,
        "seed": None,
        "model": "",
        "force": False,
        "options": {},
    }
    req.update(over)
    return req


def test_to_clip_with_ardy_spec():
    npz = _npz()
    clip = da.to_clip(npz, da.Canon(), ARDY)
    clipformat.validate(clip)
    assert str(clip["skeleton"]) == "ardy_core" and float(clip["fps"]) == 20.0
    assert clip["world_pos"].shape == (100, 27, 3)
    # identity rest rotations: the KineFX rotation is the global one transposed
    want = np.swapaxes(npz["global_rot_mats"], -1, -2)
    assert np.abs(clip["world_rot"] - want).max() < 1e-5
    names = [str(n) for n in clip["joint_names"]]
    assert np.array_equal(
        clip["contacts"][:, names.index("RightToeBase")],
        npz["foot_contacts"][:, 3],
    )
    assert clipformat.meta(clip, "source")["backend"] == "ardy"
    assert -0.05 < clip["world_pos"][..., 1].min() < 0.05


def test_times_are_ardy_samples():
    path = [
        {"pos": [0.0, 0.0, 0.0], "time_s": 0.0},
        {"pos": [0.0, 0.0, 2.0], "time_s": 2.0},
    ]
    inp = da.model_inputs(_req(root_path=path), ARDY)
    assert inp.num_frames == [100]
    assert inp.constraints[-1]["frame_indices"] == [0, 40]


def test_last_frame_key_at_20_fps_is_accepted():
    # 4 x 25 scene frames at 24 fps: each segment truncates to 20 samples,
    # 80 in all, while the last scene frame is at 4.125 s = sample 82
    segs = [{"prompt": "walk", "duration_s": 25 / 24}] * 4
    clip = da.to_clip(_npz(), da.Canon(), ARDY)
    kf = {
        "time_s": 99 / 24,
        "world_pos": clip["world_pos"][0].tolist(),
        "world_rot": clip["world_rot"][0].tolist(),
        "joints": None,
    }
    inp = da.model_inputs(_req(segments=segs, keyframes=[kf]), ARDY)
    assert sum(inp.num_frames) == 80
    assert inp.constraints[-1]["frame_indices"] == [79]


def test_times_past_the_requested_duration_are_refused():
    late = [
        {"pos": [0.0, 0.0, 0.0], "time_s": 0.0},
        {"pos": [0.0, 0.0, 1.0], "time_s": 5.5},
    ]
    with pytest.raises(da.AdapterError, match="ardy: root_path time 5.5 s"):
        da.model_inputs(_req(root_path=late), ARDY)


def test_keyframe_round_trip_with_identity_offsets():
    npz = _npz()
    canon = da.Canon(1.0, -1.0, 0.4)
    clip = da.to_clip(npz, canon, ARDY)
    kf = {
        "time_s": 30 / 20.0,
        "world_pos": clip["world_pos"][30].tolist(),
        "world_rot": clip["world_rot"][30].tolist(),
        "joints": None,
    }
    (c,) = da.keyframe_constraints([kf], canon, ARDY)
    assert c["frame_indices"] == [30]
    rot = np.array(c["global_joints_rots"][0])
    assert np.abs(rot - npz["global_rot_mats"][30]).max() < 1e-4


def test_errors_name_the_model():
    with pytest.raises(da.AdapterError, match="ardy: segment 1 has no prompt"):
        da.model_inputs(
            _req(segments=[{"prompt": "", "duration_s": 1.0}]), ARDY
        )
    kf = {
        "time_s": 0.0,
        "world_pos": [[0, 0, 0]],
        "world_rot": [np.eye(3).tolist()],
        "joints": None,
    }
    with pytest.raises(da.AdapterError, match="every ARDY_CORE joint"):
        da.keyframe_constraints([kf], da.Canon(), ARDY)


def test_segment_ranges():
    inp = da.model_inputs(
        _req(
            segments=[
                {"prompt": "a", "duration_s": 1.0},
                {"prompt": "b", "duration_s": 0.5},
            ]
        ),
        ARDY,
    )
    assert da.segment_ranges(inp) == [
        {"prompt": "a", "start": 0, "end": 19},
        {"prompt": "b", "start": 20, "end": 29},
    ]
