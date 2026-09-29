"""Kimodo <-> fxmotion conversions, against a real Kimodo NPZ and the frozen
1.1 cook."""

from pathlib import Path

import _reference_v11 as v11
import numpy as np
import pytest

import kimodo_adapter as ka
from fxmotion import clipformat

FIXTURE = Path(__file__).parent / "fixtures" / "kimodo_stop.npz"
CANONS = (ka.Canon(), ka.Canon(1.5, -2.0, 0.7))


def _npz():
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


def _req(**over):
    req = {
        "segments": [{"prompt": "a person walks", "duration_s": 2.0}],
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


@pytest.mark.parametrize("canon", CANONS)
def test_to_clip_matches_the_v11_cook(canon):
    npz = _npz()
    clip = ka.to_clip(npz, canon)
    for frame in (0, 10, 25, 49):
        pos, xform, _ = v11.cook(npz, frame, canon.ox, canon.oz, canon.ang)
        assert np.abs(clip["world_pos"][frame] - pos).max() < 1e-5
        got = clip["world_rot"][frame].reshape(-1, 9)
        assert np.abs(got - xform).max() < 1e-5


def test_to_clip_is_valid_and_keeps_contacts_native_and_canon():
    npz = _npz()
    clip = ka.to_clip(npz, ka.Canon(1.0, 2.0, 0.5), source={"model": "m"})
    clipformat.validate(clip)
    assert str(clip["skeleton"]) == "soma77" and float(clip["fps"]) == 30.0
    names = [str(n) for n in clip["joint_names"]]
    left = names.index("LeftFoot")
    assert np.array_equal(
        clip["contacts"][:, left], npz["foot_contacts"][:, 0]
    )
    for key in ka.NATIVE_KEYS:
        assert np.array_equal(clip["native_" + key], npz[key])
    src = clipformat.meta(clip, "source")
    assert src == {"model": "m", "backend": "kimodo", "canon": [1.0, 2.0, 0.5]}


def test_generated_clip_reaches_the_floor():
    low = ka.to_clip(_npz(), ka.Canon())["world_pos"][..., 1].min()
    assert -0.05 < low < 0.05


def test_canon_round_trips():
    c = ka.Canon(1.5, -2.0, 0.7)
    p = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(c.pos_to_world(c.pos_to_model(p)), p)
    r = _npz()["global_rot_mats"][3]
    assert np.allclose(c.rot_to_world(c.rot_to_model(r)), r, atol=1e-6)


def test_canon_puts_the_path_start_on_plus_z_like_v11():
    xz = [[2.0, 1.0], [3.0, 3.0], [5.0, 3.5]]
    want, (ox, oz, ang) = v11.canon_xz(xz)
    c = ka.Canon.from_xz(xz)
    assert np.allclose(c.as_list(), [ox, oz, ang])
    got = [c.pos_to_model([x, 0.0, z])[[0, 2]] for x, z in xz]
    assert np.allclose(got, want)
    assert abs(got[1][0]) < 1e-9 and got[1][1] > 0


def test_canon_ignores_coincident_points():
    assert ka.Canon.from_xz([[1.0, 1.0], [1.0, 1.0]]).as_list() == [
        1.0,
        1.0,
        0.0,
    ]
    assert ka.Canon.from_xz([[4.0, -2.0]]).as_list() == [4.0, -2.0, 0.0]


def test_timed_path_samples_at_30_fps():
    path = [
        {"pos": [0.0, 0.0, 0.0], "time_s": 0.0},
        {"pos": [0.0, 0.0, 2.0], "time_s": 1.0},
    ]
    inp = ka.kimodo_inputs(_req(root_path=path))
    root2d = inp.constraints[-1]
    assert root2d["type"] == "root2d"
    assert root2d["frame_indices"] == [0, 30]
    assert np.allclose(root2d["smooth_root_2d"], [[0.0, 0.0], [0.0, 2.0]])


def test_untimed_path_spreads_over_the_clip_like_v11():
    path = [{"pos": [float(i), 0.0, 0.0]} for i in range(4)]
    inp = ka.kimodo_inputs(_req(root_path=path))  # 2.0 s -> 60 samples
    assert inp.constraints[-1]["frame_indices"] == [0, 20, 39, 59]
    assert np.allclose(inp.canon.as_list(), [0.0, 0.0, np.pi / 2])


def test_mixed_timing_is_refused():
    path = [{"pos": [0.0, 0.0, 0.0], "time_s": 0.0}, {"pos": [1.0, 0.0, 0.0]}]
    with pytest.raises(ka.AdapterError, match="all have time_s or none"):
        ka.kimodo_inputs(_req(root_path=path))


def test_a_keyframe_round_trips_to_kimodo_model_space():
    npz = _npz()
    canon = ka.Canon(1.5, -2.0, 0.7)
    clip = ka.to_clip(npz, canon)
    kf = {
        "time_s": 10 / 30.0,
        "world_pos": clip["world_pos"][10].tolist(),
        "world_rot": clip["world_rot"][10].tolist(),
        "joints": None,
    }
    (c,) = ka.keyframe_constraints([kf], canon)
    assert c["type"] == "fullbody-global" and c["frame_indices"] == [10]
    pos = np.array(c["global_joints_positions"][0])
    rot = np.array(c["global_joints_rots"][0])
    assert np.abs(pos - npz["posed_joints"][10]).max() < 1e-4
    assert np.abs(rot - npz["global_rot_mats"][10]).max() < 1e-4
    hips = npz["posed_joints"][10][0]
    assert np.allclose(c["smooth_root_2d"][0], [hips[0], hips[2]], atol=1e-4)


def test_end_effector_keys_group_by_joints():
    clip = ka.to_clip(_npz(), ka.Canon())

    def kf(t, joints):
        return {
            "time_s": t,
            "world_pos": clip["world_pos"][0].tolist(),
            "world_rot": clip["world_rot"][0].tolist(),
            "joints": joints,
        }

    out = ka.keyframe_constraints(
        [kf(1.0, ["LeftHand"]), kf(0.5, ["LeftHand"]), kf(0.2, None)],
        ka.Canon(),
    )
    assert [c["type"] for c in out] == ["ee-global", "fullbody-global"]
    assert out[0]["joint_names"] == ["LeftHand"]
    assert out[0]["frame_indices"] == [15, 30]


def test_keyframe_needs_every_joint():
    kf = {
        "time_s": 0.0,
        "world_pos": [[0.0, 0.0, 0.0]],
        "world_rot": [np.eye(3).tolist()],
        "joints": None,
    }
    with pytest.raises(ka.AdapterError, match="every SOMA77 joint"):
        ka.keyframe_constraints([kf], ka.Canon())


def test_segments_options_and_constraint_order():
    native = [
        {"type": "root2d", "frame_indices": [0], "smooth_root_2d": [[0, 0]]}
    ]
    inp = ka.kimodo_inputs(
        _req(
            segments=[
                {"prompt": " walk ", "duration_s": 1.0},
                {"prompt": "run", "duration_s": 0.5},
            ],
            root_path=[{"pos": [0.0, 0.0, 0.0]}, {"pos": [0.0, 0.0, 1.0]}],
            options={"transition_frames": 7, "native_constraints": native},
        )
    )
    assert inp.texts == ["walk", "run"] and inp.num_frames == [30, 15]
    assert inp.transition_frames == 7
    assert [c["type"] for c in inp.constraints] == ["root2d", "root2d"]
    assert inp.constraints[0] is native[0]


def test_empty_prompt_is_refused():
    with pytest.raises(ka.AdapterError, match="segment 1 has no prompt"):
        ka.kimodo_inputs(_req(segments=[{"prompt": "  ", "duration_s": 1.0}]))


def test_canon_option_wins_over_the_path():
    inp = ka.kimodo_inputs(
        _req(
            options={"canon": [1.0, 2.0, 0.3]},
            root_path=[{"pos": [9.0, 0.0, 9.0]}],
        )
    )
    assert inp.canon.as_list() == [1.0, 2.0, 0.3]


def test_continue_from_maps_the_native_keys():
    cf = {"native_local_rot_mats": [[1]], "native_root_positions": [[2]]}
    inp = ka.kimodo_inputs(_req(continue_from=cf))
    assert inp.continue_from == {
        "local_rot_mats": [[1]],
        "root_positions": [[2]],
    }
    with pytest.raises(ka.AdapterError, match="native_root_positions"):
        ka.kimodo_inputs(_req(continue_from={"native_local_rot_mats": [[1]]}))
