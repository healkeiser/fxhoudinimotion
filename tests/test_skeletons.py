"""The skeleton registry: one source of truth for joints, rest poses and rest
geometry, shared by the model servers and the Houdini nodes."""

import numpy as np
import pytest

from fxmotion import skeletons


def test_soma77_is_one_tree_in_parent_first_order():
    s = skeletons.get("soma77")
    assert len(s.joint_names) == len(s.parents) == 77
    assert [i for i, p in enumerate(s.parents) if p < 0] == [0]
    assert all(p < i for i, p in enumerate(s.parents) if p >= 0)
    assert s.index("Hips") == 0


def test_soma77_rest_rotations_are_orthonormal():
    r = skeletons.get("soma77").rest_rot
    assert r.shape == (77, 3, 3)
    err = np.abs(r @ np.transpose(r, (0, 2, 1)) - np.eye(3)).max()
    assert err < 1e-5


def test_soma77_rest_geometry_ships_with_the_package():
    s = skeletons.get("soma77")
    assert s.skin.is_file() and s.skin.stat().st_size > 100_000
    assert s.capture_pose.is_file()


def test_unknown_skeleton_names_the_known_ones():
    with pytest.raises(KeyError, match="soma77"):
        skeletons.get("nope")


def test_get_returns_the_same_object():
    assert skeletons.get("soma77") is skeletons.get("soma77")


def test_ardy_core_is_a_parent_first_t_pose():
    s = skeletons.get("ardy_core")
    assert len(s.joint_names) == 27 and s.index("Hips") == 0
    assert all(p < i for i, p in enumerate(s.parents) if p >= 0)
    assert np.allclose(s.rest_rot, np.eye(3))  # world-axis-aligned frames
    hand, arm = s.rest_pos[s.index("LeftHand")], s.rest_pos[s.index("LeftArm")]
    assert (
        abs(hand[1] - arm[1]) < 0.05 and hand[0] > arm[0] + 0.4
    )  # arm out, level


def test_ardy_core_rest_geometry_ships_with_the_package():
    s = skeletons.get("ardy_core")
    assert s.skin.is_file() and s.skin.stat().st_size > 50_000
    assert s.capture_pose.is_file()


def test_registry_lists_both_skeletons():
    assert skeletons.names() == ("soma77", "ardy_core")
