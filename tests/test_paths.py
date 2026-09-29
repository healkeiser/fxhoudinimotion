"""Root-path thinning and scene time, against 1.1's behaviour."""

from fxmotion import paths


def test_thin_keeps_everything_when_asked_for_as_many_or_fewer_than_two():
    xz = [[float(i), 0.0] for i in range(5)]
    assert paths.thin(xz, 0) == [0, 1, 2, 3, 4]
    assert paths.thin(xz, 8) == [0, 1, 2, 3, 4]


def test_thin_by_arc_length_keeps_first_and_last():
    xz = [[float(i), 0.0] for i in range(11)]
    assert paths.thin(xz, 3) == [0, 4, 10]
    assert paths.thin(xz, 2) == [0, 10]


def test_scene_seconds_matches_v11_samples():
    # 1.1 to_sample: retime -> round((f - start) * clip_fps / scene_fps),
    # else int(f - start). 2.0 sends seconds and the server multiplies by 30,
    # so the sample *value* must match; compared before rounding, because
    # rounding two float spellings of an exact .5 can land either side.
    for scene_fps in (24, 25, 30):
        for f in range(1, 80):
            t = paths.scene_seconds(f, 1, scene_fps, 30, True)
            assert abs(t * 30 - (f - 1) * 30 / scene_fps) < 1e-9
            t = paths.scene_seconds(f, 1, scene_fps, 30, False)
            assert abs(t * 30 - (f - 1)) < 1e-9
    assert paths.scene_seconds(0, 10, 24, 30, True) == 0.0  # before the start
