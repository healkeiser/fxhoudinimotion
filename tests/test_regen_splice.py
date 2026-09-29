"""Checks for fxmotion.timeline.regen's pure functions. Plain python: python
tests/test_regen_splice.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "houdini" / "python")
)

from cliptools import tiny  # noqa: E402

from fxmotion import clipformat  # noqa: E402
from fxmotion.timeline.regen import (  # noqa: E402
    continue_payload,
    cut_sample,
    per_sample_keys,
    samples_for,
    splice,
)

# scene rates a shot gets cut at, against the 30 the SOMA models generate at,
# plus a couple of mismatched pairs to keep the rounding honest
FPS_PAIRS = ((24, 30), (25, 30), (30, 30), (24, 20), (60, 30))


def test_cut_lines_up_with_what_the_server_generates():
    """The slot a segment occupies in the clip has to be exactly as long as
    the segment the server generates for it (int(duration * fps) per
    segment, mirrored by samples_for), or every single-segment regen drops a
    sample."""
    for scene, source in FPS_PAIRS:
        for f in range(4, 120):
            frames = [f, f + 1, f + 7, f + 13]
            for i in range(1, len(frames)):
                cut = cut_sample(frames, i, scene, source)
                resume = cut_sample(frames, i + 1, scene, source)
                nf = samples_for(frames[i], scene, source)
                assert resume - cut == nf


def _clip(length, value=None):
    """A valid clip whose per-sample values say which sample they came from,
    so a misplaced join shows up as a wrong number rather than a right
    shape."""
    col = (
        np.arange(length, dtype=float)
        if value is None
        else np.full(length, value)
    )
    pos = np.zeros((length, 2, 3))
    pos[:, :, 0] = col[:, None]
    return tiny(
        length,
        world_pos=pos,
        native={"root_positions": np.repeat(col[:, None], 3, axis=1)},
    )


def test_per_sample_keys():
    assert set(per_sample_keys(_clip(4))) == {
        "world_pos",
        "world_rot",
        "native_root_positions",
    }


def test_single_segment_splice_keeps_the_clip_length_and_stays_valid():
    scene, source, n = 24, 30, 5
    frames = [31, 47, 23]
    have = cut_sample(frames, len(frames), scene, source)
    old = _clip(have)
    cut = cut_sample(frames, 1, scene, source)
    resume = cut_sample(frames, 2, scene, source)
    nf = samples_for(frames[1], scene, source)
    new = _clip(n + nf, value=-1.0)  # the blended transition, then the segment

    merged = splice(old, new, cut, n, resume_at=resume)
    clipformat.validate(merged)
    assert clipformat.frame_count(merged) == have
    x = merged["world_pos"][:, 0, 0]
    assert np.array_equal(x[: cut - n], np.arange(cut - n))
    assert np.array_equal(x[cut + nf :], np.arange(resume, have))
    assert (x[cut - n : cut + nf] == -1.0).all()


def test_run_to_the_end_splice_drops_the_old_tail():
    scene, source, n = 24, 30, 5
    frames = [31, 47, 23]
    have = cut_sample(frames, len(frames), scene, source)
    cut = cut_sample(frames, 1, scene, source)
    merged = splice(_clip(have), _clip(n + have - cut, -1.0), cut, n)
    assert clipformat.frame_count(merged) == have


def test_continue_payload_sends_the_native_tail():
    clip = tiny(
        40,
        native={
            "root_positions": np.arange(120, dtype=float).reshape(40, 3),
            "local_rot_mats": np.tile(np.eye(3), (40, 2, 1, 1)),
        },
    )
    body = continue_payload(clip, cut=30, n=5)
    assert set(body) == {"native_local_rot_mats", "native_root_positions"}
    want = clip["native_root_positions"][25:30].tolist()
    assert body["native_root_positions"] == want


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
