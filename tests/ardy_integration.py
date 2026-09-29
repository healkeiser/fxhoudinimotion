"""ARDY on the real model: segments chained, constraints kept, long timelines.

    <ardy>/.venv/Scripts/python.exe tests/ardy_integration.py [model]

Needs the Kimodo text-encoder container on http://127.0.0.1:9550/ and a GPU.
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "server"), str(REPO / "houdini" / "python")]

import ardy_backend as ab  # noqa: E402

import fxmotion_server as fs  # noqa: E402

backend = ab.ArdyBackend()
backend.load(
    sys.argv[1] if len(sys.argv) > 1 else ""
)  # a model name, e.g. Horizon8
progress = fs.Progress({})


def run(req):
    t0 = time.perf_counter()
    clip = backend.generate(fs.GenerateRequest(**req), progress)
    return clip, time.perf_counter() - t0


def steps(pos):
    return np.linalg.norm(np.diff(pos, axis=0), axis=-1).max(axis=1)


# Two segments and a timed path across both, off the origin and heading +X.
path = [
    {"pos": [1.0, 0.0, 2.0], "time_s": 0.0},
    {"pos": [3.0, 0.0, 2.0], "time_s": 2.5},
    {"pos": [3.5, 0.0, 4.5], "time_s": 5.5},  # a left turn, not a natural walk
]
clip, secs = run(
    {
        "segments": [
            {"prompt": "A person walks forward.", "duration_s": 3.0},
            {"prompt": "A person sits down on the ground.", "duration_s": 3.0},
        ],
        "root_path": path,
        "seed": 0,
    }
)
pos = clip["world_pos"]
assert pos.shape == (120, 27, 3), pos.shape
s = steps(pos)
print(
    "two segments: %.1f s, seam step %.3f m vs median %.3f m"
    % (secs, s[59], np.median(s))
)
assert s[59] < 3 * np.median(s), "the seam jumps"
for p in path:
    f = min(int(round(p["time_s"] * 20)), 119)
    err = np.hypot(pos[f, 0, 0] - p["pos"][0], pos[f, 0, 2] - p["pos"][2])
    print("  waypoint at %.1f s: %.2f m off" % (p["time_s"], err))
    assert err < 0.25, "the path is not followed"
assert pos[-1, 0, 1] < 0.6, "the second prompt (sit down) was not followed"

# Review Focus 1: a 12 s, three-segment timeline, past ARDY's 10 s window.
clip, secs = run(
    {
        "segments": [
            {"prompt": "A person walks forward.", "duration_s": 4.0},
            {
                "prompt": "A person turns around and walks back.",
                "duration_s": 4.0,
            },
            {"prompt": "A person waves with both hands.", "duration_s": 4.0},
        ],
        "seed": 0,
    }
)
s = steps(clip["world_pos"])
early, late = np.median(s[:40]), np.median(s[-40:])
jitter = np.median(np.abs(np.diff(s[-40:])))
print(
    "12 s: %.1f s, step median first 2 s %.3f, last 2 s %.3f, late jitter %.4f"
    % (secs, early, late, jitter)
)
assert jitter < 0.02, "the end of a long timeline jitters"
# Pose keys (full-body on a chunk boundary, a raised left hand), taken from
# a first walk so they are reachable. Measured at the keyed frames.
walk = {"segments": [{"prompt": "A person walks forward.", "duration_s": 6.0}]}
ref, _ = run(dict(walk, seed=1))
names = [str(n) for n in ref["joint_names"]]
hand = names.index("LeftHand")
hand_end = names.index("LeftHandEnd")
raised = ref["world_pos"][100].copy()
raised[[hand, hand_end], 1] += 0.3  # the key constrains both
keys = [
    {
        "time_s": 40 / 20,
        "world_pos": ref["world_pos"][40].tolist(),
        "world_rot": ref["world_rot"][40].tolist(),
        "joints": None,
    },
    {
        "time_s": 100 / 20,
        "world_pos": raised.tolist(),
        "world_rot": ref["world_rot"][100].tolist(),
        "joints": ["LeftHand"],
    },
]
clip, secs = run(dict(walk, seed=2, keyframes=keys))
pos = clip["world_pos"]
body = np.linalg.norm(pos[40] - ref["world_pos"][40], axis=-1).max()
reach = np.linalg.norm(pos[100, hand] - raised[hand])


def native(req):
    """ARDY's own single call (scripts/generate.py's), same keys and seed:
    the reference for what chunking may add at its seams."""
    import diffusion_adapter as da
    import torch
    from ardy.motion_rep.tools import length_to_mask
    from ardy.tools import seed_everything

    m, dev = backend._model, backend._device
    inp = da.model_inputs(fs.GenerateRequest(**req).model_dump(), ab.ARDY)
    n = sum(inp.num_frames)
    cons = ab._constraint_objects(inp.constraints, m.skeleton)
    obs, mask = m.motion_rep.create_conditions_from_constraints_batched(
        cons, torch.tensor([n], device=dev), to_normalize=True, device=dev
    )
    seed_everything(int(req["seed"]))
    with torch.no_grad():
        out = m(
            inp.texts[:1],
            n,
            num_denoising_steps=int(m.diffusion.num_base_steps),
            cfg_weight=ab.CFG_WEIGHT,
            pad_mask=length_to_mask(torch.tensor([n], device=dev)),
            first_heading_angle=torch.zeros(1, device=dev),
            motion_mask=mask,
            observed_motion=obs,
            crop_history_length=ab.history_frames(
                20.0, m.gen_horizon_len, m.num_frames_per_token, 100
            ),
        )
    return (
        m.motion_rep.inverse(out, is_normalized=True)["posed_joints"][0]
        .cpu()
        .numpy()
    )


keyed = dict(walk, seed=2, keyframes=keys)
ref_pos = native(keyed)
s, r = steps(pos), steps(ref_pos)
reach_native = np.linalg.norm(ref_pos[100, hand] - raised[hand])
worst = max(s[f - 4 : f + 4].max() / r[f - 4 : f + 4].max() for f in (40, 100))
print(
    "keys: %.1f s, full-body key %.3f m off (worst joint), hand key %.3f m off "
    "(native call %.3f m), worst step around a key %.2fx the native call"
    % (secs, body, reach, reach_native, worst)
)
assert body < 0.1, "the full-body key is not reached"
# the hand: no worse than ARDY's own single call on the same key
assert reach < max(0.1, 1.25 * reach_native), (
    "the end-effector key is not reached"
)
assert worst < 1.25, "chunking adds a pop around a key"
print("ok")
