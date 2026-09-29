"""ARDY on the real model: segments chained, constraints kept, long timelines.

    <ardy>/.venv/Scripts/python.exe tests/ardy_integration.py

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
backend.load("")
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
    {"pos": [4.0, 0.0, 2.5], "time_s": 5.5},
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
print("ok")
