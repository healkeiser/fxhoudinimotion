"""The ARDY server in MOCK_MODE (no torch, no ARDY, no GPU)."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import ardy_backend as ab
import pytest
from fastapi.testclient import TestClient

import fxmotion_server as fs
from fxmotion import clipformat

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "ardy_path.npz"


def _client(tmp_path):
    backend = ab.ArdyBackend(mock_clip=FIXTURE)
    return TestClient(fs.create_app(backend, output_dir=tmp_path, mock=True))


def _wait(client, job_id):
    end = time.monotonic() + 5
    while True:
        st = client.get("/jobs/%s" % job_id).json()
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        assert time.monotonic() < end
        time.sleep(0.02)


def test_mock_generate_returns_a_ardy_core_clip(tmp_path):
    req = {
        "segments": [{"prompt": "a person walks", "duration_s": 5.0}],
        "root_path": [{"pos": [1.0, 0.0, 2.0]}, {"pos": [4.0, 0.0, 2.0]}],
    }
    with _client(tmp_path) as client:
        health = client.get("/health").json()
        assert health["backend"] == "ardy" and health["fps"] == 20.0
        assert health["skeleton"] == "ardy_core"
        assert not health["capabilities"]["continue"]
        job = client.post("/generate", json=req).json()["job_id"]
        assert _wait(client, job)["status"] == "done"
        out = tmp_path / "got.npz"
        out.write_bytes(client.get("/jobs/%s/download" % job).content)
    clip = clipformat.load(out)
    assert str(clip["skeleton"]) == "ardy_core" and float(clip["fps"]) == 20.0
    assert clipformat.meta(clip, "source")["canon"][:2] == [1.0, 2.0]


def test_ardy_refuses_continue_from(tmp_path):
    req = {
        "segments": [{"prompt": "walk", "duration_s": 1.0}],
        "continue_from": {
            "native_local_rot_mats": [[1]],
            "native_root_positions": [[1]],
        },
    }
    with _client(tmp_path) as client:
        r = client.post("/generate", json=req)
    assert r.status_code == 422
    assert r.json()["detail"] == "ardy: not supported: continue_from"


def test_text_encoder_down_is_named():
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = "http://127.0.0.1:%d/" % port
    with pytest.raises(
        RuntimeError, match="text encoder unreachable at %s" % url
    ):
        ab.check_text_encoder(url, timeout=1.0)


def test_module_imports_and_builds_without_torch_or_ardy(tmp_path):
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join(
            [
                str(HERE.parent / "houdini" / "python"),
                str(HERE.parent / "server"),
            ]
        ),
        OUTPUT_DIR=str(tmp_path),
        MOCK_MODE="1",
    )
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['ardy'] = None; "
        "import ardy_backend; ardy_backend.make_app()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_history_defaults_to_4_s_and_fits_the_trained_window():
    # Horizon40 at 20 fps, 4-frame tokens: 200-frame window
    assert ab.history_frames(20.0, 40, 4) == 80  # the 4 s default
    assert ab.history_frames(20.0, 40, 4, 2.0) == 40
    assert ab.history_frames(20.0, 40, 4, 100.0) == 160  # window minus horizon
    assert ab.history_frames(20.0, 40, 4, 0.01) == 4  # at least one token
    assert ab.history_frames(20.0, 8, 4, 100.0) == 192


def test_segments_are_cut_into_horizons():
    assert ab.chunks([100, 30], 40, 4) == [
        (0, 0, 40),
        (0, 40, 40),
        (0, 80, 20),
        (1, 100, 30),
    ]


def test_chunks_and_histories_are_whole_tokens():
    # ARDY reshapes a history into 4-frame tokens: a 30-frame first
    # segment (1.5 s) used to hand the next chunk a 30-frame history
    got = ab.chunks([30, 50], 40, 4)
    assert got == [(0, 0, 32), (1, 32, 40), (1, 72, 8)]
    for durations in ([30, 50], [31, 49, 7], [3, 77], [62, 18, 45], [1, 1, 60]):
        got = ab.chunks(durations, 40, 4)
        assert [s for _, s, _ in got] == [0] + [s + n for _, s, n in got][:-1]
        assert sum(n for _, _, n in got) == sum(durations)
        assert all(n % 4 == 0 for _, _, n in got[:-1]), got


def test_a_segment_shorter_than_a_token_is_dropped():
    assert ab.chunks([30, 1, 49], 40, 4) == [
        (0, 0, 32),
        (2, 32, 40),
        (2, 72, 8),
    ]


def test_an_encoder_lost_mid_session_is_named():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = "http://127.0.0.1:%d/" % port

    def encode(texts):  # what ARDY's gradio client does when the port is dead
        raise ConnectionError("[WinError 10061] connection refused")

    with pytest.raises(RuntimeError, match="text encoder unreachable at"):
        ab.encode_prompts(encode, ["walk", "walk", "wave"], url)


def test_prompts_are_encoded_once_each():
    seen = []
    got = ab.encode_prompts(
        lambda t: seen.append(t) or t[0].upper(), ["a", "b", "a"], ""
    )
    assert got == {"a": "A", "b": "B"} and seen == [["a"], ["b"]]


def test_lookahead_fits_the_window_and_the_clip():
    # 80 history + 40 chunk leave room for one more horizon of 40
    assert ab.lookahead(80, 40, 40, 200, 500) == 40
    # history 160 + 40: the 200-frame window is full
    assert ab.lookahead(160, 40, 40, 200, 500) == 0
    # the clip ends 12 frames after this chunk
    assert ab.lookahead(80, 40, 40, 200, 12) == 12


def test_end_effector_sets_carry_the_hips():
    # ARDY asserts one Hips position per constrained frame (its own
    # LeftHandConstraintSet is ["LeftHand", "Hips"])
    assert ab.effector_joints(["LeftHand"]) == ["LeftHand", "Hips"]
    assert ab.effector_joints(["Hips", "LeftFoot"]) == ["Hips", "LeftFoot"]


def test_constraints_get_guidance():
    # a bare float is ARDY's text weight alone: its constraint weight is then
    # 0 and every root path and pose key is ignored (measured 1.7 m off)
    text, constraint = ab.CFG_WEIGHT
    assert text > 0 and constraint > 0
