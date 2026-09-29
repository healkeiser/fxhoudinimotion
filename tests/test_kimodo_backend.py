"""The Kimodo server end to end in MOCK_MODE (no torch, no GPU), plus its
progress stand-in."""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import fxmotion_server as fs
import kimodo_backend as kb
from fxmotion import clipformat

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "kimodo_stop.npz"


def _client(tmp_path):
    backend = kb.KimodoBackend(mock_clip=FIXTURE)
    return TestClient(fs.create_app(backend, output_dir=tmp_path, mock=True))


def _wait(client, job_id):
    end = time.monotonic() + 5
    while True:
        st = client.get("/jobs/%s" % job_id).json()
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        assert time.monotonic() < end
        time.sleep(0.02)


def test_mock_generate_returns_a_soma77_clip_on_the_path(tmp_path):
    req = {
        "segments": [{"prompt": "a person walks", "duration_s": 1.5}],
        "root_path": [{"pos": [1.0, 0.0, 2.0]}, {"pos": [1.0, 0.0, 5.0]}],
    }
    with _client(tmp_path) as client:
        caps = client.get("/health").json()["capabilities"]
        assert caps["keyframes"] and caps["continue"] and not caps["live"]
        job = client.post("/generate", json=req).json()["job_id"]
        st = _wait(client, job)
        assert st["status"] == "done", st
        out = tmp_path / "got.npz"
        out.write_bytes(client.get("/jobs/%s/download" % job).content)
    clip = clipformat.load(out)
    assert str(clip["skeleton"]) == "soma77"
    src = clipformat.meta(clip, "source")
    assert src["canon"] == [1.0, 2.0, 0.0] and src["backend"] == "kimodo"
    with np.load(FIXTURE) as z:
        root = z["posed_joints"][0, 0]
    want = root + [1.0, 0.0, 2.0]
    assert np.allclose(clip["world_pos"][0, 0], want, atol=1e-5)


def test_kimodo_refuses_styles_and_empty_prompts(tmp_path):
    with _client(tmp_path) as client:
        r = client.post(
            "/generate",
            json={"segments": [{"style": "walk", "duration_s": 1.0}]},
        )
        assert r.status_code == 422
        assert r.json()["detail"] == "kimodo: not supported: segment 1: styles"
        r = client.post(
            "/generate",
            json={"segments": [{"prompt": " ", "duration_s": 1.0}]},
        )
        assert r.status_code == 422
        assert r.json()["detail"] == "kimodo: segment 1 has no prompt"


class _Recorder:
    def __init__(self):
        self.calls = []

    def set(self, fraction, phase=None, est_s=None, span=None):
        self.calls.append((round(fraction, 3), phase, est_s))


def test_sampler_walks_encode_denoise_post_and_learns_the_encode_time():
    backend = kb.KimodoBackend(encode_est_s=30.0)
    rec = _Recorder()
    sampler = kb._Sampler(backend, rec, expected=2)
    list(sampler(range(4)))
    list(sampler(range(4)))
    phases = [p for _, p, _ in rec.calls]
    assert phases[0] == "encoding text"
    assert "denoising segment 1/2" in phases
    assert "denoising segment 2/2" in phases
    assert phases[-1] == "post-processing"
    assert backend.encode_est_s < 30.0  # replaced by the measured encode


def test_module_imports_and_builds_without_torch(tmp_path):
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
        "import sys; sys.modules['torch'] = None; sys.modules['kimodo'] = None; "
        "import kimodo_backend; kimodo_backend.make_app()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
