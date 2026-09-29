"""The shared server contract, driven with a fake backend: no model, no GPU."""

import threading
import time

import numpy as np
import pytest
from cliptools import tiny
from fastapi.testclient import TestClient

import fxmotion_server as fs
from fxmotion import clipformat

REQ = {"segments": [{"prompt": "a person walks", "duration_s": 2.0}]}


class FakeBackend(fs.Backend):
    name = "fake"
    skeleton = "test"
    fps = 30.0
    models = ("m1", "m2")
    default_model = "m1"
    capabilities = {
        "text": True,
        "styles": ["walk"],
        "root_path": True,
        "keyframes": False,
        "effectors": False,
        "continue": False,
        "live": False,
    }

    def __init__(self):
        self.calls, self.loaded, self.unloaded = [], [], 0
        self.gate, self.fail = None, None

    def validate(self, req):
        if any(s.prompt == "" for s in req.segments):
            raise ValueError("fake: empty prompt")

    def load(self, model):
        self.loaded.append(model)

    def unload(self):
        self.unloaded += 1

    def generate(self, req, progress):
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError(self.fail)
        progress.set(0.5, "working")
        self.calls.append(req)
        return tiny(4)

    def mock(self, req):
        return tiny(2)


def _serve(tmp_path, **kw):
    backend = FakeBackend()
    app = fs.create_app(backend, output_dir=tmp_path, **kw)
    return backend, app


@pytest.fixture
def server(tmp_path):
    backend, app = _serve(tmp_path)
    with TestClient(app) as client:
        yield client, backend, app


def wait(client, job_id, timeout=5.0):
    end = time.monotonic() + timeout
    while True:
        st = client.get("/jobs/%s" % job_id).json()
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        assert time.monotonic() < end, "job still %s" % st["status"]
        time.sleep(0.02)


def submit(client, req=REQ):
    r = client.post("/generate", json=req)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def test_health_reports_the_backend(server):
    client, _, _ = server
    h = client.get("/health").json()
    assert h["backend"] == "fake" and h["fps"] == 30.0
    assert h["models"] == ["m1", "m2"]
    assert h["capabilities"]["styles"] == ["walk"]
    assert h["mock_mode"] is False and h["loaded_model"] is None


def test_generate_then_download_a_valid_clip(server, tmp_path):
    client, backend, _ = server
    st = wait(client, submit(client))
    assert st["status"] == "done" and st["frames"] == 4 and st["joints"] == 2
    assert st["cached"] is False and backend.loaded == ["m1"]
    blob = client.get("/jobs/%s/download" % st["job_id"]).content
    path = tmp_path / "got.npz"
    path.write_bytes(blob)
    assert clipformat.frame_count(clipformat.load(path)) == 4


def test_unsupported_controls_are_refused_by_name(server):
    client, _, _ = server
    key = {
        "time_s": 0.0,
        "world_pos": [[0, 0, 0]],
        "world_rot": [np.eye(3).tolist()],
    }
    r = client.post("/generate", json=dict(REQ, keyframes=[key]))
    assert r.status_code == 422
    assert r.json()["detail"] == "fake: not supported: keyframes"
    r = client.post(
        "/generate", json={"segments": [{"style": "run", "duration_s": 1.0}]}
    )
    assert r.status_code == 422
    assert "style 'run' (known: walk)" in r.json()["detail"]


def test_backend_validation_is_a_422(server):
    client, _, _ = server
    r = client.post(
        "/generate", json={"segments": [{"prompt": "", "duration_s": 1.0}]}
    )
    assert r.status_code == 422 and r.json()["detail"] == "fake: empty prompt"


def test_identical_requests_hit_the_cache_unless_forced(server):
    client, backend, _ = server
    wait(client, submit(client))
    st = wait(client, submit(client))
    assert st["cached"] is True and len(backend.calls) == 1
    wait(client, submit(client, dict(REQ, force=True)))
    assert len(backend.calls) == 2


def test_non_ascii_prompt_is_cached(server):
    client, backend, _ = server
    prompt = "une personne marche é 一"
    req = {"segments": [{"prompt": prompt, "duration_s": 1.0}]}
    assert wait(client, submit(client, req))["status"] == "done"
    assert wait(client, submit(client, req))["cached"] is True
    assert backend.calls[0].segments[0].prompt.endswith("一")


def test_a_failure_is_reported(server):
    client, backend, _ = server
    backend.fail = "boom"
    st = wait(client, submit(client))
    assert st["status"] == "failed" and "boom" in st["error"]


def test_cancel_a_queued_job(server):
    client, backend, _ = server
    backend.gate = threading.Event()
    first = submit(client)
    second = submit(client, dict(REQ, force=True))
    cancelled = client.post("/jobs/%s/cancel" % second).json()
    assert cancelled["status"] == "cancelled"
    backend.gate.set()
    assert wait(client, first)["status"] == "done"
    assert wait(client, second)["status"] == "cancelled"
    assert len(backend.calls) == 1


def test_download_errors(server):
    client, backend, _ = server
    assert client.get("/jobs/nope/download").status_code == 404
    backend.gate = threading.Event()
    job = submit(client)
    assert client.get("/jobs/%s/download" % job).status_code == 409
    backend.gate.set()
    wait(client, job)


def test_switching_models_unloads_the_previous_one(server):
    client, backend, _ = server
    wait(client, submit(client))
    wait(client, submit(client, dict(REQ, model="m2")))
    assert backend.loaded == ["m1", "m2"] and backend.unloaded == 1


def test_idle_unload(tmp_path):
    backend, app = _serve(tmp_path, idle_unload_s=10.0)
    with TestClient(app) as client:
        wait(client, submit(client))
        assert app.state.reap_idle(now=time.monotonic() + 11) is True
        assert backend.unloaded == 1
        assert app.state.reap_idle(now=time.monotonic() + 11) is False


def test_mock_mode_never_loads_a_model(tmp_path):
    backend, app = _serve(tmp_path, mock=True)
    with TestClient(app) as client:
        st = wait(client, submit(client))
        assert st["frames"] == 2 and backend.loaded == []
        assert client.get("/health").json()["mock_mode"] is True
