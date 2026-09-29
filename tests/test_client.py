"""fxmotion.client against a fake requests session: no network."""

import pytest
import requests

from fxmotion import client


class FakeResponse:
    def __init__(self, status=200, body=None, content=b"", reason="OK"):
        self.status_code, self._body, self.content = status, body, content
        self.reason, self.text = reason, str(body)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def iter_content(self, chunk_size=1):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def _do(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if self.error:
            raise self.error
        return self.response

    def get(self, url, **kw):
        return self._do("GET", url, **kw)

    def post(self, url, **kw):
        return self._do("POST", url, **kw)


def test_health_strips_the_trailing_slash():
    s = FakeSession(FakeResponse(body={"backend": "kimodo"}))
    assert client.health("http://h:8001/", session=s) == {"backend": "kimodo"}
    assert s.calls[0][1] == "http://h:8001/health"


def test_unreachable_server_is_a_server_error():
    s = FakeSession(error=requests.ConnectionError("refused"))
    with pytest.raises(client.ServerError, match="unreachable at http://h"):
        client.health("http://h", session=s)


def test_a_422_is_rejected_with_the_servers_words():
    body = {"detail": "kimodo: segment 1 has no prompt"}
    s = FakeSession(FakeResponse(422, body, reason="Unprocessable"))
    with pytest.raises(client.Rejected, match="segment 1 has no prompt"):
        client.submit("http://h", {"segments": []}, session=s)


def test_submit_returns_the_job_id():
    s = FakeSession(FakeResponse(202, {"job_id": "abc", "status": "queued"}))
    assert client.submit("http://h", {"x": 1}, session=s) == "abc"
    assert s.calls[0][2]["json"] == {"x": 1}


def test_a_lost_job_carries_its_status():
    body = {"detail": "Job x not found."}
    s = FakeSession(FakeResponse(404, body, reason="Not Found"))
    with pytest.raises(client.ServerError) as e:
        client.job("http://h", "x", session=s)
    assert e.value.status == 404


def test_download_creates_the_folder(tmp_path):
    s = FakeSession(FakeResponse(content=b"NPZBYTES"))
    dest = tmp_path / "not" / "there"
    out = client.download("http://h", "job1", str(dest), session=s)
    assert out == (dest / "job1.npz").as_posix()
    assert (dest / "job1.npz").read_bytes() == b"NPZBYTES"
