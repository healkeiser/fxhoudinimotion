"""HTTP to an fxmotion model server. No hou; the session is injectable so
the tests need no network."""

from __future__ import annotations

from pathlib import Path

import requests


class ServerError(RuntimeError):
    """The server is unreachable or answered with an error."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class Rejected(ServerError):
    """422: the server understood the request and refuses it."""


def _base(url: str) -> str:
    return url.rstrip("/")


def _check(resp):
    if resp.status_code == 422:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        message = detail if isinstance(detail, str) else str(detail)
        raise Rejected(message, 422)
    if resp.status_code >= 400:
        raise ServerError(
            "%s %s: %s" % (resp.status_code, resp.reason, resp.text[:300]),
            resp.status_code,
        )
    return resp


def _call(method, url, **kw):
    try:
        return _check(method(url, **kw))
    except requests.RequestException as e:
        raise ServerError("server unreachable at %s: %s" % (url, e)) from e


def health(url, session=requests) -> dict:
    return _call(session.get, _base(url) + "/health", timeout=5).json()


def submit(url, payload, session=requests) -> str:
    target = _base(url) + "/generate"
    return _call(session.post, target, json=payload, timeout=60).json()[
        "job_id"
    ]


def job(url, job_id, session=requests) -> dict:
    target = "%s/jobs/%s" % (_base(url), job_id)
    return _call(session.get, target, timeout=10).json()


def cancel(url, job_id, session=requests) -> dict:
    target = "%s/jobs/%s/cancel" % (_base(url), job_id)
    return _call(session.post, target, timeout=10).json()


def download(url, job_id, dest_dir, session=requests) -> str:
    """Stream the job's clip to <dest_dir>/<job_id>.npz; returns that path
    with forward slashes, Houdini's own convention."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / ("%s.npz" % job_id)
    target = "%s/jobs/%s/download" % (_base(url), job_id)
    try:
        with session.get(target, timeout=120, stream=True) as resp:
            _check(resp)
            with open(out, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    except requests.RequestException as e:
        raise ServerError("download failed from %s: %s" % (target, e)) from e
    return out.as_posix()
