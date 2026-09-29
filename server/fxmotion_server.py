"""The HTTP contract every fxmotion model server speaks, in one module.

Spec: superpowers/specs/2026-09-29-fxhoudinimotion-design.md, section 4. A
model server is a Backend subclass plus `create_app(MyBackend(), ...)`. Jobs,
the request cache, progress, cancel and idle unloading live here, so each
model only says how to run itself. Times are seconds and positions Houdini
space everywhere in a request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fxmotion import clipformat

log = logging.getLogger("fxmotion_server")


class Segment(BaseModel):
    duration_s: float
    prompt: Optional[str] = None
    style: Optional[str] = None


class PathPoint(BaseModel):
    pos: list[float]  # [x, y, z], Houdini space
    time_s: Optional[float] = None  # all points timed, or none


class Keyframe(BaseModel):
    time_s: float
    world_pos: list[list[float]]  # (J, 3)
    world_rot: list[list[list[float]]]  # (J, 3, 3), KineFX row-vector
    joints: Optional[list[str]] = None  # None: whole body; names: effectors


class GenerateRequest(BaseModel):
    segments: list[Segment]
    root_path: Optional[list[PathPoint]] = None
    keyframes: Optional[list[Keyframe]] = None
    continue_from: Optional[dict] = None
    seed: Optional[int] = None
    model: str = ""
    force: bool = False  # bypass the cache
    options: dict = Field(default_factory=dict)  # backend-specific knobs


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | failed | cancelled
    prompt: Optional[str] = None
    frames: Optional[int] = None
    joints: Optional[int] = None
    error: Optional[str] = None
    elapsed: Optional[float] = None
    cached: Optional[bool] = None
    progress: Optional[float] = None
    phase: Optional[str] = None


class Progress:
    """Written by the backend on its worker thread, read by the event loop.
    Plain dict writes, no lock: each one is atomic in CPython."""

    def __init__(self, job: dict):
        self.job = job

    def set(self, fraction, phase=None, est_s=None, span=None) -> None:
        """`fraction` of the whole job done. `est_s` and `span` describe a
        phase that reports nothing while it runs (Kimodo's text encoding):
        it should take about est_s seconds and covers `span` of the job, so
        the displayed progress creeps across it instead of sitting still."""
        job = self.job
        if phase != job.get("phase"):
            job["phase_started"] = time.monotonic()
        job.update(
            progress=min(0.99, max(0.0, float(fraction))),
            phase=phase,
            est_s=est_s,
            span=span,
        )

    @property
    def cancelled(self) -> bool:
        return self.job.get("status") == "cancelled"


class Backend:
    """One model. Every method runs on a worker thread, one at a time."""

    name = "backend"
    skeleton = ""
    fps = 30.0
    models: tuple = ()
    default_model = ""
    # text, styles (list of names), root_path, keyframes, effectors,
    # continue, live
    capabilities: dict = {}

    def validate(self, req: GenerateRequest) -> None:
        """Raise ValueError for a request this model cannot run (422)."""

    def load(self, model: str) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def generate(self, req: GenerateRequest, progress: Progress) -> dict:
        raise NotImplementedError

    def mock(self, req: GenerateRequest) -> dict:
        raise NotImplementedError

    def describe(self, req: GenerateRequest) -> str:
        return " | ".join(s.prompt or s.style or "" for s in req.segments)


def unsupported(req: GenerateRequest, backend: Backend) -> list:
    """What the request asks for that the backend's capabilities lack."""
    caps = backend.capabilities
    styles = caps.get("styles") or []
    out = []
    for i, seg in enumerate(req.segments, start=1):
        if seg.prompt is not None and not caps.get("text"):
            out.append("segment %d: text prompts" % i)
        if seg.style is not None:
            if not styles:
                out.append("segment %d: styles" % i)
            elif seg.style not in styles:
                out.append(
                    "segment %d: style %r (known: %s)"
                    % (i, seg.style, ", ".join(styles))
                )
    if req.root_path and not caps.get("root_path"):
        out.append("root_path")
    kinds = {
        "effectors" if kf.joints else "keyframes" for kf in req.keyframes or []
    }
    out += sorted(k for k in kinds if not caps.get(k))
    if req.continue_from and not caps.get("continue"):
        out.append("continue_from")
    return out


def _elapsed(job: dict) -> float:
    now = time.monotonic()
    return round(now - job.get("started_at", now), 1)


def _display_progress(job: dict):
    prog, est = job.get("progress"), job.get("est_s")
    if prog is None or not est:
        return prog
    waited = time.monotonic() - job.get("phase_started", time.monotonic())
    span = job.get("span") or 0.0
    return min(0.99, prog + span * 0.9 * min(1.0, waited / est))


def create_app(
    backend: Backend,
    *,
    output_dir,
    mock: bool = False,
    idle_unload_s: float = 0.0,
    preload: bool = False,
) -> FastAPI:
    """The FastAPI app for one backend. `idle_unload_s` > 0 frees the model
    after that long without a job; `preload` loads the default model at
    startup (ignored in mock mode)."""
    # Mock clips live apart from the real cache, so switching MOCK_MODE off
    # never serves a mock clip as a cached result.
    output_dir = Path(output_dir) / ("mock" if mock else "")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs: dict = {}
    tasks: set = set()  # asyncio keeps only weak references to tasks
    lock = asyncio.Lock()  # one GPU, one generation at a time
    state = {"loaded": None, "last_used": time.monotonic()}

    def cache_key(req: GenerateRequest) -> str:
        payload = req.model_dump(exclude={"force"})
        payload["backend"] = backend.name
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def ensure_loaded(model: str) -> None:
        model = model or backend.default_model
        if state["loaded"] != model:
            if state["loaded"] is not None:
                backend.unload()
                # a load that raises below must not leave the old name
                # standing for a model that is gone
                state["loaded"] = None
            log.info("[LOAD] %s", model)
            backend.load(model)
            state["loaded"] = model
        state["last_used"] = time.monotonic()

    def reap_idle(now=None) -> bool:
        now = time.monotonic() if now is None else now
        if (
            idle_unload_s > 0
            and state["loaded"] is not None
            and not lock.locked()
            and now - state["last_used"] >= idle_unload_s
        ):
            log.info(
                "[UNLOAD] %s after %.0f s idle", state["loaded"], idle_unload_s
            )
            backend.unload()
            state["loaded"] = None
            return True
        return False

    async def reaper():
        while True:
            await asyncio.sleep(max(1.0, idle_unload_s / 4))
            reap_idle()

    @asynccontextmanager
    async def lifespan(app):
        if preload and not mock:
            await asyncio.to_thread(ensure_loaded, "")
        task = asyncio.create_task(reaper()) if idle_unload_s > 0 else None
        yield
        if task is not None:
            task.cancel()

    app = FastAPI(title="fxmotion %s server" % backend.name, lifespan=lifespan)
    app.state.reap_idle = reap_idle

    def produce(req: GenerateRequest, job: dict) -> dict:
        if mock:
            return backend.mock(req)
        ensure_loaded(req.model)
        try:
            return backend.generate(req, Progress(job))
        finally:
            state["last_used"] = time.monotonic()

    def finish(job: dict, path: Path, cached: bool) -> None:
        with np.load(path) as z:
            frames, joints = z["world_pos"].shape[:2]
        job.update(
            status="done",
            path=str(path),
            frames=int(frames),
            joints=int(joints),
            cached=cached,
            elapsed=_elapsed(job),
            progress=1.0,
            phase=None,
            est_s=None,
        )

    async def run(job_id: str, req: GenerateRequest) -> None:
        job = jobs[job_id]
        if job["status"] == "cancelled":
            return
        job.update(status="running", progress=0.0)
        path = output_dir / ("%s.npz" % cache_key(req))
        try:
            if not mock and not req.force and path.exists():
                finish(job, path, cached=True)
                return
            async with lock:
                if job["status"] == "cancelled":
                    return
                log.info("[GEN] %s %s", job_id[:8], backend.describe(req))
                clip = await asyncio.to_thread(produce, req, job)
            if job["status"] == "cancelled":  # discard the result
                return
            await asyncio.to_thread(clipformat.save, path, clip)
            finish(job, path, cached=False)
        except Exception as exc:  # never leave a job stuck in "running"
            if job.get("status") != "cancelled":
                log.exception("[FAIL] %s", job_id[:8])
                job.update(
                    status="failed",
                    error=str(exc)[-500:],
                    elapsed=_elapsed(job),
                )

    @app.get("/health")
    def health() -> dict:
        return {
            "backend": backend.name,
            "skeleton": backend.skeleton,
            "fps": backend.fps,
            "models": list(backend.models),
            "default_model": backend.default_model,
            "capabilities": backend.capabilities,
            "mock_mode": mock,
            "loaded_model": state["loaded"],
        }

    @app.post("/generate", status_code=202)
    async def generate(req: GenerateRequest) -> JobStatus:
        missing = unsupported(req, backend)
        if missing:
            raise HTTPException(
                422,
                "%s: not supported: %s" % (backend.name, "; ".join(missing)),
            )
        try:
            backend.validate(req)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        job_id = uuid.uuid4().hex
        desc = backend.describe(req)
        jobs[job_id] = {
            "status": "queued",
            "started_at": time.monotonic(),
            "prompt": desc,
        }
        task = asyncio.create_task(run(job_id, req))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return JobStatus(job_id=job_id, status="queued", prompt=desc)

    def status(job_id: str) -> JobStatus:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job %s not found." % job_id)
        elapsed = job.get("elapsed")
        if elapsed is None:
            elapsed = _elapsed(job)
        return JobStatus(
            job_id=job_id,
            status=job["status"],
            prompt=job.get("prompt"),
            frames=job.get("frames"),
            joints=job.get("joints"),
            error=job.get("error"),
            elapsed=elapsed,
            cached=job.get("cached"),
            progress=_display_progress(job),
            phase=job.get("phase"),
        )

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> JobStatus:
        return status(job_id)

    @app.get("/jobs/{job_id}/download")
    def download(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job %s not found." % job_id)
        if job["status"] != "done":
            raise HTTPException(409, "Job %s is %s." % (job_id, job["status"]))
        return FileResponse(
            job["path"],
            media_type="application/octet-stream",
            filename="%s.npz" % job_id,
        )

    @app.post("/jobs/{job_id}/cancel")
    async def cancel(job_id: str) -> JobStatus:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job %s not found." % job_id)
        if job["status"] in ("queued", "running"):
            # in-process inference cannot be interrupted: a queued job is
            # skipped and a running one's result discarded
            job.update(status="cancelled", elapsed=_elapsed(job))
            log.info("[CANCEL] %s", job_id[:8])
        return status(job_id)

    return app
