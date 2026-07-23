"""In-memory async job store for /infer.

Single-process, not persisted. Acceptable for a single-instance deployment,
but a job's state is lost on restart/recycle (callers must treat an unknown
job id as failed, not as "still processing"). Terminal jobs are swept on a
short TTL (not evicted on first read — see take_job) and abandoned
still-processing jobs are swept on a longer TTL, to avoid unbounded growth on
a memory-constrained instance.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Literal

JobStatus = Literal["processing", "completed", "failed"]

_PROCESSING_TTL_SECONDS = 3600
_TERMINAL_TTL_SECONDS = 600
_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def start_job(job_id: str) -> None:
    with _lock:
        _sweep_locked()
        _jobs[job_id] = {
            "status": "processing",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "terminal_at": None,
        }


def complete_job(job_id: str, result: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(status="completed", result=result, terminal_at=time.time())


def fail_job(job_id: str, error: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(status="failed", error=error, terminal_at=time.time())


def take_job(job_id: str) -> dict[str, Any] | None:
    """Return the job's current state.

    Deliberately non-destructive: callers may poll the same job repeatedly
    (e.g. two near-simultaneous requests from a client) and must consistently
    see the same terminal result rather than racing over who "claims" it
    first. Terminal jobs are instead cleaned up by the TTL sweep below.
    """
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _sweep_locked() -> None:
    now = time.time()
    stale = [
        job_id
        for job_id, job in _jobs.items()
        if (job["terminal_at"] is not None and job["terminal_at"] < now - _TERMINAL_TTL_SECONDS)
        or (job["terminal_at"] is None and job["created_at"] < now - _PROCESSING_TTL_SECONDS)
    ]
    for job_id in stale:
        del _jobs[job_id]
