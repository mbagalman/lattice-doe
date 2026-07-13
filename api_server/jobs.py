# api_server/jobs.py
# License: MIT
"""In-memory async job manager for long-running design searches (UX-2).

A design search can take 30–120 s. Holding an HTTP request open for that long
gives the caller no way to observe progress, cancel, or retry. This module runs
each search on a background thread and exposes its live state through a job
resource:

* ``submit`` returns a job id immediately (the router replies ``202``).
* ``get`` returns the job's state, latest progress event, and result/error.
* ``cancel`` requests cooperative cancellation (honored at the next progress
  emit inside the search).

Concurrency is bounded: at most ``max_concurrent`` searches run at once, and
further submissions are rejected with :class:`JobsAtCapacity` (the router maps
this to ``503`` + ``Retry-After``). Finished jobs are retained up to
``max_retained`` (oldest evicted first) so memory stays bounded.

This is a single-process store (state lives in the worker's memory). Under
Uvicorn ``--workers N`` each worker keeps its own jobs; a job id is only
resolvable on the worker that created it. For multi-worker durable jobs, back
this with a shared store (Redis/DB) — tracked as a follow-up.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from lattice_doe.progress import ProgressReporter, SearchCancelled


class JobsAtCapacity(RuntimeError):
    """Raised by ``submit`` when the concurrency limit is reached."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("Job manager at capacity; retry later.")
        self.retry_after = retry_after


# A runner receives a ready-made ProgressReporter (already wired to record the
# job's live progress and to honor cancellation) and returns the serialized
# result dict. Typically it just forwards the reporter to
# ``find_optimal_design(..., on_progress=reporter)``.
Runner = Callable[[ProgressReporter], Dict[str, Any]]


@dataclass
class Job:
    id: str
    kind: str
    state: str = "queued"  # queued | running | done | failed | cancelled
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        end = self.finished_at or now
        return {
            "job_id": self.id,
            "kind": self.kind,
            "state": self.state,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_sec": round(end - self.started_at, 3) if self.started_at else 0.0,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    """Thread-safe, bounded-concurrency job store."""

    def __init__(
        self,
        max_concurrent: int = 2,
        max_retained: int = 128,
        retry_after: int = 5,
    ) -> None:
        self._max_concurrent = max(1, int(max_concurrent))
        self._max_retained = max(1, int(max_retained))
        self._retry_after = max(1, int(retry_after))
        self._sema = threading.BoundedSemaphore(self._max_concurrent)
        self._lock = threading.Lock()
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()

    # -- introspection --
    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def retry_after(self) -> int:
        return self._retry_after

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.state == "running")

    # -- lifecycle --
    def submit(self, kind: str, runner: Runner) -> str:
        """Accept a job and start it on a worker thread, or raise
        :class:`JobsAtCapacity` if the concurrency limit is reached."""
        if not self._sema.acquire(blocking=False):
            raise JobsAtCapacity(self._retry_after)

        job = Job(id=uuid.uuid4().hex, kind=kind)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()

        threading.Thread(
            target=self._run, args=(job, runner), daemon=True,
            name=f"job-{job.id[:8]}",
        ).start()
        return job.id

    def _run(self, job: Job, runner: Runner) -> None:
        try:
            with self._lock:
                job.state = "running"
                job.started_at = time.time()

            def _sink(ev) -> None:
                with self._lock:
                    job.progress = ev.to_dict()

            reporter = ProgressReporter(
                _sink, min_interval=0.25, cancelled=job._cancel.is_set
            )
            result = runner(reporter)

            with self._lock:
                if job.state != "cancelled":
                    job.result = result
                    job.state = "done"
        except SearchCancelled:
            with self._lock:
                job.state = "cancelled"
                job.error = "Cancelled by request."
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            with self._lock:
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                job.finished_at = time.time()
            self._sema.release()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.snapshot() if job is not None else None

    def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state in ("queued", "running"):
                job._cancel.set()
            return job.snapshot()

    def _evict_locked(self) -> None:
        """Drop oldest *finished* jobs beyond the retention cap."""
        if len(self._jobs) <= self._max_retained:
            return
        removable = [
            jid for jid, j in self._jobs.items()
            if j.state in ("done", "failed", "cancelled")
        ]
        while len(self._jobs) > self._max_retained and removable:
            self._jobs.pop(removable.pop(0), None)


def build_job_manager() -> JobManager:
    """Construct a JobManager from environment configuration."""
    return JobManager(
        max_concurrent=int(os.environ.get("LATTICE_JOBS_MAX_CONCURRENT", "2")),
        max_retained=int(os.environ.get("LATTICE_JOBS_MAX_RETAINED", "128")),
        retry_after=int(os.environ.get("LATTICE_JOBS_RETRY_AFTER", "5")),
    )


__all__ = ["Job", "JobManager", "JobsAtCapacity", "build_job_manager"]
