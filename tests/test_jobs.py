# tests/test_jobs.py
"""Unit tests for the async job manager (UX-2)."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("fastapi")  # jobs module imports are stdlib, but keep parity

from lattice_doe.api_server.jobs import JobManager, JobsAtCapacity
from lattice_doe.progress import Phase


def _wait_state(mgr, jid, states, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = mgr.get(jid)
        if snap and snap["state"] in states:
            return snap
        time.sleep(0.02)
    return mgr.get(jid)


class TestJobManager:
    def test_successful_job_records_result(self):
        mgr = JobManager(max_concurrent=2)

        def runner(reporter):
            reporter.emit(Phase.OPTIMIZING, trial_n=10, current_power=0.9)
            return {"n": 10, "ok": True}

        jid = mgr.submit("design", runner)
        snap = _wait_state(mgr, jid, {"done"})
        assert snap["state"] == "done"
        assert snap["result"] == {"n": 10, "ok": True}
        assert snap["progress"]["trial_n"] == 10
        assert snap["error"] is None

    def test_runner_exception_marks_failed(self):
        mgr = JobManager()

        def runner(reporter):
            raise ValueError("boom")

        jid = mgr.submit("design", runner)
        snap = _wait_state(mgr, jid, {"failed"})
        assert snap["state"] == "failed"
        assert "boom" in snap["error"]
        assert snap["result"] is None

    def test_cancellation(self):
        mgr = JobManager()
        started = threading.Event()

        def runner(reporter):
            started.set()
            # Loop emitting so cancellation is honored at the next emit.
            while True:
                reporter.emit(Phase.OPTIMIZING, trial_n=1)
                time.sleep(0.01)

        jid = mgr.submit("design", runner)
        assert started.wait(2.0)
        mgr.cancel(jid)
        snap = _wait_state(mgr, jid, {"cancelled"})
        assert snap["state"] == "cancelled"

    def test_capacity_limit_and_release(self):
        mgr = JobManager(max_concurrent=1, retry_after=7)
        release = threading.Event()
        running = threading.Event()

        def blocking(reporter):
            running.set()
            release.wait(5.0)
            return {"done": True}

        jid1 = mgr.submit("design", blocking)
        assert running.wait(2.0)
        # Second submission at capacity -> raises with the configured retry.
        with pytest.raises(JobsAtCapacity) as ei:
            mgr.submit("design", blocking)
        assert ei.value.retry_after == 7
        # Release the first; slot frees; a new submission now succeeds.
        release.set()
        _wait_state(mgr, jid1, {"done"})
        jid2 = mgr.submit("design", lambda r: {"ok": True})
        assert _wait_state(mgr, jid2, {"done"})["state"] == "done"

    def test_get_unknown_returns_none(self):
        mgr = JobManager()
        assert mgr.get("nope") is None
        assert mgr.cancel("nope") is None

    def test_retention_evicts_oldest_finished(self):
        mgr = JobManager(max_concurrent=4, max_retained=3)
        ids = []
        for i in range(6):
            ids.append(mgr.submit("design", lambda r, i=i: {"i": i}))
        for jid in ids:
            _wait_state(mgr, jid, {"done"})
        # At most max_retained jobs remain; the earliest are evicted.
        present = [jid for jid in ids if mgr.get(jid) is not None]
        assert len(present) <= 3
        assert ids[-1] in present  # newest retained


class TestRetentionAfterCompletion:
    """RV-12 regression: eviction ran only at submit time, so with
    max_concurrent > max_retained, all concurrently-running jobs stayed
    stored after finishing (3 retained with a cap of 1) until the next
    submit. Terminal transitions now evict too."""

    def test_cap_enforced_when_jobs_finish(self):
        import threading
        import time

        m = JobManager(max_concurrent=3, max_retained=1, retry_after=1)
        release = threading.Event()

        def runner(reporter):
            release.wait(10)
            return {"ok": True}

        ids = [m.submit("design", runner) for _ in range(3)]
        release.set()
        deadline = time.time() + 10
        while time.time() < deadline:
            snaps = [m.get(j) for j in ids]
            if all(s is None or s["state"] == "done" for s in snaps):
                break
            time.sleep(0.05)
        time.sleep(0.3)  # let the final finally-block eviction land
        stored = sum(1 for j in ids if m.get(j) is not None)
        assert stored <= 1, f"{stored} finished jobs retained with max_retained=1"
