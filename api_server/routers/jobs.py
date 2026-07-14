# api_server/routers/jobs.py
# License: MIT
"""Asynchronous design-job endpoints (UX-2).

Long searches run in the background so callers can poll progress, cancel, or
stream events instead of holding one request open for 30–120 s:

* ``POST /jobs/design`` / ``POST /jobs/multiresponse_design`` → ``202`` + job id
* ``GET /jobs/{job_id}`` → state, live progress, and result/error
* ``DELETE /jobs/{job_id}`` → request cooperative cancellation
* ``GET /jobs/{job_id}/events`` → Server-Sent Events progress stream

When the concurrency limit is reached, submissions return ``503`` with a
``Retry-After`` header.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

import anyio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from lattice_doe import find_optimal_design
from lattice_doe.api import find_multiresponse_design
from lattice_doe.progress import ProgressReporter
from api_server.jobs import JobsAtCapacity
from api_server.models.design import DesignRequest, MultiResponseDesignRequest
from api_server.serialization import (
    factors_to_spec,
    pydantic_design_opts_to_dataclass,
    pydantic_multi_cfg_to_dataclass,
    pydantic_power_cfg_to_dataclass,
    serialize_design_result,
    serialize_multiresponse_result,
)

router = APIRouter()


def _design_runner(request: DesignRequest):
    power_cfg = pydantic_power_cfg_to_dataclass(request.power_cfg)
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)

    def run(reporter: ProgressReporter) -> Dict[str, Any]:
        result = find_optimal_design(
            formula=request.formula,
            factors=factors_to_spec(request.factors, request.formula),
            power_cfg=power_cfg,
            design_opts=design_opts,
            on_progress=reporter,
        )
        return serialize_design_result(result)

    return run


def _multiresponse_runner(request: MultiResponseDesignRequest):
    multi_cfg = pydantic_multi_cfg_to_dataclass(request.multi_cfg)
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)

    def run(reporter: ProgressReporter) -> Dict[str, Any]:
        result = find_multiresponse_design(
            formula=request.formula,
            factors=factors_to_spec(request.factors, request.formula),
            multi_cfg=multi_cfg,
            design_opts=design_opts,
            on_progress=reporter,
        )
        return serialize_multiresponse_result(result)

    return run


def _submit(request: Request, kind: str, runner) -> JSONResponse:
    manager = request.app.state.job_manager
    try:
        job_id = manager.submit(kind, runner)
    except JobsAtCapacity as exc:
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": str(exc.retry_after)},
            content={
                "error": "JobsAtCapacity",
                "detail": (
                    f"At most {manager.max_concurrent} searches run concurrently. "
                    f"Retry after {exc.retry_after}s."
                ),
            },
        )
    status_url = f"/jobs/{job_id}"
    return JSONResponse(
        status_code=202,
        headers={"Location": status_url},
        content={"job_id": job_id, "state": "queued", "status_url": status_url},
    )


@router.post(
    "/jobs/design",
    status_code=202,
    summary="Submit a design search as an asynchronous job",
    description=(
        "Accepts the same body as POST /design but returns immediately with a "
        "202 and a job id. Poll GET /jobs/{job_id} for progress and the result, "
        "or stream GET /jobs/{job_id}/events."
    ),
)
async def submit_design(request: DesignRequest, http: Request) -> JSONResponse:
    return _submit(http, "design", _design_runner(request))


@router.post(
    "/jobs/multiresponse_design",
    status_code=202,
    summary="Submit a multi-response design search as an asynchronous job",
)
async def submit_multiresponse(
    request: MultiResponseDesignRequest, http: Request
) -> JSONResponse:
    return _submit(http, "multiresponse_design", _multiresponse_runner(request))


@router.get(
    "/jobs/{job_id}",
    summary="Get a job's state, live progress, and result",
)
async def get_job(job_id: str, http: Request) -> JSONResponse:
    snap = http.app.state.job_manager.get(job_id)
    if snap is None:
        return JSONResponse(
            status_code=404,
            content={"error": "JobNotFound", "detail": f"No job with id {job_id!r}."},
        )
    return JSONResponse(status_code=200, content=snap)


@router.delete(
    "/jobs/{job_id}",
    summary="Request cancellation of a running job",
    description=(
        "Sets the job's cancellation flag. The search honors it at its next "
        "progress checkpoint (bounded by one design-build cycle), then "
        "transitions to state 'cancelled'."
    ),
)
async def cancel_job(job_id: str, http: Request) -> JSONResponse:
    snap = http.app.state.job_manager.cancel(job_id)
    if snap is None:
        return JSONResponse(
            status_code=404,
            content={"error": "JobNotFound", "detail": f"No job with id {job_id!r}."},
        )
    return JSONResponse(status_code=200, content=snap)


@router.get(
    "/jobs/{job_id}/events",
    summary="Stream job progress as Server-Sent Events",
    description=(
        "text/event-stream of progress snapshots, one SSE 'data:' line per "
        "update, terminating when the job reaches a terminal state."
    ),
)
async def stream_job_events(job_id: str, http: Request) -> Any:
    manager = http.app.state.job_manager
    if manager.get(job_id) is None:
        return JSONResponse(
            status_code=404,
            content={"error": "JobNotFound", "detail": f"No job with id {job_id!r}."},
        )

    # Emit a keep-alive comment at least this often so proxies and clients do
    # not time the stream out during a long optimizer build that produces no
    # new progress event (the search may spend many seconds inside one build).
    heartbeat_sec = 12.0

    async def event_gen():
        last_seq = None
        last_send = time.monotonic()
        terminal = {"done", "failed", "cancelled"}
        while True:
            if await http.is_disconnected():
                break
            snap = manager.get(job_id)
            if snap is None:
                break
            prog = snap.get("progress") or {}
            seq = prog.get("seq")
            now = time.monotonic()
            # Emit on a new progress event or a state change to terminal.
            if seq != last_seq or snap["state"] in terminal:
                last_seq = seq
                last_send = now
                yield f"data: {json.dumps(snap)}\n\n"
            elif now - last_send >= heartbeat_sec:
                last_send = now
                yield ": heartbeat\n\n"
            if snap["state"] in terminal:
                break
            await anyio.sleep(0.25)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


__all__ = ["router"]
