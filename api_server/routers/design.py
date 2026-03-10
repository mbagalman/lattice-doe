# api_server/routers/design.py
# License: MIT
"""POST /design — main powered-design generation endpoint."""
from __future__ import annotations

import anyio
from functools import partial

from fastapi import APIRouter

from iopt_power_design import i_optimal_powered_design
from api_server.models.design import DesignRequest, DesignResponse
from api_server.serialization import (
    pydantic_power_cfg_to_dataclass,
    pydantic_design_opts_to_dataclass,
    serialize_design_result,
)

router = APIRouter()


def _sync_run_design(request: DesignRequest) -> dict:
    """Synchronous wrapper called from the thread-pool executor."""
    power_cfg = pydantic_power_cfg_to_dataclass(request.power_cfg)
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)
    result = i_optimal_powered_design(
        formula=request.formula,
        factors=dict(request.factors),
        power_cfg=power_cfg,
        design_opts=design_opts,
    )
    return serialize_design_result(result)


@router.post(
    "/design",
    response_model=DesignResponse,
    summary="Generate an I-optimal powered design",
    description=(
        "Builds a sample-size-assured I-optimal experimental design. "
        "Iteratively finds the minimum n that achieves the target power, "
        "then returns the design matrix, run-frequency buckets, and a "
        "full diagnostics report.\n\n"
        "This endpoint runs synchronously in a background thread so the "
        "server remains responsive during long searches (30–120 s is normal "
        "for complex factor spaces with many starts)."
    ),
)
async def run_design(request: DesignRequest) -> DesignResponse:
    result_dict = await anyio.to_thread.run_sync(partial(_sync_run_design, request))
    return DesignResponse(**result_dict)
