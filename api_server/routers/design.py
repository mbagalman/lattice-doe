# api_server/routers/design.py
# License: MIT
"""POST /design and POST /multiresponse_design endpoints."""
from __future__ import annotations

import anyio
from functools import partial

from fastapi import APIRouter

from iopt_power_design import i_optimal_powered_design
from iopt_power_design.api import i_optimal_multiresponse_design
from api_server.models.design import (
    DesignRequest,
    DesignResponse,
    MultiResponseDesignRequest,
    MultiResponseDesignResponse,
)
from api_server.serialization import (
    pydantic_design_opts_to_dataclass,
    pydantic_multi_cfg_to_dataclass,
    pydantic_power_cfg_to_dataclass,
    serialize_design_result,
    serialize_multiresponse_result,
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


def _sync_run_multiresponse_design(request: MultiResponseDesignRequest) -> dict:
    """Synchronous wrapper called from the thread-pool executor."""
    multi_cfg = pydantic_multi_cfg_to_dataclass(request.multi_cfg)
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)
    result = i_optimal_multiresponse_design(
        formula=request.formula,
        factors=dict(request.factors),
        multi_cfg=multi_cfg,
        design_opts=design_opts,
    )
    return serialize_multiresponse_result(result)


@router.post(
    "/multiresponse_design",
    response_model=MultiResponseDesignResponse,
    summary="Generate an I-optimal powered design for multiple responses",
    description=(
        "Builds a sample-size-assured I-optimal experimental design that "
        "satisfies power requirements for all specified responses simultaneously. "
        "Supports 'min', 'product', and 'weighted_mean' combination rules, "
        "compound-criterion paths (per-response formulas), and Hotelling T² "
        "joint power via sigma_joint.\n\n"
        "This endpoint runs synchronously in a background thread so the "
        "server remains responsive during long searches."
    ),
)
async def run_multiresponse_design(
    request: MultiResponseDesignRequest,
) -> MultiResponseDesignResponse:
    result_dict = await anyio.to_thread.run_sync(
        partial(_sync_run_multiresponse_design, request)
    )
    return MultiResponseDesignResponse(**result_dict)
