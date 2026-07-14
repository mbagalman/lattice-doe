# api_server/routers/sensitivity.py
# License: MIT
"""POST /sensitivity and POST /mde."""
from __future__ import annotations

import anyio
from functools import partial

from fastapi import APIRouter

from lattice_doe import power_sensitivity, min_detectable_effect
from lattice_doe.config import PowerContrastConfig
from api_server.models.sensitivity import (
    MdeRequest,
    MdeResponse,
    SensitivityRequest,
    SensitivityResponse,
)
from api_server.serialization import (
    factors_to_spec,
    df_to_records,
    pydantic_design_opts_to_dataclass,
    pydantic_power_cfg_to_dataclass,
    records_to_df,
    sanitize_float,
)

router = APIRouter()


def _sync_sensitivity(request: SensitivityRequest) -> dict:
    power_cfg = pydantic_power_cfg_to_dataclass(request.power_cfg)
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)
    design_df = records_to_df(request.design_df)
    result = power_sensitivity(
        formula=request.formula,
        factors=factors_to_spec(request.factors, request.formula),
        power_cfg=power_cfg,
        design_df=design_df,
        sigma_range=request.sigma_range,
        sigma_points=request.sigma_points,
        r2_range=request.r2_range,
        r2_points=request.r2_points,
        design_opts=design_opts,
    )
    mode = "contrast" if isinstance(power_cfg, PowerContrastConfig) else "r2"
    df = result["data"]
    return {
        "mode": mode,
        "nominal_power": sanitize_float(result["nominal_power"]),
        "rows": df_to_records(df),
        "columns": list(df.columns),
    }


def _sync_mde(request: MdeRequest) -> dict:
    power_cfg = pydantic_power_cfg_to_dataclass(request.power_cfg)
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)
    design_df = records_to_df(request.design_df)
    result = min_detectable_effect(
        design_df=design_df,
        formula=request.formula,
        factors=factors_to_spec(request.factors, request.formula),
        power_cfg=power_cfg,
        target_power=request.target_power,
        design_opts=design_opts,
    )
    return {
        "mde": float(result["mde"]),
        "achieved_power": float(result["achieved_power"]),
        "n": int(result["n"]),
        "mode": result["mode"],
    }


@router.post(
    "/sensitivity",
    response_model=SensitivityResponse,
    summary="Power sensitivity sweep on a fixed design",
    description=(
        "Re-evaluates power across a sweep of the key sensitivity axis "
        "using a **fixed** design matrix. No new designs are built — this "
        "is a fast analytical sweep.\n\n"
        "* **Contrast mode**: sweeps σ (``sigma_range``).\n"
        "* **R² mode**: sweeps R² target (``r2_range``)."
    ),
)
async def sensitivity_endpoint(request: SensitivityRequest) -> SensitivityResponse:
    result = await anyio.to_thread.run_sync(partial(_sync_sensitivity, request))
    return SensitivityResponse(**result)


@router.post(
    "/mde",
    response_model=MdeResponse,
    summary="Minimum detectable effect for a fixed design",
    description=(
        "Inverts the power curve to find the smallest effect size that "
        "achieves ``target_power`` for a given fixed design. No new designs "
        "are built — bisection over effect size on the existing model matrix."
    ),
)
async def mde_endpoint(request: MdeRequest) -> MdeResponse:
    result = await anyio.to_thread.run_sync(partial(_sync_mde, request))
    return MdeResponse(**result)
