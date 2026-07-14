# api_server/routers/power_curve.py
# License: MIT
"""POST /power_curve/by_n and POST /power_curve/by_effect."""
from __future__ import annotations

import anyio
from functools import partial

from fastapi import APIRouter

from lattice_doe import power_curve_by_n, power_curve_by_effect
from api_server.models.power_curve import (
    PowerCurveByNRequest,
    PowerCurveByEffectRequest,
    PowerCurveResponse,
)
from api_server.serialization import (
    factors_to_spec,
    pydantic_power_cfg_to_dataclass,
    pydantic_design_opts_to_dataclass,
    df_to_records,
)

router = APIRouter()


def _sync_by_n(request: PowerCurveByNRequest) -> dict:
    power_cfg = pydantic_power_cfg_to_dataclass(request.power_cfg)
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)
    df = power_curve_by_n(
        formula=request.formula,
        factors=factors_to_spec(request.factors, request.formula),
        power_cfg=power_cfg,
        design_opts=design_opts,
        n_range=request.n_range,
        n_points=request.n_points,
    )
    return {"rows": df_to_records(df), "columns": list(df.columns)}


def _sync_by_effect(request: PowerCurveByEffectRequest) -> dict:
    power_cfg = pydantic_power_cfg_to_dataclass(request.power_cfg)
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)
    df = power_curve_by_effect(
        formula=request.formula,
        factors=factors_to_spec(request.factors, request.formula),
        n=request.n,
        power_cfg=power_cfg,
        design_opts=design_opts,
    )
    return {"rows": df_to_records(df), "columns": list(df.columns)}


@router.post(
    "/power_curve/by_n",
    response_model=PowerCurveResponse,
    summary="Power vs sample size curve",
    description=(
        "Builds one I-optimal design per n in the sweep range and evaluates "
        "power at each size. Returns a DataFrame (rows-as-records) with columns "
        "``n`` and ``power`` (plus ``achieved_power`` and other diagnostics)."
    ),
)
async def power_curve_by_n_endpoint(
    request: PowerCurveByNRequest,
) -> PowerCurveResponse:
    result = await anyio.to_thread.run_sync(partial(_sync_by_n, request))
    return PowerCurveResponse(**result)


@router.post(
    "/power_curve/by_effect",
    response_model=PowerCurveResponse,
    summary="Power vs effect size curve",
    description=(
        "Evaluates power at a fixed n across a sweep of effect sizes "
        "(δ scale for contrast mode, R² for global mode). Returns a "
        "DataFrame with columns ``effect_scale`` or ``r2_target`` and ``power``."
    ),
)
async def power_curve_by_effect_endpoint(
    request: PowerCurveByEffectRequest,
) -> PowerCurveResponse:
    result = await anyio.to_thread.run_sync(partial(_sync_by_effect, request))
    return PowerCurveResponse(**result)
