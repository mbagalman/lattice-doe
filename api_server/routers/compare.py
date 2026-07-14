# api_server/routers/compare.py
# License: MIT
"""POST /compare_criteria."""
from __future__ import annotations

import anyio
from functools import partial
from typing import Any, Dict

from fastapi import APIRouter

from lattice_doe import compare_criteria
from api_server.models.compare import (
    CompareCriteriaRequest,
    CompareCriteriaResponse,
    CriterionSummaryRow,
)
from api_server.models.design import DesignResponse
from api_server.serialization import (
    factors_to_spec,
    pydantic_design_opts_to_dataclass,
    pydantic_power_cfg_to_dataclass,
    sanitize_float,
    serialize_design_result,
)

router = APIRouter()


def _sync_compare(request: CompareCriteriaRequest) -> Dict[str, Any]:
    power_cfg = pydantic_power_cfg_to_dataclass(request.power_cfg)
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)
    result = compare_criteria(
        formula=request.formula,
        factors=factors_to_spec(request.factors, request.formula),
        power_cfg=power_cfg,
        design_opts=design_opts,
        criteria=request.criteria,
    )
    # result has keys: "summary" (DataFrame) and one key per criterion
    summary_df = result["summary"]
    summary_rows = []
    for _, row in summary_df.iterrows():
        summary_rows.append(CriterionSummaryRow(
            criterion=str(row["criterion"]),
            n=int(row["n"]),
            achieved_power=float(row["achieved_power"]),
            elapsed_sec=sanitize_float(row.get("elapsed_sec")),
            condition_number=sanitize_float(row.get("condition_number")),
            d_efficiency=sanitize_float(row.get("d_efficiency")),
        ))

    criteria_used = request.criteria or ["I", "D", "A"]
    results: Dict[str, DesignResponse] = {}
    for c in criteria_used:
        if c in result:
            serialized = serialize_design_result(result[c])
            results[c] = DesignResponse(**serialized)

    return {"summary": summary_rows, "results": results}


@router.post(
    "/compare_criteria",
    response_model=CompareCriteriaResponse,
    summary="Compare I, D, and A optimality criteria",
    description=(
        "Runs ``find_optimal_design`` independently for each entry in "
        "``criteria`` (default: all three — I, D, A), then assembles a "
        "side-by-side summary.\n\n"
        "⚠️ This endpoint runs up to **three full design searches** sequentially. "
        "Wall-clock time is 3× that of a single ``/design`` call. Consider "
        "using ``design_opts`` with reduced ``starts`` and ``max_n`` for "
        "exploratory comparisons."
    ),
)
async def compare_criteria_endpoint(
    request: CompareCriteriaRequest,
) -> CompareCriteriaResponse:
    result = await anyio.to_thread.run_sync(partial(_sync_compare, request))
    return CompareCriteriaResponse(**result)
