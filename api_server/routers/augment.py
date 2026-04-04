# api_server/routers/augment.py
# License: MIT
"""POST /augment."""
from __future__ import annotations

import anyio
from functools import partial

from fastapi import APIRouter

from lattice_doe import augment_design
from api_server.models.augment import AugmentRequest, AugmentResponse
from api_server.serialization import (
    df_to_records,
    pydantic_design_opts_to_dataclass,
    records_to_df,
)

router = APIRouter()


def _sync_augment(request: AugmentRequest) -> dict:
    design_opts = pydantic_design_opts_to_dataclass(request.design_opts)
    existing_df = records_to_df(request.design_df)
    augmented_df, new_runs_df = augment_design(
        design_df=existing_df,
        m=request.m,
        formula=request.formula,
        factors=dict(request.factors),
        design_opts=design_opts,
    )
    return {
        "augmented_df": df_to_records(augmented_df),
        "new_runs_df": df_to_records(new_runs_df),
        "n_original": len(existing_df),
        "n_added": len(new_runs_df),
        "n_total": len(augmented_df),
    }


@router.post(
    "/augment",
    response_model=AugmentResponse,
    summary="Augment an existing design with additional runs",
    description=(
        "Greedily adds ``m`` new runs to an existing design by iteratively "
        "selecting the candidate row that most improves the chosen optimality "
        "criterion. The existing rows are fixed — no re-optimization of the "
        "original runs is performed.\n\n"
        "Send the ``design_df`` from a prior ``/design`` response."
    ),
)
async def augment_endpoint(request: AugmentRequest) -> AugmentResponse:
    result = await anyio.to_thread.run_sync(partial(_sync_augment, request))
    return AugmentResponse(**result)
