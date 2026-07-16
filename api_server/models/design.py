# api_server/models/design.py
# License: MIT
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .common import StrictRequestModel

from api_server.models.common import (
    DesignOptionsModel,
    FactorSpec,
    MatrixSplitModel,
    MultiResponseOptionsModel,
    PowerCfgModel,
)


class DesignRequest(StrictRequestModel):
    """Request body for POST /design."""

    formula: str = Field(
        ...,
        description="Patsy formula string, e.g. '~ 1 + A + B + A:B'.",
        examples=["~ 1 + A + B"],
    )
    factors: Dict[str, FactorSpec] = Field(
        ...,
        description=(
            "Factor specifications. Continuous: [low, high]. "
            "Categorical: [\"lvl1\", \"lvl2\", ...]."
        ),
        examples=[{"A": [-1.0, 1.0], "B": [-1.0, 1.0]}],
    )
    power_cfg: PowerCfgModel = Field(..., description="Power configuration.")
    design_opts: Optional[DesignOptionsModel] = Field(
        None,
        description="Design generation options. Uses defaults when omitted.",
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "power_cfg": {"type": "r2", "r2_target": 0.15},
        }]
    }}


class DiagnosticsModel(BaseModel):
    i_criterion: Optional[float] = None
    d_efficiency: Optional[float] = None
    condition_number: Optional[float] = None
    leverages: Optional[List[float]] = None


class ReportModel(BaseModel):
    n: int
    p: int
    df_num: int
    df_denom: int
    alpha: float
    target_power: float
    achieved_power: float
    noncentrality_lambda: float
    criterion: str
    elapsed_sec: Optional[float] = None
    search_strategy: Optional[str] = None
    warnings: List[str] = []
    # Machine-readable search outcome (UX-7)
    status: Optional[str] = None            # "complete" | "partial"
    target_met: Optional[bool] = None
    termination_reason: Optional[str] = None  # target_reached | max_n | max_iter | candidate_cap
    diagnostics: Optional[DiagnosticsModel] = None
    # GLM fields (present when power_cfg.type == "glm_contrast")
    test_type: Optional[str] = None       # "f" | "wald_chi2"
    family: Optional[str] = None
    link: Optional[str] = None
    baseline: Optional[float] = None
    glm_weight: Optional[float] = None
    df2: Optional[int] = None             # None for GLM (Wald χ² has no denominator df)

    model_config = {"extra": "allow"}  # forward-compatible with new report keys


class DesignResponse(BaseModel):
    """Response body for POST /design."""

    design_df: List[Dict[str, Any]] = Field(
        ..., description="Design matrix rows as records."
    )
    buckets_df: List[Dict[str, Any]] = Field(
        ..., description="Unique run frequencies."
    )
    model_matrix: Optional[MatrixSplitModel] = Field(
        default=None,
        description=(
            "The authoritative model matrix (n x p records, parameter-"
            "named columns) the power calculation used. For formulas "
            "whose coding is learned from the data, analyze against "
            "this (and pass it to the sensitivity endpoints) instead of "
            "refitting from design_df (UX-57)."
        ),
    )
    report: ReportModel


# ---------------------------------------------------------------------------
# Multi-response design endpoint models
# ---------------------------------------------------------------------------

class MultiResponseDesignRequest(StrictRequestModel):
    """Request body for POST /multiresponse_design."""

    formula: str = Field(
        ...,
        description="Global Patsy formula string, e.g. '~ 1 + A + B + A:B'.",
        examples=["~ 1 + A + B"],
    )
    factors: Dict[str, FactorSpec] = Field(
        ...,
        description=(
            "Factor specifications. Continuous: [low, high]. "
            "Categorical: [\"lvl1\", \"lvl2\", ...]."
        ),
        examples=[{"A": [-1.0, 1.0], "B": [-1.0, 1.0]}],
    )
    multi_cfg: MultiResponseOptionsModel = Field(
        ...,
        description="Per-response power specifications and combination rule.",
    )
    design_opts: Optional[DesignOptionsModel] = Field(
        None,
        description="Design generation options. Uses defaults when omitted.",
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "multi_cfg": {
                "responses": [
                    {"name": "Y1", "power_cfg": {"type": "r2", "r2_target": 0.15}},
                    {"name": "Y2", "power_cfg": {"type": "r2", "r2_target": 0.20}},
                ],
                "power_combination": "min",
            },
        }]
    }}


class MultiResponseReportModel(BaseModel):
    """The ``report`` block of a multi-response result (UX-6).

    Shares the search-outcome and diagnostic fields with the single-response
    ``ReportModel``; adds the multi-response-specific keys. ``extra="allow"``
    keeps it forward-compatible with optional joint / split-plot fields.
    """

    n: int
    p: Optional[int] = None
    achieved_power: float
    target_power: Optional[float] = None
    combination_rule: str
    compound_criterion: bool
    responses: List[Dict[str, Any]] = Field(default_factory=list)
    criterion: Optional[str] = None
    elapsed_sec: Optional[float] = None
    search_strategy: Optional[str] = None
    iteration: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)
    # Machine-readable search outcome (UX-7)
    status: Optional[str] = None
    target_met: Optional[bool] = None
    termination_reason: Optional[str] = None
    # Hotelling T² (present when sigma_joint was supplied)
    joint_power: Optional[float] = None
    joint_lam: Optional[float] = None
    joint_df1: Optional[int] = None
    joint_df2: Optional[int] = None
    # Split-plot summary (present in split-plot mode)
    n_whole_plots: Optional[int] = None
    subplots_per_wp: Optional[int] = None

    model_config = {"extra": "allow"}


class MultiResponseDesignResponse(BaseModel):
    """Response body for POST /multiresponse_design (UX-6).

    Unified envelope: same top-level shape as ``DesignResponse``. All
    multi-response metadata lives in ``report``.
    """

    design_df: List[Dict[str, Any]] = Field(
        ..., description="Design matrix rows as records."
    )
    buckets_df: List[Dict[str, Any]] = Field(
        ..., description="Unique run-frequency buckets."
    )
    model_matrix: Optional[MatrixSplitModel] = Field(
        default=None,
        description=(
            "The authoritative model matrix (n x p records, parameter-"
            "named columns) the power calculation used. For formulas "
            "whose coding is learned from the data, analyze against "
            "this (and pass it to the sensitivity endpoints) instead of "
            "refitting from design_df (UX-57)."
        ),
    )
    model_matrices: Optional[Dict[str, MatrixSplitModel]] = Field(
        default=None,
        description=(
            "Per-response authoritative model matrices, ordered as "
            "configured. In compound mode (per-response formulas) each "
            "response's power used its own matrix — analyze that "
            "response against model_matrices[name], not model_matrix "
            "(UX-63)."
        ),
    )
    report: MultiResponseReportModel
