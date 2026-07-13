# api_server/models/design.py
# License: MIT
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .common import StrictRequestModel

from api_server.models.common import (
    DesignOptionsModel,
    FactorSpec,
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


class MultiResponseDesignResponse(BaseModel):
    """Response body for POST /multiresponse_design."""

    # Machine-readable search outcome (UX-7)
    status: Optional[str] = None
    target_met: Optional[bool] = None
    termination_reason: Optional[str] = None

    design: List[Dict[str, Any]] = Field(
        ..., description="Design matrix rows as records."
    )
    n: int = Field(..., description="Number of runs in the optimal design.")
    achieved_power: float = Field(
        ..., description="Combined power achieved by the aggregation rule."
    )
    responses: List[Dict[str, Any]] = Field(
        ..., description="Per-response power details (name, power, n, ...)."
    )
    combination_rule: str = Field(
        ..., description="Aggregation rule used (min / product / weighted_mean)."
    )
    compound_criterion: bool = Field(
        ..., description="True when per-response formulas differ (compound path)."
    )
    elapsed_sec: Optional[float] = Field(None, description="Wall-clock search time.")
    buckets: List[Dict[str, Any]] = Field(
        ..., description="Unique run-frequency buckets."
    )
    warnings: List[str] = Field(default_factory=list)
    # Search diagnostics
    search_strategy: Optional[str] = Field(None, description="Search strategy string (bisection/growth/verification).")
    p: Optional[int] = Field(None, description="Number of model parameters.")
    iteration: Optional[int] = Field(None, description="Number of bisection iterations used.")
    # Hotelling T² joint power (present when sigma_joint was supplied)
    joint_power: Optional[float] = Field(None, description="Hotelling T² joint power.")
    joint_lam: Optional[float] = Field(None, description="Hotelling T² noncentrality λ.")
    joint_df1: Optional[int] = Field(None, description="Hotelling T² numerator df.")
    joint_df2: Optional[int] = Field(None, description="Hotelling T² denominator df.")
    # Split-plot summary (present when split-plot mode was used)
    n_whole_plots: Optional[int] = Field(None, description="Number of whole plots (split-plot only).")
    subplots_per_wp: Optional[int] = Field(None, description="Sub-plots per whole plot (split-plot only).")
