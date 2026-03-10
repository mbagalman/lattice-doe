# api_server/models/design.py
# License: MIT
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from api_server.models.common import DesignOptionsModel, FactorSpec, PowerCfgModel


class DesignRequest(BaseModel):
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
    diagnostics: Optional[DiagnosticsModel] = None

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
