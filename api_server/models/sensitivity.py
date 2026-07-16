# api_server/models/sensitivity.py
# License: MIT
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .common import StrictRequestModel

from api_server.models.common import DesignOptionsModel, FactorSpec, PowerCfgModel
from api_server.models.common import MatrixSplitModel


class SensitivityRequest(StrictRequestModel):
    """Request body for POST /sensitivity."""

    formula: str
    factors: Dict[str, FactorSpec]
    power_cfg: PowerCfgModel
    design_df: List[Dict[str, Any]] = Field(
        ...,
        description="Fixed design matrix as rows-of-records (from a prior /design call).",
    )
    model_matrix: Optional[MatrixSplitModel] = Field(
        default=None,
        description=(
            "The design run's authoritative model matrix "
            "(result['model_matrix'] records). REQUIRED when the "
            "formula's coding is learned from the data (splines, "
            "derived categoricals): rebuilding from design_df would "
            "silently evaluate a different basis (UX-57)."
        ),
    )
    sigma_range: Tuple[float, float] = Field(
        (0.5, 2.0),
        description="(sigma_lo, sigma_hi) sweep range (contrast mode only).",
    )
    sigma_points: int = Field(25, ge=2)
    r2_range: Tuple[float, float] = Field(
        (0.05, 0.50),
        description="(r2_lo, r2_hi) sweep range (R² mode only).",
    )
    r2_points: int = Field(25, ge=2)
    design_opts: Optional[DesignOptionsModel] = None


class SensitivityResponse(BaseModel):
    """Response body for POST /sensitivity."""

    mode: Literal["contrast", "r2"]
    nominal_power: float
    rows: List[Dict[str, Any]]
    columns: List[str]


class MdeRequest(StrictRequestModel):
    """Request body for POST /mde."""

    design_df: List[Dict[str, Any]] = Field(
        ...,
        description="Fixed design matrix as rows-of-records.",
    )
    model_matrix: Optional[MatrixSplitModel] = Field(
        default=None,
        description=(
            "The design run's authoritative model matrix "
            "(result['model_matrix'] records). REQUIRED when the "
            "formula's coding is learned from the data (splines, "
            "derived categoricals): rebuilding from design_df would "
            "silently evaluate a different basis (UX-57)."
        ),
    )
    formula: str
    factors: Dict[str, FactorSpec]
    power_cfg: PowerCfgModel
    target_power: float = Field(0.80, gt=0.0, lt=1.0)
    design_opts: Optional[DesignOptionsModel] = None


class MdeResponse(BaseModel):
    """Response body for POST /mde."""

    mde: float
    achieved_power: float
    n: int
    mode: Literal["contrast", "r2"]
