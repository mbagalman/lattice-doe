# api_server/models/power_curve.py
# License: MIT
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .common import StrictRequestModel

from lattice_doe.api_server.models.common import DesignOptionsModel, FactorSpec, PowerCfgModel


class PowerCurveByNRequest(StrictRequestModel):
    """Request body for POST /power_curve/by_n."""

    formula: str
    factors: Dict[str, FactorSpec]
    power_cfg: PowerCfgModel
    design_opts: Optional[DesignOptionsModel] = None
    n_range: Optional[Tuple[int, int]] = Field(
        None,
        description="(n_min, n_max) sweep range. Defaults to a heuristic range.",
    )
    n_points: int = Field(20, ge=2, description="Number of n values to evaluate.")

    model_config = {"json_schema_extra": {
        "examples": [{
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "power_cfg": {"type": "r2", "r2_target": 0.15},
            "n_points": 15,
        }]
    }}


class PowerCurveByEffectRequest(StrictRequestModel):
    """Request body for POST /power_curve/by_effect."""

    formula: str
    factors: Dict[str, FactorSpec]
    n: int = Field(..., ge=1, description="Fixed sample size for the sweep.")
    power_cfg: PowerCfgModel
    design_opts: Optional[DesignOptionsModel] = None


class PowerCurveResponse(BaseModel):
    """Response body for power curve endpoints."""

    rows: List[Dict[str, Any]] = Field(
        ..., description="Power curve data as rows-of-records."
    )
    columns: List[str] = Field(..., description="Column names.")
