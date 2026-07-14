# api_server/models/compare.py
# License: MIT
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .common import StrictRequestModel

from api_server.models.common import DesignOptionsModel, FactorSpec, PowerCfgModel
from api_server.models.design import DesignResponse


class CompareCriteriaRequest(StrictRequestModel):
    """Request body for POST /compare_criteria."""

    formula: str
    factors: Dict[str, FactorSpec]
    power_cfg: PowerCfgModel
    design_opts: Optional[DesignOptionsModel] = None
    criteria: Optional[List[Literal["I", "D", "A"]]] = Field(
        None,
        description="Criteria to compare. Defaults to all three: ['I', 'D', 'A'].",
    )


class CriterionSummaryRow(BaseModel):
    criterion: str
    n: int
    achieved_power: float
    elapsed_sec: Optional[float] = None
    condition_number: Optional[float] = None
    d_efficiency: Optional[float] = None


class CompareCriteriaResponse(BaseModel):
    """Response body for POST /compare_criteria."""

    summary: List[CriterionSummaryRow]
    results: Dict[str, DesignResponse] = Field(
        ...,
        description="Full design result keyed by criterion ('I', 'D', 'A').",
    )
