# api_server/models/augment.py
# License: MIT
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from api_server.models.common import DesignOptionsModel, FactorSpec


class AugmentRequest(BaseModel):
    """Request body for POST /augment."""

    design_df: List[Dict[str, Any]] = Field(
        ...,
        description="Existing design matrix as rows-of-records.",
    )
    m: int = Field(..., ge=1, description="Number of new runs to add.")
    formula: str
    factors: Dict[str, FactorSpec]
    design_opts: Optional[DesignOptionsModel] = None

    model_config = {"json_schema_extra": {
        "examples": [{
            "design_df": [{"A": -1.0, "B": -1.0}, {"A": 1.0, "B": 1.0}],
            "m": 2,
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
        }]
    }}


class AugmentResponse(BaseModel):
    """Response body for POST /augment."""

    augmented_df: List[Dict[str, Any]] = Field(
        ..., description="Full augmented design (original + new runs)."
    )
    new_runs_df: List[Dict[str, Any]] = Field(
        ..., description="Only the newly added runs."
    )
    n_original: int
    n_added: int
    n_total: int
