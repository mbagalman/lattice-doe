# api_server/models/common.py
# License: MIT
"""
Shared Pydantic types used across all request/response models.

Key design decisions
--------------------
* ``PowerCfgModel`` uses a Pydantic v2 discriminated union on the ``type``
  field (``"r2"`` or ``"contrast"``).  The ``type`` field is synthetic —
  it does not exist on the underlying dataclasses; it is added here purely
  for unambiguous JSON deserialization.
* ``DesignOptionsModel`` accepts ``constraint_expr`` (a string) but not
  ``constraint_func`` (a callable).  The dataclass ``__post_init__`` compiles
  the string automatically.
* ``workers`` is accepted but silently forced to ``None`` inside the ASGI
  server; use Uvicorn's ``--workers`` flag for process-level parallelism.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Factor specification
# ---------------------------------------------------------------------------

# Continuous factor: (low, high) tuple encoded as a two-element list
FactorContinuous = Tuple[float, float]
# Categorical factor: list of level strings
FactorCategorical = List[str]
FactorSpec = Union[FactorContinuous, FactorCategorical]


# ---------------------------------------------------------------------------
# Power configuration models
# ---------------------------------------------------------------------------

class PowerContrastConfigModel(BaseModel):
    """Power configuration for contrast-based (L·β = δ) tests."""

    type: Literal["contrast"] = "contrast"
    L: List[List[float]] = Field(
        ...,
        description="Contrast matrix — 2-D list of floats (q rows × p columns).",
        examples=[[[0, 1, 0]], [[0, 0, 1]]],
    )
    delta: List[float] = Field(
        ...,
        description="Minimum detectable effect vector — one value per row of L.",
        examples=[[0.5]],
    )
    alpha: float = Field(0.05, ge=1e-6, lt=1.0, description="Significance level.")
    power: float = Field(0.80, gt=0.0, lt=1.0, description="Target power.")
    sigma: float = Field(1.0, gt=0.0, description="Residual standard deviation.")
    tol_power: float = Field(1e-3, gt=0.0)
    max_iter: int = Field(200, gt=0)
    max_n: int = Field(2000, gt=0, description="Hard cap on sample size.")


class PowerGLMContrastModel(BaseModel):
    """Power configuration for GLM (logistic/Poisson) Wald chi-square contrast tests."""

    type: Literal["glm_contrast"] = "glm_contrast"
    L: List[List[float]] = Field(
        ...,
        description="Contrast matrix — 2-D list of floats (q rows × p columns).",
        examples=[[[0, 1, 0]]],
    )
    delta: List[float] = Field(
        ...,
        description="Minimum detectable effect on the linear-predictor scale — one value per row of L.",
        examples=[[0.5]],
    )
    family: Literal["binomial", "poisson"] = Field(
        "binomial",
        description="GLM response family.",
    )
    link: Optional[Literal["logit", "log"]] = Field(
        None,
        description="Link function. None selects the canonical link (logit for binomial, log for Poisson).",
    )
    baseline: float = Field(
        ...,
        description="Baseline response value: event probability p₀ ∈ (0,1) for binomial; expected count μ₀ > 0 for Poisson.",
    )
    alpha: float = Field(0.05, ge=1e-6, lt=1.0, description="Significance level.")
    power: float = Field(0.80, gt=0.0, lt=1.0, description="Target power.")
    tol_power: float = Field(1e-3, gt=0.0)
    max_iter: int = Field(200, gt=0)
    max_n: int = Field(2000, gt=0, description="Hard cap on sample size.")

    @field_validator("baseline")
    @classmethod
    def validate_baseline(cls, v: float, info: Any) -> float:
        family = info.data.get("family", "binomial")
        if family == "binomial" and not (0.0 < v < 1.0):
            raise ValueError("baseline must be in (0, 1) for binomial family")
        if family == "poisson" and v <= 0:
            raise ValueError("baseline must be > 0 for Poisson family")
        return v


class PowerR2ConfigModel(BaseModel):
    """Power configuration for global R² (full-model F-test) tests."""

    type: Literal["r2"] = "r2"
    r2_target: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Target population R² effect size.",
        examples=[0.15],
    )
    alpha: float = Field(0.05, ge=1e-6, lt=1.0, description="Significance level.")
    power: float = Field(0.80, gt=0.0, lt=1.0, description="Target power.")
    tol_power: float = Field(1e-3, gt=0.0)
    max_iter: int = Field(200, gt=0)
    max_n: int = Field(2000, gt=0, description="Hard cap on sample size.")
    lambda_mode: Literal["n", "n_minus_p"] = Field(
        "n",
        description=(
            "Noncentrality convention: 'n' (G*Power / statsmodels) or "
            "'n_minus_p' (conservative)."
        ),
    )


PowerCfgModel = Annotated[
    Union[PowerContrastConfigModel, PowerR2ConfigModel, PowerGLMContrastModel],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Split-plot options model
# ---------------------------------------------------------------------------

class SplitPlotOptionsModel(BaseModel):
    """Options for split-plot (hard-to-change factor) designs.

    Mirrors ``iopt_power_design.config.SplitPlotOptions``.  Set this field
    on ``DesignOptionsModel.split_plot`` to run a split-plot search.
    """

    htc_factors: List[str] = Field(
        ...,
        min_length=1,
        description="Names of the hard-to-change (whole-plot) factors.",
        examples=[["Temperature"]],
    )
    n_whole_plots: int = Field(
        ...,
        ge=2,
        description="Number of whole plots (outer randomisation units, ≥ 2).",
        examples=[4],
    )
    eta: float = Field(
        1.0,
        ge=0.0,
        description="Variance ratio σ²_wp / σ²_sp. 0 = OLS (no WP random effect).",
    )
    subplots_per_wp: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Sub-plots per whole plot. None lets the API choose automatically: "
            "max(2, ceil(p / n_whole_plots) + 1)."
        ),
    )
    df_method: Literal["auto", "conservative", "sp_only"] = Field(
        "auto",
        description=(
            "Denominator-df assignment for contrast power: "
            "'auto' classifies WP vs SP contrasts; "
            "'conservative' always uses WP df; "
            "'sp_only' always uses SP df."
        ),
    )


# ---------------------------------------------------------------------------
# Design options model
# ---------------------------------------------------------------------------

class DesignOptionsModel(BaseModel):
    """Options controlling design generation.

    Notes
    -----
    ``constraint_func`` (Python callable) cannot travel over HTTP.
    Use ``constraint_expr`` (a string expression) instead — it is compiled
    server-side by ``DesignOptions.__post_init__`` using a sandboxed AST
    evaluator.

    ``workers > 1`` is not supported inside the ASGI server (process-pool
    executor conflicts with Uvicorn's event loop).  Use Uvicorn's own
    ``--workers`` flag for horizontal scaling.
    """

    candidate_points: int = Field(2000, gt=0)
    auto_candidate: bool = False
    cand_min: int = Field(1000, gt=0)
    cand_max: int = Field(10000, gt=0)
    random_state: int = 123
    criterion: Literal["I", "D", "A"] = "I"
    algo: Literal["fedorov", "coordinate"] = "fedorov"
    starts: int = Field(5, gt=0)
    max_iter: int = Field(1000, gt=0)
    xtx_jitter: float = Field(1e-8, gt=0.0)
    # Blocked design
    n_blocks: Optional[int] = Field(
        None,
        ge=2,
        description="Number of blocks (≥ 2 to enable blocking, None to disable).",
    )
    block_sizes: Optional[List[int]] = Field(
        None,
        description="Explicit block sizes; must sum to design n. Defaults to balanced.",
    )
    block_factor_name: str = Field(
        "Block",
        description="Name of the block column added to the design output.",
    )
    # Constraint expression (string only — no callable over HTTP)
    constraint_expr: Optional[str] = Field(
        None,
        description=(
            "Row-level constraint as a Python expression referencing factor column "
            "names. Whitelisted functions: abs, min, max, round, sqrt, log, exp, "
            "floor, ceil. Example: 'A + B <= 1'."
        ),
    )
    # Categorical pre-allocation
    preallocate_categorical: bool = False
    alloc_min_per_cell: int = Field(1, ge=1)
    alloc_max_per_cell: Optional[int] = Field(None, ge=1)
    # Split-plot (hard-to-change factor) designs
    split_plot: Optional[SplitPlotOptionsModel] = Field(
        None,
        description=(
            "Split-plot configuration. When set, the design search uses a "
            "two-stratum GLS model with whole-plot and sub-plot error terms. "
            "Cannot be combined with n_blocks."
        ),
    )


# ---------------------------------------------------------------------------
# Multi-response models
# ---------------------------------------------------------------------------

class ResponseSpecModel(BaseModel):
    """One response variable's power requirements for a multi-response design."""

    name: str = Field(..., min_length=1, description="Label for this response.")
    power_cfg: PowerCfgModel = Field(
        ...,
        description="Power requirements (r2 or contrast mode).",
    )
    formula: Optional[str] = Field(
        None,
        description=(
            "Patsy formula for this response. When None the global formula is used. "
            "Setting a per-response formula activates the compound-criterion path."
        ),
    )
    weight: float = Field(
        1.0,
        gt=0.0,
        description="Relative importance weight (used with power_combination='weighted_mean').",
    )


class MultiResponseOptionsModel(BaseModel):
    """Options for multi-response powered designs."""

    responses: List[ResponseSpecModel] = Field(
        ...,
        min_length=2,
        description="Per-response power specs. Must contain at least two entries.",
    )
    power_combination: Literal["min", "product", "weighted_mean"] = Field(
        "min",
        description=(
            "Rule for aggregating per-response powers: "
            "'min' (all responses must reach target), "
            "'product' (joint probability assuming independence), "
            "'weighted_mean' (weighted average)."
        ),
    )
    sigma_joint: Optional[List[List[float]]] = Field(
        None,
        description=(
            "Inter-response error covariance matrix (k×k, JSON list-of-lists). "
            "When set, Hotelling T² joint power is used. "
            "Only valid when all responses share the same formula and use contrast mode."
        ),
    )
