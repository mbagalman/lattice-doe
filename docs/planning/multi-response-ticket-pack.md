# Enhancement 23 — Multi-Response Designs
## Ticket Pack

**Status:** Complete ✅ — MR-1 ✅ MR-2 ✅ MR-3 ✅ MR-4 ✅ MR-5 ✅ MR-6 ✅ MR-7 ✅ MR-8 ✅ MR-9 ✅ MR-10 ✅
**Backlog entry:** ENHANCEMENTS.md § Enhancement 23
**Estimated total LOE:** 15–20 developer-days
**Value:** High (multi-response is standard in industrial DOE; required for simultaneous optimisation)

---

## Background and Motivation

Real experiments almost always measure **more than one response variable**. A polymer chemist
monitors tensile strength, elongation-at-break, and viscosity simultaneously. A pharmaceutical
process engineer tracks yield, purity, and dissolution rate. A semiconductor process targets
both etch rate and uniformity.

The naive approach — run one-at-a-time power analyses for each response and take the maximum `n`
— is wasteful and ignores a critical constraint: **all responses share the same run order and
factor settings**. A design that is individually adequate for each response may nonetheless fail
to achieve target power for some responses when those requirements are evaluated simultaneously.

Multi-response optimal design addresses this by:

1. **Shared formula path** — when all responses use the same model (same X), the single design
   matrix is built once and power is evaluated per-response using each response's own L, δ, and σ.
   The design search criterion is unchanged (I/D/A-optimal over X); only the power evaluation
   aggregation rule changes.

2. **Compound criterion path** — when responses use different model formulas (different X_k),
   a single run matrix must simultaneously serve all k designs. The design search uses a
   **compound optimality criterion**:

   ```
   C(d) = Σ_k  w_k · criterion_k(X_k(d))
   ```

   where X_k(d) is the model matrix formed from run d under formula k.

3. **Hotelling T² joint power** (Phase 2) — when response correlation Σ is known, the
   multivariate noncentrality parameter gives a tighter, statistically rigorous joint power
   estimate that accounts for the gain from correlated responses. This is a separate ticket
   (MR-6) because it requires additional user input (inter-response covariance).

### Power Combination Rules

Three philosophically distinct approaches for collapsing per-response powers into a single
scalar for the n-search binary bisection:

| Rule | Formula | When to use |
|------|---------|-------------|
| `"min"` (default) | `min(P_1, …, P_k)` | Conservative; guarantees weakest response meets target |
| `"product"` | `P_1 × … × P_k` | Independence assumption; overall probability all pass |
| `"weighted_mean"` | `Σ w_k P_k / Σ w_k` | Soft trade-off; tolerates under-powering lower-priority responses |

The `"min"` rule is the safe default: the design is judged adequate only when the least-powered
response exceeds the target, matching the standard single-response guarantee.

### Mathematical Sketch

**Shared formula.** Given n runs with model matrix X (n × p):

```
λ_k = δ_k' [L_k (X'X)⁻¹ L_k']⁺ δ_k / σ_k²      (per-response noncentrality)
df1_k = rank(L_k),   df2 = n − p                   (per-response df)
P_k = 1 − F(F_crit; df1_k, df2, λ_k)               (per-response power)
P_combined = combine(P_1, …, P_k)                   (scalar for bisection)
```

**Different formulas.** The compound I-criterion over the shared run set d:

```
C_I(d) = Σ_k  w_k · (1/n) trace[(X_k' X_k)⁻¹ M_k]
```

where M_k is the moments matrix of formula k over the candidate space. Power is then
evaluated per-response after the compound-optimal design is found.

**Hotelling T² (MR-6).** When Y = XB + E with E ~ MN(0, Σ ⊗ I_n):

```
Ω = (CBΓ − Δ)' [C(X'X)⁻¹C']⁻¹ (CBΓ − Δ) Σ⁻¹     (noncentrality matrix)
λ_H = trace(Ω)                                      (scalar noncentrality, Pillai trace)
Power = 1 − F(F_crit; df1, df2, λ_H)               (approximate F test)
```

---

## Key Architectural Decisions

### Decision 1 — New top-level function vs extending the existing one

**Option A:** Add `responses: Optional[List[ResponseSpec]] = None` to the existing
`i_optimal_powered_design()`. When `responses` is not None, switch into multi-response mode.

**Option B:** New function `i_optimal_multiresponse_design()` alongside the existing function.

**Decision: Option B.** The signature of the existing function is already complex; mixing
single- and multi-response modes via a nullable argument would make the call site confusing.
A dedicated function with a `MultiResponseOptions` argument is cleaner and more discoverable.
The existing function is untouched (no regression risk).

### Decision 2 — ResponseSpec: contrast vs R² mode

Single responses support two power modes (`PowerContrastConfig` / `PowerR2Config`). Each
`ResponseSpec` will embed its own power config object (discriminated union), matching the
existing pattern. This avoids duplicating alpha, power, sigma fields.

```python
PowerCfg = Union[PowerContrastConfig, PowerR2Config]

@dataclass
class ResponseSpec:
    name: str
    power_cfg: PowerCfg        # per-response power requirements
    formula: Optional[str] = None   # None → use global formula
    weight: float = 1.0             # for "weighted_mean" combination only
```

### Decision 3 — Formula sharing heuristic

When every `ResponseSpec.formula` is `None` or identical, the shared-formula fast path is
used (one design search). When any response has a distinct formula, the compound criterion
path is activated. A `ValueError` is raised if the user provides `responses` with differing
formulas but does not set `criterion="I"` or `criterion="D"` (A-compound is not implemented).

### Decision 4 — Binary n-search for multi-response

The existing binary bisection in `api.py` sweeps total run count `n` and calls a user-supplied
inner evaluator `_eval(n)`. For multi-response, `_eval(n)` becomes:

```
1. Build X at current n (shared formula) or build X_k at current n (compound)
2. Compute P_k for each response
3. Return combine(P_1, …, P_k)
```

The outer bisection loop is unchanged. Power monotonicity in n is preserved under `min` and
`product`; the growth-fallback already handles non-monotone cases.

### Decision 5 — Hotelling T² as a Phase 2 feature (MR-6)

Hotelling T² requires the user to supply an inter-response error covariance matrix Σ, which is
rarely known a priori. The implementation is gated behind a separate field
`MultiResponseOptions.sigma_joint`. When `None`, the per-response independence combination
(Decision 1) is used. When provided, the Hotelling T² path is activated.

Phase 2 scope: MR-6 implements Hotelling T² power for the **shared formula** path only.
Different-formula Hotelling T² (where Ω is block-structured) is deferred to a future
enhancement.

### Decision 6 — Report structure

The result dict returned by `i_optimal_multiresponse_design` mirrors the single-response
result plus a new `"responses"` key:

```python
{
    "design": pd.DataFrame,
    "n": int,
    "achieved_power": float,      # combined power (scalar)
    "responses": [                # per-response breakdown
        {"name": "Y1", "power": 0.87, "lam": 12.3, "n": 40},
        {"name": "Y2", "power": 0.81, "lam": 9.1, "n": 40},
    ],
    "combination_rule": "min",
    "compound_criterion": False,
    "elapsed_sec": 4.2,
    ...
}
```

---

## New Public API Surface

```python
# --- config.py ---

from typing import Literal, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ResponseSpec:
    """Specification of one response variable's power requirements.

    Parameters
    ----------
    name : str
        Label for this response (used in reporting and result dict keys).
    power_cfg : PowerContrastConfig | PowerR2Config
        Power requirements for this response.  Each response may use a
        different mode (contrast or R²), have its own sigma, L, delta, etc.
    formula : str or None, default None
        Patsy formula for this response.  If None, the global formula
        passed to i_optimal_multiresponse_design() is used.  Setting a
        different formula per-response activates the compound criterion path.
    weight : float, default 1.0
        Relative importance weight used when power_combination="weighted_mean".
        Weights are normalised internally (they do not need to sum to 1).
    """
    name: str
    power_cfg: Union[PowerContrastConfig, PowerR2Config]
    formula: Optional[str] = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ResponseSpec.name must be a non-empty string.")
        if self.weight <= 0:
            raise ValueError(f"ResponseSpec.weight must be > 0, got {self.weight}.")


@dataclass
class MultiResponseOptions:
    """Options for multi-response powered design.

    Parameters
    ----------
    responses : list of ResponseSpec
        One entry per response variable.  Must contain at least two entries
        (use i_optimal_powered_design for single-response problems).
    power_combination : {"min", "product", "weighted_mean"}, default "min"
        Rule for aggregating per-response powers into a single scalar used
        by the binary n-search.

        * "min"           — design adequate when *all* responses reach target
                            (conservative; recommended default).
        * "product"       — overall probability all responses pass (independence).
        * "weighted_mean" — weighted average; tolerates weaker minor responses.
    sigma_joint : ndarray (k x k) or None, default None
        Inter-response error covariance matrix for Hotelling T² joint power.
        Must be symmetric positive definite with k = len(responses).
        When None, per-response independence combination is used.
        Only valid when all responses share the same formula and use contrast mode.
    """
    responses: List[ResponseSpec]
    power_combination: Literal["min", "product", "weighted_mean"] = "min"
    sigma_joint: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if len(self.responses) < 2:
            raise ValueError(
                "MultiResponseOptions requires at least 2 ResponseSpec entries. "
                "Use i_optimal_powered_design() for single-response problems."
            )
        names = [r.name for r in self.responses]
        if len(names) != len(set(names)):
            raise ValueError("ResponseSpec names must be unique.")
        if self.power_combination not in ("min", "product", "weighted_mean"):
            raise ValueError(
                "power_combination must be 'min', 'product', or 'weighted_mean'."
            )
        if self.sigma_joint is not None:
            k = len(self.responses)
            arr = np.asarray(self.sigma_joint)
            if arr.shape != (k, k):
                raise ValueError(
                    f"sigma_joint must be ({k},{k}) for {k} responses, "
                    f"got {arr.shape}."
                )


# --- api.py / __init__.py ---

def i_optimal_multiresponse_design(
    formula: str,
    factors: Dict[str, Any],
    multi_cfg: MultiResponseOptions,
    design_opts: Optional[DesignOptions] = None,
) -> dict:
    """Find the minimum-n I-optimal design achieving target power for all responses.

    Parameters
    ----------
    formula : str
        Global Patsy formula (right-hand side, e.g. "~ 1 + A + B + A:B").
        Individual responses may override this via ResponseSpec.formula.
    factors : dict
        Factor definitions (same format as i_optimal_powered_design).
    multi_cfg : MultiResponseOptions
        Per-response power requirements and combination rule.
    design_opts : DesignOptions or None
        Design search options (criterion, starts, workers, …).
        Defaults to DesignOptions() when None.

    Returns
    -------
    dict
        Keys: "design", "n", "achieved_power", "responses", "combination_rule",
        "compound_criterion", "elapsed_sec", "buckets", "warnings", plus the
        standard single-response metadata keys ("search_strategy", etc.).
    """
    ...


# --- analysis.py / __init__.py ---

def power_curve_by_n_multiresponse(
    formula: str,
    factors: Dict[str, Any],
    multi_cfg: MultiResponseOptions,
    n_range: Tuple[int, int] = (5, 100),
    n_points: int = 20,
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
    plot_backend: str = "matplotlib",
) -> pd.DataFrame:
    """Power vs n curve for a multi-response design.

    Returns a DataFrame with columns:
        n, combined_power, <response_name>_power (one per response)
    """
    ...
```

---

## Ticket List

| ID | Title | LOE | Depends On |
|----|-------|-----|-----------|
| MR-1 | Config layer — `ResponseSpec` and `MultiResponseOptions` | 1 day | — |
| MR-2 | Per-response power evaluation wrapper | 1 day | MR-1 |
| MR-3 | Power combination rules and n-search integration | 1 day | MR-2 |
| MR-4 | Shared-formula multi-response API function | 3 days | MR-3 |
| MR-5 | Compound criterion for different formulas | 3 days | MR-3 |
| MR-6 | Hotelling T² joint power (Phase 2) | 2 days | MR-4 |
| MR-7 | Analysis functions (`power_curve_by_n_multiresponse`) | 2 days | MR-4 |
| MR-8 | CLI / Streamlit / YAML / Sheets / Excel integration | 2 days | MR-4 |
| MR-9 | REST API extension | 1 day | MR-4 |
| MR-10 | Integration and property-based tests | 3 days | MR-4, MR-5, MR-6, MR-7 |

---

## Ticket Details

---

### MR-1 — Config layer

**Goal:** Add `ResponseSpec` and `MultiResponseOptions` dataclasses to `config.py`.

**Files changed:**
- `iopt_power_design/config.py`

**What to implement:**

`ResponseSpec` dataclass:
```python
@dataclass
class ResponseSpec:
    name: str
    power_cfg: Union[PowerContrastConfig, PowerR2Config]
    formula: Optional[str] = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ResponseSpec.name must be a non-empty string.")
        if not isinstance(self.power_cfg, (PowerContrastConfig, PowerR2Config)):
            raise TypeError(
                "ResponseSpec.power_cfg must be a PowerContrastConfig or PowerR2Config."
            )
        if self.weight <= 0:
            raise ValueError(f"ResponseSpec.weight must be > 0, got {self.weight}.")
```

`MultiResponseOptions` dataclass (see New Public API Surface above for full spec).

Add `"ResponseSpec"` and `"MultiResponseOptions"` to `__all__`.

**Acceptance criteria:**
- Both dataclasses can be constructed with valid arguments.
- `ResponseSpec.formula = None` is the default (shared formula path).
- `MultiResponseOptions` raises `ValueError` on: fewer than 2 responses, duplicate names,
  invalid `power_combination`, `sigma_joint` shape mismatch.
- `dataclasses.replace` works correctly on both (needed by analysis sweep utilities).
- All existing tests still pass (no regression).

**Test class:** `TestResponseSpec` and `TestMultiResponseOptions` in `tests/test_config.py`

**Tests (minimum 15):**
- Valid construction with two contrast-mode responses.
- Valid construction with mixed contrast/R² responses.
- `power_combination` all three valid values accepted.
- `sigma_joint` shape validated (wrong size raises ValueError).
- `sigma_joint=None` is the default.
- `ResponseSpec.weight ≤ 0` raises ValueError.
- Empty `name` raises ValueError.
- Wrong `power_cfg` type raises TypeError.
- Duplicate response names raise ValueError.
- Fewer than 2 responses raise ValueError.
- `dataclasses.replace(MultiResponseOptions(...), power_combination="product")` works.
- `__all__` exports both new names.
- `ResponseSpec.formula = None` accepted.
- `ResponseSpec.formula = "~ 1 + A + B"` accepted.
- `MultiResponseOptions.responses` stored in insertion order.

---

### MR-2 — Per-response power evaluation wrapper

**Goal:** Thin wrapper function that computes power for a single `ResponseSpec` given a
fixed design matrix X (or X_k for that response's formula). Returns a dict with
`name`, `power`, `lam`, and (for contrast mode) `df1`, `df2`.

**Files changed:**
- `iopt_power_design/power.py`

**What to implement:**

```python
def eval_response_power(
    response: ResponseSpec,
    X: np.ndarray,
    p_names: List[str],
    jitter: float = 1e-8,
    split_plot_opts: Optional[SplitPlotOptions] = None,
    Z: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate power for one response given a fixed design matrix X.

    Parameters
    ----------
    response : ResponseSpec
    X : ndarray (n, p) — model matrix for this response's formula
    p_names : list of str — column names of X (from build_model_matrix)
    jitter : float — Tikhonov jitter for (X'X)⁻¹
    split_plot_opts : SplitPlotOptions or None
        When not None, uses GLS power functions (contrast_power_sp /
        global_r2_power_sp) instead of OLS versions.
    Z : ndarray (n, n_wp) or None — whole-plot indicator matrix, required
        when split_plot_opts is not None.

    Returns
    -------
    dict with keys: "name", "power", "lam", "df1" (contrast only), "df2"
    """
    cfg = response.power_cfg
    if isinstance(cfg, PowerContrastConfig):
        if split_plot_opts is not None:
            result = contrast_power_sp(X, cfg.L, cfg.delta, cfg.sigma, cfg.alpha,
                                       Z, split_plot_opts.eta, ...)
        else:
            result = contrast_power(X, cfg.L, cfg.delta, cfg.sigma, cfg.alpha, jitter=jitter)
        return {"name": response.name, "power": result.power, "lam": result.lam,
                "df1": cfg.L.shape[0], "df2": X.shape[0] - X.shape[1]}
    else:
        ...  # PowerR2Config path
```

Export `eval_response_power` from `__init__.py`.

**Acceptance criteria:**
- Contrast mode: result matches `contrast_power(X, L, delta, sigma, alpha)` exactly.
- R² mode: result matches `global_r2_power(X, r2_target, alpha)` exactly.
- Split-plot mode: delegates to `contrast_power_sp` / `global_r2_power_sp`.
- Returns a plain dict (not a NamedTuple) so keys are easy to inspect in tests.
- `name` in returned dict matches `response.name`.

**Test class:** `TestEvalResponsePower` in `tests/test_api.py` (or `tests/test_power.py`)

---

### MR-3 — Power combination rules and n-search integration

**Goal:** Implement `combine_powers(powers, weights, rule)` and integrate into the existing
n-search binary bisection pattern.

**Files changed:**
- `iopt_power_design/power.py` (new `combine_powers` function)
- `iopt_power_design/api.py` (new `_mr_eval(n)` inner function pattern)

**What to implement:**

```python
def combine_powers(
    powers: List[float],
    weights: Optional[List[float]],
    rule: Literal["min", "product", "weighted_mean"],
) -> float:
    """Combine per-response power values into a single scalar.

    Parameters
    ----------
    powers : list of floats in [0, 1]
    weights : list of floats (> 0), same length as powers.
        Ignored for "min" and "product"; required for "weighted_mean".
    rule : combination rule

    Returns
    -------
    float in [0, 1]
    """
    if rule == "min":
        return min(powers)
    elif rule == "product":
        p = 1.0
        for pv in powers:
            p *= pv
        return p
    elif rule == "weighted_mean":
        w = weights if weights is not None else [1.0] * len(powers)
        total_w = sum(w)
        return sum(pv * wv for pv, wv in zip(powers, w)) / total_w
    raise ValueError(f"Unknown combination rule: {rule!r}")
```

The `_mr_eval(n)` inner function (used by the outer bisection in `api.py`):
1. Builds X (shared formula path) or X_k per response (compound path).
2. Calls `eval_response_power(response_k, X_k, p_names_k)` for each response.
3. Calls `combine_powers([r["power"] for r in results], weights, rule)`.
4. Returns the combined scalar.

**Acceptance criteria:**
- `combine_powers([0.8, 0.9], None, "min") == 0.8`
- `combine_powers([0.8, 0.9], None, "product") == pytest.approx(0.72)`
- `combine_powers([0.8, 0.9], [2.0, 1.0], "weighted_mean") == pytest.approx(0.8333...)`
- Weights are normalized (values do not need to sum to 1).
- `combine_powers` raises ValueError for unknown rule.
- Empty powers list raises ValueError.

**Test class:** `TestCombinePowers` in `tests/test_power.py`

---

### MR-4 — Shared-formula multi-response API function

**Goal:** Implement `i_optimal_multiresponse_design()` for the shared-formula path (all
responses use the same formula and therefore the same X). This is the primary deliverable
and the prerequisite for all downstream tickets.

**Files changed:**
- `iopt_power_design/api.py` (new function `i_optimal_multiresponse_design`)
- `iopt_power_design/__init__.py` (export new function)

**What to implement:**

```python
def i_optimal_multiresponse_design(
    formula: str,
    factors: Dict[str, Any],
    multi_cfg: MultiResponseOptions,
    design_opts: Optional[DesignOptions] = None,
) -> dict:
```

Algorithm (shared formula path):

1. Validate: all `ResponseSpec.formula` are None or equal to the global `formula`.
   If any differ → raise `ValueError("Use compound criterion path: set ResponseSpec.formula")`
   (compound path is MR-5).
2. Resolve `design_opts` (default to `DesignOptions()` if None).
3. Build candidate set from `formula` and `factors` (same as single-response).
4. Extract the tightest power target: `target = min(r.power_cfg.power for r in responses)`
   as the convergence threshold for the outer bisection. Each response may have a different
   power target, but the combined scalar must exceed `target` (the minimum).
5. Define `_eval(n) -> float`:
   a. Build design at current n using `build_i_opt_design_with_idx`.
   b. Build X from formula.
   c. Call `eval_response_power(r, X, p_names)` for each response.
   d. Return `combine_powers([r["power"] for r in per_r], weights, rule)`.
6. Run the existing binary bisection loop (reuse / factor out from `api.py`).
7. Assemble and return the result dict with `"responses"` key.

**Handling mixed power targets:** When responses have different power targets (e.g. Y1
wants 0.80, Y2 wants 0.90), the bisection aims for the hardest target (0.90). The result
dict records per-response achieved power for the user to inspect. Add a `"warnings"` entry
if any response significantly exceeds its target (indicating that a smaller n may suffice
for that response alone).

**Split-plot compatibility:** If `design_opts.split_plot` is not None, delegate to the
split-plot `_sp_eval` pattern, calling `eval_response_power` with `split_plot_opts` and
`Z` passed through.

**Acceptance criteria:**
- With two identical `ResponseSpec` objects (same L, delta, sigma), returns the same n
  as `i_optimal_powered_design()` called with one of those configs.
- `result["responses"]` has length equal to `len(multi_cfg.responses)`.
- Each entry in `result["responses"]` has keys `"name"`, `"power"`, `"lam"`.
- `result["achieved_power"]` equals `combine_powers(per_response_powers, ...)`.
- `result["combination_rule"]` matches `multi_cfg.power_combination`.
- `result["compound_criterion"]` is `False` for the shared-formula path.
- When a response's per-response power at the chosen n exceeds its individual target,
  a warning is included in `result["warnings"]`.
- Raises `ValueError` if any `ResponseSpec.formula` differs from the global formula
  (compound path not yet active in this ticket — MR-5).
- Raises `ValueError` if `multi_cfg` has fewer than 2 responses.

**Test class:** `TestMultiResponseAPI` in `tests/test_api.py`

**Tests (minimum 15):**
- Two contrast-mode responses, same L/delta/sigma → n matches single-response n.
- Two contrast-mode responses, one harder → n driven by harder response.
- Three responses with `"min"` combination → weakest governs.
- Three responses with `"product"` combination → n is smaller than `"min"`.
- Two R²-mode responses → correct n.
- Mixed contrast + R² responses → runs without error, correct structure.
- `result["responses"]` length matches `len(multi_cfg.responses)`.
- `result["compound_criterion"]` is `False`.
- `result["design"]` is a DataFrame with factor columns.
- `result["buckets"]` is present.
- `result["elapsed_sec"]` > 0.
- Split-plot mode: `design_opts.split_plot` set → uses GLS power functions.
- `sigma_joint` set with mismatched shape → `ValueError` at `MultiResponseOptions` construction.
- Differing formulas raise `ValueError` (deferred to MR-5).
- `workers=2` (parallel) → same n as serial.

---

### MR-5 — Compound criterion for different formulas

**Goal:** Support the case where responses use different model formulas, requiring a compound
optimality criterion over the shared run set.

**Files changed:**
- `iopt_power_design/api.py` (extend `i_optimal_multiresponse_design` with compound path)
- `iopt_power_design/iopt_search.py` (new `build_compound_design` function)

**When activated:** Any `ResponseSpec.formula` is not None and differs from the global
formula (or from another response's formula).

**What to implement:**

Compound I-criterion scorer:

```python
def compound_i_criterion(
    indices: List[int],
    candidates_list: List[np.ndarray],  # one candidate array per formula
    weights: List[float],
    jitter: float = 1e-8,
) -> float:
    """Weighted sum of per-formula I-criterion scores over the shared run indices.

    score_k = trace[(X_k' X_k)⁻¹ M_k] where M_k is the pre-computed
    moments matrix for formula k.

    Returns: Σ_k w_k · score_k  (lower is better for I-optimal).
    """
```

Exchange loop: the Fedorov exchange operates on a **single** index set (the shared run
indices). At each candidate swap, the compound score is recomputed across all formulas.
This is the natural generalisation — the only change is the scorer function.

Candidate sets: when formulas differ, multiple candidate arrays are pre-built (one per
formula). The shared run indices index into the same physical candidate rows; each formula's
candidate array is evaluated at those rows.

**Constraint:** All response formulas must be over the **same factors** (same column names).
Raise `ValueError` if a response formula introduces a factor not in the global `factors` dict.

**Acceptance criteria:**
- Two responses with identical formula → compound criterion produces the same design as
  the standard single-formula path (MR-4).
- Two responses with different formulas (e.g. linear vs quadratic) → compound design is
  valid for both (X_k is estimable for both formulas at the chosen n).
- Compound criterion path sets `result["compound_criterion"]` to `True`.
- A-optimal compound criterion is explicitly rejected (`criterion="A"` with different
  formulas raises `NotImplementedError("A-compound not supported; use 'I' or 'D'.")`).
- Adding a formula for a factor not in `factors` raises `ValueError`.

**Test class:** `TestCompoundCriterion` in `tests/test_api.py`

---

### MR-6 — Hotelling T² joint power

**Goal:** When `MultiResponseOptions.sigma_joint` is provided, compute joint power via the
Hotelling T² (multivariate F) test instead of the per-response independence combination.
Applies only to the shared-formula contrast path.

**Files changed:**
- `iopt_power_design/power.py` (new `hotelling_t2_power` function)
- `iopt_power_design/api.py` (delegate to Hotelling T² when `sigma_joint` is set)
- `iopt_power_design/__init__.py` (export `hotelling_t2_power`)

**Mathematical derivation:**

Given k responses, shared X (n × p), and joint contrast:

```
C (q × p) — common contrast matrix (same L for all responses)
B (p × k) — coefficient matrix
Γ (k × k) — response combination matrix (usually I_k)
Σ (k × k) — inter-response error covariance (provided as sigma_joint)
Δ (q × k) — effect-size matrix (each column = delta_r for response r)

Noncentrality matrix:
    Ω = (CBΓ − Δ)' [C(X'X)⁻¹C']⁻¹ (CBΓ − Δ) Σ⁻¹

Pillai trace statistic (approximate F):
    V  = trace(Ω (I + Ω)⁻¹)   ≈ trace(Ω) for small Ω
    df1 = q · k,   df2 = n − p − k + 1   (Bartlett-Nanda-Pillai)
    λ  = V · df2 / (df1 · (1 − V/k))     (F-approximation noncentrality)
Power = 1 − F(F_crit; df1, df2, λ)
```

For the simplified case where Γ = I_k and all responses have the same L and delta:

```
Ω ≈ n · δ δ' / σ_scalar²  for scalar σ (reduces to standard contrast power)
```

This serves as a numerical sanity check (two-response Hotelling T² with Σ = I should
give power ≥ each individual response's power when responses are independent, and higher
power when they are positively correlated).

**Acceptance criteria:**
- With `sigma_joint = I_k` (identity), Hotelling T² power ≥ per-response independence
  combination under `"min"` rule for positively correlated effects.
- With `sigma_joint = I_1` and one response, matches standard `contrast_power` output.
- `hotelling_t2_power` raises `ValueError` if sigma_joint is singular (within tolerance).
- `sigma_joint` non-symmetric raises `ValueError`.
- `result["responses"]` still lists individual per-response powers alongside joint power.
- `result["joint_power"]` key present when Hotelling T² is used.
- Raises `NotImplementedError` if `sigma_joint` is set with R²-mode responses or compound-
  formula path (only contrast + shared formula is supported in Phase 2).

**Test class:** `TestHotellingT2Power` in `tests/test_power.py`

---

### MR-7 — Analysis functions

**Goal:** Multi-response equivalents of `power_curve_by_n` and `power_sensitivity`.

**Files changed:**
- `iopt_power_design/analysis.py` (two new functions)
- `iopt_power_design/__init__.py` (export both)

**What to implement:**

```python
def power_curve_by_n_multiresponse(
    formula: str,
    factors: Dict[str, Any],
    multi_cfg: MultiResponseOptions,
    n_range: Tuple[int, int] = (5, 100),
    n_points: int = 20,
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
    plot_backend: Literal["matplotlib", "plotly"] = "matplotlib",
) -> pd.DataFrame:
    """Power vs n curve for each response plus the combined power.

    Sweeps n from n_range[0] to n_range[1] at n_points evenly spaced values.
    At each n, builds an I-optimal design and evaluates per-response powers.

    Returns
    -------
    pd.DataFrame with columns:
        n, combined_power, <name>_power (one per response)
    """
```

```python
def multiresponse_sensitivity(
    formula: str,
    factors: Dict[str, Any],
    multi_cfg: MultiResponseOptions,
    fixed_n: int,
    sigma_range: Tuple[float, float] = (0.5, 3.0),
    sigma_points: int = 20,
    design_opts: Optional[DesignOptions] = None,
) -> pd.DataFrame:
    """Sigma sensitivity for a multi-response design at fixed n.

    Sweeps a common sigma scale factor across [sigma_range[0], sigma_range[1]].
    Each response's sigma is multiplied by the scale factor at each point.
    Only valid for contrast-mode responses (PowerContrastConfig).

    Returns
    -------
    pd.DataFrame with columns:
        sigma_scale, combined_power, <name>_power (one per response)
    """
```

**Acceptance criteria:**
- `power_curve_by_n_multiresponse` returns a DataFrame with `len(responses) + 2` columns
  (n, combined_power, one per response).
- `n_range` is respected; output has exactly `n_points` rows.
- Combined power column equals `combine_powers(per_response_powers_at_n, ...)` exactly.
- `plot=True` produces a figure without error (matplotlib and plotly).
- `multiresponse_sensitivity` raises `TypeError` if any response uses `PowerR2Config`.
- `multiresponse_sensitivity` result DataFrame has monotonically decreasing combined power
  as `sigma_scale` increases (with a valid design at fixed n).

**Test class:** `TestMultiResponseAnalysis` in `tests/test_api.py`

---

### MR-8 — CLI / Streamlit / YAML / Sheets / Excel integration

**Goal:** Expose multi-response design through all existing user-facing surfaces.

**Files changed:**
- `iopt_power_design/cli.py`
- `app/pages/2_Power_Config.py` (Streamlit response config panel)
- `app/pages/3_Run_Results.py` (display per-response breakdown)
- `iopt_power_design/sheets.py` (RESPONSES sheet tab)
- `iopt_power_design/excel_template.py` (RESPONSES sheet tab)

**CLI design:**

Multi-response config is expressed via a new top-level YAML key `responses`:

```yaml
responses:
  - name: Yield
    sigma: 1.2
    contrast:
      L: [[0, 1, 0, 0]]
      delta: [2.0]
    weight: 1.0
  - name: Purity
    sigma: 0.8
    formula: "~ 1 + A + B"     # optional override
    r2_target: 0.75
    weight: 2.0
power_combination: min
```

CLI flag `--multi-response` switches the run to `i_optimal_multiresponse_design()`.
Existing single-response YAML keys (`sigma`, `contrast`, `r2_target`) are still valid
for single-response runs (no regression).

**Streamlit:**

Add a "Responses" expander in `2_Power_Config.py` with:
- "Add response" button (appends a new `ResponseSpec` to session state list).
- Per-response sub-form: name, formula override, power mode (contrast/R²), sigma, L/delta
  or r2_target, weight.
- Power combination radio (min / product / weighted_mean).
- When 2+ responses configured, the "Run" button calls `i_optimal_multiresponse_design`.

**Result display (`3_Run_Results.py`):** Add a per-response power table under the main result
showing `name | power | λ | df1 | df2` for each response.

**Sheets / Excel:** Parse and write a `RESPONSES` range / sheet (same sentinel-delimited
format as existing `[CONTRAST]` / `[FACTORS]` sections).

**Acceptance criteria:**
- `--template contrast` YAML template gains a commented `responses:` block example.
- `--multi-response` with a valid YAML config produces a result and exits 0.
- Existing single-response YAML configs still work without `--multi-response`.
- Streamlit: "Add response" button renders without error in test mode.
- Streamlit results page: per-response table visible when multi-response result is present.
- `sheets.py` and `excel_template.py` do not break existing single-response round-trips.

---

### MR-9 — REST API extension

**Goal:** Expose `i_optimal_multiresponse_design` as a new HTTP endpoint.

**Files changed:**
- `api_server/models/common.py` (new Pydantic models)
- `api_server/models/power.py` (new discriminated union)
- `api_server/routers/design.py` (new `POST /multiresponse_design` endpoint)
- `api_server/serialization.py` (serialise multi-response result)
- `tests/test_api_server.py` (new endpoint tests)

**New endpoint:**

```
POST /multiresponse_design
```

Request body: `MultiResponseDesignRequest`

```python
class ResponseSpecModel(BaseModel):
    name: str
    power_cfg: PowerCfgModel          # existing discriminated union ("r2" / "contrast")
    formula: Optional[str] = None
    weight: float = 1.0

class MultiResponseOptionsModel(BaseModel):
    responses: List[ResponseSpecModel]
    power_combination: Literal["min", "product", "weighted_mean"] = "min"
    sigma_joint: Optional[List[List[float]]] = None  # JSON-serialisable k×k matrix

class MultiResponseDesignRequest(BaseModel):
    formula: str
    factors: Dict[str, Any]
    multi_cfg: MultiResponseOptionsModel
    design_opts: Optional[DesignOptionsModel] = None

class MultiResponseDesignResponse(BaseModel):
    design: List[Dict[str, Any]]
    n: int
    achieved_power: float
    responses: List[Dict[str, Any]]
    combination_rule: str
    compound_criterion: bool
    elapsed_sec: float
    buckets: List[Dict[str, Any]]
    warnings: List[str]
```

**Acceptance criteria:**
- `POST /multiresponse_design` with valid body returns 200 and correct structure.
- `responses` list in response body has length matching request.
- `sigma_joint` round-trips correctly (list-of-lists in/out).
- Invalid `power_combination` returns 422 with clear error message.
- `GET /health` still returns 200 (no regression).

**Test class:** `TestMultiResponseEndpoint` in `tests/test_api_server.py`

---

### MR-10 — Integration and property-based tests

**Goal:** End-to-end integration tests covering all code paths introduced in MR-1 through
MR-9, plus property-based parametrized tests for correctness guarantees.

**Files changed:**
- `tests/test_api.py` (new test classes)
- `tests/test_config.py` (new test classes)
- `tests/test_power.py` (new test classes)

**Integration scenarios (slow-marked, real DOE builds):**

1. **Two contrast responses, same formula, `"min"` combination** — n must be ≥ n required
   for each response individually.
2. **Two R²-mode responses, `"product"` combination** — combined power ≈ P1 × P2.
3. **Three responses with weights, `"weighted_mean"` combination** — achieved combined power
   ≥ weighted target.
4. **Compound criterion: linear vs quadratic formula** — design is estimable for both.
5. **Hotelling T² with identity sigma_joint** — joint power ≥ independence min-combination.
6. **Split-plot + multi-response** — `SplitPlotOptions` + `MultiResponseOptions` together.
7. **`power_curve_by_n_multiresponse` monotonicity** — combined power increases with n.

**Property-based tests (parametrized, fast):**

- With identical responses, `i_optimal_multiresponse_design(n=...)` power equals
  `i_optimal_powered_design(n=...)` power.
- `combine_powers` satisfies: `min ≤ weighted_mean ≤ max(powers)` for any input.
- Adding a third weaker response under `"min"` never decreases the required n vs two-response.
- `power_curve_by_n_multiresponse` DataFrame has no NaN values.
- `hotelling_t2_power` with diagonal sigma_joint with σ² → ∞ → joint power → 0.

**Target coverage:** Multi-response code paths ≥ 85% line coverage.

**Acceptance criteria:**
- All 7 integration scenarios pass (marked `@pytest.mark.slow`).
- All property-based tests pass without `pytest.mark.skip`.
- No regression in existing 821 passing tests.
- Total test count increases by ≥ 50.

---

## Implementation Order and Session Chunking

The dependency graph is linear for the happy path:

```
MR-1 → MR-2 → MR-3 → MR-4 → MR-5 (independent of MR-4 result, but needs MR-3)
                              MR-4 → MR-6 (Hotelling T²)
                              MR-4 → MR-7 (analysis curves)
                              MR-4 → MR-8 (CLI/UI)
                              MR-4 → MR-9 (REST API)
                              MR-4, MR-5, MR-6, MR-7 → MR-10 (integration tests)
```

**Recommended session breakdown:**

| Session | Tickets | Focus |
|---------|---------|-------|
| 1 | MR-1, MR-2, MR-3 | Config layer, power wrapper, combination rules (all fast) |
| 2 | MR-4 | Shared-formula API function (core deliverable) |
| 3 | MR-5 | Compound criterion (independent of MR-4 completion) |
| 4 | MR-6, MR-7 | Hotelling T² + analysis curves |
| 5 | MR-8, MR-9 | Integration surfaces (CLI, Streamlit, REST) |
| 6 | MR-10 | Full integration and property-based test suite |

Sessions 1–2 deliver a working, tested multi-response API and should be the first milestone.
Sessions 3–6 can be reordered or skipped if the compound criterion / Hotelling T² is
deprioritised.

---

## Open Questions and Risks

### Q1: Mixed power targets in the bisection
When responses have different target powers (e.g. 0.80 vs 0.90), the combined scalar must
reach the harder target. Under `"min"` combination: the harder target governs. Under
`"product"` or `"weighted_mean"`: the combined threshold is ambiguous. **Proposed resolution:**
always use `max(power_cfg.power for r in responses)` as the bisection convergence threshold,
and report per-response achieved powers in the result. Users can inspect whether lower-priority
responses exceed their individual targets.

### Q2: Compound criterion scalability
For k > 5 responses with different formulas, the compound criterion evaluation multiplies
per-iteration cost by k. This may make the search impractically slow. **Proposed mitigation:**
batch-compute all X_k simultaneously using vectorised NumPy indexing; profile before deciding
on further optimisation.

### Q3: Hotelling T² validity for R²-mode responses
The Hotelling T² derivation assumes contrast structure (L, delta, Σ). There is no clean
multivariate analogue of the global R² F-test. **Proposed resolution:** raise
`NotImplementedError` for R²-mode responses with `sigma_joint` (documented in MR-6
acceptance criteria). Future enhancement could integrate over the prior on B.

### Q4: `sigma_joint` estimation in practice
Users rarely know the true inter-response covariance. The ticket pack should note in the
CLI YAML template that `sigma_joint` can be estimated from a pilot study using
`np.cov(responses_pilot_data, rowvar=False)`. Adding a `fit_sigma_joint(data)` utility
function would be a follow-on enhancement (not in scope here).

### Q5: Streamlit complexity
Adding per-response sub-forms significantly increases Streamlit page complexity. Consider
shipping MR-8 as a phased delivery: CLI + YAML first (end of session 5), Streamlit UI
as a separate sub-task in session 6 if time permits.

---

## Reference Material

- **Existing power functions:** `iopt_power_design/power.py` — `contrast_power`,
  `global_r2_power`, `contrast_power_sp`, `global_r2_power_sp`
- **Existing config:** `iopt_power_design/config.py` — `PowerContrastConfig`,
  `PowerR2Config`, `DesignOptions`, `SplitPlotOptions`
- **Existing API:** `iopt_power_design/api.py` — `i_optimal_powered_design`,
  `_sp_eval`, binary bisection loop structure
- **Existing REST models:** `api_server/models/common.py`, `api_server/models/power.py` —
  `PowerCfgModel` discriminated union pattern
- **Prior ticket pack:** [`docs/planning/split-plot-ticket-pack.md`](split-plot-ticket-pack.md)
- **ENHANCEMENTS.md backlog entry:** Enhancement 23 row in High effort · High value table

### Key formulas summary

```
# Per-response noncentrality (OLS, contrast mode)
λ_k = δ_k' [L_k (X'X)⁻¹ L_k']⁺ δ_k / σ_k²

# Per-response noncentrality (GLS, split-plot contrast mode)
λ_k = δ_k' [L_k (X'V⁻¹X)⁻¹ L_k']⁺ δ_k / σ²_sp,k

# Hotelling T² noncentrality (shared formula, contrast mode)
Ω = (CBΓ − Δ)' [C(X'X)⁻¹C']⁻¹ (CBΓ − Δ) Σ⁻¹
λ_H ≈ trace(Ω) / df_denom  [Pillai trace F-approximation]

# Compound I-criterion
C_I(d) = Σ_k  w_k · trace[(X_k'X_k)⁻¹ M_k]

# Power combination rules
P_min = min(P_1, …, P_k)
P_prod = Π P_k
P_wmean = Σ w_k P_k / Σ w_k
```
