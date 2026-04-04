# Enhancement 22 — Split-Plot / Hard-to-Change Factors
## Ticket Pack

**Status:** Complete ✅ — SP-1 ✅ SP-2 ✅ SP-3 ✅ SP-4 ✅ SP-5 ✅ SP-6 ✅ SP-7 ✅ SP-8 ✅ SP-9 ✅ SP-10 ✅ done
**Backlog entry:** ENHANCEMENTS.md § Enhancement 22
**Estimated total LOE:** 20–30 developer-days
**Value:** Very High (split-plot is the most common industrial DOE restriction)

---

## Background and Motivation

In many industrial and scientific experiments, some factors are **hard to change (HTC)** between
runs — resetting an oven temperature, reconfiguring a manufacturing line, or changing a batch of
raw material is expensive or time-consuming. A naive fully-randomized design would require resetting
HTC factors for every run, which is impractical.

A **split-plot design** groups runs into **whole plots** (WPs). All runs within a whole plot share
the same setting of every HTC factor; only the **easy-to-change (ETC)** sub-plot factors vary
freely within a whole plot. The experiment is randomized at two levels:

- WP-level: the order of whole plots is randomized.
- SP-level: the order of sub-plots within each whole plot is randomized.

This creates a **two-stratum variance structure**:

```
y_ij = X_wp_i β + X_sp_ij β + τ_i + ε_ij

τ_i ~ N(0, σ²_wp)   (whole-plot error, shared within WP i)
ε_ij ~ N(0, σ²_sp)  (sub-plot error, independent)
η = σ²_wp / σ²_sp   (variance ratio, key nuisance parameter)
```

The full covariance matrix for the n observations is:

```
V = σ²_sp · (η Z Z' + I_n)
```

where **Z** is the `n × n_wp` whole-plot indicator matrix (Z_{ij} = 1 iff observation j belongs
to whole plot i).

Ignoring this structure and using OLS (`(X'X)⁻¹`) inflates the apparent precision of WP-factor
estimates and produces anti-conservative power calculations for those effects.

---

## Key Architectural Decisions

### Decision 1 — GLS vs OLS criterion
- **OLS criterion** (standard I/D/A): ignores V, treats all runs as independent.
- **GLS criterion** (split-plot I/D/A): uses `(X'V⁻¹X)` in the optimality objective.

**Decision: implement GLS for power calculations (mandatory for correctness) and provide GLS
criterion for design search (default when η > 0). OLS criterion remains available as a fallback
(`criterion_ignore_variance_ratio=True`) for comparison.**

### Decision 2 — n_whole_plots vs n_total
The binary n-search loop in `api.py` currently searches over total run count `n`. For split-plot:
- `n_total = n_whole_plots × subplots_per_wp` (when balanced)
- The search loop must handle both `n_whole_plots` (outer) and `subplots_per_wp` (inner) — two
  integers to sweep.

**Decision: the outer `max_n` search sweeps `n_whole_plots`. `subplots_per_wp` is fixed by the
user (or defaults to a heuristic: `ceil(p / n_whole_plots) + 1`). This matches how practitioners
think ("I can afford W whole plots; how many sub-plots per WP do I need?").**

### Decision 3 — Exchange algorithm structure
Split-plot Fedorov must propose two types of moves:
- **WP swap**: replace all HTC factor settings for an entire whole plot with a new WP candidate.
- **SP swap**: replace one sub-plot's ETC factor settings within a WP.

**Decision: implement as a two-phase exchange within each iteration:
Phase 1 sweeps WP swaps; Phase 2 sweeps SP swaps. Alternate until convergence.**

### Decision 4 — Degrees of freedom for power
WP-factor contrasts are tested against WP error (df_wp = n_wp − rank(X_wp)).
SP-factor and interaction contrasts are tested against SP error
(df_sp = n_total − n_wp − rank(X_sp|X_wp), approximately).

**Decision: provide `df_method` option (default `"auto"`):
- `"auto"`: classify each contrast row as WP or SP based on which factors it involves;
  use df_wp for WP-only contrasts and df_sp for all others.
- `"conservative"`: always use df_wp (most conservative, never anti-conservative).
- `"sp_only"`: always use df_sp (aggressive; ignores WP error for WP-factor effects).
Users who need Kenward-Roger can override via `df_override` (integer) — left for Enhancement 22+.**

---

## New Public API Surface

```python
# DesignOptions (config.py) — new field
@dataclass
class SplitPlotOptions:
    htc_factors: List[str]          # factors that are hard-to-change (whole-plot factors)
    n_whole_plots: int              # number of whole plots (outer groups)
    eta: float = 1.0                # variance ratio σ²_wp / σ²_sp  (≥ 0)
    subplots_per_wp: Optional[int] = None  # sub-plots per WP; None → auto-compute
    df_method: str = "auto"         # "auto" | "conservative" | "sp_only"
    criterion_ignore_vr: bool = False  # if True, use OLS criterion (not GLS)

@dataclass
class DesignOptions:
    ...
    split_plot: Optional[SplitPlotOptions] = None  # new field

# Top-level API — same signature, split-plot activated when split_plot is set in design_opts
result = find_optimal_design(
    formula="~ 1 + A + B + C + A:C",
    factors={"A": ("continuous", -1, 1),   # HTC
             "B": ("continuous", -1, 1),   # HTC
             "C": ("continuous", -1, 1)},  # ETC
    power_cfg=PowerContrastConfig(...),
    design_opts=DesignOptions(
        split_plot=SplitPlotOptions(
            htc_factors=["A", "B"],
            n_whole_plots=6,
            eta=2.0,
            subplots_per_wp=4,
        )
    ),
)

# New analysis function
from iopt_power_design import power_curve_by_wp

df = power_curve_by_wp(
    formula="~ 1 + A + B + C",
    factors=...,
    power_cfg=...,
    subplots_per_wp=4,
    wp_range=(3, 12),
    eta=2.0,
    htc_factors=["A", "B"],
)
```

---

## Ticket List

| ID | Title | LOE | Depends On |
|----|-------|-----|-----------|
| SP-1 | ~~Config layer — `SplitPlotOptions` and `DesignOptions.split_plot`~~ | ~~1 day~~ | ✅ Done — commit `2cf3e98`; 29 new tests in `TestSplitPlotOptions` + `TestDesignOptionsSplitPlot` |
| SP-2 | ~~Two-stratum covariance utilities (`split_plot.py`)~~ | ~~2 days~~ | ✅ Done — commit `89dbf26`; 48 new tests; `split_plot.py` at 100% coverage |
| SP-3 | Split-plot candidate generation | 2 days | SP-1 |
| SP-4 | GLS criterion scorers | 2 days | SP-2 |
| SP-5 | Split-plot Fedorov exchange algorithm | 5 days | SP-3, SP-4 |
| SP-6 | GLS power calculation | 2 days | SP-2 |
| SP-7 | Top-level API integration | 3 days | SP-5, SP-6 |
| SP-8 | Analysis functions (`power_curve_by_wp`, sensitivity) | 2 days | SP-7 |
| SP-9 | CLI / Streamlit / Sheets / Excel integration | 3 days | SP-7 |
| SP-10 | ~~Tests (per-ticket unit + integration regression)~~ | ~~4 days~~ | ✅ Done — 7 integration tests in `TestSplitPlotIntegration` (test_api.py); 19 property-based parametrized tests in `TestSP10PropertyBased` (test_split_plot.py); 814 passed, 37 skipped |

---

## Ticket Details

---

### SP-1 — Config layer

**Goal:** Add `SplitPlotOptions` dataclass and wire it into `DesignOptions`.

**Files changed:**
- `iopt_power_design/config.py`

**What to implement:**

```python
@dataclass
class SplitPlotOptions:
    """Configuration for split-plot (hard-to-change factor) designs."""

    htc_factors: List[str]
    """Factor names that are hard-to-change (whole-plot factors).
    Must be a non-empty subset of the keys in the `factors` dict passed to the API."""

    n_whole_plots: int
    """Number of whole plots (outer randomization units). Must be ≥ 2."""

    eta: float = 1.0
    """Variance ratio σ²_wp / σ²_sp.  Must be ≥ 0.  eta=0 reduces to OLS."""

    subplots_per_wp: Optional[int] = None
    """Sub-plots per whole plot.  None → auto-computed as ceil(p / n_whole_plots) + 1,
    where p = number of model parameters.  Must be ≥ 1 when provided."""

    df_method: Literal["auto", "conservative", "sp_only"] = "auto"
    """How to assign denominator df for power calculations.
    "auto" classifies each contrast by which stratum it belongs to.
    "conservative" always uses WP df (never anti-conservative).
    "sp_only" always uses SP df (may be anti-conservative for WP effects)."""

    criterion_ignore_vr: bool = False
    """If True, use the standard OLS criterion during design search and ignore
    the variance ratio.  Useful for comparison; not recommended for production use."""

    def __post_init__(self):
        if not self.htc_factors:
            raise ValueError("SplitPlotOptions.htc_factors must be non-empty.")
        if self.n_whole_plots < 2:
            raise ValueError("n_whole_plots must be ≥ 2.")
        if self.eta < 0:
            raise ValueError("eta (variance ratio) must be ≥ 0.")
        if self.subplots_per_wp is not None and self.subplots_per_wp < 1:
            raise ValueError("subplots_per_wp must be ≥ 1.")
```

Add to `DesignOptions`:
```python
split_plot: Optional[SplitPlotOptions] = None
```

Add cross-validation in `DesignOptions.__post_init__`:
- Emit a warning if `n_blocks` is also set (blocked split-plots not yet supported).

**Acceptance criteria:**
- `SplitPlotOptions` can be constructed and validated.
- Invalid inputs raise `ValueError` with clear messages.
- `dataclasses.replace(DesignOptions(..., split_plot=...), ...)` works correctly.
- All existing tests still pass (no regression from adding an optional field).

**Test class:** `TestSplitPlotConfig` in `tests/test_config.py`

---

### SP-2 — Two-stratum covariance utilities

**Goal:** New `split_plot.py` module containing all matrix-algebra helpers for the two-stratum
variance model.

**Files changed:**
- `iopt_power_design/split_plot.py` (new)
- `iopt_power_design/__init__.py` (export new public helpers)

**Key functions:**

```python
def build_whole_plot_indicator(n_total: int, n_wp: int, subplots_per_wp: int) -> np.ndarray:
    """Build the n_total × n_wp whole-plot indicator matrix Z.

    Assumes a balanced layout: runs 0..subplots_per_wp-1 belong to WP 0,
    runs subplots_per_wp..2*subplots_per_wp-1 belong to WP 1, etc.

    Returns: Z with shape (n_total, n_wp), dtype float64.
    """


def build_split_plot_covariance_inv(Z: np.ndarray, eta: float) -> np.ndarray:
    """Compute V⁻¹ where V = η Z Z' + I_n using the Woodbury identity.

    Woodbury: (I + η Z Z')⁻¹ = I - η Z (I/η + Z'Z)⁻¹ Z'
                               = I - Z (I/η + n_sp * I_nwp)⁻¹ Z'  [balanced case]

    For balanced designs (all WPs same size s):
        V⁻¹ = I - η/(1 + η*s) * Z Z'

    Returns: V_inv with shape (n_total, n_total), dtype float64.
    Note: For large n, V_inv can be dense.  The balanced closed-form avoids
    explicit matrix inversion and is preferred.
    """


def gls_information_matrix(X: np.ndarray, V_inv: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Compute M = X' V⁻¹ X + jitter * I (GLS information matrix).

    Returns: M with shape (p, p).
    """


def classify_contrasts(
    L: np.ndarray,
    htc_factor_cols: List[int],
    p: int,
) -> np.ndarray:
    """For each row of L, determine whether the contrast involves only WP columns.

    Returns: boolean array of shape (q,); True = pure WP contrast.
    """


def split_plot_df_denom(
    X: np.ndarray,
    Z: np.ndarray,
    is_wp_contrast: np.ndarray,
    df_method: str,
) -> np.ndarray:
    """Compute per-contrast denominator df.

    df_wp = n_wp - rank(X_wp)   where X_wp = unique WP rows of X
    df_sp = n_total - n_wp - (rank(X) - rank(X_wp))  (approximate)

    Returns: integer array of shape (q,) with df_denom per contrast row.
    """
```

**Acceptance criteria:**
- `build_split_plot_covariance_inv` with eta=0 returns identity matrix.
- `build_split_plot_covariance_inv` closed-form matches explicit matrix inverse for small examples.
- `gls_information_matrix` with identity V_inv matches `X'X`.
- `split_plot_df_denom` with `df_method="conservative"` always returns df_wp.
- Woodbury identity verified numerically for η ∈ {0.1, 1.0, 5.0, 100.0}.

**Test class:** `TestSplitPlotCovariance` in `tests/test_split_plot.py` (new file)

---

### SP-3 — Split-plot candidate generation

**Goal:** Extend `build_candidate` to produce a structured split-plot candidate set with WP and SP
factor columns clearly labelled, enabling the exchange algorithm to reason about nesting.

**Files changed:**
- `iopt_power_design/candidate.py`

**Approach:**

The split-plot candidate set is conceptually the **Cartesian product** of:
- `C_wp`: candidate WP factor settings (columns = htc_factors only)
- `C_sp`: candidate SP factor settings (columns = etc_factors only)

Each row of the full candidate = one (wp_setting, sp_setting) combination.

The candidate must also carry a `__wp_id__` column that groups sub-plots by WP slot.

```python
def build_split_plot_candidate(
    factors: dict,
    htc_factors: List[str],
    n_whole_plots: int,
    subplots_per_wp: int,
    *,
    random_state: Optional[int] = None,
    candidate_points: Optional[int] = None,
) -> pd.DataFrame:
    """Build split-plot candidate set.

    Structure:
      - n_whole_plots WP "slots", each with subplots_per_wp sub-plots.
      - WP slot i has a single WP factor setting, replicated across all sub-plots.
      - SP factor settings vary freely across sub-plots within a WP.

    Returns DataFrame with all factor columns + `__wp_id__` (int, 0-indexed WP slot).
    """
```

**Implementation strategy:**
1. Generate `C_wp` using the standard `build_candidate` for the HTC factors subset with
   `candidate_points = n_whole_plots * oversampling_factor` (e.g., 5×).
2. Generate `C_sp` for the ETC factors subset.
3. Cross-join: for each of the `n_whole_plots` WP slots, assign a single WP factor row
   (initially random, will be optimized by exchange), and generate `subplots_per_wp` SP rows.
4. The initial WP assignments can use a simple LHS over the WP candidate pool.

**Key constraint the exchange algorithm depends on:**
- All rows with `__wp_id__ == i` must have identical values in all HTC factor columns.
- The candidate pool is a superset; the exchange algorithm selects from it while maintaining
  this constraint.

**Acceptance criteria:**
- Returned DataFrame has `n_whole_plots * subplots_per_wp` rows.
- All rows with the same `__wp_id__` have identical HTC factor values.
- `__wp_id__` values span 0..n_whole_plots-1.
- With pure-continuous factors, WP factor settings are inside factor bounds.
- With `htc_factors=[]` (all ETC), falls back to standard candidate generation.

**Test class:** `TestSplitPlotCandidate` in `tests/test_split_plot.py`

---

### SP-4 — GLS criterion scorers

**Goal:** Add GLS variants of the I, D, A criterion scorers that incorporate `V⁻¹`.

**Files changed:**
- `iopt_power_design/iopt_search.py`

**New private functions:**

```python
def _gls_i_criterion(X_sel: np.ndarray, V_inv: np.ndarray, jitter: float = 1e-8) -> float:
    """GLS I-criterion: tr[(X'V⁻¹X)⁻¹ A] where A = X_cand'X_cand / n_cand.

    Lower is better.
    """


def _gls_d_criterion(X_sel: np.ndarray, V_inv: np.ndarray, jitter: float = 1e-8) -> float:
    """GLS D-criterion: -log det(X'V⁻¹X).

    Lower is better (maximizes determinant).
    """


def _gls_a_criterion(X_sel: np.ndarray, V_inv: np.ndarray, jitter: float = 1e-8) -> float:
    """GLS A-criterion: tr[(X'V⁻¹X)⁻¹].

    Lower is better.
    """
```

Extend `_criterion_score()` dispatcher:
```python
def _criterion_score(X_sel, criterion, *, V_inv=None, jitter=1e-8) -> float:
    if V_inv is not None:
        # GLS path
        dispatch to _gls_* functions
    else:
        # OLS path (existing behaviour unchanged)
        dispatch to existing _i/_d/_a_criterion_for_indices()
```

**Key property for tests:** At η=0 (V=I, V_inv=I), GLS criterion = OLS criterion.

**Acceptance criteria:**
- GLS I-criterion equals OLS I-criterion when V_inv = identity.
- GLS D-criterion equals OLS D-criterion when V_inv = identity.
- GLS criteria are invariant under orthogonal transformation of rows (same optimal design
  regardless of run ordering within WPs).
- `_criterion_score` dispatcher correctly routes to GLS/OLS based on `V_inv` presence.

**Test class:** `TestGLSCriterionScorers` in `tests/test_split_plot.py`

---

### SP-5 — Split-plot Fedorov exchange algorithm

**Goal:** Implement a modified Fedorov point exchange that respects the whole-plot nesting
constraint and optimizes over both WP and SP factor settings jointly using the GLS criterion.

**Files changed:**
- `iopt_power_design/iopt_search.py`

**New function:**

```python
def build_split_plot_design(
    cand: pd.DataFrame,
    formula: str,
    n_wp: int,
    subplots_per_wp: int,
    htc_factors: List[str],
    eta: float,
    *,
    criterion: str = "I",
    starts: int = 10,
    max_iter: int = 100,
    random_state: Optional[int] = None,
    jitter: float = 1e-8,
    criterion_ignore_vr: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a split-plot optimal design using a two-phase Fedorov exchange.

    Parameters
    ----------
    cand : DataFrame
        Candidate set (output of build_split_plot_candidate).
        Must have `__wp_id__` column.
    formula : str
        Patsy formula for the model (full formula including HTC and ETC factors).
    n_wp : int
        Number of whole plots.
    subplots_per_wp : int
        Sub-plots per whole plot (balanced layout assumed).
    htc_factors : List[str]
        Names of the HTC (whole-plot) factors.
    eta : float
        Variance ratio σ²_wp / σ²_sp.

    Returns
    -------
    design_df : DataFrame
        Selected runs with `__wp_id__` column.
    X : ndarray, shape (n_wp * subplots_per_wp, p)
        Model matrix for the selected design.
    """
```

**Two-phase exchange loop (per start):**

```
Initialize:
  - Start with a random selection of n_wp WP factor settings from WP candidate pool.
  - For each WP, randomly select subplots_per_wp SP factor settings from SP candidate pool.
  - Build initial X and Z; compute V_inv; compute initial GLS criterion value.

Iterate until convergence (no improving swap found) or max_iter:

  Phase 1 — WP swaps:
    For each WP i in 0..n_wp-1:
      For each WP candidate w in WP candidate pool:
        If w ≠ current WP-i setting:
          Propose: replace ALL sub-plots in WP i with new HTC = w (keep SP settings).
          Recompute X_proposed (n_total × p), Z unchanged.
          Compute criterion gain Δ = criterion(X_proposed) - criterion(X_current).
          If Δ < 0 (improvement): accept, update X_current and WP-i record.

  Phase 2 — SP swaps:
    For each sub-plot (i, j) in 0..n_total-1:
      For each SP candidate s in SP candidate pool:
        If s ≠ current SP setting for run (i,j):
          Propose: replace SP setting for run (i,j) with s (keep HTC for WP i).
          Recompute X_proposed.
          Compute criterion gain.
          If Δ < 0: accept.

Record criterion value; keep best design across all starts.
```

**Important:** `V_inv` depends only on the whole-plot structure (n_wp, subplots_per_wp, eta),
not on factor settings. Since the layout is balanced and fixed, `V_inv` is computed once at
the start and reused throughout the exchange.

**Rank-1 update shortcut for SP swaps:**
SP swaps change only one row of X. The standard Sherman-Morrison rank-1 update applies directly.
For WP swaps (which change `subplots_per_wp` rows simultaneously), use a rank-k update.
Both are extensions of the existing `_fedorov_exchange_single` logic.

**Acceptance criteria:**
- Returned design has n_wp * subplots_per_wp rows.
- All sub-plots within a WP share identical HTC factor values.
- With eta=0 and criterion_ignore_vr=True, produces a design close to standard I-optimal (not
  necessarily identical due to nesting constraint, but same criterion value within tolerance).
- Multiple starts produce different designs; best is returned.
- Runs within budget of `max_iter` iterations without infinite loops.

**Test class:** `TestSplitPlotExchange` in `tests/test_split_plot.py`

---

### SP-6 — GLS power calculation

**Goal:** Add GLS variants of the power functions in `power.py`.

**Files changed:**
- `iopt_power_design/power.py`

**New functions:**

```python
def contrast_power_sp(
    L: np.ndarray,
    delta: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    sigma_sp: float,
    eta: float,
    alpha: float,
    *,
    df_method: str = "auto",
    jitter: float = 1e-8,
) -> ContrastPowerResult:
    """Power for a linear contrast in a split-plot design.

    Uses GLS information matrix M = X'V⁻¹X where V = σ²_sp(ηZZ' + I).
    The non-centrality parameter is:
        λ = δ' [L M⁻¹ L']⁻¹ δ / σ²_sp

    Denominator df is computed per-contrast according to df_method.
    The overall power = min power across all contrast rows (same convention
    as the OLS version for multi-row L).
    """


def global_r2_power_sp(
    r2_target: float,
    X: np.ndarray,
    Z: np.ndarray,
    sigma_sp: float,
    eta: float,
    alpha: float,
    *,
    df_method: str = "auto",
    lambda_mode: str = "n",
    jitter: float = 1e-8,
) -> GlobalPowerResult:
    """Power for the global R² F-test in a split-plot design.

    Uses GLS F-statistic under two-stratum variance model.
    Non-centrality: λ = f² * tr(V⁻¹ H) where H = X(X'V⁻¹X)⁻¹X'V⁻¹ is the
    GLS hat matrix and f² = r2_target / (1 - r2_target).
    df_denom = n_total - n_wp (approximate SP df, used for global R² test).
    """
```

**Key property:** Both functions with eta=0 must return results identical (within numerical
tolerance) to the existing OLS `contrast_power` and `global_r2_power`.

**Acceptance criteria:**
- `contrast_power_sp(..., eta=0)` equals `contrast_power(...)` for the same X.
- `global_r2_power_sp(..., eta=0)` equals `global_r2_power(...)` for the same X.
- As eta increases (more whole-plot variance), power for WP-factor contrasts decreases.
- As eta increases, power for SP-factor contrasts is less affected.
- `df_method="conservative"` always returns lower power than `"sp_only"` for WP contrasts.

**Test class:** `TestSplitPlotPower` in `tests/test_split_plot.py`

---

### SP-7 — Top-level API integration

**Goal:** Wire split-plot design generation and power calculation into `find_optimal_design`.

**Files changed:**
- `iopt_power_design/api.py`

**Logic to add in `find_optimal_design`:**

```python
if design_opts.split_plot is not None:
    sp_opts = design_opts.split_plot
    # Validate that all htc_factors are keys in `factors`
    _validate_htc_factors(sp_opts.htc_factors, factors)
    # Auto-compute subplots_per_wp if not set
    subplots_per_wp = sp_opts.subplots_per_wp or _auto_subplots_per_wp(p, sp_opts.n_whole_plots)
    # Build split-plot candidate
    cand = build_split_plot_candidate(factors, sp_opts.htc_factors, sp_opts.n_whole_plots,
                                      subplots_per_wp, random_state=design_opts.random_state)
    # Build split-plot design
    design_df, X = build_split_plot_design(cand, formula, sp_opts.n_whole_plots,
                                           subplots_per_wp, sp_opts.htc_factors,
                                           sp_opts.eta, ...)
    # Build Z
    Z = build_whole_plot_indicator(len(design_df), sp_opts.n_whole_plots, subplots_per_wp)
    # Compute power
    power_result = contrast_power_sp(...) or global_r2_power_sp(...)
    # Add SP-specific report fields
    report["n_whole_plots"] = sp_opts.n_whole_plots
    report["subplots_per_wp"] = subplots_per_wp
    report["eta"] = sp_opts.eta
    report["df_method"] = sp_opts.df_method
```

**n-search loop for split-plot:**

The existing binary search sweeps over `n` (total runs). For split-plot, the outer loop sweeps
`n_whole_plots` (1..max_n_wp where max_n_wp = power_cfg.max_n // subplots_per_wp). `n_total`
is then `n_whole_plots * subplots_per_wp`.

This requires extracting the n-search loop into a helper that accepts a "design size" abstraction:
- Non-SP: design size = total run count
- SP: design size = n_whole_plots (subplots_per_wp fixed)

**New private helpers:**
- `_validate_htc_factors(htc_factors, factors)` — raises if any HTC factor not in `factors`
- `_auto_subplots_per_wp(p, n_wp)` — returns `max(2, ceil(p / n_wp) + 1)`
- `_is_split_plot(design_opts)` — convenience predicate

**Report additions:**
```python
report["split_plot"] = {
    "n_whole_plots": ...,
    "subplots_per_wp": ...,
    "n_total": ...,
    "eta": ...,
    "htc_factors": [...],
    "etc_factors": [...],
    "df_method": ...,
}
```

**Acceptance criteria:**
- `find_optimal_design` with `split_plot=None` is behaviorally unchanged (all existing tests pass).
- With `split_plot` set, returns a design where all sub-plots in each WP share HTC factor values.
- Report contains `split_plot` sub-dict.
- `ValueError` raised if any `htc_factor` is not in `factors`.
- `ValueError` raised if `n_blocks` is also set (not yet supported together).

**Test class:** `TestSplitPlotAPI` in `tests/test_split_plot.py`

---

### SP-8 — Analysis functions

**Goal:** Extend `analysis.py` with split-plot-aware analysis helpers.

**Files changed:**
- `iopt_power_design/analysis.py`
- `iopt_power_design/__init__.py`

**New function:**

```python
def power_curve_by_wp(
    formula: str,
    factors: dict,
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    subplots_per_wp: int,
    htc_factors: List[str],
    eta: float,
    *,
    wp_range: Optional[Tuple[int, int]] = None,
    wp_points: int = 10,
    design_opts: Optional[DesignOptions] = None,
    plot_backend: str = "matplotlib",
    figsize: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Power vs number of whole plots curve for a split-plot design.

    Sweeps n_whole_plots from wp_range[0] to wp_range[1], builds a new
    split-plot design at each size, evaluates GLS power, and returns a
    DataFrame with columns: n_wp, n_total, power.
    """
```

**Extend `power_sensitivity` for split-plot:**
- When a `SplitPlotOptions` is present in `design_opts`, add an `eta` sweep axis:
  vary `eta` from `eta_range[0]` to `eta_range[1]` on the fixed design.
- New parameters: `eta_range: Optional[Tuple[float, float]] = None`, `eta_points: int = 20`.

**Acceptance criteria:**
- `power_curve_by_wp` returns a DataFrame with at least `wp_points` rows.
- Power generally increases as `n_whole_plots` increases (not guaranteed but typical).
- `power_sensitivity` with `eta_range=(0, 5)` on a split-plot design shows power decreasing
  as eta increases for WP-factor contrasts.

**Test class:** `TestSplitPlotAnalysis` in `tests/test_split_plot.py`

---

### SP-9 — CLI / Streamlit / Sheets / Excel integration

**Goal:** Expose split-plot options across all user-facing surfaces.

**Files changed:**
- `iopt_power_design/cli.py`
- `app/pages/2_Power_Config.py`
- `app/state.py`
- `app/pages/3_Run_Results.py`
- `app/pages/4_Analysis.py`
- `iopt_power_design/sheets.py`
- `iopt_power_design/excel_template.py`

**CLI additions:**

```
lattice --config config.yaml \
  --htc-factors A,B \
  --n-whole-plots 6 \
  --eta 2.0 \
  --subplots-per-wp 4 \
  --df-method auto
```

Or via YAML config (new `split_plot:` section):
```yaml
split_plot:
  htc_factors: [A, B]
  n_whole_plots: 6
  eta: 2.0
  subplots_per_wp: 4
  df_method: auto
```

**Streamlit additions (2_Power_Config.py):**
- New expander section "D6 — Split-Plot / Hard-to-Change Factors"
- Toggle: "Enable split-plot design"
- Multi-select: "Hard-to-change (WP) factors" (populated from factor table)
- Number input: "Number of whole plots"
- Number input: "Sub-plots per whole plot (0 = auto)"
- Slider: "Variance ratio η (σ²_wp / σ²_sp)"
- Select: "df method"

**Sheets additions (`sheets.py`):**
- New `[SETTINGS]` rows: `htc_factors`, `n_whole_plots`, `eta`, `subplots_per_wp`, `df_method`

**Excel additions (`excel_template.py`):**
- Same 5 rows added to `[SETTINGS]` block with sensible defaults

**Acceptance criteria:**
- `lattice --htc-factors A --n-whole-plots 4 --eta 1.0 --config myconfig.yaml` runs without error.
- YAML config with `split_plot:` block round-trips correctly.
- Streamlit UI shows the SP section only when toggled on.
- Sheets/Excel connectors read and write all 5 SP fields.

**Test class:** `TestSplitPlotCLI` / `TestSplitPlotSheets` / `TestSplitPlotExcel`

---

### SP-10 — Tests

**Goal:** Comprehensive test coverage for the entire split-plot feature. Tests should be written
*alongside each ticket*, not as a separate pass; this ticket tracks the integration regression
tests only.

**Files changed:**
- `tests/test_split_plot.py` (new — unit tests, per SP-2 through SP-8)
- `tests/test_api.py` (integration regression tests added to existing file)
- `tests/test_config.py` (config validation tests added)

**Integration regression tests (added to test_api.py):**

```python
class TestSplitPlotIntegration:
    """End-to-end regression tests for split-plot designs."""

    def test_2wp_3sp_contrast_mode(self):
        """Basic 2-factor model, 1 HTC, 1 ETC, 2 WPs, 3 SP each."""
        ...

    def test_htc_factor_collision_raises(self):
        """htc_factors contains a name not in factors dict → ValueError."""
        ...

    def test_split_plot_and_blocked_raises(self):
        """Setting both n_blocks and split_plot raises ValueError."""
        ...

    def test_eta_zero_matches_ols_power(self):
        """Power at eta=0 equals OLS power for the same design."""
        ...

    def test_design_respects_nesting(self):
        """All sub-plots within a WP share identical HTC factor values."""
        ...

    def test_report_contains_split_plot_dict(self):
        """Result report includes 'split_plot' sub-dict with expected keys."""
        ...

    def test_power_curve_by_wp_returns_dataframe(self):
        """power_curve_by_wp returns DF with n_wp and power columns."""
        ...
```

**Property-based tests (using hypothesis if available, else parametrized):**
- For balanced layouts, `build_whole_plot_indicator` produces a correct block-diagonal Z'Z.
- GLS criterion scorers agree with explicit matrix computation for small random examples.
- `split_plot_df_denom` with "conservative" always ≤ result from "sp_only".

**Acceptance criteria:**
- All split-plot unit tests pass.
- All pre-existing tests continue to pass (no regressions).
- `pytest -m "not slow"` completes in < 5 minutes.

---

## Implementation Order and Suggested Chunking

The tickets are designed to be implemented in session-sized chunks:

| Session | Tickets | Deliverable |
|---------|---------|-------------|
| 1 | ~~SP-1 + SP-2~~ | ~~Config dataclass + covariance math~~ ✅ |
| 2 | SP-3 | Candidate generation for split-plot |
| 3 | SP-4 | GLS criterion scorers |
| 4 | SP-5 (init + WP-swap) | First half of exchange algorithm |
| 5 | SP-5 (SP-swap + multi-start) | Complete exchange algorithm |
| 6 | SP-6 | GLS power calculation |
| 7 | SP-7 | API integration (end-to-end first working example) |
| 8 | SP-8 | Analysis functions (`power_curve_by_wp`, sensitivity sweep) |
| 9 | SP-9 (CLI + YAML) | CLI surface |
| 10 | SP-9 (Streamlit) | Streamlit UI |
| 11 | SP-9 (Sheets + Excel) | Connector integration |
| 12 | SP-10 + cleanup | Integration regression tests, commit |

---

## Open Questions / Risks

| # | Question | Risk level | Notes |
|---|----------|-----------|-------|
| 1 | **Unbalanced WPs** (different subplots_per_wp per WP) | Medium | The Woodbury identity has a simple closed form only for balanced layouts. Unbalanced V_inv requires `n × n` matrix inversion. Start with balanced; add unbalanced as SP-11. |
| 2 | **Mixed HTC/ETC interactions** (e.g., A×C where A=HTC, C=ETC) | Low | These are SP-level effects (tested against SP error) because they vary within a WP. The `classify_contrasts` logic in SP-2 handles this correctly by checking whether *any* column is ETC. |
| 3 | **Blocked split-plots** (whole plots nested within blocks) | High | Three variance strata: block, WP-within-block, SP. Not in scope for this enhancement. |
| 4 | **Kenward-Roger df** | Medium | KR df approximation is more accurate than the simple WP/SP split, especially for unbalanced designs with interactions. Deferred to SP-11. |
| 5 | **API server (Enhancement 21) support** | Low | `SplitPlotOptions` is a dataclass; `api_server/models/common.py` needs a `SplitPlotOptionsModel` Pydantic model and a corresponding conversion in `serialization.py`. This is a small addition to SP-7. |
| 6 | **Convergence of two-phase exchange** | Medium | The alternating WP/SP swap may cycle without converging. Add a staleness counter: terminate if no improvement in 3 consecutive full passes. |

---

## Reference Material

- Jones, B. & Nachtsheim, C.J. (2009). "Split-plot designs: What, why and how." *Journal of
  Quality Technology*, 41(4), 340–361. — Definitive practitioner guide.
- Goos, P. & Jones, B. (2011). *Optimal Design of Experiments: A Case Study Approach*. — Chapters
  8–10 cover split-plot I-optimal designs with GLS criterion.
- Letsinger, J.D., Myers, R.H. & Lentner, M. (1996). "Response surface methods for bi-randomization
  structures." *Journal of Quality Technology*, 28(4), 381–397. — Original bi-randomization theory.
