# Enhancement 25 — GLM Support (Logistic / Poisson)
## Ticket Pack

**Status:** ⬜ Not started
**Target:** Add logistic and Poisson regression as first-class design families alongside the existing Gaussian linear model.

---

## Background & Scope

### What users want

Experimenters studying binary outcomes (pass/fail, conversion, survival) or count outcomes
(defects, events per unit time) need I-optimal designs whose sample sizes are calibrated
to the actual likelihood model — not a Gaussian surrogate. A logistic or Poisson regression
design requires:

1. Specifying the **distributional family** (`binomial` / `poisson`) and the **link function**
   (logit / log — canonical defaults).
2. Specifying a **baseline** mean on the response scale (e.g., 20 % event probability or
   3.5 counts per trial) so that the Fisher information at the design point can be computed.
3. Expressing the **effect size** on the **linear predictor (LP) scale** — log-odds difference
   for logistic, log-rate difference for Poisson — rather than in standard-deviation units.
4. Getting the **minimum n** to achieve a target power, computed via the **Wald chi-square
   distribution** (df = rank of contrast), not the noncentral F.

### Two-phase strategy

**Phase 1 — Null-based locally optimal design (these tickets)**
Design candidates are scored under a *null* assumption: all observations share the same
baseline probability / rate (β = β₀ intercept-only model). Under this assumption the
Fisher information matrix reduces to:

    M = w · X′X    where  w = p₀(1 − p₀)  [binomial]
                           w = μ₀          [Poisson]

Because `w` is a positive scalar, the **I/D/A-optimal design structure is identical to OLS**.
The existing Fedorov exchange in `iopt_search.py` requires *no algorithmic changes* for
Phase 1. Only the power functions and config layer change.

**Phase 2 — Alternative-based locally optimal design (GL-10, GL-11 — deferred)**
For large effects where the linear predictor moves substantially across the design space,
the weights `wᵢ = μᵢ(1 − μᵢ)` vary by row. This requires a *weighted* Fedorov exchange
(`M = X′WX` with non-constant `W`). Tickets GL-10 and GL-11 cover this.

### Mathematical summary

| Quantity | Linear (OLS) | GLM Phase 1 (null-based) |
|---|---|---|
| Information matrix M | X′X / σ² | w · X′X (w from baseline) |
| Noncentrality λ | δᵀ [L M⁻¹ Lᵀ]⁻¹ δ | δᵀ [L M⁻¹ Lᵀ]⁻¹ δ (no σ²) |
| Test distribution | Noncentral F(q, n−p, λ) | Noncentral χ²(q, λ) |
| Effect size δ | Absolute on response scale | LP scale (log-odds / log-rate) |
| Effective σ² | σ² (user-supplied) | 1/w (derived from baseline) |

> **Code reuse insight:** The noncentrality formula for a GLM null-based design is
> equivalent to OLS with `sigma_eff = 1/√w`. The only substantive code change is
> replacing `scipy.stats.ncf` (noncentral F) with `scipy.stats.ncx2` (noncentral χ²).

---

## Tickets

| ID | Title | Files | Est. Tests | Dep |
|---|---|---|---|---|
| GL-1 | GLM config dataclasses | `config.py` | 25 | — |
| GL-2 | GLM power functions (Wald χ²) | `power.py` | 30 | GL-1 |
| GL-3 | GLM design API integration | `api.py`, `analysis.py`, `__init__.py` | 20 | GL-1, GL-2 |
| GL-4 | GLM power curves and baseline sweep | `analysis.py` | 20 | GL-3 |
| GL-5 | GLM min-detectable-effect and compare_criteria | `analysis.py` | 15 | GL-4 |
| GL-6 | CLI and YAML template support | `cli.py` | 15 | GL-3 |
| GL-7 | REST API GLM support | `api_server/` | 20 | GL-3 |
| GL-8 | Streamlit UI for GLM | `app/` | manual | GL-3 |
| GL-9 | Sheets and Excel GLM connector support | `sheets.py`, `excel_template.py` | 20 | GL-3 |
| GL-10 | Weighted Fedorov exchange (Phase 2 — deferred) | `iopt_search.py` | 25 | GL-2 |
| GL-11 | Alternative-based locally optimal GLM designs (Phase 2 — deferred) | `api.py`, `analysis.py` | 20 | GL-10 |

---

## GL-1 — GLM Config Dataclasses

**Status:** ✅ Completed
**Files:** `iopt_power_design/config.py`
**Estimated new tests:** 25

### Goal

Add `PowerGLMContrastConfig` as the primary GLM power configuration dataclass and update
the `PowerCfg` union type throughout the codebase so GLM configs flow through the same
infrastructure as OLS configs.

### New dataclass: `PowerGLMContrastConfig`

```python
@dataclass
class PowerGLMContrastConfig:
    """Power configuration for GLM contrast test (Wald chi-square).

    Parameters
    ----------
    L : ndarray, shape (q, p)
        Contrast matrix. Same semantics as PowerContrastConfig.L.
    delta : ndarray, shape (q,)
        Effect sizes on the LINEAR PREDICTOR scale.
        Logistic: log-odds difference (e.g. 0.5 means ~0.5 nat increase in logit).
        Poisson:  log-rate difference (e.g. 0.3 means ~30 % increase in rate).
    baseline : float
        Baseline mean on the RESPONSE scale under the null / reference scenario.
        Binomial: event probability ∈ (0, 1), e.g. 0.20 for 20 % baseline rate.
        Poisson:  expected count > 0, e.g. 2.5 events per unit.
    family : {'binomial', 'poisson'}, default 'binomial'
    link : {'logit', 'log'} or None, default None
        None selects the canonical link for the family (logit for binomial, log for Poisson).
    alpha : float, default 0.05
    power : float, default 0.80
    tol_power : float, default 1e-3
    max_iter : int, default 200
    max_n : int, default 2000
    """
    L: np.ndarray
    delta: np.ndarray
    baseline: float
    family: Literal["binomial", "poisson"] = "binomial"
    link: Optional[Literal["logit", "log"]] = None
    alpha: float = 0.05
    power: float = 0.80
    tol_power: float = 1e-3
    max_iter: int = 200
    max_n: int = 2000
```

### Validation (`__post_init__`)

- `L` must be 2-D; `delta` length must equal `L.shape[0]` (one effect per contrast row).
- `family` must be one of `"binomial"`, `"poisson"`.
- `link` must be `None` or `"logit"` (binomial) / `"log"` (Poisson); cross-family link raises
  `ValueError`.
- `baseline ∈ (0, 1)` for `"binomial"`, `baseline > 0` for `"poisson"`.
- `alpha ∈ (0, 0.5)`, `power ∈ (0, 1)`.
- `tol_power > 0`, `max_iter ≥ 1`, `max_n ≥ 1`.

### Helper: `glm_fisher_weight`

Add a module-level helper (not exported to `__init__`) that returns the scalar Fisher
information weight at the baseline:

```python
def glm_fisher_weight(cfg: PowerGLMContrastConfig) -> float:
    """Return w = variance function evaluated at baseline mean.

    Binomial: w = p₀(1 − p₀)
    Poisson:  w = μ₀
    """
```

### Union type update

Update `PowerCfg` type alias wherever it appears in `config.py`, `api.py`, `analysis.py`,
`power.py`, and docstrings:

```python
# Before
PowerCfg = Union[PowerContrastConfig, PowerR2Config]

# After
PowerCfg = Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig]
```

Update `ResponseSpec.__post_init__` validation if it checks `isinstance(power_cfg, ...)`.

### Exports

Add `PowerGLMContrastConfig` and `glm_fisher_weight` to `__init__.py` (public API).

### Tests (`tests/test_config.py` — new class `TestPowerGLMContrastConfig`)

1. `test_default_family_is_binomial` — field default
2. `test_canonical_link_none_accepted`
3. `test_logit_link_accepted_for_binomial`
4. `test_log_link_accepted_for_poisson`
5. `test_log_link_rejected_for_binomial` — ValueError
6. `test_logit_link_rejected_for_poisson` — ValueError
7. `test_baseline_zero_raises_for_binomial`
8. `test_baseline_one_raises_for_binomial`
9. `test_baseline_negative_raises_for_poisson`
10. `test_baseline_zero_raises_for_poisson`
11. `test_delta_length_mismatch_raises`
12. `test_L_1d_raises` (must be 2-D)
13. `test_alpha_out_of_range_raises`
14. `test_valid_binomial_config_creates`
15. `test_valid_poisson_config_creates`
16. `test_glm_fisher_weight_binomial` — p₀=0.5 → w=0.25
17. `test_glm_fisher_weight_binomial_asymmetric` — p₀=0.2 → w=0.16
18. `test_glm_fisher_weight_poisson` — μ₀=2.5 → w=2.5
19. `test_glm_config_is_valid_power_cfg_type` — isinstance check passes
20. `test_dataclasses_replace_safe` — `dataclasses.replace(cfg, alpha=0.01)` works
21. `test_response_spec_accepts_glm_config` — `ResponseSpec("Y", glm_cfg)` OK
22. `test_max_n_default_is_2000`
23. `test_tol_power_nonpositive_raises`
24. `test_max_iter_zero_raises`
25. `test_multiline_L_matrix_accepted`

---

## GL-2 — GLM Power Functions (Wald χ²)

**Status:** ✅ Completed
**Files:** `iopt_power_design/power.py`
**Estimated new tests:** 30
**Depends on:** GL-1

### Goal

Add `glm_contrast_power()` which computes power for a GLM contrast test using the **Wald
chi-square** approximation and the **null-based locally optimal** information matrix
`M = w · X′X`. Extend `eval_response_power` to dispatch to this function when given a
`PowerGLMContrastConfig`.

### New function: `glm_contrast_power`

```python
def glm_contrast_power(
    cfg: PowerGLMContrastConfig,
    X: np.ndarray,
    jitter: float = 1e-8,
) -> ContrastPowerResult:
    """Wald chi-square power for a GLM contrast.

    Uses the null-based locally optimal information matrix M = w·X′X where
    w = glm_fisher_weight(cfg).  The test statistic is:

        T = (L β̂)ᵀ [L (X′WX)⁻¹ Lᵀ]⁻¹ (L β̂)  ~  χ²(q)  under H₀

    Under the alternative Lβ = δ, T ~ χ²(q, λ) (noncentral) with:

        λ = δᵀ [L · (w · X′X)⁻¹ · Lᵀ]⁻¹ δ
          = w · δᵀ [L · (X′X)⁻¹ · Lᵀ]⁻¹ δ

    Power = P(χ²(q, λ) > χ²_{α}(q))

    Parameters
    ----------
    cfg : PowerGLMContrastConfig
    X : ndarray, shape (n, p)
        Model matrix for the proposed design (already built from formula + design df).
    jitter : float
        Ridge added to X′X before inversion (same role as OLS jitter).

    Returns
    -------
    ContrastPowerResult(power, lam)
        power : Achieved power ∈ [0, 1].
        lam   : Noncentrality parameter λ.
    """
```

**Implementation sketch:**

```python
_require_scipy()
from scipy.stats import chi2, ncx2

w = glm_fisher_weight(cfg)                    # scalar
XtX_inv = _pinv_xtx(X, jitter)               # reuse existing helper
LXtXinvLt = cfg.L @ XtX_inv @ cfg.L.T        # (q, q)
# Noncentrality: lambda = w * delta' @ inv(L(X'X)^-1 L') @ delta
lam = float(w * cfg.delta @ np.linalg.pinv(LXtXinvLt) @ cfg.delta)
q = cfg.L.shape[0]
chi2_crit = chi2.ppf(1.0 - cfg.alpha, df=q)
power = float(1.0 - ncx2.cdf(chi2_crit, df=q, nc=lam))
return ContrastPowerResult(power=power, lam=lam)
```

> **Note:** `_pinv_xtx` and `ContrastPowerResult` already exist in `power.py`. No new helpers
> needed beyond `scipy.stats.ncx2`.

### Extend `eval_response_power`

The existing dispatcher (lines ~621–714) currently handles `PowerContrastConfig` and
`PowerR2Config` branches. Add a third branch:

```python
elif isinstance(response.power_cfg, PowerGLMContrastConfig):
    result = glm_contrast_power(response.power_cfg, X, jitter=jitter)
    return {
        "name": response.name,
        "power": result.power,
        "lam": result.lam,
        "df1": response.power_cfg.L.shape[0],   # q (contrast rows)
        "df2": None,                              # no denominator df in chi-square test
        "family": response.power_cfg.family,
        "baseline": response.power_cfg.baseline,
    }
```

### `eval_response_power` return dict changes

`df2` can now be `None` for GLM paths. Callers that display `df2` must guard against `None`.

### Tests (`tests/test_power.py` — new class `TestGLMContrastPower`)

**Numerical correctness:**

1. `test_binomial_power_increases_with_n` — build X for n=10, n=20, n=40; power rises
2. `test_binomial_power_monotone_in_delta` — larger δ → higher power
3. `test_binomial_power_symmetric_around_baseline_half` — p₀=0.5 is max-info baseline
4. `test_binomial_lower_power_at_extreme_baseline` — p₀=0.1 < power(p₀=0.5) for same n,δ
5. `test_poisson_power_increases_with_n`
6. `test_poisson_power_increases_with_baseline_rate` — higher μ₀ → more Fisher info → higher power
7. `test_power_returns_named_tuple`
8. `test_lam_matches_manual_formula` — compute λ by hand, compare
9. `test_identity_design_matrix_gives_finite_power`
10. `test_singular_X_with_jitter_doesnt_crash`

**Equivalence with OLS:**

11. `test_glm_matches_ols_at_sigma_eff` — GLM power with p₀=0.5 (w=0.25, σ_eff=2)
    must equal OLS contrast power with identical X, L, δ, σ=2, using F→χ² approximation
    (large n limit: F(1, n-p, λ) ≈ χ²(1, λ) for large n)

**Edge cases:**

12. `test_zero_delta_gives_alpha_power` — λ=0, power=α
13. `test_large_delta_gives_power_near_one`
14. `test_multirow_L_matrix` — rank-2 contrast
15. `test_power_below_alpha_is_valid` — very small n, power < α is OK

**`eval_response_power` dispatch:**

16. `test_eval_dispatch_glm_config` — returns dict with correct keys
17. `test_eval_df2_is_none_for_glm`
18. `test_eval_family_key_present`
19. `test_eval_baseline_key_present`
20. `test_eval_glm_power_matches_direct_call`

**Poisson-specific:**

21. `test_poisson_weight_equals_baseline_rate`
22. `test_poisson_power_for_log_rate_delta`
23. `test_poisson_high_rate_high_power` — μ₀=10, δ=0.2

**Regression guard:**

24. `test_ols_contrast_power_unchanged` — existing OLS tests still pass
25. `test_ols_r2_power_unchanged` — existing R² tests still pass

**Extra coverage:**

26. `test_glm_power_zero_n_raises` — n=0 raises gracefully
27. `test_glm_power_all_same_rows` — degenerate constant design
28. `test_glm_power_with_categorical_X` — categorical-coded design matrix
29. `test_eval_glm_response_spec_integration`
30. `test_glm_lam_nonnegative`

---

## GL-3 — GLM Design API Integration

**Status:** ✅ Completed
**Files:** `iopt_power_design/api.py`, `iopt_power_design/__init__.py`
**Estimated new tests:** 20
**Depends on:** GL-1, GL-2

### Goal

Make `find_optimal_design` accept a `PowerGLMContrastConfig`, run the **existing**
Fedorov exchange unchanged (Phase 1 — null-based optimal design), and evaluate power via
`glm_contrast_power`. The `report` dict gains GLM-specific keys.

### Changes to `find_optimal_design`

1. **Input validation:** extend `_validate_api_inputs` to accept `PowerGLMContrastConfig`
   in addition to existing types.

2. **Power evaluation dispatch:** `eval_response_power` already dispatches on config type
   (after GL-2). No further changes needed in the design-search loop — GLM uses the same
   I/D/A criteria over `X′X` (not `X′WX`, which is Phase 2).

3. **Report dict:** Add GLM-specific fields when `isinstance(power_cfg, PowerGLMContrastConfig)`:

   ```python
   # Additional report keys for GLM
   "family":   power_cfg.family,           # "binomial" | "poisson"
   "link":     power_cfg.link or "<canonical>",
   "baseline": power_cfg.baseline,
   "glm_weight": glm_fisher_weight(power_cfg),  # w scalar
   "test_type":  "wald_chi2",              # distinguish from "f" (OLS)
   "df1":    power_cfg.L.shape[0],         # contrast rows
   "df2":    None,                         # no denominator df for chi-square
   ```

   Keep existing OLS report keys (`sigma`, `noncentrality_lambda`, etc.) under their
   original names. Add `"test_type": "f"` to existing OLS report for symmetry.

4. **Sigma guard:** If user passes `PowerGLMContrastConfig` but the code path accidentally
   tries to read `.sigma`, raise a clear `AttributeError` message: "GLM configs do not
   have a sigma parameter. Use `baseline` instead."

5. **`n_search` loop:** No changes needed — it already calls `eval_response_power` each
   iteration, which now dispatches to `glm_contrast_power`.

### Export updates (`__init__.py`)

Add `PowerGLMContrastConfig` to the public `__all__` export list. Also export `glm_fisher_weight`.

### Tests (`tests/test_api.py` — new class `TestGLMDesignAPI`)

**Happy path:**

1. `test_binomial_design_runs_without_error` — 2 continuous factors, logistic, baseline=0.2
2. `test_binomial_report_has_family_key`
3. `test_binomial_report_has_test_type_wald_chi2`
4. `test_binomial_design_df_has_factor_columns`
5. `test_binomial_achieved_power_between_0_and_1`
6. `test_poisson_design_runs_without_error`
7. `test_poisson_report_has_baseline_key`

**n-search convergence:**

8. `test_binomial_n_search_converges_to_target_power`
   (small target, e.g., power=0.70 with δ=0.5 log-odds, p₀=0.3 — should converge quickly)
9. `test_binomial_n_search_grows_for_harder_problem`
   (baseline=0.05 is extreme, more n needed)

**Report structure:**

10. `test_ols_report_has_test_type_f` — verify backward compat key added
11. `test_glm_report_df2_is_none`
12. `test_glm_report_glm_weight_close_to_p0_times_1_minus_p0`

**Design options:**

13. `test_glm_with_d_criterion`
14. `test_glm_with_a_criterion`
15. `test_glm_with_constraint_expr`

**Categorical factors:**

16. `test_binomial_with_categorical_factor`

**Error handling:**

17. `test_glm_config_with_missing_baseline_raises` — impossible to construct (caught by GL-1 validation)
18. `test_glm_config_wrong_family_raises` — ValueError at config construction

**Backward compat:**

19. `test_ols_contrast_api_unchanged` — existing result shape preserved
20. `test_ols_r2_api_unchanged`

---

## GL-4 — GLM Power Curves and Baseline Sweep

**Status:** ✅ Completed
**Files:** `iopt_power_design/analysis.py`
**Estimated new tests:** 20
**Depends on:** GL-3

### Goal

`power_curve_by_n` already iterates over n values and calls `eval_response_power` — after
GL-3, it will automatically support GLM configs with no changes. This ticket adds:

1. Verify / guard existing `power_curve_by_n` and `power_curve_by_effect` for GLM configs.
2. Add new **`power_curve_by_baseline`** — sweeps the baseline probability/rate and shows
   how power changes (unique to GLMs; no OLS analogue).
3. Update `multiresponse` curves to handle `PowerGLMContrastConfig` in `ResponseSpec`.

### Changes to `power_curve_by_n`

Audit the function for any code path that assumes `.sigma` on the config. Guard with
`isinstance` check and fall through to GLM path. The returned DataFrame column `sigma`
becomes `None` (or is absent) for GLM runs; add a `baseline` column instead.

### New function: `power_curve_by_baseline`

```python
def power_curve_by_baseline(
    formula: str,
    factors: Dict[str, Any],
    design_df: pd.DataFrame,
    cfg: PowerGLMContrastConfig,
    baseline_range: Tuple[float, float] = (0.05, 0.95),
    baseline_points: int = 30,
    design_opts: Optional[DesignOptions] = None,
) -> pd.DataFrame:
    """Power as a function of baseline event probability / rate.

    Holds the design fixed and sweeps the baseline mean, recomputing GLM power
    at each point.  Useful for sensitivity analysis: "if my baseline rate is
    really 10 % instead of 20 %, how does my power change?"

    Returns a DataFrame with columns: [baseline, power, lam, family, link].
    """
```

This function is analogous to `power_sensitivity` (which sweeps σ for OLS). It does
**not** re-run the design search — it reuses the provided `design_df`.

### Changes to `power_curve_by_effect`

`power_curve_by_effect` currently sweeps `delta` (absolute, OLS) or `r2_target` (R²).
Add a GLM branch that sweeps `delta` on the LP scale (log-odds / log-rate) while holding
the baseline fixed.

Signature addition: detect `PowerGLMContrastConfig` and:
- Scale `cfg.delta` from the sweep range
- Recompute `glm_contrast_power` at each point

### Tests (`tests/test_analysis.py` — new class `TestGLMPowerCurves`)

**`power_curve_by_n` with GLM:**

1. `test_power_curve_by_n_glm_returns_dataframe`
2. `test_power_curve_by_n_glm_power_increases_with_n`
3. `test_power_curve_by_n_glm_has_baseline_column`
4. `test_power_curve_by_n_glm_no_sigma_column`

**`power_curve_by_baseline`:**

5. `test_by_baseline_returns_dataframe`
6. `test_by_baseline_has_correct_columns`
7. `test_by_baseline_power_peaks_near_p0_half_for_binomial`
   (p₀=0.5 maximises w for binomial → max power)
8. `test_by_baseline_strictly_monotone_away_from_half`
   (power decreases as p₀ → 0 or p₀ → 1)
9. `test_by_baseline_poisson_power_increases_with_baseline`
   (higher μ₀ → more info → more power)
10. `test_by_baseline_uses_provided_design_not_re_optimize`
11. `test_by_baseline_range_respected`

**`power_curve_by_effect` with GLM:**

12. `test_by_effect_glm_returns_dataframe`
13. `test_by_effect_glm_power_increases_with_delta`

**`power_curve_by_n_multiresponse` with mixed configs:**

14. `test_multiresponse_mixed_ols_glm_configs` — one OLS + one GLM response
15. `test_multiresponse_glm_responses_only`

**Regression guards:**

16. `test_ols_power_curve_by_n_unchanged`
17. `test_ols_power_sensitivity_unchanged`

**Edge cases:**

18. `test_by_baseline_single_point_range`
19. `test_by_baseline_poisson_large_rate`
20. `test_by_baseline_very_small_baseline`

---

## GL-5 — GLM min_detectable_effect and compare_criteria

**Status:** ✅ Completed
**Files:** `iopt_power_design/analysis.py`
**Estimated new tests:** 15
**Depends on:** GL-4

### Goal

Extend `min_detectable_effect` and `compare_criteria` to work with `PowerGLMContrastConfig`.

### Changes to `min_detectable_effect`

Currently bisects over a `delta` *scale factor* for contrast mode, or over `r2_target` for
R² mode. Add a GLM branch that bisects over `cfg.delta` (on the LP scale):

```python
elif isinstance(power_cfg, PowerGLMContrastConfig):
    # Bisect over a scale factor s ∈ [lo, hi] applied to cfg.delta
    # Return minimum |delta| on LP scale that achieves target power
```

Return format addition: `min_delta_lp` (minimum effect on linear predictor scale), alongside
or replacing `min_delta` for GLM results.

### Changes to `compare_criteria`

The function currently runs I/D/A-optimal searches and compares power. For GLM configs, the
power comparison still makes sense (I/D/A give different n and hence different power).
Add dispatch so that when `power_cfg` is `PowerGLMContrastConfig`, the result table includes
`family` and `baseline` columns instead of `sigma`.

### Tests (`tests/test_analysis.py` — new class `TestGLMMDE`)

1. `test_mde_glm_returns_dict`
2. `test_mde_glm_has_min_delta_lp_key`
3. `test_mde_glm_larger_n_smaller_min_delta`
4. `test_mde_glm_lower_baseline_larger_min_delta` — extreme baseline → harder detection
5. `test_mde_glm_poisson`
6. `test_mde_glm_respects_target_power`
7. `test_compare_criteria_glm_runs`
8. `test_compare_criteria_glm_returns_dataframe`
9. `test_compare_criteria_glm_has_three_rows` — I, D, A
10. `test_compare_criteria_glm_power_column`
11. `test_compare_criteria_glm_no_sigma_column`
12. `test_compare_criteria_glm_has_family_column`
13. `test_mde_ols_unchanged` — regression guard
14. `test_compare_criteria_ols_unchanged` — regression guard
15. `test_mde_glm_target_power_respected_within_tol`

---

## GL-6 — CLI and YAML Template Support

**Status:** ✅ Completed
**Files:** `iopt_power_design/cli.py`
**Estimated new tests:** 15
**Depends on:** GL-3

### Goal

Expose GLM options through the command-line interface and YAML config format.

### New CLI flags

```
--family {binomial,poisson}   GLM family (default: not set → linear model)
--link   {logit,log}          Link function (default: canonical for family)
--baseline FLOAT              Baseline event probability (binomial) or rate (Poisson).
                              Required when --family is set.
```

When `--family` is specified, `--sigma` is silently ignored (or a warning is emitted).
`--delta` still specifies the effect size, now interpreted on the LP scale.

### YAML schema additions

```yaml
# GLM single-response example
power_mode: contrast     # still required
family: binomial         # new — triggers GLM path
link: logit              # optional; defaults to canonical
baseline: 0.20           # required when family is set
delta: 0.5               # now in log-odds units
L_row: 0, 1, 0           # unchanged
```

Add a `--template glm-binomial` and `--template glm-poisson` template mode that emits a
fully-commented YAML scaffold with GLM fields.

### Template updates

The existing `--template r2` and `--template contrast` templates gain a comment block
explaining the new `family` / `baseline` / `link` fields with an example.

### Config parsing in `run()`

Extend `_make_power_cfg(cfg)` to check `cfg.get("family")`:
- If present: build `PowerGLMContrastConfig` from `family`, `link`, `baseline`, `L`, `delta`
- If absent: existing OLS path unchanged

### Tests (`tests/test_cli.py` — new class `TestGLMCLI`)

1. `test_glm_template_binomial_prints_yaml`
2. `test_glm_template_poisson_prints_yaml`
3. `test_glm_yaml_parses_to_glm_config` — parse a YAML with `family: binomial`
4. `test_glm_yaml_without_baseline_raises`
5. `test_glm_yaml_wrong_family_raises`
6. `test_glm_yaml_with_constraint_expr` — regression: constraints still work
7. `test_glm_template_is_parseable` — round-trip YAML template → parse → run (mocked)
8. `test_linear_yaml_unchanged` — existing contrast YAML not broken
9. `test_family_flag_overrides_yaml_default`
10. `test_baseline_flag_sets_baseline`
11. `test_sigma_ignored_with_family_flag` — no crash
12. `test_link_defaults_to_canonical_logit`
13. `test_link_defaults_to_canonical_log`
14. `test_glm_template_has_comment_explaining_lp_scale`
15. `test_make_power_cfg_glm_returns_glm_config_type`

---

## GL-7 — REST API GLM Support

**Status:** ✅ Completed
**Files:** `api_server/models/common.py`, `api_server/models/design.py`,
           `api_server/routers/design.py`, `api_server/serialization.py`
**Estimated new tests:** 20
**Depends on:** GL-3

### Goal

Extend the REST API so that `POST /design` accepts a GLM power config. No new endpoint
needed — `PowerCfgModel` gains a new discriminated-union variant.

### Pydantic model: `PowerGLMContrastModel`

```python
class PowerGLMContrastModel(BaseModel):
    type: Literal["glm_contrast"] = "glm_contrast"
    L: List[List[float]]              # (q, p) contrast matrix as nested list
    delta: List[float]                # (q,) effect on LP scale
    baseline: float                   # ∈ (0,1) for binomial; > 0 for Poisson
    family: Literal["binomial", "poisson"] = "binomial"
    link: Optional[Literal["logit", "log"]] = None
    alpha: float = 0.05
    power: float = 0.80
    tol_power: float = 1e-3
    max_iter: int = 200
    max_n: int = 2000

    @field_validator("baseline")
    @classmethod
    def validate_baseline(cls, v, info):
        family = info.data.get("family", "binomial")
        if family == "binomial" and not (0.0 < v < 1.0):
            raise ValueError("baseline must be in (0, 1) for binomial family")
        if family == "poisson" and v <= 0:
            raise ValueError("baseline must be > 0 for Poisson family")
        return v
```

### Update `PowerCfgModel`

Add `"glm_contrast"` as a discriminated-union variant:

```python
PowerCfgModel = Annotated[
    Union[PowerR2Model, PowerContrastModel, PowerGLMContrastModel],
    Field(discriminator="type"),
]
```

### Serialization: `pydantic_power_cfg_to_dataclass`

Add branch for `PowerGLMContrastModel`:

```python
elif model.type == "glm_contrast":
    return PowerGLMContrastConfig(
        L=np.array(model.L, dtype=float),
        delta=np.array(model.delta, dtype=float),
        baseline=model.baseline,
        family=model.family,
        link=model.link,
        alpha=model.alpha,
        power=model.power,
        tol_power=model.tol_power,
        max_iter=model.max_iter,
        max_n=model.max_n,
    )
```

### Response model updates

`ReportModel` gains optional GLM fields:

```python
class ReportModel(BaseModel):
    # ... existing fields ...
    test_type: Optional[str] = None      # "f" | "wald_chi2"
    family: Optional[str] = None
    link: Optional[str] = None
    baseline: Optional[float] = None
    glm_weight: Optional[float] = None
    df2: Optional[int] = None            # now Optional (None for GLM)
```

### Tests (`tests/test_api_server.py` — new class `TestGLMDesignEndpoint`)

**Binomial:**

1. `test_glm_binomial_returns_200`
2. `test_glm_binomial_response_has_design_df`
3. `test_glm_binomial_report_has_family`
4. `test_glm_binomial_report_test_type_wald_chi2`
5. `test_glm_binomial_report_df2_is_none`
6. `test_glm_binomial_no_nan_in_json`

**Poisson:**

7. `test_glm_poisson_returns_200`
8. `test_glm_poisson_report_has_baseline`

**Validation:**

9. `test_glm_baseline_out_of_range_returns_422`
10. `test_glm_wrong_family_returns_422`
11. `test_glm_missing_baseline_returns_422`
12. `test_glm_L_delta_mismatch_returns_422`

**Round-trip:**

13. `test_glm_result_json_parseable`
14. `test_glm_design_has_factor_columns`
15. `test_glm_report_achieved_power_between_0_and_1`

**Backward compat:**

16. `test_ols_contrast_endpoint_unchanged`
17. `test_ols_r2_endpoint_unchanged`
18. `test_glm_type_discriminator_routes_correctly`

**Both asyncio and trio backends:**

19. `test_glm_asyncio_backend[asyncio]`
20. `test_glm_asyncio_backend[trio]`

---

## GL-8 — Streamlit UI for GLM

**Status:** ✅ Completed
**Files:** `app/state.py`, `app/pages/2_Power_Config.py`, `app/pages/3_Run_Results.py`
**Estimated new tests:** manual / smoke
**Depends on:** GL-3

### Goal

Extend the Streamlit front-end to expose GLM options when the user selects
"GLM (logistic/Poisson)" as the power mode. No automated tests — validate via manual
smoke testing of the UI.

### `app/state.py` additions

```python
# GLM options
"glm_family": "binomial",    # "binomial" | "poisson"
"glm_link": "",              # "" = canonical
"glm_baseline": 0.20,        # event probability or baseline rate
```

### `app/pages/2_Power_Config.py` changes

**Power mode radio** adds a third option: `"GLM (logistic/Poisson)"`.

When selected, show:
- **Family** select box: `Binomial (logistic)` / `Poisson (log)`
- **Baseline** number input with label that changes by family:
  - Binomial: "Baseline event probability (0–1)" with range [0.001, 0.999]
  - Poisson: "Baseline event rate (> 0)" with min=0.001
- **Effect size help text** updates dynamically: "δ is on the log-odds scale for logistic
  (e.g. 0.5 ≈ 0.5 nat increase in log-odds, corresponding to an odds ratio of 1.65)"
- **σ input** is hidden when GLM family is selected (not applicable)

The L matrix and δ vector inputs remain in contrast mode; for GLM, δ is interpreted on the
LP scale.

### `app/pages/3_Run_Results.py` changes

Add `_is_glm` detection (`isinstance(power_cfg, PowerGLMContrastConfig)` after building it).
When GLM:
- Report table shows `family`, `baseline`, `test_type = "Wald χ²"` instead of σ
- Power curve (E5) shows an info message: "Power curve by n is available for GLM — see
  Analysis tab."
- Summary metric labels update (e.g., no "σ" row in the summary table)

### Sidebar update (`app/state.py` sidebar renderer)

When GLM is active, sidebar shows `family`, `baseline` instead of σ:

```
- Mode: GLM (binomial)
- baseline=0.20 · α=0.05 · power=0.80
```

### Manual smoke test checklist

- [ ] Select GLM → binomial family shows
- [ ] Baseline slider respects (0, 1) bounds
- [ ] δ input label says "log-odds"
- [ ] Run produces result with `family: binomial`
- [ ] Result page shows family + baseline in summary
- [ ] σ input hidden when GLM selected
- [ ] Reset All clears GLM state

---

## GL-9 — Sheets and Excel GLM Connector Support

**Status:** ✅ Completed
**Files:** `iopt_power_design/sheets.py`, `iopt_power_design/excel_template.py`
**Estimated new tests:** 20
**Depends on:** GL-3

### Goal

Extend the `[SETTINGS]` section of both Google Sheets and Excel connectors to accept the
new GLM fields (`family`, `link`, `baseline`). When `family` is set, construct a
`PowerGLMContrastConfig` instead of `PowerContrastConfig`.

### Config schema additions (`[SETTINGS]`)

```
family     | binomial    # "binomial" | "poisson" | "" (= linear model)
link       |             # "logit" | "log" | "" (= canonical)
baseline   | 0.20        # event probability (binomial) or rate (Poisson)
```

When `family` is non-empty, the parser must:
- Ignore `sigma` (or warn and skip)
- Require `baseline`
- Build `PowerGLMContrastConfig` (if `power_mode == "contrast"`) or raise `SheetsError` /
  `ExcelError` explaining that GLM + R² mode is not supported yet

### Template updates

Add a `"glm-binomial"` and `"glm-poisson"` template to `_TEMPLATE_ROWS` (Sheets) and
`create_excel_template` (Excel). Template includes `family`, `baseline`, and a contrast
with `delta` comment explaining LP-scale units.

### Per-response GLM in `[RESPONSES]` section

Extend the per-response row schema to support GLM configs by re-using the existing
`lambda_mode` / `max_n` / `max_iter` / `tol_power` columns (from CR-34) and adding:

- Col 14: `family` (`"binomial"` / `"poisson"` / `""`)
- Col 15: `baseline` (float or blank)

When `family` is non-blank for a response row, build `PowerGLMContrastConfig` for that
response instead of `PowerContrastConfig`.

### Tests

**Sheets (`tests/test_sheets.py` — new class `TestCR34ExtGLM`):**

1. `test_glm_binomial_settings_parses`
2. `test_glm_poisson_settings_parses`
3. `test_glm_family_builds_glm_config`
4. `test_glm_missing_baseline_raises`
5. `test_glm_with_invalid_mode_raises`
6. `test_glm_sigma_ignored_gracefully`
7. `test_glm_template_binomial_parseable`
8. `test_glm_template_poisson_parseable`
9. `test_responses_glm_per_response_family`
10. `test_responses_glm_baseline_forwarded`

**Excel (`tests/test_excel_template.py` — new class `TestExcelGLMSupport`):**

11. `test_excel_glm_settings_parse`
12. `test_excel_glm_family_builds_glm_config`
13. `test_excel_glm_missing_baseline_raises`
14. `test_excel_glm_template_binomial_creates`
15. `test_excel_glm_template_round_trips`
16. `test_excel_responses_glm_family_per_row`
17. `test_excel_responses_glm_baseline_forwarded`
18. `test_linear_sheets_unchanged` — regression guard
19. `test_linear_excel_unchanged` — regression guard
20. `test_glm_template_has_baseline_row`

---

## GL-10 — Weighted Fedorov Exchange (Phase 2 — Deferred)

**Status:** ⬜ Deferred
**Files:** `iopt_power_design/iopt_search.py`, `iopt_power_design/candidate.py`
**Estimated new tests:** 25
**Depends on:** GL-2

### Context

Phase 1 uses a null-based locally optimal design — the design search is identical to OLS
because the Fisher weight `w` is a scalar that cancels out of the I/D/A criteria. Phase 2
implements *alternative-based* locally optimal designs where the Fisher weights vary by
candidate row: `wᵢ = μᵢ(1 − μᵢ)` (binomial) or `μᵢ` (Poisson), evaluated at the
**alternative** β.

### Why it matters

For large effect sizes (e.g., δ = 1.0 log-odds), predicted probabilities range significantly
across the design space. The optimal design under the alternative β differs from the OLS
design. Phase 2 produces designs with better power for these cases.

### Algorithmic approach

**Approximate locally optimal design at alternative β:**

1. User provides `PowerGLMContrastConfig` with `L`, `delta`, `baseline`.
2. Back-compute β_alt from `baseline` (intercept) and `delta` (slope for contrast direction).
3. For each candidate row xᵢ: compute `η_i = xᵢᵀ β_alt`, then `wᵢ = μᵢ(1 − μᵢ)` or `μᵢ`.
4. Run Fedorov exchange with **weighted information matrix** `M = X′WX` where `W = diag(w)`.
5. Apply Sherman-Morrison updates using `w`-scaled inner products.

### New argument in `_fedorov_exchange_single`

Add `row_weights: Optional[np.ndarray] = None` parameter. When provided:
- `lev_t = x_t' (X'WX)^{-1} x_t * w_t` (leverage under W-metric)
- Swap gain uses `wᵢ`-scaled updates

### New criterion scorers

Add `_i_criterion_weighted`, `_d_criterion_weighted`, `_a_criterion_weighted` that mirror
existing OLS scorers but use `M = X'WX`.

### Tests (`tests/test_iopt_search.py` — new class `TestWeightedFedorov`)

1. `test_uniform_weights_equals_unweighted`
2. `test_zero_weight_row_not_selected`
3. `test_high_weight_regions_preferred`
4. `test_d_criterion_weighted_decreases_per_swap`
5. `test_i_criterion_weighted_valid_score`
6. `test_weighted_d_vs_unweighted_d_differ_for_nonuniform_weights`
7. ... (25 total covering correctness, edge cases, and criterion variants)

---

## GL-11 — Alternative-Based GLM Design API (Phase 2 — Deferred)

**Status:** ⬜ Deferred
**Files:** `iopt_power_design/api.py`, `iopt_power_design/analysis.py`
**Estimated new tests:** 20
**Depends on:** GL-10

### Context

Expose the Phase 2 weighted Fedorov exchange through the main API. Activated by a new
`DesignOptions` field `glm_design_strategy: Literal["null_based", "alt_based"] = "null_based"`.

### New `DesignOptions` field

```python
glm_design_strategy: Literal["null_based", "alt_based"] = "null_based"
```

- `"null_based"`: Phase 1 (no weights, identical to OLS design structure).
- `"alt_based"`: Phase 2 (weighted Fedorov at alternative β, better for large effects).

### Changes to `find_optimal_design`

When `isinstance(power_cfg, PowerGLMContrastConfig)` and
`design_opts.glm_design_strategy == "alt_based"`:
1. Compute β_alt from `power_cfg.baseline` and `power_cfg.delta`
2. Evaluate predicted values at each candidate row
3. Compute per-row weights
4. Run weighted Fedorov (GL-10)

### Power curve for alternative-based designs

Add `power_curve_by_n_glm_alt_based` to `analysis.py` that shows the performance gain from
using alternative-based vs null-based designs as n grows.

---

## Architecture diagram

```
config.py                    power.py                 iopt_search.py
---------                    --------                 --------------
PowerGLMContrastConfig  -->  glm_contrast_power  -->  Fedorov exchange
  .family                      (Wald χ²)               (unchanged Ph1;
  .link                        ncx2 distribution        X'WX in Ph2)
  .baseline                    noncentrality λ
  .L, .delta                   = w · δ'(L(X'X)⁻¹L')⁻¹δ
  .alpha, .power
           |
           v
       glm_fisher_weight(cfg)
           w = p₀(1−p₀) [binomial]
           w = μ₀        [Poisson]
```

---

## Open questions and decisions

| # | Question | Recommended decision |
|---|---|---|
| 1 | Should GLM + R² mode be supported? | **No in v1.** Pseudo-R² (McFadden, Nagelkerke) exists but has different properties. Implement contrast-only for v1; add pseudo-R² as separate ticket if demanded. |
| 2 | What happens with split-plot + GLM? | **Not supported in v1.** The split-plot GLS framework and GLM weighted Fisher info are orthogonal complexities. Note in docs; leave for future work. |
| 3 | How to handle `sigma` field in existing YAML configs when switching to GLM? | Silently ignore with a `warnings.warn()`. Do not raise an error — user may be adapting an existing config. |
| 4 | Should the LP-scale `delta` be automatically rescaled from odds ratios? | **Add a convenience constructor or docstring note** showing how to convert: `delta = log(odds_ratio)`. Do not auto-detect or convert silently. |
| 5 | Should we support `probit` link for binomial? | **No in v1.** Logit is canonical and most common. Probit requires `scipy.stats.norm.ppf` and different weight formula. Add as future ticket. |
| 6 | Can `PowerGLMContrastConfig` be used in `MultiResponseOptions`? | **Yes from day 1.** `ResponseSpec` is duck-typed; `eval_response_power` dispatches by isinstance. Multi-response GLM is automatically supported. |
| 7 | Should n-search warn if baseline is extreme (< 0.05 or > 0.95)? | **Yes** — emit `RuntimeWarning("GLM baseline near boundary; required n may be very large.")` when `min(p₀, 1−p₀) < 0.05`. |

---

## Estimated test count summary

| Ticket | New tests |
|---|---|
| GL-1 | 25 |
| GL-2 | 30 |
| GL-3 | 20 |
| GL-4 | 20 |
| GL-5 | 15 |
| GL-6 | 15 |
| GL-7 | 20 |
| GL-8 | manual |
| GL-9 | 20 |
| GL-10 | 25 |
| GL-11 | 20 |
| **Total (Phase 1)** | **~165** |
| **Total (Phases 1+2)** | **~210** |

---

## Suggested implementation order

```
GL-1 → GL-2 → GL-3 → GL-4 → GL-5    # Core library (no UI/connectors)
                 └→ GL-6              # CLI (can parallelize with GL-4)
                 └→ GL-7              # REST API (can parallelize with GL-4)
                 └→ GL-8              # Streamlit (after GL-3)
                 └→ GL-9              # Sheets/Excel (after GL-3)

[deferred]
GL-3 → GL-10 → GL-11
```

GL-1 through GL-5 form the minimum shippable slice — a user can call
`find_optimal_design(formula, factors, PowerGLMContrastConfig(...))` from Python with
correct power calculations and n-search. Tickets GL-6 through GL-9 bring connectors and
UIs up to feature parity.
