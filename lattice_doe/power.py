# power.py
# License: MIT
"""
Power calculations for linear models (F-tests and GLM Wald chi-square)
=======================================================================

This module provides power calculations for:
  - Linear contrasts (Wald F-test on Lβ = δ)
  - Global R² (full model F-test)
  - GLM contrasts (Wald chi-square test, null-based locally optimal)

Core steps:
-----------
1. Compute noncentrality parameter λ based on the model matrix X
   and effect size specification (contrast, R², or GLM contrast).
2. Compute power using:
   - Noncentral F distribution for OLS/GLS tests:
       power = 1 - F_{df1, df2, λ}(Fcrit)
   - Noncentral chi-square for GLM Wald tests:
       power = 1 - χ²_{q, λ}(χ²crit)
     where the noncentrality λ = w · δᵀ [L(X'X)⁻¹Lᵀ]⁺ δ
     and w = glm_fisher_weight(cfg) (scalar Fisher information weight).

Notes:
------
• Shapes are validated and broadcast where sensible.
• We use numerically-stable pseudo-inverses with small Tikhonov jitter for X'X.
• We avoid importing heavy dependencies outside SciPy / NumPy.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, List, Literal, NamedTuple, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from .config import ResponseSpec, SplitPlotOptions

try:
    # Only import when actually computing power
    from scipy.stats import ncf as ncf_dist
    from scipy.stats import f as f_dist
    from scipy.stats import ncx2 as ncx2_dist
    from scipy.stats import chi2 as chi2_dist
except Exception as e:  # pragma: no cover
    # Delay the error until one of the functions is actually used.
    ncf_dist = None
    f_dist = None
    ncx2_dist = None
    chi2_dist = None
    _scipy_import_error = e
else:
    _scipy_import_error = None


# --- Custom Types for readable return values ---
ContrastPowerResult = NamedTuple("ContrastPowerResult", [("power", float), ("lam", float)])
GlobalPowerResult = NamedTuple("GlobalPowerResult", [("power", float), ("lam", float)])
HotellingT2Result = NamedTuple(
    "HotellingT2Result",
    [("power", float), ("lam", float), ("df1", int), ("df2", int)],
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _require_scipy() -> None:
    if _scipy_import_error is not None:
        raise RuntimeError(
            "scipy is required for power calculations but could not be imported."
        ) from _scipy_import_error


def _pinv_xtx(X: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """
    Compute pinv(X'X + ridge) with a small scale-relative ridge for stability.

    The ridge added to each diagonal entry is ``jitter * (X'X)_ii`` (with 1.0
    substituted for all-zero columns), i.e. the regularization is applied in
    a column-equilibrated space. An absolute ridge ``jitter * I`` would be
    huge relative to columns expressed in small physical units (e.g. mole
    fractions ~1e-5), shrinking the contrast variance and silently inflating
    the noncentrality parameter — power estimates became anti-conservative
    for small-scale factors (SR-8). The relative ridge makes power results
    invariant to the units of the factor columns.

    Parameters
    ----------
    X : ndarray (n x p)
        Model/design matrix.
    jitter : float
        Relative ridge magnitude; each diagonal entry is inflated by the
        factor ``(1 + jitter)``.

    Returns
    -------
    ndarray (p x p)
        Moore–Penrose inverse of the (regularized) X'X.
    """
    XtX = X.T @ X
    diag = np.diag(XtX)
    ridge = np.where(diag > 0, diag, 1.0)
    XtX_reg = XtX + float(jitter) * np.diag(ridge)
    # Invert in a column-equilibrated space: with badly scaled columns the
    # direct (pseudo-)inverse loses precision (the condition number grows
    # with the squared column-scale ratio); scaling to unit diagonal keeps
    # the inversion accurate, and the ridge makes XtX_reg full rank so the
    # scaled-back result equals pinv(XtX_reg) exactly in real arithmetic.
    d = np.sqrt(np.diag(XtX_reg))
    d = np.where(d > 0, d, 1.0)
    S = 1.0 / d
    A = XtX_reg * S[:, None] * S[None, :]
    return np.linalg.pinv(A) * S[:, None] * S[None, :]


def _symmetrize(A: np.ndarray) -> np.ndarray:
    """Force numerical symmetry."""
    return 0.5 * (A + A.T)


def _check_delta_consistency(
    V: np.ndarray,
    V_pinv: np.ndarray,
    delta: np.ndarray,
) -> None:
    """Raise if the hypothesis L·β = δ is infeasible for every β (SR-9).

    When rows of L are linearly dependent, V = L M⁻¹ Lᵀ is rank-deficient and
    the components of L·β satisfy the same linear relations as the rows of L.
    A δ violating those relations specifies a hypothesis no β can satisfy;
    the pseudo-inverse would silently project δ onto range(V), changing the
    tested hypothesis without warning (e.g. duplicated contrast rows with a
    sign-flipped δ yield λ = 0 and power = α). Feasibility is equivalent to
    δ ∈ range(V), checked as ‖δ − V V⁺ δ‖ ≈ 0. For full-rank V the projector
    is the identity and the check always passes.

    Accepts a vector δ (shape (q,)) or a matrix Δ (shape (q, k), checked
    column-wise).
    """
    resid = delta - V @ (V_pinv @ delta)
    resid_norm = float(np.linalg.norm(resid))
    if resid_norm > 1e-8 * max(1.0, float(np.linalg.norm(delta))):
        raise ValueError(
            "delta is inconsistent with the linear dependencies among the "
            "rows of L: the hypothesis L·beta = delta cannot hold for any "
            f"beta (||delta − proj(delta)|| = {resid_norm:.3g}). Remove "
            "redundant contrast rows, or make the delta entries of dependent "
            "rows satisfy the same linear relation as the rows themselves."
        )


def _r2_df_num(X: np.ndarray) -> int:
    """Numerator df for the global R² F-test (slopes only, intercept excluded).

    Matches the G*Power / ``pwr.f2.test`` convention: if the model contains
    an intercept, df_num = rank(X) - 1; otherwise df_num = rank(X).  This is
    the single authoritative source of truth shared by ``global_r2_power``
    and any caller that needs the same value for reporting.

    An intercept is detected as the constant vector lying in the column span
    of X — ``rank([X, 1]) == rank(X)`` — not as a literal all-ones column.
    This correctly handles implicit intercepts such as cell-means coding
    (``0 + C(group)``: the dummies sum to 1, so "all group means equal" has
    k − 1 numerator df) and constant columns with values other than 1 (SR-10).
    """
    rank_X = int(np.linalg.matrix_rank(X))
    if X.shape[1] == 0 or rank_X == 0:
        return rank_X
    ones = np.ones((X.shape[0], 1), dtype=float)
    rank_aug = int(np.linalg.matrix_rank(np.hstack([np.asarray(X, dtype=float), ones])))
    has_intercept = rank_aug == rank_X
    return (rank_X - 1) if (has_intercept and rank_X > 1) else rank_X


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def contrast_power(
    L: np.ndarray,
    delta: np.ndarray,
    X: np.ndarray,
    sigma: float = 1.0,
    alpha: float = 0.05,
    jitter: float = 1e-8,
) -> ContrastPowerResult:
    """
    Power for a linear contrast (Wald/F-test): H0: Lβ = 0 vs H1: Lβ = δ.

    Noncentrality parameter:
        λ = (δᵀ [L (X'X)^-1 Lᵀ]⁺ δ) / σ²
    with df1 = rank(L), df2 = n − p.

    Parameters
    ----------
    L : ndarray (q x p) or (p,)
        Contrast rows. If (p,), it is treated as a single-row matrix.
    delta : ndarray (q,) or (q x 1)
        Effect size under H1 for each contrast row.
    X : ndarray (n x p)
        Model/design matrix corresponding to the fitted model.
    sigma : float
        Residual standard deviation (σ). Must be positive.
    alpha : float
        Significance level.
    jitter : float
        Small ridge added to X'X before pseudo-inversion.

    Returns
    -------
    ContrastPowerResult
        A NamedTuple containing:
        - power (float): 1 - β for the noncentral F. Clipped to [0, 1].
        - lam (float): Noncentrality parameter λ.
    """
    _require_scipy()

    # --- Parameter Validation ---
    if not sigma > 0:
        raise ValueError(f"sigma (residual std dev) must be positive; got {sigma}.")

    X = np.asarray(X)
    L = np.asarray(L)
    delta = np.asarray(delta).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"Design matrix X must be 2D; got {X.ndim} dimensions.")
    n, p = X.shape

    # Normalize L shape to (q x p)
    if L.ndim == 1:
        L = L.reshape(1, -1)
    if L.ndim != 2 or L.shape[1] != p:
        raise ValueError(
            f"Contrast matrix L has incompatible shape. "
            f"Expected (q, p={p}), got {L.shape}."
        )

    q = L.shape[0]
    if delta.shape[0] != q:
        raise ValueError(
            f"Effect size delta has incompatible length. "
            f"Expected length q={q}, got {delta.shape[0]}."
        )

    # Degrees of freedom
    df_num = int(np.linalg.matrix_rank(L))
    df_denom = int(n - np.linalg.matrix_rank(X))
    if df_num <= 0:
        raise ValueError(
            f"Contrast matrix L has rank {df_num}. "
            "Must have rank > 0 to be testable."
        )
    if df_denom <= 0:
        raise ValueError(
            f"Denominator degrees of freedom must be positive. "
            f"Got {df_denom} (n={n}, rank(X)={n - df_denom})."
        )

    # --- Calculation ---

    # Inverses
    XtX_inv = _pinv_xtx(X, jitter=jitter)

    # Var(L beta_hat) = sigma^2 * L (X'X)^-1 L'
    V_unscaled = _symmetrize(L @ XtX_inv @ L.T)

    # Rank 0 of the contrast variance matrix means the contrast lies in the
    # null space of X — there is no variance to test against.
    rank_V = np.linalg.matrix_rank(V_unscaled)
    if rank_V == 0:
        raise ValueError(
            "Contrast variance matrix (L @ (X'X_inv) @ L.T) has rank 0. "
            "The contrast is not testable (e.g., it is in the null space of X)."
        )
    # If rank_V < df_num the contrast rows are linearly dependent; pinv
    # handles the inversion, but delta must satisfy the same dependencies
    # for the hypothesis to be well-posed — checked below (SR-9).

    V = V_unscaled * (sigma ** 2)

    # Use pseudo-inverse in case V is singular
    V_inv = np.linalg.pinv(V)
    _check_delta_consistency(V, V_inv, delta)
    lam = float(delta.T @ V_inv @ delta)

    # A quadratic form should be >= 0; clip small negative values from float error,
    # but treat anything below tolerance as a numerical bug.
    if lam < -1e-8:  # Allow for small float tolerance
        raise ValueError(
            f"Computed noncentrality parameter lambda is negative ({lam}), "
            "indicating a numerical issue."
        )
    lam = max(0.0, lam)  # Clip small negative values to 0

    # Critical F and power
    Fcrit = f_dist.isf(alpha, df_num, df_denom)
    power = float(1.0 - ncf_dist.cdf(Fcrit, df_num, df_denom, lam))

    power = np.clip(power, 0.0, 1.0)

    return ContrastPowerResult(power=power, lam=lam)


# ---------------------------------------------------------------------
def glm_contrast_power(
    cfg: "PowerGLMContrastConfig",
    X: np.ndarray,
    jitter: float = 1e-8,
) -> ContrastPowerResult:
    """Power for a GLM linear contrast via Wald chi-square test.

    Uses a null-based locally optimal approximation: the Fisher information
    weight ``w = glm_fisher_weight(cfg)`` is a positive scalar evaluated at
    the null (baseline), so noncentrality is:

        λ = w · δᵀ [L (X'X)⁻¹ Lᵀ]⁺ δ

    Power is then computed from the noncentral chi-square distribution:

        χ²_crit = χ²_{1−α}(q)
        power   = 1 − χ²_{q, λ}(χ²_crit)

    where ``q = rank(L)``.  This is equivalent to OLS contrast power with
    effective residual std ``σ_eff = 1/√w``.

    .. note:: **Approximation scope.**
        ``w`` is a single scalar derived from ``cfg.baseline`` and applied
        uniformly across all rows of ``X``.  In the general GLM design
        setting the weight varies per design point (it depends on the
        linear predictor at each point under the true parameter vector).
        The approximation is accurate when the true means stay close to
        the baseline and degrades as slope magnitudes and factor ranges
        grow.  The direction of the error depends on where the baseline
        sits: for a logistic model with baseline probability near 0.5 the
        null weight w = p(1−p) is at its maximum, so real effects (which
        push probabilities toward the extremes) make this estimate
        **optimistic**; for baselines near 0 or 1 with effects toward 0.5
        it is conservative.  Use results from this function as a planning
        approximation and validate via simulation for studies with large
        effects or wide covariate ranges (per-point weights under H1 are
        tracked as GL-10/GL-11).

    Parameters
    ----------
    cfg : PowerGLMContrastConfig
        GLM contrast specification (family, link, baseline, L, delta, alpha).
    X : ndarray (n, p)
        Model/design matrix corresponding to the fitted model.
    jitter : float
        Small ridge added to X'X before pseudo-inversion.

    Returns
    -------
    ContrastPowerResult
        NamedTuple with ``power`` (float) and ``lam`` (float).
    """
    _require_scipy()
    from .config import glm_fisher_weight  # avoid circular at module level

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Design matrix X must be 2D; got {X.ndim} dimensions.")
    n, p = X.shape

    L = np.asarray(cfg.L)
    if L.ndim == 1:
        L = L.reshape(1, -1)
    delta = np.asarray(cfg.delta).reshape(-1)

    q = int(np.linalg.matrix_rank(L))
    if q <= 0:
        raise ValueError("Contrast matrix L has rank 0; must have rank > 0.")

    # Fisher information weight at null (scalar)
    w = glm_fisher_weight(cfg)

    # Weighted XtX inverse (w cancels in ratio, kept for clarity)
    XtX_inv = _pinv_xtx(X, jitter=jitter)
    V_unscaled = _symmetrize(L @ XtX_inv @ L.T)

    rank_V = np.linalg.matrix_rank(V_unscaled)
    if rank_V == 0:
        raise ValueError(
            "Contrast variance matrix L @ (X'X)⁻¹ @ Lᵀ has rank 0; "
            "the contrast is not testable."
        )

    V_inv = np.linalg.pinv(V_unscaled)
    _check_delta_consistency(V_unscaled, V_inv, delta)
    lam = float(w * (delta @ V_inv @ delta))
    lam = max(0.0, lam)

    # Critical chi-square value and power
    chi2_crit = float(chi2_dist.isf(cfg.alpha, q))
    power = float(1.0 - ncx2_dist.cdf(chi2_crit, q, lam))
    power = float(np.clip(power, 0.0, 1.0))

    return ContrastPowerResult(power=power, lam=lam)


# ---------------------------------------------------------------------
def global_r2_power(
    r2_target: float,
    X: np.ndarray,
    alpha: float,
    lambda_mode: Literal["n", "n_minus_p"] = "n",
    df_num: Optional[int] = None,
) -> GlobalPowerResult:
    """
    Power for the global R² test (full model F-test): H0: R² = 0 vs H1: R² = r2_target.

    We parameterize effect size via Cohen's f² = R² / (1 - R²).

    Noncentrality parameter (approximate):
        f² = R² / (1 - R²)
        λ ≈ f² * n       (if lambda_mode="n")
        λ ≈ f² * (n - p) (if lambda_mode="n_minus_p")
    with df1 = rank(X), df2 = n − rank(X).

    Parameters
    ----------
    r2_target : float
        Target population R² (0 < r2_target < 1).
    X : ndarray (n x p)
        Model/design matrix.
    alpha : float
        Significance level.
    lambda_mode : {"n", "n_minus_p"}
        How to compute λ from f²: either f² * n (default) or f² * (n - p).
    df_num : int, optional
        Override for the numerator df. When None (default) it is derived
        from X via ``_r2_df_num`` (slopes only, intercept excluded). Pass
        the tested-predictor count explicitly when X contains adjustment
        columns that are not under test — e.g. blocked designs pass the
        treatment-slope count so block dummies reduce error df but are not
        counted as tested predictors (SR-17).

    Returns
    -------
    GlobalPowerResult
        A NamedTuple containing:
        - power (float): Power against the specified R². Clipped to [0, 1].
        - lam (float): Noncentrality parameter λ used.
    """
    _require_scipy()

    # --- Parameter Validation ---
    if not (0.0 < r2_target < 1.0):
        # IMPROVED: More specific error message
        raise ValueError(f"r2_target must be in the range (0, 1); got {r2_target}.")

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Design matrix X must be 2D; got {X.ndim} dimensions.")
        
    n, p = X.shape
    rank_X = int(np.linalg.matrix_rank(X))
    df_denom = int(n - rank_X)

    # Standard global F-test: df_num = slopes only (intercept excluded).
    # Delegates to _r2_df_num so the convention is defined in one place.
    # Callers may override df_num when only a subset of the columns in X is
    # under test — e.g. blocked designs pass the treatment-slope count so
    # block-dummy adjustment columns are charged to error df but not counted
    # as tested predictors (SR-17).
    df_num = _r2_df_num(X) if df_num is None else int(df_num)

    if df_num <= 0:
        raise ValueError(
            f"Numerator degrees of freedom must be positive; got {df_num} "
            f"(rank(X)={rank_X}). Check that the model matrix is not rank-deficient."
        )
    if df_denom <= 0:
        raise ValueError(
            f"Denominator degrees of freedom (n - rank(X)) must be positive; "
            f"Got {df_denom} (n={n}, rank(X)={rank_X})."
        )

    # --- Calculation ---
    
    # Cohen's f²
    f2 = r2_target / (1.0 - r2_target)

    # Compute λ
    if lambda_mode == "n":
        lam = float(f2 * n)
    else:
        lam = float(f2 * df_denom)  # f2 * (n - p); df_denom = n - rank(X)

    # Always non-negative if r2_target > 0; defensive guard against numerical issues.
    if lam < 0.0:
        raise ValueError(
            f"Computed noncentrality parameter lambda is negative ({lam}), "
            "indicating a numerical issue."
        )

    Fcrit = f_dist.isf(alpha, df_num, df_denom)
    power = float(1.0 - ncf_dist.cdf(Fcrit, df_num, df_denom, lam))

    power = np.clip(power, 0.0, 1.0)

    return GlobalPowerResult(power=power, lam=lam)



# ---------------------------------------------------------------------
# Split-plot (GLS) power functions
# ---------------------------------------------------------------------
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
    htc_factor_cols: Optional[List[int]] = None,
    jitter: float = 1e-8,
) -> ContrastPowerResult:
    """Power for a linear contrast in a split-plot design.

    Uses GLS information matrix M = X'V⁻¹X where V = σ²_sp (η ZZ' + I).

    **Joint test (default).**  When every contrast row is assigned the same
    denominator df — always true for single-row L, for multi-row L whose rows
    all belong to one error stratum, and for ``df_method="conservative"`` or
    ``"sp_only"`` — the power of the joint Wald F-test is computed, mirroring
    the OLS ``contrast_power``:

        λ = δᵀ [L M⁻¹ Lᵀ]⁺ δ / σ²_sp,   df1 = rank(L)

    **Stratum-spanning fallback.**  When rows of L span both strata under
    ``df_method="auto"``, no single denominator df exists for a joint F-test.
    In that case each row is tested individually (1 numerator df, its own
    stratum's denominator df) and the *minimum* power across rows is
    returned.  This is a conservative worst-case-per-row bound, **not** the
    omnibus power; supply ``df_method="conservative"`` to force a joint test
    with whole-plot df instead.

    Denominator df is assigned per contrast row via df_method:
    - "auto"         : WP df for pure-WP contrasts, SP df for others.
    - "conservative" : always WP df.
    - "sp_only"      : always SP df.

    At eta = 0 the result is identical to ``contrast_power``.  As eta → 0⁺
    the joint-test λ converges to the OLS λ (denominator df may differ
    because it comes from the stratum heuristic below).

    .. note:: **Denominator df approximation.**
        df assignment uses a stratum-classification heuristic rather than
        a full small-sample mixed-model method (Satterthwaite or
        Kenward-Roger).  For balanced designs with a single variance
        component (η) this gives exact denominator df.  For unbalanced
        designs, near-singular settings, or more complex variance
        structures the heuristic can produce conservative or
        anti-conservative power estimates.  Use ``df_method="conservative"``
        when in doubt; Satterthwaite/KR support is a planned future
        enhancement.

    Parameters
    ----------
    L : ndarray (q, p) or (p,)
        Contrast matrix.
    delta : ndarray (q,)
        Effect sizes under H1, one per contrast row.
    X : ndarray (n, p)
        Model / design matrix.
    Z : ndarray (n, n_wp)
        Whole-plot indicator matrix (from ``build_whole_plot_indicator``).
    sigma_sp : float
        Sub-plot residual standard deviation (σ_sp > 0).
    eta : float
        Variance ratio σ²_wp / σ²_sp (≥ 0).
    alpha : float
        Significance level.
    df_method : {"auto", "conservative", "sp_only"}
        How to assign denominator df per contrast row.
    htc_factor_cols : list of int or None
        Column indices in X corresponding to HTC (whole-plot) factors.
        Required for df_method="auto" and "conservative" to classify
        contrasts as WP vs SP.  When None all contrasts are treated as SP.
    jitter : float
        Small ridge added to M for numerical stability.

    Returns
    -------
    ContrastPowerResult
        power : joint Wald F-test power (or the min per-row power in the
        stratum-spanning fallback), lam : the corresponding λ.
    """
    _require_scipy()

    if not sigma_sp > 0:
        raise ValueError(f"sigma_sp must be positive; got {sigma_sp}.")
    if eta < 0:
        raise ValueError(f"eta must be ≥ 0; got {eta}.")

    X = np.asarray(X, dtype=float)
    L = np.asarray(L, dtype=float)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.ndim}D.")

    # eta=0 shortcut: exact OLS equivalence
    if eta == 0.0:
        return contrast_power(L, delta, X, sigma_sp, alpha, jitter=jitter)

    n, p = X.shape
    if L.ndim == 1:
        L = L.reshape(1, -1)
    if L.ndim != 2 or L.shape[1] != p:
        raise ValueError(
            f"L has incompatible shape; expected (q, p={p}), got {L.shape}."
        )
    q = L.shape[0]
    if delta.shape[0] != q:
        raise ValueError(
            f"delta length {delta.shape[0]} does not match L rows {q}."
        )

    from .split_plot import (
        build_split_plot_covariance_inv,
        gls_information_matrix,
        classify_contrasts,
        split_plot_df_denom,
    )

    V_inv = build_split_plot_covariance_inv(Z, eta)
    M = gls_information_matrix(X, V_inv, jitter=jitter)
    M_inv = np.linalg.inv(M)

    htc_cols: List[int] = list(htc_factor_cols) if htc_factor_cols is not None else []
    is_wp = classify_contrasts(L, htc_cols, p)
    df_denoms = split_plot_df_denom(X, Z, is_wp, df_method, htc_cols or None)

    # Joint Wald F-test whenever a single denominator df applies to every row.
    if np.all(df_denoms == df_denoms[0]):
        V_c = _symmetrize(L @ M_inv @ L.T)
        if np.linalg.matrix_rank(V_c) == 0:
            return ContrastPowerResult(power=0.0, lam=0.0)
        df_num = int(np.linalg.matrix_rank(L))
        V_c_pinv = np.linalg.pinv(V_c)
        _check_delta_consistency(V_c, V_c_pinv, delta)
        lam = max(0.0, float(delta @ V_c_pinv @ delta) / (sigma_sp ** 2))
        df_d = int(df_denoms[0])
        if df_d <= 0:
            return ContrastPowerResult(power=0.0, lam=lam)
        Fcrit = f_dist.isf(alpha, df_num, df_d)
        power = float(np.clip(1.0 - ncf_dist.cdf(Fcrit, df_num, df_d, lam), 0.0, 1.0))
        return ContrastPowerResult(power=power, lam=lam)

    # Rows span both strata under df_method="auto": no single denominator df
    # exists for a joint F-test, so bound the omnibus power from below by the
    # worst per-row 1-df test, each with its own stratum's df.
    powers: List[float] = []
    lams: List[float] = []
    for i in range(q):
        l_i = L[i : i + 1, :]  # (1, p)
        d_i = float(delta[i])
        v_i = float((l_i @ M_inv @ l_i.T).item())
        if v_i <= 0.0:
            powers.append(0.0)
            lams.append(0.0)
            continue
        lam_i = max(0.0, d_i ** 2 / (sigma_sp ** 2 * v_i))
        df_d_i = int(df_denoms[i])
        if df_d_i <= 0:
            powers.append(0.0)
            lams.append(float(lam_i))
            continue
        Fcrit = f_dist.isf(alpha, 1, df_d_i)
        power_i = float(np.clip(1.0 - ncf_dist.cdf(Fcrit, 1, df_d_i, lam_i), 0.0, 1.0))
        powers.append(power_i)
        lams.append(float(lam_i))

    min_idx = int(np.argmin(powers))
    return ContrastPowerResult(power=powers[min_idx], lam=lams[min_idx])


# ---------------------------------------------------------------------
def global_r2_power_sp(
    r2_target: float,
    X: np.ndarray,
    Z: np.ndarray,
    sigma_sp: float,
    eta: float,
    alpha: float,
    *,
    df_method: str = "auto",
    lambda_mode: Literal["n", "n_minus_p"] = "n",
    jitter: float = 1e-8,
    htc_factor_cols: Optional[List[int]] = None,
) -> GlobalPowerResult:
    """Power for the global R² F-test in a split-plot design.

    Uses GLS noncentrality:
        f² = r2_target / (1 − r2_target)
        λ = f² · tr(Ṽ⁻¹)           (lambda_mode="n")
        λ = f² · max(1, tr(Ṽ⁻¹) − df_num)   (lambda_mode="n_minus_p")
    where Ṽ⁻¹ = (η ZZ' + I)⁻¹ is the scaled inverse covariance.

    Denominator df is the sub-plot stratum df, consistent with
    ``split_plot_df_denom`` (SR-11):

        df_denom = n − n_wp − (rank(X) − rank(X_wp))

    where rank(X_wp) is computed from *htc_factor_cols* when provided and
    approximated as 1 (intercept only) otherwise — the fallback is
    conservative (fewer df).  A ``ValueError`` is raised when df_denom ≤ 0
    (the sub-plot stratum cannot support the test), instead of silently
    clamping to 1 as earlier versions did.

    At eta = 0 the result is identical to ``global_r2_power``.  Note the
    denominator df steps at η → 0⁺: the OLS shortcut uses the pooled
    df = n − rank(X), while any η > 0 uses the sub-plot stratum df above.
    λ itself is continuous (tr(Ṽ⁻¹) → n); the df step reflects the change
    from a pooled error estimate to a stratum error estimate.

    .. note:: **Approximation scope.**
        A target R² does not specify how the explained variance splits
        between the whole-plot and sub-plot strata, so no single
        noncentrality parameter can be exact.  tr(Ṽ⁻¹) blends the two
        strata — it equals n at η = 0 and approaches n − n_wp as η → ∞ —
        which implicitly assumes the signal is spread proportionally across
        strata.  If the true signal is carried mainly by whole-plot (HTC)
        factors, this estimate is **optimistic** (whole-plot effects are
        really estimated from only n_wp plots); if carried mainly by
        sub-plot factors it is mildly conservative.  A ``UserWarning`` is
        emitted whenever η > 0; validate by simulation when whole-plot
        factors dominate the model.

    Parameters
    ----------
    r2_target : float
        Target population R² (0 < r2_target < 1).
    X : ndarray (n, p)
        Model / design matrix.
    Z : ndarray (n, n_wp)
        Whole-plot indicator matrix.
    sigma_sp : float
        Sub-plot residual standard deviation (not used in λ computation but
        validated for consistency; must be positive).
    eta : float
        Variance ratio σ²_wp / σ²_sp (≥ 0).
    alpha : float
        Significance level.
    df_method : str
        Kept for API symmetry with contrast_power_sp; currently unused
        (the global test always uses the sub-plot stratum df).
    lambda_mode : {"n", "n_minus_p"}
        Whether λ scales by tr(Ṽ⁻¹) or tr(Ṽ⁻¹) − df_num.
    jitter : float
        Ridge for numerical stability.
    htc_factor_cols : list of int or None
        Column indices in X of the HTC (whole-plot) factor terms, used to
        compute rank(X_wp) for the denominator df.  When None, rank(X_wp)
        is approximated as 1 (conservative).

    Returns
    -------
    GlobalPowerResult
        power : power for the global F-test, lam : noncentrality λ.
    """
    _require_scipy()

    if not (0.0 < r2_target < 1.0):
        raise ValueError(f"r2_target must be in (0, 1); got {r2_target}.")
    if not sigma_sp > 0:
        raise ValueError(f"sigma_sp must be positive; got {sigma_sp}.")
    if eta < 0:
        raise ValueError(f"eta must be ≥ 0; got {eta}.")

    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.ndim}D.")

    # eta=0 shortcut: exact OLS equivalence
    if eta == 0.0:
        return global_r2_power(r2_target, X, alpha, lambda_mode=lambda_mode)

    warnings.warn(
        "global_r2_power_sp approximates the split-plot noncentrality with a "
        "blended effective sample size tr((η ZZ' + I)⁻¹), which assumes the R² "
        "signal is spread proportionally across whole-plot and sub-plot strata. "
        "If the signal comes mainly from whole-plot (HTC) factors the power "
        "estimate can be optimistic; validate by simulation.",
        UserWarning,
        stacklevel=2,
    )

    n, p = X.shape
    n_wp = Z.shape[1]

    from .split_plot import build_split_plot_covariance_inv, split_plot_rank_wp

    df_num = _r2_df_num(X)
    if df_num <= 0:
        raise ValueError(
            f"Numerator df must be positive; got {df_num} (rank(X)={np.linalg.matrix_rank(X)})."
        )
    # Sub-plot stratum df, consistent with split_plot_df_denom (SR-11).
    rank_X = int(np.linalg.matrix_rank(X))
    rank_X_wp = split_plot_rank_wp(X, Z, htc_factor_cols)
    df_denom = n - n_wp - (rank_X - rank_X_wp)
    if df_denom <= 0:
        raise ValueError(
            f"Sub-plot denominator df must be positive; got {df_denom} "
            f"(n={n}, n_wp={n_wp}, rank(X)={rank_X}, rank(X_wp)={rank_X_wp}). "
            "The sub-plot stratum cannot support the global R² test — "
            "increase subplots per whole plot or reduce the model size."
        )

    V_inv = build_split_plot_covariance_inv(Z, eta)
    eff_n = float(np.trace(V_inv))

    f2 = r2_target / (1.0 - r2_target)
    if lambda_mode == "n":
        lam = float(f2 * eff_n)
    else:
        lam = float(f2 * max(1.0, eff_n - df_num))

    lam = max(0.0, lam)
    Fcrit = f_dist.isf(alpha, df_num, df_denom)
    power = float(np.clip(1.0 - ncf_dist.cdf(Fcrit, df_num, df_denom, lam), 0.0, 1.0))

    return GlobalPowerResult(power=power, lam=lam)


def combine_powers(
    powers: List[float],
    weights: Optional[List[float]],
    rule: Literal["min", "product", "weighted_mean"],
) -> float:
    """Combine per-response power values into a single scalar.

    Parameters
    ----------
    powers : list of float in [0, 1]
        Per-response power values.
    weights : list of float (> 0) or None
        Relative weights, same length as *powers*.  Ignored for ``"min"`` and
        ``"product"``; required (but defaults to equal weights) for
        ``"weighted_mean"``.  Weights are normalised internally.
    rule : {"min", "product", "weighted_mean"}
        Combination rule.

        * ``"min"`` — combined power = min(p_i).  Conservative; the combined
          power equals the worst individual response.  No statistical
          assumptions about dependence between responses.
        * ``"product"`` — combined power = ∏ p_i.  Interprets combined power
          as the probability that **all** responses simultaneously achieve
          significance.  **This is the exact joint probability only when
          responses are independent.**  When responses are correlated
          (shared experimental units, common error) the true joint
          probability differs; use ``sigma_joint`` (Hotelling T²) for the
          correlated OLS contrast case instead.
        * ``"weighted_mean"`` — combined power = Σ(w_i p_i) / Σw_i.  A
          soft aggregation that allows high-power responses to partially
          compensate for low-power ones.

    Returns
    -------
    float in [0, 1]

    Raises
    ------
    ValueError
        If *powers* is empty or *rule* is not one of the three valid values.
    """
    if len(powers) == 0:
        raise ValueError("combine_powers: powers list must not be empty.")
    if rule == "min":
        return float(min(powers))
    elif rule == "product":
        result = 1.0
        for pv in powers:
            result *= pv
        return float(result)
    elif rule == "weighted_mean":
        w = weights if weights is not None else [1.0] * len(powers)
        total_w = sum(w)
        return float(sum(pv * wv for pv, wv in zip(powers, w)) / total_w)
    raise ValueError(
        f"combine_powers: unknown combination rule {rule!r}. "
        "Must be 'min', 'product', or 'weighted_mean'."
    )


def eval_response_power(
    response: "ResponseSpec",
    X: np.ndarray,
    p_names: List[str],
    jitter: float = 1e-8,
    split_plot_opts: Optional["SplitPlotOptions"] = None,
    Z: Optional[np.ndarray] = None,
    all_factor_names: Optional[List[str]] = None,
) -> Dict:
    """Evaluate power for one response given a fixed design matrix X.

    Parameters
    ----------
    response : ResponseSpec
        Specification of the response's power requirements.
    X : ndarray (n, p)
        Model matrix for this response's formula.
    p_names : list of str
        Column names of X (from ``build_model_matrix``).
    jitter : float
        Tikhonov jitter for (X'X)⁻¹.
    split_plot_opts : SplitPlotOptions or None
        When not None, uses GLS power functions (``contrast_power_sp`` /
        ``global_r2_power_sp``) instead of OLS versions.
    Z : ndarray (n, n_wp) or None
        Whole-plot indicator matrix; required when *split_plot_opts* is not None.
    all_factor_names : list of str or None
        All factor names in the design (HTC + ETC).  Used to resolve
        HTC column indices when *split_plot_opts* is provided.  When None,
        ``split_plot_opts.htc_factors`` is used as a best-effort fallback.

    Returns
    -------
    dict
        Keys: ``"name"``, ``"power"``, ``"lam"``, ``"df2"``,
        and ``"df1"`` (contrast mode only).
    """
    from .config import PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig  # avoid circular at module level

    cfg = response.power_cfg
    n, p = X.shape
    df2 = int(n - int(np.linalg.matrix_rank(X)))

    if isinstance(cfg, PowerGLMContrastConfig):
        result = glm_contrast_power(cfg, X, jitter=jitter)
        return {
            "name": response.name,
            "power": result.power,
            "lam": result.lam,
            "df1": int(np.linalg.matrix_rank(np.asarray(cfg.L))),
            "df2": None,
            "family": cfg.family,
            "baseline": cfg.baseline,
        }
    elif isinstance(cfg, PowerContrastConfig):
        if split_plot_opts is not None:
            if Z is None:
                raise ValueError(
                    "eval_response_power: Z (whole-plot indicator) is required "
                    "when split_plot_opts is provided."
                )
            from .split_plot import htc_factor_cols_from_names

            _all_fnames = all_factor_names if all_factor_names is not None else list(
                split_plot_opts.htc_factors
            )
            htc_cols = htc_factor_cols_from_names(
                p_names, split_plot_opts.htc_factors, _all_fnames
            )
            result = contrast_power_sp(
                cfg.L, cfg.delta, X, Z,
                sigma_sp=cfg.sigma, eta=split_plot_opts.eta, alpha=cfg.alpha,
                df_method=split_plot_opts.df_method, htc_factor_cols=htc_cols,
                jitter=jitter,
            )
        else:
            result = contrast_power(cfg.L, cfg.delta, X, cfg.sigma, cfg.alpha, jitter=jitter)
        return {
            "name": response.name,
            "power": result.power,
            "lam": result.lam,
            "df1": int(np.linalg.matrix_rank(cfg.L)),
            "df2": df2,
        }
    else:  # PowerR2Config (fallback)
        if split_plot_opts is not None:
            if Z is None:
                raise ValueError(
                    "eval_response_power: Z (whole-plot indicator) is required "
                    "when split_plot_opts is provided."
                )
            from .split_plot import htc_factor_cols_from_names

            _all_fnames_r2 = all_factor_names if all_factor_names is not None else list(
                split_plot_opts.htc_factors
            )
            _htc_cols_r2 = htc_factor_cols_from_names(
                p_names, split_plot_opts.htc_factors, _all_fnames_r2
            )
            result = global_r2_power_sp(
                cfg.r2_target, X, Z,
                sigma_sp=cfg.sigma, eta=split_plot_opts.eta, alpha=cfg.alpha,
                df_method=split_plot_opts.df_method, lambda_mode=cfg.lambda_mode,
                jitter=jitter, htc_factor_cols=_htc_cols_r2,
            )
        else:
            result = global_r2_power(cfg.r2_target, X, cfg.alpha, lambda_mode=cfg.lambda_mode)
        return {
            "name": response.name,
            "power": result.power,
            "lam": result.lam,
            "df2": df2,
        }


def hotelling_t2_power(
    L: np.ndarray,
    Delta: np.ndarray,
    X: np.ndarray,
    sigma_joint: np.ndarray,
    alpha: float = 0.05,
    jitter: float = 1e-8,
) -> HotellingT2Result:
    """Joint power for k simultaneous linear contrasts (Hotelling-Lawley trace).

    Computes the multivariate power for testing H0: CΒ = 0 vs H1: CΒ = Δ,
    where all k responses share the common contrast matrix L and the inter-response
    error covariance is Σ (``sigma_joint``).

    Noncentrality matrix
    --------------------
    Ω = Δ' [L(X'X)⁻¹L']⁻¹ Δ Σ⁻¹          (k × k)

    T²-style F approximation
    ------------------------
    λ   = trace(Ω)              (the Hotelling-Lawley noncentrality)
    df1 = rank(L) · k
    df2 = n − rank(X) − k + 1
    F_crit = F_{1−α}(df1, df2)
    Power  = 1 − ncF(F_crit; df1, df2, λ)

    .. note:: **Approximation scope (SR-20b).**
        These df are the Hotelling T² form, exact when
        s = min(rank(L), k) = 1 (single contrast row, or single response) —
        MC-verified. For s ≥ 2 the approximation is slightly
        **conservative**: Monte-Carlo calibration against the true
        Hotelling-Lawley trace test measured power understated by
        ≲ 0.015 at small n (e.g. 0.538 vs 0.550 at n=16, q=k=2) and
        converging with n. The textbook one-moment HL df2
        (s·(ve − k − 1) + 2) with unscaled λ was tested and rejected: it is
        systematically anti-conservative (e.g. 0.619 vs 0.550 in the same
        configuration).

    This reduces exactly to ``contrast_power`` when k = 1 and
    ``sigma_joint = [[σ²]]``.

    Parameters
    ----------
    L : ndarray (q × p)
        Common contrast matrix shared by all k responses.  If 1-D, treated as
        a single-row matrix.
    Delta : ndarray (q × k)
        Effect-size matrix; column r is δ_r for response r.  If 1-D (q,),
        treated as q × 1 (single response).
    X : ndarray (n × p)
        Design/model matrix.
    sigma_joint : ndarray (k × k)
        Inter-response error covariance matrix.  Must be symmetric and
        positive definite.
    alpha : float
        Significance level (default 0.05).
    jitter : float
        Small ridge added to X'X for numerical stability.

    Returns
    -------
    HotellingT2Result
        NamedTuple with fields ``power``, ``lam``, ``df1``, ``df2``.

    Raises
    ------
    ValueError
        If ``sigma_joint`` is non-symmetric or singular.
    ValueError
        If df2 ≤ 0 (n too small for k responses).
    """
    _require_scipy()

    L = np.asarray(L, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    X = np.asarray(X, dtype=float)
    sigma_joint = np.asarray(sigma_joint, dtype=float)

    if L.ndim == 1:
        L = L.reshape(1, -1)
    if Delta.ndim == 1:
        Delta = Delta.reshape(-1, 1)

    n, p = X.shape
    q, p_L = L.shape
    q_D, k = Delta.shape

    if p_L != p:
        raise ValueError(
            f"L has {p_L} columns but X has {p} columns; shapes incompatible."
        )
    if q_D != q:
        raise ValueError(
            f"L has {q} rows but Delta has {q_D} rows; both must equal q."
        )
    if sigma_joint.shape != (k, k):
        raise ValueError(
            f"sigma_joint must be ({k}, {k}) for k={k} responses, "
            f"got {sigma_joint.shape}."
        )

    # Symmetry check
    if not np.allclose(sigma_joint, sigma_joint.T, atol=1e-8):
        raise ValueError(
            "sigma_joint must be symmetric; "
            f"max asymmetry = {float(np.max(np.abs(sigma_joint - sigma_joint.T))):.3e}."
        )

    # Positive-definiteness: check via slogdet
    sign, _ = np.linalg.slogdet(sigma_joint)
    if sign <= 0:
        raise ValueError(
            "sigma_joint is singular (non-positive determinant). "
            "The inter-response covariance must be positive definite."
        )

    rank_X = int(np.linalg.matrix_rank(X))
    # df1 counts independent hypothesis constraints: rank(L), not L's row
    # count — duplicated contrast rows leave λ unchanged (pinv projects) but
    # would otherwise inflate df1 and understate power (SR-20a).
    q_eff = int(np.linalg.matrix_rank(L))
    df1 = q_eff * k
    df2 = n - rank_X - k + 1

    if df1 <= 0:
        raise ValueError(f"df1 = rank(L)*k = {df1} ≤ 0; L or Delta has zero rank.")
    if df2 <= 0:
        raise ValueError(
            f"Hotelling T² df2 = n − rank(X) − k + 1 = {df2} ≤ 0. "
            f"Increase n (currently {n}) or reduce k (currently {k})."
        )

    # [L (X'X)⁻¹ L']⁻¹  (q × q)
    XtX_inv = _pinv_xtx(X, jitter=jitter)
    V_unscaled = _symmetrize(L @ XtX_inv @ L.T)
    if np.linalg.matrix_rank(V_unscaled) < q:
        V_inv = np.linalg.pinv(V_unscaled)
    else:
        try:
            V_inv = np.linalg.inv(V_unscaled)
        except np.linalg.LinAlgError:
            V_inv = np.linalg.pinv(V_unscaled)
    _check_delta_consistency(V_unscaled, V_inv, Delta)

    # Σ⁻¹  (k × k)
    Sigma_inv = np.linalg.inv(sigma_joint)

    # Noncentrality matrix  Ω = Δ' V⁻¹ Δ Σ⁻¹  (k × k)
    Omega = Delta.T @ V_inv @ Delta @ Sigma_inv

    # λ = trace(Ω) — Pillai-Bartlett approximation; equals contrast λ for k=1
    lam = float(max(0.0, np.trace(Omega)))

    F_crit = float(f_dist.ppf(1.0 - alpha, df1, df2))
    power = float(np.clip(1.0 - ncf_dist.cdf(F_crit, df1, df2, lam), 0.0, 1.0))

    return HotellingT2Result(power=power, lam=lam, df1=df1, df2=df2)


__all__ = [
    "contrast_power",
    "global_r2_power",
    "contrast_power_sp",
    "global_r2_power_sp",
    "eval_response_power",
    "combine_powers",
    "hotelling_t2_power",
    "ContrastPowerResult",
    "GlobalPowerResult",
    "HotellingT2Result",
    "_r2_df_num",
]
