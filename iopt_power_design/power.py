# power.py
# License: MIT
"""
Power calculations for linear models (F-tests)
=============================================

This module provides power calculations for:
  - Linear contrasts (Wald test on Lβ = δ)
  - Global R² (full model F-test)

Core steps:
-----------
1. Compute noncentrality parameter λ based on the model matrix X
   and effect size specification (contrast or R²).
2. Compute power using the noncentral F distribution:
      power = 1 - F_{df1, df2, λ}(Fcrit)
   where Fcrit is the critical value at significance level α.

Notes:
------
• Shapes are validated and broadcast where sensible.
• We use numerically-stable pseudo-inverses with small Tikhonov jitter for X'X.
• We avoid importing heavy dependencies outside SciPy / NumPy.
"""

from __future__ import annotations

from typing import Tuple, Literal, NamedTuple
import numpy as np

try:
    # Only import when actually computing power
    from scipy.stats import ncf as ncf_dist
    from scipy.stats import f as f_dist
except Exception as e:  # pragma: no cover
    # Delay the error until one of the functions is actually used.
    ncf_dist = None
    f_dist = None
    _scipy_import_error = e
else:
    _scipy_import_error = None


# --- Custom Types for readable return values ---
ContrastPowerResult = NamedTuple("ContrastPowerResult", [("power", float), ("lam", float)])
GlobalPowerResult = NamedTuple("GlobalPowerResult", [("power", float), ("lam", float)])


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
    Compute pinv(X'X + jitter*I) with a small ridge for stability.

    Parameters
    ----------
    X : ndarray (n x p)
        Model/design matrix.
    jitter : float
        Small positive number added to the diagonal before inversion.

    Returns
    -------
    ndarray (p x p)
        Moore–Penrose inverse of the (regularized) X'X.
    """
    XtX = X.T @ X
    p = XtX.shape[0]
    XtX_reg = XtX + float(jitter) * np.eye(p, dtype=XtX.dtype)
    # Return the (regularized) pseudo-inverse
    return np.linalg.pinv(XtX_reg)


def _symmetrize(A: np.ndarray) -> np.ndarray:
    """Force numerical symmetry."""
    return 0.5 * (A + A.T)


def _r2_df_num(X: np.ndarray) -> int:
    """Numerator df for the global R² F-test (slopes only, intercept excluded).

    Matches the G*Power / ``pwr.f2.test`` convention: if X contains an
    intercept column (all-ones, std ≈ 0, mean ≈ 1), df_num = rank(X) - 1;
    otherwise df_num = rank(X).  This is the single authoritative source of
    truth shared by ``global_r2_power`` and any caller that needs the same
    value for reporting.
    """
    rank_X = int(np.linalg.matrix_rank(X))
    p = X.shape[1]
    has_intercept = any(
        float(np.std(X[:, j])) < 1e-12
        and bool(np.allclose(X[:, j].mean(), 1.0, atol=1e-8))
        for j in range(p)
    )
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

    # ADDED: Validate for edge cases (zero variance)
    # Check if the variance-covariance matrix of the contrast is rank-deficient
    # A rank of 0 means the contrast has 0 variance and is untestable.
    rank_V = np.linalg.matrix_rank(V_unscaled)
    if rank_V == 0:
        raise ValueError(
            "Contrast variance matrix (L @ (X'X_inv) @ L.T) has rank 0. "
            "The contrast is not testable (e.g., it is in the null space of X)."
        )
    # Note: If rank_V < df_num, contrasts are linearly dependent,
    # but pinv will correctly handle this, so we don't error.

    V = V_unscaled * (sigma ** 2)

    # Use pseudo-inverse in case V is singular
    V_inv = np.linalg.pinv(V)
    lam = float(delta.T @ V_inv @ delta)

    # ADDED: Add validation that noncentrality parameter is non-negative
    # A quadratic form should be >= 0, but clip to handle float error
    if lam < -1e-8:  # Allow for small float tolerance
        raise ValueError(
            f"Computed noncentrality parameter lambda is negative ({lam}), "
            "indicating a numerical issue."
        )
    lam = max(0.0, lam)  # Clip small negative values to 0

    # Critical F and power
    Fcrit = f_dist.isf(alpha, df_num, df_denom)
    power = float(1.0 - ncf_dist.cdf(Fcrit, df_num, df_denom, lam))

    # ADDED: Clip power values to [0, 1] range
    power = np.clip(power, 0.0, 1.0)

    return ContrastPowerResult(power=power, lam=lam)


# ---------------------------------------------------------------------
def global_r2_power(
    r2_target: float,
    X: np.ndarray,
    alpha: float,
    lambda_mode: Literal["n", "n_minus_p"] = "n",
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
    df_num = _r2_df_num(X)

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

    # ADDED: Add validation that noncentrality parameter is non-negative
    # This should always be true if r2_target > 0, but good for safety.
    if lam < 0.0:
        raise ValueError(
            f"Computed noncentrality parameter lambda is negative ({lam}), "
            "indicating a numerical issue."
        )

    Fcrit = f_dist.isf(alpha, df_num, df_denom)
    power = float(1.0 - ncf_dist.cdf(Fcrit, df_num, df_denom, lam))

    # ADDED: Clip power values to [0, 1] range
    power = np.clip(power, 0.0, 1.0)

    return GlobalPowerResult(power=power, lam=lam)


__all__ = ["contrast_power", "global_r2_power", "ContrastPowerResult", "GlobalPowerResult", "_r2_df_num"]
