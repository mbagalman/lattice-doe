# split_plot.py
# License: MIT
"""
Two-stratum covariance utilities for split-plot designs.

This module provides the matrix-algebra building blocks for split-plot (hard-to-change
factor) experiments.  All functions operate on NumPy arrays and have no dependency on
the rest of the package beyond this module, making them independently testable.

Two-stratum variance model
--------------------------
The observations in a balanced split-plot experiment with n_wp whole plots of
subplots_per_wp sub-plots each satisfy::

    y_ij = X β + τ_i + ε_ij

    τ_i  ~ N(0, σ²_wp)    (whole-plot error, shared within WP i)
    ε_ij ~ N(0, σ²_sp)    (sub-plot error, independent)

The full variance-covariance matrix is::

    V = σ²_sp · (η Z Z' + I_n)   where η = σ²_wp / σ²_sp

All functions that return V⁻¹ return the *scaled* inverse::

    Ṽ⁻¹ = (η Z Z' + I_n)⁻¹

so that σ²_sp cancels when the caller computes e.g. M = X' Ṽ⁻¹ X / σ²_sp.
"""
from __future__ import annotations

import re
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Whole-plot indicator matrix
# ---------------------------------------------------------------------------

def build_whole_plot_indicator(n_total: int, n_wp: int, subplots_per_wp: int) -> np.ndarray:
    """Build the n_total × n_wp whole-plot indicator matrix Z.

    Assumes a **balanced** layout: runs 0 .. subplots_per_wp-1 belong to WP 0,
    runs subplots_per_wp .. 2*subplots_per_wp-1 belong to WP 1, and so on.

    Parameters
    ----------
    n_total : int
        Total number of observations (must equal n_wp * subplots_per_wp).
    n_wp : int
        Number of whole plots.
    subplots_per_wp : int
        Number of sub-plots per whole plot.

    Returns
    -------
    Z : ndarray, shape (n_total, n_wp), dtype float64
        Z[i, k] = 1 if observation i belongs to whole plot k, else 0.

    Raises
    ------
    ValueError
        If n_total != n_wp * subplots_per_wp.
    """
    expected = n_wp * subplots_per_wp
    if n_total != expected:
        raise ValueError(
            f"n_total ({n_total}) must equal n_wp * subplots_per_wp "
            f"({n_wp} × {subplots_per_wp} = {expected})."
        )
    Z = np.zeros((n_total, n_wp), dtype=np.float64)
    for k in range(n_wp):
        start = k * subplots_per_wp
        Z[start : start + subplots_per_wp, k] = 1.0
    return Z


# ---------------------------------------------------------------------------
# Scaled covariance inverse using the Woodbury identity
# ---------------------------------------------------------------------------

def build_split_plot_covariance_inv(Z: np.ndarray, eta: float) -> np.ndarray:
    """Compute Ṽ⁻¹ = (η Z Z' + I_n)⁻¹ using the Woodbury matrix identity.

    Uses the identity::

        (I + η Z Z')⁻¹ = I − η Z (I_{n_wp} + η Z'Z)⁻¹ Z'

    Because Z is a block-indicator matrix, Z'Z is diagonal with entries equal
    to the size of each whole plot.  This makes the inner n_wp × n_wp inverse
    trivial (diagonal inversion), avoiding any full n × n matrix inverse.

    For balanced designs (all WPs the same size s)::

        Ṽ⁻¹[i, i] = 1 − η / (1 + η s)         (diagonal)
        Ṽ⁻¹[i, j] = −η / (1 + η s)             (same WP, off-diagonal)
        Ṽ⁻¹[i, j] = 0                           (different WPs)

    Parameters
    ----------
    Z : ndarray, shape (n, n_wp)
        Whole-plot indicator matrix (output of ``build_whole_plot_indicator``).
    eta : float
        Variance ratio σ²_wp / σ²_sp.  Must be ≥ 0.
        When eta = 0 the function returns the identity matrix exactly.

    Returns
    -------
    V_inv : ndarray, shape (n, n), dtype float64
        The scaled inverse (η Z Z' + I_n)⁻¹.

    Raises
    ------
    ValueError
        If eta < 0.
    """
    if eta < 0:
        raise ValueError(f"eta must be ≥ 0, got {eta}.")

    n = Z.shape[0]

    if eta == 0.0:
        return np.eye(n, dtype=np.float64)

    # D = I_{n_wp} + η Z'Z  →  diagonal entries = 1 + η * (wp_size_i)
    ZtZ_diag = np.sum(Z ** 2, axis=0)          # shape (n_wp,); wp_sizes
    D_diag = 1.0 + eta * ZtZ_diag              # shape (n_wp,)
    D_inv_diag = 1.0 / D_diag                  # shape (n_wp,)

    # Ṽ⁻¹ = I − η Z diag(D_inv) Z'
    # = I − η (Z * D_inv_diag) Z'   [broadcast scale each column of Z]
    ZD = Z * D_inv_diag[np.newaxis, :]         # shape (n, n_wp)
    correction = eta * (ZD @ Z.T)              # shape (n, n)

    V_inv = np.eye(n, dtype=np.float64) - correction
    return V_inv


# ---------------------------------------------------------------------------
# GLS information matrix
# ---------------------------------------------------------------------------

def gls_information_matrix(
    X: np.ndarray,
    V_inv: np.ndarray,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Compute the GLS information matrix M = X' V⁻¹ X with a relative ridge.

    The ridge added to each diagonal entry is ``jitter * M_ii`` (with 1.0
    substituted for zero diagonal entries), i.e. relative to each column's
    own scale. An absolute ridge ``jitter * I`` dominated columns expressed
    in small physical units, silently inflating GLS noncentrality parameters
    (anti-conservative power, SR-8); the relative ridge makes results
    invariant to factor-column units.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Model (design) matrix.
    V_inv : ndarray, shape (n, n)
        Scaled inverse covariance (η Z Z' + I_n)⁻¹.  Pass ``np.eye(n)``
        to recover the standard OLS information matrix X'X.
    jitter : float, default 1e-8
        Relative ridge magnitude; each diagonal entry is inflated by the
        factor ``(1 + jitter)``.

    Returns
    -------
    M : ndarray, shape (p, p), dtype float64
        Symmetric positive semi-definite GLS information matrix.
    """
    M = X.T @ V_inv @ X
    M = 0.5 * (M + M.T)  # enforce symmetry
    diag = np.diag(M)
    ridge = np.where(diag > 0, diag, 1.0)
    M += jitter * np.diag(ridge)
    return M


# ---------------------------------------------------------------------------
# Map HTC factor names → model-matrix column indices
# ---------------------------------------------------------------------------

def htc_factor_cols_from_names(
    p_names: List[str],
    htc_factors: List[str],
    all_factor_names: List[str],
) -> List[int]:
    """Return model-matrix column indices that involve only HTC (whole-plot) factors.

    A column is classified as WP-pure when its Patsy/dmatrix label contains no
    ETC (sub-plot) factor name as a word token.  The ``"Intercept"`` column is
    always classified as WP.

    Parameters
    ----------
    p_names : list of str
        Column names returned by ``build_model_matrix`` (length p).
    htc_factors : list of str
        Names of the hard-to-change (whole-plot) factors.
    all_factor_names : list of str
        All factor names present in the design (HTC + ETC).  Used to derive
        which identifier tokens in column labels are ETC factors.

    Returns
    -------
    list of int
        Zero-based column indices of WP-pure model-matrix columns.
    """
    if not htc_factors:
        return []

    htc_set = set(htc_factors)
    etc_set = {f for f in all_factor_names if f not in htc_set}

    result: List[int] = []
    for i, name in enumerate(p_names):
        if name == "Intercept":
            result.append(i)
            continue
        # Extract all Python-identifier tokens from the column label.
        tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", name))
        # A column is WP-pure when none of its tokens match an ETC factor name.
        if not tokens.intersection(etc_set):
            result.append(i)
    return result


# ---------------------------------------------------------------------------
# Contrast classification (WP vs SP)
# ---------------------------------------------------------------------------

def classify_contrasts(
    L: np.ndarray,
    htc_factor_cols: List[int],
    p: int,
) -> np.ndarray:
    """Classify each row of L as a whole-plot (WP) or sub-plot (SP) contrast.

    A contrast row is classified as **pure WP** if every non-zero entry falls
    within the columns corresponding to HTC (whole-plot) factors.  SP effects
    include any contrast involving an ETC factor column.

    Parameters
    ----------
    L : ndarray, shape (q, p)
        Contrast matrix.
    htc_factor_cols : list of int
        Column indices of the model matrix X that correspond to HTC factors
        (and the intercept, if you want intercept contrasts treated as WP).
    p : int
        Total number of model-matrix columns (used only for shape validation).

    Returns
    -------
    is_wp : ndarray, shape (q,), dtype bool
        ``True`` for rows that involve only WP factor columns.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L.reshape(1, -1)

    htc_set = set(htc_factor_cols)
    is_wp = np.zeros(L.shape[0], dtype=bool)
    for i, row in enumerate(L):
        nonzero_cols = set(int(j) for j in np.where(np.abs(row) > 1e-12)[0])
        is_wp[i] = nonzero_cols.issubset(htc_set)
    return is_wp


# ---------------------------------------------------------------------------
# Per-contrast denominator degrees of freedom
# ---------------------------------------------------------------------------

def split_plot_df_denom(
    X: np.ndarray,
    Z: np.ndarray,
    is_wp_contrast: np.ndarray,
    df_method: str,
    htc_factor_cols: Optional[List[int]] = None,
) -> np.ndarray:
    """Compute per-contrast denominator degrees of freedom for a split-plot design.

    Two error strata are available:

    * **WP df** — ``df_wp = n_wp − rank(X_wp)`` where X_wp is the submatrix of
      X restricted to HTC-factor columns, with one representative row per WP.
      If *htc_factor_cols* is ``None``, the approximation ``n_wp − 1`` is used.

    * **SP df** — ``df_sp = n_total − n_wp − (rank(X) − rank(X_wp))``.
      If *htc_factor_cols* is ``None``, the approximation ``n_total − n_wp − 1``
      is used.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Full model matrix.
    Z : ndarray, shape (n, n_wp)
        Whole-plot indicator matrix.
    is_wp_contrast : ndarray, shape (q,), dtype bool
        Output of ``classify_contrasts`` — True for pure WP contrasts.
    df_method : {"auto", "conservative", "sp_only"}
        How to assign df:

        * ``"auto"`` — WP df for pure-WP contrasts, SP df for all others.
        * ``"conservative"`` — always WP df.
        * ``"sp_only"`` — always SP df.
    htc_factor_cols : list of int or None
        Column indices of HTC factors in X (used for rank computation).
        When ``None``, simpler approximations are used.

    Returns
    -------
    df : ndarray, shape (q,), dtype int
        Denominator df for each contrast row.

    Raises
    ------
    ValueError
        If df_method is not one of the accepted values.
    """
    if df_method not in ("auto", "conservative", "sp_only"):
        raise ValueError(
            f"df_method must be 'auto', 'conservative', or 'sp_only'; "
            f"got {df_method!r}."
        )

    n_total, n_wp = Z.shape
    rank_X = int(np.linalg.matrix_rank(X))

    # Compute rank(X_wp) — the rank of the WP-factor model sub-matrix.
    if htc_factor_cols is not None and len(htc_factor_cols) > 0:
        # One representative row per WP (first row belonging to each WP).
        wp_rep_rows = [int(np.where(Z[:, k] > 0)[0][0]) for k in range(n_wp)]
        X_wp = X[np.ix_(wp_rep_rows, list(htc_factor_cols))]
        rank_X_wp = int(np.linalg.matrix_rank(X_wp))
    else:
        # Fallback: approximate rank_X_wp as 1 (intercept only).
        rank_X_wp = 1

    df_wp = max(1, n_wp - rank_X_wp)
    df_sp = max(1, n_total - n_wp - (rank_X - rank_X_wp))

    is_wp = np.asarray(is_wp_contrast, dtype=bool)
    q = len(is_wp)

    if df_method == "conservative":
        return np.full(q, df_wp, dtype=int)
    if df_method == "sp_only":
        return np.full(q, df_sp, dtype=int)
    # "auto"
    df = np.where(is_wp, df_wp, df_sp).astype(int)
    return df


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "build_whole_plot_indicator",
    "build_split_plot_covariance_inv",
    "gls_information_matrix",
    "classify_contrasts",
    "split_plot_df_denom",
    "htc_factor_cols_from_names",
]
