"""Pure-NumPy diagnostic metrics — no matplotlib dependency."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import warnings

__all__ = ["compute_leverages", "compute_design_metrics"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _xtx(X: np.ndarray) -> np.ndarray:
    return X.T @ X


def _pinv(M: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(M)


def _has_intercept(col: np.ndarray, atol: float = 1e-12) -> bool:
    """Heuristic: column is (near) constant ones."""
    if np.std(col) < atol and np.allclose(col.mean(), 1.0, atol=1e-8):
        return True
    return False


def _compute_vif(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    *,
    detect_intercept: bool = True,
    jitter: float = 1e-12,
) -> pd.DataFrame:
    """Compute variance inflation factors (VIF) for columns of X.

    Returns DataFrame with feature names and VIF values for better
    interpretability.
    """
    n, p = X.shape

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(p)]

    # Identify non-intercept columns
    keep_idx = []
    keep_names = []
    for j in range(p):
        if detect_intercept and _has_intercept(X[:, j]):
            continue
        keep_idx.append(j)
        keep_names.append(feature_names[j])

    if not keep_idx:
        return pd.DataFrame(columns=["feature", "vif"])

    # Compute VIFs for non-intercept columns
    Z = X[:, keep_idx].astype(float)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0, ddof=1)

    # Handle zero variance columns (perfectly constant)
    zero_sd_mask = sd == 0
    if np.any(zero_sd_mask):
        sd[zero_sd_mask] = 1.0

    Z = (Z - mu) / sd

    try:
        R = (Z.T @ Z) / max(n - 1, 1)
        R = R + jitter * np.eye(R.shape[0])
        Rinv = np.linalg.pinv(R)
        vif_values = np.diag(Rinv).copy()
        vif_values[np.isinf(vif_values)] = 1e12
    except np.linalg.LinAlgError:
        warnings.warn(
            "VIF calculation failed due to a linear algebra error. "
            f"Returning NaN for {len(keep_names)} features.",
            RuntimeWarning,
        )
        vif_values = np.full(len(keep_names), np.nan)

    return pd.DataFrame({"feature": keep_names, "vif": vif_values})


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_leverages(X: np.ndarray) -> np.ndarray:
    """Compute leverage values (diagonal of hat matrix).

    Leverage indicates influence of each design point on predictions.
    High leverage points (> 2p/n) may be overly influential.

    Parameters
    ----------
    X : ndarray (n x p)
        Design matrix.

    Returns
    -------
    ndarray (n,)
        Leverage value for each design point.
    """
    XtX_inv = _pinv(_xtx(X))
    H = X @ XtX_inv @ X.T
    return np.diag(H)


def compute_design_metrics(
    X: np.ndarray,
    *,
    include_vif: bool = False,
    X_cand: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute core design diagnostics from a model matrix X.

    Parameters
    ----------
    X : ndarray (n x p)
        Design (model) matrix.
    include_vif : bool, default False
        If True, compute VIFs as a DataFrame.
    X_cand : ndarray (N_cand x p), optional
        Candidate-region model matrix for I-criterion.
    feature_names : list of str, optional
        Names for model matrix columns (for VIF reporting).

    Returns
    -------
    dict
        {
          'condition_number' : float,
          'd_efficiency'     : float,
          'leverage_mean'    : float,
          'leverage_max'     : float,
          'i_criterion'      : float (if X_cand provided),
          'i_criterion_n_cand': int (if X_cand provided),
          'vif_df'           : DataFrame (if include_vif=True),
          'leverages'        : ndarray (always included for plotting)
        }
    """
    n, p = X.shape
    if p == 0:
        return {
            "condition_number": np.nan,
            "d_efficiency": np.nan,
            "leverage_mean": np.nan,
            "leverage_max": np.nan,
            "leverages": np.array([]),
        }

    XtX = _xtx(X)
    cond = float(np.linalg.cond(XtX))

    # D-efficiency — normalised to [0, 1] via (det(X'X) / n^p)^(1/p).
    # Clamped to 1.0: values above 1 can arise from the continuous
    # normalisation baseline and would be uninterpretable in the [0,1] scale.
    sign, logdet = np.linalg.slogdet(XtX)
    if sign <= 0:
        d_eff = 0.0
    else:
        log_d_eff = (1.0 / p) * logdet - np.log(n)
        d_eff = min(1.0, float(np.exp(log_d_eff)))

    # Leverage statistics
    try:
        leverages = compute_leverages(X)
        leverage_mean = float(np.mean(leverages))
        leverage_max = float(np.max(leverages))
    except np.linalg.LinAlgError:
        warnings.warn("Leverage calculation failed due to singular matrix.")
        leverages = np.full(n, np.nan)
        leverage_mean = np.nan
        leverage_max = np.nan

    out: Dict[str, Any] = {
        "condition_number": cond,
        "d_efficiency": d_eff,
        "leverage_mean": leverage_mean,
        "leverage_max": leverage_max,
        "leverages": leverages,
    }

    # I-criterion over candidate region
    if X_cand is not None and X_cand.size > 0:
        n_cand = X_cand.shape[0]
        try:
            XtX_inv = _pinv(XtX)
            Mcand = X_cand.T @ X_cand
            out["i_criterion"] = float(np.trace(XtX_inv @ Mcand) / n_cand)
            out["i_criterion_n_cand"] = n_cand
        except np.linalg.LinAlgError:
            out["i_criterion"] = np.nan
            out["i_criterion_n_cand"] = n_cand

    if include_vif:
        out["vif_df"] = _compute_vif(X, feature_names)

    return out
