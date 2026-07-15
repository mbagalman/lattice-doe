# iopt_search.py
# License: MIT
"""
I-, D-, and A-optimal design search
=====================================

This module provides the Fedorov point-exchange optimizer and all
surrounding orchestration needed to build optimal experimental designs:

  - ``_fedorov_exchange_single`` — single-start rank-1 exchange loop
  - ``_optimal_indices_from_X`` — multi-start serial wrapper
  - ``_i/_d/_a_criterion_for_indices`` — lower-is-better criterion scorers
  - ``_criterion_score`` — dispatcher
  - ``_score_design`` — direct matrix scorer (used by augment_design)
  - ``_one_start_worker`` — picklable worker for ProcessPoolExecutor
  - ``build_i_opt_design_with_idx`` — main search entry point (returns idx)
  - ``build_i_opt_design`` — convenience wrapper (returns DataFrame only)
  - ``augment_design`` — greedy one-point-at-a-time augmentation

Dependencies (all within this package):
  candidate.py  →  model_matrix.py  →  iopt_search.py  →  config.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Runtime import (not TYPE_CHECKING): typing.get_type_hints() on the public
# functions must be able to resolve the alias. utils has no imports from this
# module, so there is no circularity.
from .utils import FactorSpec

from .candidate import build_candidate, build_search_candidate
from .model_matrix import build_model_matrix
from .config import DesignOptions
from .split_plot import (
    gls_information_matrix,
    build_whole_plot_indicator,
    build_split_plot_covariance_inv,
)


# ---------------------------------------------------------------------
# Internal Fedorov point-exchange optimizer + scoring helpers
# ---------------------------------------------------------------------
def _fedorov_exchange_single(
    X_cand: np.ndarray,
    n: int,
    *,
    criterion: str,
    max_iter: int,
    seed: int,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Single-start Fedorov point-exchange on a pre-built model matrix.

    Uses rank-1 Sherman-Morrison updates so that the inner candidate-swap
    loop is fully vectorised over all non-design points — no per-swap matrix
    inversion is required.

    Parameters
    ----------
    X_cand : ndarray (n_cand, p)
        Pre-built model matrix over the full candidate set.
    n : int
        Number of design rows to select.
    criterion : {"I", "D", "A"}
        Optimality criterion (lower is better in all cases).
    max_iter : int
        Maximum exchange-iteration rounds.
    seed : int
        Random seed for the initial random selection.
    jitter : float
        Diagonal ridge added to X'X for numerical stability.

    Returns
    -------
    idx : ndarray[int] of shape (n,)
        Row indices into X_cand forming the selected design.
    """
    rng = np.random.default_rng(seed)
    n_cand, p = X_cand.shape

    # --- Initial random design ---
    if n > n_cand:
        raise ValueError(
            f"n={n} exceeds candidate set size n_cand={n_cand}."
        )
    idx = rng.choice(n_cand, size=n, replace=False)
    in_design = np.zeros(n_cand, dtype=bool)
    in_design[idx] = True

    # Candidate moment matrix for I-criterion (unchanged across iterations)
    Mcand: Optional[np.ndarray] = (X_cand.T @ X_cand) if criterion == "I" else None

    for _iter in range(max_iter):
        # Current moment matrix and its regularised inverse
        X_d = X_cand[idx]                        # n × p
        M = X_d.T @ X_d + jitter * np.eye(p)    # p × p
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            break  # singular; stop improving

        # H[:, t] = M^-1 x_t  for every candidate t  →  shape (p, n_cand)
        H = M_inv @ X_cand.T                     # p × n_cand
        # leverages[t] = x_t' M^-1 x_t
        leverages = np.einsum("pt,pt->t", X_cand.T, H)   # (n_cand,)

        # Pre-gather non-design rows once per iteration
        non_idx = np.where(~in_design)[0]        # (n_non,)
        if len(non_idx) == 0:
            break
        X_non = X_cand[non_idx]                  # n_non × p
        H_non = H[:, non_idx]                    # p × n_non
        lev_non = leverages[non_idx]             # (n_non,)

        # Current criterion score used for A / I gain calculations
        if criterion == "A":
            current_score = float(np.trace(M_inv))
        elif criterion == "I":
            current_score = float(np.trace(M_inv @ Mcand))

        best_gain = 0.0   # only accept strictly improving swaps
        best_s_pos = -1
        best_t_local = -1   # index into non_idx

        for s_pos in range(len(idx)):
            s = idx[s_pos]
            d_s = float(leverages[s])
            h_s = H[:, s]                        # M^-1 x_s  (p,)
            denom_s = 1.0 - d_s
            if denom_s < 1e-10:
                continue  # near-degenerate; can't safely remove this point

            # w_s_all[k] = x_{non_idx[k]}' M^-1 x_s  (vectorised over all t)
            w_s_all = X_non @ h_s                # (n_non,)

            if criterion == "D":
                # Fedorov det-ratio (exact rank-2 update):
                #   v_t' = lev_non + w^2 / denom_s   (leverage w.r.t. M' = M - x_s x_s')
                #   ratio = denom_s * (1 + v_t')   →  gain = ratio - 1
                v_t_prime = lev_non + w_s_all * w_s_all / denom_s
                gains = denom_s * (1.0 + v_t_prime) - 1.0       # (n_non,)

            elif criterion == "A":
                # Step 1 – remove x_s:  trace(M'^-1) = trace(M^-1) + ||h_s||^2/denom_s
                trace_minus = current_score + float(h_s @ h_s) / denom_s
                # Step 2 – add x_t:  M'^-1 x_t via Sherman-Morrison (vectorised)
                mp_inv_xnon = H_non + np.outer(h_s, w_s_all) / denom_s   # p × n_non
                d_t_prime = np.einsum("pt,pt->t", X_non.T, mp_inv_xnon)  # (n_non,)
                norm2 = np.einsum("pt,pt->t", mp_inv_xnon, mp_inv_xnon)  # (n_non,)
                new_traces = trace_minus - norm2 / (1.0 + d_t_prime)     # (n_non,)
                gains = current_score - new_traces                        # (n_non,)

            else:  # criterion == "I"
                # Step 1 – remove x_s
                Mcand_hs = Mcand @ h_s                                    # p,
                trace_I_minus = current_score + float(h_s @ Mcand_hs) / denom_s
                # Step 2 – add x_t
                mp_inv_xnon = H_non + np.outer(h_s, w_s_all) / denom_s   # p × n_non
                d_t_prime = np.einsum("pt,pt->t", X_non.T, mp_inv_xnon)  # (n_non,)
                Mcand_mp = Mcand @ mp_inv_xnon                            # p × n_non
                delta_I = (
                    np.einsum("pt,pt->t", mp_inv_xnon, Mcand_mp)
                    / (1.0 + d_t_prime)
                )                                                         # (n_non,)
                gains = current_score - (trace_I_minus - delta_I)        # (n_non,)

            # Track the globally best swap across all design points
            local_best = int(np.argmax(gains))
            if gains[local_best] > best_gain:
                best_gain = float(gains[local_best])
                best_s_pos = s_pos
                best_t_local = local_best

        if best_s_pos == -1:
            break  # converged — no improving swap found

        # Apply the best swap
        old_pt = idx[best_s_pos]
        new_pt = non_idx[best_t_local]
        in_design[old_pt] = False
        in_design[new_pt] = True
        idx[best_s_pos] = new_pt

    return idx


def _optimal_indices_from_X(
    X_cand: np.ndarray,
    n: int,
    *,
    criterion: str,
    algo: str,
    n_start: int,
    max_iter: int,
    random_state: Optional[int],
    jitter: float = 1e-8,
) -> np.ndarray:
    """Run optimal design search on X_cand and return the best row-index set.

    Uses the internal Fedorov point-exchange algorithm (``_fedorov_exchange_single``)
    that works directly on the pre-built model matrix.  Multiple random starts are
    supported via ``n_start``; the start with the best criterion score is returned.

    Parameters
    ----------
    X_cand : ndarray (n_cand, p)
        Pre-built model matrix.
    n : int
        Number of design runs to select.
    criterion : {"I", "D", "A"}
        Optimality criterion (lower is better).
    algo : {"fedorov", "coordinate"}
        Retained for API compatibility; both map to the Fedorov exchange.
    n_start : int
        Number of independent random starts.
    max_iter : int
        Maximum exchange iterations per start.
    random_state : int or None
        Base random seed.  Start k uses seed ``base + k * 1337``.
    jitter : float
        Diagonal ridge added to X'X for numerical stability.

    Returns
    -------
    ndarray[int]
        Row indices into X_cand.
    """
    base = int(random_state) if random_state is not None else 0
    best_score = np.inf
    best_idx: Optional[np.ndarray] = None

    for k in range(max(1, n_start)):
        seed = base + k * 1337
        idx = _fedorov_exchange_single(
            X_cand,
            n,
            criterion=criterion,
            max_iter=max_iter,
            seed=seed,
            jitter=jitter,
        )
        score = _criterion_score(criterion, X_cand, idx, jitter=jitter)
        if score < best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:  # pragma: no cover – defensive
        raise RuntimeError("No valid design found in any random start.")
    return best_idx


def _i_criterion_for_indices(X_cand: np.ndarray, idx: np.ndarray, jitter: float = 1e-8) -> float:
    """Compute I-criterion over candidate region for a selected index set.

    I = (1/Ncand) * trace( (X'X)^-1 * (X_cand' X_cand) )
    where X is the design matrix formed by X_cand[idx, :].

    Lower values indicate better designs (lower average prediction variance).
    """
    X = X_cand[idx, :]
    p = X.shape[1]
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX + jitter * np.eye(p))
    Mcand = X_cand.T @ X_cand
    return float(np.trace(XtX_inv @ Mcand) / X_cand.shape[0])


def _d_criterion_for_indices(X_cand: np.ndarray, idx: np.ndarray, jitter: float = 1e-8) -> float:
    """Compute D-criterion score for a selected index set.

    D-optimal designs maximise det(X'X), equivalently maximise log det(X'X).
    We return -log det(X'X + jitter·I) so that *lower is better*, consistent
    with the I-criterion sign convention used for parallel-start selection.

    Parameters
    ----------
    X_cand : ndarray (n_cand x p)
        Full candidate model matrix.
    idx : ndarray of int
        Selected row indices into X_cand.
    jitter : float
        Small ridge added to X'X for numerical stability before computing
        the determinant.

    Returns
    -------
    float
        Negative log-determinant of the regularised X'X matrix; lower is
        better.  Returns ``float('inf')`` if X'X is (near-)singular even
        after regularisation.
    """
    X = X_cand[idx, :]
    p = X.shape[1]
    XtX = X.T @ X + jitter * np.eye(p)
    sign, logdet = np.linalg.slogdet(XtX)
    if sign <= 0:
        return float("inf")
    return -logdet


def _a_criterion_for_indices(X_cand: np.ndarray, idx: np.ndarray, jitter: float = 1e-8) -> float:
    """Compute A-criterion score for a selected index set.

    A-optimal designs minimise ``trace((X'X)^-1)``, the sum of variances of
    all coefficient estimates.  We return this trace directly (with a small
    ridge for numerical stability) — it is already a *lower-is-better* score,
    consistent with the I- and D-criterion sign conventions used in the
    parallel-start selector.

    Parameters
    ----------
    X_cand : ndarray (n_cand x p)
        Full candidate model matrix.
    idx : ndarray of int
        Selected row indices into X_cand.
    jitter : float
        Small ridge added to X'X for numerical stability.

    Returns
    -------
    float
        Trace of (X'X + jitter·I)^-1; lower is better.
        Returns ``float('inf')`` for (near-)singular designs.
    """
    X = X_cand[idx, :]
    p = X.shape[1]
    XtX = X.T @ X + jitter * np.eye(p)
    try:
        XtX_inv = np.linalg.inv(XtX)
        score = float(np.trace(XtX_inv))
        return score if np.isfinite(score) else float("inf")
    except np.linalg.LinAlgError:
        return float("inf")


# ---------------------------------------------------------------------
# GLS criterion scorers (split-plot / two-stratum variance model)
# ---------------------------------------------------------------------

def _gls_i_criterion(
    X: np.ndarray,
    V_inv: np.ndarray,
    Mcand: Optional[np.ndarray] = None,
    N_cand: int = 1,
    jitter: float = 1e-8,
) -> float:
    """GLS I-criterion: tr[(X'V⁻¹X + jitter·I)⁻¹ Mcand] / N_cand.

    Lower is better.  When *Mcand* is ``None``, the design's own moment
    matrix ``X'X`` is used with ``N_cand = n`` (treats the design as its own
    candidate pool).  The preferred usage is to pass ``Mcand = X_cand'X_cand``
    and ``N_cand = n_cand``.

    At V_inv = I the result equals the OLS I-criterion.

    Parameters
    ----------
    X : ndarray (n, p)
        Selected design matrix.
    V_inv : ndarray (n, n)
        Scaled inverse covariance ``(η Z Z' + I_n)⁻¹``.
    Mcand : ndarray (p, p), optional
        Candidate moment matrix ``X_cand'X_cand``.
    N_cand : int
        Number of candidate rows (denominator for normalisation).
    jitter : float
        Small ridge added to ``X'V⁻¹X`` for numerical stability.

    Returns
    -------
    float
        GLS I-criterion score; lower is better.  Returns ``float('inf')``
        for singular or near-singular designs.
    """
    M = gls_information_matrix(X, V_inv, jitter=jitter)   # p × p, PD
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return float("inf")

    if Mcand is None:
        Mcand = X.T @ X
        N_cand = max(X.shape[0], 1)

    score = float(np.trace(M_inv @ Mcand)) / max(N_cand, 1)
    return score if np.isfinite(score) else float("inf")


def _gls_d_criterion(
    X: np.ndarray,
    V_inv: np.ndarray,
    jitter: float = 1e-8,
) -> float:
    """GLS D-criterion: -log det(X'V⁻¹X + jitter·I).

    Lower is better (maximises GLS determinant).  At V_inv = I the result
    equals the OLS D-criterion.

    Parameters
    ----------
    X : ndarray (n, p)
        Selected design matrix.
    V_inv : ndarray (n, n)
        Scaled inverse covariance.
    jitter : float
        Small ridge added for numerical stability.

    Returns
    -------
    float
        Negative log-determinant; lower is better.  Returns ``float('inf')``
        for singular designs.
    """
    M = gls_information_matrix(X, V_inv, jitter=jitter)
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return float("inf")
    return float(-logdet)


def _gls_a_criterion(
    X: np.ndarray,
    V_inv: np.ndarray,
    jitter: float = 1e-8,
) -> float:
    """GLS A-criterion: tr[(X'V⁻¹X + jitter·I)⁻¹].

    Lower is better.  At V_inv = I the result equals the OLS A-criterion.

    Parameters
    ----------
    X : ndarray (n, p)
        Selected design matrix.
    V_inv : ndarray (n, n)
        Scaled inverse covariance.
    jitter : float
        Small ridge added for numerical stability.

    Returns
    -------
    float
        Trace of inverse GLS information matrix; lower is better.  Returns
        ``float('inf')`` for singular designs.
    """
    M = gls_information_matrix(X, V_inv, jitter=jitter)
    try:
        M_inv = np.linalg.inv(M)
        score = float(np.trace(M_inv))
        return score if np.isfinite(score) else float("inf")
    except np.linalg.LinAlgError:
        return float("inf")


def _score_design_gls(
    criterion: str,
    X: np.ndarray,
    V_inv: np.ndarray,
    Mcand: Optional[np.ndarray] = None,
    N_cand: int = 1,
    jitter: float = 1e-8,
) -> float:
    """Score a design matrix X under a GLS criterion.

    GLS analogue of ``_score_design``.  Used by the split-plot Fedorov
    exchange to evaluate candidate designs during each swap proposal.

    Parameters
    ----------
    criterion : {"I", "D", "A"}
        Optimality criterion; lower is better.
    X : ndarray (n, p)
        Current design matrix.
    V_inv : ndarray (n, n)
        Scaled inverse covariance ``(η Z Z' + I_n)⁻¹``.
    Mcand : ndarray (p, p), optional
        Candidate moment matrix ``X_cand'X_cand``.  Required for
        ``criterion="I"``; ignored for ``"D"`` and ``"A"``.
    N_cand : int, default 1
        Number of candidate rows (denominator for I-criterion).
    jitter : float
        Diagonal regularisation.

    Returns
    -------
    float
        GLS criterion score; lower is better.

    Raises
    ------
    ValueError
        If *criterion* is not ``"I"``, ``"D"``, or ``"A"``.
    """
    if criterion == "I":
        return _gls_i_criterion(X, V_inv, Mcand=Mcand, N_cand=N_cand, jitter=jitter)
    elif criterion == "D":
        return _gls_d_criterion(X, V_inv, jitter=jitter)
    elif criterion == "A":
        return _gls_a_criterion(X, V_inv, jitter=jitter)
    else:
        raise ValueError(
            f"Unknown optimality criterion {criterion!r}. "
            "Supported values are 'I' (I-optimal), 'D' (D-optimal), and 'A' (A-optimal)."
        )


def _criterion_score(
    criterion: str,
    X_cand: np.ndarray,
    idx: np.ndarray,
    jitter: float = 1e-8,
    *,
    V_inv: Optional[np.ndarray] = None,
) -> float:
    """Dispatch to the appropriate criterion scoring function.

    All criteria use a *lower-is-better* convention so that the parallel
    multi-start loop can compare scores uniformly.

    Parameters
    ----------
    criterion : {"I", "D", "A"}
        Optimality criterion.

        * ``"I"`` — I-optimal: minimise average prediction variance over the
          candidate region (preferred when prediction accuracy across the
          factor space matters).
        * ``"D"`` — D-optimal: maximise ``det(X'X)`` (preferred when precise
          coefficient estimation matters).
        * ``"A"`` — A-optimal: minimise ``trace((X'X)^-1)``, the sum of
          coefficient-estimate variances (preferred when all coefficients
          should be estimated with equal precision).

    X_cand : ndarray (n_cand x p)
        Full candidate model matrix.
    idx : ndarray of int
        Selected row indices into X_cand.
    jitter : float
        Small ridge for numerical stability passed to the underlying scorer.
    V_inv : ndarray (n_sel, n_sel), optional
        Scaled inverse covariance for the selected rows ``X_cand[idx]``.
        When provided, the GLS criterion path is used (split-plot).
        When ``None`` (default), the standard OLS path is used.

    Returns
    -------
    float
        Criterion score; lower is better in all cases.

    Raises
    ------
    ValueError
        If *criterion* is not ``"I"``, ``"D"``, or ``"A"``.
    """
    if V_inv is not None:
        # GLS path — split-plot / two-stratum variance model
        X_sel = X_cand[idx]
        n_cand = X_cand.shape[0]
        if criterion == "I":
            Mcand = X_cand.T @ X_cand
            return _gls_i_criterion(X_sel, V_inv, Mcand=Mcand, N_cand=n_cand, jitter=jitter)
        elif criterion == "D":
            return _gls_d_criterion(X_sel, V_inv, jitter=jitter)
        elif criterion == "A":
            return _gls_a_criterion(X_sel, V_inv, jitter=jitter)
        else:
            raise ValueError(
                f"Unknown optimality criterion {criterion!r}. "
                "Supported values are 'I' (I-optimal), 'D' (D-optimal), and 'A' (A-optimal)."
            )
    # OLS path — existing behaviour unchanged
    if criterion == "I":
        return _i_criterion_for_indices(X_cand, idx, jitter=jitter)
    elif criterion == "D":
        return _d_criterion_for_indices(X_cand, idx, jitter=jitter)
    elif criterion == "A":
        return _a_criterion_for_indices(X_cand, idx, jitter=jitter)
    else:
        raise ValueError(
            f"Unknown optimality criterion {criterion!r}. "
            "Supported values are 'I' (I-optimal), 'D' (D-optimal), and 'A' (A-optimal)."
        )


def _score_design(
    criterion: str,
    X: np.ndarray,
    Mcand: Optional[np.ndarray] = None,
    N_cand: int = 1,
    jitter: float = 1e-8,
) -> float:
    """Score a design matrix X directly under a given criterion.

    Unlike ``_criterion_score``, which operates via an index set into a
    candidate matrix, this helper accepts the design matrix X directly.
    It is used by ``augment_design`` where the growing design matrix is
    assembled incrementally and indexed rows are not available.

    Parameters
    ----------
    criterion : {"I", "D", "A"}
        Optimality criterion; lower is better in all cases.
    X : ndarray (n x p)
        Current design matrix.
    Mcand : ndarray (p x p), optional
        Candidate moment matrix X_cand'X_cand.  Required when
        ``criterion="I"``; ignored otherwise.
    N_cand : int, default 1
        Number of candidate rows (denominator for I-criterion normalisation).
    jitter : float
        Diagonal regularisation added to X'X.

    Returns
    -------
    float
        Criterion score; lower is better.
        Returns ``float('inf')`` for singular or near-singular designs.
    """
    p = X.shape[1]
    XtX = X.T @ X + jitter * np.eye(p)
    if criterion == "I":
        if Mcand is None:
            raise ValueError("Mcand (candidate moment matrix) is required for criterion='I'.")
        XtX_inv = np.linalg.pinv(XtX)
        return float(np.trace(XtX_inv @ Mcand) / max(N_cand, 1))
    elif criterion == "D":
        sign, logdet = np.linalg.slogdet(XtX)
        if sign <= 0:
            return float("inf")
        return float(-logdet)
    elif criterion == "A":
        try:
            XtX_inv = np.linalg.inv(XtX)
            score = float(np.trace(XtX_inv))
            return score if np.isfinite(score) else float("inf")
        except np.linalg.LinAlgError:
            return float("inf")
    else:
        raise ValueError(
            f"Unknown optimality criterion {criterion!r}. "
            "Supported values are 'I' (I-optimal), 'D' (D-optimal), and 'A' (A-optimal)."
        )


def _one_start_worker(
    X_cand: np.ndarray,
    n: int,
    seed: int,
    *,
    algo: str,
    criterion: str,
    max_iter: int,
    jitter: float = 1e-8,
) -> Tuple[float, np.ndarray]:
    """Run a single-start optimization with a fixed seed and return (score, idx).

    We fix n_start=1 to ensure each worker runs exactly one
    independent start. The returned score is the criterion value (lower is
    better for both "I" and "D").
    """
    idx = _optimal_indices_from_X(
        X_cand,
        n,
        criterion=criterion,
        algo=algo,
        n_start=1,  # force single-start per worker
        max_iter=max_iter,
        random_state=seed,
        jitter=jitter,
    )
    score = _criterion_score(criterion, X_cand, idx, jitter=jitter)
    return score, np.asarray(idx, dtype=int)


# ---------------------------------------------------------------------
# I-optimal design search (serial + optional parallel starts)
# ---------------------------------------------------------------------
def _preallocated_design(
    cand: pd.DataFrame,
    formula: str,
    n: int,
    *,
    criterion: str,
    algo: str,
    n_start: int,
    max_iter: int,
    random_state: Optional[int],
    jitter: float,
    alloc_min_per_cell: int,
    alloc_max_per_cell: Optional[int],
    alloc_wynn_max_iter: int,
    alloc_wynn_tol: float,
    cat_cells_cap: int,
    cat_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Run pre-allocation then per-cell Fedorov exchange.

    Identifies categorical columns (non-numeric dtype) in *cand*, calls
    the Wynn allocation on the candidate set to determine integer run
    counts per cell, then runs a serial ``_optimal_indices_from_X`` search
    within each non-empty cell stratum.

    Allocation counts are honored exactly (SR-6): when a cell's allocation
    exceeds its number of distinct candidate rows, the surplus runs are
    **replicates** — exact optimal designs repeat design points — assigned
    round-robin across the cell's selected rows (exact for single-row cells,
    e.g. pure-categorical spaces, and balanced otherwise).  The returned
    design therefore always has exactly *n* rows, and ``selected_idx`` may
    contain repeated indices.

    Falls back to the standard single-pool search when no categorical columns
    are present in *cand*, or when *n* is too small to give every cell its
    ``alloc_min_per_cell`` minimum (the plain search then selects a subset of
    cells, which keeps small-n probes of an n-search feasible).
    """
    # Categorical columns: prefer the caller-supplied factor-spec metadata —
    # dtype inference misclassifies numeric-coded categories such as
    # {"g": [0, 1, 2]} as continuous, bypassing allocation/replication
    # entirely (SR-28). Fall back to dtype detection for direct callers.
    if cat_cols is not None:
        cat_cols = [c for c in cat_cols if c in cand.columns]
    else:
        cat_cols = [
            c for c in cand.columns
            if not pd.api.types.is_numeric_dtype(cand[c])
        ]

    if not cat_cols:
        # No categorical columns — fall back to normal search
        X_cand, p_names = build_model_matrix(formula, cand)
        selected_idx = _optimal_indices_from_X(
            X_cand, n,
            criterion=criterion, algo=algo, n_start=n_start,
            max_iter=max_iter, random_state=random_state, jitter=jitter,
        )
        design_df = cand.iloc[selected_idx].reset_index(drop=True)
        return design_df, selected_idx, p_names

    # Enumerate unique categorical cells present in the candidate set
    cell_df = cand[cat_cols].drop_duplicates().reset_index(drop=True)
    k = len(cell_df)

    if k > cat_cells_cap:
        raise ValueError(
            f"Categorical cell count ({k}) in the candidate set exceeds "
            f"cat_cells_cap ({cat_cells_cap}). Reduce categorical levels or "
            "raise DesignOptions.cat_cells_cap."
        )

    # Too few runs to give every cell its minimum: allocation is infeasible,
    # but a design selecting a subset of cells still is. Fall back to the
    # plain search so small-n probes (e.g. during the n-search bisection)
    # work instead of raising from the allocation feasibility guard.
    if n < k * max(1, alloc_min_per_cell) and n <= len(cand):
        X_cand, p_names = build_model_matrix(formula, cand)
        selected_idx = _optimal_indices_from_X(
            X_cand, n,
            criterion=criterion, algo=algo, n_start=n_start,
            max_iter=max_iter, random_state=random_state, jitter=jitter,
        )
        design_df = cand.iloc[selected_idx].reset_index(drop=True)
        return design_df, selected_idx, p_names

    # Represent each cell by its centroid (midpoint of any continuous columns)
    cont_cols = [c for c in cand.columns if pd.api.types.is_numeric_dtype(cand[c])]
    rep_rows = []
    for _, row in cell_df.iterrows():
        rep_row = dict(row)
        for cc in cont_cols:
            rep_row[cc] = float(cand[cc].mean())
        rep_rows.append(rep_row)
    rep_df = pd.DataFrame(rep_rows, columns=list(cand.columns))

    try:
        X_cells, _ = build_model_matrix(formula, rep_df)
    except Exception as e:
        raise ValueError(
            f"Pre-allocation: failed to build model matrix for representative "
            f"cell points. Original error: {e}"
        ) from e

    # Import allocation helpers here (avoids circular import at module level)
    from .allocation import _wynn_multiplicative_I, _round_allocation  # noqa: PLC0415

    weights = _wynn_multiplicative_I(
        X_cells=X_cells,
        jitter=jitter,
        max_iter=alloc_wynn_max_iter,
        tol=alloc_wynn_tol,
    )
    counts = _round_allocation(
        weights=weights,
        n=n,
        min_per_cell=alloc_min_per_cell,
        max_per_cell=alloc_max_per_cell,
    )

    # Build model matrix for the full candidate set (needed for p_names)
    X_cand_full, p_names = build_model_matrix(formula, cand)

    # Per-cell Fedorov exchange
    all_global_idx: List[np.ndarray] = []
    cell_random_state = int(0 if random_state is None else random_state)

    for ci in range(k):
        n_cell = int(counts[ci])
        if n_cell == 0:
            continue

        # Build boolean mask for this cell
        mask = np.ones(len(cand), dtype=bool)
        for col in cat_cols:
            mask &= (cand[col].values == cell_df.iloc[ci][col])
        cell_positions = np.where(mask)[0]  # positional indices into cand

        if len(cell_positions) == 0:
            continue

        n_avail = len(cell_positions)
        n_distinct = min(n_cell, n_avail)

        if n_distinct == n_avail:
            # Every candidate row in the cell is used — no search needed.
            local_idx = np.arange(n_avail)
        else:
            # Extract per-cell model matrix (using positional index slice)
            X_cell = X_cand_full[cell_positions, :]
            local_idx = _optimal_indices_from_X(
                X_cell, n_distinct,
                criterion=criterion, algo=algo, n_start=n_start,
                max_iter=max_iter, random_state=cell_random_state, jitter=jitter,
            )

        chosen = cell_positions[local_idx]
        if n_cell > n_distinct:
            # The allocation exceeds the cell's distinct candidate rows: the
            # surplus runs are replicates (exact optimal designs repeat
            # design points). Round-robin over the selected rows — exact for
            # single-row cells, balanced otherwise (SR-6; formerly a silent
            # clamp that returned fewer rows than requested).
            chosen = np.resize(chosen, n_cell)
        all_global_idx.append(chosen)
        cell_random_state += 1337  # decorrelate seeds between cells

    if not all_global_idx:
        raise RuntimeError(
            "Pre-allocation produced no design points. "
            "All categorical cells were empty after candidate filtering."
        )

    selected_idx = np.concatenate(all_global_idx)
    design_df = cand.iloc[selected_idx].reset_index(drop=True)
    return design_df, selected_idx, p_names


def build_i_opt_design_with_idx(
    cand: pd.DataFrame,
    formula: str,
    n: int,
    criterion: str = "I",
    n_start: int = 5,
    algo: str = "fedorov",
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    workers: Optional[int] = None,
    parallel_seed_stride: int = 10_000,
    memory_limit_gb: float = 1.0,
    jitter: float = 1e-8,
    preallocate_categorical: bool = False,
    alloc_min_per_cell: int = 1,
    alloc_max_per_cell: Optional[int] = None,
    alloc_wynn_max_iter: int = 500,
    alloc_wynn_tol: float = 1e-6,
    cat_cells_cap: int = 10_000,
    cat_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Build an I-optimal design and also return selected row indices.

    If `workers` >= 2, perform parallelized random starts:
      - Launch `starts` single-start trials in parallel processes (n_start=1 each)
      - Score each result by I-criterion over the candidate region
      - Return the best-scoring index set

    Otherwise (workers is None/0/1), run the standard serial search with
    `n_start=starts` inside the Fedorov exchange.

    Parameters
    ----------
    cand : DataFrame
        Candidate set.
    formula : str
        Patsy model formula used to build X_cand.
    n : int
        Target number of design runs.
    criterion : str, default 'I'
        Optimality criterion for design search.
    n_start : int, default 5
        Total number of random starts to attempt (serial mode).
        In parallel mode, this is the total number of single-start trials.
    algo : {'fedorov', 'coordinate'}, default 'fedorov'
        Algorithm for optimal design search.
    max_iter : int, default 1000
        Max iterations for design search.
    random_state : int, optional
        Base random seed for reproducibility across starts.
    workers : int, optional
        Number of parallel workers (processes). If None or <=1, runs serially.
    parallel_seed_stride : int, default 10000
        Offset between per-start seeds to minimize correlation across workers.
    memory_limit_gb : float, default 1.0
        Warn if the candidate model matrix X_cand is estimated to exceed this size.
    jitter : float, default 1e-8
        Diagonal ridge added to X'X for numerical stability in the Fedorov
        exchange.  Passed through from ``DesignOptions.xtx_jitter``.
    preallocate_categorical : bool, default False
        If True and *cand* contains non-numeric (categorical) columns, run the
        Wynn multiplicative pre-allocation step before the Fedorov exchange.
        Each categorical cell receives an integer run count, then an
        independent Fedorov search is run within that cell stratum.
    alloc_min_per_cell : int, default 1
        Minimum runs assigned to each occupied categorical cell.
    alloc_max_per_cell : int or None, default None
        Maximum runs per cell (``None`` = unconstrained).
    alloc_wynn_max_iter : int, default 500
        Maximum iterations for the Wynn multiplicative update.
    alloc_wynn_tol : float, default 1e-6
        Convergence tolerance for the Wynn algorithm.
    cat_cells_cap : int, default 10 000
        Maximum number of categorical cells; raises if exceeded.
    cat_cols : list of str, optional
        Names of the categorical factor columns, taken from the original
        factor specification (SR-28). When omitted, categorical columns are
        inferred from non-numeric dtypes — which misclassifies numeric-coded
        categories such as ``[0, 1, 2]`` levels as continuous.

    Returns
    -------
    (design_df, selected_idx, p_names)
        design_df : DataFrame of length n
        selected_idx : np.ndarray[int] indices into cand
        p_names : list[str] of parameter names from the model matrix
    """
    if len(cand) == 0:
        raise ValueError("Candidate set 'cand' is empty.")

    # Pre-allocation path — delegates entirely to _preallocated_design
    if preallocate_categorical:
        return _preallocated_design(
            cand=cand, formula=formula, n=n,
            criterion=criterion, algo=algo, n_start=n_start,
            max_iter=max_iter, random_state=random_state, jitter=jitter,
            alloc_min_per_cell=alloc_min_per_cell,
            alloc_max_per_cell=alloc_max_per_cell,
            alloc_wynn_max_iter=alloc_wynn_max_iter,
            alloc_wynn_tol=alloc_wynn_tol,
            cat_cells_cap=cat_cells_cap,
            cat_cols=cat_cols,
        )

    p_names: List[str] = []
    try:
        # Rough pre-build column count from a head-sample, used only for the
        # memory heuristic below. This can undercount categorical models (the
        # head rows may omit some levels); the full candidate set built below
        # is authoritative, so we reconcile silently rather than warning (P2).
        sample_size = max(5, min(len(cand), 50))
        X_sample, p_names = build_model_matrix(formula, cand.head(sample_size))
        p = X_sample.shape[1]
    except Exception as e:
        raise ValueError(
            f"Failed to build sample model matrix from formula='{formula}'. "
            f"Check formula and factor levels. Original error: {e}"
        ) from e

    n_cand = len(cand)
    if n > n_cand:
        _has_cat = bool(cat_cols) or any(
            not pd.api.types.is_numeric_dtype(cand[c]) for c in cand.columns
        )
        _hint = (
            " For categorical factor spaces, set preallocate_categorical=True "
            "to allow replicated runs (n may then exceed the number of "
            "distinct cells)."
            if _has_cat
            else ""
        )
        raise ValueError(
            f"Requested design size n={n} exceeds the candidate set size "
            f"n_cand={n_cand}. Increase candidate_points (or disable "
            "auto_candidate and set a larger value) to generate a bigger "
            f"candidate pool, or reduce the target sample size.{_hint}"
        )
    estimated_bytes = n_cand * p * 8  # 8 bytes per float64
    limit_bytes = memory_limit_gb * (1024**3)

    if estimated_bytes > limit_bytes:
        warnings.warn(
            f"Candidate model matrix X_cand (shape ~({n_cand}, {p})) is estimated "
            f"to require {estimated_bytes / (1024**3):.2f}GB, "
            f"exceeding the limit of {memory_limit_gb}GB. "
            "This may cause memory errors. Consider reducing candidate_points "
            "or simplifying the model formula.",
            ResourceWarning,
        )

    # Build cached model matrix for candidates once — this is the authoritative
    # column count. The head-sample p above may be lower for categorical models
    # (a subset can omit levels); adopt the full count silently. A genuine loss
    # of levels from the whole candidate set is surfaced upstream in
    # find_optimal_design against the full categorical-level preview (P2).
    X_cand, p_names_cand = build_model_matrix(formula, cand)

    if p != X_cand.shape[1] or not p_names:
        p = X_cand.shape[1]
        p_names = p_names_cand

    # Parallelized multi-start path
    if workers is not None and workers > 1 and n_start > 1:
        base = int(0 if random_state is None else random_state)
        seeds = [base + (i + 1) * parallel_seed_stride for i in range(n_start)]

        best_score = np.inf
        best_idx: Optional[np.ndarray] = None

        successful_results: list[Tuple[float, np.ndarray]] = []
        failed_workers = 0

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _one_start_worker,
                    X_cand,
                    n,
                    seed,
                    algo=algo,
                    criterion=criterion,
                    max_iter=max_iter,
                    jitter=jitter,
                )
                for seed in seeds
            ]
            for fut in as_completed(futures):
                try:
                    score, idx = fut.result()
                    successful_results.append((score, idx))
                except Exception as e:
                    warnings.warn(
                        f"A parallel optimization worker failed with error: {e}",
                        RuntimeWarning,
                    )
                    failed_workers += 1

        if not successful_results:
            raise RuntimeError(
                f"All {n_start} parallel optimization workers failed. "
                "Unable to find an optimal design. "
                "Check worker errors above for details."
            )

        for score, idx in successful_results:
            if score < best_score:
                best_score = score
                best_idx = idx

        if failed_workers > 0:
            warnings.warn(
                f"{failed_workers}/{n_start} optimization workers failed. "
                f"The returned design is the best from the "
                f"{len(successful_results)} successful runs.",
                RuntimeWarning,
            )

        if best_idx is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "Parallel optimization completed but no best index was found."
            )

        design_df = cand.iloc[best_idx].reset_index(drop=True)
        return design_df, best_idx, p_names

    # Serial path: run multi-start Fedorov exchange internally
    selected_idx = _optimal_indices_from_X(
        X_cand,
        n,
        criterion=criterion,
        algo=algo,
        n_start=n_start,
        max_iter=max_iter,
        random_state=random_state,
        jitter=jitter,
    )
    design_df = cand.iloc[selected_idx].reset_index(drop=True)
    return design_df, selected_idx, p_names


def build_i_opt_design(
    cand: pd.DataFrame,
    formula: str,
    n: int,
    criterion: str = "I",
    n_start: int = 5,
    algo: str = "fedorov",
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    workers: Optional[int] = None,
    memory_limit_gb: float = 1.0,
    jitter: float = 1e-8,
) -> pd.DataFrame:
    """Build an I-optimal design from a candidate set.

    Parameters
    ----------
    cand : DataFrame
        Candidate set.
    formula : str
        Patsy model formula.
    n : int
        Target number of design runs.
    criterion : str, default 'I'
        Optimality criterion for design search.
    n_start : int, default 5
        Number of random starts to avoid local optima (serial), or total
        single-start trials in parallel mode.
    algo : {'fedorov', 'coordinate'}, default 'fedorov'
        Algorithm for optimal design search.
    max_iter : int, default 1000
        Max iterations for design search.
    random_state : int, optional
        Random seed.
    workers : int, optional
        Number of parallel workers (processes). If None or <=1, runs serially.
    memory_limit_gb : float, default 1.0
        Warn if the candidate model matrix X_cand is estimated to exceed this size.
    jitter : float, default 1e-8
        Diagonal ridge added to X'X for numerical stability in the Fedorov
        exchange.  Passed through from ``DesignOptions.xtx_jitter``.

    Returns
    -------
    DataFrame
        Selected design points (n rows).
    """
    design_df, _idx, _p_names = build_i_opt_design_with_idx(
        cand=cand,
        formula=formula,
        n=n,
        criterion=criterion,
        n_start=n_start,
        algo=algo,
        max_iter=max_iter,
        random_state=random_state,
        workers=workers,
        memory_limit_gb=memory_limit_gb,
        jitter=jitter,
    )
    return design_df


def augment_design(
    design_df: pd.DataFrame,
    m: int,
    formula: str,
    factors: FactorSpec,
    design_opts: Optional[DesignOptions] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Augment an existing design by greedily adding *m* new runs.

    Fixes the rows from *design_df* in place and iteratively selects the
    single candidate row that most improves the chosen optimality criterion.
    No new I-optimal search is performed — the augmentation is a greedy
    one-point-at-a-time exchange that is fast relative to a full design rebuild.

    Common use cases
    ----------------
    - Preliminary data already exists and additional runs must be added
      sequentially without discarding existing observations.
    - A design was built at a pilot sample size and needs top-up runs
      because the achieved power was slightly below target.

    Parameters
    ----------
    design_df : DataFrame
        Existing design rows (n_existing runs).
    m : int
        Number of new runs to add (must be ≥ 1).
    formula : str
        Patsy formula matching the original design.
    factors : dict
        Factor specifications (continuous tuples, categorical lists).
    design_opts : DesignOptions, optional
        Controls candidate generation and the optimality criterion.
        Defaults to ``DesignOptions()``.

    Returns
    -------
    (augmented_df, new_runs_df)
        augmented_df : DataFrame with ``n_existing + m`` rows.
        new_runs_df  : DataFrame with only the ``m`` newly added rows.

    Notes
    -----
    The greedy step uses the same criterion as specified in
    ``design_opts.criterion`` (``"I"``, ``"D"``, or ``"A"``).

    For ``criterion="I"``, the candidate moment matrix
    ``Mcand = X_cand'X_cand`` is computed once and reused across iterations.
    For ``"D"`` and ``"A"``, only X'X is required, so each iteration is
    O(N_cand × p²).

    The new candidate set is drawn fresh (same parameter settings as a
    normal design call), so augmented runs explore the full factor space
    rather than being confined to the original design's candidate set.
    """
    if design_opts is None:
        design_opts = DesignOptions()

    if m <= 0:
        raise ValueError(f"m must be a positive integer; got {m!r}.")
    if len(design_df) == 0:
        raise ValueError("design_df is empty; there are no existing rows to augment.")

    # Resolve discriminated factor-spec dict forms before any factor use (UX-5).
    from .utils import normalize_factors

    factors = normalize_factors(factors, formula)

    criterion = design_opts.criterion
    jitter = design_opts.xtx_jitter

    # Build a fresh candidate set — sized and generated by the shared helper,
    # the single definition of the candidate set a run with these design_opts
    # selects from (UX-48).
    cand, candidate_points = build_search_candidate(formula, factors, design_opts)
    X_cand, _ = build_model_matrix(formula, cand)
    N_cand = X_cand.shape[0]

    # Pre-compute candidate moment matrix once (used by I-criterion only)
    Mcand = X_cand.T @ X_cand if criterion == "I" else None

    # Initialise X_current from the existing (fixed) design
    X_current, _ = build_model_matrix(formula, design_df)

    new_rows: List[pd.DataFrame] = []
    for _step in range(m):
        best_score = np.inf
        best_j = -1
        for j in range(N_cand):
            X_try = np.vstack([X_current, X_cand[j : j + 1, :]])
            score = _score_design(criterion, X_try, Mcand=Mcand, N_cand=N_cand, jitter=jitter)
            if score < best_score:
                best_score = score
                best_j = j
        if best_j < 0:  # pragma: no cover — degenerate edge case
            break
        X_current = np.vstack([X_current, X_cand[best_j : best_j + 1, :]])
        new_rows.append(cand.iloc[[best_j]])

    new_runs_df = (
        pd.concat(new_rows, ignore_index=True) if new_rows
        else pd.DataFrame(columns=design_df.columns)
    )
    augmented_df = pd.concat(
        [design_df.reset_index(drop=True), new_runs_df], ignore_index=True
    )
    return augmented_df, new_runs_df


# ---------------------------------------------------------------------
# Split-plot Fedorov exchange algorithm (SP-5)
# ---------------------------------------------------------------------

def build_split_plot_design(
    cand: pd.DataFrame,
    formula: str,
    n_wp: int,
    subplots_per_wp: int,
    htc_factors: List[str],
    eta: float,
    *,
    factors: Optional[FactorSpec] = None,
    criterion: str = "I",
    starts: int = 5,
    max_iter: int = 100,
    random_state: Optional[int] = None,
    jitter: float = 1e-8,
    criterion_ignore_vr: bool = False,
    n_wp_cand: int = 30,
    n_sp_cand: int = 50,
    constraint_func: Optional[callable] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a split-plot optimal design using a two-phase Fedorov exchange.

    Implements a modified Fedorov point-exchange that respects the whole-plot
    nesting constraint.  HTC (hard-to-change) factor settings are constant
    within each whole plot; ETC (easy-to-change) settings vary across sub-plots.
    The GLS information matrix ``M = X'V⁻¹X`` is used for the criterion, with
    ``V_inv`` computed once from the balanced layout and reused throughout.

    Parameters
    ----------
    cand : DataFrame
        Initial candidate structure (e.g. from ``build_split_plot_candidate``).
        Must include all factor columns and a ``__wp_id__`` column.  Used to
        determine factor names and column ordering.
    formula : str
        Patsy formula for the model (references HTC and/or ETC factors).
    n_wp : int
        Number of whole plots (≥ 2).
    subplots_per_wp : int
        Sub-plots per whole plot (balanced layout assumed).
    htc_factors : list of str
        Names of the hard-to-change (whole-plot) factors.
    eta : float
        Variance ratio σ²_wp / σ²_sp.  ``eta=0`` is equivalent to OLS.
    factors : dict, optional
        Full factor specifications ``{name: (low, high) or [levels]}``.
        When provided, WP and SP candidate pools are generated from this dict
        (sizes controlled by *n_wp_cand* and *n_sp_cand*).  When ``None``,
        candidate pools are derived from the unique rows in *cand*.
    criterion : {"I", "D", "A"}, default "I"
        GLS optimality criterion; lower is better in all cases.
    starts : int, default 5
        Number of independent random starts.  The start with the best
        criterion score is returned.
    max_iter : int, default 100
        Maximum exchange iterations per start.  Each iteration runs one
        complete WP-swap phase followed by one SP-swap phase.
    random_state : int, optional
        Base random seed for reproducibility.
    jitter : float, default 1e-8
        Diagonal ridge added to ``X'V⁻¹X`` for numerical stability.
    criterion_ignore_vr : bool, default False
        If ``True``, set ``V_inv = I`` (OLS criterion) regardless of *eta*.
        Useful for comparing GLS and OLS on the same nested structure.
    n_wp_cand : int, default 30
        WP candidate pool size when *factors* is provided.
    n_sp_cand : int, default 50
        SP candidate pool size when *factors* is provided.

    Returns
    -------
    design_df : DataFrame
        Optimal split-plot design with ``n_wp * subplots_per_wp`` rows.
        All factor columns and ``__wp_id__`` are present.  All rows sharing
        the same ``__wp_id__`` have identical HTC factor values.
    X : ndarray, shape (n_total, p)
        Ordinary (unweighted) model matrix for the returned design.
    """
    n_total = n_wp * subplots_per_wp
    htc_set = set(htc_factors)

    # Factor columns in original order (excludes __wp_id__)
    factor_cols = [c for c in cand.columns if c != "__wp_id__"]
    etc_factors = [c for c in factor_cols if c not in htc_set]

    # V_inv: computed once — depends only on layout and eta, not factor values
    Z = build_whole_plot_indicator(n_total, n_wp, subplots_per_wp)
    if criterion_ignore_vr or eta == 0.0:
        V_inv = np.eye(n_total, dtype=np.float64)
    else:
        V_inv = build_split_plot_covariance_inv(Z, eta)

    # Balanced layout: run r belongs to WP slot r // subplots_per_wp
    wp_slot_of_run = np.repeat(np.arange(n_wp), subplots_per_wp)

    # --- Candidate pools ---
    seed_base = int(random_state) if random_state is not None else 0

    # Wrap constraint_func to gracefully skip evaluation when a row is missing
    # columns referenced by the constraint (e.g. a cross-stratum constraint
    # evaluated on an HTC-only or ETC-only pool row).  Missing columns produce
    # a KeyError inside the user function; we treat those rows as feasible here
    # because the authoritative gate is the full-row feasibility mask applied
    # to every WP × SP combination below (SR-5).
    def _pool_constraint(row: "pd.Series") -> bool:
        try:
            return bool(constraint_func(row))
        except (KeyError, TypeError, ValueError, NameError):
            # Cross-stratum constraints reference columns absent from this pool;
            # treat as feasible — the combined-row mask below enforces them.
            return True

    pool_cfunc = _pool_constraint if constraint_func is not None else None

    if factors is not None:
        htc_factor_dict = {k: v for k, v in factors.items() if k in htc_set}
        etc_factor_dict = {k: v for k, v in factors.items() if k not in htc_set}
        wp_pool_df = build_candidate(
            htc_factor_dict, candidate_points=n_wp_cand, seed=seed_base,
            constraint_func=pool_cfunc,
        )
        sp_pool_df = (
            build_candidate(
                etc_factor_dict, candidate_points=n_sp_cand, seed=seed_base + 1,
                constraint_func=pool_cfunc,
            )
            if etc_factor_dict
            else pd.DataFrame()
        )
    else:
        wp_pool_df = cand[htc_factors].drop_duplicates().reset_index(drop=True)
        sp_pool_df = (
            cand[etc_factors].drop_duplicates().reset_index(drop=True)
            if etc_factors
            else pd.DataFrame()
        )

    has_etc = bool(etc_factors and len(sp_pool_df) > 0)
    n_wp_pool = len(wp_pool_df)
    n_sp_pool = len(sp_pool_df) if has_etc else 1

    if n_wp_pool == 0:
        raise ValueError(
            "WP candidate pool is empty.  Provide a non-empty factors dict or "
            "a cand DataFrame with at least one distinct HTC factor setting."
        )

    # --- Pre-build model matrix for all (WP_cand × SP_cand) combinations ---
    # X_combo_3d[k, m] = model matrix row for (wp_pool[k], sp_pool[m])
    # We compute all n_wp_pool × n_sp_pool rows in ONE patsy call.
    combo_rows: List[Dict] = []
    for k in range(n_wp_pool):
        wp_row = wp_pool_df.iloc[k]
        for m in range(n_sp_pool):
            row: Dict[str, Any] = dict(wp_row)
            if has_etc:
                row.update(dict(sp_pool_df.iloc[m]))
            combo_rows.append(row)

    combo_df = pd.DataFrame(combo_rows, columns=factor_cols)
    X_combo, p_names = build_model_matrix(formula, combo_df)
    p = X_combo.shape[1]
    # Reshape: C-order → X_combo_3d[k, m] == X_combo[k * n_sp_pool + m]
    X_combo_3d = X_combo.reshape(n_wp_pool, n_sp_pool, p)

    # --- Full-row feasibility mask over WP × SP combinations (SR-5) ---
    # The stratum pools above can only enforce same-stratum constraints; a
    # cross-stratum constraint (e.g. "H + E <= 1") must be checked on the
    # combined row. Every combination the exchange can propose is a row of
    # combo_df, so masking here guarantees the returned design is feasible.
    if constraint_func is not None:
        _feas_flat = combo_df.apply(
            lambda _row: bool(constraint_func(_row)), axis=1
        ).to_numpy()
        feasible = _feas_flat.reshape(n_wp_pool, n_sp_pool)
        wp_feasible = feasible.any(axis=1)  # WP candidates with ≥1 feasible SP
        if not wp_feasible.any():
            raise ValueError(
                "constraint_func eliminates every WP × SP combination in the "
                "split-plot candidate pools; the constrained design space is "
                "empty. Relax the constraint or enlarge the candidate pools "
                "(n_wp_cand / n_sp_cand)."
            )
    else:
        feasible = np.ones((n_wp_pool, n_sp_pool), dtype=bool)
        wp_feasible = np.ones(n_wp_pool, dtype=bool)
    feasible_wp_idx = np.flatnonzero(wp_feasible)
    feasible_sp_for_wp = [np.flatnonzero(feasible[k]) for k in range(n_wp_pool)]

    # Candidate moment matrix for I-criterion: feasible combinations only,
    # so the I-criterion averages prediction variance over the reachable
    # design region rather than over infeasible points.
    _feas_rows = feasible.reshape(-1)
    Mcand = X_combo[_feas_rows].T @ X_combo[_feas_rows]
    N_cand_total = int(_feas_rows.sum())

    # --- Multi-start two-phase exchange ---
    rng = np.random.default_rng(random_state)
    best_score = np.inf
    best_design_df: Optional[pd.DataFrame] = None
    best_X: Optional[np.ndarray] = None

    for _start in range(max(1, starts)):
        # Random initialisation: draw WP and SP pool indices from the
        # feasible combinations only (each run's SP index must be feasible
        # for its whole plot's WP candidate).
        wp_slots = feasible_wp_idx[
            rng.integers(0, len(feasible_wp_idx), size=n_wp)
        ]
        sp_runs = np.zeros(n_total, dtype=np.intp)
        if has_etc:
            for r in range(n_total):
                _opts = feasible_sp_for_wp[int(wp_slots[wp_slot_of_run[r]])]
                sp_runs[r] = int(_opts[rng.integers(0, len(_opts))])

        # Initial model matrix from pool indices
        X_current = np.vstack([
            X_combo_3d[wp_slots[wp_slot_of_run[r]], sp_runs[r]]
            for r in range(n_total)
        ])
        current_score = _score_design_gls(
            criterion, X_current, V_inv,
            Mcand=Mcand, N_cand=N_cand_total, jitter=jitter,
        )

        # Alternating WP / SP exchange until convergence or max_iter
        for _iter in range(max_iter):
            improved = False

            # --- Phase 1: WP swaps ---
            # For each WP slot, find the best WP candidate and accept if improving.
            for i in range(n_wp):
                wp_rows = np.where(wp_slot_of_run == i)[0]
                best_k = int(wp_slots[i])
                best_k_score = current_score

                for k in range(n_wp_pool):
                    if k == wp_slots[i]:
                        continue
                    # Feasibility: candidate k must be compatible with every
                    # current SP assignment in this whole plot.
                    if not feasible[k, sp_runs[wp_rows]].all():
                        continue
                    # Propose: replace all sub-plots in WP i with WP candidate k
                    X_prop = X_current.copy()
                    for r in wp_rows:
                        X_prop[r] = X_combo_3d[k, sp_runs[r]]
                    score_prop = _score_design_gls(
                        criterion, X_prop, V_inv,
                        Mcand=Mcand, N_cand=N_cand_total, jitter=jitter,
                    )
                    if score_prop < best_k_score - 1e-10:
                        best_k = k
                        best_k_score = score_prop

                if best_k != wp_slots[i]:
                    for r in wp_rows:
                        X_current[r] = X_combo_3d[best_k, sp_runs[r]]
                    wp_slots[i] = best_k
                    current_score = best_k_score
                    improved = True

            # --- Phase 2: SP swaps ---
            # For each run, find the best SP candidate and accept if improving.
            if has_etc:
                for r in range(n_total):
                    wp_i = int(wp_slot_of_run[r])
                    best_m = int(sp_runs[r])
                    best_m_score = current_score

                    for m in feasible_sp_for_wp[int(wp_slots[wp_i])]:
                        if m == sp_runs[r]:
                            continue
                        X_prop = X_current.copy()
                        X_prop[r] = X_combo_3d[wp_slots[wp_i], m]
                        score_prop = _score_design_gls(
                            criterion, X_prop, V_inv,
                            Mcand=Mcand, N_cand=N_cand_total, jitter=jitter,
                        )
                        if score_prop < best_m_score - 1e-10:
                            best_m = m
                            best_m_score = score_prop

                    if best_m != sp_runs[r]:
                        X_current[r] = X_combo_3d[wp_slots[wp_i], best_m]
                        sp_runs[r] = best_m
                        current_score = best_m_score
                        improved = True

            if not improved:
                break  # local optimum reached — converged

        # Record best across all starts
        if current_score < best_score:
            best_score = current_score
            best_X = X_current.copy()
            # Reconstruct design DataFrame from pool indices
            design_rows: List[Dict] = []
            for r in range(n_total):
                i = int(wp_slot_of_run[r])
                row = dict(wp_pool_df.iloc[wp_slots[i]])
                if has_etc:
                    row.update(dict(sp_pool_df.iloc[sp_runs[r]]))
                row["__wp_id__"] = i
                design_rows.append(row)
            best_design_df = pd.DataFrame(design_rows, columns=list(cand.columns))

    if best_design_df is None or best_X is None:  # pragma: no cover
        raise RuntimeError("build_split_plot_design: no valid design found.")

    return best_design_df.reset_index(drop=True), best_X


# =====================================================================
# Compound criterion Fedorov exchange  (MR-5)
# =====================================================================

def compound_i_criterion(
    indices: np.ndarray,
    candidates_list: List[np.ndarray],
    weights: List[float],
    jitter: float = 1e-8,
) -> float:
    """Weighted sum of per-formula I-criterion scores over shared run indices.

    score_k = trace[(X_k' X_k)^-1 M_k] / N_cand
    where M_k = X_cand_k' X_cand_k (pre-computed candidate moment matrix).

    Returns
    -------
    float
        Σ_k (w_k / Σw) · score_k   (lower is better, consistent with I-criterion).
    """
    w_total = sum(weights)
    score = 0.0
    for X_cand_k, w_k in zip(candidates_list, weights):
        score += (w_k / w_total) * _i_criterion_for_indices(X_cand_k, indices, jitter=jitter)
    return score


def _compound_criterion_score(
    criterion: str,
    candidates_list: List[np.ndarray],
    weights: List[float],
    idx: np.ndarray,
    jitter: float = 1e-8,
) -> float:
    """Weighted compound criterion score (I or D); lower is better."""
    w_total = sum(weights)
    score = 0.0
    for X_cand_k, w_k in zip(candidates_list, weights):
        if criterion == "I":
            score += (w_k / w_total) * _i_criterion_for_indices(X_cand_k, idx, jitter=jitter)
        else:  # D
            score += (w_k / w_total) * _d_criterion_for_indices(X_cand_k, idx, jitter=jitter)
    return score


def _compound_fedorov_single(
    candidates_list: List[np.ndarray],
    weights: List[float],
    n: int,
    *,
    criterion: str,
    max_iter: int,
    seed: int,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Single-start Fedorov exchange for compound I/D criterion across multiple formulas.

    Operates on a **shared** index set (same rows for all formulas).  At every
    candidate swap, per-formula gains are weight-summed to obtain a single
    compound gain, enabling the exchange to simultaneously optimise all
    formula-specific I (or D) criteria.

    Parameters
    ----------
    candidates_list : list of ndarray (n_cand, p_k)
        One pre-built model matrix per formula; all must have the same row
        count n_cand.
    weights : list of float
        Per-formula weights (need not sum to 1; normalised internally).
    n : int
        Number of design rows to select.
    criterion : {"I", "D"}
        Compound optimality criterion.  "A" is not supported.
    max_iter : int
        Maximum exchange-iteration rounds.
    seed : int
        Random seed for the initial random selection.
    jitter : float
        Diagonal ridge for numerical stability.

    Returns
    -------
    idx : ndarray[int] of shape (n,)
        Row indices into the shared candidate set.
    """
    rng = np.random.default_rng(seed)
    n_cand = candidates_list[0].shape[0]
    w_total = sum(weights)
    w_norm = [w / w_total for w in weights]

    if n > n_cand:
        raise ValueError(f"n={n} exceeds candidate set size n_cand={n_cand}.")

    idx = rng.choice(n_cand, size=n, replace=False)
    in_design = np.zeros(n_cand, dtype=bool)
    in_design[idx] = True

    # Precompute candidate moment matrices (I-criterion only; unchanged across iterations)
    Mcand_list: Optional[List[np.ndarray]] = (
        [X_cand_k.T @ X_cand_k for X_cand_k in candidates_list]
        if criterion == "I" else None
    )

    for _iter in range(max_iter):
        non_idx = np.where(~in_design)[0]
        if len(non_idx) == 0:
            break

        # ------------------------------------------------------------------
        # Precompute per-formula quantities for this iteration
        # ------------------------------------------------------------------
        per_k_data: List[Dict] = []
        skip_iter = False
        for k_idx, (X_cand_k, w_k) in enumerate(zip(candidates_list, w_norm)):
            p_k = X_cand_k.shape[1]
            X_d_k = X_cand_k[idx]
            M_k = X_d_k.T @ X_d_k + jitter * np.eye(p_k)
            try:
                M_inv_k = np.linalg.inv(M_k)
            except np.linalg.LinAlgError:
                skip_iter = True
                break

            H_k = M_inv_k @ X_cand_k.T          # p_k × n_cand
            leverages_k = np.einsum("pt,pt->t", X_cand_k.T, H_k)   # (n_cand,)
            X_non_k = X_cand_k[non_idx]          # n_non × p_k
            H_non_k = H_k[:, non_idx]            # p_k × n_non
            lev_non_k = leverages_k[non_idx]     # (n_non,)

            if criterion == "I":
                Mcand_k = Mcand_list[k_idx]
                current_score_k = float(np.trace(M_inv_k @ Mcand_k))
            else:  # D
                sign_k, logdet_k = np.linalg.slogdet(M_k)
                current_score_k = -float(logdet_k) if sign_k > 0 else float("inf")

            per_k_data.append({
                "H_k": H_k,
                "leverages_k": leverages_k,
                "X_non_k": X_non_k,
                "H_non_k": H_non_k,
                "lev_non_k": lev_non_k,
                "current_score_k": current_score_k,
                "Mcand_k": Mcand_list[k_idx] if criterion == "I" else None,
                "w_k": w_k,
            })

        if skip_iter:
            break  # singular formula matrix — cannot safely continue

        best_gain = 0.0
        best_s_pos = -1
        best_t_local = -1

        for s_pos in range(len(idx)):
            s = idx[s_pos]
            compound_gains = np.zeros(len(non_idx))
            denom_bad = False

            for pk in per_k_data:
                d_s_k = float(pk["leverages_k"][s])
                h_s_k = pk["H_k"][:, s]           # (p_k,)
                denom_s_k = 1.0 - d_s_k
                if denom_s_k < 1e-10:
                    denom_bad = True
                    break

                w_s_all_k = pk["X_non_k"] @ h_s_k  # (n_non,)

                if criterion == "D":
                    v_t_prime_k = pk["lev_non_k"] + w_s_all_k * w_s_all_k / denom_s_k
                    # Exact per-formula improvement of −logdet is log(det ratio).
                    # (ratio − 1 is only a first-order surrogate: it is a valid
                    # monotone transform for a SINGLE formula, but weight-summed
                    # across formulas it can disagree in sign with the true
                    # compound delta, accepting score-worsening swaps and
                    # oscillating forever — SR-12.)
                    det_ratio_k = denom_s_k * (1.0 + v_t_prime_k)
                    gains_k = np.log(np.maximum(det_ratio_k, 1e-300))
                else:  # I
                    Mcand_hs_k = pk["Mcand_k"] @ h_s_k                               # (p_k,)
                    trace_I_minus_k = (
                        pk["current_score_k"] + float(h_s_k @ Mcand_hs_k) / denom_s_k
                    )
                    mp_inv_xnon_k = (
                        pk["H_non_k"] + np.outer(h_s_k, w_s_all_k) / denom_s_k
                    )                                                                  # p_k × n_non
                    d_t_prime_k = np.einsum(
                        "pt,pt->t", pk["X_non_k"].T, mp_inv_xnon_k
                    )                                                                  # (n_non,)
                    Mcand_mp_k = pk["Mcand_k"] @ mp_inv_xnon_k                        # p_k × n_non
                    delta_I_k = (
                        np.einsum("pt,pt->t", mp_inv_xnon_k, Mcand_mp_k)
                        / (1.0 + d_t_prime_k)
                    )                                                                  # (n_non,)
                    gains_k = pk["current_score_k"] - (trace_I_minus_k - delta_I_k)

                compound_gains += pk["w_k"] * gains_k

            if denom_bad:
                continue  # skip near-degenerate design point

            local_best = int(np.argmax(compound_gains))
            if compound_gains[local_best] > best_gain:
                best_gain = float(compound_gains[local_best])
                best_s_pos = s_pos
                best_t_local = local_best

        if best_s_pos == -1:
            break  # converged — no improving swap found

        old_pt = idx[best_s_pos]
        new_pt = non_idx[best_t_local]
        in_design[old_pt] = False
        in_design[new_pt] = True
        idx[best_s_pos] = new_pt

    return idx


def build_compound_design(
    candidates_list: List[np.ndarray],
    weights: List[float],
    n: int,
    *,
    criterion: str = "I",
    n_start: int = 5,
    max_iter: int = 100,
    random_state: Optional[int] = None,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Multi-start compound Fedorov exchange for responses with different formulas.

    Runs ``n_start`` independent random starts of :func:`_compound_fedorov_single`
    and returns the index set with the best compound criterion score.

    Parameters
    ----------
    candidates_list : list of ndarray (n_cand, p_k)
        One pre-built model matrix per response formula.  All arrays must
        share the same number of rows (the candidate set size).
    weights : list of float
        Per-formula importance weights (normalised internally).
    n : int
        Number of design rows to select.
    criterion : {"I", "D"}
        Compound optimality criterion.  "A" is not supported.
    n_start : int
        Number of independent random starts.
    max_iter : int
        Maximum Fedorov exchange iterations per start.
    random_state : int or None
        Base random seed; start k uses seed ``base + k * 1337``.
    jitter : float
        Diagonal ridge added to X'X for numerical stability.

    Returns
    -------
    ndarray[int] of shape (n,)
        Row indices into the shared candidate set.
    """
    if criterion == "A":
        raise NotImplementedError("A-compound not supported; use 'I' or 'D'.")

    base = int(random_state) if random_state is not None else 0
    best_score = float("inf")
    best_idx: Optional[np.ndarray] = None

    for k in range(max(1, n_start)):
        seed = base + k * 1337
        idx = _compound_fedorov_single(
            candidates_list, weights, n,
            criterion=criterion, max_iter=max_iter, seed=seed, jitter=jitter,
        )
        score = _compound_criterion_score(
            criterion, candidates_list, weights, idx, jitter=jitter
        )
        if score < best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:  # pragma: no cover
        raise RuntimeError("build_compound_design: no valid design found.")
    return best_idx


__all__ = [
    "build_i_opt_design",
    "build_i_opt_design_with_idx",
    "build_split_plot_design",
    "_score_design",
    "_score_design_gls",
    "_gls_i_criterion",
    "_gls_d_criterion",
    "_gls_a_criterion",
    "augment_design",
    "compound_i_criterion",
    "build_compound_design",
]
