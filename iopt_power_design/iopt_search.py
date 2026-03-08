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

from .candidate import estimate_candidate_size, build_candidate
from .model_matrix import build_model_matrix
from .config import DesignOptions


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


def _criterion_score(
    criterion: str,
    X_cand: np.ndarray,
    idx: np.ndarray,
    jitter: float = 1e-8,
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

    Returns
    -------
    float
        Criterion score; lower is better in all cases.

    Raises
    ------
    ValueError
        If *criterion* is not ``"I"``, ``"D"``, or ``"A"``.
    """
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
    )
    score = _criterion_score(criterion, X_cand, idx, jitter=jitter)
    return score, np.asarray(idx, dtype=int)


# ---------------------------------------------------------------------
# I-optimal design search (serial + optional parallel starts)
# ---------------------------------------------------------------------
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

    Returns
    -------
    (design_df, selected_idx, p_names)
        design_df : DataFrame of length n
        selected_idx : np.ndarray[int] indices into cand
        p_names : list[str] of parameter names from the model matrix
    """
    if len(cand) == 0:
        raise ValueError("Candidate set 'cand' is empty.")

    p_names: List[str] = []
    try:
        # Build matrix from a small but representative sample
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
        raise ValueError(
            f"Requested design size n={n} exceeds the candidate set size "
            f"n_cand={n_cand}. Increase candidate_points (or disable "
            "auto_candidate and set a larger value) to generate a bigger "
            "candidate pool, or reduce the target sample size."
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

    # Build cached model matrix for candidates once
    X_cand, p_names_cand = build_model_matrix(formula, cand)

    if p != X_cand.shape[1]:
        warnings.warn(
            f"Model parameter count mismatch between sample ({p}) and full "
            f"candidate set ({X_cand.shape[1]}). Using full set count.",
            RuntimeWarning
        )
        p = X_cand.shape[1]
        p_names = p_names_cand
    elif not p_names:
        p_names = p_names_cand
        p = X_cand.shape[1]

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
    factors: Dict[str, Any],
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

    criterion = design_opts.criterion
    jitter = design_opts.xtx_jitter

    # Build a fresh candidate set
    if design_opts.auto_candidate:
        candidate_points = estimate_candidate_size(
            formula=formula,
            factors=factors,
            cand_min=design_opts.cand_min,
            cand_max=design_opts.cand_max,
            cat_cells_cap=design_opts.cat_cells_cap,
            per_cell_alpha=design_opts.per_cell_alpha,
            per_cell_min=design_opts.per_cell_min,
            per_cell_max=design_opts.per_cell_max,
            seed=design_opts.random_state,
        )
    else:
        candidate_points = int(design_opts.candidate_points)

    cand = build_candidate(
        factors=factors,
        candidate_points=candidate_points,
        seed=design_opts.random_state,
        constraint_func=design_opts.constraint_func,
        cat_cells_cap=design_opts.cat_cells_cap,
    )
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


__all__ = [
    "build_i_opt_design",
    "build_i_opt_design_with_idx",
    "_score_design",
    "augment_design",
]
