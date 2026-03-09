# allocation.py
# License: MIT
"""
I-optimal categorical pre-allocation
======================================

Implements a Wynn multiplicative algorithm for I-optimal allocation of
experimental runs across categorical cells, with integer rounding and
per-cell lower/upper bounds.

Public API
----------
``i_optimal_allocation(formula, factors, n, design_opts)``
    Return a mapping from each categorical cell (tuple of level values) to
    the integer number of runs allocated to that cell.

Algorithm
---------
Given k categorical cells represented by their model-matrix rows x_1, …, x_k:

1. Start with uniform weights  w_i = 1/k.
2. Iterate the Wynn multiplicative update for I-optimality:

       M(w)  = Σ_i w_i x_i x_i'          (weighted moment matrix)
       A     = (1/k) Σ_i x_i x_i'         (candidate moment matrix)
       φ_i   = x_i' M^{-1} A M^{-1} x_i   (sensitivity function)
       φ_bar = trace(M^{-1} A)             (current I-criterion)
       w_i  ← w_i × φ_i / φ_bar           (multiplicative update)

   Normalise weights after each step.  Stop when the maximum relative
   deviation max_i |φ_i / φ_bar − 1| < tol, or after max_iter steps.

3. Round fractional allocations w_i × n to integers using the Hamilton
   (largest-remainder) method, then clamp to [min_per_cell, max_per_cell].

References
----------
* Wynn (1970) "The Sequential Generation of D-Optimum Experimental Designs."
* Atkinson, Donev & Tobias (2007) "Optimum Experimental Designs, with SAS."
  Chapter 9 on I-optimal and continuous design measures.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import DesignOptions
from .model_matrix import build_model_matrix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_continuous(spec: Any) -> bool:
    """Return True if *spec* is a 2-element numeric tuple/list (continuous factor)."""
    return (
        isinstance(spec, (tuple, list))
        and len(spec) == 2
        and all(isinstance(x, (int, float)) for x in spec)
    )


def _wynn_multiplicative_I(
    X_cells: np.ndarray,
    jitter: float = 1e-8,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> np.ndarray:
    """Wynn multiplicative algorithm for I-optimal continuous design measure.

    Parameters
    ----------
    X_cells : ndarray, shape (k, p)
        Model-matrix rows for each of the k categorical cells.
    jitter : float
        Diagonal ridge added to the moment matrix before inversion.
    max_iter : int
        Maximum number of multiplicative updates.
    tol : float
        Convergence tolerance on max relative sensitivity deviation.

    Returns
    -------
    w : ndarray, shape (k,)
        Normalised optimal weights (sum to 1).
    """
    k, p = X_cells.shape

    if k == 1:
        return np.ones(1)

    # Candidate moment matrix: A = (1/k) X' X
    A = (X_cells.T @ X_cells) / k  # (p, p)

    # Uniform initial weights
    w = np.ones(k) / k

    for _ in range(max_iter):
        # Weighted moment matrix: M = X' diag(w) X
        M = (X_cells * w[:, None]).T @ X_cells + jitter * np.eye(p)

        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            break  # singular — return current weights

        # M^{-1} A M^{-1}
        M_inv_A_M_inv = M_inv @ A @ M_inv  # (p, p)

        # Sensitivity function φ_i = x_i' (M^{-1} A M^{-1}) x_i
        temp = X_cells @ M_inv_A_M_inv   # (k, p)
        phi = np.sum(temp * X_cells, axis=1)  # (k,)  element-wise then sum

        # Current criterion value φ_bar = trace(M^{-1} A)
        phi_bar = float(np.trace(M_inv @ A))
        if phi_bar < 1e-14:
            break

        # Multiplicative update
        w_new = w * phi / phi_bar

        # Normalise
        w_sum = float(w_new.sum())
        if w_sum < 1e-14:
            break
        w_new /= w_sum

        # Convergence: max relative deviation of sensitivity values
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_dev = np.abs(phi / phi_bar - 1.0)
        if np.max(rel_dev) < tol:
            w = w_new
            break

        w = w_new

    return w


def _round_allocation(
    weights: np.ndarray,
    n: int,
    min_per_cell: int = 1,
    max_per_cell: Optional[int] = None,
) -> np.ndarray:
    """Convert continuous weights to integer allocations.

    Uses the Hamilton (largest-remainder) method to distribute exactly n runs
    across k cells while respecting per-cell lower and upper bounds.

    Parameters
    ----------
    weights : ndarray, shape (k,)
        Normalised weights (must sum to 1).
    n : int
        Total number of runs to distribute.
    min_per_cell : int
        Minimum runs per cell.  Cells that the algorithm would leave with 0
        are boosted to *min_per_cell* by stealing from the largest allocations.
    max_per_cell : int or None
        Maximum runs per cell.  ``None`` means unconstrained.

    Returns
    -------
    counts : ndarray[int], shape (k,)
        Integer allocations; counts.sum() == n.

    Raises
    ------
    ValueError
        If bounds are infeasible given *n* and the number of cells *k*.
    """
    k = len(weights)

    # Feasibility guard
    if min_per_cell > 0 and k * min_per_cell > n:
        raise ValueError(
            f"Cannot allocate at least min_per_cell={min_per_cell} run(s) to "
            f"each of {k} cells with only n={n} total runs. "
            f"Lower min_per_cell or increase n."
        )
    if max_per_cell is not None and k * max_per_cell < n:
        raise ValueError(
            f"Cannot distribute n={n} runs with max_per_cell={max_per_cell} "
            f"across {k} cells (maximum possible = {k * max_per_cell}). "
            f"Raise max_per_cell or reduce n."
        )

    # Step 1: Hamilton (largest-remainder) rounding
    f = weights * n
    counts = np.floor(f).astype(int)
    remainders = f - counts
    shortage = n - int(counts.sum())
    if shortage > 0:
        top_idx = np.argsort(-remainders)[:shortage]
        counts[top_idx] += 1

    # Step 2: Enforce min_per_cell (boost deficit cells, steal from surplus)
    if min_per_cell > 0:
        for i in range(k):
            deficit = min_per_cell - counts[i]
            if deficit > 0:
                counts[i] = min_per_cell
                # Steal one run at a time from the cell with the largest allocation
                for _ in range(deficit):
                    donor = int(np.argmax(counts))
                    if counts[donor] > min_per_cell:
                        counts[donor] -= 1

    # Step 3: Enforce max_per_cell (cap excess cells, redistribute)
    if max_per_cell is not None:
        for i in range(k):
            excess = counts[i] - max_per_cell
            if excess > 0:
                counts[i] = max_per_cell
                for _ in range(excess):
                    recipient = int(np.argmin(counts))
                    if counts[recipient] < max_per_cell:
                        counts[recipient] += 1

    return counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def i_optimal_allocation(
    formula: str,
    factors: Dict[str, Any],
    n: int,
    design_opts: Optional[DesignOptions] = None,
) -> Dict[Tuple[Any, ...], int]:
    """Compute I-optimal run allocations across categorical cells.

    Identifies every categorical factor in *factors*, enumerates all
    combinations of their levels (categorical cells), and uses the Wynn
    multiplicative algorithm to find a continuous I-optimal design measure
    over those cells.  The resulting fractional weights are rounded to
    integer run counts that sum to *n*.

    A representative model-matrix row is built for each cell using the
    midpoint of every continuous factor range.

    Parameters
    ----------
    formula : str
        Patsy model formula (same one used by ``i_optimal_powered_design``).
    factors : dict
        Factor specification dict.  Continuous factors are ``(lo, hi)`` tuples
        or two-element numeric lists.  Categorical factors are lists with two
        or more string (or mixed) levels.
    n : int
        Total number of runs to allocate.
    design_opts : DesignOptions, optional
        Provides ``xtx_jitter``, ``alloc_min_per_cell``, ``alloc_max_per_cell``,
        ``alloc_wynn_max_iter``, ``alloc_wynn_tol``, and ``cat_cells_cap``.
        Defaults to ``DesignOptions()`` when *None*.

    Returns
    -------
    dict
        Mapping from ``(level_factor_A, level_factor_B, ...)`` tuples to
        integer run counts.  Only cells with a non-zero allocation are
        included.  The tuple ordering matches the order of categorical factors
        as they appear in *factors*.

    Raises
    ------
    ValueError
        If *factors* contains no categorical factors, if *n* is less than 1,
        or if the per-cell bounds are infeasible for the given *n* and cell
        count.

    Examples
    --------
    >>> from iopt_power_design import DesignOptions
    >>> from iopt_power_design.allocation import i_optimal_allocation
    >>> factors = {
    ...     "Material": ["Steel", "Aluminum", "Titanium"],
    ...     "Temp":     (-10.0, 50.0),
    ...     "Pressure": (1.0,   5.0),
    ... }
    >>> alloc = i_optimal_allocation("1 + Material + Temp + Pressure", factors, n=24)
    >>> sum(alloc.values())
    24
    """
    if design_opts is None:
        design_opts = DesignOptions()

    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}.")

    # Separate categorical and continuous factors
    cat_names: List[str] = [k for k, v in factors.items() if not _is_continuous(v)]
    cont_names: List[str] = [k for k, v in factors.items() if _is_continuous(v)]

    if not cat_names:
        raise ValueError(
            "i_optimal_allocation requires at least one categorical factor. "
            "All factors in this specification are continuous."
        )

    # Enumerate categorical cells (Cartesian product of levels)
    cat_levels = [factors[k] for k in cat_names]
    total_cells = 1
    for lv in cat_levels:
        total_cells *= len(lv)

    if total_cells > design_opts.cat_cells_cap:
        raise ValueError(
            f"Categorical cell count ({total_cells}) exceeds cat_cells_cap "
            f"({design_opts.cat_cells_cap}). Reduce the number of categorical "
            "levels or raise DesignOptions.cat_cells_cap."
        )

    cells: List[Tuple[Any, ...]] = list(itertools.product(*cat_levels))
    k = len(cells)

    # Build representative DataFrame — one row per cell
    # Continuous factors are fixed at their midpoint
    cont_midpoints = {
        fname: (factors[fname][0] + factors[fname][1]) / 2.0
        for fname in cont_names
    }

    rep_rows = []
    for cell in cells:
        row: Dict[str, Any] = {}
        for fname, level in zip(cat_names, cell):
            row[fname] = level
        for fname, midpt in cont_midpoints.items():
            row[fname] = midpt
        rep_rows.append(row)

    # Preserve original factor ordering for Patsy to work correctly
    col_order = list(factors.keys())
    rep_df = pd.DataFrame(rep_rows, columns=col_order)

    # Build model matrix for the representative points
    try:
        X_cells, _ = build_model_matrix(formula, rep_df)
    except Exception as e:
        raise ValueError(
            f"Failed to build model matrix from representative cells. "
            f"Check that all factor levels appear in the formula. "
            f"Original error: {e}"
        ) from e

    # Run Wynn multiplicative algorithm for I-optimality
    weights = _wynn_multiplicative_I(
        X_cells=X_cells,
        jitter=design_opts.xtx_jitter,
        max_iter=design_opts.alloc_wynn_max_iter,
        tol=design_opts.alloc_wynn_tol,
    )

    # Round to integer counts
    counts = _round_allocation(
        weights=weights,
        n=n,
        min_per_cell=design_opts.alloc_min_per_cell,
        max_per_cell=design_opts.alloc_max_per_cell,
    )

    # Return only non-zero cells
    return {
        cell: int(cnt)
        for cell, cnt in zip(cells, counts)
        if cnt > 0
    }


__all__ = ["i_optimal_allocation"]
