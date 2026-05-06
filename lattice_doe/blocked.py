# blocked.py
# License: MIT
"""
Blocked design utilities for I-optimal experimental designs
===========================================================

Implements utilities for constructing blocked designs where experimental
runs are grouped into blocks to account for nuisance variation (e.g.,
day-to-day variation, different operators, batch effects).

Block effects are treated as nuisance factors: included in the model for
adjustment but not tested as primary hypotheses.  Adding block effects
to the model costs (n_blocks - 1) denominator degrees of freedom.

Public API
----------
``balanced_block_sizes(n, n_blocks)``
    Return block sizes that are as equal as possible and sum to *n*.

``blocked_formula(formula, block_factor_name)``
    Append the block factor to a Patsy formula string.

``build_blocked_design(cand, formula, n, n_blocks, block_sizes,
                       block_factor_name, aug_formula, *, ...)``
    Run independent within-block I-optimal searches and combine results.
    Returns (design_df, X_full).

References
----------
* Goos & Jones (2011) "Optimal Design of Experiments: A Case Study Approach."
* Atkinson, Donev & Tobias (2007) "Optimum Experimental Designs, with SAS."
  Chapter 13 on blocking.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .model_matrix import build_model_matrix


def balanced_block_sizes(n: int, n_blocks: int) -> List[int]:
    """Compute balanced block sizes that sum to n.

    Distributes n runs as evenly as possible across n_blocks blocks.
    The first ``n % n_blocks`` blocks receive one extra run.

    Parameters
    ----------
    n : int
        Total number of runs.
    n_blocks : int
        Number of blocks (>= 2).

    Returns
    -------
    list of int
        Block sizes; ``len == n_blocks`` and ``sum == n``.

    Examples
    --------
    >>> balanced_block_sizes(10, 3)
    [4, 3, 3]
    >>> balanced_block_sizes(12, 4)
    [3, 3, 3, 3]
    """
    if n_blocks < 2:
        raise ValueError(f"n_blocks must be >= 2; got {n_blocks}.")
    if n < n_blocks:
        raise ValueError(
            f"n ({n}) must be >= n_blocks ({n_blocks}): "
            "each block must have at least 1 run."
        )
    base = n // n_blocks
    remainder = n % n_blocks
    return [base + (1 if i < remainder else 0) for i in range(n_blocks)]


def blocked_formula(formula: str, block_factor_name: str = "Block") -> str:
    """Append the block factor to a Patsy formula string.

    Parameters
    ----------
    formula : str
        Original Patsy formula (e.g., ``"1 + A + B"``).
    block_factor_name : str, default ``"Block"``
        Name of the block column in the design DataFrame.

    Returns
    -------
    str
        Augmented formula, e.g. ``"1 + A + B + C(Block)"``.
    """
    return f"{formula} + C({block_factor_name})"


def build_blocked_design(
    cand: pd.DataFrame,
    formula: str,
    n: int,
    n_blocks: int,
    block_sizes: Optional[List[int]],
    block_factor_name: str,
    aug_formula: str,
    *,
    criterion: str,
    n_start: int,
    algo: str,
    max_iter: int,
    random_state: Optional[int],
    workers: Optional[int],
    parallel_seed_stride: int,
    jitter: float,
    preallocate_categorical: bool,
    alloc_min_per_cell: int,
    alloc_max_per_cell: Optional[int],
    alloc_wynn_max_iter: int,
    alloc_wynn_tol: float,
    cat_cells_cap: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run within-block I-optimal searches and combine into a blocked design.

    For each block, runs an independent I-optimal search on the treatment
    candidate set (without block column) for the block's allocated number of
    runs.  Combines block-level designs and adds the block column, then builds
    the augmented model matrix using *aug_formula* (which includes block term).

    Parameters
    ----------
    cand : DataFrame
        Candidate set (treatment factors only — no block column).
    formula : str
        Treatment-only Patsy formula (no block term).
    n : int
        Total number of runs across all blocks.
    n_blocks : int
        Number of blocks.
    block_sizes : list of int or None
        Sizes for each block.  If None, uses ``balanced_block_sizes(n, n_blocks)``.
    block_factor_name : str
        Column name for the block assignment (e.g. ``"Block"``).
    aug_formula : str
        Patsy formula including the block term (from ``blocked_formula``).
    criterion, n_start, algo, max_iter, random_state, workers, ... :
        Forwarded to ``build_i_opt_design_with_idx`` for each within-block search.
        Each block gets seed ``random_state + b_idx * parallel_seed_stride``.

    Returns
    -------
    design_df : DataFrame
        Combined design with block column added (n rows).
    X_full : ndarray, shape (n, p_full)
        Augmented model matrix built from *design_df* using *aug_formula*.
        Columns = treatment columns + (n_blocks - 1) block dummy columns.
    """
    from .iopt_search import build_i_opt_design_with_idx

    if block_sizes is None:
        block_sizes = balanced_block_sizes(n, n_blocks)

    if len(block_sizes) != n_blocks:
        raise ValueError(
            f"len(block_sizes)={len(block_sizes)} != n_blocks={n_blocks}."
        )
    if sum(block_sizes) != n:
        raise ValueError(
            f"sum(block_sizes)={sum(block_sizes)} != n={n}. "
            "Adjust block_sizes so they sum to the total run count."
        )

    # Guard against block_factor_name colliding with a treatment column.
    if block_factor_name in cand.columns:
        raise ValueError(
            f"block_factor_name={block_factor_name!r} is already a column in the "
            f"candidate set. Choose a name not in: {list(cand.columns)}."
        )

    X_cand_treat, _ = build_model_matrix(formula, cand)
    block_labels = [f"B{i + 1}" for i in range(n_blocks)]

    block_frames: List[pd.DataFrame] = []

    for b_idx, (b_label, b_size) in enumerate(zip(block_labels, block_sizes)):
        # Give each block a distinct seed to decorrelate the within-block searches
        seed_b = (
            (random_state + b_idx * parallel_seed_stride)
            if random_state is not None
            else None
        )
        design_b, _sel_idx_b, _ = build_i_opt_design_with_idx(
            cand=cand,
            formula=formula,
            n=b_size,
            criterion=criterion,
            n_start=n_start,
            algo=algo,
            max_iter=max_iter,
            random_state=seed_b,
            workers=workers,
            parallel_seed_stride=parallel_seed_stride,
            jitter=jitter,
            preallocate_categorical=preallocate_categorical,
            alloc_min_per_cell=alloc_min_per_cell,
            alloc_max_per_cell=alloc_max_per_cell,
            alloc_wynn_max_iter=alloc_wynn_max_iter,
            alloc_wynn_tol=alloc_wynn_tol,
            cat_cells_cap=cat_cells_cap,
        )
        design_b = design_b.copy()
        design_b[block_factor_name] = b_label
        block_frames.append(design_b)

    design_df = pd.concat(block_frames, ignore_index=True)

    # Build augmented model matrix (includes block dummy columns)
    X_full, _ = build_model_matrix(aug_formula, design_df)

    return design_df, X_full


__all__ = ["balanced_block_sizes", "blocked_formula", "build_blocked_design"]
