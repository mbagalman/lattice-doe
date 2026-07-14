# candidate.py
# License: MIT
"""
Candidate set generation for optimal experimental designs
=========================================================

This module provides utilities for building the candidate pool from which
I-, D-, and A-optimal designs are selected:

  - ``estimate_candidate_size`` — adaptive sizing based on factor complexity
  - ``build_candidate`` — space-filling candidate generation (LHS for
    continuous factors, Cartesian enumeration / sampling for categorical,
    stratified combination for mixed designs)

These functions are called by ``iopt_search.py`` during every design build
and are independent of the exchange algorithm and criterion scoring.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube


# ---------------------------------------------------------------------
# Factor-type helper (single source of truth, mirrors utils.validate_factors)
# ---------------------------------------------------------------------
def _is_continuous_spec(spec: Any) -> bool:
    """Return True iff *spec* represents a continuous factor.

    Delegates to the shared classifier so discriminated-spec markers
    (``_CategoricalSpec`` / ``_ContinuousSpec``, UX-5) win over the legacy
    two-element-numeric heuristic.
    """
    from .utils import _spec_is_continuous

    return _spec_is_continuous(spec)


# ---------------------------------------------------------------------
# Adaptive candidate sizing
# ---------------------------------------------------------------------
def estimate_candidate_size(
    formula: str,
    factors: Dict[str, Any],
    cand_min: int = 1000,
    cand_max: int = 10000,
    cat_cells_cap: int = 10000,
    per_cell_alpha: float = 1.5,
    per_cell_min: int = 5,
    per_cell_max: int = 20,
    seed: Optional[int] = None,
) -> int:
    """Estimate appropriate candidate size based on factor complexity.

    This function analyzes the factor space to determine an appropriate
    number of candidate points, balancing coverage quality against
    computational cost.

    Parameters
    ----------
    formula : str
        Model formula (used to estimate parameter count for complex models).
    factors : dict
        Factor specifications. Continuous factors are tuples (low, high),
        categorical factors are lists of levels.
    cand_min : int, default 1000
        Minimum candidate points to ensure basic coverage.
    cand_max : int, default 10000
        Maximum candidate points to control memory/compute.
    cat_cells_cap : int, default 10000
        Cap on categorical cells to avoid combinatorial explosion.
    per_cell_alpha : float, default 1.5
        Multiplier for pure categorical designs (points = cells × alpha).
    per_cell_min : int, default 5
        Minimum points per categorical cell in mixed designs.
    per_cell_max : int, default 20
        Maximum points per categorical cell in mixed designs.
    seed : int, optional
        Random seed (currently unused but kept for API consistency).

    Returns
    -------
    int
        Recommended number of candidate points.

    Notes
    -----
    Strategy by factor type:
    - Pure continuous: Uses cand_min (LHS scales well with dimension)
    - Pure categorical: min(cells × per_cell_alpha, cand_max)
    - Mixed: Scales with both categorical cells and continuous dimensions,
      using per_cell_min/max to control sampling density within strata

    The formula parameter enables future enhancements where model complexity
    (interactions, polynomials) could influence candidate sizing.
    """
    # Separate continuous and categorical factors
    cont = {k: v for k, v in factors.items() if _is_continuous_spec(v)}
    cat = {k: v for k, v in factors.items() if not _is_continuous_spec(v)}

    # Case 1: Purely continuous design
    if not cat:
        # LHS handles high dimensions well, so base size is sufficient
        cont_dims = len(cont)
        raw_points = cand_min
        if cont_dims > 10:
            # Modest scaling for high-dimensional continuous spaces
            raw_points = int(cand_min * math.sqrt(cont_dims / 10))

        final_points = min(raw_points, cand_max)

        if final_points == cand_max and raw_points > cand_max:
            warnings.warn(
                f"Estimated candidate size ({raw_points}) for continuous factors "
                f"exceeded cand_max ({cand_max}). Clipping to {final_points} points.",
                UserWarning
            )
        return final_points

    # Count categorical cells (with cap to prevent explosion)
    cat_cells = 1
    for factor_name, levels in cat.items():
        cat_cells *= len(levels)
        if cat_cells > cat_cells_cap:
            cat_cells = cat_cells_cap
            break

    # Case 2: Purely categorical design
    if not cont:
        # Need enough points to cover cells, but not excessive. Note that
        # per_cell_alpha only inflates the size *estimate* here: for pure
        # categorical spaces build_candidate enumerates the distinct cells,
        # so no replicated candidate rows materialise (replication of runs
        # is handled at design time via preallocate_categorical, SR-6).
        candidate_points_raw = int(cat_cells * per_cell_alpha)
        candidate_points = min(max(candidate_points_raw, cand_min), cand_max)

        if candidate_points == cand_max and candidate_points_raw > cand_max:
            warnings.warn(
                f"Estimated candidate size ({candidate_points_raw}) for categorical factors "
                f"exceeded cand_max ({cand_max}). Clipping to {candidate_points} points.",
                UserWarning
            )
        elif candidate_points == cand_min and candidate_points_raw < cand_min:
            warnings.warn(
                f"Estimated candidate size ({candidate_points_raw}) for categorical factors "
                f"was below cand_min ({cand_min}). Setting to {candidate_points} points.",
                UserWarning
            )
        return candidate_points

    # Case 3: Mixed categorical-continuous design (most complex case)
    # Strategy: Sample enough points within each categorical cell to
    # adequately explore the continuous subspace
    cont_dims = len(cont)

    # Scale points per cell with continuous dimensionality
    # More continuous dims => need more points per cell for coverage
    if cont_dims <= 2:
        points_per_cell = per_cell_min
    elif cont_dims <= 5:
        # Linear interpolation between min and max based on dimensions
        alpha = (cont_dims - 2) / 3.0
        points_per_cell = per_cell_min + alpha * (per_cell_max - per_cell_min)
    else:
        # High-dimensional continuous space within cells
        points_per_cell = per_cell_max

    # Total candidates = cells × points_per_cell
    candidate_points_raw = int(cat_cells * points_per_cell)

    candidate_points = min(max(candidate_points_raw, cand_min), cand_max)

    if candidate_points == cand_max and candidate_points_raw > cand_max:
        warnings.warn(
            f"Estimated candidate size ({candidate_points_raw}) for mixed factors "
            f"exceeded cand_max ({cand_max}). Clipping to {candidate_points} points.",
            UserWarning
        )
    return candidate_points


# ---------------------------------------------------------------------
# Candidate set generation
# ---------------------------------------------------------------------
def build_candidate(
    factors: Dict[str, Any],
    candidate_points: int = 2000,
    seed: Optional[int] = 123,
    constraint_func: Optional[callable] = None,
    cat_cells_cap: int = 10000,
) -> pd.DataFrame:
    """Build a candidate set from factor specifications.

    This function generates a space-filling candidate set that balances
    coverage of continuous factors (via Latin Hypercube Sampling) with
    enumeration or sampling of categorical combinations.

    Parameters
    ----------
    factors : dict
        Factor specifications. Continuous factors are tuples (low, high),
        categorical factors are lists of levels.
    candidate_points : int, default 2000
        Target number of candidate points. For mixed designs, this is
        approximate due to stratification constraints.
    seed : int, optional
        Random seed for reproducibility in LHS and sampling.
    constraint_func : callable, optional
        Function applied to candidate rows to filter infeasible points.
        Must accept a pandas Series and return bool.
    cat_cells_cap : int, default 10000
        Maximum categorical cells to enumerate. If exceeded, samples
        a subset to avoid memory explosion.

    Returns
    -------
    DataFrame
        Candidate set with columns for each factor. For mixed designs,
        attempts to balance continuous sampling within categorical strata.

    Notes
    -----
    For purely continuous designs, uses Latin Hypercube Sampling (LHS)
    for optimal space-filling properties.

    For purely categorical designs, enumerates all combinations if
    feasible, otherwise samples uniformly.

    For mixed designs, uses a stratified approach:
    - If categorical cells <= cap: enumerate all, sample continuous within each
    - If categorical cells > cap: sample categorical combinations, then continuous

    The constraint_func is applied after generation, so the returned
    DataFrame may have fewer rows than requested if constraints are tight.

    Continuous candidates come from a Latin hypercube, whose stratified
    samples exclude the exact region vertices (expected shortfall
    ~1/(2·candidate_points) per side), so selected designs sit very slightly
    inside the factor ranges. Negligible at default sizes (SR-24f).
    """
    # Separate factor types
    cont = {k: v for k, v in factors.items() if _is_continuous_spec(v)}
    cat = {k: v for k, v in factors.items() if not _is_continuous_spec(v)}

    # Case 1: Continuous-only candidate space
    cont_df = pd.DataFrame()
    if cont:
        sampler = LatinHypercube(d=len(cont), seed=seed)
        lhs_samples = sampler.random(n=candidate_points)  # Unit hypercube [0,1]^d

        # Scale from [0,1] to actual factor ranges
        cont_df = pd.DataFrame(lhs_samples, columns=list(cont.keys()))
        for col_name in cont_df.columns:
            low, high = cont[col_name]
            cont_df[col_name] = low + cont_df[col_name] * (high - low)

    # Case 2: Categorical-only or mixed space
    cat_df = pd.DataFrame()
    if cat:
        # Count total categorical combinations
        total_cells = 1
        for levels in cat.values():
            total_cells *= len(levels)

        if total_cells <= cat_cells_cap:
            # Enumerate all categorical combinations (Cartesian product)
            from itertools import product

            cat_combinations = list(product(*[cat[k] for k in cat.keys()]))
            cat_df = pd.DataFrame(cat_combinations, columns=list(cat.keys()))
        else:
            # Too many combinations - sample a subset
            # Use random sampling with replacement to get cat_cells_cap combinations
            import random
            rng = random.Random(seed)

            sampled_combinations = []
            cat_factor_names = list(cat.keys())
            cat_factor_levels = [cat[k] for k in cat_factor_names]

            for _ in range(min(cat_cells_cap, candidate_points)):
                combo = [rng.choice(levels) for levels in cat_factor_levels]
                sampled_combinations.append(combo)

            cat_df = pd.DataFrame(sampled_combinations, columns=cat_factor_names)
            # Remove duplicates while preserving order
            cat_df = cat_df.drop_duplicates().reset_index(drop=True)

            # Level-coverage repair (SR-13): random combination sampling can
            # miss entire levels, making their model columns unestimable and
            # those treatments unreachable in any design. Append one
            # combination per missing level (other factors at their first
            # level) so every level appears at least once.
            for fname, levels in zip(cat_factor_names, cat_factor_levels):
                present = set(cat_df[fname])
                for lv in levels:
                    if lv not in present:
                        repair = {
                            fn: lvs[0]
                            for fn, lvs in zip(cat_factor_names, cat_factor_levels)
                        }
                        repair[fname] = lv
                        cat_df = pd.concat(
                            [cat_df, pd.DataFrame([repair])], ignore_index=True
                        )
            cat_df = cat_df.drop_duplicates().reset_index(drop=True)

    # Combine continuous and categorical spaces
    if cont and cat:
        # Mixed design: stratified sampling
        # Goal: sample continuous space within each categorical stratum

        n_cat_cells = len(cat_df)

        # Every categorical cell must stay represented (SR-13): dropping
        # cells makes those treatment combinations unreachable in any design
        # and can leave entire levels' model columns unestimable. When the
        # budget allows fewer than 2 continuous samples per cell, keep one
        # row per cell — the candidate set may then exceed candidate_points,
        # which is a documented soft target.
        if n_cat_cells * 2 > candidate_points:
            warnings.warn(
                f"candidate_points={candidate_points} allows fewer than 2 "
                f"continuous samples per categorical cell ({n_cat_cells} "
                "cells). Keeping at least one candidate row per cell; the "
                "candidate set may exceed the requested size. Increase "
                "candidate_points for better continuous-space coverage.",
                UserWarning,
            )

        # Determine samples per categorical cell
        samples_per_cell = max(1, candidate_points // n_cat_cells)

        # Generate stratified candidate set
        frames = []
        for idx, cat_row in cat_df.iterrows():
            # Sample continuous within this categorical stratum. Even with a
            # single sample per cell the point must vary across cells: a
            # constant midpoint in every cell would make each continuous
            # column collinear with the intercept and thus unestimable.
            sampler = LatinHypercube(d=len(cont), seed=seed + idx if seed is not None else idx)
            lhs_samples = sampler.random(n=samples_per_cell)

            # Scale continuous samples
            cont_subset = pd.DataFrame(lhs_samples, columns=list(cont.keys()))
            for col_name in cont_subset.columns:
                low, high = cont[col_name]
                cont_subset[col_name] = low + cont_subset[col_name] * (high - low)

            # Combine with categorical values
            for col_name, value in cat_row.items():
                cont_subset[col_name] = value

            frames.append(cont_subset)

        cand = pd.concat(frames, ignore_index=True)

        # No trimming below one row per cell: with samples_per_cell =
        # max(1, candidate_points // n_cat_cells) the set only exceeds
        # candidate_points in the one-row-per-cell regime, where any trim
        # would drop cells (SR-13).

    elif cont:
        # Pure continuous
        cand = cont_df
    else:
        # Pure categorical
        cand = cat_df

    # Apply constraint filtering if provided
    if constraint_func is not None:
        mask = cand.apply(constraint_func, axis=1)
        cand = cand.loc[mask].reset_index(drop=True)

        # Warn if constraints eliminate many points
        if len(cand) < candidate_points * 0.5:
            warnings.warn(
                f"Constraints eliminated {candidate_points - len(cand)} of {candidate_points} "
                f"candidate points ({100 * (1 - len(cand)/candidate_points):.1f}% removed). "
                "Consider relaxing constraints or increasing candidate_points."
            )

    return cand.reset_index(drop=True)


# ---------------------------------------------------------------------
# Split-plot candidate set generation
# ---------------------------------------------------------------------
def build_split_plot_candidate(
    factors: Dict[str, Any],
    htc_factors: List[str],
    n_whole_plots: int,
    subplots_per_wp: int,
    *,
    random_state: Optional[int] = None,
    candidate_points: Optional[int] = None,
    constraint_func: Optional[callable] = None,
) -> pd.DataFrame:
    """Build an initial split-plot candidate set.

    Generates a structured candidate set with *n_whole_plots* whole-plot (WP)
    slots, each containing *subplots_per_wp* sub-plot runs.  Hard-to-change
    (HTC) factor settings are constant within each WP slot; easy-to-change
    (ETC) factor settings vary freely across sub-plots.

    Parameters
    ----------
    factors : dict
        All factor specifications.  Continuous factors are tuples ``(low, high)``;
        categorical factors are lists of level values.
    htc_factors : list of str
        Names of hard-to-change (whole-plot) factors.  Every name must be a key
        in *factors*.
    n_whole_plots : int
        Number of whole-plot slots (≥ 2).
    subplots_per_wp : int
        Number of sub-plots per whole-plot slot (≥ 1).
    random_state : int, optional
        Random seed for reproducibility.
    candidate_points : int, optional
        Size of the WP candidate pool from which *n_whole_plots* settings are
        drawn.  Defaults to ``max(n_whole_plots * 3, 50)``.
    constraint_func : callable, optional
        Row-wise filter applied to the combined candidate set.  Must accept a
        pandas Series and return ``bool``.

    Returns
    -------
    DataFrame
        Exactly ``n_whole_plots * subplots_per_wp`` rows (before constraint
        filtering).  Columns are all factor names (in original order) followed
        by ``'__wp_id__'`` (integer, 0-indexed WP slot).  All rows sharing the
        same ``__wp_id__`` have identical HTC factor values.

    Raises
    ------
    ValueError
        If *n_whole_plots* < 2, *subplots_per_wp* < 1, *htc_factors* is empty,
        or any name in *htc_factors* is absent from *factors*.
    """
    # --- Input validation ---
    if not isinstance(n_whole_plots, int) or isinstance(n_whole_plots, bool):
        raise ValueError("n_whole_plots must be a plain integer.")
    if n_whole_plots < 2:
        raise ValueError(f"n_whole_plots must be ≥ 2, got {n_whole_plots}.")
    if not isinstance(subplots_per_wp, int) or isinstance(subplots_per_wp, bool):
        raise ValueError("subplots_per_wp must be a plain integer.")
    if subplots_per_wp < 1:
        raise ValueError(f"subplots_per_wp must be ≥ 1, got {subplots_per_wp}.")
    if not htc_factors:
        raise ValueError("htc_factors must be a non-empty list.")
    missing = set(htc_factors) - set(factors.keys())
    if missing:
        raise ValueError(
            f"htc_factors contains names not found in factors: {sorted(missing)}."
        )

    # --- Split factor specs into HTC and ETC ---
    htc_set = set(htc_factors)
    htc_factor_dict = {k: factors[k] for k in factors if k in htc_set}
    etc_factor_dict = {k: factors[k] for k in factors if k not in htc_set}

    n_sp_runs = n_whole_plots * subplots_per_wp

    # --- Generate WP settings (one per WP slot) ---
    wp_pool_size = max(n_whole_plots * 3, candidate_points or 0, 50)
    wp_pool = build_candidate(htc_factor_dict, candidate_points=wp_pool_size, seed=random_state)

    rng = np.random.default_rng(random_state)
    if len(wp_pool) >= n_whole_plots:
        wp_idx = rng.choice(len(wp_pool), size=n_whole_plots, replace=False)
    else:
        wp_idx = rng.choice(len(wp_pool), size=n_whole_plots, replace=True)
    wp_settings = wp_pool.iloc[wp_idx].reset_index(drop=True)

    # --- Generate SP settings (ETC factors, n_sp_runs rows) ---
    sp_seed = (random_state + 1) if random_state is not None else None
    if etc_factor_dict:
        sp_pool_size = max(n_sp_runs * 2, 100)
        sp_pool = build_candidate(etc_factor_dict, candidate_points=sp_pool_size, seed=sp_seed)
        if len(sp_pool) >= n_sp_runs:
            sp_idx = rng.choice(len(sp_pool), size=n_sp_runs, replace=False)
        else:
            sp_idx = rng.choice(len(sp_pool), size=n_sp_runs, replace=True)
        sp_settings = sp_pool.iloc[sp_idx].reset_index(drop=True)
    else:
        sp_settings = None

    # --- Build structured candidate set ---
    frames: List[pd.DataFrame] = []
    for wp_i in range(n_whole_plots):
        wp_row = wp_settings.iloc[wp_i]
        sp_start = wp_i * subplots_per_wp
        sp_end = sp_start + subplots_per_wp

        block: Dict[str, Any] = {}
        for col in factors:
            if col in htc_set:
                block[col] = [wp_row[col]] * subplots_per_wp
            else:
                block[col] = sp_settings[col].iloc[sp_start:sp_end].values
        block["__wp_id__"] = [wp_i] * subplots_per_wp
        frames.append(pd.DataFrame(block))

    cand = pd.concat(frames, ignore_index=True)
    cand["__wp_id__"] = cand["__wp_id__"].astype(int)

    # --- Apply constraint filtering ---
    if constraint_func is not None:
        factor_cols = list(factors.keys())
        mask = cand[factor_cols].apply(constraint_func, axis=1)
        n_before = len(cand)
        cand = cand.loc[mask].reset_index(drop=True)
        n_removed = n_before - len(cand)
        if n_removed > 0 and len(cand) < n_before * 0.5:
            warnings.warn(
                f"Constraints eliminated {n_removed} of {n_before} split-plot candidate "
                f"rows ({100 * n_removed / n_before:.1f}% removed). "
                "Consider relaxing constraints or increasing candidate_points.",
                UserWarning,
            )

    return cand


__all__ = ["estimate_candidate_size", "build_candidate", "build_split_plot_candidate"]
