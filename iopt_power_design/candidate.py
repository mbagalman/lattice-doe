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

from typing import Any, Dict, Optional
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube


# ---------------------------------------------------------------------
# Factor-type helper (single source of truth, mirrors utils.validate_factors)
# ---------------------------------------------------------------------
def _is_continuous_spec(spec: Any) -> bool:
    """Return True iff *spec* represents a continuous factor (2-element numeric sequence)."""
    return (
        isinstance(spec, (tuple, list))
        and len(spec) == 2
        and all(isinstance(x, (int, float)) for x in spec)
    )


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

        # --- ADDED: Clipping check ---
        if final_points == cand_max and raw_points > cand_max:
            warnings.warn(
                f"Estimated candidate size ({raw_points}) for continuous factors "
                f"exceeded cand_max ({cand_max}). Clipping to {final_points} points.",
                UserWarning
            )
        # --- END ADDED ---
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
        # Need enough points to cover cells, but not excessive
        # per_cell_alpha > 1 provides some replication for numerical stability
        candidate_points_raw = int(cat_cells * per_cell_alpha)
        candidate_points = min(max(candidate_points_raw, cand_min), cand_max)

        # --- ADDED: Clipping check ---
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
        # --- END ADDED ---
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

    # --- ADDED: Clipping check ---
    if candidate_points == cand_max and candidate_points_raw > cand_max:
        warnings.warn(
            f"Estimated candidate size ({candidate_points_raw}) for mixed factors "
            f"exceeded cand_max ({cand_max}). Clipping to {candidate_points} points.",
            UserWarning
        )
    # --- END ADDED ---
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

    # Combine continuous and categorical spaces
    if cont and cat:
        # Mixed design: stratified sampling
        # Goal: sample continuous space within each categorical stratum

        n_cat_cells = len(cat_df)

        if n_cat_cells * 2 > candidate_points:
            # Not enough budget for multiple samples per cell
            # Take subset of categorical cells
            n_cells_to_use = max(1, candidate_points // 2)
            cat_df = cat_df.sample(n=min(n_cells_to_use, len(cat_df)),
                                  random_state=seed).reset_index(drop=True)
            n_cat_cells = len(cat_df)

        # Determine samples per categorical cell
        samples_per_cell = max(1, candidate_points // n_cat_cells)

        # Generate stratified candidate set
        frames = []
        for idx, cat_row in cat_df.iterrows():
            # Sample continuous within this categorical stratum
            if samples_per_cell > 1:
                sampler = LatinHypercube(d=len(cont), seed=seed + idx if seed else idx)
                lhs_samples = sampler.random(n=samples_per_cell)
            else:
                # Single sample - use center or random point
                lhs_samples = np.array([[0.5] * len(cont)])

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

        # Ensure we don't exceed requested size
        if len(cand) > candidate_points:
            cand = cand.sample(n=candidate_points, random_state=seed).reset_index(drop=True)

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


__all__ = ["estimate_candidate_size", "build_candidate"]
