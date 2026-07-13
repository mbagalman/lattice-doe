# utils.py
# License: MIT
"""
General utilities for Lattice DOE
---------------------------------

Small, dependency-light helpers used across modules (validation, sizing, etc.).
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union, Literal
import numpy as np


FactorSpec = Dict[str, Union[List[Union[int, float, str]], Tuple[float, float]]]


def validate_factors(factors: FactorSpec) -> None:
    """Validate factor specifications.

    Rules
    -----
    - `factors` must be non-empty.
    - Factor names must be unique (case-insensitive).
    - Continuous factors are `(low, high)` with `low < high` and finite values.
    - Categorical factors are non-empty sequences (list/tuple) of unique levels.
    """
    if not isinstance(factors, dict) or len(factors) == 0:
        raise ValueError("factors must be a non-empty dict")

    seen_factor_names: Dict[str, str] = {}

    for name, spec in factors.items():
        name_lower = name.lower()
        if name_lower in seen_factor_names:
            original_name = seen_factor_names[name_lower]
            raise ValueError(
                f"Duplicate factor name: '{name}' is not unique "
                f"(conflicts with '{original_name}', case-insensitive)."
            )
        seen_factor_names[name_lower] = name

        # --- Distinguish continuous vs categorical ---
        
        # Continuous: tuple(low, high) or list[low, high] with numeric content
        is_cont = False
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            if all(isinstance(x, (int, float)) for x in spec):
                is_cont = True
        
        if is_cont:
            lo, hi = spec
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except Exception as e:
                raise ValueError(
                    f"Continuous factor '{name}' bounds must be numeric: {e}"
                )
            
            if not (np.isfinite(lo_f) and np.isfinite(hi_f)):
                raise ValueError(
                    f"Continuous factor '{name}' bounds must be finite numbers; "
                    f"got ({lo}, {hi})"
                )
                
            if not (lo_f < hi_f):
                raise ValueError(
                    f"Continuous factor '{name}' needs (low, high) with low < high; "
                    f"got ({lo}, {hi})"
                )
        
        # Categorical: sequence of levels
        elif isinstance(spec, (list, tuple)):
            levels = list(spec)
            if len(levels) == 0:
                raise ValueError(
                    f"Categorical factor '{name}' must have at least one level."
                )
                
            if len(levels) != len(set(levels)):
                raise ValueError(
                    f"Categorical factor '{name}' contains duplicate levels."
                )
        
        # Invalid spec type
        else:
            raise ValueError(
                f"Factor '{name}' spec is invalid. "
                "Expected (low, high) tuple for continuous or list of levels "
                f"for categorical, but got {type(spec).__name__}."
            )


def initial_n_guess(p: int, mode: Literal["contrast", "r2"]) -> int:
    """Conservative initial guess for design size `n`.

    Heuristic aims to ensure `n >= p+1` and avoid repeated infeasible proposals
    when the model is rich. The outer loop in `api.py` will grow `n` as needed.
    """
    base = 4 * p if mode == "contrast" else 3 * p
    return max(base, p + 1, 16)


def model_matrix_preview(
    formula: str,
    factors: Dict[str, Union[Tuple[float, float], List]],
    max_preview_rows: int = 10_000,
) -> Tuple[int, List[str]]:
    """Return (p, column_names) for *formula* over a representative frame.

    Builds the preview frame from the full Cartesian cross of every
    categorical factor's levels, with continuous factors at their range
    midpoints (UX-1). A single-row frame containing only the first level of
    each categorical would make Patsy silently drop the remaining dummy
    columns and every categorical interaction column, so the reported model
    size p would undercount — e.g. a three-level ``C(g)`` model shown as
    p = 1 instead of p = 3 — misleading users constructing contrast
    matrices against the displayed columns.

    Parameters
    ----------
    formula : str
        Patsy model formula.
    factors : dict
        Factor spec — continuous factors as 2-tuples/lists of numbers,
        categorical factors as lists of levels (package convention).
    max_preview_rows : int, default 10 000
        Guard against categorical-level explosions; raises ValueError when
        the level cross exceeds this.

    Returns
    -------
    (p, column_names)
        Model-matrix column count and Patsy column labels.
    """
    import itertools

    import pandas as pd

    from .model_matrix import build_model_matrix

    def _is_cont(spec) -> bool:
        return (
            isinstance(spec, (tuple, list))
            and len(spec) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in spec)
        )

    cat = {k: list(v) for k, v in factors.items() if not _is_cont(v)}
    cont = {k: v for k, v in factors.items() if _is_cont(v)}

    n_rows = 1
    for levels in cat.values():
        if not levels:
            raise ValueError(
                "Every categorical factor needs at least one level for the "
                "model preview."
            )
        n_rows *= len(levels)
    if n_rows > max_preview_rows:
        raise ValueError(
            f"Categorical level cross has {n_rows} combinations, exceeding "
            f"the preview cap of {max_preview_rows}."
        )

    if cat:
        combos = list(itertools.product(*cat.values()))
        frame = {name: [c[i] for c in combos]
                 for i, name in enumerate(cat.keys())}
    else:
        combos = [()]
        frame = {}
    for name, (lo, hi) in cont.items():
        frame[name] = [(float(lo) + float(hi)) / 2.0] * len(combos)

    X, col_names = build_model_matrix(formula, pd.DataFrame(frame))
    return X.shape[1], list(col_names)


__all__ = ["validate_factors", "initial_n_guess", "model_matrix_preview"]
