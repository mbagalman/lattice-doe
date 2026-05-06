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


__all__ = ["validate_factors", "initial_n_guess"]
