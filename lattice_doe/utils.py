# utils.py
# License: MIT
"""
General utilities for Lattice DOE
---------------------------------

Small, dependency-light helpers used across modules (validation, sizing, etc.).
"""
from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import numpy as np


FactorSpec = Dict[str, Union[List[Union[int, float, str]], Tuple[float, float]]]


# --- Discriminated factor-spec markers (UX-5) -----------------------------
# The legacy shorthand classifies any two-element numeric sequence as
# continuous, so a binary numeric CATEGORY like ``[0, 1]`` is ambiguous and
# cannot be expressed. The discriminated dict forms
#   {"type": "continuous", "low": lo, "high": hi}
#   {"type": "categorical", "levels": [...]}
# normalize to these marker subclasses, which ARE a tuple / list (so all
# downstream ``lo, hi = spec`` / ``list(spec)`` code keeps working) but let the
# spec classifiers resolve the type unambiguously regardless of level dtype.


class _ContinuousSpec(tuple):
    """A ``(low, high)`` continuous spec, explicitly typed."""

    __slots__ = ()


class _CategoricalSpec(list):
    """A list of levels, explicitly typed as categorical."""

    __slots__ = ()


def _spec_is_continuous(spec: Any) -> bool:
    """Authoritative continuous/categorical classifier honoring markers.

    Marker subclasses win; otherwise the legacy heuristic (a two-element
    all-numeric sequence is continuous) applies.
    """
    if isinstance(spec, _CategoricalSpec):
        return False
    if isinstance(spec, _ContinuousSpec):
        return True
    return (
        isinstance(spec, (tuple, list))
        and len(spec) == 2
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in spec)
    )


def normalize_factors(
    factors: FactorSpec,
    formula: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize factor specs, resolving the discriminated dict forms (UX-5).

    Accepts, per factor:
      * ``{"type": "continuous", "low": lo, "high": hi}`` → ``_ContinuousSpec``
      * ``{"type": "categorical", "levels": [...]}``      → ``_CategoricalSpec``
      * legacy ``(lo, hi)`` / ``[lo, hi]``                → passed through
      * legacy ``[level, ...]``                           → passed through

    When *formula* is supplied, a legacy two-element numeric spec that the
    formula wraps in ``C(name)`` (the one case where the intended type visibly
    conflicts with the heuristic) triggers a ``DeprecationWarning`` steering the
    caller to the explicit categorical dict form.

    Returns a new dict; the input is not mutated.
    """
    out: Dict[str, Any] = {}
    for name, spec in factors.items():
        if isinstance(spec, dict) and "type" in spec:
            kind = spec.get("type")
            if kind == "continuous":
                if "low" not in spec or "high" not in spec:
                    raise ValueError(
                        f"Continuous factor '{name}' needs 'low' and 'high' keys."
                    )
                out[name] = _ContinuousSpec((spec["low"], spec["high"]))
            elif kind == "categorical":
                levels = spec.get("levels")
                if not isinstance(levels, (list, tuple)) or len(levels) == 0:
                    raise ValueError(
                        f"Categorical factor '{name}' needs a non-empty 'levels' list."
                    )
                out[name] = _CategoricalSpec(list(levels))
            else:
                raise ValueError(
                    f"Factor '{name}' has unknown type {kind!r}; use "
                    "'continuous' or 'categorical'."
                )
            continue

        # Legacy form — pass through, but flag the ambiguous C(...) case.
        # Markers are explicit already, so they never warn.
        if (
            formula
            and not isinstance(spec, (_CategoricalSpec, _ContinuousSpec))
            and isinstance(spec, (tuple, list))
            and len(spec) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in spec)
            and re.search(r"\bC\(\s*" + re.escape(name) + r"\s*[,)]", formula)
        ):
            warnings.warn(
                f"Factor '{name}' is given as a two-number list {list(spec)!r} "
                f"(treated as a CONTINUOUS range) but the formula wraps it in "
                f"C({name}). If you meant a categorical factor with levels "
                f"{list(spec)!r}, use the explicit form "
                f'{{"type": "categorical", "levels": {list(spec)!r}}}; the '
                "ambiguous shorthand is deprecated for this case.",
                DeprecationWarning,
                stacklevel=2,
            )
        out[name] = spec
    return out


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

        # --- Distinguish continuous vs categorical (markers win, UX-5) ---
        is_cont = _spec_is_continuous(spec)

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

    factors = normalize_factors(factors)
    cat = {k: list(v) for k, v in factors.items() if not _spec_is_continuous(v)}
    cont = {k: v for k, v in factors.items() if _spec_is_continuous(v)}

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


__all__ = [
    "validate_factors",
    "initial_n_guess",
    "model_matrix_preview",
    "normalize_factors",
]
