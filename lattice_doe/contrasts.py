# contrasts.py
# License: MIT
"""
Convenience builders for contrasts
---------------------------------

Helpers to build contrast matrices `L` and SESOI vectors `delta` from
human-friendly descriptions (e.g., two named scenarios in factor space).

Public functions
----------------
- contrast_from_scenarios(...): single-row L (1 x p) for a pairwise difference

Implementation note
-------------------
To guarantee that the model coding (dummy columns, interactions, etc.) used to
construct scenario rows matches the coding used during design generation, we
build a *single* model matrix that includes:
  1) at least one candidate row from the factor space (to anchor all levels), and
  2) both scenarios.
We then slice the last two rows to obtain x_a and x_b and form L = x_b - x_a.
This avoids subtle column-misalignment when not all levels appear in the two
scenarios themselves.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union, Any
import numpy as np
import pandas as pd

from .candidate import build_candidate
from .model_matrix import build_model_matrix

FactorSpec = Dict[str, Union[list, tuple]]
Scenario = Dict[str, Union[int, float, str]]


def _validate_scenario(
    scenario_name: str,
    scenario: Scenario,
    factors: FactorSpec
) -> None:
    """
    Validate a scenario dict against the project's factor specification.

    Raises
    ------
    KeyError
        If the scenario is missing factors or has extra factors.
    ValueError
        If a scenario value is outside the allowed range (continuous)
        or not in the list of levels (categorical).
    """
    factor_names = set(factors.keys())
    scenario_names = set(scenario.keys())

    # Check for missing factors
    missing = factor_names - scenario_names
    if missing:
        raise KeyError(
            f"Validation failed for '{scenario_name}': "
            f"Missing required factors: {sorted(list(missing))}"
        )

    # Check for extra/unknown factors
    extra = scenario_names - factor_names
    if extra:
        raise KeyError(
            f"Validation failed for '{scenario_name}': "
            f"Unknown factors provided: {sorted(list(extra))}"
        )

    # Check each value
    for factor_name, value in scenario.items():
        spec = factors[factor_name]

        # Case 1: Continuous factor (spec is a tuple)
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            low, high = float(spec[0]), float(spec[1])
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Validation failed for '{scenario_name}', factor '{factor_name}': "
                    f"Expected a numerical value, but got '{value}' (type {type(value).__name__})."
                )
            if not (low <= float(value) <= high):
                raise ValueError(
                    f"Validation failed for '{scenario_name}', factor '{factor_name}': "
                    f"Value {value} is outside the allowed continuous range [{low}, {high}]."
                )

        # Case 2: Categorical factor (spec is a list)
        elif isinstance(spec, (list, tuple)):
            levels = list(spec)
            if value not in levels:
                raise ValueError(
                    f"Validation failed for '{scenario_name}', factor '{factor_name}': "
                    f"Value '{value}' is not one of the allowed categorical levels: {levels}."
                )
        
        else:
            raise TypeError(
                f"Invalid factor specification for '{factor_name}'. "
                f"Expected tuple (min, max) or list of levels, but got {spec}."
            )


def contrast_from_scenarios(
    formula: str,
    factors: FactorSpec,
    scenario_a: Scenario,
    scenario_b: Scenario,
    sesoi: float,
    *,
    candidate_points: int = 10,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build `L` and `delta` for comparing two concrete scenarios.

    L is a single row formed by subtracting the model rows for the two
    scenarios: ``x_b - x_a``. The SESOI vector is ``(sesoi,)``.

    Parameters
    ----------
    formula : str
        Patsy-style model formula (e.g., "~ 1 + A + B + A:B").
    factors : dict
        Factor specifications (same structure used for building candidates).
    scenario_a : dict
        Mapping of factor -> level/value for scenario A.
    scenario_b : dict
        Mapping of factor -> level/value for scenario B.
    sesoi : float
        Smallest effect size of interest on the response scale.
    candidate_points : int, default 10
        Tiny candidate set size to anchor model coding (patsy/pyDOE3). Only a
        *single* candidate row is used in the actual matrix below.
    seed : int, default 0
        Random seed used only if continuous LHS is needed when building the
        tiny candidate set.

    Returns
    -------
    (L, delta) : (np.ndarray, np.ndarray)
        ``L`` has shape ``(1, p)``; ``delta`` has shape ``(1,)``.
    """
    _validate_scenario("scenario_a", scenario_a, factors)
    _validate_scenario("scenario_b", scenario_b, factors)
    if not isinstance(sesoi, (int, float)) or sesoi <= 0:
        raise ValueError(
            f"sesoi must be a positive number, but got {sesoi}."
        )

    # Build a tiny candidate to ensure patsy coding is anchored consistently and
    # that all categorical levels are represented somewhere in the matrix.
    tmp_cand = build_candidate(
        factors, candidate_points=candidate_points, seed=seed, constraint_func=None
    )

    a_df = pd.DataFrame([scenario_a])
    b_df = pd.DataFrame([scenario_b])

    # Concatenate one candidate row + the two scenarios, then build ONE model matrix
    # Use at least one row from the *start* of the candidate set to anchor coding
    anchor_row = tmp_cand.iloc[:1]
    
    both = pd.concat([anchor_row, a_df, b_df], ignore_index=True)
    
    try:
        X_both, _ = build_model_matrix(formula, both)
    except Exception as e:
        raise ValueError(
             f"Failed to build model matrix from scenarios. "
             f"This can happen if scenario values cause patsy errors "
             f"(e.g., divide by zero in formula like 'I(1/A)'). "
             f"Patsy error: {e}"
        ) from e


    # Last two rows correspond to scenarios A and B in order
    x_a = X_both[-2, :]
    x_b = X_both[-1, :]

    L = (x_b - x_a).reshape(1, -1)
    delta = np.array([float(sesoi)], dtype=float)
    return L, delta


__all__ = ["contrast_from_scenarios"]