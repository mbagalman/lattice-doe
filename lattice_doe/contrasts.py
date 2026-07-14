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
  1) a level-covering anchor frame containing every level of every categorical
     factor (so no dummy column can be silently dropped, TD-7), and
  2) both scenarios.
We then slice the last two rows to obtain x_a and x_b and form L = x_b - x_a.
This avoids subtle column-misalignment when not all levels appear in the two
scenarios themselves.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd

from .model_matrix import build_model_matrix
from .utils import (
    FactorSpec,
    normalize_factors,
    _spec_is_continuous,
    _representative_frame,
)

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

        # Case 1: Continuous factor — classified by the shared package-wide
        # helper so explicit markers (UX-5) win over the two-numeric heuristic
        # (a typed binary numeric category like [0, 1] must NOT land here).
        if _spec_is_continuous(spec):
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
        Unused; retained for backward compatibility. Model coding is now
        anchored by a deterministic level-covering frame (TD-7), not a
        sampled candidate row.
    seed : int, default 0
        Unused; retained for backward compatibility (see candidate_points).

    Returns
    -------
    (L, delta) : (np.ndarray, np.ndarray)
        ``L`` has shape ``(1, p)``; ``delta`` has shape ``(1,)``.
    """
    # Resolve discriminated factor-spec dict forms before validation (UX-5);
    # scenario validation and the anchor candidate both need typed specs.
    factors = normalize_factors(factors, formula)
    _validate_scenario("scenario_a", scenario_a, factors)
    _validate_scenario("scenario_b", scenario_b, factors)
    if not isinstance(sesoi, (int, float)) or sesoi <= 0:
        raise ValueError(
            f"sesoi must be a positive number, but got {sesoi}."
        )

    # Anchor Patsy's coding with a LEVEL-COVERING frame, not a single candidate
    # row (TD-7): when both scenarios share a categorical level and the lone
    # anchor row happened to share it too, Patsy never saw the other levels and
    # silently dropped their dummy columns, returning an L narrower than the
    # design model (which find_optimal_design then rejects). The representative
    # frame contains every level of every categorical factor, so the model
    # coding always matches design generation.
    anchor_frame, _exact = _representative_frame(factors)

    a_df = pd.DataFrame([scenario_a])
    b_df = pd.DataFrame([scenario_b])

    # Concatenate the anchor frame + the two scenarios, then build ONE model
    # matrix so all rows share identical column coding.
    both = pd.concat([anchor_frame, a_df, b_df], ignore_index=True)
    
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