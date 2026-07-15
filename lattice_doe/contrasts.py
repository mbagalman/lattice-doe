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
construct scenario rows matches the coding used during design generation, the
coding is established with :func:`patsy.incr_dbuilder` over an anchor that
covers every level of every formula-referenced categorical factor (streamed in
chunks — no anchor model matrix is ever materialized, TD-7/UX-36), plus the
two scenario rows. Only the 2 × p scenario matrix is built; L = x_b − x_a.
Callers holding realized data (a candidate set or design) can pass it as
``coding_data`` to make that data the authority instead.
"""
from __future__ import annotations

import itertools
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import patsy

from .utils import (
    FactorSpec,
    normalize_factors,
    _spec_is_continuous,
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
        # Any-typed local: after runtime classification the spec is indexed
        # positionally / iterated, which mypy cannot narrow through the union.
        spec: Any = factors[factor_name]

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


_CONTRAST_CROSS_CAP = 100_000  # max cross rows per combination-derived GROUP
_ANCHOR_CHUNK_ROWS = 5_000     # rows per chunk in the incremental coding scan

#: Patsy stateful transforms: their coding parameters (knots, means, scales…)
#: are LEARNED from the input data, so an internally generated anchor would
#: silently produce a numerically different — same-width — contrast than the
#: realized design coding. They therefore require authoritative coding_data.
_STATEFUL_TRANSFORMS = frozenset(
    {"bs", "cr", "cc", "te", "center", "standardize", "scale"}
)


def _formula_factor_codes(formula: str) -> List[str]:
    """Per-EvalFactor code strings from Patsy's parsed model description.

    Falls back to the whole formula as a single code if parsing fails, which
    only makes downstream checks more conservative (never less)."""
    try:
        desc = patsy.ModelDesc.from_formula(formula)
        codes = [f.code for t in desc.rhs_termlist for f in t.factors]
        return codes if codes else [formula]
    except Exception:
        return [formula]


def _code_identifiers(code: str) -> set:
    """Names a factor-code expression can reference: Python identifiers from
    Patsy's own AST walk, plus ``Q("...")``-quoted names (which resolve via
    the data and can contain non-identifier characters)."""
    names: set = set()
    try:
        from patsy.eval import ast_names

        names |= set(ast_names(code))
    except Exception:
        # Conservative fallback: any identifier-shaped token.
        names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))
    for m in re.finditer(r"""Q\(\s*['"](.+?)['"]\s*\)""", code):
        names.add(m.group(1))
    return names


def _merge_groups(groups: List[set]) -> List[set]:
    """Union-find style merge of overlapping sets."""
    merged: List[set] = []
    for g in groups:
        g = set(g)
        keep: List[set] = []
        for m in merged:
            if m & g:
                g |= m
            else:
                keep.append(m)
        keep.append(g)
        merged = keep
    return merged


def _iter_anchor_chunks(
    group_blocks: List[Tuple[List[str], List[tuple]]],
    pinned: Dict[str, Any],
    n_rows: int,
    chunk_rows: int = _ANCHOR_CHUNK_ROWS,
) -> "Iterator[pd.DataFrame]":
    """Yield the anchor frame in chunks for Patsy's incremental coding scan.

    Each *group_blocks* entry is ``(column_names, value_tuples)``: the value
    tuples cycle independently per group, so factors combined inside one
    derived expression stay jointly enumerated while everything else only
    needs its own levels covered."""
    for start in range(0, n_rows, chunk_rows):
        size = min(chunk_rows, n_rows - start)
        frame: Dict[str, list] = {}
        for names, block in group_blocks:
            for i, nm in enumerate(names):
                frame[nm] = [block[(start + r) % len(block)][i]
                             for r in range(size)]
        for nm, val in pinned.items():
            frame[nm] = [val] * size
        yield pd.DataFrame(frame)


def contrast_from_scenarios(
    formula: str,
    factors: FactorSpec,
    scenario_a: Scenario,
    scenario_b: Scenario,
    sesoi: float,
    *,
    candidate_points: int = 10,
    seed: int = 0,
    coding_data: Optional[pd.DataFrame] = None,
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
        derived from an incremental Patsy scan (TD-7), not a sampled
        candidate row.
    seed : int, default 0
        Unused; retained for backward compatibility (see candidate_points).
    coding_data : DataFrame, optional
        Authoritative data the model coding is learned from EXCLUSIVELY —
        pass the realized candidate set (``build_candidate(factors, ...)``)
        or an existing design (``result["design_df"]``). The scenario rows
        are then transformed against that fixed coding; they cannot widen
        the model or shift stateful transform parameters, and a scenario
        using a level absent from the coding data raises. Required when the
        formula (a) derives CATEGORICAL levels from continuous data (e.g.
        ``C(I(x // 1))``) or (b) uses stateful Patsy transforms (``bs``,
        ``cr``, ``cc``, ``te``, ``center``, ``standardize``, ``scale``),
        because both learn their coding from realized data that an
        internally generated anchor cannot reproduce.

    Returns
    -------
    (L, delta) : (np.ndarray, np.ndarray)
        ``L`` has shape ``(1, p)``; ``delta`` has shape ``(1,)``.

    Notes
    -----
    The model coding is established with :func:`patsy.incr_dbuilder`, which
    scans the anchor data in chunks WITHOUT materializing a model matrix, and
    only the two scenario rows are ever built (``2 × p``). Memory therefore
    stays flat even when a large categorical cross implies thousands of model
    columns.

    Factor references are taken from Patsy's parsed factor expressions (plus
    ``Q("...")`` decoding), not from raw text matching. Only factors whose
    values are COMBINED inside one derived expression (e.g. ``C(a + b)``) are
    jointly enumerated; conventional main effects and ``:``/``*``
    interactions build their columns structurally from per-factor codings,
    so they need only a cycling level cover. Unreferenced factors cannot
    affect the coding and are pinned to a single value.
    """
    # Resolve discriminated factor-spec dict forms before validation (UX-5).
    factors = normalize_factors(factors, formula)
    _validate_scenario("scenario_a", scenario_a, factors)
    _validate_scenario("scenario_b", scenario_b, factors)
    if not isinstance(sesoi, (int, float)) or sesoi <= 0:
        raise ValueError(
            f"sesoi must be a positive number, but got {sesoi}."
        )

    scenarios_df = pd.concat(
        [pd.DataFrame([scenario_a]), pd.DataFrame([scenario_b])],
        ignore_index=True,
    )

    if coding_data is not None:
        # Authoritative path: the coding is learned EXCLUSIVELY from the
        # caller-supplied realized data (candidate set or design). Scenario
        # rows are transformed against that fixed coding afterwards — they
        # must not be able to widen the model or shift stateful transform
        # parameters (UX-40); a scenario level absent from the coding data is
        # an error, raised below.
        def _iter_maker() -> Iterator[pd.DataFrame]:
            return iter([coding_data])
    else:
        # Internal anchor (TD-7): every level of every REFERENCED categorical
        # factor must be visible to Patsy, or dummy columns get silently
        # dropped and L comes out narrower than the design model. The anchor
        # is streamed in chunks (see Notes), never materialized as a model
        # matrix. Unreferenced factors are pinned: they cannot affect the
        # coding, so they impose no cost (UX-38).
        codes = _formula_factor_codes(formula)
        code_names = [_code_identifiers(c) for c in codes]
        referenced = {
            name for name in factors
            if any(name in names for names in code_names)
        }

        # Stateful transforms (bs, cr, center, …) LEARN their parameters from
        # the data, so an internal anchor would yield a same-width but
        # numerically different contrast than the realized design coding —
        # silently changing the estimand (UX-41).
        _stateful_used = sorted(
            set().union(*code_names) & _STATEFUL_TRANSFORMS
        ) if code_names else []
        if _stateful_used:
            raise ValueError(
                f"Formula uses stateful Patsy transform(s) "
                f"{_stateful_used} whose coding parameters are learned from "
                "the data. An internally generated anchor cannot reproduce "
                "the realized design's coding, so the contrast would be "
                "silently wrong. Pass coding_data= with the realized "
                "candidate set (build_candidate(factors, ...)) or an "
                "existing design (result['design_df'])."
            )

        # A categorical DERIVED from continuous data cannot be anchored from
        # a static frame: the derived levels depend on the realized values.
        for code, names in zip(codes, code_names):
            if "C" not in names:
                continue
            for name in names:
                if name in factors and _spec_is_continuous(factors[name]):
                    raise ValueError(
                        f"Formula derives categorical levels from continuous "
                        f"factor '{name}' (term {code!r}). The set of "
                        "derived levels depends on the realized data, so an "
                        "internally generated anchor cannot guarantee a "
                        "correct contrast. Pass coding_data= with the "
                        "realized candidate set (build_candidate(factors, "
                        "...)) or an existing design (result['design_df']), "
                        "or construct L explicitly."
                    )

        cat_ref = {
            k: list(v) for k, v in factors.items()
            if k in referenced and not _spec_is_continuous(factors[k])
        }

        # Only factors whose values are COMBINED inside a single derived
        # expression (e.g. C(a + b)) need joint enumeration; conventional
        # main effects and ':'/'*' interactions build their columns
        # structurally from per-factor codings, so a cycling level cover is
        # exact for them (UX-42). Overlapping combination groups are merged.
        combo_groups = _merge_groups([
            {n for n in names if n in cat_ref}
            for names in code_names
            if len({n for n in names if n in cat_ref}) >= 2
        ])
        grouped = set().union(*combo_groups) if combo_groups else set()
        groups: List[List[str]] = [sorted(g) for g in combo_groups] + [
            [k] for k in cat_ref if k not in grouped
        ]

        group_blocks: List[Tuple[List[str], List[tuple]]] = []
        n_rows = 1
        for gnames in groups:
            block_size = 1
            for nm in gnames:
                block_size *= max(len(cat_ref[nm]), 1)
            if block_size > _CONTRAST_CROSS_CAP:
                raise ValueError(
                    "contrast_from_scenarios cannot anchor this model: the "
                    f"factors {gnames} are combined inside one model term, "
                    f"and their level cross exceeds {_CONTRAST_CROSS_CAP:,} "
                    "combinations. Pass coding_data= with the realized "
                    "candidate set or design, reduce the number of levels, "
                    "or construct L explicitly."
                )
            block = list(itertools.product(*(cat_ref[nm] for nm in gnames)))
            group_blocks.append((gnames, block))
            n_rows = max(n_rows, len(block))

        # Pin everything that is not a referenced categorical: continuous
        # factors at their midpoint, unreferenced categoricals at their first
        # level.
        pinned: Dict[str, Any] = {}
        for k, v in factors.items():
            if k in cat_ref:
                continue
            if _spec_is_continuous(v):
                pinned[k] = (float(v[0]) + float(v[1])) / 2.0
            else:
                pinned[k] = list(v)[0]

        def _iter_maker() -> Iterator[pd.DataFrame]:
            return _iter_anchor_chunks(group_blocks, pinned, n_rows)

    try:
        design_info = patsy.incr_dbuilder(formula, _iter_maker)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to build the model coding. "
            f"This can happen if the formula causes patsy errors "
            f"(e.g., divide by zero in formula like 'I(1/A)'). "
            f"Patsy error: {e}"
        ) from e

    try:
        (X_sc,) = patsy.build_design_matrices([design_info], scenarios_df)
    except Exception as e:
        raise ValueError(
            "Failed to evaluate the scenarios against the model coding. "
            "This usually means a scenario uses a categorical level (or a "
            "derived level) that is absent from the coding data — the "
            "authoritative model has no column for it. Extend coding_data "
            "to include that level, or change the scenario. "
            f"Patsy error: {e}"
        ) from e

    X_sc = np.asarray(X_sc)
    x_a = X_sc[0, :]
    x_b = X_sc[1, :]

    L = (x_b - x_a).reshape(1, -1)
    delta = np.array([float(sesoi)], dtype=float)
    return L, delta


__all__ = ["contrast_from_scenarios"]