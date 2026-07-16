# contrasts.py
# License: MIT
"""
Convenience builders for contrasts
---------------------------------

Helpers to build contrast matrices `L` and SESOI vectors `delta` from
human-friendly descriptions (e.g., two named scenarios in factor space).

Public API
----------
- contrast_from_scenarios(...): single-row L (1 x p) for a pairwise difference
- coding_is_data_dependent(...): why a formula needs coding_data, or None
- ContrastCodingError: raised when the coding cannot be derived from the spec

Implementation note
-------------------
To guarantee that the model coding (dummy columns, interactions, etc.) used to
construct scenario rows matches the coding used during design generation, the
coding is established with :func:`patsy.incr_dbuilder` over an anchor that
covers every level of every formula-referenced categorical factor (streamed in
chunks — no anchor model matrix is ever materialized, TD-7/UX-36). Only the
2 × p scenario matrix is built; L = x_b − x_a.

Factors are enumerated jointly only where they are combined inside a single
*categorical-valued* derived term such as ``C(a + b)``, whose level set is the
set of realized combinations. Main effects, ``:``/``*`` interactions (UX-42)
and numeric derived terms such as ``I(a + b)`` (UX-45) code structurally, so a
per-factor level cover is exact — and far cheaper — for them. Each term
is scanned in its OWN segment, so overlapping terms cost the sum of their
crosses rather than the product of their union (UX-47).

Some codings cannot be derived from the factor spec at all, because they are
learned from the data: stateful transforms (``bs``, ``center``, …) and
categoricals derived from continuous factors. Those raise
:class:`ContrastCodingError` unless the caller passes ``coding_data``, which
then becomes the exclusive coding authority.
"""
from __future__ import annotations

import ast
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


#: Default remedy for :class:`ContrastCodingError`, phrased for the Python API.
PY_CODING_REMEDY = (
    "Pass coding_data= with the realized candidate set "
    "(build_candidate(factors, ...)) or an existing design "
    "(result['design_df']), or construct L explicitly."
)

#: Python-facing default remedies for scenario_contrast_for_run's refusals;
#: interfaces override them with wording their users can act on (UX-46).
#: Every default must only recommend arguments THIS function accepts —
#: PY_CODING_REMEDY's "pass coding_data=" advice belongs to
#: contrast_from_scenarios and would raise TypeError here.
PY_NO_DESIGN_OPTS_REMEDY = (
    "Pass design_opts= with the run's DesignOptions so the contrast is "
    "coded against the candidate set that run will search, or call "
    "contrast_from_scenarios(..., coding_data=...) directly with a "
    "realized candidate set or design, or construct L explicitly."
)
PY_RUN_CODING_REMEDY = (
    "Construct L explicitly, or call "
    "contrast_from_scenarios(..., coding_data=...) directly with a "
    "realized candidate set or design."
)
PY_SPLIT_PLOT_REMEDY = (
    "This run uses split-plot options, and a split-plot search learns its "
    "coding from separately built whole-plot/sub-plot pools — no candidate "
    "set built up front can be its authority. Construct L explicitly, or "
    "use a formula whose coding does not depend on the data."
)
PY_GROWTH_REMEDY = (
    "allow_candidate_growth lets the search rebuild the candidate set "
    "mid-run and re-derive the coding while L stays fixed, so the contrast "
    "would silently stop matching the model. Disable candidate growth, or "
    "construct L explicitly."
)


class ContrastCodingError(ValueError):
    """The model coding cannot be reproduced from the factor spec alone.

    Raised when the formula's coding is learned from realized data (stateful
    transforms, categoricals derived from continuous factors) or when the
    level cross needed to anchor it is too large.

    The message is split into a *reason* (what about the formula makes the
    coding unreproducible — the same for every caller) and a *remedy* (what to
    do about it — different for every caller). Interfaces that cannot supply
    ``coding_data``, such as the CLI and the Streamlit scenario builder,
    re-raise with their own remedy rather than repeating advice the user
    cannot act on (UX-46)::

        except ContrastCodingError as exc:
            raise ContrastCodingError(exc.reason, MY_REMEDY) from exc

    Subclasses :class:`ValueError` for backward compatibility.
    """

    def __init__(self, reason: str, remedy: str = PY_CODING_REMEDY) -> None:
        self.reason = reason
        self.remedy = remedy
        super().__init__(f"{reason} {remedy}".strip())


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


#: Spline transforms whose coding is FULLY SPECIFIED — hence data-independent
#: — when the call supplies literal knots and both bounds (UX-55).
_SPLINE_TRANSFORMS = frozenset({"bs", "cr", "cc"})


def _is_literal_node(node: "ast.AST") -> bool:
    """True for a non-None constant, a signed constant, or a list/tuple of them.

    ``None`` is deliberately NOT a literal here: in every parameter this
    predicate inspects (``knots=``, ``lower_bound=``, ``upper_bound=``,
    ``levels=``), an explicit ``None`` means "learn it from the data" — the
    opposite of a fixed coding contract (UX-59)."""
    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant) and node.operand.value is not None
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_is_literal_node(e) for e in node.elts)
    return False


def _call_name(node: "ast.Call") -> Optional[str]:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _learned_stateful_calls(code: str) -> set:
    """Stateful-transform calls in *code* that still have parameters to LEARN.

    ``bs``/``cr``/``cc`` with literal ``knots``, ``lower_bound`` and
    ``upper_bound`` are fully specified — the coding is identical for any
    input data, so they are NOT learned (UX-55). A ``constraints=`` argument
    on cr/cc re-introduces data dependence (the constraint matrix is computed
    from the data), so it disqualifies. Everything else in
    :data:`_STATEFUL_TRANSFORMS` — center, standardize, scale, te, and any
    spline whose knots or bounds are left to the data or supplied as
    non-literal expressions — counts as learned."""
    try:
        tree = ast.parse(code.strip(), mode="eval")
    except SyntaxError:
        # Conservative fallback: any call-shaped stateful name is learned.
        return set(
            re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
        ) & _STATEFUL_TRANSFORMS
    learned: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in _STATEFUL_TRANSFORMS:
            continue
        if name in _SPLINE_TRANSFORMS:
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            fully_specified = (
                all(
                    k in kw and _is_literal_node(kw[k])
                    for k in ("knots", "lower_bound", "upper_bound")
                )
                and "constraints" not in kw
            )
            if fully_specified:
                continue
        learned.add(name)
    return learned


def _c_call_with_explicit_levels(code: str) -> bool:
    """True when *code* is a top-level ``C(..., levels=[...])`` call whose
    level set is a literal — an explicit, data-independent coding contract
    (UX-56)."""
    try:
        tree = ast.parse(code.strip(), mode="eval")
    except SyntaxError:
        return False
    node = tree.body
    if not (isinstance(node, ast.Call) and _call_name(node) == "C"):
        return False
    for k in node.keywords:
        if k.arg == "levels" and _is_literal_node(k.value):
            return True
    return False


def _code_calls(code: str) -> set:
    """Names of functions *called* in a factor-code expression.

    ``scale(x)`` reports ``{"scale"}``; a bare factor named ``scale`` reports
    nothing. Keeping the two apart is what lets a factor whose name collides
    with a Patsy transform stay usable (UX-44)."""
    try:
        tree = ast.parse(code.strip(), mode="eval")
    except SyntaxError:
        # Conservative fallback: anything in call position.
        return set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", code))
    calls: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                calls.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                calls.add(fn.attr)
    return calls


def _code_identifiers(code: str) -> set:
    """Data names a factor-code expression can reference.

    Names in call position (``scale`` in ``scale(x)``) are excluded: they
    resolve to transforms, not to data columns. ``Q("...")``-quoted names are
    included — they resolve via the data and may contain characters that are
    not valid Python identifiers."""
    names: set = set()
    try:
        tree = ast.parse(code.strip(), mode="eval")
    except SyntaxError:
        try:
            from patsy.eval import ast_names

            names |= set(ast_names(code))
        except Exception:
            # Conservative fallback: any identifier-shaped token.
            names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))
    else:
        called = {n.func for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node not in called:
                names.add(node.id)
    for m in re.finditer(r"""Q\(\s*['"](.+?)['"]\s*\)""", code):
        names.add(m.group(1))
    return names


def _coding_dependency_reason(
    factors: FactorSpec,
    codes: List[str],
    code_names: List[set],
    code_calls: List[set],
) -> Optional[str]:
    """Why the coding cannot be derived from *factors* alone, or None.

    Takes the already-parsed factor codes so the caller does not pay to parse
    the formula twice; :func:`coding_is_data_dependent` is the public wrapper.
    """
    # Stateful transforms (bs, cr, center, …) LEARN their parameters from the
    # data, so an internal anchor would yield a same-width but numerically
    # different contrast than the realized design coding — silently changing
    # the estimand (UX-41). Only an actual CALL counts (a factor merely
    # *named* `scale` codes like any other column, UX-44), and only a call
    # with parameters left to learn: a spline with literal knots and bounds
    # codes identically on any data, so it is NOT data-dependent (UX-55).
    stateful = sorted(
        set().union(*(_learned_stateful_calls(c) for c in codes))
    ) if codes else []
    if stateful:
        return (
            f"Formula uses stateful Patsy transform(s) {stateful} whose "
            "coding parameters (knots, means, scales) are learned from the "
            "data, so the model coding cannot be derived from the factor "
            "specification alone — a guessed anchor would give a same-width "
            "but numerically different contrast, with nothing to signal it. "
            "(A spline with explicit literal knots=, lower_bound= and "
            "upper_bound= is fully specified and is not affected.)"
        )

    # A categorical DERIVED from continuous data cannot be anchored from a
    # static frame: the derived levels depend on the realized values. Decided
    # by EVALUATING the term on a range-spanning probe and reading Patsy's own
    # FactorInfo.type — syntax is not enough, because Patsy also treats
    # object-valued results as categorical without any C(...) call, e.g.
    # ``I(np.where(x < 0.5, "lo", "hi"))`` (UX-51). The C-call test survives
    # only as a conservative fallback for terms the probe cannot evaluate.
    _probe: Optional[pd.DataFrame] = None
    for code, names, calls in zip(codes, code_names, code_calls):
        cont_refs = sorted(
            n for n in names
            if n in factors and _spec_is_continuous(factors[n])
        )
        if not cont_refs:
            # Terms over categorical factors only: every level combination is
            # enumerable from the spec, so the coding is derivable.
            continue
        if _c_call_with_explicit_levels(code):
            # C(..., levels=[...]) with a literal level set is an explicit
            # coding contract — the columns are fixed regardless of which
            # values the data realizes (UX-56). A realized value outside the
            # declared levels fails loudly at transform time.
            continue
        if _probe is None:
            _probe = _spanning_probe(factors)
        is_cat = _derived_result_is_categorical(code, _probe)
        if is_cat is True or (is_cat is None and "C" in calls):
            return (
                f"Formula derives categorical levels from continuous "
                f"factor(s) {cont_refs} (term {code!r}). Which levels exist "
                "depends on the realized data, so the model coding cannot "
                "be derived from the factor specification alone. (An "
                "explicit literal C(..., levels=[...]) fixes the level set "
                "and is not affected.)"
            )
    return None


def _spanning_probe(factors: FactorSpec, rows: int = 5) -> "pd.DataFrame":
    """A tiny frame spanning each factor's range, for result-TYPE probes.

    Continuous factors get evenly spaced values across [low, high] — a single
    pinned value would let a thresholding expression such as
    ``np.where(x < 0.5, ...)`` realize only one branch. Categorical factors
    cycle their levels. The dtype of an expression does not depend on how
    many rows it sees, so a handful is enough."""
    cols: Dict[str, list] = {}
    for k, v in factors.items():
        # Any-typed local: after runtime classification the spec is indexed
        # positionally / iterated, which mypy cannot narrow through the union.
        spec: Any = v
        if _spec_is_continuous(spec):
            lo, hi = float(spec[0]), float(spec[1])
            cols[k] = list(np.linspace(lo, hi, rows))
        else:
            levels = list(spec)
            cols[k] = [levels[i % len(levels)] for i in range(rows)]
    return pd.DataFrame(cols)


def coding_is_data_dependent(
    formula: str, factors: FactorSpec
) -> Optional[str]:
    """Why *formula*'s coding cannot be derived from *factors* alone, or None.

    Returns a human-readable reason string when :func:`contrast_from_scenarios`
    would need authoritative ``coding_data`` — the formula uses a stateful
    Patsy transform, or derives categorical levels from a continuous factor —
    and ``None`` when the factor specification is sufficient on its own.

    Interfaces use this to decide *up front* whether they must supply
    ``coding_data``, and whether options that would change the coding mid-run
    (candidate growth) are safe to leave enabled (UX-48).

    Examples
    --------
    >>> coding_is_data_dependent("~ 1 + C(g)", {"g": ["a", "b"]}) is None
    True
    >>> "stateful" in coding_is_data_dependent("~ bs(x, df=3)", {"x": (0, 1)})
    True
    """
    factors = normalize_factors(factors, formula)
    codes = _formula_factor_codes(formula)
    return _coding_dependency_reason(
        factors,
        codes,
        [_code_identifiers(c) for c in codes],
        [_code_calls(c) for c in codes],
    )


def _derived_result_is_categorical(
    code: str, probe: "pd.DataFrame"
) -> Optional[bool]:
    """Whether a factor-code expression *evaluates* to a categorical column.

    Only a categorical result needs joint level enumeration: its level set is
    the set of realized value combinations, so every combination must be
    visible to Patsy. A numeric derived column (``I(a + b)``) contributes
    exactly one column no matter which combinations appear, so a per-factor
    per-factor level cover codes it exactly (UX-45).

    Note this is decided by evaluating the expression, not by its syntax:
    ``I(a + b)`` is numeric over numeric-coded levels but is a string
    concatenation — hence categorical — over string levels.

    Returns ``None`` when the expression cannot be evaluated on *probe*, which
    the caller treats conservatively (assume categorical)."""
    try:
        design_info = patsy.incr_dbuilder("0 + " + code, lambda: iter([probe]))
        infos = list(design_info.factor_infos.values())
    except Exception:
        return None
    if len(infos) != 1:
        return None
    return bool(infos[0].type == "categorical")


def _iter_anchor_chunks(
    group_blocks: List[Tuple[List[str], List[tuple]]],
    base_row: Dict[str, Any],
    chunk_rows: int = _ANCHOR_CHUNK_ROWS,
) -> "Iterator[pd.DataFrame]":
    """Yield the anchor frame in chunks for Patsy's incremental coding scan.

    Each *group_blocks* entry is ``(column_names, value_tuples)`` and gets its
    OWN segment of rows, with every other factor held at its *base_row* value.
    Because Patsy accumulates levels across the whole scan, each derived term
    still sees every combination it can take, while overlapping terms — say
    ``C(a + b)`` and ``C(b + c)`` — cost the SUM of their two crosses instead
    of the product of their union (UX-47).

    Factors that only ever code structurally (main effects, ``:``/``*``
    interactions) get a one-factor group, so their levels are covered in a
    segment of their own."""
    if not group_blocks:
        yield pd.DataFrame({k: [v] for k, v in base_row.items()})
        return
    for names, block in group_blocks:
        for start in range(0, len(block), chunk_rows):
            rows = block[start:start + chunk_rows]
            frame: Dict[str, list] = {
                k: [v] * len(rows) for k, v in base_row.items()
            }
            for i, nm in enumerate(names):
                frame[nm] = [r[i] for r in rows]
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

    Raises
    ------
    ContrastCodingError
        If the coding cannot be derived from *factors* alone and no
        ``coding_data`` was supplied. See the class docstring for how callers
        that cannot supply it re-phrase the remedy.
    KeyError
        If a scenario omits a factor or names an unknown one.
    ValueError
        If a scenario value is out of range or not a declared level, or if
        ``sesoi`` is not positive.

    Notes
    -----
    The model coding is established with :func:`patsy.incr_dbuilder`, which
    scans the anchor data in chunks WITHOUT materializing a model matrix, and
    only the two scenario rows are ever built (``2 × p``). Memory therefore
    stays flat even when a large categorical cross implies thousands of model
    columns.

    Factor references are taken from Patsy's parsed factor expressions (plus
    ``Q("...")`` decoding), not from raw text matching, and function names are
    distinguished from column names — a factor named ``scale`` is a factor,
    not a transform (UX-44).

    Only factors COMBINED inside one derived expression that itself evaluates
    to a CATEGORICAL column (e.g. ``C(a + b)``) are jointly enumerated, since
    there the level set is the set of realized combinations. Main effects,
    ``:``/``*`` interactions and numeric derived terms such as ``I(a + b)``
    code structurally, so a per-factor level cover is exact for them (UX-42,
    UX-45). Note this is decided by evaluating the term, not by its syntax:
    ``I(a + b)`` is numeric over numeric levels but is a string concatenation
    — hence categorical — over string levels. Unreferenced factors cannot
    affect the coding and are pinned to a single value.

    Examples
    --------
    >>> L, delta = contrast_from_scenarios(
    ...     "~ 1 + C(g) + x", {"g": ["a", "b"], "x": (0.0, 1.0)},
    ...     {"g": "a", "x": 0.0}, {"g": "b", "x": 0.0}, sesoi=0.5,
    ... )
    >>> L.shape, delta
    ((1, 3), array([0.5]))
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
        code_calls = [_code_calls(c) for c in codes]
        referenced = {
            name for name in factors
            if any(name in names for names in code_names)
        }

        # Codings that are learned from realized data cannot be reproduced
        # from the spec; the caller must name an authority instead.
        _reason = _coding_dependency_reason(
            factors, codes, code_names, code_calls
        )
        if _reason:
            raise ContrastCodingError(_reason)

        cat_ref = {
            k: list(v) for k, v in factors.items()
            if k in referenced and not _spec_is_continuous(factors[k])
        }

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

        # Small frame used only to decide the RESULT TYPE of derived terms.
        # A handful of rows is enough — the dtype of an expression does not
        # depend on how many combinations it sees.
        _probe_rows = min(
            max((len(v) for v in cat_ref.values()), default=1), 8
        )
        probe = pd.DataFrame({
            **{k: [v[i % len(v)] for i in range(_probe_rows)]
               for k, v in cat_ref.items()},
            **{k: [v] * _probe_rows for k, v in pinned.items()},
        })

        # Only factors whose values are COMBINED inside a single derived
        # expression that is itself CATEGORICAL (e.g. C(a + b)) need joint
        # enumeration — there the level set is the set of realized
        # combinations. Conventional main effects and ':'/'*' interactions
        # build their columns structurally from per-factor codings (UX-42),
        # and a numeric derived column such as I(a + b) is one column
        # regardless (UX-45); a per-factor level cover is exact for both.
        #
        # Each such term keeps its OWN group, scanned in its own segment.
        # Overlapping terms are NOT unioned: C(a + b) + C(b + c) needs the
        # a×b cross and the b×c cross, never the a×b×c cross (UX-47).
        _combo: List[Tuple[List[str], str]] = []   # (factor names, term code)
        _seen: set = set()
        for code, names in zip(codes, code_names):
            group = {n for n in names if n in cat_ref}
            if len(group) < 2:
                continue
            # None (un-evaluable) is treated as categorical: conservative, and
            # the cap below then names coding_data as the way out.
            if _derived_result_is_categorical(code, probe) is False:
                continue
            key = frozenset(group)
            if key in _seen:      # two terms over the same factors: one scan
                continue
            _seen.add(key)
            _combo.append((sorted(group), code))

        grouped = set().union(*(set(g) for g, _ in _combo)) if _combo else set()
        # Structural-only factors each get a one-factor group: their levels
        # need covering, but never jointly with anything else.
        groups: List[Tuple[List[str], Optional[str]]] = _combo + [
            ([k], None) for k in cat_ref if k not in grouped
        ]

        group_blocks: List[Tuple[List[str], List[tuple]]] = []
        for gnames, term_code in groups:
            block_size = 1
            for nm in gnames:
                block_size *= max(len(cat_ref[nm]), 1)
            # The cap guards joint enumeration only. A single factor's own
            # levels are always coverable — the scan is chunked.
            if term_code is not None and block_size > _CONTRAST_CROSS_CAP:
                raise ContrastCodingError(
                    "contrast_from_scenarios cannot anchor this model: the "
                    f"factors {gnames} are combined inside the categorical "
                    f"term {term_code!r}, whose level cross ({block_size:,}) "
                    f"exceeds {_CONTRAST_CROSS_CAP:,} combinations, so its "
                    "derived level set cannot be enumerated."
                )
            block = list(itertools.product(*(cat_ref[nm] for nm in gnames)))
            group_blocks.append((gnames, block))

        # Every factor gets a resting value; each segment overrides only its
        # own group's columns.
        base_row: Dict[str, Any] = dict(pinned)
        for k, levels in cat_ref.items():
            base_row[k] = levels[0]

        def _iter_maker() -> Iterator[pd.DataFrame]:
            return _iter_anchor_chunks(group_blocks, base_row)

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
            "Either a scenario uses a categorical level (or a derived level) "
            "that is absent from the coding data — the authoritative model "
            "has no column for it — or a continuous scenario value lies "
            "outside the range the coding data covers, which a spline term "
            "(bs/cr/cc) cannot extrapolate beyond its outermost knots. Note "
            "a sampled candidate set does not quite reach a factor's declared "
            "bounds, so a scenario sitting exactly on a bound can fall "
            "outside them; move the scenario inside the realized range. (Do "
            "NOT extend coding_data with extra rows to cover the scenario: "
            "the design search still builds its own candidate, and added "
            "rows shift stateful coding parameters such as spline knots, so "
            "L would silently stop matching the design's model.) "
            f"Patsy error: {e}"
        ) from e

    X_sc = np.asarray(X_sc)
    x_a = X_sc[0, :]
    x_b = X_sc[1, :]

    L = (x_b - x_a).reshape(1, -1)
    delta = np.array([float(sesoi)], dtype=float)
    return L, delta


def scenario_contrast_for_run(
    formula: str,
    factors: FactorSpec,
    scenario_a: Scenario,
    scenario_b: Scenario,
    sesoi: float,
    *,
    design_opts: Optional[Any] = None,
    sizing_formula: Optional[str] = None,
    remedies: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scenario contrast coded against the run's own candidate set.

    THE one decision tree for every interface (UX-48/50/59; previously
    duplicated between the CLI and the Streamlit app, which let the two
    drift): when the formula's coding is learned from realized data, the
    coding authority must be the candidate set the run will actually
    search — built here from the run's own *design_opts* — and
    configurations whose run cannot honor that authority are refused up
    front:

    * no *design_opts* (a preview with no run configured) — nothing built
      here can be the authority;
    * split-plot runs — their coding comes from separately built
      whole-plot/sub-plot pools, not the ordinary candidate set;
    * candidate growth — the search may rebuild the candidate set mid-run
      and re-derive the coding while L stays fixed.

    ``sizing_formula`` is the formula the run sizes its candidate set by.
    It matters in multi-response mode, where a response may carry its own
    *formula* but the run builds ONE candidate set from the global formula
    — sizing here by the response's formula would break the
    shared-authority invariant the moment candidate sizing starts reading
    the formula.

    ``remedies`` maps ``"preview"`` / ``"split_plot"`` / ``"growth"`` /
    ``"coding"`` to interface-specific advice (UX-46); unset keys fall
    back to the Python-facing defaults. The *reason* half of every error
    passes through unchanged.
    """
    r = remedies or {}
    reason = coding_is_data_dependent(formula, factors)
    coding_data = None
    if reason is not None:
        if design_opts is None:
            raise ContrastCodingError(
                reason, r.get("preview", PY_NO_DESIGN_OPTS_REMEDY)
            )
        if design_opts.split_plot is not None:
            raise ContrastCodingError(
                reason, r.get("split_plot", PY_SPLIT_PLOT_REMEDY)
            )
        if design_opts.allow_candidate_growth:
            raise ContrastCodingError(
                reason, r.get("growth", PY_GROWTH_REMEDY)
            )
        from .candidate import build_search_candidate

        coding_data, _ = build_search_candidate(
            sizing_formula or formula, factors, design_opts,
        )
    try:
        return contrast_from_scenarios(
            formula, factors, scenario_a, scenario_b, sesoi,
            coding_data=coding_data,
        )
    except ContrastCodingError as exc:
        # Rewrap even without an override: contrast_from_scenarios' own
        # remedy recommends its coding_data= parameter, which this function
        # does not accept — the advice must fit the API that raised it.
        raise ContrastCodingError(
            exc.reason, r.get("coding", PY_RUN_CODING_REMEDY)
        ) from exc


__all__ = [
    "contrast_from_scenarios",
    "coding_is_data_dependent",
    "scenario_contrast_for_run",
    "ContrastCodingError",
]
