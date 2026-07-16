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
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
    Literal,
    overload,
)
import numpy as np


class ContinuousFactorSpec(TypedDict):
    """Explicit continuous factor: ``{"type": "continuous", "low": lo, "high": hi}``."""

    type: Literal["continuous"]
    low: float
    high: float


class CategoricalFactorSpec(TypedDict):
    """Explicit categorical factor: ``{"type": "categorical", "levels": [...]}``.

    Levels may be numeric — this is the only unambiguous way to express a
    numeric-coded category such as ``[0, 1]`` (the legacy shorthand treats
    any two-element numeric sequence as a continuous range).
    """

    type: Literal["categorical"]
    levels: List[Union[int, float, str]]


#: One factor's specification: explicit discriminated dict (recommended) or
#: legacy shorthand — ``(low, high)`` for continuous, ``[level, ...]`` for
#: categorical.
FactorSpecValue = Union[
    ContinuousFactorSpec,
    CategoricalFactorSpec,
    List[Union[int, float, str]],
    Tuple[float, float],
]

#: Factor-name → spec mapping accepted by every public factor-taking API.
#: ``Mapping`` (read-only, covariant in the value type) rather than ``Dict``:
#: the APIs never mutate the caller's dict, and covariance lets callers pass
#: plain ``dict``s of any compatible value type without invariance friction.
FactorSpec = Mapping[str, FactorSpecValue]


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


def _representative_frame(
    factors: Dict[str, Any],
    max_cross_rows: int = 10_000,
) -> Tuple["Any", bool]:
    """Build a frame exposing every categorical level to Patsy.

    Returns ``(frame, cross_exact)``:

    * When the categorical Cartesian cross has at most *max_cross_rows*
      combinations, the frame IS the full cross and ``cross_exact=True`` —
      every CATEGORICAL level combination is materialized, so even derived
      cross-factor terms like ``C(a + b)`` (whose levels depend on
      combinations) are coded completely.
    * Above the cap, the frame is a compact level cover (each categorical
      column cycles its own levels; length = largest level list) and
      ``cross_exact=False``. Patsy derives standard codings per-factor and
      builds ordinary interaction columns structurally, so the compact count
      matches the full cross for conventional formulas — but derived
      cross-factor terms can be undercounted, so callers must treat the
      result as provisional and defer hard dimension checks to a realized
      model matrix.

    NOTE: the flag speaks only for the categorical space. Continuous factors
    are represented by a single midpoint value, so formulas that derive
    columns from continuous DATA (e.g. ``C(I(x // 1))``) can gain columns on
    realized data even when ``cross_exact=True``; callers whose exactness
    contract covers continuous factors must weaken the flag themselves (see
    ``model_matrix_preview``).

    *factors* must already be normalized.
    """
    import itertools

    import pandas as pd

    cat = {k: list(v) for k, v in factors.items() if not _spec_is_continuous(v)}
    cont = {k: v for k, v in factors.items() if _spec_is_continuous(v)}

    for name, levels in cat.items():
        if not levels:
            raise ValueError(
                "Every categorical factor needs at least one level for the "
                f"model preview (factor '{name}' has none)."
            )

    cross_size = 1
    for levels in cat.values():
        cross_size *= len(levels)
        if cross_size > max_cross_rows:
            break

    if cross_size <= max_cross_rows:
        # Full Cartesian cross — exact for any Patsy expression.
        combos = list(itertools.product(*cat.values())) if cat else [()]
        frame = {name: [c[i] for c in combos]
                 for i, name in enumerate(cat.keys())}
        n_rows = len(combos)
        exact = True
    else:
        # Compact level cover — provisional for derived cross-factor terms.
        n_rows = max(len(levels) for levels in cat.values())
        if n_rows > max_cross_rows:
            raise ValueError(
                f"A categorical factor has {n_rows} levels, exceeding the "
                f"preview cap of {max_cross_rows}."
            )
        frame = {
            name: list(itertools.islice(itertools.cycle(levels), n_rows))
            for name, levels in cat.items()
        }
        exact = False

    for name, (lo, hi) in cont.items():
        frame[name] = [(float(lo) + float(hi)) / 2.0] * n_rows

    return pd.DataFrame(frame), exact


@overload
def model_matrix_preview(
    formula: str,
    factors: FactorSpec,
    max_preview_rows: int = ...,
    return_exact: Literal[False] = ...,
) -> Tuple[int, List[str]]: ...


@overload
def model_matrix_preview(
    formula: str,
    factors: FactorSpec,
    max_preview_rows: int = ...,
    *,
    return_exact: Literal[True],
) -> Tuple[int, List[str], bool]: ...


def model_matrix_preview(
    formula: str,
    factors: FactorSpec,
    max_preview_rows: int = 10_000,
    return_exact: bool = False,
) -> Union[Tuple[int, List[str]], Tuple[int, List[str], bool]]:
    """Return (p, column_names) for *formula* over a representative frame.

    Uses the full categorical Cartesian cross when it has at most
    *max_preview_rows* combinations; larger spaces fall back to a compact
    level cover (each categorical column cycles its own levels), which
    matches the full cross for conventional formulas (dummies + interactions
    are built structurally from per-factor codings) but can undercount
    derived cross-factor terms such as ``C(a + b)``. A single-row frame
    containing only the first level of each categorical would undercount p
    for every categorical model (UX-1), which is why a representative frame
    is used at all.

    The reported count is **exact** only when the categorical cross was fully
    materialized AND there are no continuous factors: continuous factors are
    represented by a single midpoint value, so formulas that derive columns
    from continuous data (e.g. ``C(I(x // 1))`` over a range spanning several
    integers) can gain columns on realized data. Pass ``return_exact=True``
    to receive that status; downstream validation in ``find_optimal_design``
    reconciles a provisional count against the realized candidate model
    matrix either way.

    Parameters
    ----------
    formula : str
        Patsy model formula.
    factors : dict
        Factor spec — continuous factors as 2-tuples/lists of numbers,
        categorical factors as lists of levels (package convention);
        discriminated dict forms are normalized.
    max_preview_rows : int, default 10 000
        Largest categorical cross to materialize before switching to the
        compact cover; also caps a single factor's level count.
    return_exact : bool, default False
        When True, return ``(p, column_names, exact)`` so UI/API callers can
        label a provisional count as such.

    Returns
    -------
    (p, column_names) or (p, column_names, exact)
        Model-matrix column count, Patsy column labels, and (optionally)
        whether the count is guaranteed to match any realized model matrix.
    """
    from .model_matrix import build_model_matrix

    factors = normalize_factors(factors)
    frame, cross_exact = _representative_frame(factors, max_cross_rows=max_preview_rows)
    has_continuous = any(_spec_is_continuous(v) for v in factors.values())
    exact = cross_exact and not has_continuous
    X, col_names = build_model_matrix(formula, frame)
    if return_exact:
        return X.shape[1], list(col_names), exact
    return X.shape[1], list(col_names)



def safe_name_slug(
    name: str,
    existing: "Optional[set]" = None,
    maxlen: int = 60,
    prefix: str = "",
) -> str:
    """A filesystem/sheet-safe identifier for a user-supplied label (UX-67).

    Response names are free-form (``ResponseSpec`` accepts ``"Yield/Day"``),
    but they end up inside filenames, Excel sheet titles and widget keys,
    where path separators and other reserved characters raise or corrupt the
    target. Reserved characters become ``_``, leading/trailing dots and
    spaces are stripped, the result is truncated to *maxlen*, and an empty
    result falls back to ``"response"``.

    When *existing* (a set of already-taken names) is given, collisions are
    resolved deterministically by appending ``_2``, ``_3``, … in call order,
    and the chosen name is added to the set. Collision comparison is
    CASE-INSENSITIVE (``casefold``): Windows filenames and Excel worksheet
    titles do not distinguish case, so the distinct response names ``Yield``
    and ``yield`` must not map to case-only-different slugs — that silently
    overwrites files and desyncs sheet indexes (UX-69). Elements already in
    *existing* may be any case; returned slugs keep the original casing.
    Callers keep the ORIGINAL name alongside the slug (e.g. in report
    metadata) so nothing is lost.

    When the slug is embedded in a longer name (Excel/Sheets titles are
    ``MM_<slug>``), pass that *prefix*: collisions are then checked — and
    recorded in *existing* — on the COMPLETE ``prefix + slug`` name, so the
    set may be seeded directly with existing worksheet titles. Checking the
    bare slug against prefixed titles misses the collision, and the
    spreadsheet backend then renames the sheet behind the caller's back
    while the index records the requested title (UX-73). *maxlen* still
    bounds the slug alone; the returned value never includes *prefix*.
    """
    cleaned = "".join(
        "_" if (c in '\\/:*?"<>|[]\'' or ord(c) < 32) else c
        for c in str(name)
    ).strip(". ")
    cleaned = cleaned[:maxlen].strip(". ") or "response"
    if existing is None:
        return cleaned

    def _taken(candidate: str) -> bool:
        folded = (prefix + candidate).casefold()
        return any(folded == e.casefold() for e in existing)

    slug = cleaned
    k = 2
    while _taken(slug):
        suffix = f"_{k}"
        slug = cleaned[: maxlen - len(suffix)] + suffix
        k += 1
    existing.add(prefix + slug)
    return slug

__all__ = [
    "validate_factors",
    "safe_name_slug",
    "initial_n_guess",
    "model_matrix_preview",
    "normalize_factors",
    "FactorSpec",
    "FactorSpecValue",
    "ContinuousFactorSpec",
    "CategoricalFactorSpec",
]
