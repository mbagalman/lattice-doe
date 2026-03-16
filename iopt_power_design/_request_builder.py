# _request_builder.py
# License: MIT
"""Shared request/config translation layer for iopt_power_design.

All interface layers (CLI, Sheets, Excel, Streamlit) normalise their raw input
into plain Python dicts and call the builder functions here to construct the
authoritative config dataclasses.  This eliminates duplicated construction
logic across every interface.

Normalised dict schemas
-----------------------
Each builder accepts a plain ``dict``.  Missing keys fall back to sensible
defaults (matching the underlying dataclass defaults or interface conventions).
Values must already be typed (int, float, bool, str, list) — raw string
parsing belongs in the calling interface, not here.

``constraint_func`` (a Python callable) is **never** accepted here.  Callers
that need a callable constraint must construct :class:`DesignOptions` directly.

``L`` and ``delta`` may be supplied as ``list[list[float]]`` / ``list[float]``,
or as any array-like accepted by ``numpy.asarray``.  The builder always
converts to ``numpy.ndarray``.

``sigma_joint`` in :func:`build_multi_response` may be a
``list[list[float]]`` or ``None``; the builder converts to ``numpy.ndarray``.

Key schemas (all keys optional unless noted)
--------------------------------------------
**build_power_cfg**
    ``power_mode``  (required) ``"contrast"`` | ``"r2"`` | ``"glm"``
    Shared: ``alpha``, ``power``, ``max_n``, ``max_iter``, ``tol_power``
    contrast/glm: ``L`` (required), ``delta`` (required), ``sigma`` (contrast only)
    r2: ``r2_target`` (required), ``sigma``, ``lambda_mode``
    glm: ``baseline`` (required), ``family``, ``link``

**build_split_plot_opts**
    ``htc_factors`` (required), ``n_whole_plots`` (required)
    Optional: ``eta``, ``subplots_per_wp`` (0 → None), ``df_method``

**build_design_opts**
    All keys optional; ``n_blocks`` forwarded only when ≥ 2,
    ``alloc_max_per_cell`` forwarded only when > 0.
    Nested ``split_plot`` dict → :func:`build_split_plot_opts`.

**build_response_spec**
    ``name`` (required), ``power_cfg`` dict (required)
    Optional: ``formula``, ``weight``

**build_multi_response**
    ``responses`` list of response-spec dicts (required, ≥ 2)
    Optional: ``power_combination``, ``sigma_joint``
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from .config import (
    DesignOptions,
    MultiResponseOptions,
    PowerContrastConfig,
    PowerGLMContrastConfig,
    PowerR2Config,
    ResponseSpec,
    SplitPlotOptions,
)

__all__ = [
    "build_power_cfg",
    "build_split_plot_opts",
    "build_design_opts",
    "build_response_spec",
    "build_multi_response",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_L(val: Any, error_cls: Type[Exception], ctx: str) -> np.ndarray:
    """Convert *val* to a float 2-D ndarray; raise *error_cls* if absent/invalid."""
    if val is None:
        raise error_cls(f"{ctx}'L' is required for contrast/glm mode.")
    try:
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except (ValueError, TypeError) as e:
        raise error_cls(f"{ctx}'L' is not a valid numeric matrix: {e}") from e


def _coerce_delta(val: Any, error_cls: Type[Exception], ctx: str) -> np.ndarray:
    """Convert *val* to a float 1-D ndarray; raise *error_cls* if absent/invalid."""
    if val is None:
        raise error_cls(f"{ctx}'delta' is required for contrast/glm mode.")
    try:
        return np.asarray(val, dtype=float).ravel()
    except (ValueError, TypeError) as e:
        raise error_cls(f"{ctx}'delta' is not a valid numeric vector: {e}") from e


def _ctx_prefix(context: str) -> str:
    """Return *context* with a trailing ': ' separator, or empty string."""
    return f"{context}: " if context else ""


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_power_cfg(
    d: Dict[str, Any],
    *,
    error_cls: Type[Exception] = ValueError,
    context: str = "",
) -> Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig]:
    """Construct a power-config dataclass from a normalised dict.

    Parameters
    ----------
    d : dict
        Normalised dict.  ``power_mode`` is required; all other keys are
        optional and fall back to sensible defaults.
    error_cls : type
        Exception class raised on validation errors.  Pass ``SheetsError``,
        ``ExcelError``, etc. for interface-specific error messages.
    context : str
        Human-readable prefix prepended to error messages.

    Returns
    -------
    PowerContrastConfig | PowerR2Config | PowerGLMContrastConfig
    """
    _ctx = _ctx_prefix(context)
    mode = str(d.get("power_mode", "contrast")).lower()

    alpha     = float(d.get("alpha",     0.05))
    power     = float(d.get("power",     0.80))
    max_n     = int(d.get("max_n",       500))
    max_iter  = int(d.get("max_iter",    200))
    tol_power = float(d.get("tol_power", 1e-3))

    if mode == "r2":
        r2_target = d.get("r2_target")
        if r2_target is None:
            raise error_cls(f"{_ctx}r2 mode requires 'r2_target'.")
        sigma       = float(d.get("sigma",       1.0))
        lambda_mode = str(d.get("lambda_mode",   "n"))
        try:
            return PowerR2Config(
                r2_target=float(r2_target),
                alpha=alpha, power=power, sigma=sigma,
                max_n=max_n, max_iter=max_iter, tol_power=tol_power,
                lambda_mode=lambda_mode,
            )
        except (ValueError, TypeError) as e:
            raise error_cls(f"{_ctx}invalid PowerR2Config: {e}") from e

    if mode == "glm":
        L     = _coerce_L(d.get("L"), error_cls, _ctx)
        delta = _coerce_delta(d.get("delta"), error_cls, _ctx)
        baseline = d.get("baseline")
        if baseline is None:
            raise error_cls(f"{_ctx}glm mode requires 'baseline'.")
        family = str(d.get("family", "binomial"))
        link   = d.get("link") or None
        try:
            return PowerGLMContrastConfig(
                L=L, delta=delta,
                baseline=float(baseline),
                family=family, link=link,
                alpha=alpha, power=power,
                max_n=max_n, max_iter=max_iter, tol_power=tol_power,
            )
        except (ValueError, TypeError) as e:
            raise error_cls(f"{_ctx}invalid PowerGLMContrastConfig: {e}") from e

    # Default: contrast
    L     = _coerce_L(d.get("L"), error_cls, _ctx)
    delta = _coerce_delta(d.get("delta"), error_cls, _ctx)
    sigma = float(d.get("sigma", 1.0))
    try:
        return PowerContrastConfig(
            L=L, delta=delta,
            alpha=alpha, power=power, sigma=sigma,
            max_n=max_n, max_iter=max_iter, tol_power=tol_power,
        )
    except (ValueError, TypeError) as e:
        raise error_cls(f"{_ctx}invalid PowerContrastConfig: {e}") from e


def build_split_plot_opts(
    d: Dict[str, Any],
    *,
    error_cls: Type[Exception] = ValueError,
    context: str = "",
) -> SplitPlotOptions:
    """Construct :class:`SplitPlotOptions` from a normalised dict.

    Parameters
    ----------
    d : dict
        Must contain ``htc_factors`` (list[str]) and ``n_whole_plots`` (int).
        ``subplots_per_wp=0`` is treated as ``None`` (auto).
    error_cls : type
        Exception class raised on validation errors.
    context : str
        Human-readable prefix prepended to error messages.
    """
    _ctx = _ctx_prefix(context)
    htc_factors   = list(d.get("htc_factors", []))
    n_whole_plots = int(d.get("n_whole_plots", 4))
    eta           = float(d.get("eta", 1.0))

    spwp_raw = d.get("subplots_per_wp")
    if spwp_raw is None:
        subplots_per_wp: Optional[int] = None
    else:
        spwp_int = int(spwp_raw)
        subplots_per_wp = spwp_int if spwp_int > 0 else None

    df_method = str(d.get("df_method", "auto"))

    try:
        return SplitPlotOptions(
            htc_factors=htc_factors,
            n_whole_plots=n_whole_plots,
            eta=eta,
            subplots_per_wp=subplots_per_wp,
            df_method=df_method,
        )
    except (ValueError, TypeError) as e:
        raise error_cls(f"{_ctx}invalid SplitPlotOptions: {e}") from e


def build_design_opts(
    d: Dict[str, Any],
    *,
    error_cls: Type[Exception] = ValueError,
    context: str = "",
) -> DesignOptions:
    """Construct :class:`DesignOptions` from a normalised dict.

    All keys are optional; defaults match the :class:`DesignOptions` field
    defaults.

    Notes
    -----
    ``constraint_func`` (a Python callable) is **never** accepted here.
    Pass ``constraint_expr`` (a string) or construct :class:`DesignOptions`
    directly for callable constraints.

    ``n_blocks`` is forwarded only when ≥ 2.
    ``alloc_max_per_cell`` is forwarded only when > 0.
    A nested ``split_plot`` dict is built via :func:`build_split_plot_opts`.

    Parameters
    ----------
    d : dict
        Normalised dict with typed Python values.
    error_cls : type
        Exception class raised on validation errors.
    context : str
        Human-readable prefix prepended to error messages.
    """
    _ctx = _ctx_prefix(context)

    kwargs: Dict[str, Any] = dict(
        criterion              = str(d.get("criterion",           "I")),
        starts                 = int(d.get("starts",              5)),
        max_iter               = int(d.get("max_iter",            1000)),
        random_state           = int(d.get("random_state",        123)),
        candidate_points       = int(d.get("candidate_points",    2000)),
        auto_candidate         = bool(d.get("auto_candidate",     False)),
        cand_min               = int(d.get("cand_min",            1000)),
        cand_max               = int(d.get("cand_max",            10000)),
        cat_cells_cap          = int(d.get("cat_cells_cap",       10000)),
        per_cell_alpha         = float(d.get("per_cell_alpha",    1.5)),
        per_cell_min           = int(d.get("per_cell_min",        5)),
        per_cell_max           = int(d.get("per_cell_max",        20)),
        allow_candidate_growth = bool(d.get("allow_candidate_growth", False)),
        growth_factor          = float(d.get("growth_factor",     2.0)),
        xtx_jitter             = float(d.get("xtx_jitter",        1e-8)),
        algo                   = str(d.get("algo",                "fedorov")),
        parallel_seed_stride   = int(d.get("parallel_seed_stride", 10000)),
        workers                = d.get("workers"),          # None is valid
        block_factor_name      = str(d.get("block_factor_name",  "Block")),
        preallocate_categorical= bool(d.get("preallocate_categorical", False)),
        alloc_min_per_cell     = int(d.get("alloc_min_per_cell", 1)),
    )

    # Conditional: n_blocks (only when ≥ 2)
    n_blocks = d.get("n_blocks")
    if n_blocks is not None and int(n_blocks) >= 2:
        kwargs["n_blocks"] = int(n_blocks)

    # Conditional: block_sizes (only when provided)
    block_sizes = d.get("block_sizes")
    if block_sizes is not None:
        kwargs["block_sizes"] = list(block_sizes)

    # Conditional: alloc_max_per_cell (only when > 0)
    alloc_max = d.get("alloc_max_per_cell")
    if alloc_max is not None and int(alloc_max) > 0:
        kwargs["alloc_max_per_cell"] = int(alloc_max)

    # Conditional: constraint_expr (only when non-empty)
    constraint_expr = d.get("constraint_expr")
    if constraint_expr:
        kwargs["constraint_expr"] = str(constraint_expr)

    # Nested split-plot options
    sp_d = d.get("split_plot")
    if sp_d is not None and isinstance(sp_d, dict):
        kwargs["split_plot"] = build_split_plot_opts(
            sp_d, error_cls=error_cls, context=context
        )

    try:
        return DesignOptions(**kwargs)
    except (ValueError, TypeError) as e:
        raise error_cls(f"{_ctx}invalid DesignOptions: {e}") from e


def build_response_spec(
    d: Dict[str, Any],
    *,
    error_cls: Type[Exception] = ValueError,
    context: str = "",
) -> ResponseSpec:
    """Construct a :class:`ResponseSpec` from a normalised dict.

    The nested ``power_cfg`` dict is built via :func:`build_power_cfg`.

    Parameters
    ----------
    d : dict
        Must contain ``name`` (str) and ``power_cfg`` (dict).
        Optional: ``formula`` (str | None), ``weight`` (float, default 1.0).
    error_cls : type
        Exception class raised on validation errors.
    context : str
        Human-readable prefix prepended to error messages.
    """
    _ctx = _ctx_prefix(context)
    name = str(d.get("name", "")).strip()
    if not name:
        raise error_cls(f"{_ctx}response spec requires a non-empty 'name'.")

    pcfg_d = d.get("power_cfg")
    if not isinstance(pcfg_d, dict):
        raise error_cls(f"{_ctx}response '{name}': 'power_cfg' must be a dict.")

    pcfg = build_power_cfg(
        pcfg_d,
        error_cls=error_cls,
        context=f"{context}: response '{name}'" if context else f"response '{name}'",
    )
    formula = d.get("formula") or None
    weight  = float(d.get("weight", 1.0))

    try:
        return ResponseSpec(name=name, power_cfg=pcfg, formula=formula, weight=weight)
    except (ValueError, TypeError) as e:
        raise error_cls(f"{_ctx}response '{name}': {e}") from e


def build_multi_response(
    d: Dict[str, Any],
    *,
    error_cls: Type[Exception] = ValueError,
    context: str = "",
) -> MultiResponseOptions:
    """Construct :class:`MultiResponseOptions` from a normalised dict.

    Parameters
    ----------
    d : dict
        Must contain ``responses`` (list of response-spec dicts, ≥ 2 entries).
        Optional: ``power_combination`` (default ``"min"``),
        ``sigma_joint`` (list[list[float]] | None, default ``None``).
    error_cls : type
        Exception class raised on validation errors.
    context : str
        Human-readable prefix prepended to error messages.
    """
    _ctx = _ctx_prefix(context)
    raw_responses: List[Dict[str, Any]] = d.get("responses", [])
    if len(raw_responses) < 2:
        raise error_cls(
            f"{_ctx}multi-response requires at least 2 responses; "
            f"got {len(raw_responses)}."
        )

    specs = [
        build_response_spec(r, error_cls=error_cls, context=context)
        for r in raw_responses
    ]

    power_combination = str(d.get("power_combination", "min"))

    sigma_joint_arr: Optional[np.ndarray] = None
    sj = d.get("sigma_joint")
    if sj is not None:
        try:
            sigma_joint_arr = np.array(sj, dtype=float)
        except (ValueError, TypeError) as e:
            raise error_cls(f"{_ctx}sigma_joint: invalid matrix: {e}") from e

    try:
        return MultiResponseOptions(
            responses=specs,
            power_combination=power_combination,
            sigma_joint=sigma_joint_arr,
        )
    except (ValueError, TypeError) as e:
        raise error_cls(f"{_ctx}invalid MultiResponseOptions: {e}") from e
