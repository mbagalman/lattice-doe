# api_server/serialization.py
# License: MIT
"""
Serialization helpers: numpy/pandas ↔ JSON-safe Python types.

These converters live here (not in the models) so they can be used by
every router without circular imports.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from iopt_power_design.config import DesignOptions, MultiResponseOptions, PowerContrastConfig, PowerGLMContrastConfig, PowerR2Config, ResponseSpec, SplitPlotOptions


# ---------------------------------------------------------------------------
# Scalar / float sanitization
# ---------------------------------------------------------------------------

def sanitize_float(v: Any) -> Optional[float]:
    """Convert numpy floats and handle nan/inf → None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def sanitize_value(v: Any) -> Any:
    """Recursively convert numpy scalars to JSON-safe Python types."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return sanitize_float(float(v))
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: sanitize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [sanitize_value(i) for i in v]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ---------------------------------------------------------------------------
# DataFrame serialization
# ---------------------------------------------------------------------------

def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a DataFrame to a list of JSON-safe row dicts."""
    records = []
    for row in df.to_dict(orient="records"):
        records.append({k: sanitize_value(v) for k, v in row.items()})
    return records


def records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Reconstruct a DataFrame from list-of-dicts (rows-as-records)."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Config conversion: Pydantic model → dataclass
# ---------------------------------------------------------------------------

def pydantic_power_cfg_to_dataclass(
    model: Any,
) -> Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig]:
    """Convert a Pydantic power-config model to the matching dataclass."""
    if model.type == "contrast":
        return PowerContrastConfig(
            L=np.array(model.L, dtype=float),
            delta=np.array(model.delta, dtype=float),
            alpha=model.alpha,
            power=model.power,
            sigma=model.sigma,
            tol_power=model.tol_power,
            max_iter=model.max_iter,
            max_n=model.max_n,
        )
    if model.type == "glm_contrast":
        return PowerGLMContrastConfig(
            L=np.array(model.L, dtype=float),
            delta=np.array(model.delta, dtype=float),
            baseline=model.baseline,
            family=model.family,
            link=model.link,
            alpha=model.alpha,
            power=model.power,
            tol_power=model.tol_power,
            max_iter=model.max_iter,
            max_n=model.max_n,
        )
    return PowerR2Config(
        r2_target=model.r2_target,
        alpha=model.alpha,
        power=model.power,
        tol_power=model.tol_power,
        max_iter=model.max_iter,
        max_n=model.max_n,
        lambda_mode=model.lambda_mode,
    )


def pydantic_design_opts_to_dataclass(model: Optional[Any]) -> DesignOptions:
    """Convert a Pydantic DesignOptionsModel to DesignOptions.

    ``constraint_func`` is never accepted over HTTP — only ``constraint_expr``
    (a string) is accepted; ``DesignOptions.__post_init__`` compiles it.

    ``workers`` is forced to ``None`` (serial execution) inside the ASGI
    server; use Uvicorn's own ``--workers`` flag for concurrency instead.
    """
    if model is None:
        return DesignOptions()

    kwargs: dict = dict(
        candidate_points=model.candidate_points,
        auto_candidate=model.auto_candidate,
        cand_min=model.cand_min,
        cand_max=model.cand_max,
        random_state=model.random_state,
        criterion=model.criterion,
        algo=model.algo,
        starts=model.starts,
        max_iter=model.max_iter,
        xtx_jitter=model.xtx_jitter,
        workers=None,  # always serial inside ASGI — see docstring
        block_factor_name=model.block_factor_name,
        preallocate_categorical=model.preallocate_categorical,
        alloc_min_per_cell=model.alloc_min_per_cell,
    )
    if model.n_blocks is not None:
        kwargs["n_blocks"] = model.n_blocks
    if model.block_sizes is not None:
        kwargs["block_sizes"] = model.block_sizes
    if model.alloc_max_per_cell is not None:
        kwargs["alloc_max_per_cell"] = model.alloc_max_per_cell
    if model.constraint_expr:
        kwargs["constraint_expr"] = model.constraint_expr
    if model.split_plot is not None:
        sp = model.split_plot
        kwargs["split_plot"] = SplitPlotOptions(
            htc_factors=list(sp.htc_factors),
            n_whole_plots=sp.n_whole_plots,
            eta=sp.eta,
            subplots_per_wp=sp.subplots_per_wp,
            df_method=sp.df_method,
        )
    return DesignOptions(**kwargs)


# ---------------------------------------------------------------------------
# Result serialization: find_optimal_design return dict → response dict
# ---------------------------------------------------------------------------

def serialize_report(report: dict) -> dict:
    """Strip internal keys and sanitize numpy scalars from a report dict."""
    skip = {"_X", "_selected_idx", "_X_cand", "figure"}
    out: dict = {}
    for k, v in report.items():
        if k in skip:
            continue
        out[k] = sanitize_value(v)
    return out


def serialize_design_result(result: dict) -> dict:
    """Convert an find_optimal_design result dict to JSON-safe form."""
    return {
        "design_df": df_to_records(result["design_df"]),
        "buckets_df": df_to_records(result["buckets_df"]),
        "report": serialize_report(result["report"]),
    }


def pydantic_multi_cfg_to_dataclass(model: Any) -> MultiResponseOptions:
    """Convert a Pydantic MultiResponseOptionsModel to MultiResponseOptions."""
    responses = [
        ResponseSpec(
            name=r.name,
            power_cfg=pydantic_power_cfg_to_dataclass(r.power_cfg),
            formula=r.formula,
            weight=r.weight,
        )
        for r in model.responses
    ]
    kwargs: dict = dict(
        responses=responses,
        power_combination=model.power_combination,
    )
    if model.sigma_joint is not None:
        kwargs["sigma_joint"] = np.array(model.sigma_joint, dtype=float)
    return MultiResponseOptions(**kwargs)


def serialize_multiresponse_result(result: dict) -> dict:
    """Convert an find_multiresponse_design result dict to JSON-safe form."""
    design_df = result.get("design")
    buckets_df = result.get("buckets")
    out: dict = {
        "design": df_to_records(design_df) if isinstance(design_df, pd.DataFrame) else [],
        "n": int(result["n"]),
        "achieved_power": sanitize_float(result["achieved_power"]),
        "responses": [sanitize_value(r) for r in result.get("responses", [])],
        "combination_rule": str(result.get("combination_rule", "min")),
        "compound_criterion": bool(result.get("compound_criterion", False)),
        "elapsed_sec": sanitize_float(result.get("elapsed_sec")),
        "buckets": df_to_records(buckets_df) if isinstance(buckets_df, pd.DataFrame) else [],
        "warnings": list(result.get("warnings", [])),
        # Search diagnostics
        "search_strategy": result.get("search_strategy"),
        "p": int(result["p"]) if result.get("p") is not None else None,
        "iteration": int(result["iteration"]) if result.get("iteration") is not None else None,
        # Hotelling T² (optional — present when sigma_joint was supplied)
        "joint_power": sanitize_float(result.get("joint_power")),
        "joint_lam": sanitize_float(result.get("joint_lam")),
        "joint_df1": int(result["joint_df1"]) if result.get("joint_df1") is not None else None,
        "joint_df2": int(result["joint_df2"]) if result.get("joint_df2") is not None else None,
        # Split-plot summary (optional — present when split-plot mode was used)
        "n_whole_plots": int(result["n_whole_plots"]) if result.get("n_whole_plots") is not None else None,
        "subplots_per_wp": int(result["subplots_per_wp"]) if result.get("subplots_per_wp") is not None else None,
    }
    return out
