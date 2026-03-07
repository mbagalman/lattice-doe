# api.py
# License: MIT
"""
High-level API for Power-Assured I-Optimal Experimental Designs
===============================================================

This module exposes these public entry points:

    i_optimal_powered_design(formula, factors, power_cfg, design_opts, export_diagnostics_to=None)
    power_curve_by_n(...)
    power_curve_by_effect(...)
    generate_power_curves(...)
    power_sensitivity(...)
    min_detectable_effect(...)
    compare_criteria(...)

It orchestrates:
  1) Candidate set sizing (fixed or adaptive) and construction
  2) Model matrix building from Patsy formula
  3) I-optimal design selection (multi-start, optional parallel)
  4) Power evaluation:
        - global R² target OR
        - linear contrast target (via L, δ)
  5) Optional diagnostics export
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Literal, Tuple, Callable
import dataclasses
import time
import numpy as np
import pandas as pd
import warnings
from patsy import dmatrix  # ADDED: For early formula validation

from .config import PowerContrastConfig, PowerR2Config, DesignOptions
from .design import (
    estimate_candidate_size,
    build_candidate,
    build_model_matrix,
    build_i_opt_design_with_idx,
)
from .diag_metrics import compute_design_metrics
from .diag_export import export_diagnostics
from .power import contrast_power, global_r2_power, _r2_df_num
from .utils import validate_factors, initial_n_guess
from .power_curves import (
    power_curve_by_n as _power_curve_by_n_impl,
    power_curve_by_effect as _power_curve_by_effect_impl,
)


def _buckets_df(design_df: pd.DataFrame) -> pd.DataFrame:
    """Frequency of unique rows with a pandas-version-safe fallback."""
    if hasattr(pd.DataFrame, "value_counts"):
        return design_df.value_counts(dropna=False).rename("count").reset_index()
    # fallback for older pandas
    return (
        design_df.groupby(list(design_df.columns), dropna=False)
        .size()
        .reset_index(name="count")
    )


def _validate_api_inputs(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
) -> int:
    """
    Validate formula, factors, and configuration before expensive computations.
    Returns 'p' (number of model parameters) if successful.
    """
    # 1. Validate formula syntax and get 'p'
    try:
        # Build a tiny, sample DataFrame to test formula
        sample_data = {}
        for k, v in factors.items():
            if isinstance(v, (list, tuple)) and len(v) > 0:
                # If categorical, use first level. If continuous, use low bound.
                sample_data[k] = [v[0]]
            else:
                # Fallback for unexpected factor specs
                sample_data[k] = [1] 
                
        X_sample = dmatrix(formula, pd.DataFrame(sample_data), return_type="dataframe")
        p = X_sample.shape[1]
    except Exception as e:
        raise ValueError(
            f"Formula validation failed: '{formula}'. "
            f"Ensure all factors in formula are defined in 'factors' "
            f"and syntax is correct. Original patsy error: {e}"
        ) from e

    if p <= 0:
        raise ValueError(f"Model formula '{formula}' resulted in p=0 parameters.")

    # 2. Check n > p constraint
    if power_cfg.max_n <= p:
        raise ValueError(
            f"power_cfg.max_n ({power_cfg.max_n}) must be greater than "
            f"the number of model parameters p ({p})."
        )
        
    return p


def i_optimal_powered_design(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    design_opts: Optional[DesignOptions] = None,
    export_diagnostics_to: Optional[str] = None,
    export_report_to: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,  # ADDED
) -> Dict[str, Any]:
    """
    Build an I-optimal design and ensure (or approach) the requested power.

    Parameters
    ----------
    formula : str
        Patsy model formula (e.g., "~ 1 + A + B + A:B").
    factors : dict
        Factor specifications:
          • continuous: (low, high) tuples
          • categorical: list of levels
    power_cfg : PowerContrastConfig or PowerR2Config
        Target test configuration (contrast-based or global R²).
    design_opts : DesignOptions, optional
        Design generation and numerical options.
    export_diagnostics_to : str, optional
        If provided, exports diagnostics (plots/tables) to this path.
    export_report_to : str, optional
        If provided, writes a self-contained HTML report to this path (or
        directory).  Requires ``jinja2``; install with
        ``pip install "iopt-power-design[report]"``.  A failure here does
        **not** prevent the design result from being returned.
    progress_callback : callable, optional
        Function to call after each iteration, passing the current 'report' dict.
        Useful for logging or progress bars.

    Returns
    -------
    dict with keys:
      - design_df (DataFrame)     : chosen design (n x k)
      - buckets_df (DataFrame)    : frequency of unique rows
      - report (dict)             : iteration info, dfs, power, λ, diagnostics, etc.
    """
    if design_opts is None:
        design_opts = DesignOptions()

    # --- 1. Validate factors and inputs (early exit) ---
    validate_factors(factors)
    # ADDED: Early validation of formula, p, and max_n
    p = _validate_api_inputs(formula, factors, power_cfg)

    # --- 2. Decide candidate size ---
    if design_opts.auto_candidate:
        candidate_points = estimate_candidate_size(
            formula=formula,
            factors=factors,
            cand_min=design_opts.cand_min,
            cand_max=design_opts.cand_max,
            cat_cells_cap=design_opts.cat_cells_cap,
            per_cell_alpha=design_opts.per_cell_alpha,
            per_cell_min=design_opts.per_cell_min,
            per_cell_max=design_opts.per_cell_max,
            seed=design_opts.random_state,
        )
    else:
        candidate_points = int(design_opts.candidate_points)

    # --- 3. Build candidate set & cached model matrix ---
    cand = build_candidate(
        factors=factors,
        candidate_points=candidate_points,
        seed=design_opts.random_state,
        constraint_func=design_opts.constraint_func,
        cat_cells_cap=design_opts.cat_cells_cap,
    )
    X_cand, _ = build_model_matrix(formula, cand)

    # Re-check p just in case the full candidate set changed it (e.g., empty levels)
    if X_cand.shape[1] != p:
        warnings.warn(
            f"Model parameter count changed from p={p} (sample) to "
            f"p={X_cand.shape[1]} (full candidate set). Using new p. "
            "This may happen if some factor levels had 0 candidates."
        )
        p = X_cand.shape[1]
        if power_cfg.max_n <= p:
            raise ValueError(
                f"power_cfg.max_n ({power_cfg.max_n}) must be > p ({p}). "
                "p changed after building full candidate set."
            )

    # --- 4. Two-phase search for minimum n that meets the power target ---
    # Phase 1 — bisection: narrows the feasible range in O(log max_n) iterations.
    # Phase 2 — linear verification: because each n rebuilds an independent
    # I-optimal design via random starts, achieved power can be non-monotone.
    # After bisection finds a candidate n*, we scan linearly downward through a
    # small window to catch any smaller feasible n that bisection may have jumped.
    mode: Literal["contrast", "r2"] = "contrast" if isinstance(power_cfg, PowerContrastConfig) else "r2"
    target = power_cfg.power
    tol = power_cfg.tol_power

    # Validate mode-specific contrast dimensions up front (before the loop)
    if mode == "contrast":
        if power_cfg.L.shape[1] != p:
            raise ValueError(
                f"Contrast L has {power_cfg.L.shape[1]} columns but model has p={p} parameters."
            )
        q = power_cfg.L.shape[0]
        if power_cfg.delta.shape != (q,):
            raise ValueError(f"delta must be shape ({q},), got {power_cfg.delta.shape}.")

    lo = p + 1
    hi = power_cfg.max_n + 1  # exclusive sentinel — hi is only updated to evaluated achievers

    best: Optional[Dict[str, Any]] = None
    grew_candidates_once = False
    it = 0  # iteration counter

    # --- Run-metadata tracking (enriches the final report) ---
    t_start = time.perf_counter()
    _run_warnings: List[str] = []          # compact list of warning messages issued
    _verify_window: int = 0                # Phase 2 window size (0 if Phase 2 didn't run)
    _ran_phase2: bool = False              # True if at least one Phase 2 iteration executed
    _strategy_parts: List[str] = ["bisection"]  # build search_strategy string incrementally

    while lo < hi and it < power_cfg.max_iter:
        n = (lo + hi) // 2
        if n > power_cfg.max_n:
            lo = hi  # out of range; exit
            break
        it += 1

        # 1) Build I-optimal design at n
        design_df, selected_idx, _ = build_i_opt_design_with_idx(
            cand=cand,
            formula=formula,
            n=n,
            criterion=design_opts.criterion,
            n_start=design_opts.starts,
            algo=design_opts.algo,
            max_iter=design_opts.max_iter,
            random_state=design_opts.random_state,
            workers=design_opts.workers,
            parallel_seed_stride=design_opts.parallel_seed_stride,
        )
        X = X_cand[selected_idx, :]

        # 2) Compute power
        if mode == "contrast":
            power, lam = contrast_power(
                L=power_cfg.L, delta=power_cfg.delta, X=X,
                sigma=power_cfg.sigma, alpha=power_cfg.alpha,
                jitter=design_opts.xtx_jitter,
            )
            df_num = int(np.linalg.matrix_rank(power_cfg.L))
        else:
            power, lam = global_r2_power(
                r2_target=power_cfg.r2_target,
                X=X,
                alpha=power_cfg.alpha,
                lambda_mode=power_cfg.lambda_mode,
            )
            df_num = _r2_df_num(X)  # slopes only, matching global_r2_power convention

        # NaN → singular design at this n; treat as insufficient and search higher
        if np.isnan(power):
            _msg = f"Power is NaN at n={n} (singular design). Searching higher n."
            warnings.warn(_msg, RuntimeWarning)
            _run_warnings.append(_msg)
            lo = n + 1
            continue

        df_denom = int(X.shape[0] - np.linalg.matrix_rank(X))
        diags = compute_design_metrics(X, X_cand=X_cand)

        # Optional one-time candidate growth if conditioning is poor; re-evaluate at same n
        if (
            design_opts.allow_candidate_growth
            and not grew_candidates_once
            and diags.get("condition_number", np.inf) > 1e6
        ):
            grew_candidates_once = True
            _strategy_parts.append("growth")
            candidate_points = min(
                int(candidate_points * design_opts.growth_factor),
                design_opts.cand_max,
            )
            cand = build_candidate(
                factors=factors,
                candidate_points=candidate_points,
                seed=design_opts.random_state,
                constraint_func=design_opts.constraint_func,
                cat_cells_cap=design_opts.cat_cells_cap,
            )
            X_cand, _ = build_model_matrix(formula, cand)
            p = X_cand.shape[1]
            it += 1  # count the re-evaluation
            design_df, selected_idx, _ = build_i_opt_design_with_idx(
                cand=cand, formula=formula, n=n,
                criterion=design_opts.criterion,
                n_start=design_opts.starts,
                algo=design_opts.algo,
                max_iter=design_opts.max_iter,
                random_state=design_opts.random_state,
                workers=design_opts.workers,
                parallel_seed_stride=design_opts.parallel_seed_stride,
            )
            X = X_cand[selected_idx, :]
            if mode == "contrast":
                power, lam = contrast_power(
                    L=power_cfg.L, delta=power_cfg.delta, X=X,
                    sigma=power_cfg.sigma, alpha=power_cfg.alpha,
                    jitter=design_opts.xtx_jitter,
                )
                df_num = int(np.linalg.matrix_rank(power_cfg.L))
            else:
                power, lam = global_r2_power(
                    r2_target=power_cfg.r2_target, X=X,
                    alpha=power_cfg.alpha, lambda_mode=power_cfg.lambda_mode,
                )
                df_num = _r2_df_num(X)
            if np.isnan(power):
                _msg = f"Power is NaN at n={n} after candidate growth. Searching higher n."
                warnings.warn(_msg, RuntimeWarning)
                _run_warnings.append(_msg)
                lo = n + 1
                continue
            df_denom = int(X.shape[0] - np.linalg.matrix_rank(X))
            diags = compute_design_metrics(X, X_cand=X_cand)

        buckets = _buckets_df(design_df)
        report = {
            "iteration": it,
            "n": int(n),
            "p": int(p),
            "df_num": int(df_num),
            "df_denom": int(df_denom),
            "alpha": float(power_cfg.alpha),
            "target_power": float(target),
            "achieved_power": float(power),
            "noncentrality_lambda": float(lam),
            "diagnostics": diags,
            "criterion": design_opts.criterion,
            "algo": design_opts.algo,
            "starts": design_opts.starts,
            "workers": design_opts.workers,
            "candidate_points": int(candidate_points),
        }

        if progress_callback:
            try:
                progress_callback(report)
            except Exception as e:
                warnings.warn(f"Progress callback failed: {e}", RuntimeWarning)

        # Bisection step: if power achieved, record minimum-n achiever and search lower;
        # otherwise record best non-achiever as fallback and search higher.
        if power + tol >= target:
            if best is None or int(n) <= int(best["report"]["n"]):
                best = {
                    "design_df": design_df,
                    "buckets_df": buckets,
                    "report": report,
                    "_selected_idx": selected_idx,
                    "_X_cand": X_cand,
                }
            hi = n  # search for smaller achiever
        else:
            if best is None or (
                best["report"]["achieved_power"] + tol < target
                and power > best["report"]["achieved_power"]
            ):
                best = {
                    "design_df": design_df,
                    "buckets_df": buckets,
                    "report": report,
                    "_selected_idx": selected_idx,
                    "_X_cand": X_cand,
                }
            lo = n + 1  # need more runs

    # --- Phase 2: Linear verification scan ---
    # If bisection found an achiever at n*, scan n*-1, n*-2, … down to
    # max(p+1, n*-verify_window) to find any smaller n that also achieves target.
    if best is not None and best["report"]["achieved_power"] + tol >= target:
        n_star = int(best["report"]["n"])
        verify_window = min(max(5, n_star // 10), max(0, power_cfg.max_iter - it))
        _verify_window = verify_window  # record for final report
        for n_check in range(n_star - 1, max(p, n_star - verify_window - 1), -1):
            if it >= power_cfg.max_iter:
                break
            _ran_phase2 = True
            it += 1
            design_df_v, sel_idx_v, _ = build_i_opt_design_with_idx(
                cand=cand, formula=formula, n=n_check,
                criterion=design_opts.criterion, n_start=design_opts.starts,
                algo=design_opts.algo, max_iter=design_opts.max_iter,
                random_state=design_opts.random_state, workers=design_opts.workers,
                parallel_seed_stride=design_opts.parallel_seed_stride,
            )
            X_v = X_cand[sel_idx_v, :]
            if mode == "contrast":
                power_v, lam_v = contrast_power(
                    L=power_cfg.L, delta=power_cfg.delta, X=X_v,
                    sigma=power_cfg.sigma, alpha=power_cfg.alpha,
                    jitter=design_opts.xtx_jitter,
                )
                df_num_v = int(np.linalg.matrix_rank(power_cfg.L))
            else:
                power_v, lam_v = global_r2_power(
                    r2_target=power_cfg.r2_target, X=X_v,
                    alpha=power_cfg.alpha, lambda_mode=power_cfg.lambda_mode,
                )
                df_num_v = _r2_df_num(X_v)
            if np.isnan(power_v) or power_v + tol < target:
                continue  # not feasible; keep scanning
            df_denom_v = int(X_v.shape[0] - np.linalg.matrix_rank(X_v))
            diags_v = compute_design_metrics(X_v, X_cand=X_cand)
            report_v = {
                "iteration": it,
                "n": int(n_check),
                "p": int(p),
                "df_num": int(df_num_v),
                "df_denom": int(df_denom_v),
                "alpha": float(power_cfg.alpha),
                "target_power": float(target),
                "achieved_power": float(power_v),
                "noncentrality_lambda": float(lam_v),
                "diagnostics": diags_v,
                "criterion": design_opts.criterion,
                "algo": design_opts.algo,
                "starts": design_opts.starts,
                "workers": design_opts.workers,
                "candidate_points": int(candidate_points),
            }
            if progress_callback:
                try:
                    progress_callback(report_v)
                except Exception as e:
                    warnings.warn(f"Progress callback failed: {e}", RuntimeWarning)
            best = {
                "design_df": design_df_v,
                "buckets_df": _buckets_df(design_df_v),
                "report": report_v,
                "_selected_idx": sel_idx_v,
                "_X_cand": X_cand,
            }
            # Found a smaller achiever; keep scanning downward for an even smaller one

    # Record Phase 2 in strategy before computing final metadata
    if _ran_phase2:
        _strategy_parts.append("verification")

    # Compute total elapsed time (covers bisection + Phase 2; export excluded)
    elapsed_sec = time.perf_counter() - t_start

    # --- 5. Validate results and warn on non-convergence ---
    if best is None:
        raise RuntimeError(
            "Failed to generate any valid design. "
            "This can happen if power calculation failed repeatedly."
        )

    # MODIFIED: Get key metrics for validation and warning
    final_power = best["report"]["achieved_power"]
    target_power = best["report"]["target_power"]
    final_n = best["report"]["n"]
    final_p = best["report"]["p"]

    # MODIFIED: Warn with n and p if max_iter was hit without success
    if final_power + power_cfg.tol_power < target_power:
        _msg = (
            f"Design generation finished without converging to target power. "
            f"Max iterations ({power_cfg.max_iter}) or max_n ({power_cfg.max_n}) reached. "
            f"Final power: {final_power:.4f} (Target: {target_power:.4f}) "
            f"at n={final_n}, p={final_p}."
        )
        warnings.warn(_msg, RuntimeWarning)
        _run_warnings.append(_msg)

    # ADDED: Final result validation
    if len(best["design_df"]) != final_n:
        raise RuntimeError(
            f"Result validation failed: Final design_df has {len(best['design_df'])} rows, "
            f"but report indicates n={final_n}."
        )
    
    # MODIFIED: Reconstruct final X matrix for validation
    final_X = best["_X_cand"][best["_selected_idx"], :]
    
    # MODIFIED: Add X.shape check per review
    if final_X.shape != (final_n, final_p):
        raise RuntimeError(
            f"Result validation failed: Final design matrix X has shape {final_X.shape}, "
            f"but report indicates (n, p) = ({final_n}, {final_p})."
        )
    
    if np.isnan(best["report"]["achieved_power"]):
         raise RuntimeError(f"Result validation failed: Final reported power is NaN.")
    # --- End validation ---

    # --- Enrich final report with run-metadata ---
    best["report"].update({
        "elapsed_sec": round(float(elapsed_sec), 4),
        "search_strategy": "+".join(_strategy_parts),
        "verify_window": int(_verify_window),
        "random_state": int(design_opts.random_state) if design_opts.random_state is not None else None,
        "warnings": list(_run_warnings),
    })

    # 6. Optional export
    if export_diagnostics_to:
        try:
            # MODIFIED: final_X is already defined
            export_paths = export_diagnostics(
                X=final_X,
                design_df=best["design_df"],
                output_path=export_diagnostics_to,
                feature_names=None,
                formats=["html", "csv"],
                include_data=True,
            )
            best["report"]["diagnostic_exports"] = {
                k: str(v) for k, v in export_paths.items()
            }
        except Exception as e:
            # Don’t fail main computation due to export issues
            best["report"]["diagnostic_exports_error"] = str(e)

    # 7. Optional HTML report export
    if export_report_to is not None:
        try:
            from .report import generate_report
            report_path = generate_report(
                result=best,
                formula=formula,
                factors=factors,
                power_cfg=power_cfg,
                output_path=export_report_to,
                include_power_curve=False,  # skip curve to keep API call fast
            )
            best["report"]["report_path"] = str(report_path)
        except Exception as e:
            best["report"]["report_path_error"] = str(e)

    # Strip internal cache
    best.pop("_selected_idx", None)
    best.pop("_X_cand", None)
    return best


def power_curve_by_n(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
) -> pd.DataFrame:
    """Sweep n to visualize power as design size grows.

    This function is a compatibility wrapper around the canonical implementation
    in ``power_curves.py`` and returns only the curve DataFrame.
    """
    out = _power_curve_by_n_impl(
        formula=formula,
        factors=factors,
        power_cfg=power_cfg,
        design_opts=design_opts,
        plot=plot,
    )
    return out["data"]


def power_curve_by_effect(
    formula: str,
    factors: Dict[str, Any],
    n: int,
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
) -> pd.DataFrame:
    """Sweep effect size (δ for contrast, R² for global) at fixed n.

    This function is a compatibility wrapper around the canonical implementation
    in ``power_curves.py`` and returns only the curve DataFrame.
    """
    out = _power_curve_by_effect_impl(
        formula=formula,
        factors=factors,
        n=n,
        power_cfg=power_cfg,
        design_opts=design_opts,
        plot=plot,
    )
    df = out["data"].copy()
    if "effect_size" in df.columns:
        if isinstance(power_cfg, PowerContrastConfig):
            df = df.rename(columns={"effect_size": "effect_scale"})
        else:
            df = df.rename(columns={"effect_size": "r2_target"})
    return df


def generate_power_curves(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    curve_type: Literal["by_n", "by_effect", "both"] = "both",
    n_for_effect: Optional[int] = None,
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
) -> Dict[str, Any]:
    """Generate power curves for sensitivity analysis."""
    if design_opts is None:
        design_opts = DesignOptions()

    results: Dict[str, Any] = {}

    if curve_type in ("by_n", "both"):
        results["by_n"] = power_curve_by_n(
            formula, factors, power_cfg, design_opts=design_opts, plot=plot
        )

    if curve_type in ("by_effect", "both"):
        if n_for_effect is None:
            design_result = i_optimal_powered_design(
                formula, factors, power_cfg, design_opts=design_opts
            )
            n_for_effect = int(design_result["report"]["n"])

        results["by_effect"] = power_curve_by_effect(
            formula, factors, n_for_effect, power_cfg, design_opts=design_opts, plot=plot
        )

    return results


def power_sensitivity(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    design_df: pd.DataFrame,
    sigma_range: Tuple[float, float] = (0.5, 2.0),
    sigma_points: int = 25,
    r2_range: Tuple[float, float] = (0.05, 0.50),
    r2_points: int = 25,
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
    figsize: Tuple[float, float] = (8, 5),
) -> Dict[str, Any]:
    """Assess how achieved power changes when a key assumption varies.

    Re-evaluates power across a sweep of the key sensitivity parameter using
    the **fixed** design matrix from *design_df*.  No new I-optimal designs
    are built — this is a fast, purely analytical sweep.

    * For **contrast mode** (``PowerContrastConfig``): sweeps *sigma* from
      ``sigma_range[0]`` to ``sigma_range[1]``.  Useful for understanding
      how robust the design is if the noise assumption is wrong.
    * For **R² mode** (``PowerR2Config``): sweeps *r2_target* from
      ``r2_range[0]`` to ``r2_range[1]``.  Since sigma does not enter the
      R² power formula, varying sigma is meaningless; varying the assumed
      effect size (R²) is the natural sensitivity axis instead.

    Parameters
    ----------
    formula : str
        Patsy formula used when generating *design_df*.
    factors : dict
        Factor specifications matching the original design (needed only to
        rebuild the model matrix from *design_df*).
    power_cfg : PowerContrastConfig or PowerR2Config
        Power configuration.  The type determines which sensitivity axis is
        swept (sigma for contrast, r2_target for R²).
    design_df : DataFrame
        Fixed design to evaluate (e.g. ``result["design_df"]``).
    sigma_range : tuple of (sigma_lo, sigma_hi), default (0.5, 2.0)
        Absolute sigma values to sweep (contrast mode only).  The defaults
        span half to double the nominal sigma.
    sigma_points : int, default 25
        Number of evenly-spaced sigma values in the sweep (contrast mode).
    r2_range : tuple of (r2_lo, r2_hi), default (0.05, 0.50)
        R² values to sweep (R² mode only).  Both endpoints must be in (0, 1).
    r2_points : int, default 25
        Number of evenly-spaced R² values in the sweep (R² mode).
    design_opts : DesignOptions, optional
        Used only for ``xtx_jitter``.  Defaults to ``DesignOptions()``.
    plot : bool, default False
        If True and matplotlib is available, attach a Figure to the result.
    figsize : tuple, default (8, 5)
        Figure size when plotting.

    Returns
    -------
    dict — keys depend on *power_cfg* type:

    **Contrast mode** (``PowerContrastConfig``):
        ``data``           DataFrame — columns: sigma, power, noncentrality_lambda
        ``nominal_power``  float — power at ``power_cfg.sigma``
        ``sigma_nominal``  float — the nominal sigma from ``power_cfg``
        ``figure``         matplotlib Figure if *plot* is True, else None

    **R² mode** (``PowerR2Config``):
        ``data``           DataFrame — columns: r2_target, power, noncentrality_lambda
        ``nominal_power``  float — power at ``power_cfg.r2_target``
        ``r2_nominal``     float — the nominal r2_target from ``power_cfg``
        ``figure``         matplotlib Figure if *plot* is True, else None
    """
    if design_opts is None:
        design_opts = DesignOptions()

    # Rebuild X from the fixed design (no new DOE search needed)
    X, _ = build_model_matrix(formula, design_df)
    n = int(X.shape[0])
    jitter = design_opts.xtx_jitter

    # ------------------------------------------------------------------ #
    # R² mode: sweep r2_target                                            #
    # ------------------------------------------------------------------ #
    if isinstance(power_cfg, PowerR2Config):
        if r2_range[0] <= 0 or r2_range[1] <= 0:
            raise ValueError("r2_range values must be > 0")
        if r2_range[0] >= r2_range[1]:
            raise ValueError("r2_range[0] must be < r2_range[1]")
        if r2_range[0] >= 1 or r2_range[1] >= 1:
            raise ValueError("r2_range values must be < 1")
        if r2_points < 2:
            raise ValueError("r2_points must be >= 2")

        r2_vals = np.linspace(r2_range[0], r2_range[1], r2_points)
        rows = []
        for r2 in r2_vals:
            pwr, lam = global_r2_power(
                r2_target=float(r2),
                X=X,
                alpha=power_cfg.alpha,
                lambda_mode=power_cfg.lambda_mode,
            )
            rows.append({
                "r2_target": float(r2),
                "power": float(pwr),
                "noncentrality_lambda": float(lam),
            })

        df = pd.DataFrame(rows)

        # Nominal power at the configured r2_target
        nominal_pwr, _ = global_r2_power(
            r2_target=power_cfg.r2_target,
            X=X,
            alpha=power_cfg.alpha,
            lambda_mode=power_cfg.lambda_mode,
        )

        fig = None
        if plot:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(df["r2_target"], df["power"], "b-", linewidth=2, label="Power")
                ax.axvline(
                    x=power_cfg.r2_target, color="gray", linestyle="--",
                    label=f"Nominal R² = {power_cfg.r2_target}",
                )
                ax.axhline(
                    y=power_cfg.power, color="r", linestyle="--",
                    label=f"Target power = {power_cfg.power:.2f}",
                )
                ax.axhline(
                    y=float(nominal_pwr), color="steelblue", linestyle=":",
                    label=f"Power @ nominal R²: {float(nominal_pwr):.3f}",
                )
                ax.set_xlabel("R² (population effect size)")
                ax.set_ylabel("Statistical Power")
                ax.set_ylim([0, 1.05])
                ax.set_title(f"Power Sensitivity to R²  (n = {n})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
            except ImportError:
                pass  # matplotlib unavailable — return fig=None

        return {
            "data": df,
            "nominal_power": float(nominal_pwr),
            "r2_nominal": float(power_cfg.r2_target),
            "figure": fig,
        }

    # ------------------------------------------------------------------ #
    # Contrast mode: sweep sigma                                          #
    # ------------------------------------------------------------------ #
    if sigma_range[0] <= 0 or sigma_range[1] <= 0:
        raise ValueError("sigma_range values must be > 0")
    if sigma_range[0] >= sigma_range[1]:
        raise ValueError("sigma_range[0] must be < sigma_range[1]")
    if sigma_points < 2:
        raise ValueError("sigma_points must be >= 2")

    sigma_vals = np.linspace(sigma_range[0], sigma_range[1], sigma_points)
    rows = []
    for sigma in sigma_vals:
        pwr, lam = contrast_power(
            L=power_cfg.L,
            delta=power_cfg.delta,
            X=X,
            sigma=float(sigma),
            alpha=power_cfg.alpha,
            jitter=jitter,
        )
        rows.append({
            "sigma": float(sigma),
            "power": float(pwr),
            "noncentrality_lambda": float(lam),
        })

    df = pd.DataFrame(rows)

    # Nominal power at the configured sigma (reference line)
    nominal_pwr, _ = contrast_power(
        L=power_cfg.L,
        delta=power_cfg.delta,
        X=X,
        sigma=power_cfg.sigma,
        alpha=power_cfg.alpha,
        jitter=jitter,
    )

    fig = None
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(df["sigma"], df["power"], "b-", linewidth=2, label="Power")
            ax.axvline(
                x=power_cfg.sigma, color="gray", linestyle="--",
                label=f"Nominal σ = {power_cfg.sigma}",
            )
            ax.axhline(
                y=power_cfg.power, color="r", linestyle="--",
                label=f"Target power = {power_cfg.power:.2f}",
            )
            ax.axhline(
                y=float(nominal_pwr), color="steelblue", linestyle=":",
                label=f"Power @ nominal σ: {float(nominal_pwr):.3f}",
            )
            ax.set_xlabel("σ  (residual standard deviation)")
            ax.set_ylabel("Statistical Power")
            ax.set_ylim([0, 1.05])
            ax.set_title(f"Power Sensitivity to σ  (n = {n})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
        except ImportError:
            pass  # matplotlib unavailable — return fig=None

    return {
        "data": df,
        "nominal_power": float(nominal_pwr),
        "sigma_nominal": float(power_cfg.sigma),
        "figure": fig,
    }


def min_detectable_effect(
    design_df: pd.DataFrame,
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    target_power: float = 0.80,
    design_opts: Optional[DesignOptions] = None,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> Dict[str, Any]:
    """Find the minimum detectable effect (MDE) for a fixed design.

    Inverts the power curve analytically: given a fixed design matrix X and
    a power target, returns the smallest effect size that achieves that power.
    No new I-optimal designs are built.

    For **contrast mode** (``PowerContrastConfig``), the MDE is expressed as
    a *scale factor* on ``power_cfg.delta``.  A scale of 1.0 means the
    original ``sesoi`` (delta) is exactly detectable at *target_power*; a
    scale > 1.0 means only *larger* effects are detectable.

    For **R² mode** (``PowerR2Config``), the MDE is the minimum
    ``r2_target`` value at which *target_power* is achieved; values near 0
    indicate the design can detect small R² effects.

    Parameters
    ----------
    design_df : DataFrame
        Fixed design (e.g. ``result["design_df"]``).
    formula : str
        Patsy formula used to build *design_df*.
    factors : dict
        Factor specifications (needed to rebuild the model matrix).
    power_cfg : PowerContrastConfig or PowerR2Config
        Power configuration supplying contrast details (L, delta, sigma) or
        R² parameters (alpha, lambda_mode).
    target_power : float, default 0.80
        Power level to invert; must be in (0, 1).
    design_opts : DesignOptions, optional
        Used only for ``xtx_jitter``.  Defaults to ``DesignOptions()``.
    tol : float, default 1e-4
        Bisection convergence tolerance on the effect parameter.
    max_iter : int, default 60
        Maximum bisection iterations.

    Returns
    -------
    dict with keys:
        ``mde``            Minimum detectable effect.
                           *Contrast mode*: scale factor on delta (float).
                           *R² mode*: minimum r2_target (float).
        ``achieved_power`` Power at the MDE (should be ≈ target_power).
        ``n``              Number of runs in the fixed design.
        ``mode``           ``"contrast"`` or ``"r2"``.
    """
    if not (0 < target_power < 1):
        raise ValueError(f"target_power must be in (0, 1); got {target_power!r}.")
    if design_opts is None:
        design_opts = DesignOptions()

    X, _ = build_model_matrix(formula, design_df)
    n = int(X.shape[0])
    jitter = design_opts.xtx_jitter

    if isinstance(power_cfg, PowerContrastConfig):
        # Bisect over a multiplicative scale on delta.
        # scale→0 means effect→0 (power→alpha); scale→large means power→1.
        def _pwr(scale: float) -> float:
            pwr, _ = contrast_power(
                L=power_cfg.L,
                delta=power_cfg.delta * scale,
                X=X,
                sigma=power_cfg.sigma,
                alpha=power_cfg.alpha,
                jitter=jitter,
            )
            return float(pwr)

        lo_s, hi_s = 0.0, 20.0
        # Expand hi_s until power at hi_s ≥ target_power
        for _ in range(20):
            if _pwr(hi_s) >= target_power:
                break
            hi_s *= 2.0
        else:
            return {
                "mde": float("inf"),
                "achieved_power": _pwr(hi_s),
                "n": n,
                "mode": "contrast",
            }

        for _ in range(max_iter):
            mid = (lo_s + hi_s) / 2.0
            if _pwr(mid) >= target_power:
                hi_s = mid
            else:
                lo_s = mid
            if hi_s - lo_s < tol:
                break

        mde = (lo_s + hi_s) / 2.0
        return {
            "mde": float(mde),
            "achieved_power": float(_pwr(mde)),
            "n": n,
            "mode": "contrast",
        }

    else:  # PowerR2Config
        # Bisect over r2_target in (0, 1).
        # Higher R² → higher power (monotone).
        def _pwr_r2(r2: float) -> float:
            pwr, _ = global_r2_power(
                r2_target=r2,
                X=X,
                alpha=power_cfg.alpha,
                lambda_mode=power_cfg.lambda_mode,
            )
            return float(pwr)

        lo_r2, hi_r2 = 1e-6, 1.0 - 1e-6

        if _pwr_r2(hi_r2) < target_power:
            return {
                "mde": float("inf"),
                "achieved_power": _pwr_r2(hi_r2),
                "n": n,
                "mode": "r2",
            }
        if _pwr_r2(lo_r2) >= target_power:
            return {
                "mde": float(lo_r2),
                "achieved_power": _pwr_r2(lo_r2),
                "n": n,
                "mode": "r2",
            }

        for _ in range(max_iter):
            mid = (lo_r2 + hi_r2) / 2.0
            if _pwr_r2(mid) >= target_power:
                hi_r2 = mid
            else:
                lo_r2 = mid
            if hi_r2 - lo_r2 < tol:
                break

        mde = (lo_r2 + hi_r2) / 2.0
        return {
            "mde": float(mde),
            "achieved_power": float(_pwr_r2(mde)),
            "n": n,
            "mode": "r2",
        }


def compare_criteria(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    design_opts: Optional[DesignOptions] = None,
    criteria: Optional[List[str]] = None,
    plot: bool = False,
    figsize: Tuple[float, float] = (8, 5),
) -> Dict[str, Any]:
    """Run the powered-design search under multiple optimality criteria and compare.

    Executes ``i_optimal_powered_design`` independently for each entry in
    *criteria* (default: all three — ``"I"``, ``"D"``, ``"A"``), then assembles
    a side-by-side summary to support criterion choice.  All runs share the same
    formula, factors, and power configuration; only the ``criterion`` field of
    *design_opts* is swapped per run.

    Parameters
    ----------
    formula : str
        Patsy formula string (e.g. ``"~ 1 + A + B + A:B"``).
    factors : dict
        Factor specifications (same format as ``i_optimal_powered_design``).
    power_cfg : PowerContrastConfig or PowerR2Config
        Shared power configuration applied to all criterion runs.
    design_opts : DesignOptions, optional
        Base design options.  The ``criterion`` field is overridden per run.
        Defaults to ``DesignOptions()``.  The original object is never mutated.
    criteria : list of str, optional
        Criteria to compare.  Any non-empty subset of ``["I", "D", "A"]``.
        Defaults to all three: ``["I", "D", "A"]``.
    plot : bool, default False
        If True and matplotlib is available, produce a grouped bar chart
        comparing achieved power (and sample size) across criteria.
    figsize : tuple of float, default (8, 5)
        Figure dimensions ``(width, height)`` in inches when *plot* is True.

    Returns
    -------
    dict with keys:

    ``summary`` : DataFrame
        One row per criterion.  Columns:

        ==================  =================================================
        ``criterion``       "I", "D", or "A"
        ``n``               Minimum sample size that achieved target power
        ``achieved_power``  Statistical power of the returned design
        ``elapsed_sec``     Wall-clock seconds for that criterion's run
        ``condition_number``Condition number of X'X (from diagnostics)
        ``d_efficiency``    D-efficiency relative to D-optimal reference
        ==================  =================================================

    ``results`` : dict
        Maps each criterion string to the full dict returned by
        ``i_optimal_powered_design`` (``design_df``, ``buckets_df``, ``report``).

    ``figure`` : matplotlib Figure or None
        Bar chart if *plot* is True and matplotlib is importable; else None.

    Raises
    ------
    ValueError
        If *criteria* is empty or contains an unrecognised criterion string.

    Examples
    --------
    >>> comparison = compare_criteria(formula, factors, power_cfg, design_opts=opts)
    >>> print(comparison["summary"])
    #   criterion   n   achieved_power  elapsed_sec  condition_number  d_efficiency
    #   I          24        0.814          1.23          12.5             0.81
    #   D          22        0.823          1.17          10.2             1.00
    #   A          23        0.811          1.19          11.8             0.94
    >>> comparison["results"]["I"]["design_df"]   # full I-optimal design DataFrame
    """
    if design_opts is None:
        design_opts = DesignOptions()
    if criteria is None:
        criteria = ["I", "D", "A"]

    valid_criteria = {"I", "D", "A"}
    if len(criteria) == 0:
        raise ValueError("criteria must contain at least one entry.")
    bad = set(criteria) - valid_criteria
    if bad:
        raise ValueError(
            f"Invalid criteria {sorted(bad)!r}. "
            f"Each must be one of {sorted(valid_criteria)}."
        )

    all_results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for criterion in criteria:
        # Swap criterion without mutating the caller's DesignOptions
        run_opts = dataclasses.replace(design_opts, criterion=criterion)
        res = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=run_opts,
        )
        all_results[criterion] = res
        rpt = res["report"]
        diag = rpt.get("diagnostics") or {}
        rows.append({
            "criterion": criterion,
            "n": int(rpt["n"]),
            "achieved_power": float(rpt["achieved_power"]),
            "elapsed_sec": float(rpt.get("elapsed_sec", float("nan"))),
            "condition_number": float(diag.get("condition_number", float("nan"))),
            "d_efficiency": float(diag.get("d_efficiency", float("nan"))),
        })

    summary = pd.DataFrame(rows)

    fig = None
    if plot:
        try:
            import matplotlib.pyplot as plt

            _palette = ["steelblue", "darkorange", "mediumseagreen"]
            _colors = [_palette[i % len(_palette)] for i in range(len(criteria))]

            fig, axes = plt.subplots(1, 2, figsize=figsize)

            # --- Left: achieved power ---
            axes[0].bar(summary["criterion"], summary["achieved_power"], color=_colors)
            axes[0].axhline(
                y=float(power_cfg.power),
                color="red",
                linestyle="--",
                label=f"Target = {power_cfg.power:.2f}",
            )
            axes[0].set_ylim([0, 1.05])
            axes[0].set_xlabel("Criterion")
            axes[0].set_ylabel("Achieved power")
            axes[0].set_title("Achieved Power")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3, axis="y")

            # --- Right: sample size n ---
            axes[1].bar(summary["criterion"], summary["n"], color=_colors)
            axes[1].set_xlabel("Criterion")
            axes[1].set_ylabel("n (runs)")
            axes[1].set_title("Sample Size Required")
            axes[1].grid(True, alpha=0.3, axis="y")

            plt.suptitle("Criterion Comparison", fontsize=12, fontweight="bold")
            plt.tight_layout()
        except ImportError:
            pass  # matplotlib unavailable — return fig=None

    return {
        "summary": summary,
        "results": all_results,
        "figure": fig,
    }


__all__ = [
    "i_optimal_powered_design",
    "power_curve_by_n",
    "power_curve_by_effect",
    "generate_power_curves",
    "power_sensitivity",
    "min_detectable_effect",
    "compare_criteria",
]
