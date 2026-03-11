# api.py
# License: MIT
"""
I-Optimal Powered Design — orchestration entry point
=====================================================

This module exposes a single public function:

    i_optimal_powered_design(formula, factors, power_cfg, design_opts, ...)

It orchestrates:
  1) Candidate set sizing (fixed or adaptive) and construction
  2) Model matrix building from Patsy formula
  3) I-optimal design selection (multi-start, optional parallel)
  4) Power evaluation:
        - global R² target OR
        - linear contrast target (via L, δ)
  5) Optional diagnostics export

All analysis utilities (power_curve_by_n, power_curve_by_effect,
generate_power_curves, power_sensitivity, min_detectable_effect,
compare_criteria) live in ``analysis.py`` and are re-exported via
``iopt_power_design.__init__``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Literal, Tuple, Callable
import math
import time
import numpy as np
import pandas as pd
import warnings

from patsy import dmatrix
from .config import PowerContrastConfig, PowerR2Config, DesignOptions
from .candidate import estimate_candidate_size, build_candidate, build_split_plot_candidate
from .model_matrix import build_model_matrix
from .iopt_search import build_i_opt_design_with_idx, build_split_plot_design
from .split_plot import build_whole_plot_indicator, htc_factor_cols_from_names
from .diag_metrics import compute_design_metrics
from .diag_export import export_diagnostics
from .power import (
    contrast_power, global_r2_power, _r2_df_num,
    contrast_power_sp, global_r2_power_sp,
)
from .utils import validate_factors, initial_n_guess
from .blocked import blocked_formula, build_blocked_design


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


def _validate_htc_factors(htc_factors: List[str], factors: Dict[str, Any]) -> None:
    """Raise ValueError if any HTC factor name is not a key in factors."""
    bad = [f for f in htc_factors if f not in factors]
    if bad:
        raise ValueError(
            f"htc_factors {bad} not found in factors dict. "
            f"Valid factor names: {list(factors)}."
        )


def _auto_subplots_per_wp(p: int, n_wp: int) -> int:
    """Default subplots per whole plot: max(2, ceil(p / n_wp) + 1)."""
    return max(2, math.ceil(p / n_wp) + 1)


def _is_split_plot(design_opts: Optional[DesignOptions]) -> bool:
    """True if split-plot options are configured."""
    return design_opts is not None and design_opts.split_plot is not None


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

    # --- Blocking setup ---
    is_blocked = design_opts.n_blocks is not None and design_opts.n_blocks >= 2
    is_sp = _is_split_plot(design_opts)
    if is_sp and is_blocked:
        raise ValueError(
            "n_blocks and split_plot cannot both be set. "
            "Blocked split-plot (three-stratum) designs are not yet supported."
        )
    if is_blocked:
        # CR-17: Reject block_factor_name that collides with a treatment factor.
        if design_opts.block_factor_name in factors:
            raise ValueError(
                f"block_factor_name={design_opts.block_factor_name!r} collides with "
                f"an existing treatment factor. Choose a name not in: {list(factors)}."
            )
        aug_formula = blocked_formula(formula, design_opts.block_factor_name)
    else:
        aug_formula = formula
    p_treat = p  # treatment-only parameter count (for L validation)

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
        p_treat = p  # keep p_treat in sync after candidate set update
        if power_cfg.max_n <= p:
            raise ValueError(
                f"power_cfg.max_n ({power_cfg.max_n}) must be > p ({p}). "
                "p changed after building full candidate set."
            )

    # For blocked designs: compute p_full from augmented formula.
    # CR-18: Use the full candidate set (all categorical treatment levels present)
    # augmented with cyclic block labels so every block level appears at least once.
    # The previous approach pinned each categorical treatment factor to a single level,
    # causing Patsy to omit their dummy columns and undercount p_full relative to
    # p_treat, triggering a false "no block dummy columns" error.
    if is_blocked:
        _blk_labels = [f"B{i + 1}" for i in range(design_opts.n_blocks)]
        _cand_aug_tmp = cand.copy()
        _n_cand_tmp = len(_cand_aug_tmp)
        # Ensure every block label appears at least once (pad if candidate is tiny)
        if _n_cand_tmp < design_opts.n_blocks:
            _pad = pd.concat(
                [cand.iloc[[0]]] * (design_opts.n_blocks - _n_cand_tmp),
                ignore_index=True,
            )
            _cand_aug_tmp = pd.concat([_cand_aug_tmp, _pad], ignore_index=True)
        _cand_aug_tmp[design_opts.block_factor_name] = [
            _blk_labels[i % design_opts.n_blocks]
            for i in range(len(_cand_aug_tmp))
        ]
        try:
            _X_aug_sample, _ = build_model_matrix(aug_formula, _cand_aug_tmp)
            p_full = _X_aug_sample.shape[1]
        except Exception as _e:
            raise ValueError(
                f"Failed to build augmented blocked formula {aug_formula!r}: {_e}"
            ) from _e
        p_block_cols = p_full - p_treat
        if p_block_cols < 1:
            raise ValueError(
                f"Blocked model has no block dummy columns "
                f"(p_full={p_full}, p_treat={p_treat}). "
                "Check that block_factor_name does not conflict with existing factors."
            )
    else:
        p_full = p_treat
        p_block_cols = 0

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
        if power_cfg.L.shape[1] != p_treat:
            raise ValueError(
                f"Contrast L has {power_cfg.L.shape[1]} columns but model has "
                f"p_treat={p_treat} treatment parameters."
            )
        q = power_cfg.L.shape[0]
        if power_cfg.delta.shape != (q,):
            raise ValueError(f"delta must be shape ({q},), got {power_cfg.delta.shape}.")
        # Pad L with zeros for block dummy columns when blocked
        if is_blocked and p_block_cols > 0:
            L_eff = np.hstack([power_cfg.L, np.zeros((q, p_block_cols))])
        else:
            L_eff = power_cfg.L
    else:
        L_eff = None

    # =========================================================================
    # Split-plot path — bisection over n_whole_plots, then Phase 2 scan
    # =========================================================================
    if is_sp:
        sp_opts = design_opts.split_plot
        _validate_htc_factors(sp_opts.htc_factors, factors)
        subplots_per_wp = (
            sp_opts.subplots_per_wp
            if sp_opts.subplots_per_wp is not None
            else _auto_subplots_per_wp(p, sp_opts.n_whole_plots)
        )
        etc_factors = [f for f in factors if f not in sp_opts.htc_factors]
        sigma_sp = power_cfg.sigma if mode == "contrast" else 1.0

        # Split candidate budget proportionally across WP and SP strata.
        _n_htc = len(sp_opts.htc_factors)
        _n_etc = len(etc_factors)
        _n_all = max(1, _n_htc + _n_etc)
        _n_wp_cand = max(10, int(candidate_points * _n_htc / _n_all))
        _n_sp_cand = max(10, int(candidate_points * _n_etc / _n_all)) if _n_etc > 0 else 1

        max_n_wp = power_cfg.max_n // subplots_per_wp
        lo_wp = sp_opts.n_whole_plots   # user's minimum
        hi_wp = max_n_wp + 1           # exclusive sentinel

        best: Optional[Dict[str, Any]] = None
        it = 0
        _run_warnings: List[str] = []
        _strategy_parts: List[str] = ["bisection"]
        _ran_phase2 = False
        _verify_window = 0
        t_start = time.perf_counter()

        def _sp_eval(n_wp_: int) -> Optional[Dict[str, Any]]:
            """Build SP design at n_wp_, compute power; return result dict or None."""
            n_total_ = n_wp_ * subplots_per_wp
            sp_cand_ = build_split_plot_candidate(
                factors, sp_opts.htc_factors, n_wp_, subplots_per_wp,
                random_state=design_opts.random_state,
                candidate_points=candidate_points,
                constraint_func=design_opts.constraint_func,
            )
            design_df_, X_ = build_split_plot_design(
                sp_cand_, formula, n_wp_, subplots_per_wp,
                sp_opts.htc_factors, sp_opts.eta,
                factors=factors,
                criterion=design_opts.criterion,
                starts=design_opts.starts,
                max_iter=design_opts.max_iter,
                random_state=design_opts.random_state,
                jitter=design_opts.xtx_jitter,
                constraint_func=design_opts.constraint_func,
                n_wp_cand=_n_wp_cand,
                n_sp_cand=_n_sp_cand,
            )
            Z_ = build_whole_plot_indicator(n_total_, n_wp_, subplots_per_wp)
            if mode == "contrast":
                _, _p_names = build_model_matrix(formula, design_df_)
                _all_fcols = [c for c in design_df_.columns if c != "__wp_id__"]
                _htc_cols = htc_factor_cols_from_names(
                    _p_names, sp_opts.htc_factors, _all_fcols,
                )
                pr_ = contrast_power_sp(
                    L_eff, power_cfg.delta, X_, Z_,
                    sigma_sp=sigma_sp, eta=sp_opts.eta, alpha=power_cfg.alpha,
                    df_method=sp_opts.df_method, jitter=design_opts.xtx_jitter,
                    htc_factor_cols=_htc_cols,
                )
                df_num_ = int(np.linalg.matrix_rank(power_cfg.L))
            else:
                pr_ = global_r2_power_sp(
                    power_cfg.r2_target, X_, Z_, sigma_sp=sigma_sp,
                    eta=sp_opts.eta, alpha=power_cfg.alpha,
                    df_method=sp_opts.df_method,
                    lambda_mode=power_cfg.lambda_mode,
                    jitter=design_opts.xtx_jitter,
                )
                df_num_ = _r2_df_num(X_)
            if np.isnan(pr_.power):
                return None
            return {
                "design_df": design_df_,
                "buckets_df": _buckets_df(design_df_),
                "_X": X_,
                "_n_wp": n_wp_,
                "report": {
                    "n": int(n_total_),
                    "p": int(p),
                    "df_num": int(df_num_),
                    "df_denom": int(n_total_ - n_wp_),
                    "alpha": float(power_cfg.alpha),
                    "target_power": float(target),
                    "achieved_power": float(pr_.power),
                    "noncentrality_lambda": float(pr_.lam),
                    "diagnostics": compute_design_metrics(X_),
                    "criterion": design_opts.criterion,
                    "algo": design_opts.algo,
                    "starts": design_opts.starts,
                    "workers": design_opts.workers,
                    "candidate_points": int(candidate_points),
                    "block_structure": None,
                    "p_treat": int(p_treat),
                    "split_plot": {
                        "n_whole_plots": int(n_wp_),
                        "subplots_per_wp": int(subplots_per_wp),
                        "n_total": int(n_total_),
                        "eta": float(sp_opts.eta),
                        "htc_factors": list(sp_opts.htc_factors),
                        "etc_factors": list(etc_factors),
                        "df_method": str(sp_opts.df_method),
                    },
                },
            }

        # Phase 1 — bisection over n_whole_plots
        while lo_wp < hi_wp and it < power_cfg.max_iter:
            n_wp = (lo_wp + hi_wp) // 2
            it += 1
            ev = _sp_eval(n_wp)
            if ev is None:
                _run_warnings.append(f"Power is NaN at n_wp={n_wp}. Searching higher.")
                lo_wp = n_wp + 1
                continue
            ev["report"]["iteration"] = it
            if progress_callback:
                try:
                    progress_callback(ev["report"])
                except Exception as e:
                    warnings.warn(f"Progress callback failed: {e}", RuntimeWarning)
            if ev["report"]["achieved_power"] + tol >= target:
                if best is None or n_wp <= best["_n_wp"]:
                    best = ev
                hi_wp = n_wp
            else:
                if best is None or (
                    best["report"]["achieved_power"] + tol < target
                    and ev["report"]["achieved_power"] > best["report"]["achieved_power"]
                ):
                    best = ev
                lo_wp = n_wp + 1

        # Phase 2 — linear scan downward
        if best is not None and best["report"]["achieved_power"] + tol >= target:
            n_wp_star = best["_n_wp"]
            verify_window = min(max(3, n_wp_star // 5), max(0, power_cfg.max_iter - it))
            _verify_window = verify_window
            for n_wp_chk in range(
                n_wp_star - 1,
                max(max(2, sp_opts.n_whole_plots), n_wp_star - verify_window - 1),
                -1,
            ):
                if it >= power_cfg.max_iter:
                    break
                _ran_phase2 = True
                it += 1
                ev = _sp_eval(n_wp_chk)
                if ev is None or ev["report"]["achieved_power"] + tol < target:
                    continue
                ev["report"]["iteration"] = it
                if progress_callback:
                    try:
                        progress_callback(ev["report"])
                    except Exception as e:
                        warnings.warn(f"Progress callback failed: {e}", RuntimeWarning)
                best = ev

        if _ran_phase2:
            _strategy_parts.append("verification")
        elapsed_sec = time.perf_counter() - t_start

        if best is None:
            raise RuntimeError(
                "Failed to generate any valid split-plot design. "
                "Try increasing power_cfg.max_n or max_iter."
            )

        final_power = best["report"]["achieved_power"]
        if final_power + power_cfg.tol_power < target:
            _msg = (
                f"Split-plot design finished without converging to target power. "
                f"max_iter ({power_cfg.max_iter}) or max_n ({power_cfg.max_n}) reached. "
                f"Final power: {final_power:.4f} (Target: {target:.4f})."
            )
            warnings.warn(_msg, RuntimeWarning)
            _run_warnings.append(_msg)

        best["report"].update({
            "iteration": it,
            "elapsed_sec": round(float(elapsed_sec), 4),
            "search_strategy": "+".join(_strategy_parts),
            "verify_window": int(_verify_window),
            "random_state": int(design_opts.random_state) if design_opts.random_state is not None else None,
            "warnings": list(_run_warnings),
        })

        if export_diagnostics_to:
            try:
                export_paths = export_diagnostics(
                    X=best["_X"], design_df=best["design_df"],
                    output_path=export_diagnostics_to,
                    feature_names=None, formats=["html", "csv"], include_data=True,
                )
                best["report"]["diagnostic_exports"] = {k: str(v) for k, v in export_paths.items()}
            except Exception as e:
                best["report"]["diagnostic_exports_error"] = str(e)

        if export_report_to is not None:
            try:
                from .report import generate_report
                report_path = generate_report(
                    result=best, formula=formula, factors=factors,
                    power_cfg=power_cfg, output_path=export_report_to,
                    include_power_curve=False,
                )
                best["report"]["report_path"] = str(report_path)
            except Exception as e:
                best["report"]["report_path_error"] = str(e)

        best.pop("_n_wp", None)
        best.pop("_X", None)
        return best
    # =========================================================================
    # End split-plot path
    # =========================================================================

    lo = max(p_full + 1, design_opts.n_blocks if is_blocked else 1)
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

    # Common kwargs forwarded to every build_i_opt_design_with_idx call
    _search_kwargs: Dict[str, Any] = dict(
        cand=cand,
        formula=formula,
        criterion=design_opts.criterion,
        n_start=design_opts.starts,
        algo=design_opts.algo,
        max_iter=design_opts.max_iter,
        random_state=design_opts.random_state,
        workers=design_opts.workers,
        parallel_seed_stride=design_opts.parallel_seed_stride,
        jitter=design_opts.xtx_jitter,
        preallocate_categorical=design_opts.preallocate_categorical,
        alloc_min_per_cell=design_opts.alloc_min_per_cell,
        alloc_max_per_cell=design_opts.alloc_max_per_cell,
        alloc_wynn_max_iter=design_opts.alloc_wynn_max_iter,
        alloc_wynn_tol=design_opts.alloc_wynn_tol,
        cat_cells_cap=design_opts.cat_cells_cap,
    )

    # Kwargs for build_blocked_design (blocked mode only)
    _blocked_kwargs = dict(
        cand=cand,
        formula=formula,
        n_blocks=design_opts.n_blocks,
        block_sizes=design_opts.block_sizes,
        block_factor_name=design_opts.block_factor_name,
        aug_formula=aug_formula,
        criterion=design_opts.criterion,
        n_start=design_opts.starts,
        algo=design_opts.algo,
        max_iter=design_opts.max_iter,
        random_state=design_opts.random_state,
        workers=design_opts.workers,
        parallel_seed_stride=design_opts.parallel_seed_stride,
        jitter=design_opts.xtx_jitter,
        preallocate_categorical=design_opts.preallocate_categorical,
        alloc_min_per_cell=design_opts.alloc_min_per_cell,
        alloc_max_per_cell=design_opts.alloc_max_per_cell,
        alloc_wynn_max_iter=design_opts.alloc_wynn_max_iter,
        alloc_wynn_tol=design_opts.alloc_wynn_tol,
        cat_cells_cap=design_opts.cat_cells_cap,
    ) if is_blocked else {}

    while lo < hi and it < power_cfg.max_iter:
        n = (lo + hi) // 2
        if n > power_cfg.max_n:
            lo = hi  # out of range; exit
            break
        it += 1

        # 1) Build I-optimal design at n
        if is_blocked:
            design_df, X = build_blocked_design(n=n, **_blocked_kwargs)
            selected_idx = None
        else:
            design_df, selected_idx, _ = build_i_opt_design_with_idx(
                n=n, **_search_kwargs
            )
            X = X_cand[selected_idx, :]

        # 2) Compute power
        if mode == "contrast":
            power, lam = contrast_power(
                L=L_eff, delta=power_cfg.delta, X=X,
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
        diags = compute_design_metrics(X, X_cand=X_cand if not is_blocked else None)

        # Optional one-time candidate growth if conditioning is poor; re-evaluate at same n
        if (
            design_opts.allow_candidate_growth
            and not is_blocked
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
            # Update cand in shared kwargs after candidate growth
            _search_kwargs["cand"] = cand
            design_df, selected_idx, _ = build_i_opt_design_with_idx(
                n=n, **_search_kwargs
            )
            X = X_cand[selected_idx, :]
            if mode == "contrast":
                power, lam = contrast_power(
                    L=L_eff, delta=power_cfg.delta, X=X,
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
            diags = compute_design_metrics(X, X_cand=X_cand if not is_blocked else None)

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
            "block_structure": {
                "n_blocks": design_opts.n_blocks,
                "block_factor_name": design_opts.block_factor_name,
            } if is_blocked else None,
            "p_treat": int(p_treat),
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
                    "_X": X,
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
                    "_X": X,
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
            if is_blocked:
                design_df_v, X_v = build_blocked_design(n=n_check, **_blocked_kwargs)
                sel_idx_v = None
            else:
                design_df_v, sel_idx_v, _ = build_i_opt_design_with_idx(
                    n=n_check, **_search_kwargs
                )
                X_v = X_cand[sel_idx_v, :]
            if mode == "contrast":
                power_v, lam_v = contrast_power(
                    L=L_eff, delta=power_cfg.delta, X=X_v,
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
            diags_v = compute_design_metrics(X_v, X_cand=X_cand if not is_blocked else None)
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
                "block_structure": {
                    "n_blocks": design_opts.n_blocks,
                    "block_factor_name": design_opts.block_factor_name,
                } if is_blocked else None,
                "p_treat": int(p_treat),
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
                "_X": X_v,
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
    if is_blocked:
        final_X = best["_X"]
        # For blocked designs, p_full was estimated from the full candidate set.
        # The actual X_full from build_blocked_design is the ground truth for
        # column count — it can differ when n_b < p_treat within a block, causing
        # Patsy to omit columns for unseen factor levels.
        p_full = final_X.shape[1]
    else:
        final_X = best["_X_cand"][best["_selected_idx"], :]

    # MODIFIED: Add X.shape check per review
    if final_X.shape != (final_n, p_full):
        raise RuntimeError(
            f"Result validation failed: Final design matrix X has shape {final_X.shape}, "
            f"but report indicates (n, p_full) = ({final_n}, {p_full})."
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


__all__ = ["i_optimal_powered_design"]
