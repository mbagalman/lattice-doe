# analysis.py
# License: MIT
"""
Post-hoc analysis utilities for power-assured experimental designs
===================================================================

This module provides functions for sensitivity analysis, minimum detectable
effect computation, and criterion comparison.  All functions either operate on
a **fixed** design matrix (no new DOE builds) or delegate design construction
to the canonical implementations in ``power_curves.py`` / ``api.py``.

Public functions
----------------
power_curve_by_n(...)      — sweep sample size; returns a plain DataFrame.
power_curve_by_effect(...) — sweep effect size at fixed n; returns a DataFrame.
generate_power_curves(...) — convenience wrapper combining both sweeps.
power_sensitivity(...)     — sweep sigma or R² on a fixed design.
min_detectable_effect(...) — invert the power curve to find the MDE.
compare_criteria(...)      — run I/D/A designs in one call and compare results.
robustness_report(...)     — multi-axis uncertainty summary for a fixed design.

Note on return types
--------------------
``power_curve_by_n`` and ``power_curve_by_effect`` in this module return plain
``pd.DataFrame`` objects (not dicts) for a simpler default experience.  To
access the figure object or the full result dict, call the canonical
implementations in ``iopt_power_design.power_curves`` directly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Literal, Tuple
import dataclasses
import numpy as np
import pandas as pd

import copy

from .config import PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig, DesignOptions
from .config import glm_fisher_weight
from .model_matrix import build_model_matrix
from .power import contrast_power, global_r2_power, contrast_power_sp, global_r2_power_sp, glm_contrast_power
from .split_plot import build_whole_plot_indicator, htc_factor_cols_from_names
from .power_curves import (
    power_curve_by_n as _power_curve_by_n_impl,
    power_curve_by_effect as _power_curve_by_effect_impl,
)


# ---------------------------------------------------------------------------
# Convenience wrappers — DataFrame-returning surface over power_curves.py
# ---------------------------------------------------------------------------

def power_curve_by_n(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig],
    design_opts: Optional[DesignOptions] = None,
    n_range: Optional[tuple] = None,
    n_points: int = 20,
    plot: bool = False,
    figsize: tuple = (8, 5),
    plot_backend: str = "matplotlib",
) -> pd.DataFrame:
    """Sweep n to visualize power as design size grows.

    Thin adapter over the canonical ``power_curves.power_curve_by_n``;
    returns only the curve DataFrame (discards the figure).

    Parameters
    ----------
    n_range : tuple of (int, int), optional
        ``(n_min, n_max)`` sweep bounds.  Defaults to a heuristic range
        based on the number of model parameters.
    n_points : int, optional
        Number of evenly-spaced n values to evaluate.  Default 20.
    figsize : tuple of (float, float), optional
        Matplotlib figure size.  Default ``(8, 5)``.  Ignored when
        ``plot_backend="plotly"``.

    Note: To access the figure object directly, call
    ``iopt_power_design.power_curves.power_curve_by_n()`` directly.
    """
    out = _power_curve_by_n_impl(
        formula=formula,
        factors=factors,
        power_cfg=power_cfg,
        design_opts=design_opts,
        n_range=n_range,
        n_points=n_points,
        plot=plot,
        figsize=figsize,
        plot_backend=plot_backend,
    )
    return out["data"]


def power_curve_by_effect(
    formula: str,
    factors: Dict[str, Any],
    n: int,
    power_cfg: Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig],
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
    plot_backend: str = "matplotlib",
) -> pd.DataFrame:
    """Sweep effect size (δ for contrast, R² for global) at fixed n.

    Thin adapter over the canonical ``power_curves.power_curve_by_effect``;
    returns only the curve DataFrame (discards the figure).  Column names
    are normalised for clarity:

    * Contrast mode: ``effect_size`` → ``effect_scale``
    * R² mode: ``effect_size`` → ``r2_target``

    Note: To access the figure object directly, call
    ``iopt_power_design.power_curves.power_curve_by_effect()`` directly.
    """
    out = _power_curve_by_effect_impl(
        formula=formula,
        factors=factors,
        n=n,
        power_cfg=power_cfg,
        design_opts=design_opts,
        plot=plot,
        plot_backend=plot_backend,
    )
    df = out["data"].copy()
    if "effect_size" in df.columns:
        if isinstance(power_cfg, (PowerContrastConfig, PowerGLMContrastConfig)):
            df = df.rename(columns={"effect_size": "effect_scale"})
        else:
            df = df.rename(columns={"effect_size": "r2_target"})
    return df


# ---------------------------------------------------------------------------
# generate_power_curves — combined convenience wrapper
# ---------------------------------------------------------------------------

def generate_power_curves(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    curve_type: Literal["by_n", "by_effect", "both"] = "both",
    n_for_effect: Optional[int] = None,
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
    plot_backend: str = "matplotlib",
) -> Dict[str, Any]:
    """Generate power curves for sensitivity analysis."""
    # Deferred import avoids a load-time circular dependency between
    # analysis.py → api.py → (nothing that imports analysis.py).
    from .api import find_optimal_design

    if design_opts is None:
        design_opts = DesignOptions()

    results: Dict[str, Any] = {}

    if curve_type in ("by_n", "both"):
        results["by_n"] = power_curve_by_n(
            formula, factors, power_cfg, design_opts=design_opts, plot=plot,
            plot_backend=plot_backend,
        )

    if curve_type in ("by_effect", "both"):
        if n_for_effect is None:
            design_result = find_optimal_design(
                formula, factors, power_cfg, design_opts=design_opts
            )
            n_for_effect = int(design_result["report"]["n"])

        results["by_effect"] = power_curve_by_effect(
            formula, factors, n_for_effect, power_cfg, design_opts=design_opts,
            plot=plot, plot_backend=plot_backend,
        )

    return results


# ---------------------------------------------------------------------------
# power_sensitivity — analytical sweep on a fixed design
# ---------------------------------------------------------------------------

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
    plot_backend: str = "matplotlib",
    eta_range: Optional[Tuple[float, float]] = None,
    eta_points: int = 20,
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
    X, _p_names = build_model_matrix(formula, design_df)
    n = int(X.shape[0])
    jitter = design_opts.xtx_jitter

    # --- Eta sweep (split-plot designs only) ---
    # When design_df has __wp_id__ and eta_range is supplied, sweep eta on the
    # fixed X/Z structure.  Z is inferred from __wp_id__ (balanced assumption).
    _eta_sweep_df: Optional[pd.DataFrame] = None
    if eta_range is not None and "__wp_id__" in design_df.columns:
        _n_wp = int(design_df["__wp_id__"].nunique())
        _s_per_wp = n // _n_wp
        Z_sp = build_whole_plot_indicator(n, _n_wp, _s_per_wp)
        _df_method = (
            design_opts.split_plot.df_method
            if design_opts.split_plot is not None
            else "sp_only"
        )
        _htc_factors_eta = (
            design_opts.split_plot.htc_factors
            if design_opts.split_plot is not None
            else []
        )
        _all_fcols_eta = [c for c in design_df.columns if c != "__wp_id__"]
        _htc_cols_eta = htc_factor_cols_from_names(
            _p_names, _htc_factors_eta, _all_fcols_eta,
        )
        _sigma_sp = power_cfg.sigma if isinstance(power_cfg, PowerContrastConfig) else 1.0
        _eta_rows = []
        for _eta in np.linspace(eta_range[0], eta_range[1], max(2, eta_points)):
            if isinstance(power_cfg, PowerContrastConfig):
                _pr = contrast_power_sp(
                    power_cfg.L, power_cfg.delta, X, Z_sp,
                    sigma_sp=_sigma_sp, eta=float(_eta),
                    alpha=power_cfg.alpha, df_method=_df_method, jitter=jitter,
                    htc_factor_cols=_htc_cols_eta,
                )
            else:
                _pr = global_r2_power_sp(
                    power_cfg.r2_target, X, Z_sp, sigma_sp=_sigma_sp,
                    eta=float(_eta), alpha=power_cfg.alpha,
                    lambda_mode=power_cfg.lambda_mode, jitter=jitter,
                )
            _eta_rows.append({
                "eta": float(_eta),
                "power": float(_pr.power),
                "noncentrality_lambda": float(_pr.lam),
            })
        _eta_sweep_df = pd.DataFrame(_eta_rows)

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
            if plot_backend == "plotly":
                from .plot_backends import plotly_sensitivity as _plotly_sensitivity
                fig = _plotly_sensitivity(df, power_cfg, float(nominal_pwr), n)
            else:
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
            "eta_sweep": _eta_sweep_df,
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
        if plot_backend == "plotly":
            from .plot_backends import plotly_sensitivity as _plotly_sensitivity
            fig = _plotly_sensitivity(df, power_cfg, float(nominal_pwr), n)
        else:
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
        "eta_sweep": _eta_sweep_df,
    }


# ---------------------------------------------------------------------------
# power_curve_by_baseline — GLM-specific: sweep baseline mean/probability
# ---------------------------------------------------------------------------

def power_curve_by_baseline(
    formula: str,
    factors: Dict[str, Any],
    design_df: pd.DataFrame,
    cfg: "PowerGLMContrastConfig",
    baseline_range: Tuple[float, float] = (0.05, 0.95),
    baseline_points: int = 30,
    design_opts: Optional["DesignOptions"] = None,
) -> pd.DataFrame:
    """Power as a function of GLM baseline event probability or rate.

    Holds the design fixed and sweeps the baseline mean (p₀ for binomial,
    μ₀ for Poisson), recomputing GLM power at each point.  Useful for
    sensitivity analysis: "if my true baseline rate differs from my
    assumption, how does power change?"

    For binomial models, p₀ = 0.5 maximises the Fisher information weight
    w = p₀(1−p₀), so power peaks near 0.5 and declines toward 0 or 1.
    For Poisson models, power increases monotonically with μ₀ because
    w = μ₀.

    Parameters
    ----------
    formula : str
        Patsy formula used when generating *design_df*.
    factors : dict
        Factor specifications matching the original design.
    design_df : DataFrame
        Fixed design to evaluate (no new search is performed).
    cfg : PowerGLMContrastConfig
        GLM power configuration.  ``baseline`` is swept; all other fields
        are held fixed.
    baseline_range : tuple of (lo, hi), default (0.05, 0.95)
        Range of baseline values to sweep.  For binomial, both endpoints
        must be in (0, 1).  For Poisson, both must be > 0.
    baseline_points : int, default 30
        Number of evenly-spaced baseline values.
    design_opts : DesignOptions, optional
        Used only for ``xtx_jitter``.  Defaults to ``DesignOptions()``.

    Returns
    -------
    pd.DataFrame
        Columns: ``baseline``, ``power``, ``lam``, ``family``, ``link``.
    """
    if design_opts is None:
        design_opts = DesignOptions()

    lo, hi = float(baseline_range[0]), float(baseline_range[1])
    if lo > hi:
        raise ValueError("baseline_range[0] must be <= baseline_range[1].")
    if baseline_points < 1:
        raise ValueError("baseline_points must be >= 1.")

    X, _ = build_model_matrix(formula, design_df)
    jitter = design_opts.xtx_jitter

    baseline_vals = np.linspace(lo, hi, max(1, baseline_points))
    rows = []
    for b in baseline_vals:
        _tmp = copy.copy(cfg)
        object.__setattr__(_tmp, "baseline", float(b))
        res = glm_contrast_power(_tmp, X, jitter=jitter)
        rows.append({
            "baseline": float(b),
            "power": float(res.power),
            "lam": float(res.lam),
            "family": cfg.family,
            "link": cfg.link,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# min_detectable_effect — bisection inversion of the power curve
# ---------------------------------------------------------------------------

def min_detectable_effect(
    design_df: pd.DataFrame,
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig],
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

    elif isinstance(power_cfg, PowerGLMContrastConfig):
        # GLM: bisect over a multiplicative scale on delta (LP scale).
        base_delta = np.asarray(power_cfg.delta)

        def _pwr_glm(scale: float) -> float:
            _tmp = copy.copy(power_cfg)
            object.__setattr__(_tmp, "delta", base_delta * scale)
            res = glm_contrast_power(_tmp, X, jitter=jitter)
            return float(res.power)

        lo_s, hi_s = 0.0, 20.0
        for _ in range(20):
            if _pwr_glm(hi_s) >= target_power:
                break
            hi_s *= 2.0
        else:
            return {
                "mde": float("inf"),
                "min_delta_lp": float("inf"),
                "achieved_power": _pwr_glm(hi_s),
                "n": n,
                "mode": "glm",
                "family": power_cfg.family,
                "baseline": float(power_cfg.baseline),
            }

        for _ in range(max_iter):
            mid = (lo_s + hi_s) / 2.0
            if _pwr_glm(mid) >= target_power:
                hi_s = mid
            else:
                lo_s = mid
            if hi_s - lo_s < tol:
                break

        mde = (lo_s + hi_s) / 2.0
        return {
            "mde": float(mde),
            "min_delta_lp": float(mde * float(np.linalg.norm(base_delta))),
            "achieved_power": float(_pwr_glm(mde)),
            "n": n,
            "mode": "glm",
            "family": power_cfg.family,
            "baseline": float(power_cfg.baseline),
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


# ---------------------------------------------------------------------------
# compare_criteria — run I/D/A designs side-by-side
# ---------------------------------------------------------------------------

def compare_criteria(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig],
    design_opts: Optional[DesignOptions] = None,
    criteria: Optional[List[str]] = None,
    plot: bool = False,
    figsize: Tuple[float, float] = (8, 5),
) -> Dict[str, Any]:
    """Run the powered-design search under multiple optimality criteria and compare.

    Executes ``find_optimal_design`` independently for each entry in
    *criteria* (default: all three — ``"I"``, ``"D"``, ``"A"``), then assembles
    a side-by-side summary to support criterion choice.  All runs share the same
    formula, factors, and power configuration; only the ``criterion`` field of
    *design_opts* is swapped per run.

    Parameters
    ----------
    formula : str
        Patsy formula string (e.g. ``"~ 1 + A + B + A:B"``).
    factors : dict
        Factor specifications (same format as ``find_optimal_design``).
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
        ``find_optimal_design`` (``design_df``, ``buckets_df``, ``report``).

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
    # Deferred import avoids a load-time circular dependency.
    from .api import find_optimal_design

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
        res = find_optimal_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=run_opts,
        )
        all_results[criterion] = res
        rpt = res["report"]
        diag = rpt.get("diagnostics") or {}
        row: Dict[str, Any] = {
            "criterion": criterion,
            "n": int(rpt["n"]),
            "achieved_power": float(rpt["achieved_power"]),
            "elapsed_sec": float(rpt.get("elapsed_sec", float("nan"))),
            "condition_number": float(diag.get("condition_number", float("nan"))),
            "d_efficiency": float(diag.get("d_efficiency", float("nan"))),
        }
        if isinstance(power_cfg, PowerGLMContrastConfig):
            row["family"] = power_cfg.family
            row["baseline"] = float(power_cfg.baseline)
        rows.append(row)

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


# ---------------------------------------------------------------------------
# robustness_report — multi-axis uncertainty summary for a fixed design
# ---------------------------------------------------------------------------

def _threshold_crossing(
    values: np.ndarray,
    powers: np.ndarray,
    target: float,
    increasing: bool,
) -> Optional[float]:
    """Return the linearly-interpolated x at which *powers* cross *target*.

    Parameters
    ----------
    values : array-like
        Swept x values (monotonically ordered).
    powers : array-like
        Corresponding power values.
    target : float
        The threshold to locate.
    increasing : bool
        True when power increases with *values* (e.g. effect size or r²).
        False when power decreases with *values* (e.g. sigma).

    Returns
    -------
    float or None
        Interpolated crossing x, or None if the threshold is never crossed.
    """
    passing = powers >= target
    if passing.all():
        # All scenarios pass: return the "safe" edge value
        return float(values[0]) if increasing else float(values[-1])
    if not passing.any():
        return None

    if increasing:
        # Power rises — find the first index that passes; interpolate left edge
        first_pass = int(np.where(passing)[0][0])
        if first_pass == 0:
            return float(values[0])
        x0, x1 = float(values[first_pass - 1]), float(values[first_pass])
        p0, p1 = float(powers[first_pass - 1]), float(powers[first_pass])
    else:
        # Power falls — find the last index that passes; interpolate right edge
        last_pass = int(np.where(passing)[0][-1])
        if last_pass == len(values) - 1:
            return float(values[-1])
        x0, x1 = float(values[last_pass]), float(values[last_pass + 1])
        p0, p1 = float(powers[last_pass]), float(powers[last_pass + 1])

    if p1 == p0:
        return x0
    return float(x0 + (target - p0) * (x1 - x0) / (p1 - p0))


def robustness_report(
    design_df: pd.DataFrame,
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    design_opts: Optional[DesignOptions] = None,
    sigma_range: Tuple[float, float] = (0.5, 2.0),
    sigma_points: int = 11,
    effect_range: Optional[Tuple[float, float]] = None,
    effect_points: int = 11,
    alpha_range: Tuple[float, float] = (0.01, 0.10),
    alpha_points: int = 9,
    plot: bool = False,
    figsize: Tuple[float, float] = (10, 4),
    plot_backend: str = "matplotlib",
) -> Dict[str, Any]:
    """Multi-axis robustness summary for a fixed design.

    Re-evaluates power across three independent assumption axes — effect size,
    noise (sigma), and significance level (alpha) — using the **fixed** design
    matrix from *design_df*.  No new I-optimal designs are built; all sweeps
    are fast, purely analytical evaluations of the power formula.

    For each axis the function reports:

    * A 1-D sweep DataFrame (power as a function of that axis).
    * A **threshold crossing**: the boundary value at which power transitions
      through the target (e.g. the maximum sigma that still achieves the target
      power).

    Summary statistics (worst, median, best power; fraction of all scenarios
    that pass the target) are computed by pooling every scenario from all three
    sweeps.

    Parameters
    ----------
    design_df : DataFrame
        The fixed design to evaluate (e.g. ``result["design_df"]``).
    formula : str
        Patsy formula used when building *design_df*.
    factors : dict
        Factor specifications (only needed to rebuild the model matrix).
    power_cfg : PowerContrastConfig or PowerR2Config
        Power configuration.  The type determines how each axis is interpreted.
    design_opts : DesignOptions, optional
        Used only for ``xtx_jitter``.  Defaults to ``DesignOptions()``.
    sigma_range : (lo, hi), default (0.5, 2.0)
        Absolute sigma values for the sigma sweep.  **Contrast mode only**;
        ignored in R² mode (sigma does not enter the R² power formula).
    sigma_points : int, default 11
        Number of evenly-spaced sigma values in the sweep.
    effect_range : (lo, hi) or None
        Range for the effect-size sweep.

        * **Contrast mode** — scale factors applied to ``power_cfg.delta``
          (e.g. ``(0.5, 2.0)`` means 0.5× to 2.0× the nominal delta).
          Defaults to ``(0.5, 2.0)``.
        * **R² mode** — direct ``r2_target`` values (e.g. ``(0.05, 0.50)``).
          Defaults to ``(0.05, 0.50)``.
    effect_points : int, default 11
        Number of evenly-spaced values in the effect sweep.
    alpha_range : (lo, hi), default (0.01, 0.10)
        Significance levels for the alpha sweep (both modes).
    alpha_points : int, default 9
        Number of evenly-spaced alpha values in the sweep.
    plot : bool, default False
        If True, produce a summary figure (2 panels for R² mode, 3 panels for
        contrast mode).
    figsize : tuple, default (10, 4)
        Figure size passed to matplotlib.
    plot_backend : str, default "matplotlib"
        Plotting backend; only ``"matplotlib"`` is currently supported.

    Returns
    -------
    dict with keys:

    ``mode``
        ``"contrast"`` or ``"r2"``.
    ``nominal_power``
        Power at the configured nominal assumptions.
    ``effect_sweep``
        DataFrame — columns: ``effect_scale`` (contrast) or ``r2_target`` (R²),
        ``power``, ``noncentrality_lambda``.
    ``sigma_sweep``
        DataFrame — columns: ``sigma``, ``power``, ``noncentrality_lambda``.
        ``None`` in R² mode.
    ``alpha_sweep``
        DataFrame — columns: ``alpha``, ``power``, ``noncentrality_lambda``.
    ``summary``
        dict — ``worst_power``, ``median_power``, ``best_power``,
        ``power_target``, ``pct_scenarios_passing``.
    ``thresholds``
        dict — ``max_sigma_for_target`` (contrast only, else ``None``),
        ``min_effect_for_target``, ``min_alpha_for_target``.
    ``figure``
        matplotlib Figure if *plot* is True, else ``None``.

    Raises
    ------
    ValueError
        For invalid range parameters.

    Examples
    --------
    >>> report = robustness_report(
    ...     result["design_df"], formula, factors, power_cfg
    ... )
    >>> report["summary"]
    {'worst_power': 0.42, 'median_power': 0.81, 'best_power': 0.99,
     'power_target': 0.80, 'pct_scenarios_passing': 0.61}
    >>> report["thresholds"]["max_sigma_for_target"]
    1.43
    """
    if design_opts is None:
        design_opts = DesignOptions()

    # Build the model matrix once from the fixed design — all sweeps reuse X
    X, _ = build_model_matrix(formula, design_df)
    n = int(X.shape[0])
    jitter = design_opts.xtx_jitter
    mode = "contrast" if isinstance(power_cfg, PowerContrastConfig) else "r2"
    power_target = float(power_cfg.power)

    # ------------------------------------------------------------------ #
    # Input validation                                                     #
    # ------------------------------------------------------------------ #
    if sigma_range[0] <= 0 or sigma_range[1] <= 0:
        raise ValueError("sigma_range values must be > 0")
    if sigma_range[0] >= sigma_range[1]:
        raise ValueError("sigma_range[0] must be < sigma_range[1]")
    if sigma_points < 2:
        raise ValueError("sigma_points must be >= 2")
    if alpha_range[0] <= 0 or alpha_range[1] <= 0:
        raise ValueError("alpha_range values must be > 0")
    if alpha_range[0] >= alpha_range[1]:
        raise ValueError("alpha_range[0] must be < alpha_range[1]")
    if alpha_range[0] >= 1 or alpha_range[1] >= 1:
        raise ValueError("alpha_range values must be < 1")
    if alpha_points < 2:
        raise ValueError("alpha_points must be >= 2")

    # Set mode-appropriate defaults for effect_range
    if effect_range is None:
        effect_range = (0.5, 2.0) if mode == "contrast" else (0.05, 0.50)

    if effect_range[0] <= 0 or effect_range[1] <= 0:
        raise ValueError("effect_range values must be > 0")
    if effect_range[0] >= effect_range[1]:
        raise ValueError("effect_range[0] must be < effect_range[1]")
    if mode == "r2" and (effect_range[0] >= 1 or effect_range[1] >= 1):
        raise ValueError(
            "effect_range values must be < 1 in R² mode (they are r2_target values)"
        )
    if effect_points < 2:
        raise ValueError("effect_points must be >= 2")

    # ------------------------------------------------------------------ #
    # Nominal power at configured assumptions                              #
    # ------------------------------------------------------------------ #
    if mode == "contrast":
        nominal_pwr, _ = contrast_power(
            L=power_cfg.L,
            delta=power_cfg.delta,
            X=X,
            sigma=power_cfg.sigma,
            alpha=power_cfg.alpha,
            jitter=jitter,
        )
    else:
        nominal_pwr, _ = global_r2_power(
            r2_target=power_cfg.r2_target,
            X=X,
            alpha=power_cfg.alpha,
            lambda_mode=power_cfg.lambda_mode,
        )

    # ------------------------------------------------------------------ #
    # 1. Effect sweep                                                      #
    # ------------------------------------------------------------------ #
    effect_vals = np.linspace(effect_range[0], effect_range[1], effect_points)
    effect_rows = []
    for ev in effect_vals:
        if mode == "contrast":
            pwr, lam = contrast_power(
                L=power_cfg.L,
                delta=power_cfg.delta * float(ev),  # scale the nominal delta
                X=X,
                sigma=power_cfg.sigma,
                alpha=power_cfg.alpha,
                jitter=jitter,
            )
            effect_rows.append({
                "effect_scale": float(ev),
                "power": float(pwr),
                "noncentrality_lambda": float(lam),
            })
        else:
            pwr, lam = global_r2_power(
                r2_target=float(ev),
                X=X,
                alpha=power_cfg.alpha,
                lambda_mode=power_cfg.lambda_mode,
            )
            effect_rows.append({
                "r2_target": float(ev),
                "power": float(pwr),
                "noncentrality_lambda": float(lam),
            })
    effect_sweep_df = pd.DataFrame(effect_rows)
    effect_col = "effect_scale" if mode == "contrast" else "r2_target"

    # ------------------------------------------------------------------ #
    # 2. Sigma sweep (contrast mode only)                                  #
    # ------------------------------------------------------------------ #
    sigma_sweep_df: Optional[pd.DataFrame] = None
    if mode == "contrast":
        sigma_vals = np.linspace(sigma_range[0], sigma_range[1], sigma_points)
        sigma_rows = []
        for sigma in sigma_vals:
            pwr, lam = contrast_power(
                L=power_cfg.L,
                delta=power_cfg.delta,
                X=X,
                sigma=float(sigma),
                alpha=power_cfg.alpha,
                jitter=jitter,
            )
            sigma_rows.append({
                "sigma": float(sigma),
                "power": float(pwr),
                "noncentrality_lambda": float(lam),
            })
        sigma_sweep_df = pd.DataFrame(sigma_rows)

    # ------------------------------------------------------------------ #
    # 3. Alpha sweep (both modes)                                          #
    # ------------------------------------------------------------------ #
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], alpha_points)
    alpha_rows = []
    for alpha in alpha_vals:
        if mode == "contrast":
            pwr, lam = contrast_power(
                L=power_cfg.L,
                delta=power_cfg.delta,
                X=X,
                sigma=power_cfg.sigma,
                alpha=float(alpha),
                jitter=jitter,
            )
        else:
            pwr, lam = global_r2_power(
                r2_target=power_cfg.r2_target,
                X=X,
                alpha=float(alpha),
                lambda_mode=power_cfg.lambda_mode,
            )
        alpha_rows.append({
            "alpha": float(alpha),
            "power": float(pwr),
            "noncentrality_lambda": float(lam),
        })
    alpha_sweep_df = pd.DataFrame(alpha_rows)

    # ------------------------------------------------------------------ #
    # 4. Summary statistics (pooled across all sweeps)                     #
    # ------------------------------------------------------------------ #
    all_powers_parts = [
        effect_sweep_df["power"].values,
        alpha_sweep_df["power"].values,
    ]
    if sigma_sweep_df is not None:
        all_powers_parts.append(sigma_sweep_df["power"].values)
    all_powers = np.concatenate(all_powers_parts)

    summary = {
        "worst_power": float(np.min(all_powers)),
        "median_power": float(np.median(all_powers)),
        "best_power": float(np.max(all_powers)),
        "power_target": power_target,
        "pct_scenarios_passing": float(np.mean(all_powers >= power_target)),
    }

    # ------------------------------------------------------------------ #
    # 5. Threshold crossings                                               #
    # ------------------------------------------------------------------ #
    # Effect sweep: power increases with effect size — find smallest value
    # at which power reaches the target.
    min_effect = _threshold_crossing(
        values=effect_sweep_df[effect_col].values,
        powers=effect_sweep_df["power"].values,
        target=power_target,
        increasing=True,
    )

    # Alpha sweep: as alpha increases power increases — find the smallest
    # (most conservative) alpha that still meets the target.
    min_alpha = _threshold_crossing(
        values=alpha_sweep_df["alpha"].values,
        powers=alpha_sweep_df["power"].values,
        target=power_target,
        increasing=True,
    )

    # Sigma sweep (contrast only): as sigma increases power decreases — find
    # the largest sigma that still meets the target.
    max_sigma: Optional[float] = None
    if sigma_sweep_df is not None:
        max_sigma = _threshold_crossing(
            values=sigma_sweep_df["sigma"].values,
            powers=sigma_sweep_df["power"].values,
            target=power_target,
            increasing=False,
        )

    thresholds: Dict[str, Optional[float]] = {
        "max_sigma_for_target": max_sigma,
        "min_effect_for_target": min_effect,
        "min_alpha_for_target": min_alpha,
    }

    # ------------------------------------------------------------------ #
    # 6. Optional plot                                                     #
    # ------------------------------------------------------------------ #
    fig = None
    if plot:
        try:
            import matplotlib.pyplot as plt

            n_panels = 3 if mode == "contrast" else 2
            fig, axes = plt.subplots(1, n_panels, figsize=figsize)
            if n_panels == 2:
                axes = list(axes)
            else:
                axes = list(axes)

            def _decorate(ax: Any, xlabel: str, nominal_x: float, title: str) -> None:
                ax.axhline(y=power_target, color="red", linestyle="--", linewidth=1.2,
                           label=f"Target power = {power_target:.2f}")
                ax.axvline(x=nominal_x, color="gray", linestyle="--", linewidth=1,
                           label=f"Nominal = {nominal_x:.3g}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Statistical Power")
                ax.set_ylim([0, 1.05])
                ax.set_title(title)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            # Panel 0: effect sweep
            ax0 = axes[0]
            if mode == "contrast":
                ax0.plot(effect_sweep_df["effect_scale"], effect_sweep_df["power"],
                         "b-", linewidth=2)
                _decorate(ax0, "Effect scale (× δ)", 1.0, "Effect Sweep")
                if min_effect is not None:
                    ax0.axvline(x=min_effect, color="darkorange", linestyle=":",
                                linewidth=1.2, label=f"Threshold ≈ {min_effect:.3g}")
                    ax0.legend(fontsize=7)
            else:
                ax0.plot(effect_sweep_df["r2_target"], effect_sweep_df["power"],
                         "b-", linewidth=2)
                _decorate(ax0, "R² target", power_cfg.r2_target, "Effect Sweep (R²)")
                if min_effect is not None:
                    ax0.axvline(x=min_effect, color="darkorange", linestyle=":",
                                linewidth=1.2, label=f"Threshold ≈ {min_effect:.3g}")
                    ax0.legend(fontsize=7)

            # Panel 1: sigma sweep (contrast) or alpha sweep (r2)
            if mode == "contrast":
                ax1 = axes[1]
                ax1.plot(sigma_sweep_df["sigma"], sigma_sweep_df["power"],
                         "b-", linewidth=2)
                _decorate(ax1, "σ (residual std dev)", power_cfg.sigma, "Sigma Sweep")
                if max_sigma is not None:
                    ax1.axvline(x=max_sigma, color="darkorange", linestyle=":",
                                linewidth=1.2, label=f"Threshold ≈ {max_sigma:.3g}")
                    ax1.legend(fontsize=7)
                # Alpha sweep in Panel 2
                ax2 = axes[2]
                ax2.plot(alpha_sweep_df["alpha"], alpha_sweep_df["power"],
                         "b-", linewidth=2)
                _decorate(ax2, "α (significance level)", power_cfg.alpha, "Alpha Sweep")
                if min_alpha is not None:
                    ax2.axvline(x=min_alpha, color="darkorange", linestyle=":",
                                linewidth=1.2, label=f"Threshold ≈ {min_alpha:.3g}")
                    ax2.legend(fontsize=7)
            else:
                # R² mode has no sigma sweep — alpha in Panel 1
                ax1 = axes[1]
                ax1.plot(alpha_sweep_df["alpha"], alpha_sweep_df["power"],
                         "b-", linewidth=2)
                _decorate(ax1, "α (significance level)", power_cfg.alpha, "Alpha Sweep")
                if min_alpha is not None:
                    ax1.axvline(x=min_alpha, color="darkorange", linestyle=":",
                                linewidth=1.2, label=f"Threshold ≈ {min_alpha:.3g}")
                    ax1.legend(fontsize=7)

            plt.suptitle(
                f"Robustness Report  (n = {n}, nominal power = {float(nominal_pwr):.3f})",
                fontsize=11,
                fontweight="bold",
            )
            plt.tight_layout()
        except ImportError:
            pass  # matplotlib unavailable — return fig=None

    return {
        "mode": mode,
        "nominal_power": float(nominal_pwr),
        "effect_sweep": effect_sweep_df,
        "sigma_sweep": sigma_sweep_df,
        "alpha_sweep": alpha_sweep_df,
        "summary": summary,
        "thresholds": thresholds,
        "figure": fig,
    }


# ---------------------------------------------------------------------------
# power_curve_by_wp — power vs n_whole_plots for split-plot designs
# ---------------------------------------------------------------------------

def power_curve_by_wp(
    formula: str,
    factors: Dict[str, Any],
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    subplots_per_wp: int,
    htc_factors: List[str],
    eta: float,
    *,
    wp_range: Optional[Tuple[int, int]] = None,
    wp_points: int = 10,
    design_opts: Optional[DesignOptions] = None,
    plot_backend: str = "matplotlib",
    figsize: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Power vs number of whole plots curve for a split-plot design.

    Sweeps ``n_whole_plots`` from ``wp_range[0]`` to ``wp_range[1]``,
    builds a new split-plot design at each size, evaluates GLS power, and
    returns a DataFrame with columns: ``n_wp``, ``n_total``, ``power``,
    ``noncentrality_lambda``.

    Parameters
    ----------
    formula : str
        Patsy model formula.
    factors : dict
        Factor specifications (continuous tuples or categorical lists).
    power_cfg : PowerContrastConfig or PowerR2Config
        Power target configuration.
    subplots_per_wp : int
        Fixed number of sub-plots per whole plot.
    htc_factors : list of str
        Names of the hard-to-change (whole-plot) factors.
    eta : float
        Variance ratio σ²_wp / σ²_sp.
    wp_range : (int, int), optional
        ``(min_n_wp, max_n_wp)`` sweep bounds.  Defaults to
        ``(2, 2 + wp_points)``.
    wp_points : int, default 10
        Number of evenly-spaced n_whole_plots values to evaluate.
    design_opts : DesignOptions, optional
        Controls design-build settings (starts, random_state, etc.).
    plot_backend : str, default "matplotlib"
        Ignored (reserved for future plot support).
    figsize : tuple, optional
        Ignored (reserved for future plot support).

    Returns
    -------
    pd.DataFrame
        Columns: ``n_wp``, ``n_total``, ``power``, ``noncentrality_lambda``.
        Has exactly ``wp_points`` rows (one per evaluated n_wp value).
    """
    # Deferred imports to avoid circular dependencies
    from .candidate import build_split_plot_candidate
    from .iopt_search import build_split_plot_design

    if design_opts is None:
        design_opts = DesignOptions()

    if wp_range is None:
        wp_range = (2, max(3, 2 + wp_points - 1))

    n_wp_vals = np.round(
        np.linspace(wp_range[0], wp_range[1], max(1, wp_points))
    ).astype(int)

    sigma_sp = power_cfg.sigma if isinstance(power_cfg, PowerContrastConfig) else 1.0
    df_method = (
        design_opts.split_plot.df_method
        if design_opts.split_plot is not None
        else "sp_only"
    )

    # Compute candidate pool sizes from DesignOptions.candidate_points.
    _htc_set = set(htc_factors)
    _etc_factors_wp = [f for f in factors if f not in _htc_set]
    _n_htc_wp = len(htc_factors)
    _n_etc_wp = len(_etc_factors_wp)
    _n_all_wp = max(1, _n_htc_wp + _n_etc_wp)
    _cand_pts = int(design_opts.candidate_points)
    _n_wp_cand = max(10, int(_cand_pts * _n_htc_wp / _n_all_wp))
    _n_sp_cand = max(10, int(_cand_pts * _n_etc_wp / _n_all_wp)) if _n_etc_wp > 0 else 1

    rows = []
    for n_wp in n_wp_vals:
        n_wp = int(n_wp)
        n_total = n_wp * subplots_per_wp
        try:
            sp_cand = build_split_plot_candidate(
                factors, htc_factors, n_wp, subplots_per_wp,
                random_state=design_opts.random_state,
            )
            design_df_, X_ = build_split_plot_design(
                sp_cand, formula, n_wp, subplots_per_wp,
                htc_factors, eta,
                factors=factors,
                criterion=design_opts.criterion,
                starts=design_opts.starts,
                max_iter=design_opts.max_iter,
                random_state=design_opts.random_state,
                jitter=design_opts.xtx_jitter,
                n_wp_cand=_n_wp_cand,
                n_sp_cand=_n_sp_cand,
            )
            Z_ = build_whole_plot_indicator(n_total, n_wp, subplots_per_wp)
            if isinstance(power_cfg, PowerContrastConfig):
                _, _p_names_ = build_model_matrix(formula, design_df_)
                _all_fcols_ = [c for c in design_df_.columns if c != "__wp_id__"]
                _htc_cols_ = htc_factor_cols_from_names(
                    _p_names_, htc_factors, _all_fcols_,
                )
                pr = contrast_power_sp(
                    power_cfg.L, power_cfg.delta, X_, Z_,
                    sigma_sp=sigma_sp, eta=eta, alpha=power_cfg.alpha,
                    df_method=df_method, jitter=design_opts.xtx_jitter,
                    htc_factor_cols=_htc_cols_,
                )
            else:
                pr = global_r2_power_sp(
                    power_cfg.r2_target, X_, Z_, sigma_sp=sigma_sp,
                    eta=eta, alpha=power_cfg.alpha,
                    lambda_mode=power_cfg.lambda_mode,
                    jitter=design_opts.xtx_jitter,
                )
            rows.append({
                "n_wp": n_wp,
                "n_total": n_total,
                "power": float(pr.power),
                "noncentrality_lambda": float(pr.lam),
            })
        except Exception:
            rows.append({
                "n_wp": n_wp,
                "n_total": n_total,
                "power": float("nan"),
                "noncentrality_lambda": float("nan"),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MR-7: Multi-response analysis functions
# ---------------------------------------------------------------------------

def power_curve_by_n_multiresponse(
    formula: str,
    factors: Dict[str, Any],
    multi_cfg: "MultiResponseOptions",
    n_range: Tuple[int, int] = (5, 100),
    n_points: int = 20,
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
    plot_backend: Literal["matplotlib", "plotly"] = "matplotlib",
) -> pd.DataFrame:
    """Power vs n curve for each response plus the combined power.

    Sweeps n from ``n_range[0]`` to ``n_range[1]`` at ``n_points`` evenly
    spaced values.  At each n, builds an I-optimal design (shared or compound
    criterion) and evaluates per-response powers plus the combined power.

    Parameters
    ----------
    formula : str
        Global Patsy formula (right-hand side).
    factors : dict
        Factor definitions.
    multi_cfg : MultiResponseOptions
        Per-response power configs and combination rule.
    n_range : tuple of (int, int), default (5, 100)
        Inclusive sweep bounds ``(n_min, n_max)``.
    n_points : int, default 20
        Number of evenly-spaced n values to evaluate.  Output has exactly
        this many rows (ties in the integer grid are kept as-is).
    design_opts : DesignOptions or None
        Design search options.  Defaults to ``DesignOptions()``.
    plot : bool, default False
        If True, produce a power-vs-n line chart (one line per response
        plus a dashed combined-power line).
    plot_backend : {"matplotlib", "plotly"}, default "matplotlib"
        Plotting backend.

    Returns
    -------
    pd.DataFrame
        Columns: ``n``, ``combined_power``, and ``<name>_power`` for each
        response in ``multi_cfg.responses``.  Total: ``len(responses) + 2``
        columns; exactly ``n_points`` rows.
    """
    # Deferred imports keep module load fast and avoid circular refs.
    from .candidate import build_candidate, estimate_candidate_size
    from .iopt_search import build_i_opt_design_with_idx, build_compound_design
    from .utils import validate_factors
    from .config import MultiResponseOptions as _MROpt
    from .power import eval_response_power, combine_powers as _combine

    if design_opts is None:
        design_opts = DesignOptions()

    validate_factors(factors)

    n_min, n_max = int(n_range[0]), int(n_range[1])
    if n_min < 1:
        raise ValueError(f"n_range[0] must be >= 1, got {n_min}.")
    if n_min >= n_max:
        raise ValueError("n_range[0] must be < n_range[1].")
    if n_points < 1:
        raise ValueError("n_points must be >= 1.")

    # --- Candidate set (built once) ---
    if design_opts.auto_candidate:
        candidate_points = estimate_candidate_size(
            formula=formula, factors=factors,
            cand_min=design_opts.cand_min, cand_max=design_opts.cand_max,
            cat_cells_cap=design_opts.cat_cells_cap,
            per_cell_alpha=design_opts.per_cell_alpha,
            per_cell_min=design_opts.per_cell_min,
            per_cell_max=design_opts.per_cell_max,
            seed=design_opts.random_state,
        )
    else:
        candidate_points = int(design_opts.candidate_points)

    cand = build_candidate(
        factors=factors, candidate_points=candidate_points,
        seed=design_opts.random_state,
        constraint_func=design_opts.constraint_func,
        cat_cells_cap=design_opts.cat_cells_cap,
    )

    # Detect compound path (any response formula differs from global).
    _compound = any(
        r.formula is not None and r.formula != formula
        for r in multi_cfg.responses
    )

    rule = multi_cfg.power_combination
    weights = [r.weight for r in multi_cfg.responses]
    resp_names = [r.name for r in multi_cfg.responses]

    if _compound:
        candidates_list: List[np.ndarray] = []
        p_names_list: List[List[str]] = []
        for r in multi_cfg.responses:
            f_k = r.formula if r.formula is not None else formula
            X_cand_k, p_names_k = build_model_matrix(f_k, cand)
            candidates_list.append(X_cand_k)
            p_names_list.append(list(p_names_k))
        n_cand = candidates_list[0].shape[0]
    else:
        X_cand, p_names_global = build_model_matrix(formula, cand)
        n_cand = X_cand.shape[0]

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

    # n sweep: exactly n_points values (may include integer duplicates for small ranges)
    n_vals: List[int] = np.round(np.linspace(n_min, n_max, n_points)).astype(int).tolist()

    rows: List[Dict[str, Any]] = []
    for n_ in n_vals:
        n_safe = min(int(n_), n_cand)
        try:
            if _compound:
                idx_ = build_compound_design(
                    candidates_list, weights, n_safe,
                    criterion=design_opts.criterion,
                    n_start=design_opts.starts,
                    max_iter=design_opts.max_iter,
                    random_state=design_opts.random_state,
                    jitter=design_opts.xtx_jitter,
                )
                per_r_ = [
                    eval_response_power(r_k, X_cand_k[idx_], p_names_k,
                                        jitter=design_opts.xtx_jitter)
                    for r_k, X_cand_k, p_names_k in zip(
                        multi_cfg.responses, candidates_list, p_names_list
                    )
                ]
            else:
                _, idx_, _ = build_i_opt_design_with_idx(n=n_safe, **_search_kwargs)
                X_ = X_cand[idx_]
                per_r_ = [
                    eval_response_power(r, X_, p_names_global,
                                        jitter=design_opts.xtx_jitter)
                    for r in multi_cfg.responses
                ]
            combined_ = _combine([d["power"] for d in per_r_], weights, rule)
            row: Dict[str, Any] = {"n": int(n_), "combined_power": float(combined_)}
            for rd in per_r_:
                row[f"{rd['name']}_power"] = float(rd["power"])
        except Exception:
            row = {"n": int(n_), "combined_power": float("nan")}
            for name in resp_names:
                row[f"{name}_power"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)

    if plot:
        if plot_backend == "plotly":
            try:
                import plotly.graph_objects as go  # type: ignore[import]
                fig = go.Figure()
                for name in resp_names:
                    col = f"{name}_power"
                    if col in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df["n"], y=df[col], mode="lines", name=name,
                        ))
                fig.add_trace(go.Scatter(
                    x=df["n"], y=df["combined_power"], mode="lines",
                    name="combined", line=dict(dash="dash"),
                ))
                fig.update_layout(
                    xaxis_title="n",
                    yaxis_title="Power",
                    title="Multi-Response Power vs n",
                )
            except ImportError:
                pass
        else:
            try:
                import matplotlib.pyplot as plt  # type: ignore[import]
                fig_mr, ax = plt.subplots(figsize=(8, 5))
                for name in resp_names:
                    col = f"{name}_power"
                    if col in df.columns:
                        ax.plot(df["n"], df[col], label=name)
                ax.plot(df["n"], df["combined_power"], "--", label="combined", linewidth=2)
                ax.set_xlabel("n (sample size)")
                ax.set_ylabel("Statistical Power")
                ax.set_ylim([0, 1.05])
                ax.set_title("Multi-Response Power vs n")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
            except ImportError:
                pass

    return df


def multiresponse_sensitivity(
    formula: str,
    factors: Dict[str, Any],
    multi_cfg: "MultiResponseOptions",
    fixed_n: int,
    sigma_range: Tuple[float, float] = (0.5, 3.0),
    sigma_points: int = 20,
    design_opts: Optional[DesignOptions] = None,
) -> pd.DataFrame:
    """Sigma sensitivity for a multi-response design at fixed n.

    Builds a single I-optimal design at ``fixed_n`` runs, then sweeps a
    common sigma scale factor across ``sigma_range``.  Each response's
    ``sigma`` is multiplied by the scale factor at every point.

    Only valid for contrast-mode responses (``PowerContrastConfig``).

    Parameters
    ----------
    formula : str
        Global Patsy formula (right-hand side).
    factors : dict
        Factor definitions.
    multi_cfg : MultiResponseOptions
        Per-response power configs. All must be ``PowerContrastConfig``.
    fixed_n : int
        Number of runs; one design is built at this n for all sweep points.
    sigma_range : tuple of (lo, hi), default (0.5, 3.0)
        Scale-factor range.  Each response's sigma is multiplied by the
        scale factor; must satisfy 0 < lo < hi.
    sigma_points : int, default 20
        Number of evenly-spaced scale values; must be >= 2.
    design_opts : DesignOptions or None
        Design search options.  Defaults to ``DesignOptions()``.

    Returns
    -------
    pd.DataFrame
        Columns: ``sigma_scale``, ``combined_power``, and ``<name>_power``
        for each response.  Power decreases monotonically as sigma_scale
        increases (larger sigma → smaller noncentrality → lower power).

    Raises
    ------
    TypeError
        If any response uses ``PowerR2Config``; sigma scaling is undefined
        for R²-mode responses.
    """
    from .candidate import build_candidate, estimate_candidate_size
    from .iopt_search import build_i_opt_design_with_idx
    from .utils import validate_factors
    from .config import PowerR2Config as _PowerR2Config
    from .power import combine_powers as _combine

    if design_opts is None:
        design_opts = DesignOptions()

    for r in multi_cfg.responses:
        if isinstance(r.power_cfg, _PowerR2Config):
            raise TypeError(
                f"multiresponse_sensitivity only supports PowerContrastConfig responses; "
                f"response '{r.name}' uses PowerR2Config. "
                "Sigma scaling is undefined for R²-mode responses."
            )

    validate_factors(factors)

    if sigma_range[0] <= 0 or sigma_range[1] <= 0:
        raise ValueError("sigma_range values must be > 0.")
    if sigma_range[0] >= sigma_range[1]:
        raise ValueError("sigma_range[0] must be < sigma_range[1].")
    if sigma_points < 2:
        raise ValueError("sigma_points must be >= 2.")
    if fixed_n < 1:
        raise ValueError("fixed_n must be >= 1.")

    if design_opts.auto_candidate:
        candidate_points = estimate_candidate_size(
            formula=formula, factors=factors,
            cand_min=design_opts.cand_min, cand_max=design_opts.cand_max,
            cat_cells_cap=design_opts.cat_cells_cap,
            per_cell_alpha=design_opts.per_cell_alpha,
            per_cell_min=design_opts.per_cell_min,
            per_cell_max=design_opts.per_cell_max,
            seed=design_opts.random_state,
        )
    else:
        candidate_points = int(design_opts.candidate_points)

    cand = build_candidate(
        factors=factors, candidate_points=candidate_points,
        seed=design_opts.random_state,
        constraint_func=design_opts.constraint_func,
        cat_cells_cap=design_opts.cat_cells_cap,
    )
    X_cand, p_names_global = build_model_matrix(formula, cand)

    n_safe = min(fixed_n, X_cand.shape[0])
    _, idx_, _ = build_i_opt_design_with_idx(
        n=n_safe,
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
    X = X_cand[idx_]

    rule = multi_cfg.power_combination
    weights = [r.weight for r in multi_cfg.responses]
    scale_vals = np.linspace(sigma_range[0], sigma_range[1], sigma_points).tolist()

    rows: List[Dict[str, Any]] = []
    for scale in scale_vals:
        per_r_: List[Dict[str, Any]] = []
        for r in multi_cfg.responses:
            cfg = r.power_cfg  # PowerContrastConfig guaranteed
            scaled_sigma = float(cfg.sigma) * float(scale)
            pwr, lam = contrast_power(
                L=cfg.L,
                delta=cfg.delta,
                X=X,
                sigma=scaled_sigma,
                alpha=cfg.alpha,
                jitter=design_opts.xtx_jitter,
            )
            per_r_.append({"name": r.name, "power": float(pwr), "lam": float(lam)})
        combined_ = _combine([d["power"] for d in per_r_], weights, rule)
        row: Dict[str, Any] = {
            "sigma_scale": float(scale),
            "combined_power": float(combined_),
        }
        for rd in per_r_:
            row[f"{rd['name']}_power"] = float(rd["power"])
        rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    "power_curve_by_n",
    "power_curve_by_effect",
    "generate_power_curves",
    "power_sensitivity",
    "power_curve_by_baseline",
    "min_detectable_effect",
    "compare_criteria",
    "robustness_report",
    "power_curve_by_wp",
    "power_curve_by_n_multiresponse",
    "multiresponse_sensitivity",
]
