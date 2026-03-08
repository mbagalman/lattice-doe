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

from .config import PowerContrastConfig, PowerR2Config, DesignOptions
from .model_matrix import build_model_matrix
from .power import contrast_power, global_r2_power
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
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
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
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
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
        if isinstance(power_cfg, PowerContrastConfig):
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
    from .api import i_optimal_powered_design

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
            design_result = i_optimal_powered_design(
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
    }


# ---------------------------------------------------------------------------
# min_detectable_effect — bisection inversion of the power curve
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# compare_criteria — run I/D/A designs side-by-side
# ---------------------------------------------------------------------------

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
    # Deferred import avoids a load-time circular dependency.
    from .api import i_optimal_powered_design

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
    "power_curve_by_n",
    "power_curve_by_effect",
    "generate_power_curves",
    "power_sensitivity",
    "min_detectable_effect",
    "compare_criteria",
]
