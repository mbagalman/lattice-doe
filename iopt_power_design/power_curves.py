# power_curves.py
# License: MIT
"""
Power curve generation for sensitivity analysis
===============================================

This module provides functions to generate power curves showing how
statistical power varies with:
  - Sample size (n)
  - Effect size (delta for contrasts, R² for global tests)
  - Residual standard deviation (sigma)
  - Significance level (alpha)

Power curves help researchers understand:
  - How robust their design is to assumption violations
  - The trade-off between sample size and detectable effect size
  - Where diminishing returns occur in increasing n

The module supports both interactive (matplotlib) and data-only outputs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd

from .config import PowerContrastConfig, PowerR2Config, DesignOptions
from .candidate import build_candidate, estimate_candidate_size
from .model_matrix import build_model_matrix
from .iopt_search import build_i_opt_design, build_i_opt_design_with_idx
from .power import contrast_power, global_r2_power
from .diag_metrics import compute_design_metrics

# Optional plotting support
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def power_curve_by_n(
    formula: str,
    factors: dict,
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    n_range: Optional[Tuple[int, int]] = None,
    n_points: int = 20,
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
    figsize: Tuple[float, float] = (8, 5),
    plot_backend: Literal["matplotlib", "plotly"] = "matplotlib",
) -> Dict[str, Union[pd.DataFrame, Optional["Figure"]]]:
    """Generate power curve as a function of sample size n.
    
    For each n value, builds an I-optimal design and computes actual
    power based on the realized design matrix.
    
    Parameters
    ----------
    formula : str
        Patsy-style model formula.
    factors : dict
        Factor specifications.
    power_cfg : PowerContrastConfig or PowerR2Config
        Power configuration (effect size, alpha, sigma stay fixed).
    n_range : tuple of (min_n, max_n), optional
        Range of n values to evaluate. If None, automatically determined
        based on achieving 0.1 to 0.99 power.
    n_points : int, default 20
        Number of n values to evaluate (evenly spaced or log-spaced).
    design_opts : DesignOptions, optional
        Design generation options.
    plot : bool, default False
        If True and matplotlib available, return a figure object.
    figsize : tuple, default (8, 5)
        Figure size if plotting.
    
    Returns
    -------
    dict
        {
          'data': DataFrame with columns [n, power, lambda, d_efficiency, i_criterion],
          'figure': matplotlib Figure if plot=True and available, else None,
          'target_n': int, approximate n to achieve target power
        }
    
    Notes
    -----
    This function is computationally intensive as it generates a full
    I-optimal design for each n value. Consider using fewer n_points
    for initial exploration.
    """
    # --- Reviewer Feedback: Validation ---
    if n_points <= 0:
        raise ValueError("n_points must be > 0")
        
    if design_opts is None:
        design_opts = DesignOptions()
    
    # Build candidate set once (shared across all n values for consistency)
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
    cand = build_candidate(
        factors,
        candidate_points=candidate_points,
        seed=design_opts.random_state,
        constraint_func=design_opts.constraint_func,
        cat_cells_cap=design_opts.cat_cells_cap,
    )
    X_cand, _ = build_model_matrix(formula, cand)
    p = X_cand.shape[1]
    
    # Determine n range if not provided
    if n_range is None:
        # Heuristic: start from p+1, go up to where we expect >0.99 power
        min_n = p + 1
        # Rough estimate for max_n (this could be refined)
        if isinstance(power_cfg, PowerContrastConfig):
            # Conservative estimate based on effect size
            effect_magnitude = np.linalg.norm(power_cfg.delta) / power_cfg.sigma
            max_n = min(int(100 / (effect_magnitude ** 2)), 500)
        else:
            # R² mode
            f2 = power_cfg.r2_target / (1 - power_cfg.r2_target)
            max_n = min(int(50 / f2), 500)
        max_n = max(max_n, min_n + 20)
    else:
        min_n, max_n = n_range
        min_n = max(min_n, p + 1)
    
    # Generate n values (use geometric spacing for better resolution at small n)
    n_values = np.unique(np.geomspace(min_n, max_n, n_points).astype(int))
    
    results = []
    target_n = None
    
    for n in n_values:
        # Build I-optimal design at this n
        design_df = build_i_opt_design(
            cand=cand,
            formula=formula,
            n=int(n),
            criterion=design_opts.criterion,
            n_start=design_opts.starts,
            algo=design_opts.algo,
            max_iter=design_opts.max_iter,
            random_state=design_opts.random_state,
            jitter=design_opts.xtx_jitter,
        )

        # Get design matrix
        X, _ = build_model_matrix(formula, design_df)
        
        # Compute power
        if isinstance(power_cfg, PowerContrastConfig):
            power, lam = contrast_power(
                L=power_cfg.L,
                delta=power_cfg.delta,
                X=X,
                sigma=power_cfg.sigma,
                alpha=power_cfg.alpha,
                jitter=design_opts.xtx_jitter,
            )
        else:
            power, lam = global_r2_power(
                power_cfg.r2_target,
                X,
                alpha=power_cfg.alpha,
                lambda_mode=power_cfg.lambda_mode,
            )
        
        # Compute design metrics for additional insight
        metrics = compute_design_metrics(X, X_cand=X_cand)
        
        results.append({
            'n': int(n),
            'power': float(power),
            'lambda': float(lam),
            'd_efficiency': float(metrics['d_efficiency']),
            'i_criterion': float(metrics['i_criterion']),
            'condition_number': float(metrics['condition_number']),
        })
        
        # Track first n achieving target power
        if target_n is None and power >= power_cfg.power:
            target_n = int(n)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Optional plotting
    fig = None
    if plot:
        if plot_backend == "plotly":
            from .plot_backends import plotly_curve_by_n as _plotly_curve_by_n
            fig = _plotly_curve_by_n(df, power_cfg, target_n)
        elif _HAS_MATPLOTLIB:
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

            # Power curve
            ax1 = axes[0]
            ax1.plot(df['n'], df['power'], 'b-', linewidth=2, label='Power')
            ax1.axhline(y=power_cfg.power, color='r', linestyle='--',
                        label=f'Target ({power_cfg.power:.2f})')
            ax1.axhline(y=0.80, color='gray', linestyle=':', alpha=0.5)
            if target_n:
                ax1.axvline(x=target_n, color='g', linestyle='--', alpha=0.5,
                            label=f'n={target_n}')
            ax1.set_ylabel('Statistical Power')
            ax1.set_ylim([0, 1.05])
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Design quality metrics
            ax2 = axes[1]
            ax2_twin = ax2.twinx()

            line1 = ax2.plot(df['n'], df['i_criterion'], 'g-',
                             label='I-criterion (left)')
            line2 = ax2_twin.plot(df['n'], df['d_efficiency'], 'orange',
                                  label='D-efficiency (right)')

            ax2.set_xlabel('Sample Size (n)')
            ax2.set_ylabel('I-criterion (lower is better)', color='g')
            ax2_twin.set_ylabel('D-efficiency (higher is better)', color='orange')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2_twin.tick_params(axis='y', labelcolor='orange')
            ax2.grid(True, alpha=0.3)

            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='best')

            # --- Reviewer Feedback: Titles ---
            if isinstance(power_cfg, PowerContrastConfig):
                effect_desc = f"Effect Norm={np.linalg.norm(power_cfg.delta):.2f}, $\\sigma$={power_cfg.sigma}"
                title = f"Power vs. Sample Size (n) for Contrast Test\n({effect_desc}, $\\alpha$={power_cfg.alpha})"
            else:
                effect_desc = f"Target R²={power_cfg.r2_target}"
                title = f"Power vs. Sample Size (n) for Global F-Test\n({effect_desc}, $\\alpha$={power_cfg.alpha})"

            plt.suptitle(title)
            plt.tight_layout()
    
    return {
        'data': df,
        'figure': fig,
        'target_n': target_n,
    }


def power_curve_by_effect(
    formula: str,
    factors: dict,
    n: int,
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    effect_range: Optional[Tuple[float, float]] = None,
    effect_points: int = 30,
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
    figsize: Tuple[float, float] = (8, 5),
    plot_backend: Literal["matplotlib", "plotly"] = "matplotlib",
) -> Dict[str, Union[pd.DataFrame, Optional["Figure"]]]:
    """Generate power curve as a function of effect size.
    
    Fixes n and varies the effect size (delta for contrasts, R² for global).
    Uses a single I-optimal design computed at the specified n.
    
    Parameters
    ----------
    formula : str
        Patsy-style model formula.
    factors : dict
        Factor specifications.
    n : int
        Fixed sample size.
    power_cfg : PowerContrastConfig or PowerR2Config
        Base configuration (effect size will be varied).
    effect_range : tuple of (min_effect, max_effect), optional
        Range of effect sizes. If None, automatically determined.
        For contrasts: multiplier on base delta (0.5 to 2.0).
        For R²: actual R² values (0.01 to 0.5).
    effect_points : int, default 30
        Number of effect sizes to evaluate.
    design_opts : DesignOptions, optional
        Design generation options.
    plot : bool, default False
        If True and matplotlib available, return a figure object.
    figsize : tuple, default (8, 5)
        Figure size if plotting.
    
    Returns
    -------
    dict
        {
          'data': DataFrame with columns [effect_size, power, lambda],
          'figure': matplotlib Figure if plot=True and available, else None,
          'min_detectable_effect': float at 80% power
        }
    """
    # --- Reviewer Feedback: Validation ---
    if effect_points <= 0:
        raise ValueError("effect_points must be > 0")
        
    if design_opts is None:
        design_opts = DesignOptions()
    
    # Build the design once at fixed n
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
    cand = build_candidate(
        factors,
        candidate_points=candidate_points,
        seed=design_opts.random_state,
        constraint_func=design_opts.constraint_func,
        cat_cells_cap=design_opts.cat_cells_cap,
    )
    
    design_df = build_i_opt_design(
        cand=cand,
        formula=formula,
        n=n,
        criterion=design_opts.criterion,
        n_start=design_opts.starts,
        algo=design_opts.algo,
        max_iter=design_opts.max_iter,
        random_state=design_opts.random_state,
        jitter=design_opts.xtx_jitter,
    )


    # --- Reviewer Feedback: Caching ---
    # Model matrix X is built ONCE and reused in the loop, as suggested.
    X, _ = build_model_matrix(formula, design_df)
    
    results = []
    min_detectable = None
    
    if isinstance(power_cfg, PowerContrastConfig):
        # Vary delta magnitude
        if effect_range is None:
            effect_range = (0.1, 2.5)  # multipliers on base delta
        
        effect_multipliers = np.linspace(effect_range[0], effect_range[1], effect_points)
        base_delta = power_cfg.delta
        
        for mult in effect_multipliers:
            scaled_delta = base_delta * mult
            power, lam = contrast_power(
                L=power_cfg.L,
                delta=scaled_delta,
                X=X,
                sigma=power_cfg.sigma,
                alpha=power_cfg.alpha,
            )
            
            results.append({
                'effect_size': float(mult),
                'power': float(power),
                'lambda': float(lam),
                'actual_delta_norm': float(np.linalg.norm(scaled_delta)),
            })
            
            if min_detectable is None and power >= 0.80:
                min_detectable = float(mult)
    
    else:  # R² mode
        if effect_range is None:
            effect_range = (0.01, 0.5)
        
        r2_values = np.linspace(effect_range[0], effect_range[1], effect_points)
        
        for r2 in r2_values:
            power, lam = global_r2_power(
                r2,
                X,
                alpha=power_cfg.alpha,
                lambda_mode=power_cfg.lambda_mode,
            )
            
            results.append({
                'effect_size': float(r2),
                'power': float(power),
                'lambda': float(lam),
            })
            
            if min_detectable is None and power >= 0.80:
                min_detectable = float(r2)
    
    df = pd.DataFrame(results)
    
    # Optional plotting
    fig = None
    if plot:
        if plot_backend == "plotly":
            from .plot_backends import plotly_curve_by_effect as _plotly_curve_by_effect
            fig = _plotly_curve_by_effect(df, power_cfg, min_detectable, n)
        elif _HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(df['effect_size'], df['power'], 'b-', linewidth=2)
            ax.axhline(y=0.80, color='r', linestyle='--', label='80% Power')
            ax.axhline(y=power_cfg.power, color='g', linestyle='--',
                      label=f'Target ({power_cfg.power:.2f})')

            if min_detectable:
                ax.axvline(x=min_detectable, color='orange', linestyle=':',
                          label=f'MDE={min_detectable:.3f}')

            # --- Reviewer Feedback: Titles & Labels ---
            if isinstance(power_cfg, PowerContrastConfig):
                ax.set_xlabel('Effect Size Multiplier (on base norm)')
                title = f'Power vs. Effect Size at n={n}\n(Contrast Test, $\\sigma$={power_cfg.sigma}, $\\alpha$={power_cfg.alpha})'
                ax.set_title(title)
            else:
                ax.set_xlabel('R² Effect Size')
                title = f'Power vs. Effect Size at n={n}\n(Global F-Test, $\\alpha$={power_cfg.alpha})'
                ax.set_title(title)

            ax.set_ylabel('Statistical Power')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
    
    return {
        'data': df,
        'figure': fig,
        'min_detectable_effect': min_detectable,
    }


def power_surface_2d(
    formula: str,
    factors: dict,
    power_cfg: Union[PowerContrastConfig, PowerR2Config],
    param1: Literal['n', 'effect', 'sigma', 'alpha'],
    param1_range: Tuple[float, float],
    param2: Literal['n', 'effect', 'sigma', 'alpha'],
    param2_range: Tuple[float, float],
    grid_points: int = 20,
    design_opts: Optional[DesignOptions] = None,
    plot: bool = False,
    figsize: Tuple[float, float] = (10, 8),
    plot_backend: Literal["matplotlib", "plotly"] = "matplotlib",
) -> Dict[str, Union[pd.DataFrame, Optional["Figure"]]]:
    """Generate 2D power surface varying two parameters simultaneously.
    
    Creates a grid of power values for sensitivity analysis across two
    dimensions (e.g., n vs effect size, or effect vs sigma).
    
    Parameters
    ----------
    formula : str
        Patsy-style model formula.
    factors : dict
        Factor specifications.
    power_cfg : PowerContrastConfig or PowerR2Config
        Base configuration.
    param1, param2 : {'n', 'effect', 'sigma', 'alpha'}
        Parameters to vary.
    param1_range, param2_range : tuple
        Ranges for the parameters.
    grid_points : int, default 20
        Points along each axis (total evaluations = grid_points²).
    design_opts : DesignOptions, optional
        Design generation options.
    plot : bool, default False
        If True and matplotlib available, return a contour plot.
    figsize : tuple, default (10, 8)
        Figure size if plotting.
    
    Returns
    -------
    dict
        {
          'data': DataFrame with columns [param1, param2, power],
          'figure': matplotlib Figure with contour plot if requested,
          'power_grid': 2D numpy array of power values
        }
    
    Notes
    -----
    This is computationally expensive, especially if 'n' is varied,
    as it may require generating many I-optimal designs.
    """
    _VALID_PARAMS = {"n", "effect", "sigma", "alpha"}
    if grid_points <= 0:
        raise ValueError("grid_points must be > 0")
    if param1 not in _VALID_PARAMS:
        raise ValueError(f"param1 must be one of {_VALID_PARAMS}, got {param1!r}")
    if param2 not in _VALID_PARAMS:
        raise ValueError(f"param2 must be one of {_VALID_PARAMS}, got {param2!r}")
    if param1 == param2:
        raise ValueError(f"param1 and param2 must be different (both are {param1!r})")

    is_contrast = isinstance(power_cfg, PowerContrastConfig)
    if "sigma" in (param1, param2) and not is_contrast:
        raise ValueError(
            "'sigma' sweep is only valid for PowerContrastConfig. "
            "PowerR2Config uses R² directly and does not depend on sigma."
        )

    if design_opts is None:
        design_opts = DesignOptions()

    # Build candidate set once (shared across all grid evaluations)
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
    cand = build_candidate(
        factors,
        candidate_points=candidate_points,
        seed=design_opts.random_state,
        constraint_func=design_opts.constraint_func,
        cat_cells_cap=design_opts.cat_cells_cap,
    )
    X_cand, _ = build_model_matrix(formula, cand)
    p = X_cand.shape[1]

    # Build parameter grids
    def _make_axis(param: str, rng: tuple) -> np.ndarray:
        lo, hi = float(rng[0]), float(rng[1])
        if param == "n":
            # Integer grid, log-spaced for better resolution at small n
            return np.unique(np.geomspace(max(lo, p + 1), max(hi, p + 2), grid_points).astype(int))
        return np.linspace(lo, hi, grid_points)

    axis1 = _make_axis(param1, param1_range)
    axis2 = _make_axis(param2, param2_range)

    # Base values (used when the parameter is held fixed)
    base_alpha = float(power_cfg.alpha)
    base_sigma = float(power_cfg.sigma) if is_contrast else 1.0
    # For 'effect': contrast → multiplier on delta (1.0 = nominal);
    #               R²       → actual r2_target value
    base_effect = 1.0 if is_contrast else float(power_cfg.r2_target)

    # Cache of design matrices keyed by n to avoid redundant DOE calls
    _x_cache: dict = {}

    def _get_X(n_val: int) -> np.ndarray:
        if n_val not in _x_cache:
            _, sel_idx, _ = build_i_opt_design_with_idx(
                cand=cand,
                formula=formula,
                n=n_val,
                criterion=design_opts.criterion,
                n_start=design_opts.starts,
                algo=design_opts.algo,
                max_iter=design_opts.max_iter,
                random_state=design_opts.random_state,
                workers=design_opts.workers,
                parallel_seed_stride=design_opts.parallel_seed_stride,
                jitter=design_opts.xtx_jitter,
            )
            _x_cache[n_val] = X_cand[sel_idx, :]
        return _x_cache[n_val]

    # For purely analytical sweeps (neither param is 'n'), build one fixed design
    fixed_n: Optional[int] = None
    if "n" not in (param1, param2):
        # Use a reasonable representative n: midpoint of [p+1, max_n]
        fixed_n = max(p + 1, min(power_cfg.max_n, (p + 1 + power_cfg.max_n) // 2))

    def _compute_power(v1: float, v2: float) -> Tuple[float, float]:
        """Return (power, lam) for a grid cell (v1, v2)."""
        n_val   = int(v1) if param1 == "n" else (int(v2) if param2 == "n" else fixed_n)
        alpha   = v1 if param1 == "alpha"  else (v2 if param2 == "alpha"  else base_alpha)
        sigma   = v1 if param1 == "sigma"  else (v2 if param2 == "sigma"  else base_sigma)
        effect  = v1 if param1 == "effect" else (v2 if param2 == "effect" else base_effect)

        X = _get_X(n_val)

        if is_contrast:
            # effect is a scale multiplier on the base delta vector
            delta_scaled = power_cfg.delta * float(effect)
            pwr, lam = contrast_power(
                L=power_cfg.L, delta=delta_scaled, X=X,
                sigma=float(sigma), alpha=float(alpha),
                jitter=design_opts.xtx_jitter,
            )
        else:
            # effect IS the r2_target
            pwr, lam = global_r2_power(
                r2_target=float(effect), X=X,
                alpha=float(alpha), lambda_mode=power_cfg.lambda_mode,
            )
        return float(pwr), float(lam)

    # Evaluate the grid
    rows = []
    power_grid = np.full((len(axis1), len(axis2)), np.nan)

    for i, v1 in enumerate(axis1):
        for j, v2 in enumerate(axis2):
            pwr, lam = _compute_power(v1, v2)
            power_grid[i, j] = pwr
            rows.append({param1: v1, param2: v2, "power": pwr, "noncentrality_lambda": lam})

    df = pd.DataFrame(rows)

    # Optional contour plot
    fig = None
    if plot:
        if plot_backend == "plotly":
            from .plot_backends import plotly_surface_2d as _plotly_surface_2d
            fig = _plotly_surface_2d(power_grid, axis1, axis2, power_cfg, param1, param2)
        elif _HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=figsize)
            G2, G1 = np.meshgrid(axis2, axis1)
            levels = np.linspace(0.0, 1.0, 21)
            cs = ax.contourf(G2, G1, power_grid, levels=levels, cmap="viridis")
            plt.colorbar(cs, ax=ax, label="Power")
            # Mark the target-power contour in white
            try:
                ax.contour(G2, G1, power_grid, levels=[power_cfg.power],
                           colors="white", linewidths=2, linestyles="--")
            except Exception:
                pass
            param1_label = "n" if param1 == "n" else param1
            param2_label = "n" if param2 == "n" else param2
            ax.set_xlabel(param2_label)
            ax.set_ylabel(param1_label)
            ax.set_title(
                f"Power Surface: {param1_label} × {param2_label}"
                f"  (target = {power_cfg.power:.2f}, white contour)"
            )
            plt.tight_layout()

    return {
        "data": df,
        "power_grid": power_grid,
        "param1_values": axis1,
        "param2_values": axis2,
        "figure": fig,
    }


__all__ = [
    'power_curve_by_n',
    'power_curve_by_effect', 
    'power_surface_2d',
]
