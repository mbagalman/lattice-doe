"""Matplotlib diagnostic figures."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from .diag_metrics import _compute_vif, _has_intercept, compute_leverages

__all__ = ["create_diagnostic_plots"]


def create_diagnostic_plots(
    X: np.ndarray,
    design_df: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 10),
) -> "Optional[Figure]":
    """Create comprehensive diagnostic plots for design assessment.

    Generates a multi-panel figure with:
    - VIF bar chart (if p > 1)
    - Leverage plot with threshold lines
    - Correlation heatmap
    - Design point distribution (if design_df provided)

    Parameters
    ----------
    X : ndarray (n x p)
        Design matrix.
    design_df : DataFrame, optional
        Original design DataFrame for scatter plots.
    feature_names : list of str, optional
        Names for features.
    figsize : tuple, default (12, 10)
        Figure size.

    Returns
    -------
    Figure or None
        matplotlib Figure object if plotting available, else None.
    """
    if not _HAS_MATPLOTLIB:
        warnings.warn("matplotlib not available; cannot create diagnostic plots")
        return None

    n, p = X.shape
    if p == 0:
        warnings.warn("Design matrix has 0 columns; cannot create diagnostic plots")
        return None

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(p)]

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    # Determine grid layout based on what we can show
    if p > 1:  # Can show VIF and correlation
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    else:  # Only leverage meaningful
        gs = fig.add_gridspec(2, 1, hspace=0.3)

    # 1. VIF Bar Chart (if multiple predictors)
    if p > 1:
        ax_vif = fig.add_subplot(gs[0, 0])
        vif_df = _compute_vif(X, feature_names)
        if len(vif_df) > 0:
            colors = ['red' if v > 10 else 'orange' if v > 5 else 'green'
                      for v in vif_df['vif']]
            ax_vif.bar(range(len(vif_df)), vif_df['vif'], color=colors)
            ax_vif.set_xticks(range(len(vif_df)))
            ax_vif.set_xticklabels(vif_df['feature'], rotation=45, ha='right')
            ax_vif.axhline(y=5, color='orange', linestyle='--', alpha=0.5,
                           label='Moderate (VIF=5)')
            ax_vif.axhline(y=10, color='red', linestyle='--', alpha=0.5,
                           label='High (VIF=10)')
            ax_vif.set_ylabel('VIF')
            ax_vif.set_title('Variance Inflation Factors')
            ax_vif.legend(loc='best', fontsize='small')
            ax_vif.grid(True, alpha=0.3)

    # 2. Leverage Plot
    if p > 1:
        ax_lev = fig.add_subplot(gs[0, 1])
    else:
        ax_lev = fig.add_subplot(gs[0, 0])

    leverages = compute_leverages(X)
    run_numbers = np.arange(1, n + 1)

    threshold_high = 3 * p / n
    threshold_moderate = 2 * p / n
    colors = ['red' if lev > threshold_high else 'orange' if lev > threshold_moderate else 'blue'
              for lev in leverages]

    ax_lev.scatter(run_numbers, leverages, c=colors, alpha=0.6, s=30)
    ax_lev.axhline(y=p / n, color='green', linestyle='-', alpha=0.5,
                   label=f'Mean (p/n={p/n:.3f})')
    ax_lev.axhline(y=threshold_moderate, color='orange', linestyle='--', alpha=0.5,
                   label=f'Moderate (2p/n={threshold_moderate:.3f})')
    ax_lev.axhline(y=threshold_high, color='red', linestyle='--', alpha=0.5,
                   label=f'High (3p/n={threshold_high:.3f})')
    ax_lev.set_xlabel('Run Number')
    ax_lev.set_ylabel('Leverage')
    ax_lev.set_title('Leverage Values (Design Point Influence)')
    ax_lev.legend(loc='best', fontsize='small')
    ax_lev.grid(True, alpha=0.3)

    # 3. Correlation Heatmap (if multiple predictors)
    if p > 1:
        ax_corr = fig.add_subplot(gs[1, :])

        non_intercept_idx = [i for i in range(p) if not _has_intercept(X[:, i])]
        if len(non_intercept_idx) > 1:
            X_for_corr = X[:, non_intercept_idx]
            feature_names_corr = [feature_names[i] for i in non_intercept_idx]

            with np.errstate(divide='ignore', invalid='ignore'):
                X_std = X_for_corr - X_for_corr.mean(axis=0)
                std_devs = X_for_corr.std(axis=0, ddof=1)
                std_devs[std_devs == 0] = 1.0
                X_std = X_std / std_devs
                corr_matrix = np.corrcoef(X_std.T)

            if np.any(np.isnan(corr_matrix)):
                warnings.warn(
                    "NaN values encountered in correlation matrix; plotting may be affected."
                )
                corr_matrix = np.nan_to_num(corr_matrix)

            im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax_corr.set_xticks(range(len(feature_names_corr)))
            ax_corr.set_yticks(range(len(feature_names_corr)))
            ax_corr.set_xticklabels(feature_names_corr, rotation=45, ha='right')
            ax_corr.set_yticklabels(feature_names_corr)
            ax_corr.set_title('Correlation Matrix (Non-Intercept Terms)')

            for i in range(len(feature_names_corr)):
                for j in range(len(feature_names_corr)):
                    val = corr_matrix[i, j]
                    ax_corr.text(j, i, f'{val:.2f}',
                                 ha="center", va="center",
                                 color="white" if abs(val) > 0.5 else "black",
                                 fontsize=8)

            plt.colorbar(im, ax=ax_corr, label='Correlation')

    # 4. Design Space Coverage (if design_df provided and has 1-2 continuous factors)
    if design_df is not None:
        cont_cols = []
        for col in design_df.columns:
            if design_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if len(design_df[col].unique()) > 10:
                    cont_cols.append(col)

        if len(cont_cols) >= 2:
            ax_space = fig.add_subplot(gs[2, :] if p > 1 else gs[1, :])
            scatter = ax_space.scatter(
                design_df[cont_cols[0]],
                design_df[cont_cols[1]],
                c=leverages,
                s=50,
                cmap='viridis',
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5,
            )
            ax_space.set_xlabel(cont_cols[0])
            ax_space.set_ylabel(cont_cols[1])
            ax_space.set_title('Design Space Coverage (colored by leverage)')
            plt.colorbar(scatter, ax=ax_space, label='Leverage')
            ax_space.grid(True, alpha=0.3)

        elif len(cont_cols) == 1:
            ax_space = fig.add_subplot(gs[2, :] if p > 1 else gs[1, :])
            y_jitter = np.random.normal(0, 0.02, size=len(design_df))
            scatter = ax_space.scatter(
                design_df[cont_cols[0]],
                y_jitter,
                c=leverages,
                s=50,
                cmap='viridis',
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5,
            )
            ax_space.set_xlabel(cont_cols[0])
            ax_space.set_ylabel('(jittered for visibility)')
            ax_space.set_title('Design Points Distribution (colored by leverage)')
            ax_space.set_ylim([-0.1, 0.1])
            plt.colorbar(scatter, ax=ax_space, label='Leverage')
            ax_space.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Design Diagnostics Report', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig
