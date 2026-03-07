# diagnostics.py (enhanced version)
# License: MIT
"""
Design diagnostics for linear-model DOE with visualization support
------------------------------------------------------------------

Lightweight metrics to summarize numerical quality of a given design matrix `X`,
plus optional visualization of key diagnostic plots for design assessment.

Typical ranges (rules of thumb)
-------------------------------
- condition_number: < ~30 usually OK; > ~1000 often problematic
- d_efficiency    : closer to 1.0 is better (scale-normalized); compare across designs
- leverage_mean   : equals rank(X)/n (≈ p/n if full rank); informational only
- i_criterion     : average prediction variance over the *candidate region*;
                    lower is better (requires X_cand)
- VIF             : < 5 good, 5-10 moderate collinearity, > 10 problematic

Public functions
----------------
- compute_design_metrics(X, include_vif=False, X_cand=None):
    condition #, D-efficiency, leverage_mean, optional I-criterion over candidates,
    and optional VIFs.
- create_diagnostic_plots(X, design_df=None, feature_names=None):
    Generate matplotlib figures for design diagnostics.
- export_diagnostics(X, design_df, output_path, formats=['html', 'pdf']):
    Export comprehensive diagnostic report with plots.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, List, Union, Tuple
from pathlib import Path
import os  # ADDED: For write permission checks
import numpy as np
import pandas as pd
import warnings

# Optional plotting/export support
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _xtx(X: np.ndarray) -> np.ndarray:
    return X.T @ X


def _pinv(M: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(M)


def _has_intercept(col: np.ndarray, atol: float = 1e-12) -> bool:
    """Heuristic: column is (near) constant ones."""
    if np.std(col) < atol and np.allclose(col.mean(), 1.0, atol=1e-8):
        return True
    return False


def _compute_vif(
    X: np.ndarray, 
    feature_names: Optional[List[str]] = None,
    *, 
    detect_intercept: bool = True, 
    jitter: float = 1e-12
) -> pd.DataFrame:
    """Compute variance inflation factors (VIF) for columns of X.
    
    Returns DataFrame with feature names and VIF values for better
    interpretability.
    """
    n, p = X.shape
    
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(p)]
    
    # Identify non-intercept columns
    keep_idx = []
    keep_names = []
    for j in range(p):
        if detect_intercept and _has_intercept(X[:, j]):
            continue
        keep_idx.append(j)
        keep_names.append(feature_names[j])
    
    if not keep_idx:
        return pd.DataFrame(columns=['feature', 'vif'])
    
    # Compute VIFs for non-intercept columns
    Z = X[:, keep_idx].astype(float)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0, ddof=1)
    
    # Handle zero variance columns (perfectly constant)
    zero_sd_mask = (sd == 0)
    if np.any(zero_sd_mask):
        # A non-intercept column has zero variance.
        # This will lead to division by zero. Set std to 1 to avoid.
        sd[zero_sd_mask] = 1.0
    
    Z = (Z - mu) / sd
    
    # ADDED: Handle VIF calculation failures
    try:
        R = (Z.T @ Z) / max(n - 1, 1)
        R = R + jitter * np.eye(R.shape[0])
        Rinv = np.linalg.pinv(R)
        vif_values = np.diag(Rinv)
        
        # Replace infinities (from perfect collinearity) with a large finite number
        vif_values[np.isinf(vif_values)] = 1e12 
        
    except np.linalg.LinAlgError:
        # Fallback in case pinv fails (unlikely, but safe)
        warnings.warn(
            "VIF calculation failed due to a linear algebra error. "
            f"Returning NaN for {len(keep_names)} features.",
            RuntimeWarning
        )
        vif_values = np.full(len(keep_names), np.nan)
    
    # Return as DataFrame for clarity
    return pd.DataFrame({
        'feature': keep_names,
        'vif': vif_values
    })


def compute_leverages(X: np.ndarray) -> np.ndarray:
    """Compute leverage values (diagonal of hat matrix).
    
    Leverage indicates influence of each design point on predictions.
    High leverage points (> 2p/n) may be overly influential.
    
    Parameters
    ----------
    X : ndarray (n x p)
        Design matrix.
    
    Returns
    -------
    ndarray (n,)
        Leverage value for each design point.
    """
    XtX_inv = _pinv(_xtx(X))
    H = X @ XtX_inv @ X.T
    return np.diag(H)


def compute_design_metrics(
    X: np.ndarray,
    *,
    include_vif: bool = False,
    X_cand: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute core design diagnostics from a model matrix X.
    
    Parameters
    ----------
    X : ndarray (n x p)
        Design (model) matrix.
    include_vif : bool, default False
        If True, compute VIFs as a DataFrame.
    X_cand : ndarray (N_cand x p), optional
        Candidate-region model matrix for I-criterion.
    feature_names : list of str, optional
        Names for model matrix columns (for VIF reporting).
    
    Returns
    -------
    dict
        {
          'condition_number' : float,
          'd_efficiency'     : float,
          'leverage_mean'    : float,
          'leverage_max'     : float,
          'i_criterion'      : float (if X_cand provided),
          'i_criterion_n_cand': int (if X_cand provided),
          'vif_df'           : DataFrame (if include_vif=True),
          'leverages'        : ndarray (always included for plotting)
        }
    """
    n, p = X.shape
    if p == 0:
        return {
            "condition_number": np.nan,
            "d_efficiency": np.nan,
            "leverage_mean": np.nan,
            "leverage_max": np.nan,
            "leverages": np.array([]),
        }
        
    XtX = _xtx(X)
    cond = float(np.linalg.cond(XtX))
    
    # D-efficiency
    # ADDED: Use slogdet for numerical stability and protection
    sign, logdet = np.linalg.slogdet(XtX)
    if sign <= 0:
        d_eff = 0.0  # Singular matrix, D-efficiency is 0
    else:
        # Standard D-efficiency = (det(X'X)^(1/p)) / n
        # log(D-eff) = (1/p) * logdet - log(n)
        # Note: The p==0 case is handled by the guard at the function start.
        log_d_eff = (1.0 / p) * logdet - np.log(n)
        d_eff = float(np.exp(log_d_eff))
    
    # Leverage statistics
    try:
        leverages = compute_leverages(X)
        leverage_mean = float(np.mean(leverages))
        leverage_max = float(np.max(leverages))
    except np.linalg.LinAlgError:
        warnings.warn("Leverage calculation failed due to singular matrix.")
        leverages = np.full(n, np.nan)
        leverage_mean = np.nan
        leverage_max = np.nan
    
    out: Dict[str, Any] = {
        "condition_number": cond,
        "d_efficiency": d_eff,
        "leverage_mean": leverage_mean,
        "leverage_max": leverage_max,
        "leverages": leverages,  # Always include for potential plotting
    }
    
    # I-criterion over candidate region
    if X_cand is not None and X_cand.size > 0:
        n_cand = X_cand.shape[0]  # TWEAK: Store candidate set size
        try:
            XtX_inv = _pinv(XtX)
            Mcand = X_cand.T @ X_cand
            out["i_criterion"] = float(np.trace(XtX_inv @ Mcand) / n_cand)
            out["i_criterion_n_cand"] = n_cand  # TWEAK: Add to metrics for reproducibility
        except np.linalg.LinAlgError:
            out["i_criterion"] = np.nan
            out["i_criterion_n_cand"] = n_cand  # TWEAK: Store size even on failure
    
    if include_vif:
        out["vif_df"] = _compute_vif(X, feature_names)
    
    return out


def create_diagnostic_plots(
    X: np.ndarray,
    design_df: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 10),
) -> Optional[Figure]:
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
    
    plot_idx = 0
    
    # 1. VIF Bar Chart (if multiple predictors)
    if p > 1:
        ax_vif = fig.add_subplot(gs[0, 0])
        vif_df = _compute_vif(X, feature_names)
        if len(vif_df) > 0:
            colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' 
                     for v in vif_df['vif']]
            bars = ax_vif.bar(range(len(vif_df)), vif_df['vif'], color=colors)
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
    
    # Color by leverage magnitude
    threshold_high = 3 * p / n
    threshold_moderate = 2 * p / n
    colors = ['red' if lev > threshold_high else 'orange' if lev > threshold_moderate else 'blue'
             for lev in leverages]
    
    ax_lev.scatter(run_numbers, leverages, c=colors, alpha=0.6, s=30)
    ax_lev.axhline(y=p/n, color='green', linestyle='-', alpha=0.5,
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
        
        # Compute correlation matrix (excluding intercept if detected)
        non_intercept_idx = [i for i in range(p) 
                           if not _has_intercept(X[:, i])]
        if len(non_intercept_idx) > 1:
            X_for_corr = X[:, non_intercept_idx]
            feature_names_corr = [feature_names[i] for i in non_intercept_idx]
            
            # Standardize and compute correlation
            with np.errstate(divide='ignore', invalid='ignore'):
                X_std = (X_for_corr - X_for_corr.mean(axis=0))
                std_devs = X_for_corr.std(axis=0, ddof=1)
                std_devs[std_devs == 0] = 1.0 # Avoid division by zero
                X_std = X_std / std_devs
                
                corr_matrix = np.corrcoef(X_std.T)
            
            if np.any(np.isnan(corr_matrix)):
                warnings.warn("NaN values encountered in correlation matrix; plotting may be affected.")
                corr_matrix = np.nan_to_num(corr_matrix)

            # Plot heatmap
            im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax_corr.set_xticks(range(len(feature_names_corr)))
            ax_corr.set_yticks(range(len(feature_names_corr)))
            ax_corr.set_xticklabels(feature_names_corr, rotation=45, ha='right')
            ax_corr.set_yticklabels(feature_names_corr)
            ax_corr.set_title('Correlation Matrix (Non-Intercept Terms)')
            
            # Add correlation values
            for i in range(len(feature_names_corr)):
                for j in range(len(feature_names_corr)):
                    val = corr_matrix[i, j]
                    text = ax_corr.text(j, i, f'{val:.2f}',
                                       ha="center", va="center",
                                       color="white" if abs(val) > 0.5 else "black",
                                       fontsize=8)
            
            plt.colorbar(im, ax=ax_corr, label='Correlation')
    
    # 4. Design Space Coverage (if design_df provided and has 1-2 continuous factors)
    if design_df is not None:
        # Find continuous factors
        cont_cols = []
        for col in design_df.columns:
            if design_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if len(design_df[col].unique()) > 10:  # Likely continuous
                    cont_cols.append(col)
        
        if len(cont_cols) >= 2:
            # 2D scatter plot of first two continuous factors
            ax_space = fig.add_subplot(gs[2, :] if p > 1 else gs[1, :])
            
            # Count replicates
            design_counts = design_df.groupby(list(design_df.columns)).size().reset_index(name='count')
            
            scatter = ax_space.scatter(design_df[cont_cols[0]], 
                                      design_df[cont_cols[1]],
                                      c=leverages, 
                                      s=50,
                                      cmap='viridis',
                                      alpha=0.6,
                                      edgecolors='black',
                                      linewidth=0.5)
            ax_space.set_xlabel(cont_cols[0])
            ax_space.set_ylabel(cont_cols[1])
            ax_space.set_title('Design Space Coverage (colored by leverage)')
            plt.colorbar(scatter, ax=ax_space, label='Leverage')
            ax_space.grid(True, alpha=0.3)
        
        elif len(cont_cols) == 1:
            # 1D strip plot
            ax_space = fig.add_subplot(gs[2, :] if p > 1 else gs[1, :])
            
            y_jitter = np.random.normal(0, 0.02, size=len(design_df))
            scatter = ax_space.scatter(design_df[cont_cols[0]], 
                                      y_jitter,
                                      c=leverages,
                                      s=50,
                                      cmap='viridis',
                                      alpha=0.6,
                                      edgecolors='black',
                                      linewidth=0.5)
            ax_space.set_xlabel(cont_cols[0])
            ax_space.set_ylabel('(jittered for visibility)')
            ax_space.set_title('Design Points Distribution (colored by leverage)')
            ax_space.set_ylim([-0.1, 0.1])
            plt.colorbar(scatter, ax=ax_space, label='Leverage')
            ax_space.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Design Diagnostics Report', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def export_diagnostics(
    X: np.ndarray,
    design_df: pd.DataFrame,
    output_path: Union[str, Path],
    feature_names: Optional[List[str]] = None,
    formats: List[str] = ['html', 'pdf'],
    include_data: bool = True,
) -> Dict[str, Path]:
    """Export comprehensive diagnostic report with plots and data.
    
    Parameters
    ----------
    X : ndarray (n x p)
        Design matrix.
    design_df : DataFrame
        Original design DataFrame.
    output_path : str or Path
        Base path for output files (without extension).
    feature_names : list of str, optional
        Names for model matrix columns.
    formats : list of str, default ['html', 'pdf']
        Output formats. Options: 'html', 'pdf', 'png', 'csv', 'xlsx'.
    include_data : bool, default True
        Whether to export data tables alongside plots.
    
    Returns
    -------
    dict
        Mapping of format to output Path.
    """
    output_path = Path(output_path)
    # VERIFIED: This line is already here, as requested
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ADDED: Validate write permissions
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(
            f"No write permissions for output directory: {output_path.parent}"
        )

    # Compute all metrics
    metrics = compute_design_metrics(X, include_vif=True, X_cand=None, feature_names=feature_names) # Example, X_cand would be passed if available
    
    outputs = {}
    
    # Export data tables if requested
    if include_data:
        if 'csv' in formats or 'xlsx' in formats:
            # Create summary DataFrame
            summary_data = {
                'Metric': ['Condition Number', 'D-Efficiency', 'Mean Leverage', 
                          'Max Leverage', 'I-Criterion', 'I-Criterion Cand. Size'],
                'Value': [
                    metrics['condition_number'],
                    metrics['d_efficiency'],
                    metrics['leverage_mean'],
                    metrics['leverage_max'],
                    metrics.get('i_criterion', np.nan),
                    metrics.get('i_criterion_n_cand', np.nan) # TWEAK: Added for export
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            
            lev_df = pd.DataFrame({
                'run': range(1, len(metrics['leverages']) + 1),
                'leverage': metrics['leverages']
            })

            if 'csv' in formats:
                # Save summary
                summary_path = output_path.with_suffix('.summary.csv')
                summary_df.to_csv(summary_path, index=False)
                outputs['summary_csv'] = summary_path
                
                # Save VIF data
                if 'vif_df' in metrics:
                    vif_path = output_path.with_suffix('.vif.csv')
                    metrics['vif_df'].to_csv(vif_path, index=False)
                    outputs['vif_csv'] = vif_path
                
                # Save leverages
                lev_path = output_path.with_suffix('.leverages.csv')
                lev_df.to_csv(lev_path, index=False)
                outputs['leverages_csv'] = lev_path
            
            if 'xlsx' in formats:
                xlsx_path = output_path.with_suffix('.diagnostics.xlsx')
                with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    if 'vif_df' in metrics:
                        metrics['vif_df'].to_excel(writer, sheet_name='VIF', index=False)
                    lev_df.to_excel(writer, sheet_name='Leverages', index=False)
                outputs['xlsx'] = xlsx_path
    
    # Generate plots
    if _HAS_MATPLOTLIB and any(fmt in ['html', 'pdf', 'png'] for fmt in formats):
        fig = create_diagnostic_plots(X, design_df, feature_names)
        
        if fig is not None:
            if 'pdf' in formats:
                pdf_path = output_path.with_suffix('.diagnostics.pdf')
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches='tight')
                outputs['pdf'] = pdf_path
            
            if 'png' in formats:
                png_path = output_path.with_suffix('.diagnostics.png')
                fig.savefig(png_path, dpi=150, bbox_inches='tight')
                outputs['png'] = png_path
            
            if 'html' in formats:
                # Create HTML report with embedded plots and tables
                import base64
                from io import BytesIO
                
                # Save figure to base64 for embedding
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                
                # Build HTML
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Design Diagnostics Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        h2 {{ color: #666; }}
                        table {{ border-collapse: collapse; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        .metric-value {{ font-weight: bold; }}
                        .warning {{ color: orange; }}
                        .error {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>Design Diagnostics Report</h1>
                    
                    <h2>Summary Metrics</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                        <tr>
                            <td>Condition Number</td>
                            <td class="metric-value">{metrics['condition_number']:.2f}</td>
                            <td class="{'error' if metrics['condition_number'] > 1000 else 'warning' if metrics['condition_number'] > 30 else ''}">
                                {'Poor' if metrics['condition_number'] > 1000 else 'Moderate' if metrics['condition_number'] > 30 else 'Good'}
                            </td>
                        </tr>
                        <tr>
                            <td>D-Efficiency</td>
                            <td class="metric-value">{metrics['d_efficiency']:.4f}</td>
                            <td>{'Good' if metrics['d_efficiency'] > 0.8 else 'Moderate'}</td>
                        </tr>
                        <tr>
                            <td>Mean Leverage</td>
                            <td class="metric-value">{metrics['leverage_mean']:.4f}</td>
                            <td>Expected: {X.shape[1]/X.shape[0]:.4f}</td>
                        </tr>
                        <tr>
                            <td>Max Leverage</td>
                            <td class="metric-value">{metrics['leverage_max']:.4f}</td>
                            <td class="{'error' if metrics['leverage_max'] > 3*X.shape[1]/X.shape[0] else 'warning' if metrics['leverage_max'] > 2*X.shape[1]/X.shape[0] else ''}">
                                {'High' if metrics['leverage_max'] > 3*X.shape[1]/X.shape[0] else 'Moderate' if metrics['leverage_max'] > 2*X.shape[1]/X.shape[0] else 'Good'}
                            </td>
                        </tr>
                        <!-- TWEAK: Added I-Criterion to HTML report -->
                        <tr>
                            <td>I-Criterion</td>
                            <td class="metric-value">{metrics.get('i_criterion', 'N/A')}</td>
                            <td>(Lower is better)</td>
                        </tr>
                        <tr>
                            <td>I-Criterion Cand. Size</td>
                            <td class="metric-value">{metrics.get('i_criterion_n_cand', 'N/A')}</td>
                            <td>(For reference)</td>
                        </tr>
                    </table>
                    
                    <h2>Diagnostic Plots</h2>
                    <img src="data:image/png;base64,{img_base64}" alt="Diagnostic Plots" style="max-width:100%;">
                    
                    <h2>Interpretation Guide</h2>
                    <ul>
                        <li><b>VIF &lt; 5:</b> Low multicollinearity (good)</li>
                        <li><b>VIF 5-10:</b> Moderate multicollinearity (acceptable)</li>
                        <li><b>VIF &gt; 10:</b> High multicollinearity (problematic)</li>
                        <li><b>Leverage &gt; 2p/n:</b> Potentially influential point</li>
                        <li><b>Leverage &gt; 3p/n:</b> Highly influential point</li>
                        <li><b>Condition Number &lt; 30:</b> Well-conditioned</li>
                        <li><b>Condition Number &gt; 1000:</b> Ill-conditioned</li>
                    </ul>
                </body>
                </html>
                """
                
                html_path = output_path.with_suffix('.diagnostics.html')
                html_path.write_text(html_content)
                outputs['html'] = html_path
            
            # This is the "headless safety" check
            plt.close(fig)
    
    return outputs


__all__ = [
    "compute_design_metrics",
    "compute_leverages", 
    "create_diagnostic_plots",
    "export_diagnostics",
]
