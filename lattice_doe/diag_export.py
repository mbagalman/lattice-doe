"""Diagnostic file export (CSV, PNG, HTML)."""
from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from .diag_metrics import compute_design_metrics
from .diag_plots import create_diagnostic_plots

__all__ = ["export_diagnostics"]


def export_diagnostics(
    X: np.ndarray,
    design_df: pd.DataFrame,
    output_path: Union[str, Path],
    feature_names: Optional[List[str]] = None,
    formats: Optional[List[str]] = None,
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
    if formats is None:
        formats = ["html", "pdf"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(
            f"No write permissions for output directory: {output_path.parent}"
        )

    # Compute all metrics
    metrics = compute_design_metrics(X, include_vif=True, X_cand=None, feature_names=feature_names)

    outputs = {}

    # Export data tables if requested
    if include_data:
        if "csv" in formats or "xlsx" in formats:
            summary_data = {
                "Metric": [
                    "Condition Number",
                    "D-Efficiency",
                    "Mean Leverage",
                    "Max Leverage",
                    "I-Criterion",
                    "I-Criterion Cand. Size",
                ],
                "Value": [
                    metrics["condition_number"],
                    metrics["d_efficiency"],
                    metrics["leverage_mean"],
                    metrics["leverage_max"],
                    metrics.get("i_criterion", np.nan),
                    metrics.get("i_criterion_n_cand", np.nan),
                ],
            }
            summary_df = pd.DataFrame(summary_data)

            lev_df = pd.DataFrame({
                "run": range(1, len(metrics["leverages"]) + 1),
                "leverage": metrics["leverages"],
            })

            if "csv" in formats:
                summary_path = output_path.with_suffix(".summary.csv")
                summary_df.to_csv(summary_path, index=False)
                outputs["summary_csv"] = summary_path

                if "vif_df" in metrics:
                    vif_path = output_path.with_suffix(".vif.csv")
                    metrics["vif_df"].to_csv(vif_path, index=False)
                    outputs["vif_csv"] = vif_path

                lev_path = output_path.with_suffix(".leverages.csv")
                lev_df.to_csv(lev_path, index=False)
                outputs["leverages_csv"] = lev_path

            if "xlsx" in formats:
                xlsx_path = output_path.with_suffix(".diagnostics.xlsx")
                with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    if "vif_df" in metrics:
                        metrics["vif_df"].to_excel(writer, sheet_name="VIF", index=False)
                    lev_df.to_excel(writer, sheet_name="Leverages", index=False)
                outputs["xlsx"] = xlsx_path

    # Generate plots
    if _HAS_MATPLOTLIB and any(fmt in ["html", "pdf", "png"] for fmt in formats):
        fig = create_diagnostic_plots(X, design_df, feature_names)

        if fig is not None:
            if "pdf" in formats:
                pdf_path = output_path.with_suffix(".diagnostics.pdf")
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")
                outputs["pdf"] = pdf_path

            if "png" in formats:
                png_path = output_path.with_suffix(".diagnostics.png")
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                outputs["png"] = png_path

            if "html" in formats:
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()

                n, p = X.shape
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
                            <td>D-Efficiency (0–1)</td>
                            <td class="metric-value">{metrics['d_efficiency']:.4f}</td>
                            <td>{'Good (≥0.8)' if metrics['d_efficiency'] >= 0.8 else 'Moderate (<0.8)'}</td>
                        </tr>
                        <tr>
                            <td>Mean Leverage</td>
                            <td class="metric-value">{metrics['leverage_mean']:.4f}</td>
                            <td>Expected: {p/n:.4f}</td>
                        </tr>
                        <tr>
                            <td>Max Leverage</td>
                            <td class="metric-value">{metrics['leverage_max']:.4f}</td>
                            <td class="{'error' if metrics['leverage_max'] > 3*p/n else 'warning' if metrics['leverage_max'] > 2*p/n else ''}">
                                {'High' if metrics['leverage_max'] > 3*p/n else 'Moderate' if metrics['leverage_max'] > 2*p/n else 'Good'}
                            </td>
                        </tr>
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

                html_path = output_path.with_suffix(".diagnostics.html")
                html_path.write_text(html_content)
                outputs["html"] = html_path

            plt.close(fig)

    return outputs
