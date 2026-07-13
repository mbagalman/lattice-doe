# report.py
# License: MIT
"""
lattice_doe.report
========================
Generate self-contained HTML (and optionally PDF) summary reports for
powered optimal designs.

Public API
----------
generate_report(result, formula, factors, power_cfg, output_path, ...)
    Render and write a shareable report.  Returns the Path written.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["generate_report"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_jinja_env():
    """Return a Jinja2 Environment that loads templates from this package."""
    try:
        from jinja2 import Environment, PackageLoader, select_autoescape
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "HTML report generation requires jinja2. "
            'Install it with: pip install "lattice-doe[report]"'
        ) from exc

    return Environment(
        loader=PackageLoader("lattice_doe", "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _fig_to_base64(fig) -> str | None:
    """Convert a Plotly or Matplotlib figure to a base64-encoded PNG string.

    Returns None if conversion fails or the figure type is unrecognised.
    """
    # --- Plotly (requires kaleido for to_image) ---
    try:
        if hasattr(fig, "to_image"):
            png_bytes = fig.to_image(format="png", width=800, height=350)
            return base64.b64encode(png_bytes).decode("ascii")
    except Exception:
        pass

    # --- Matplotlib ---
    try:
        if hasattr(fig, "savefig"):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# B1 — Config summary context
# ---------------------------------------------------------------------------


def _build_config_ctx(formula: str, factors: dict, power_cfg: Any) -> dict:
    """Build the template context dict for the Config Summary section (B1).

    Handles both PowerContrastConfig and PowerR2Config.
    """
    from .config import PowerContrastConfig, PowerR2Config

    # Normalise factors to a list of dicts for the template
    factor_list = []
    for name, spec in factors.items():
        if isinstance(spec, (list, tuple)):
            # Detect categorical vs continuous:
            # categorical = list of strings; continuous = 2-element numeric tuple/list
            is_continuous = (
                len(spec) == 2
                and all(isinstance(v, (int, float)) for v in spec)
            )
            if is_continuous:
                factor_list.append(
                    {"name": name, "type": "continuous", "low": spec[0], "high": spec[1]}
                )
            else:
                factor_list.append(
                    {"name": name, "type": "categorical", "levels": list(spec)}
                )
        else:
            # Fallback: treat as opaque string
            factor_list.append({"name": name, "type": "unknown", "spec": str(spec)})

    ctx: dict = {
        "formula": formula,
        "factors": factor_list,
        "alpha": power_cfg.alpha,
        "power_target": f"{power_cfg.power * 100:.1f} %",
        "max_n": power_cfg.max_n,
    }

    if isinstance(power_cfg, PowerContrastConfig):
        ctx["power_mode"] = "Contrast-based"
        ctx["sigma"] = power_cfg.sigma
        L = np.atleast_2d(power_cfg.L)
        ctx["L_shape"] = f"{L.shape[0]} \u00d7 {L.shape[1]}"
        delta = np.atleast_1d(power_cfg.delta)
        ctx["delta"] = "[" + ", ".join(f"{d:.4g}" for d in delta) + "]"
    elif isinstance(power_cfg, PowerR2Config):
        ctx["power_mode"] = "Global R\u00b2"
        ctx["r2_target"] = power_cfg.r2_target
        ctx["lambda_mode"] = power_cfg.lambda_mode
    else:
        ctx["power_mode"] = type(power_cfg).__name__

    return ctx


# ---------------------------------------------------------------------------
# B2 — Power metrics context
# ---------------------------------------------------------------------------


def _build_metrics_ctx(report: dict) -> dict:
    """Build the template context dict for the Power Metrics section (B2)."""
    achieved = float(report.get("achieved_power", 0.0))
    target = float(report.get("target_power", report.get("power", 0.0)))

    # Colour class: green if met, amber if within 5 pp, red otherwise
    diff = achieved - target
    if diff >= 0:
        power_class = "pass"
    elif diff >= -0.05:
        power_class = "warn"
    else:
        power_class = "fail"

    # Format elapsed time
    elapsed_raw = report.get("elapsed_sec")
    if elapsed_raw is not None:
        elapsed_str = f"{float(elapsed_raw):.2f} s"
    else:
        elapsed_str = "—"

    # Format noncentrality
    lam = report.get("noncentrality_lambda")
    lam_str = f"{float(lam):.3f}" if lam is not None else "—"

    return {
        "n": int(report.get("n", 0)),
        "achieved_power": f"{achieved * 100:.1f} %",
        "target_power": f"{target * 100:.1f} %",
        "power_class": power_class,
        "power_ok": diff >= 0,
        "noncentrality_lambda": lam_str,
        "df_num": report.get("df_num", "—"),
        "df_denom": report.get("df_denom", "—"),
        "criterion": report.get("criterion", "—"),
        "elapsed_sec": elapsed_str,
        "search_strategy": report.get("search_strategy", "—"),
        "random_state": report.get("random_state"),
        "warnings": list(report.get("warnings", [])),
    }


# ---------------------------------------------------------------------------
# B3 — DataFrame → HTML table
# ---------------------------------------------------------------------------


def _df_to_html(df: Any, max_rows: int) -> tuple[str, bool, int]:
    """Convert a DataFrame to an HTML table string.

    Returns (html_str, was_truncated, total_rows).
    """
    total = len(df)
    truncated = total > max_rows
    subset = df.head(max_rows)
    html = subset.to_html(
        index=False,
        border=0,
        classes="report-table",
        na_rep="—",
        float_format=lambda x: f"{x:.4g}",
    )
    return html, truncated, total


# ---------------------------------------------------------------------------
# B4 — Diagnostics context
# ---------------------------------------------------------------------------


def _build_diagnostics_ctx(report: dict) -> dict | None:
    """Build the template context dict for the Diagnostics section (B4).

    Returns None when diagnostics are absent, so the template section is skipped.
    """
    diag = report.get("diagnostics")
    if not diag:
        return None

    # Condition number badge — κ(X), Belsley scale (SR-21):
    # < 30 well-conditioned, 30–1000 moderate, > 1000 ill-conditioned.
    cond_raw = diag.get("condition_number")
    if cond_raw is not None:
        cond_val = float(cond_raw)
        cond_str = f"{cond_val:.2f}"
        if cond_val < 30:
            cond_badge = "pass"
        elif cond_val < 1000:
            cond_badge = "warn"
        else:
            cond_badge = "fail"
    else:
        cond_str = None
        cond_badge = None

    # D-efficiency
    d_eff = diag.get("d_efficiency")
    d_eff_str = f"{float(d_eff):.4f}" if d_eff is not None else None

    # I-criterion
    i_crit = diag.get("i_criterion")
    i_crit_str = f"{float(i_crit):.6f}" if i_crit is not None else None

    # VIFs — dict of {term: vif_value}
    vifs_raw = diag.get("vifs")
    if vifs_raw and isinstance(vifs_raw, dict):
        vifs = {k: f"{float(v):.3f}" for k, v in vifs_raw.items()}
    else:
        vifs = None

    return {
        "condition_number": cond_str,
        "condition_badge": cond_badge,
        "d_efficiency": d_eff_str,
        "i_criterion": i_crit_str,
        "vifs": vifs,
    }


# ---------------------------------------------------------------------------
# B5 — Embedded power curve figure
# ---------------------------------------------------------------------------


def _build_power_curve_figure(
    result: dict,
    formula: str,
    factors: dict,
    power_cfg: Any,
) -> str | None:
    """Generate a power-vs-n curve and return it as a base64 PNG string (B5).

    Tries Plotly first (rasterised via kaleido), then Matplotlib.
    Returns None if neither backend is available or figure generation fails.
    """
    from .config import DesignOptions
    from .power_curves import power_curve_by_n

    report = result.get("report", {})
    n_result = int(report.get("n", 10))
    random_state = int(report.get("random_state", 42))

    # Sweep from p+1 up to 2× chosen n, capped at max_n
    p = int(report.get("p", 2))
    max_n = int(getattr(power_cfg, "max_n", 500))
    n_min = max(p + 1, 2)
    n_max = min(n_result * 2, max_n)
    if n_max <= n_min:
        n_max = min(n_min + 40, max_n)

    opts = DesignOptions(
        auto_candidate=True,
        starts=2,
        random_state=random_state,
        criterion=report.get("criterion", "I"),
    )

    try:
        curve_result = power_curve_by_n(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=opts,
            n_range=(n_min, n_max),
            n_points=25,
        )
        curve_df = curve_result["data"]
    except Exception:
        return None

    target_power = float(getattr(power_cfg, "power", 0.80))

    # --- Try Plotly ---
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_df["n"], y=curve_df["power"],
            mode="lines+markers",
            name="Power",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
        ))
        fig.add_hline(
            y=target_power,
            line_dash="dash", line_color="#d62728",
            annotation_text=f"Target {target_power:.0%}",
            annotation_position="bottom right",
        )
        fig.add_vline(
            x=n_result,
            line_dash="dot", line_color="#2ca02c",
            annotation_text=f"n={n_result}",
            annotation_position="top left",
        )
        fig.update_layout(
            xaxis_title="Sample size (n)",
            yaxis_title="Power",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            margin=dict(l=50, r=30, t=30, b=50),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="sans-serif", size=12),
        )
        b64 = _fig_to_base64(fig)
        if b64:
            return b64
    except Exception:
        pass

    # --- Fallback: Matplotlib ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(curve_df["n"], curve_df["power"], color="#1f77b4", marker="o",
                markersize=4, linewidth=2)
        ax.axhline(target_power, color="#d62728", linestyle="--",
                   label=f"Target {target_power:.0%}")
        ax.axvline(n_result, color="#2ca02c", linestyle=":",
                   label=f"n = {n_result}")
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Power")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
        )
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        if b64:
            return b64
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# C1/C2 — Context assembly helper
# ---------------------------------------------------------------------------


def _build_context(
    result: dict,
    formula: str,
    factors: dict,
    power_cfg: Any,
    title: str,
    include_power_curve: bool,
    design_rows_shown: int,
) -> dict:
    """Assemble the full Jinja2 template context from all B-epic helpers."""
    from . import __version__

    report = result.get("report", {})
    design_df = result.get("design_df")
    buckets_df = result.get("buckets_df")

    # Header
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # B1 — config
    config_ctx = _build_config_ctx(formula, factors, power_cfg)

    # B2 — metrics
    metrics_ctx = _build_metrics_ctx(report)

    # B3 — design table
    if design_df is not None and len(design_df) > 0:
        design_html, design_truncated, design_total = _df_to_html(design_df, design_rows_shown)
    else:
        design_html, design_truncated, design_total = None, False, 0

    # B3 — buckets table (always show all rows)
    if buckets_df is not None and len(buckets_df) > 0:
        buckets_html, _, _ = _df_to_html(buckets_df, max_rows=len(buckets_df))
    else:
        buckets_html = None

    # B4 — diagnostics
    diagnostics_ctx = _build_diagnostics_ctx(report)

    # B5 — power curve figure
    power_curve_b64 = None
    include_power_curve_note = False
    if include_power_curve:
        power_curve_b64 = _build_power_curve_figure(result, formula, factors, power_cfg)
        if power_curve_b64 is None:
            include_power_curve_note = True

    return {
        "title": title,
        "generated_at": generated_at,
        "version": __version__,
        "config": config_ctx,
        "metrics": metrics_ctx,
        "design_html": design_html,
        "design_truncated": design_truncated,
        "design_rows_shown": design_rows_shown,
        "design_total_rows": design_total,
        "buckets_html": buckets_html,
        "diagnostics": diagnostics_ctx,
        "power_curve_b64": power_curve_b64,
        "include_power_curve_note": include_power_curve_note,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    result: dict,
    formula: str,
    factors: dict,
    power_cfg: Any,
    output_path: str | Path,
    title: str = "Lattice DOE Report",
    include_power_curve: bool = True,
    design_rows_shown: int = 30,
) -> Path:
    """Render and write a shareable HTML (or PDF) report.

    Parameters
    ----------
    result:
        The dict returned by ``find_optimal_design()``.
    formula:
        Patsy formula string used to generate the design.
    factors:
        Factor specification dict used to generate the design.
    power_cfg:
        ``PowerContrastConfig`` or ``PowerR2Config`` instance.
    output_path:
        Destination file path.  The suffix determines the format:
        ``.html`` (default) or ``.pdf`` (requires ``weasyprint``).
        If the path is a directory, ``iopt_report.html`` is written inside it.
    title:
        Report title shown in the HTML ``<title>`` tag and heading.
    include_power_curve:
        Whether to generate and embed a power-curve figure.  Set to
        ``False`` to skip (faster but less informative).
    design_rows_shown:
        Maximum number of design-table rows to include; a note is shown
        when the design is larger.

    Returns
    -------
    Path
        Resolved path of the file that was written.
    """
    # --- Resolve output path ---
    out = Path(output_path)
    if out.is_dir():
        out = out / "iopt_report.html"
    elif out.suffix not in (".html", ".pdf"):
        out = out.with_suffix(".html")

    out.parent.mkdir(parents=True, exist_ok=True)

    # --- Build template context (shared for both HTML and PDF) ---
    ctx = _build_context(
        result=result,
        formula=formula,
        factors=factors,
        power_cfg=power_cfg,
        title=title,
        include_power_curve=include_power_curve,
        design_rows_shown=design_rows_shown,
    )

    # --- Render template ---
    env = _get_jinja_env()
    template = env.get_template("report_template.html")
    html_str = template.render(**ctx)

    # --- C1: HTML output ---
    if out.suffix == ".html":
        out.write_text(html_str, encoding="utf-8")
        return out.resolve()

    # --- C2: PDF output (requires weasyprint) ---
    try:
        from weasyprint import HTML as WeasyprintHTML
    except ImportError as exc:
        raise ImportError(
            "PDF export requires weasyprint. "
            'Install it with: pip install "lattice-doe[report-pdf]"'
        ) from exc

    WeasyprintHTML(string=html_str).write_pdf(str(out))
    return out.resolve()
