"""Plotly figure builders for power curve functions."""
from __future__ import annotations

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

__all__ = [
    "plotly_curve_by_n",
    "plotly_curve_by_effect",
    "plotly_surface_2d",
    "plotly_sensitivity",
]

_INSTALL_HINT = (
    'plotly is required for plot_backend="plotly". '
    'Install it with: pip install "lattice-doe[viz]"'
)


def plotly_curve_by_n(df, power_cfg, target_n):
    """Two-panel Plotly figure: power vs n (top) + design metrics (bottom).

    Parameters
    ----------
    df : DataFrame
        Output of the power_curve_by_n computation loop.
        Columns: n, power, lambda, d_efficiency, i_criterion, condition_number.
    power_cfg : PowerContrastConfig or PowerR2Config
        Used for reference lines and title text.
    target_n : int or None
        First n that achieves target power; draws a vertical reference line.
    """
    if not _HAS_PLOTLY:
        raise ImportError(_INSTALL_HINT)

    from .config import PowerContrastConfig

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        specs=[[{}], [{"secondary_y": True}]],
        subplot_titles=("Statistical Power vs n", "Design Quality Metrics"),
        vertical_spacing=0.10,
    )

    # --- Row 1: power curve ---
    fig.add_trace(
        go.Scatter(
            x=df["n"], y=df["power"],
            mode="lines+markers",
            name="Power",
            line=dict(color="royalblue", width=2),
            marker=dict(size=5),
            hovertemplate="n=%{x}<br>power=%{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # Target power reference line
    fig.add_hline(
        y=power_cfg.power,
        line_dash="dash", line_color="red", line_width=1.5,
        annotation_text=f"Target {power_cfg.power:.0%}",
        annotation_position="top right",
        row=1, col=1,
    )
    # Target n vertical line
    if target_n is not None:
        fig.add_vline(
            x=target_n,
            line_dash="dash", line_color="green", line_width=1.5,
            annotation_text=f"n={target_n}",
            annotation_position="top right",
            row=1, col=1,
        )

    # --- Row 2: I-criterion (left) + D-efficiency (right) ---
    fig.add_trace(
        go.Scatter(
            x=df["n"], y=df["i_criterion"],
            mode="lines+markers",
            name="I-criterion",
            line=dict(color="green", width=2),
            marker=dict(size=5),
            hovertemplate="n=%{x}<br>I-crit=%{y:.4f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["n"], y=df["d_efficiency"],
            mode="lines+markers",
            name="D-efficiency",
            line=dict(color="darkorange", width=2),
            marker=dict(size=5),
            hovertemplate="n=%{x}<br>D-eff=%{y:.4f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=True,
    )

    # --- Title ---
    if isinstance(power_cfg, PowerContrastConfig):
        title = (
            f"Power vs Sample Size — Contrast Test "
            f"(σ={power_cfg.sigma}, α={power_cfg.alpha})"
        )
    else:
        title = (
            f"Power vs Sample Size — Global F-Test "
            f"(R²={power_cfg.r2_target}, α={power_cfg.alpha})"
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Statistical Power", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="I-criterion (lower ↓)", title_font_color="green",
                     row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="D-efficiency (higher ↑)", title_font_color="darkorange",
                     row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Sample Size (n)", row=2, col=1)

    return fig


def plotly_curve_by_effect(df, power_cfg, min_detectable, n):
    """Single-panel Plotly figure: power vs effect size.

    Parameters
    ----------
    df : DataFrame
        Output of power_curve_by_effect. Columns: effect_size, power, lambda
        (plus actual_delta_norm for contrast mode).
    power_cfg : PowerContrastConfig or PowerR2Config
        Used for reference lines and title text.
    min_detectable : float or None
        Effect size where power first reaches 80%; draws a vertical line.
    n : int
        Fixed sample size used; shown in the title.
    """
    if not _HAS_PLOTLY:
        raise ImportError(_INSTALL_HINT)

    from .config import PowerContrastConfig

    fig = go.Figure()

    # Power curve
    fig.add_trace(
        go.Scatter(
            x=df["effect_size"], y=df["power"],
            mode="lines+markers",
            name="Power",
            line=dict(color="royalblue", width=2),
            marker=dict(size=5),
            hovertemplate="effect=%{x:.4f}<br>power=%{y:.3f}<extra></extra>",
        )
    )

    # 80% reference line
    fig.add_hline(
        y=0.80,
        line_dash="dash", line_color="red", line_width=1.5,
        annotation_text="80% Power",
        annotation_position="top left",
    )

    # Target power reference line (only if different from 80%)
    if abs(power_cfg.power - 0.80) > 1e-6:
        fig.add_hline(
            y=power_cfg.power,
            line_dash="dash", line_color="green", line_width=1.5,
            annotation_text=f"Target {power_cfg.power:.0%}",
            annotation_position="top right",
        )

    # MDE vertical line
    if min_detectable is not None:
        fig.add_vline(
            x=min_detectable,
            line_dash="dot", line_color="darkorange", line_width=1.5,
            annotation_text=f"MDE={min_detectable:.3f}",
            annotation_position="top right",
        )

    # Title and axis labels
    if isinstance(power_cfg, PowerContrastConfig):
        x_label = "Effect Size Multiplier (on base norm)"
        title = (
            f"Power vs Effect Size at n={n} — Contrast Test "
            f"(σ={power_cfg.sigma}, α={power_cfg.alpha})"
        )
    else:
        x_label = "R² Effect Size"
        title = (
            f"Power vs Effect Size at n={n} — Global F-Test "
            f"(α={power_cfg.alpha})"
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=x_label,
        yaxis_title="Statistical Power",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plotly_surface_2d(power_grid, axis1, axis2, power_cfg, param1, param2):
    """Heatmap + target-power contour overlay.

    Parameters
    ----------
    power_grid : 2D ndarray, shape (len(axis1), len(axis2))
        Power values on the grid; axis1 → rows (y), axis2 → cols (x).
    axis1 : 1D ndarray
        Values for param1 (y-axis).
    axis2 : 1D ndarray
        Values for param2 (x-axis).
    power_cfg : PowerContrastConfig or PowerR2Config
        Used for target power contour level and title text.
    param1 : str
        Name of the parameter on the y-axis (e.g. 'n', 'sigma').
    param2 : str
        Name of the parameter on the x-axis.
    """
    if not _HAS_PLOTLY:
        raise ImportError(_INSTALL_HINT)

    fig = go.Figure()

    # Filled heatmap (viridis-equivalent, power clamped to [0, 1])
    fig.add_trace(
        go.Heatmap(
            x=list(axis2),
            y=list(axis1),
            z=power_grid.tolist(),
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Power"),
            hovertemplate=(
                f"{param2}=%{{x}}<br>{param1}=%{{y}}<br>power=%{{z:.3f}}<extra></extra>"
            ),
        )
    )

    # Target-power contour line drawn on top
    target = float(power_cfg.power)
    fig.add_trace(
        go.Contour(
            x=list(axis2),
            y=list(axis1),
            z=power_grid.tolist(),
            contours=dict(
                start=target,
                end=target + 1e-9,
                size=1.0,
                showlabels=True,
                labelfont=dict(color="white"),
            ),
            line=dict(color="white", width=2, dash="dash"),
            showscale=False,
            name=f"Target {target:.0%}",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=(
            f"Power Surface: {param1} \u00d7 {param2}"
            f"  (target = {target:.2f}, white contour)"
        ),
        xaxis_title=param2,
        yaxis_title=param1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plotly_sensitivity(df, power_cfg, nominal_pwr, n):
    """Single-panel Plotly figure: power vs sigma (contrast) or r2_target (R²).

    Parameters
    ----------
    df : DataFrame
        Output of power_sensitivity.
        Contrast mode columns: sigma, power, noncentrality_lambda.
        R² mode columns: r2_target, power, noncentrality_lambda.
    power_cfg : PowerContrastConfig or PowerR2Config
        Used for nominal reference lines and title text.
    nominal_pwr : float
        Power evaluated at the nominal parameter value (reference hline).
    n : int
        Fixed sample size; shown in the title.
    """
    if not _HAS_PLOTLY:
        raise ImportError(_INSTALL_HINT)

    from .config import PowerContrastConfig

    is_contrast = isinstance(power_cfg, PowerContrastConfig)

    if is_contrast:
        x_col = "sigma"
        x_label = "\u03c3  (residual standard deviation)"
        nominal_x = float(power_cfg.sigma)
        nominal_label = f"Nominal \u03c3 = {power_cfg.sigma}"
        title = f"Power Sensitivity to \u03c3  (n = {n})"
    else:
        x_col = "r2_target"
        x_label = "R\u00b2 (population effect size)"
        nominal_x = float(power_cfg.r2_target)
        nominal_label = f"Nominal R\u00b2 = {power_cfg.r2_target}"
        title = f"Power Sensitivity to R\u00b2  (n = {n})"

    fig = go.Figure()

    # Power curve
    fig.add_trace(
        go.Scatter(
            x=df[x_col], y=df["power"],
            mode="lines",
            name="Power",
            line=dict(color="royalblue", width=2),
            hovertemplate=f"{x_col}=%{{x:.4f}}<br>power=%{{y:.3f}}<extra></extra>",
        )
    )

    # Nominal parameter vline (gray dashed)
    fig.add_vline(
        x=nominal_x,
        line_dash="dash", line_color="gray", line_width=1.5,
        annotation_text=nominal_label,
        annotation_position="top left",
    )

    # Target power hline (red dashed)
    fig.add_hline(
        y=power_cfg.power,
        line_dash="dash", line_color="red", line_width=1.5,
        annotation_text=f"Target power = {power_cfg.power:.2f}",
        annotation_position="top right",
    )

    # Nominal power hline (steelblue dotted)
    fig.add_hline(
        y=float(nominal_pwr),
        line_dash="dot", line_color="steelblue", line_width=1.5,
        annotation_text=f"Power @ nominal: {float(nominal_pwr):.3f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=x_label,
        yaxis_title="Statistical Power",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
