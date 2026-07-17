"""
Plotly chart helpers for power curves, sensitivity plots, and comparison charts.

Ticket: E5, F1, F3.
Status: stub — implemented in Epics E and F.
"""

from __future__ import annotations

import pandas as pd


def power_curve_figure(df: pd.DataFrame, target_power: float, chosen_n: int):
    """
    Build a Plotly line chart of power vs n.

    Parameters
    ----------
    df : DataFrame with columns 'n' and 'power'.
    target_power : float — horizontal reference line.
    chosen_n : int — vertical reference line.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    raise NotImplementedError("Implemented in ticket E5.")


def sensitivity_figure(df: pd.DataFrame, x_col: str, nominal_power: float):
    """
    Build a Plotly line chart for sensitivity analysis.

    Parameters
    ----------
    df : DataFrame with x_col and 'power' columns.
    x_col : str — name of the x-axis column (e.g. 'sigma' or 'r2_target').
    nominal_power : float — horizontal reference line.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    raise NotImplementedError("Implemented in ticket F1.")


def criteria_comparison_figure(summary_df: pd.DataFrame):
    """
    Build a grouped bar chart comparing n and achieved_power across criteria.

    Parameters
    ----------
    summary_df : DataFrame from compare_criteria()["summary"].

    Returns
    -------
    plotly.graph_objects.Figure
    """
    raise NotImplementedError("Implemented in ticket F3.")
