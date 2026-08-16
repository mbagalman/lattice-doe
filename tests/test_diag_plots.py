# tests/test_diag_plots.py
"""Unit tests for diag_plots.create_diagnostic_plots (TD-13).

The module previously had NO direct tests — its coverage was entirely
incidental. create_diagnostic_plots is the sole public function and is
re-exported from the documented `lattice_doe.diagnostics` compat wrapper.

matplotlib is a soft dependency: the None-return paths are tested
everywhere (by patching the availability flag); the figure-producing
paths skip when matplotlib is absent, matching the CI no-extras install.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lattice_doe import diag_plots
from lattice_doe.diag_plots import create_diagnostic_plots

requires_matplotlib = pytest.mark.skipif(
    not diag_plots._HAS_MATPLOTLIB, reason="matplotlib not installed"
)


def _X(n=12, p=3):
    """Full-rank test matrix: intercept + linear + quadratic columns."""
    x = np.linspace(-1.0, 1.0, n)
    cols = [np.ones(n), x, x**2][:p]
    return np.column_stack(cols)


class TestUnavailableAndDegenerate:
    def test_returns_none_without_matplotlib(self):
        with patch.object(diag_plots, "_HAS_MATPLOTLIB", False):
            with pytest.warns(UserWarning, match="matplotlib not available"):
                assert create_diagnostic_plots(_X()) is None

    @requires_matplotlib
    def test_returns_none_for_zero_columns(self):
        with pytest.warns(UserWarning, match="0 columns"):
            assert create_diagnostic_plots(np.empty((5, 0))) is None


@requires_matplotlib
class TestFigureStructure:
    """Panel structure asserted via exact axis titles, which are the
    function's public visual contract (axes counts also include colorbars,
    so titles are the stable handle)."""

    @staticmethod
    def _titles(fig):
        return {ax.get_title() for ax in fig.axes if ax.get_title()}

    def test_multi_predictor_panels(self):
        fig = create_diagnostic_plots(_X(p=3))
        try:
            titles = self._titles(fig)
            assert "Variance Inflation Factors" in titles
            assert "Leverage Values (Design Point Influence)" in titles
            assert "Correlation Matrix (Non-Intercept Terms)" in titles
        finally:
            diag_plots.plt.close(fig)

    def test_single_predictor_is_leverage_only(self):
        fig = create_diagnostic_plots(_X(p=1))
        try:
            titles = self._titles(fig)
            assert "Leverage Values (Design Point Influence)" in titles
            assert "Variance Inflation Factors" not in titles
            assert "Correlation Matrix (Non-Intercept Terms)" not in titles
        finally:
            diag_plots.plt.close(fig)

    def test_design_space_panel_with_two_continuous_factors(self):
        n = 12
        design_df = pd.DataFrame({"x1": np.linspace(-1, 1, n), "x2": np.linspace(0, 5, n)})
        X = np.column_stack([np.ones(n), design_df["x1"], design_df["x2"]])
        fig = create_diagnostic_plots(X, design_df=design_df)
        try:
            assert "Design Space Coverage (colored by leverage)" in self._titles(fig)
        finally:
            diag_plots.plt.close(fig)

    def test_distribution_panel_with_one_continuous_factor(self):
        n = 12
        design_df = pd.DataFrame({"x1": np.linspace(-1, 1, n)})
        X = np.column_stack([np.ones(n), design_df["x1"]])
        fig = create_diagnostic_plots(X, design_df=design_df)
        try:
            assert "Design Points Distribution (colored by leverage)" in self._titles(fig)
        finally:
            diag_plots.plt.close(fig)

    def test_custom_feature_names_reach_the_vif_axis(self):
        names = ["Intercept", "Dose", "Dose2"]
        fig = create_diagnostic_plots(_X(p=3), feature_names=names)
        try:
            for ax in fig.axes:
                if ax.get_title() == "Variance Inflation Factors":
                    labels = [t.get_text() for t in ax.get_xticklabels()]
                    # VIF drops the intercept column; the named factors remain.
                    assert "Dose" in labels and "Dose2" in labels
                    break
            else:
                pytest.fail("VIF axis not found")
        finally:
            diag_plots.plt.close(fig)
