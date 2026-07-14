# tests/test_plot_backends.py
"""Tests for the Plotly figure backends (plot_backends.py) and related wiring.

All tests that require plotly are skipped when plotly is not installed.
Tests that mock the _HAS_PLOTLY flag run regardless of whether plotly is present.
"""
from __future__ import annotations

import importlib.util
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lattice_doe.config import DesignOptions, PowerContrastConfig, PowerR2Config

# ---------------------------------------------------------------------------
# Availability flag used for skipif markers
# ---------------------------------------------------------------------------

_PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FORMULA = "x1 + x2"
FACTORS = {"x1": (-1.0, 1.0), "x2": (-1.0, 1.0)}

FAST_OPTS = DesignOptions(
    candidate_points=80,
    starts=1,
    max_iter=20,
    random_state=0,
)

# Larger pool for wrapper tests (which can't pass n_range and auto-compute up to n=100)
FAST_OPTS_WIDE = DesignOptions(
    candidate_points=200,
    starts=1,
    max_iter=20,
    random_state=0,
)


def _contrast_cfg() -> PowerContrastConfig:
    # model matrix has 3 cols: intercept, x1, x2
    L = np.array([[0.0, 1.0, 0.0]])
    delta = np.array([1.0])
    return PowerContrastConfig(L=L, delta=delta, sigma=1.0, power=0.80, alpha=0.05, max_n=30)


def _r2_cfg() -> PowerR2Config:
    return PowerR2Config(r2_target=0.3, power=0.80, alpha=0.05, max_n=30)


def _tiny_design_df() -> pd.DataFrame:
    """5-row design for x1, x2 — more than p=3 parameters so X is full-rank."""
    return pd.DataFrame({
        "x1": [-1.0, -1.0,  0.0,  1.0,  1.0],
        "x2": [-1.0,  1.0,  0.0, -1.0,  1.0],
    })


# ---------------------------------------------------------------------------
# TestPlotlyBackendImport — runs regardless of whether plotly is installed
# ---------------------------------------------------------------------------

class TestPlotlyBackendImport:
    """Verify that each helper raises ImportError when plotly is absent."""

    def _call_all_helpers_with_no_plotly(self):
        import lattice_doe.plot_backends as pb

        dummy_df = pd.DataFrame({"n": [5], "power": [0.5], "i_criterion": [1.0], "d_efficiency": [0.5]})
        dummy_grid = np.array([[0.5, 0.6], [0.7, 0.8]])
        cfg = _contrast_cfg()

        with patch.object(pb, "_HAS_PLOTLY", False):
            with pytest.raises(ImportError, match="plotly"):
                pb.plotly_curve_by_n(dummy_df, cfg, target_n=5)
            with pytest.raises(ImportError, match="plotly"):
                pb.plotly_curve_by_effect(dummy_df, cfg, min_detectable=None, n=10)
            with pytest.raises(ImportError, match="plotly"):
                pb.plotly_surface_2d(dummy_grid, np.array([5, 10]), np.array([0.1, 0.3]),
                                     cfg, "n", "effect")
            with pytest.raises(ImportError, match="plotly"):
                pb.plotly_sensitivity(dummy_df, cfg, nominal_pwr=0.75, n=10)

    def test_all_helpers_raise_import_error_when_plotly_absent(self):
        self._call_all_helpers_with_no_plotly()

    def test_install_hint_mentions_viz(self):
        import lattice_doe.plot_backends as pb
        assert "viz" in pb._INSTALL_HINT or "plotly" in pb._INSTALL_HINT


# ---------------------------------------------------------------------------
# TestPowerSurface2dExport — runs regardless of plotly
# ---------------------------------------------------------------------------

class TestPowerSurface2dExport:
    def test_power_surface_2d_importable(self):
        from lattice_doe import power_surface_2d
        assert callable(power_surface_2d)

    def test_power_surface_2d_in_all(self):
        import lattice_doe as iop
        assert "power_surface_2d" in iop.__all__


# ---------------------------------------------------------------------------
# Plotly-dependent tests — skipped when plotly is not installed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _PLOTLY_AVAILABLE, reason="plotly not installed")
class TestPlotlyCurveByN:
    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        from lattice_doe.power_curves import power_curve_by_n

        result = power_curve_by_n(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            n_range=(5, 12),
            n_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        assert isinstance(result["figure"], go.Figure)

    def test_has_expected_traces(self):
        from lattice_doe.power_curves import power_curve_by_n

        result = power_curve_by_n(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_r2_cfg(),
            n_range=(5, 12),
            n_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        fig = result["figure"]
        trace_names = [t.name for t in fig.data]
        assert "Power" in trace_names
        assert "I-criterion" in trace_names
        assert "D-efficiency" in trace_names

    def test_plot_false_ignores_backend(self):
        from lattice_doe.power_curves import power_curve_by_n

        result = power_curve_by_n(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            n_range=(5, 12),
            n_points=3,
            design_opts=FAST_OPTS,
            plot=False,
            plot_backend="plotly",
        )
        assert result["figure"] is None

    def test_wrapper_accepts_plot_backend(self):
        # Wrapper has no n_range param; use FAST_OPTS_WIDE so pool >= auto-computed max_n
        from lattice_doe import power_curve_by_n as wrapper

        df = wrapper(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            design_opts=FAST_OPTS_WIDE,
            plot=False,
            plot_backend="plotly",
        )
        assert isinstance(df, pd.DataFrame)


@pytest.mark.skipif(not _PLOTLY_AVAILABLE, reason="plotly not installed")
class TestPlotlyCurveByEffect:
    def test_contrast_returns_plotly_figure(self):
        import plotly.graph_objects as go
        from lattice_doe.power_curves import power_curve_by_effect

        result = power_curve_by_effect(
            formula=FORMULA,
            factors=FACTORS,
            n=10,
            power_cfg=_contrast_cfg(),
            effect_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        assert isinstance(result["figure"], go.Figure)

    def test_r2_returns_plotly_figure(self):
        import plotly.graph_objects as go
        from lattice_doe.power_curves import power_curve_by_effect

        result = power_curve_by_effect(
            formula=FORMULA,
            factors=FACTORS,
            n=10,
            power_cfg=_r2_cfg(),
            effect_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        assert isinstance(result["figure"], go.Figure)

    def test_has_power_trace(self):
        from lattice_doe.power_curves import power_curve_by_effect

        result = power_curve_by_effect(
            formula=FORMULA,
            factors=FACTORS,
            n=10,
            power_cfg=_contrast_cfg(),
            effect_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        fig = result["figure"]
        assert any(t.name == "Power" for t in fig.data)

    def test_plot_false_ignores_backend(self):
        from lattice_doe.power_curves import power_curve_by_effect

        result = power_curve_by_effect(
            formula=FORMULA,
            factors=FACTORS,
            n=10,
            power_cfg=_contrast_cfg(),
            effect_points=3,
            design_opts=FAST_OPTS,
            plot=False,
            plot_backend="plotly",
        )
        assert result["figure"] is None


@pytest.mark.skipif(not _PLOTLY_AVAILABLE, reason="plotly not installed")
class TestPlotlySurface2d:
    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        from lattice_doe.power_curves import power_surface_2d

        result = power_surface_2d(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_r2_cfg(),
            param1="n",
            param1_range=(5, 15),
            param2="effect",
            param2_range=(0.05, 0.40),
            grid_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        assert isinstance(result["figure"], go.Figure)

    def test_has_heatmap_and_contour(self):
        from lattice_doe.power_curves import power_surface_2d

        result = power_surface_2d(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_r2_cfg(),
            param1="n",
            param1_range=(5, 15),
            param2="effect",
            param2_range=(0.05, 0.40),
            grid_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        trace_types = [t.type for t in result["figure"].data]
        assert "heatmap" in trace_types
        assert "contour" in trace_types

    def test_plot_false_ignores_backend(self):
        from lattice_doe.power_curves import power_surface_2d

        result = power_surface_2d(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_r2_cfg(),
            param1="n",
            param1_range=(5, 15),
            param2="effect",
            param2_range=(0.05, 0.40),
            grid_points=3,
            design_opts=FAST_OPTS,
            plot=False,
            plot_backend="plotly",
        )
        assert result["figure"] is None


@pytest.mark.skipif(not _PLOTLY_AVAILABLE, reason="plotly not installed")
class TestPlotlySensitivity:
    def test_contrast_returns_plotly_figure(self):
        import plotly.graph_objects as go
        from lattice_doe import power_sensitivity

        result = power_sensitivity(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            design_df=_tiny_design_df(),
            sigma_range=(0.5, 2.0),
            sigma_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        assert isinstance(result["figure"], go.Figure)

    def test_r2_returns_plotly_figure(self):
        import plotly.graph_objects as go
        from lattice_doe import power_sensitivity

        result = power_sensitivity(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_r2_cfg(),
            design_df=_tiny_design_df(),
            r2_range=(0.05, 0.45),
            r2_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        assert isinstance(result["figure"], go.Figure)

    def test_has_power_trace_and_reference_lines(self):
        from lattice_doe import power_sensitivity

        result = power_sensitivity(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            design_df=_tiny_design_df(),
            sigma_range=(0.5, 2.0),
            sigma_points=3,
            design_opts=FAST_OPTS,
            plot=True,
            plot_backend="plotly",
        )
        fig = result["figure"]
        assert any(t.name == "Power" for t in fig.data)
        # 3 reference lines: nominal vline + target hline + nominal_pwr hline
        assert len(fig.layout.shapes) == 3

    def test_plot_false_ignores_backend(self):
        from lattice_doe import power_sensitivity

        result = power_sensitivity(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            design_df=_tiny_design_df(),
            sigma_range=(0.5, 2.0),
            sigma_points=3,
            design_opts=FAST_OPTS,
            plot=False,
            plot_backend="plotly",
        )
        assert result["figure"] is None
