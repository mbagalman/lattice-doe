# tests/test_widgets.py
# License: MIT
"""Unit tests for iopt_power_design.widgets.

Layer 1 — pure Python helpers (_parse_matrix, _parse_vector,
           _build_power_cfg_from_state, _build_design_opts_from_state):
           run unconditionally with no ipywidgets dependency.

Layer 2 — DesignWidget construction and interaction: skipped when
           ipywidgets is not installed; all UI actions are exercised by
           calling private methods directly (no live kernel needed).

Layer 3 — Run callback: find_optimal_design is mocked so tests
           pass without running a real design search.

Layer 4 — Import guard: patches _HAS_WIDGETS=False to verify WidgetsError
           is raised; runs in any CI environment.
"""
from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from iopt_power_design.widgets import (
    WidgetsError,
    _parse_matrix,
    _parse_vector,
    _build_power_cfg_from_state,
    _build_design_opts_from_state,
    _approx_power_curve,
    design_widget,
    DesignWidget,
)
from iopt_power_design.config import DesignOptions, PowerContrastConfig, PowerR2Config

_HAS_WIDGETS = importlib.util.find_spec("ipywidgets") is not None

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _minimal_result():
    """Minimal result dict matching find_optimal_design() output."""
    design_df = pd.DataFrame({"A": [0.1, 0.5, -0.5], "B": [-1.0, 1.0, 0.0]})
    buckets_df = pd.DataFrame({"A": [0.1], "count": [3]})
    report = {
        "n": 10,
        "p": 3,
        "df_num": 2,
        "df_denom": 7,
        "alpha": 0.05,
        "target_power": 0.80,
        "achieved_power": 0.85,
        "noncentrality_lambda": 8.0,
        "criterion": "I",
        "elapsed_sec": 1.23,
        "search_strategy": "bisection",
        "warnings": [],
        "diagnostics": {
            "i_criterion": 0.45,
            "d_efficiency": 0.91,
            "condition_number": 3.2,
            "leverages": [0.3, 0.3, 0.4],
        },
    }
    return {"design_df": design_df, "buckets_df": buckets_df, "report": report}


def _r2_state(**overrides) -> dict:
    """Default R² widget state dict."""
    state = {
        "power_mode": "r2",
        "alpha": 0.05,
        "power_target": 0.80,
        "sigma": 1.0,
        "max_n": 500,
        "r2_target": 0.15,
        "lambda_mode": "n",
        "criterion": "I",
        "starts": 5,
        "random_state": 123,
        "auto_candidate": True,
        "candidate_points": 2000,
        "constraint_expr": "",
    }
    state.update(overrides)
    return state


def _contrast_state(**overrides) -> dict:
    """Default contrast widget state dict."""
    state = {
        "power_mode": "contrast",
        "alpha": 0.05,
        "power_target": 0.80,
        "sigma": 1.0,
        "max_n": 500,
        "L_text": "0 1 0",
        "delta_text": "0.5",
        "criterion": "I",
        "starts": 5,
        "random_state": 123,
        "auto_candidate": True,
        "candidate_points": 2000,
        "constraint_expr": "",
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Layer 4 — Import guard (always runs)
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_widgets_error_is_importable(self):
        """WidgetsError is always importable regardless of ipywidgets presence."""
        assert issubclass(WidgetsError, RuntimeError)

    def test_design_widget_raises_when_ipywidgets_absent(self):
        """design_widget() raises WidgetsError when _HAS_WIDGETS is False."""
        with patch("iopt_power_design.widgets._HAS_WIDGETS", False):
            with pytest.raises(WidgetsError, match="ipywidgets"):
                design_widget()

    def test_DesignWidget_raises_when_ipywidgets_absent(self):
        """DesignWidget() raises WidgetsError when _HAS_WIDGETS is False."""
        with patch("iopt_power_design.widgets._HAS_WIDGETS", False):
            with pytest.raises(WidgetsError, match="ipywidgets"):
                DesignWidget()

    def test_helpers_importable_without_widgets(self):
        """Pure helpers are always importable and callable."""
        arr = _parse_matrix("1 2\n3 4")
        assert arr.shape == (2, 2)


# ---------------------------------------------------------------------------
# Layer 1 — Pure-Python parse helpers
# ---------------------------------------------------------------------------

class TestParseMatrix:
    def test_space_separated(self):
        result = _parse_matrix("0 1 0\n0 0 1")
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result[0], [0, 1, 0])

    def test_comma_separated(self):
        result = _parse_matrix("0,1,0\n0,0,1")
        assert result.shape == (2, 3)

    def test_mixed_separators(self):
        result = _parse_matrix("1, 2 3\n4 5, 6")
        assert result.shape == (2, 3)

    def test_single_row(self):
        result = _parse_matrix("0 1 0")
        assert result.shape == (1, 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_matrix("")

    def test_blank_lines_skipped(self):
        result = _parse_matrix("\n1 2\n\n3 4\n")
        assert result.shape == (2, 2)


class TestParseVector:
    def test_space_separated(self):
        result = _parse_vector("0.5 1.0 -0.5")
        np.testing.assert_allclose(result, [0.5, 1.0, -0.5])

    def test_comma_separated(self):
        result = _parse_vector("0.5,1.0,-0.5")
        np.testing.assert_allclose(result, [0.5, 1.0, -0.5])

    def test_single_value(self):
        result = _parse_vector("2.5")
        np.testing.assert_allclose(result, [2.5])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_vector("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_vector("   ")


# ---------------------------------------------------------------------------
# Layer 1 — _build_power_cfg_from_state
# ---------------------------------------------------------------------------

class TestBuildPowerCfgFromState:
    def test_r2_mode_returns_PowerR2Config(self):
        cfg = _build_power_cfg_from_state(_r2_state())
        assert isinstance(cfg, PowerR2Config)

    def test_r2_fields_propagated(self):
        cfg = _build_power_cfg_from_state(_r2_state(r2_target=0.25, alpha=0.01, power_target=0.90, max_n=300))
        assert cfg.r2_target == pytest.approx(0.25)
        assert cfg.alpha == pytest.approx(0.01)
        assert cfg.power == pytest.approx(0.90)
        assert cfg.max_n == 300

    def test_r2_lambda_mode_propagated(self):
        cfg = _build_power_cfg_from_state(_r2_state(lambda_mode="n_minus_p"))
        assert cfg.lambda_mode == "n_minus_p"

    def test_contrast_mode_returns_PowerContrastConfig(self):
        cfg = _build_power_cfg_from_state(_contrast_state())
        assert isinstance(cfg, PowerContrastConfig)

    def test_contrast_L_shape(self):
        cfg = _build_power_cfg_from_state(_contrast_state(L_text="0 1 0\n0 0 1", delta_text="0.5 0.5"))
        assert cfg.L.shape == (2, 3)

    def test_contrast_sigma_propagated(self):
        cfg = _build_power_cfg_from_state(_contrast_state(sigma=2.5))
        assert cfg.sigma == pytest.approx(2.5)

    def test_contrast_bad_L_raises(self):
        with pytest.raises((ValueError, Exception)):
            _build_power_cfg_from_state(_contrast_state(L_text="not a number"))

    def test_contrast_empty_delta_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _build_power_cfg_from_state(_contrast_state(delta_text=""))

    def test_r2_defaults_are_valid(self):
        """Default state must not raise during config construction."""
        cfg = _build_power_cfg_from_state(_r2_state())
        assert cfg.alpha > 0
        assert cfg.power > 0


# ---------------------------------------------------------------------------
# Layer 1 — _build_design_opts_from_state
# ---------------------------------------------------------------------------

class TestBuildDesignOptsFromState:
    def test_returns_DesignOptions(self):
        opts = _build_design_opts_from_state(_r2_state())
        assert isinstance(opts, DesignOptions)

    def test_criterion_propagated(self):
        opts = _build_design_opts_from_state(_r2_state(criterion="D"))
        assert opts.criterion == "D"

    def test_starts_propagated(self):
        opts = _build_design_opts_from_state(_r2_state(starts=10))
        assert opts.starts == 10

    def test_random_state_propagated(self):
        opts = _build_design_opts_from_state(_r2_state(random_state=42))
        assert opts.random_state == 42

    def test_auto_candidate_true(self):
        opts = _build_design_opts_from_state(_r2_state(auto_candidate=True))
        assert opts.auto_candidate is True

    def test_auto_candidate_false_passes_candidate_points(self):
        opts = _build_design_opts_from_state(_r2_state(auto_candidate=False, candidate_points=500))
        assert opts.auto_candidate is False
        assert opts.candidate_points == 500

    def test_constraint_expr_wired(self):
        opts = _build_design_opts_from_state(_r2_state(constraint_expr="A + B <= 1"))
        assert opts.constraint_expr == "A + B <= 1"

    def test_empty_constraint_expr_not_set(self):
        opts = _build_design_opts_from_state(_r2_state(constraint_expr=""))
        assert not opts.constraint_expr

    def test_extra_kwargs_passthrough(self):
        """Non-exposed kwargs (e.g. n_blocks) are merged from extra_kwargs."""
        opts = _build_design_opts_from_state(_r2_state(), extra_kwargs={"n_blocks": 3})
        assert opts.n_blocks == 3

    def test_widget_fields_override_extra_kwargs(self):
        """Widget criterion overrides any criterion in extra_kwargs."""
        opts = _build_design_opts_from_state(
            _r2_state(criterion="A"),
            extra_kwargs={"criterion": "D"},  # should be overridden
        )
        assert opts.criterion == "A"


# ---------------------------------------------------------------------------
# Layer 1 — _approx_power_curve
# ---------------------------------------------------------------------------

class TestApproxPowerCurve:
    _report = {
        "n": 20, "p": 3, "df_num": 2,
        "df_denom": 17, "noncentrality_lambda": 10.0,
    }

    def test_returns_list_same_length_as_n_vals(self):
        n_vals = list(range(5, 30))
        powers = _approx_power_curve(n_vals, self._report, 0.05, "r2", "n")
        assert len(powers) == len(n_vals)

    def test_power_increases_with_n(self):
        n_vals = list(range(10, 50))
        powers = _approx_power_curve(n_vals, self._report, 0.05, "r2", "n")
        # Power should be non-decreasing for a well-conditioned noncentrality
        for i in range(len(powers) - 1):
            assert powers[i+1] >= powers[i] - 1e-9

    def test_power_zero_for_n_le_p(self):
        powers = _approx_power_curve([1, 2, 3], self._report, 0.05, "r2", "n")
        assert all(pw == 0.0 for pw in powers)

    def test_n_minus_p_lambda_mode(self):
        powers = _approx_power_curve([20], self._report, 0.05, "r2", "n_minus_p")
        assert 0.0 <= powers[0] <= 1.0

    def test_contrast_mode(self):
        powers = _approx_power_curve([20], self._report, 0.05, "contrast", "n")
        assert 0.0 <= powers[0] <= 1.0


# ---------------------------------------------------------------------------
# Layer 2 — DesignWidget construction (requires ipywidgets)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_WIDGETS, reason="ipywidgets not installed")
class TestDesignWidgetConstruction:
    def test_default_construction_does_not_raise(self):
        w = DesignWidget()
        assert w is not None

    def test_get_result_returns_none_before_run(self):
        w = DesignWidget()
        assert w.get_result() is None

    def test_get_design_df_returns_none_before_run(self):
        w = DesignWidget()
        assert w.get_design_df() is None

    def test_get_report_returns_none_before_run(self):
        w = DesignWidget()
        assert w.get_report() is None

    def test_construction_with_r2_mode(self):
        w = DesignWidget(power_mode="r2", r2_target=0.25, alpha=0.01)
        assert w._mode_toggle.value == "r2"
        assert w._r2_slider.value == pytest.approx(0.25)
        assert w._alpha_slider.value == pytest.approx(0.01)

    def test_construction_with_contrast_mode(self):
        w = DesignWidget(power_mode="contrast")
        assert w._mode_toggle.value == "contrast"

    def test_formula_pre_filled(self):
        w = DesignWidget(formula="~ 1 + A + B + A:B")
        assert w._formula_widget.value == "~ 1 + A + B + A:B"

    def test_factors_pre_populate_table(self):
        w = DesignWidget(factors={"X1": (-1.0, 1.0), "X2": (-1.0, 1.0), "X3": [-1.0, 1.0]})
        assert len(w._factor_rows) == 3

    def test_design_opts_pre_fills_advanced_widgets(self):
        opts = DesignOptions(starts=20, criterion="D", random_state=42)
        w = DesignWidget(design_opts=opts)
        assert w._starts_slider.value == 20
        assert w._criterion_dd.value == "D"
        assert w._seed_widget.value == 42

    def test_extra_do_kwargs_captures_non_exposed_fields(self):
        opts = DesignOptions(n_blocks=3)
        w = DesignWidget(design_opts=opts)
        assert w._extra_do_kwargs.get("n_blocks") == 3


@pytest.mark.skipif(not _HAS_WIDGETS, reason="ipywidgets not installed")
class TestDesignWidgetFactorTable:
    def test_add_factor_row_increases_count(self):
        w = DesignWidget(factors={})  # start with no initial factors
        initial = len(w._factor_rows)
        w._add_factor_row(name="NewFactor")
        assert len(w._factor_rows) == initial + 1

    def test_remove_factor_row_decreases_count(self):
        w = DesignWidget(factors={"A": (-1, 1), "B": (-1, 1)})
        initial = len(w._factor_rows)
        row_to_remove = w._factor_rows[0]
        w._remove_factor_row(row_to_remove)
        assert len(w._factor_rows) == initial - 1

    def test_get_factor_spec_continuous(self):
        w = DesignWidget(factors={"A": (-2.0, 2.0)})
        spec = w._get_factor_spec()
        assert "A" in spec
        assert spec["A"] == (-2.0, 2.0)

    def test_get_factor_spec_categorical(self):
        w = DesignWidget(factors={"Cat": ["low", "med", "high"]})
        spec = w._get_factor_spec()
        assert "Cat" in spec
        assert spec["Cat"] == ["low", "med", "high"]

    def test_get_factor_spec_raises_on_empty_name(self):
        w = DesignWidget(factors={})
        w._add_factor_row(name="")  # empty name
        with pytest.raises(ValueError, match="name"):
            w._get_factor_spec()

    def test_validate_inputs_empty_factor_list(self):
        w = DesignWidget(factors={})
        w._factor_rows.clear()
        w._factor_table_box.children = ()
        errors = w._validate_inputs()
        assert any("factor" in e.lower() for e in errors)

    def test_validate_inputs_valid_state(self):
        w = DesignWidget(factors={"A": (-1, 1), "B": (-1, 1)})
        errors = w._validate_inputs()
        assert errors == []


@pytest.mark.skipif(not _HAS_WIDGETS, reason="ipywidgets not installed")
class TestDesignWidgetModeToggle:
    def test_r2_box_visible_in_r2_mode(self):
        w = DesignWidget(power_mode="r2")
        assert w._r2_box.layout.display == ""
        assert w._contrast_box.layout.display == "none"

    def test_contrast_box_visible_in_contrast_mode(self):
        w = DesignWidget(power_mode="contrast")
        assert w._contrast_box.layout.display == ""
        assert w._r2_box.layout.display == "none"

    def test_toggling_mode_updates_visibility(self):
        w = DesignWidget(power_mode="r2")
        w._mode_toggle.value = "contrast"
        assert w._contrast_box.layout.display == ""
        assert w._r2_box.layout.display == "none"


@pytest.mark.skipif(not _HAS_WIDGETS, reason="ipywidgets not installed")
class TestDesignWidgetGetState:
    def test_get_state_r2_mode(self):
        w = DesignWidget(power_mode="r2", r2_target=0.20, alpha=0.01)
        state = w._get_state()
        assert state["power_mode"] == "r2"
        assert state["r2_target"] == pytest.approx(0.20)
        assert state["alpha"] == pytest.approx(0.01)
        assert "L_text" not in state  # contrast fields absent in r2 mode

    def test_get_state_contrast_mode_includes_L(self):
        w = DesignWidget(power_mode="contrast")
        state = w._get_state()
        assert state["power_mode"] == "contrast"
        assert "L_text" in state
        assert "delta_text" in state


@pytest.mark.skipif(not _HAS_WIDGETS, reason="ipywidgets not installed")
class TestDesignWidgetReset:
    def test_reset_clears_result(self):
        w = DesignWidget()
        w._result = _minimal_result()
        w.reset()
        assert w.get_result() is None

    def test_reset_restores_formula(self):
        w = DesignWidget(formula="~ 1 + A")
        w._formula_widget.value = "~ 1 + A + B"
        w.reset()
        assert w._formula_widget.value == "~ 1 + A"

    def test_reset_restores_alpha(self):
        w = DesignWidget(alpha=0.01)
        w._alpha_slider.value = 0.10
        w.reset()
        assert w._alpha_slider.value == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Layer 3 — Run callback (mock find_optimal_design)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_WIDGETS, reason="ipywidgets not installed")
class TestDesignWidgetRunCallback:
    @patch("iopt_power_design.widgets.find_optimal_design")
    def test_run_populates_result(self, mock_run):
        mock_run.return_value = _minimal_result()
        w = DesignWidget(factors={"A": (-1, 1), "B": (-1, 1)}, power_mode="r2")
        w._on_run_clicked(None)
        assert w.get_result() is not None

    @patch("iopt_power_design.widgets.find_optimal_design")
    def test_run_populates_design_df(self, mock_run):
        mock_run.return_value = _minimal_result()
        w = DesignWidget(factors={"A": (-1, 1), "B": (-1, 1)}, power_mode="r2")
        w._on_run_clicked(None)
        assert w.get_design_df() is not None
        assert isinstance(w.get_design_df(), pd.DataFrame)

    @patch("iopt_power_design.widgets.find_optimal_design")
    def test_run_populates_report(self, mock_run):
        mock_run.return_value = _minimal_result()
        w = DesignWidget(factors={"A": (-1, 1), "B": (-1, 1)}, power_mode="r2")
        w._on_run_clicked(None)
        assert w.get_report() is not None
        assert "n" in w.get_report()

    @patch("iopt_power_design.widgets.find_optimal_design")
    def test_run_error_leaves_result_none(self, mock_run):
        mock_run.side_effect = ValueError("bad formula")
        w = DesignWidget(factors={"A": (-1, 1)}, power_mode="r2")
        w._on_run_clicked(None)
        assert w.get_result() is None

    @patch("iopt_power_design.widgets.find_optimal_design")
    def test_run_error_shown_in_status(self, mock_run):
        mock_run.side_effect = RuntimeError("oops")
        w = DesignWidget(factors={"A": (-1, 1)}, power_mode="r2")
        w._on_run_clicked(None)
        assert "Error" in w._status_html.value or "oops" in w._status_html.value

    def test_run_with_no_factors_shows_validation_error(self):
        w = DesignWidget(factors={})
        w._factor_rows.clear()
        w._factor_table_box.children = ()
        w._on_run_clicked(None)
        assert w.get_result() is None
        assert "factor" in w._status_html.value.lower()

    @patch("iopt_power_design.widgets.find_optimal_design")
    def test_run_re_enables_button_after_success(self, mock_run):
        mock_run.return_value = _minimal_result()
        w = DesignWidget(factors={"A": (-1, 1)}, power_mode="r2")
        w._on_run_clicked(None)
        assert w._run_btn.disabled is False

    @patch("iopt_power_design.widgets.find_optimal_design")
    def test_run_re_enables_button_after_error(self, mock_run):
        mock_run.side_effect = RuntimeError("fail")
        w = DesignWidget(factors={"A": (-1, 1)}, power_mode="r2")
        w._on_run_clicked(None)
        assert w._run_btn.disabled is False

    @patch("iopt_power_design.widgets.find_optimal_design")
    def test_extra_do_kwargs_forwarded_to_api(self, mock_run):
        """Non-exposed DesignOptions fields (n_blocks) are passed to the API."""
        mock_run.return_value = _minimal_result()
        opts = DesignOptions(n_blocks=3)
        w = DesignWidget(factors={"A": (-1, 1)}, design_opts=opts)
        w._on_run_clicked(None)
        _, call_kwargs = mock_run.call_args
        assert call_kwargs["design_opts"].n_blocks == 3


# ---------------------------------------------------------------------------
# Layer 3 — Power curve rendering (mock plotly)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_WIDGETS, reason="ipywidgets not installed")
class TestPowerCurveRendering:
    @patch("iopt_power_design.widgets.find_optimal_design")
    @patch("iopt_power_design.widgets._HAS_PLOTLY", True)
    def test_render_does_not_raise_with_plotly(self, mock_run):
        mock_run.return_value = _minimal_result()
        w = DesignWidget(factors={"A": (-1, 1)}, power_mode="r2")
        # Should complete without exception even if plotly rendering is mocked
        w._on_run_clicked(None)
        assert w.get_result() is not None

    @patch("iopt_power_design.widgets.find_optimal_design")
    @patch("iopt_power_design.widgets._HAS_PLOTLY", False)
    def test_render_does_not_raise_without_plotly(self, mock_run):
        mock_run.return_value = _minimal_result()
        w = DesignWidget(factors={"A": (-1, 1)}, power_mode="r2")
        w._on_run_clicked(None)
        assert w.get_result() is not None


# ---------------------------------------------------------------------------
# Public API surface check
# ---------------------------------------------------------------------------

class TestPublicAPIExports:
    def test_WidgetsError_importable_from_package(self):
        from iopt_power_design import WidgetsError as WE
        assert issubclass(WE, RuntimeError)

    def test_DesignWidget_importable_from_package(self):
        from iopt_power_design import DesignWidget as DW
        assert DW is DesignWidget

    def test_design_widget_importable_from_package(self):
        from iopt_power_design import design_widget as dw
        assert callable(dw)
