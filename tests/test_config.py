# tests/test_config.py
"""Unit tests for config.py dataclasses."""
import numpy as np
import pandas as pd
import pytest

import dataclasses
import warnings

from iopt_power_design.config import (
    DesignOptions,
    PowerContrastConfig,
    PowerR2Config,
    SplitPlotOptions,
    _compile_constraint_expr,
)


# ---------------------------------------------------------------------------
# PowerContrastConfig
# ---------------------------------------------------------------------------

class TestPowerContrastConfig:
    def test_valid_construction(self):
        cfg = PowerContrastConfig(L=[[0, 1]], delta=[0.5])
        assert cfg.L.shape == (1, 2)
        assert cfg.delta.shape == (1,)
        assert cfg.alpha == 0.05
        assert cfg.power == 0.8
        assert cfg.sigma == 1.0

    def test_1d_L_promoted_to_2d(self):
        cfg = PowerContrastConfig(L=[0, 1], delta=[0.5])
        assert cfg.L.ndim == 2
        assert cfg.L.shape == (1, 2)

    def test_delta_promoted_to_1d(self):
        cfg = PowerContrastConfig(L=[[0, 1]], delta=0.5)
        assert cfg.delta.ndim == 1

    def test_raises_on_all_zero_L_row(self):
        with pytest.raises(ValueError, match="all-zero row"):
            PowerContrastConfig(L=[[0, 0]], delta=[0.5])

    def test_raises_on_zero_delta(self):
        with pytest.raises(ValueError, match="zero"):
            PowerContrastConfig(L=[[0, 1]], delta=[0.0])

    def test_raises_on_L_delta_shape_mismatch(self):
        with pytest.raises(ValueError):
            PowerContrastConfig(L=[[0, 1], [1, 0]], delta=[0.5])  # 2 rows, 1 delta

    def test_raises_on_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            PowerContrastConfig(L=[[0, 1]], delta=[0.5], alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            PowerContrastConfig(L=[[0, 1]], delta=[0.5], alpha=0.0)

    def test_raises_on_power_out_of_range(self):
        with pytest.raises(ValueError, match="power"):
            PowerContrastConfig(L=[[0, 1]], delta=[0.5], power=0.0)
        with pytest.raises(ValueError, match="power"):
            PowerContrastConfig(L=[[0, 1]], delta=[0.5], power=1.0)

    def test_raises_on_non_positive_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            PowerContrastConfig(L=[[0, 1]], delta=[0.5], sigma=0.0)
        with pytest.raises(ValueError, match="sigma"):
            PowerContrastConfig(L=[[0, 1]], delta=[0.5], sigma=-1.0)

    def test_raises_on_non_positive_max_n(self):
        with pytest.raises(ValueError, match="max_n"):
            PowerContrastConfig(L=[[0, 1]], delta=[0.5], max_n=0)

    def test_str_representation(self):
        cfg = PowerContrastConfig(L=[[0, 1]], delta=[0.5])
        s = str(cfg)
        assert "PowerContrastConfig" in s
        assert "alpha" in s


# ---------------------------------------------------------------------------
# PowerR2Config
# ---------------------------------------------------------------------------

class TestPowerR2Config:
    def test_valid_construction(self):
        cfg = PowerR2Config(r2_target=0.2)
        assert cfg.r2_target == 0.2
        assert cfg.alpha == 0.05
        assert cfg.power == 0.8
        assert cfg.lambda_mode == "n"

    def test_valid_lambda_modes(self):
        PowerR2Config(r2_target=0.2, lambda_mode="n")
        PowerR2Config(r2_target=0.2, lambda_mode="n_minus_p")

    def test_raises_on_r2_out_of_range(self):
        with pytest.raises(ValueError):
            PowerR2Config(r2_target=0.0)
        with pytest.raises(ValueError):
            PowerR2Config(r2_target=1.0)
        with pytest.raises(ValueError):
            PowerR2Config(r2_target=1.5)

    def test_raises_on_invalid_lambda_mode(self):
        with pytest.raises(ValueError, match="lambda_mode"):
            PowerR2Config(r2_target=0.2, lambda_mode="noncentral")

    def test_raises_on_alpha_out_of_range(self):
        with pytest.raises(ValueError):
            PowerR2Config(r2_target=0.2, alpha=0.0)

    def test_raises_on_non_positive_max_n(self):
        with pytest.raises(ValueError):
            PowerR2Config(r2_target=0.2, max_n=-5)

    def test_str_representation(self):
        cfg = PowerR2Config(r2_target=0.3)
        s = str(cfg)
        assert "PowerR2Config" in s
        assert "r2_target" in s


# ---------------------------------------------------------------------------
# DesignOptions
# ---------------------------------------------------------------------------

class TestDesignOptions:
    def test_defaults(self):
        opts = DesignOptions()
        assert opts.algo == "fedorov"
        assert opts.starts == 5
        assert opts.xtx_jitter > 0
        assert opts.workers is None
        assert opts.auto_candidate is False

    def test_raises_on_invalid_algo(self):
        with pytest.raises(ValueError, match="algo"):
            DesignOptions(algo="detmax")

    def test_raises_on_zero_candidate_points(self):
        with pytest.raises(ValueError, match="candidate_points"):
            DesignOptions(candidate_points=0)

    def test_raises_on_cand_max_less_than_min(self):
        with pytest.raises(ValueError):
            DesignOptions(cand_min=500, cand_max=100)

    def test_raises_on_growth_factor_leq_one(self):
        with pytest.raises(ValueError, match="growth_factor"):
            DesignOptions(growth_factor=1.0)

    def test_raises_on_non_positive_jitter(self):
        with pytest.raises(ValueError, match="xtx_jitter"):
            DesignOptions(xtx_jitter=0.0)

    def test_zero_workers_treated_as_serial(self):
        opts = DesignOptions(workers=0)
        assert opts.workers is None

    def test_negative_workers_treated_as_serial(self):
        opts = DesignOptions(workers=-1)
        assert opts.workers is None

    def test_positive_workers_preserved(self):
        opts = DesignOptions(workers=4)
        assert opts.workers == 4

    def test_str_representation(self):
        opts = DesignOptions()
        s = str(opts)
        assert "DesignOptions" in s
        assert "algo" in s

    def test_default_criterion_is_I(self):
        opts = DesignOptions()
        assert opts.criterion == "I"

    def test_d_criterion_accepted(self):
        opts = DesignOptions(criterion="D")
        assert opts.criterion == "D"

    def test_a_criterion_accepted(self):
        opts = DesignOptions(criterion="A")
        assert opts.criterion == "A"

    def test_raises_on_invalid_criterion(self):
        with pytest.raises(ValueError, match="criterion"):
            DesignOptions(criterion="E")


# ---------------------------------------------------------------------------
# Enhancement 12 — Declarative constraint expressions
# ---------------------------------------------------------------------------

class TestCompileConstraintExpr:
    """Unit tests for the _compile_constraint_expr helper."""

    def _row(self, **kwargs) -> pd.Series:
        """Build a one-row pandas Series from keyword arguments."""
        return pd.Series(kwargs)

    # --- Compilation ---
    def test_compiles_without_error(self):
        fn = _compile_constraint_expr("A <= 10")
        assert callable(fn)

    def test_syntax_error_raises_value_error(self):
        with pytest.raises(ValueError, match="syntax error"):
            _compile_constraint_expr("A <=> 10")

    def test_syntax_error_message_contains_expression(self):
        with pytest.raises(ValueError, match="constraint_expr"):
            _compile_constraint_expr("A <=> 10")

    # --- Evaluation: simple comparisons ---
    def test_simple_le_passes(self):
        fn = _compile_constraint_expr("Temperature <= 100")
        assert fn(self._row(Temperature=50.0)) is True

    def test_simple_le_fails(self):
        fn = _compile_constraint_expr("Temperature <= 100")
        assert fn(self._row(Temperature=150.0)) is False

    # --- Evaluation: compound expressions ---
    def test_compound_and_passes(self):
        fn = _compile_constraint_expr("A <= 2 * B and C > 0")
        assert fn(self._row(A=4.0, B=3.0, C=1.0)) is True

    def test_compound_and_fails(self):
        fn = _compile_constraint_expr("A <= 2 * B and C > 0")
        assert fn(self._row(A=10.0, B=3.0, C=1.0)) is False

    def test_compound_or(self):
        fn = _compile_constraint_expr("A > 5 or B > 5")
        assert fn(self._row(A=1.0, B=10.0)) is True
        assert fn(self._row(A=1.0, B=1.0)) is False

    # --- Evaluation: math functions ---
    def test_sqrt_function(self):
        fn = _compile_constraint_expr("sqrt(A) <= 3")
        assert fn(self._row(A=4.0)) is True   # sqrt(4)=2 <= 3
        assert fn(self._row(A=16.0)) is False  # sqrt(16)=4 > 3

    def test_abs_function(self):
        fn = _compile_constraint_expr("abs(A - B) <= 5")
        assert fn(self._row(A=3.0, B=7.0)) is True
        assert fn(self._row(A=0.0, B=10.0)) is False

    # --- Evaluation: undefined name error ---
    def test_undefined_column_raises_value_error(self):
        fn = _compile_constraint_expr("NonExistent <= 10")
        with pytest.raises(ValueError, match="undefined"):
            fn(self._row(A=5.0))

    # --- DesignOptions integration ---
    def test_design_opts_constraint_expr_sets_func(self):
        opts = DesignOptions(constraint_expr="A <= 10")
        assert opts.constraint_func is not None
        assert callable(opts.constraint_func)

    def test_design_opts_constraint_expr_evaluates(self):
        opts = DesignOptions(constraint_expr="Temperature <= 100")
        assert opts.constraint_func(pd.Series({"Temperature": 50.0})) is True
        assert opts.constraint_func(pd.Series({"Temperature": 150.0})) is False

    def test_design_opts_expr_and_func_expr_wins(self):
        """When both constraint_expr and constraint_func are provided, expr wins."""
        always_false = lambda row: False
        opts = DesignOptions(
            constraint_expr="A <= 100",
            constraint_func=always_false,
        )
        # The compiled expr says True for A=50; if func had won, it would be False
        assert opts.constraint_func(pd.Series({"A": 50.0})) is True

    def test_design_opts_no_constraint(self):
        opts = DesignOptions()
        assert opts.constraint_func is None
        assert opts.constraint_expr is None

    def test_design_opts_constraint_expr_in_str(self):
        opts = DesignOptions(constraint_expr="A <= 5")
        assert "expr=" in str(opts)

    def test_dataclasses_replace_preserves_expr(self):
        """dataclasses.replace should keep constraint_expr and recompile correctly."""
        import dataclasses
        opts = DesignOptions(constraint_expr="A <= 10", criterion="I")
        new_opts = dataclasses.replace(opts, criterion="D")
        assert new_opts.constraint_expr == "A <= 10"
        assert callable(new_opts.constraint_func)
        assert new_opts.constraint_func(pd.Series({"A": 5.0})) is True


# ---------------------------------------------------------------------------
# Regression: _validate_config_keys raises cleanly on malformed contrast block
# ---------------------------------------------------------------------------

class TestValidateConfigKeysContrastType:
    """_validate_config_keys must raise KeyError with an actionable message
    when 'contrast' is present but is not a dict (issue #3)."""

    from iopt_power_design.cli import _validate_config_keys as _vcfg
    _vcfg = staticmethod(_vcfg)  # prevent Python from treating it as an unbound method

    _BASE = {
        "formula": "~ 1 + A",
        "factors": {"A": [0.0, 1.0]},
    }

    def _cfg(self, contrast_val):
        return {**self._BASE, "contrast": contrast_val}

    def test_contrast_as_string_raises_key_error(self):
        with pytest.raises(KeyError, match="mapping"):
            self._vcfg(self._cfg("linear"))

    def test_contrast_as_list_raises_key_error(self):
        with pytest.raises(KeyError, match="mapping"):
            self._vcfg(self._cfg(["scenario_a", "scenario_b"]))

    def test_contrast_as_int_raises_key_error(self):
        with pytest.raises(KeyError, match="mapping"):
            self._vcfg(self._cfg(42))

    def test_contrast_as_null_raises_key_error(self):
        # None is falsy; old code silently replaced it with {} and passed validation
        with pytest.raises(KeyError, match="mapping"):
            self._vcfg(self._cfg(None))

    def test_valid_contrast_scenario_passes(self):
        cfg = self._cfg({"scenario_a": {"A": 0}, "scenario_b": {"A": 1}, "sesoi": 0.5})
        self._vcfg(cfg)  # must not raise

    def test_valid_contrast_explicit_passes(self):
        cfg = self._cfg({"L": [[1, 0]], "delta": [0.5]})
        self._vcfg(cfg)  # must not raise


# ---------------------------------------------------------------------------
# SplitPlotOptions  (SP-1)
# ---------------------------------------------------------------------------

class TestSplitPlotOptions:
    """Unit tests for the SplitPlotOptions dataclass."""

    # ------------------------------------------------------------------
    # Valid construction
    # ------------------------------------------------------------------

    def test_minimal_valid(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=4)
        assert sp.htc_factors == ["A"]
        assert sp.n_whole_plots == 4
        assert sp.eta == 1.0
        assert sp.subplots_per_wp is None
        assert sp.df_method == "auto"
        assert sp.criterion_ignore_vr is False

    def test_full_explicit(self):
        sp = SplitPlotOptions(
            htc_factors=["A", "B"],
            n_whole_plots=6,
            eta=2.5,
            subplots_per_wp=4,
            df_method="conservative",
            criterion_ignore_vr=True,
        )
        assert sp.htc_factors == ["A", "B"]
        assert sp.n_whole_plots == 6
        assert sp.eta == 2.5
        assert sp.subplots_per_wp == 4
        assert sp.df_method == "conservative"
        assert sp.criterion_ignore_vr is True

    def test_eta_zero_allowed(self):
        # eta=0 is the OLS limiting case — must be valid.
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=2, eta=0.0)
        assert sp.eta == 0.0

    def test_subplots_per_wp_one_allowed(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=2, subplots_per_wp=1)
        assert sp.subplots_per_wp == 1

    def test_all_df_methods_accepted(self):
        for method in ("auto", "conservative", "sp_only"):
            sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=2, df_method=method)
            assert sp.df_method == method

    # ------------------------------------------------------------------
    # htc_factors validation
    # ------------------------------------------------------------------

    def test_empty_htc_factors_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SplitPlotOptions(htc_factors=[], n_whole_plots=4)

    def test_htc_factors_with_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            SplitPlotOptions(htc_factors=["A", ""], n_whole_plots=4)

    def test_htc_factors_duplicate_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            SplitPlotOptions(htc_factors=["A", "A"], n_whole_plots=4)

    # ------------------------------------------------------------------
    # n_whole_plots validation
    # ------------------------------------------------------------------

    def test_n_whole_plots_one_raises(self):
        with pytest.raises(ValueError, match="≥ 2"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=1)

    def test_n_whole_plots_zero_raises(self):
        with pytest.raises(ValueError, match="≥ 2"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=0)

    def test_n_whole_plots_negative_raises(self):
        with pytest.raises(ValueError, match="≥ 2"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=-3)

    def test_n_whole_plots_bool_raises(self):
        # bool is a subclass of int but should be rejected
        with pytest.raises(ValueError, match="integer"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=True)

    def test_n_whole_plots_float_raises(self):
        with pytest.raises(ValueError, match="integer"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=4.0)

    # ------------------------------------------------------------------
    # eta validation
    # ------------------------------------------------------------------

    def test_negative_eta_raises(self):
        with pytest.raises(ValueError, match="≥ 0"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=4, eta=-0.1)

    # ------------------------------------------------------------------
    # subplots_per_wp validation
    # ------------------------------------------------------------------

    def test_subplots_per_wp_zero_raises(self):
        with pytest.raises(ValueError, match="≥ 1"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=4, subplots_per_wp=0)

    def test_subplots_per_wp_negative_raises(self):
        with pytest.raises(ValueError, match="≥ 1"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=4, subplots_per_wp=-2)

    def test_subplots_per_wp_bool_raises(self):
        with pytest.raises(ValueError, match="integer or None"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=4, subplots_per_wp=True)

    # ------------------------------------------------------------------
    # df_method validation
    # ------------------------------------------------------------------

    def test_invalid_df_method_raises(self):
        with pytest.raises(ValueError, match="df_method"):
            SplitPlotOptions(htc_factors=["A"], n_whole_plots=4, df_method="kenward_roger")

    # ------------------------------------------------------------------
    # dataclasses.replace compatibility
    # ------------------------------------------------------------------

    def test_replace_preserves_unchanged_fields(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=4, eta=1.0)
        sp2 = dataclasses.replace(sp, eta=3.0)
        assert sp2.htc_factors == ["A"]
        assert sp2.n_whole_plots == 4
        assert sp2.eta == 3.0

    def test_replace_with_invalid_value_raises(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=4)
        with pytest.raises(ValueError):
            dataclasses.replace(sp, n_whole_plots=1)

    # ------------------------------------------------------------------
    # __str__
    # ------------------------------------------------------------------

    def test_str_contains_htc_factors(self):
        sp = SplitPlotOptions(htc_factors=["Temp", "Pressure"], n_whole_plots=6)
        s = str(sp)
        assert "Temp" in s
        assert "Pressure" in s
        assert "n_whole_plots=6" in s
        assert "auto" in s  # subplots_per_wp=auto

    def test_str_with_explicit_subplots(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=4, subplots_per_wp=3)
        assert "subplots_per_wp=3" in str(sp)


# ---------------------------------------------------------------------------
# DesignOptions.split_plot field  (SP-1)
# ---------------------------------------------------------------------------

class TestDesignOptionsSplitPlot:
    """Tests for the split_plot field wired into DesignOptions."""

    def test_default_is_none(self):
        do = DesignOptions()
        assert do.split_plot is None

    def test_split_plot_accepted(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=4)
        do = DesignOptions(split_plot=sp)
        assert do.split_plot is sp

    def test_replace_preserves_split_plot(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=4)
        do = DesignOptions(split_plot=sp)
        do2 = dataclasses.replace(do, starts=10)
        assert do2.split_plot is sp
        assert do2.starts == 10

    def test_replace_clears_split_plot(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=4)
        do = DesignOptions(split_plot=sp)
        do2 = dataclasses.replace(do, split_plot=None)
        assert do2.split_plot is None

    def test_split_plot_and_n_blocks_emits_warning(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DesignOptions(split_plot=sp, n_blocks=3)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "n_blocks" in str(w[0].message).lower() or "split_plot" in str(w[0].message).lower()

    def test_split_plot_without_n_blocks_no_warning(self):
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DesignOptions(split_plot=sp)
        assert len(w) == 0

    def test_split_plot_importable_from_top_level(self):
        import iopt_power_design
        assert hasattr(iopt_power_design, "SplitPlotOptions")
