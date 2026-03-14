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
    PowerGLMContrastConfig,
    glm_fisher_weight,
    SplitPlotOptions,
    ResponseSpec,
    MultiResponseOptions,
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


# ---------------------------------------------------------------------------
# ResponseSpec
# ---------------------------------------------------------------------------

def _contrast_cfg(**kw):
    defaults = dict(L=[[0, 1]], delta=[0.5])
    defaults.update(kw)
    return PowerContrastConfig(**defaults)


def _r2_cfg(**kw):
    defaults = dict(r2_target=0.5)
    defaults.update(kw)
    return PowerR2Config(**defaults)


class TestResponseSpec:
    def test_valid_contrast_mode(self):
        rs = ResponseSpec(name="Y1", power_cfg=_contrast_cfg())
        assert rs.name == "Y1"
        assert rs.formula is None
        assert rs.weight == 1.0

    def test_valid_r2_mode(self):
        rs = ResponseSpec(name="Y2", power_cfg=_r2_cfg())
        assert rs.name == "Y2"

    def test_formula_none_default(self):
        rs = ResponseSpec(name="Y1", power_cfg=_contrast_cfg())
        assert rs.formula is None

    def test_formula_string_accepted(self):
        rs = ResponseSpec(name="Y1", power_cfg=_contrast_cfg(), formula="~ 1 + A + B")
        assert rs.formula == "~ 1 + A + B"

    def test_weight_custom(self):
        rs = ResponseSpec(name="Y1", power_cfg=_contrast_cfg(), weight=2.5)
        assert rs.weight == 2.5

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ResponseSpec(name="", power_cfg=_contrast_cfg())

    def test_wrong_power_cfg_type_raises(self):
        with pytest.raises(TypeError, match="PowerContrastConfig"):
            ResponseSpec(name="Y1", power_cfg="not_a_config")

    def test_weight_zero_raises(self):
        with pytest.raises(ValueError, match="weight must be > 0"):
            ResponseSpec(name="Y1", power_cfg=_contrast_cfg(), weight=0.0)

    def test_weight_negative_raises(self):
        with pytest.raises(ValueError, match="weight must be > 0"):
            ResponseSpec(name="Y1", power_cfg=_contrast_cfg(), weight=-1.0)

    def test_dataclasses_replace_works(self):
        rs = ResponseSpec(name="Y1", power_cfg=_contrast_cfg())
        rs2 = dataclasses.replace(rs, name="Y2")
        assert rs2.name == "Y2"
        assert rs2.power_cfg is rs.power_cfg

    def test_all_exports_response_spec(self):
        import iopt_power_design.config as cfg_module
        assert "ResponseSpec" in cfg_module.__all__


# ---------------------------------------------------------------------------
# MultiResponseOptions
# ---------------------------------------------------------------------------

class TestMultiResponseOptions:
    def _two_contrast(self, **kw):
        r1 = ResponseSpec(name="Y1", power_cfg=_contrast_cfg())
        r2 = ResponseSpec(name="Y2", power_cfg=_contrast_cfg())
        return MultiResponseOptions(responses=[r1, r2], **kw)

    def test_valid_two_contrast_responses(self):
        mro = self._two_contrast()
        assert len(mro.responses) == 2
        assert mro.power_combination == "min"
        assert mro.sigma_joint is None

    def test_valid_mixed_contrast_r2(self):
        r1 = ResponseSpec(name="Y1", power_cfg=_contrast_cfg())
        r2 = ResponseSpec(name="Y2", power_cfg=_r2_cfg())
        mro = MultiResponseOptions(responses=[r1, r2])
        assert len(mro.responses) == 2

    def test_power_combination_min(self):
        mro = self._two_contrast(power_combination="min")
        assert mro.power_combination == "min"

    def test_power_combination_product(self):
        mro = self._two_contrast(power_combination="product")
        assert mro.power_combination == "product"

    def test_power_combination_weighted_mean(self):
        mro = self._two_contrast(power_combination="weighted_mean")
        assert mro.power_combination == "weighted_mean"

    def test_sigma_joint_none_default(self):
        mro = self._two_contrast()
        assert mro.sigma_joint is None

    def test_sigma_joint_valid_shape(self):
        sigma = np.eye(2)
        mro = self._two_contrast(sigma_joint=sigma)
        assert mro.sigma_joint is not None

    def test_sigma_joint_wrong_shape_raises(self):
        sigma = np.eye(3)  # wrong: 3x3 for 2 responses
        with pytest.raises(ValueError, match="sigma_joint must be"):
            self._two_contrast(sigma_joint=sigma)

    def test_fewer_than_two_responses_raises(self):
        r1 = ResponseSpec(name="Y1", power_cfg=_contrast_cfg())
        with pytest.raises(ValueError, match="at least 2"):
            MultiResponseOptions(responses=[r1])

    def test_duplicate_names_raises(self):
        r1 = ResponseSpec(name="Y1", power_cfg=_contrast_cfg())
        r2 = ResponseSpec(name="Y1", power_cfg=_contrast_cfg())
        with pytest.raises(ValueError, match="unique"):
            MultiResponseOptions(responses=[r1, r2])

    def test_invalid_power_combination_raises(self):
        with pytest.raises(ValueError, match="power_combination"):
            self._two_contrast(power_combination="bad_rule")

    def test_dataclasses_replace_power_combination(self):
        mro = self._two_contrast()
        mro2 = dataclasses.replace(mro, power_combination="product")
        assert mro2.power_combination == "product"

    def test_responses_stored_in_insertion_order(self):
        names = ["Alpha", "Beta", "Gamma"]
        responses = [ResponseSpec(name=n, power_cfg=_contrast_cfg()) for n in names]
        mro = MultiResponseOptions(responses=responses)
        assert [r.name for r in mro.responses] == names

    def test_all_exports_multi_response_options(self):
        import iopt_power_design.config as cfg_module
        assert "MultiResponseOptions" in cfg_module.__all__


# ---------------------------------------------------------------------------
# MR-10 — Additional ResponseSpec and MultiResponseOptions edge-case tests
# ---------------------------------------------------------------------------


def _r2_cfg_simple():
    return PowerR2Config(r2_target=0.3)


def _contrast_cfg_simple():
    return PowerContrastConfig(L=np.array([[0, 1]]), delta=np.array([0.5]))


class TestMR10ResponseSpecEdgeCases:
    """MR-10: edge cases for ResponseSpec not covered by TestResponseSpec."""

    def test_r2_config_with_nondefault_weight(self):
        """R2Config ResponseSpec stores non-default weight correctly."""
        r = ResponseSpec(name="Y1", power_cfg=_r2_cfg_simple(), weight=3.5)
        assert r.weight == pytest.approx(3.5)

    def test_contrast_config_weight_very_small_positive(self):
        """Very small positive weight (1e-10) is accepted."""
        r = ResponseSpec(name="Y1", power_cfg=_contrast_cfg_simple(), weight=1e-10)
        assert r.weight > 0

    def test_formula_override_stored_separately_from_global(self):
        """Per-response formula is independent of other ResponseSpec instances."""
        r1 = ResponseSpec("Y1", _contrast_cfg_simple(), formula="~ 1 + A")
        r2 = ResponseSpec("Y2", _contrast_cfg_simple())
        assert r1.formula == "~ 1 + A"
        assert r2.formula is None

    def test_dataclasses_replace_weight(self):
        """dataclasses.replace updates weight without touching other fields."""
        r = ResponseSpec("Y1", _contrast_cfg_simple(), weight=1.0)
        r2 = dataclasses.replace(r, weight=5.0)
        assert r2.weight == pytest.approx(5.0)
        assert r2.name == "Y1"

    def test_dataclasses_replace_formula(self):
        """dataclasses.replace updates formula independently."""
        r = ResponseSpec("Y1", _contrast_cfg_simple())
        r2 = dataclasses.replace(r, formula="~ 1 + A + B")
        assert r2.formula == "~ 1 + A + B"
        assert r.formula is None  # original unchanged

    def test_power_cfg_r2_type_stored(self):
        """R2Config stored as PowerR2Config type."""
        r = ResponseSpec("Y1", _r2_cfg_simple())
        assert isinstance(r.power_cfg, PowerR2Config)

    def test_power_cfg_contrast_type_stored(self):
        """ContrastConfig stored as PowerContrastConfig type."""
        r = ResponseSpec("Y1", _contrast_cfg_simple())
        assert isinstance(r.power_cfg, PowerContrastConfig)


class TestMR10MultiResponseOptionsEdgeCases:
    """MR-10: edge cases for MultiResponseOptions not covered by TestMultiResponseOptions."""

    def _two(self, rule="min", **kw):
        r1 = ResponseSpec("Y1", _contrast_cfg_simple())
        r2 = ResponseSpec("Y2", _contrast_cfg_simple())
        return MultiResponseOptions([r1, r2], power_combination=rule, **kw)

    def test_three_responses_accepted(self):
        """Three responses are accepted."""
        responses = [ResponseSpec(f"Y{i}", _r2_cfg_simple()) for i in range(1, 4)]
        mro = MultiResponseOptions(responses)
        assert len(mro.responses) == 3

    def test_five_responses_accepted(self):
        """Five responses are accepted (no upper bound on k)."""
        responses = [ResponseSpec(f"Y{i}", _r2_cfg_simple()) for i in range(1, 6)]
        mro = MultiResponseOptions(responses)
        assert len(mro.responses) == 5

    def test_sigma_joint_stored_as_ndarray(self):
        """sigma_joint is stored as a numpy ndarray, not a list."""
        sigma = np.eye(2)
        mro = self._two(sigma_joint=sigma)
        assert isinstance(mro.sigma_joint, np.ndarray)

    def test_sigma_joint_three_responses_3x3_accepted(self):
        """3x3 sigma_joint is accepted for 3 responses."""
        responses = [ResponseSpec(f"Y{i}", _contrast_cfg_simple()) for i in range(1, 4)]
        sigma = np.eye(3)
        mro = MultiResponseOptions(responses, sigma_joint=sigma)
        assert mro.sigma_joint.shape == (3, 3)

    def test_sigma_joint_wrong_k_raises_shape_error(self):
        """2x2 sigma_joint for 3 responses raises ValueError."""
        responses = [ResponseSpec(f"Y{i}", _contrast_cfg_simple()) for i in range(1, 4)]
        with pytest.raises(ValueError, match="sigma_joint"):
            MultiResponseOptions(responses, sigma_joint=np.eye(2))

    def test_weights_on_individual_responses_stored(self):
        """Weights set on individual ResponseSpec objects are preserved in responses list."""
        r1 = ResponseSpec("Y1", _contrast_cfg_simple(), weight=3.0)
        r2 = ResponseSpec("Y2", _contrast_cfg_simple(), weight=1.0)
        mro = MultiResponseOptions([r1, r2])
        assert mro.responses[0].weight == pytest.approx(3.0)
        assert mro.responses[1].weight == pytest.approx(1.0)

    def test_mixed_r2_contrast_three_responses(self):
        """Mix of R2 and contrast responses is accepted for 3 responses."""
        r1 = ResponseSpec("Y1", _contrast_cfg_simple())
        r2 = ResponseSpec("Y2", _r2_cfg_simple())
        r3 = ResponseSpec("Y3", _contrast_cfg_simple())
        mro = MultiResponseOptions([r1, r2, r3])
        assert len(mro.responses) == 3

    def test_default_power_combination_is_min(self):
        """Default power_combination is 'min'."""
        r1 = ResponseSpec("Y1", _r2_cfg_simple())
        r2 = ResponseSpec("Y2", _r2_cfg_simple())
        mro = MultiResponseOptions([r1, r2])
        assert mro.power_combination == "min"

    def test_responses_length_matches_input(self):
        """len(mro.responses) matches the number of responses passed."""
        responses = [ResponseSpec(f"Y{i}", _r2_cfg_simple()) for i in range(1, 5)]
        mro = MultiResponseOptions(responses)
        assert len(mro.responses) == 4


# ---------------------------------------------------------------------------
# GL-1: PowerGLMContrastConfig and glm_fisher_weight
# ---------------------------------------------------------------------------

def _glm_binomial_cfg(**kwargs):
    """Minimal valid binomial GLM config."""
    defaults = dict(L=[[0, 1]], delta=[0.5], baseline=0.20, family="binomial")
    defaults.update(kwargs)
    return PowerGLMContrastConfig(**defaults)


def _glm_poisson_cfg(**kwargs):
    """Minimal valid Poisson GLM config."""
    defaults = dict(L=[[0, 1]], delta=[0.3], baseline=2.5, family="poisson")
    defaults.update(kwargs)
    return PowerGLMContrastConfig(**defaults)


class TestPowerGLMContrastConfig:
    # ------------------------------------------------------------------ #
    # Construction and defaults
    # ------------------------------------------------------------------ #
    def test_valid_binomial_construction(self):
        cfg = _glm_binomial_cfg()
        assert cfg.family == "binomial"
        assert cfg.link == "logit"          # canonical link resolved
        assert cfg.baseline == pytest.approx(0.20)
        assert cfg.L.shape == (1, 2)
        assert cfg.delta.shape == (1,)

    def test_valid_poisson_construction(self):
        cfg = _glm_poisson_cfg()
        assert cfg.family == "poisson"
        assert cfg.link == "log"
        assert cfg.baseline == pytest.approx(2.5)

    def test_default_family_is_binomial(self):
        cfg = PowerGLMContrastConfig(L=[[0, 1]], delta=[0.5], baseline=0.3)
        assert cfg.family == "binomial"

    def test_link_none_resolves_to_canonical_logit(self):
        cfg = PowerGLMContrastConfig(L=[[0, 1]], delta=[0.5], baseline=0.3,
                                      family="binomial", link=None)
        assert cfg.link == "logit"

    def test_link_none_resolves_to_canonical_log_for_poisson(self):
        cfg = PowerGLMContrastConfig(L=[[0, 1]], delta=[0.3], baseline=1.0,
                                      family="poisson", link=None)
        assert cfg.link == "log"

    def test_explicit_logit_link_accepted(self):
        cfg = _glm_binomial_cfg(link="logit")
        assert cfg.link == "logit"

    def test_explicit_log_link_accepted_for_poisson(self):
        cfg = _glm_poisson_cfg(link="log")
        assert cfg.link == "log"

    def test_alpha_default(self):
        assert _glm_binomial_cfg().alpha == pytest.approx(0.05)

    def test_power_default(self):
        assert _glm_binomial_cfg().power == pytest.approx(0.80)

    def test_max_n_default(self):
        assert _glm_binomial_cfg().max_n == 2000

    def test_max_iter_default(self):
        assert _glm_binomial_cfg().max_iter == 200

    def test_tol_power_default(self):
        assert _glm_binomial_cfg().tol_power == pytest.approx(1e-3)

    def test_multirow_L_accepted(self):
        cfg = PowerGLMContrastConfig(
            L=[[0, 1, 0], [0, 0, 1]], delta=[0.5, 0.4], baseline=0.3)
        assert cfg.L.shape == (2, 3)
        assert cfg.delta.shape == (2,)

    # ------------------------------------------------------------------ #
    # Family / link validation
    # ------------------------------------------------------------------ #
    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="family"):
            PowerGLMContrastConfig(L=[[0, 1]], delta=[0.5], baseline=0.3,
                                   family="gaussian")

    def test_log_link_rejected_for_binomial(self):
        with pytest.raises(ValueError, match="link"):
            PowerGLMContrastConfig(L=[[0, 1]], delta=[0.5], baseline=0.3,
                                   family="binomial", link="log")

    def test_logit_link_rejected_for_poisson(self):
        with pytest.raises(ValueError, match="link"):
            PowerGLMContrastConfig(L=[[0, 1]], delta=[0.3], baseline=1.0,
                                   family="poisson", link="logit")

    # ------------------------------------------------------------------ #
    # Baseline validation
    # ------------------------------------------------------------------ #
    def test_baseline_zero_raises_for_binomial(self):
        with pytest.raises(ValueError, match="baseline"):
            _glm_binomial_cfg(baseline=0.0)

    def test_baseline_one_raises_for_binomial(self):
        with pytest.raises(ValueError, match="baseline"):
            _glm_binomial_cfg(baseline=1.0)

    def test_baseline_negative_raises_for_poisson(self):
        with pytest.raises(ValueError, match="baseline"):
            _glm_poisson_cfg(baseline=-1.0)

    def test_baseline_zero_raises_for_poisson(self):
        with pytest.raises(ValueError, match="baseline"):
            _glm_poisson_cfg(baseline=0.0)

    def test_extreme_baseline_emits_warning(self):
        with pytest.warns(RuntimeWarning, match="boundary"):
            _glm_binomial_cfg(baseline=0.03)

    def test_baseline_just_above_zero_ok_with_warning(self):
        # 0.01 is extreme → warning but no error
        with pytest.warns(RuntimeWarning):
            cfg = _glm_binomial_cfg(baseline=0.01)
        assert cfg.baseline == pytest.approx(0.01)

    # ------------------------------------------------------------------ #
    # Numeric range validation
    # ------------------------------------------------------------------ #
    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            _glm_binomial_cfg(alpha=1.5)

    def test_power_out_of_range_raises(self):
        with pytest.raises(ValueError, match="power"):
            _glm_binomial_cfg(power=0.0)

    def test_tol_power_nonpositive_raises(self):
        with pytest.raises(ValueError, match="tol_power"):
            _glm_binomial_cfg(tol_power=0.0)

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            _glm_binomial_cfg(max_iter=0)

    def test_max_n_zero_raises(self):
        with pytest.raises(ValueError, match="max_n"):
            _glm_binomial_cfg(max_n=0)

    # ------------------------------------------------------------------ #
    # Contrast content validation
    # ------------------------------------------------------------------ #
    def test_all_zero_L_row_raises(self):
        with pytest.raises(ValueError, match="all-zero"):
            PowerGLMContrastConfig(L=[[0, 0]], delta=[0.5], baseline=0.3)

    def test_zero_delta_raises(self):
        with pytest.raises(ValueError, match="zero"):
            _glm_binomial_cfg(delta=[0.0])

    # ------------------------------------------------------------------ #
    # Dataclasses interop
    # ------------------------------------------------------------------ #
    def test_dataclasses_replace_safe(self):
        cfg = _glm_binomial_cfg()
        cfg2 = dataclasses.replace(cfg, alpha=0.01)
        assert cfg2.alpha == pytest.approx(0.01)
        assert cfg2.baseline == pytest.approx(cfg.baseline)

    def test_str_representation_contains_family(self):
        cfg = _glm_binomial_cfg()
        assert "binomial" in str(cfg)
        assert "logit" in str(cfg)

    # ------------------------------------------------------------------ #
    # Integration with ResponseSpec and union type
    # ------------------------------------------------------------------ #
    def test_response_spec_accepts_glm_config(self):
        cfg = _glm_binomial_cfg()
        spec = ResponseSpec("Y1", cfg)
        assert spec.power_cfg is cfg

    def test_response_spec_accepts_poisson_config(self):
        cfg = _glm_poisson_cfg()
        spec = ResponseSpec("Count", cfg)
        assert spec.power_cfg is cfg

    def test_response_spec_rejects_plain_object(self):
        with pytest.raises(TypeError):
            ResponseSpec("Y", object())

    def test_glm_config_is_instance_of_expected_types(self):
        cfg = _glm_binomial_cfg()
        # Not an OLS config
        assert not isinstance(cfg, PowerContrastConfig)
        assert not isinstance(cfg, PowerR2Config)
        assert isinstance(cfg, PowerGLMContrastConfig)

    def test_public_export_accessible(self):
        import iopt_power_design as pkg
        assert hasattr(pkg, "PowerGLMContrastConfig")
        assert pkg.PowerGLMContrastConfig is PowerGLMContrastConfig


class TestGLMFisherWeight:
    def test_binomial_half_gives_quarter(self):
        cfg = _glm_binomial_cfg(baseline=0.5)
        assert glm_fisher_weight(cfg) == pytest.approx(0.25)

    def test_binomial_p0_formula(self):
        p0 = 0.20
        cfg = _glm_binomial_cfg(baseline=p0)
        assert glm_fisher_weight(cfg) == pytest.approx(p0 * (1 - p0))

    def test_binomial_asymmetric_baseline(self):
        p0 = 0.1
        cfg = _glm_binomial_cfg(baseline=p0)
        assert glm_fisher_weight(cfg) == pytest.approx(0.09)

    def test_binomial_weight_maximised_at_half(self):
        w_half = glm_fisher_weight(_glm_binomial_cfg(baseline=0.5))
        w_low  = glm_fisher_weight(_glm_binomial_cfg(baseline=0.2))
        w_high = glm_fisher_weight(_glm_binomial_cfg(baseline=0.8))
        assert w_half > w_low
        assert w_half > w_high

    def test_poisson_weight_equals_baseline(self):
        mu0 = 3.7
        cfg = _glm_poisson_cfg(baseline=mu0)
        assert glm_fisher_weight(cfg) == pytest.approx(mu0)

    def test_poisson_weight_increases_with_rate(self):
        w1 = glm_fisher_weight(_glm_poisson_cfg(baseline=1.0))
        w5 = glm_fisher_weight(_glm_poisson_cfg(baseline=5.0))
        assert w5 > w1

    def test_weight_is_strictly_positive_binomial(self):
        assert glm_fisher_weight(_glm_binomial_cfg(baseline=0.3)) > 0

    def test_weight_is_strictly_positive_poisson(self):
        assert glm_fisher_weight(_glm_poisson_cfg(baseline=0.5)) > 0

    def test_public_export_accessible(self):
        import iopt_power_design as pkg
        assert hasattr(pkg, "glm_fisher_weight")
        assert pkg.glm_fisher_weight is glm_fisher_weight
