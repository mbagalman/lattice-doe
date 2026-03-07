# tests/test_config.py
"""Unit tests for config.py dataclasses."""
import numpy as np
import pandas as pd
import pytest

from iopt_power_design.config import (
    DesignOptions,
    PowerContrastConfig,
    PowerR2Config,
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
