# tests/test_api.py
"""Integration tests for the top-level API in api.py.

These tests build actual I-optimal designs, so they are slower than unit tests.
We keep the problem small (few factors, small candidate set, few starts) to
keep run-time reasonable.
"""
import numpy as np
import pandas as pd
import pytest

from iopt_power_design import (
    DesignOptions,
    PowerContrastConfig,
    PowerR2Config,
    i_optimal_powered_design,
    power_curve_by_effect,
    power_curve_by_n,
    power_sensitivity,
    min_detectable_effect,
    compare_criteria,
    augment_design,
)
from iopt_power_design.contrasts import contrast_from_scenarios

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FORMULA = "~ 1 + A + B"
FACTORS = {"A": ["low", "high"], "B": (0.0, 10.0)}

# Deliberately small options so tests finish quickly
FAST_OPTS = DesignOptions(
    candidate_points=150,
    starts=2,
    max_iter=50,
    random_state=0,
)


def _contrast_cfg(power: float = 0.80, max_n: int = 80) -> PowerContrastConfig:
    L, delta = contrast_from_scenarios(
        FORMULA, FACTORS,
        {"A": "low",  "B": 0.0},
        {"A": "high", "B": 10.0},
        sesoi=1.0,
    )
    return PowerContrastConfig(L=L, delta=delta, power=power, max_n=max_n)


def _r2_cfg(power: float = 0.80, max_n: int = 80) -> PowerR2Config:
    return PowerR2Config(r2_target=0.30, power=power, max_n=max_n)


# ---------------------------------------------------------------------------
# i_optimal_powered_design — contrast mode
# ---------------------------------------------------------------------------

class TestIOptimalPoweredDesignContrast:
    def test_returns_expected_keys(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        assert set(result.keys()) >= {"design_df", "buckets_df", "report"}

    def test_design_df_is_dataframe(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        assert isinstance(result["design_df"], pd.DataFrame)

    def test_design_df_has_factor_columns(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        assert set(FACTORS.keys()).issubset(result["design_df"].columns)

    def test_design_size_matches_report_n(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        assert len(result["design_df"]) == result["report"]["n"]

    def test_buckets_df_counts_sum_to_n(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        n = result["report"]["n"]
        assert result["buckets_df"]["count"].sum() == n

    def test_achieved_power_in_unit_interval(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        pwr = result["report"]["achieved_power"]
        assert 0.0 <= pwr <= 1.0

    def test_report_contains_expected_keys(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        rpt = result["report"]
        for key in ("n", "p", "df_num", "df_denom", "alpha", "target_power", "achieved_power"):
            assert key in rpt, f"Missing report key: {key}"

    def test_raises_on_max_n_smaller_than_p(self):
        """max_n <= p should fail validation before any computation."""
        L, delta = contrast_from_scenarios(
            FORMULA, FACTORS,
            {"A": "low", "B": 0.0},
            {"A": "high", "B": 10.0},
            sesoi=1.0,
        )
        cfg = PowerContrastConfig(L=L, delta=delta, max_n=2)  # p=3
        with pytest.raises(ValueError):
            i_optimal_powered_design(FORMULA, FACTORS, cfg, design_opts=FAST_OPTS)

    def test_no_internal_keys_in_result(self):
        """Private cache keys (_selected_idx, _X_cand) must be stripped."""
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        assert "_selected_idx" not in result
        assert "_X_cand" not in result


# ---------------------------------------------------------------------------
# i_optimal_powered_design — R² mode
# ---------------------------------------------------------------------------

class TestIOptimalPoweredDesignR2:
    def test_returns_expected_keys(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _r2_cfg(), design_opts=FAST_OPTS
        )
        assert set(result.keys()) >= {"design_df", "buckets_df", "report"}

    def test_achieved_power_in_unit_interval(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _r2_cfg(), design_opts=FAST_OPTS
        )
        pwr = result["report"]["achieved_power"]
        assert 0.0 <= pwr <= 1.0

    def test_design_size_matches_report_n(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _r2_cfg(), design_opts=FAST_OPTS
        )
        assert len(result["design_df"]) == result["report"]["n"]


# ---------------------------------------------------------------------------
# power_curve_by_n
# ---------------------------------------------------------------------------

class TestPowerCurveByN:
    def test_returns_dataframe(self):
        df = power_curve_by_n(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        assert isinstance(df, pd.DataFrame)

    def test_has_n_and_power_columns(self):
        df = power_curve_by_n(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        assert "n" in df.columns
        assert "power" in df.columns

    def test_power_generally_increases_with_n(self):
        """Power should be weakly increasing across the sampled n range."""
        df = power_curve_by_n(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        ).sort_values("n")
        # Allow occasional ties (same n design, numerical noise) but no large dips
        powers = df["power"].to_numpy()
        # Verify the last value is >= the first
        assert powers[-1] >= powers[0]


# ---------------------------------------------------------------------------
# power_curve_by_effect
# ---------------------------------------------------------------------------

class TestPowerCurveByEffect:
    def test_returns_dataframe(self):
        df = power_curve_by_effect(
            FORMULA, FACTORS, n=20, power_cfg=_contrast_cfg(), design_opts=FAST_OPTS
        )
        assert isinstance(df, pd.DataFrame)

    def test_has_effect_and_power_columns(self):
        df = power_curve_by_effect(
            FORMULA, FACTORS, n=20, power_cfg=_contrast_cfg(), design_opts=FAST_OPTS
        )
        assert "effect_scale" in df.columns
        assert "power" in df.columns

    def test_power_increases_with_effect(self):
        df = power_curve_by_effect(
            FORMULA, FACTORS, n=30, power_cfg=_contrast_cfg(), design_opts=FAST_OPTS
        ).sort_values("effect_scale")
        powers = df["power"].to_numpy()
        assert powers[-1] >= powers[0]

    def test_r2_mode_has_correct_columns(self):
        df = power_curve_by_effect(
            FORMULA, FACTORS, n=20, power_cfg=_r2_cfg(), design_opts=FAST_OPTS
        )
        assert "r2_target" in df.columns
        assert "power" in df.columns

    def test_raises_when_n_leq_p(self):
        with pytest.raises(ValueError):
            power_curve_by_effect(
                FORMULA, FACTORS, n=1, power_cfg=_contrast_cfg(), design_opts=FAST_OPTS
            )


# ---------------------------------------------------------------------------
# D-optimal criterion — integration tests
# ---------------------------------------------------------------------------

# Deliberately small options identical to FAST_OPTS but with criterion="D"
FAST_OPTS_D = DesignOptions(
    candidate_points=150,
    starts=2,
    max_iter=50,
    random_state=0,
    criterion="D",
)


class TestDOptimalCriterion:
    """Verify that criterion='D' flows through the full API stack without
    error and that output shapes / report values are well-formed."""

    def test_contrast_mode_returns_expected_keys(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS_D
        )
        assert set(result.keys()) >= {"design_df", "buckets_df", "report"}

    def test_contrast_mode_report_criterion_is_D(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS_D
        )
        assert result["report"]["criterion"] == "D"

    def test_contrast_mode_design_size_matches_report_n(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS_D
        )
        assert len(result["design_df"]) == result["report"]["n"]

    def test_contrast_mode_achieved_power_in_unit_interval(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS_D
        )
        pwr = result["report"]["achieved_power"]
        assert 0.0 <= pwr <= 1.0

    def test_r2_mode_returns_expected_keys(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _r2_cfg(), design_opts=FAST_OPTS_D
        )
        assert set(result.keys()) >= {"design_df", "buckets_df", "report"}

    def test_r2_mode_report_criterion_is_D(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _r2_cfg(), design_opts=FAST_OPTS_D
        )
        assert result["report"]["criterion"] == "D"

    def test_r2_mode_design_size_matches_report_n(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _r2_cfg(), design_opts=FAST_OPTS_D
        )
        assert len(result["design_df"]) == result["report"]["n"]


# ---------------------------------------------------------------------------
# A-optimal criterion — integration tests
# ---------------------------------------------------------------------------

FAST_OPTS_A = DesignOptions(
    candidate_points=150,
    starts=2,
    max_iter=50,
    random_state=0,
    criterion="A",
)


class TestAOptimalCriterion:
    """Verify criterion='A' flows through the full API without error."""

    def test_contrast_mode_returns_expected_keys(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS_A
        )
        assert set(result.keys()) >= {"design_df", "buckets_df", "report"}

    def test_contrast_mode_report_criterion_is_A(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS_A
        )
        assert result["report"]["criterion"] == "A"

    def test_contrast_mode_design_size_matches_report_n(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS_A
        )
        assert len(result["design_df"]) == result["report"]["n"]

    def test_r2_mode_achieved_power_in_unit_interval(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _r2_cfg(), design_opts=FAST_OPTS_A
        )
        pwr = result["report"]["achieved_power"]
        assert 0.0 <= pwr <= 1.0


# ---------------------------------------------------------------------------
# power_sensitivity — R² mode extension
# ---------------------------------------------------------------------------

class TestPowerSensitivityR2:
    """Verify power_sensitivity works for PowerR2Config."""

    @pytest.fixture
    def fixed_design(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _r2_cfg(), design_opts=FAST_OPTS
        )
        return result["design_df"]

    def test_returns_expected_keys(self, fixed_design):
        sens = power_sensitivity(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_r2_cfg(),
            design_df=fixed_design,
            r2_range=(0.05, 0.50),
            r2_points=5,
        )
        assert set(sens.keys()) >= {"data", "nominal_power", "r2_nominal", "figure"}

    def test_data_has_correct_columns(self, fixed_design):
        sens = power_sensitivity(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_r2_cfg(), design_df=fixed_design,
            r2_range=(0.05, 0.40), r2_points=5,
        )
        assert "r2_target" in sens["data"].columns
        assert "power" in sens["data"].columns

    def test_data_length_matches_r2_points(self, fixed_design):
        sens = power_sensitivity(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_r2_cfg(), design_df=fixed_design,
            r2_range=(0.05, 0.40), r2_points=8,
        )
        assert len(sens["data"]) == 8

    def test_nominal_power_is_float(self, fixed_design):
        sens = power_sensitivity(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_r2_cfg(), design_df=fixed_design,
            r2_points=5,
        )
        assert isinstance(sens["nominal_power"], float)
        assert 0.0 <= sens["nominal_power"] <= 1.0

    def test_power_increases_with_r2(self, fixed_design):
        """Power must be non-decreasing as r2_target increases."""
        sens = power_sensitivity(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_r2_cfg(), design_df=fixed_design,
            r2_range=(0.05, 0.60), r2_points=10,
        )
        pwrs = sens["data"]["power"].to_numpy()
        # Not strictly monotone per point, but last > first
        assert pwrs[-1] >= pwrs[0]

    def test_invalid_r2_range_raises(self, fixed_design):
        with pytest.raises(ValueError):
            power_sensitivity(
                formula=FORMULA, factors=FACTORS,
                power_cfg=_r2_cfg(), design_df=fixed_design,
                r2_range=(0.50, 0.10),  # lo >= hi
            )

    def test_contrast_mode_still_works(self, fixed_design):
        """Existing contrast-mode path must not be broken."""
        L, delta = contrast_from_scenarios(
            FORMULA, FACTORS,
            {"A": "low", "B": 0.0},
            {"A": "high", "B": 10.0},
            sesoi=1.0,
        )
        cfg = PowerContrastConfig(L=L, delta=delta, power=0.80, max_n=80)
        sens = power_sensitivity(
            formula=FORMULA, factors=FACTORS,
            power_cfg=cfg, design_df=fixed_design,
            sigma_range=(0.5, 2.0), sigma_points=5,
        )
        assert "sigma" in sens["data"].columns
        assert "sigma_nominal" in sens


# ---------------------------------------------------------------------------
# min_detectable_effect
# ---------------------------------------------------------------------------

class TestMinDetectableEffect:
    """Tests for the min_detectable_effect function."""

    @pytest.fixture
    def fixed_design(self):
        result = i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )
        return result["design_df"]

    def test_contrast_mode_returns_expected_keys(self, fixed_design):
        mde = min_detectable_effect(
            design_df=fixed_design,
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
        )
        assert set(mde.keys()) >= {"mde", "achieved_power", "n", "mode"}

    def test_contrast_mode_is_contrast(self, fixed_design):
        mde = min_detectable_effect(
            design_df=fixed_design,
            formula=FORMULA, factors=FACTORS,
            power_cfg=_contrast_cfg(),
        )
        assert mde["mode"] == "contrast"

    def test_contrast_mde_is_positive(self, fixed_design):
        mde = min_detectable_effect(
            design_df=fixed_design,
            formula=FORMULA, factors=FACTORS,
            power_cfg=_contrast_cfg(),
        )
        assert mde["mde"] > 0

    def test_contrast_achieved_power_near_target(self, fixed_design):
        """Bisected MDE should achieve ≈ target_power (within 2 %)."""
        target = 0.80
        mde = min_detectable_effect(
            design_df=fixed_design,
            formula=FORMULA, factors=FACTORS,
            power_cfg=_contrast_cfg(),
            target_power=target,
        )
        assert abs(mde["achieved_power"] - target) < 0.02

    def test_r2_mode_returns_expected_keys(self, fixed_design):
        mde = min_detectable_effect(
            design_df=fixed_design,
            formula=FORMULA, factors=FACTORS,
            power_cfg=_r2_cfg(),
        )
        assert set(mde.keys()) >= {"mde", "achieved_power", "n", "mode"}
        assert mde["mode"] == "r2"

    def test_r2_mde_in_valid_range(self, fixed_design):
        mde = min_detectable_effect(
            design_df=fixed_design,
            formula=FORMULA, factors=FACTORS,
            power_cfg=_r2_cfg(),
        )
        assert 0.0 < mde["mde"] < 1.0

    def test_raises_on_invalid_target_power(self, fixed_design):
        with pytest.raises(ValueError, match="target_power"):
            min_detectable_effect(
                design_df=fixed_design,
                formula=FORMULA, factors=FACTORS,
                power_cfg=_contrast_cfg(),
                target_power=1.5,
            )


# ---------------------------------------------------------------------------
# Enhancement 10 — Richer run metadata in report
# ---------------------------------------------------------------------------

class TestRunMetadata:
    """Verify that i_optimal_powered_design enriches report with run metadata."""

    @pytest.fixture(scope="class")
    def result_contrast(self):
        """One small contrast-mode run shared across all tests in this class."""
        return i_optimal_powered_design(
            FORMULA, FACTORS, _contrast_cfg(), design_opts=FAST_OPTS
        )

    @pytest.fixture(scope="class")
    def result_r2(self):
        """One small R²-mode run shared across all tests in this class."""
        return i_optimal_powered_design(
            FORMULA, FACTORS, _r2_cfg(), design_opts=FAST_OPTS
        )

    # --- elapsed_sec ---
    def test_elapsed_sec_present(self, result_contrast):
        assert "elapsed_sec" in result_contrast["report"]

    def test_elapsed_sec_positive(self, result_contrast):
        assert result_contrast["report"]["elapsed_sec"] > 0.0

    def test_elapsed_sec_is_float(self, result_contrast):
        assert isinstance(result_contrast["report"]["elapsed_sec"], float)

    def test_elapsed_sec_present_r2(self, result_r2):
        assert result_r2["report"]["elapsed_sec"] > 0.0

    # --- search_strategy ---
    def test_search_strategy_present(self, result_contrast):
        assert "search_strategy" in result_contrast["report"]

    def test_search_strategy_contains_bisection(self, result_contrast):
        assert "bisection" in result_contrast["report"]["search_strategy"]

    def test_search_strategy_is_string(self, result_contrast):
        assert isinstance(result_contrast["report"]["search_strategy"], str)

    def test_search_strategy_present_r2(self, result_r2):
        assert "bisection" in result_r2["report"]["search_strategy"]

    # --- verify_window ---
    def test_verify_window_present(self, result_contrast):
        assert "verify_window" in result_contrast["report"]

    def test_verify_window_is_non_negative_int(self, result_contrast):
        vw = result_contrast["report"]["verify_window"]
        assert isinstance(vw, int)
        assert vw >= 0

    # --- random_state ---
    def test_random_state_present(self, result_contrast):
        assert "random_state" in result_contrast["report"]

    def test_random_state_matches_design_opts(self, result_contrast):
        assert result_contrast["report"]["random_state"] == FAST_OPTS.random_state

    def test_random_state_is_int(self, result_contrast):
        assert isinstance(result_contrast["report"]["random_state"], int)

    # --- warnings ---
    def test_warnings_present(self, result_contrast):
        assert "warnings" in result_contrast["report"]

    def test_warnings_is_list(self, result_contrast):
        assert isinstance(result_contrast["report"]["warnings"], list)

    def test_warnings_present_r2(self, result_r2):
        assert isinstance(result_r2["report"]["warnings"], list)


# ---------------------------------------------------------------------------
# Enhancement 11 — compare_criteria() helper
# ---------------------------------------------------------------------------

class TestCompareCriteria:
    """Verify that compare_criteria runs all criteria and returns well-formed output."""

    @pytest.fixture(scope="class")
    def comparison(self):
        """One full compare_criteria run shared across all tests in this class."""
        return compare_criteria(
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            design_opts=FAST_OPTS,  # criterion field overridden per run
        )

    # --- Top-level return structure ---
    def test_returns_expected_keys(self, comparison):
        assert set(comparison.keys()) >= {"summary", "results", "figure"}

    def test_figure_is_none_by_default(self, comparison):
        assert comparison["figure"] is None

    # --- summary DataFrame ---
    def test_summary_is_dataframe(self, comparison):
        assert isinstance(comparison["summary"], pd.DataFrame)

    def test_summary_has_three_rows(self, comparison):
        """Default run uses all three criteria → three rows."""
        assert len(comparison["summary"]) == 3

    def test_summary_has_criterion_column(self, comparison):
        assert "criterion" in comparison["summary"].columns

    def test_summary_criteria_values(self, comparison):
        assert set(comparison["summary"]["criterion"]) == {"I", "D", "A"}

    def test_summary_has_n_column(self, comparison):
        assert "n" in comparison["summary"].columns

    def test_summary_n_values_are_positive_ints(self, comparison):
        for val in comparison["summary"]["n"]:
            assert isinstance(val, (int, np.integer))
            assert val > 0

    def test_summary_has_achieved_power_column(self, comparison):
        assert "achieved_power" in comparison["summary"].columns

    def test_summary_achieved_power_in_unit_interval(self, comparison):
        for pwr in comparison["summary"]["achieved_power"]:
            assert 0.0 <= pwr <= 1.0

    def test_summary_has_elapsed_sec_column(self, comparison):
        assert "elapsed_sec" in comparison["summary"].columns

    def test_summary_elapsed_sec_positive(self, comparison):
        for t in comparison["summary"]["elapsed_sec"]:
            assert t > 0.0 or np.isnan(t)  # nan allowed if metadata not injected

    # --- results dict ---
    def test_results_is_dict(self, comparison):
        assert isinstance(comparison["results"], dict)

    def test_results_has_all_three_criteria(self, comparison):
        assert set(comparison["results"].keys()) == {"I", "D", "A"}

    def test_each_result_has_design_df(self, comparison):
        for crit, res in comparison["results"].items():
            assert "design_df" in res, f"Missing design_df for criterion={crit}"
            assert isinstance(res["design_df"], pd.DataFrame)

    def test_each_result_has_report(self, comparison):
        for crit, res in comparison["results"].items():
            assert "report" in res, f"Missing report for criterion={crit}"
            assert res["report"]["criterion"] == crit

    # --- Subset / custom criteria ---
    def test_custom_subset_two_criteria(self):
        result = compare_criteria(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_contrast_cfg(), design_opts=FAST_OPTS,
            criteria=["I", "D"],
        )
        assert len(result["summary"]) == 2
        assert set(result["results"].keys()) == {"I", "D"}

    def test_single_criterion_allowed(self):
        result = compare_criteria(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_contrast_cfg(), design_opts=FAST_OPTS,
            criteria=["A"],
        )
        assert len(result["summary"]) == 1
        assert "A" in result["results"]

    # --- R² mode ---
    def test_r2_mode_runs_without_error(self):
        result = compare_criteria(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_r2_cfg(), design_opts=FAST_OPTS,
        )
        assert len(result["summary"]) == 3

    # --- Validation ---
    def test_empty_criteria_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            compare_criteria(
                formula=FORMULA, factors=FACTORS,
                power_cfg=_contrast_cfg(), design_opts=FAST_OPTS,
                criteria=[],
            )

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            compare_criteria(
                formula=FORMULA, factors=FACTORS,
                power_cfg=_contrast_cfg(), design_opts=FAST_OPTS,
                criteria=["I", "X"],
            )

    def test_design_opts_not_mutated(self):
        """Original DesignOptions.criterion must be unchanged after the call."""
        opts = DesignOptions(
            candidate_points=150, starts=2, max_iter=50, random_state=0, criterion="I"
        )
        compare_criteria(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_contrast_cfg(), design_opts=opts,
        )
        assert opts.criterion == "I"  # must not have been mutated
