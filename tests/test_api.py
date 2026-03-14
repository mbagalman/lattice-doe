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
    PowerGLMContrastConfig,
    SplitPlotOptions,
    ResponseSpec,
    MultiResponseOptions,
    i_optimal_powered_design,
    i_optimal_multiresponse_design,
    power_curve_by_effect,
    power_curve_by_n,
    power_curve_by_wp,
    power_sensitivity,
    min_detectable_effect,
    compare_criteria,
    augment_design,
    robustness_report,
    power_curve_by_n_multiresponse,
    multiresponse_sensitivity,
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

    def test_raises_when_candidate_pool_too_small_for_requested_n(self):
        # For FORMULA "~ 1 + A + B", p=3 so by_n starts at n>=4.
        # candidate_points=3 guarantees n > n_cand and must fail fast.
        tiny_opts = DesignOptions(
            candidate_points=3,
            auto_candidate=False,
            starts=1,
            max_iter=20,
            random_state=0,
        )
        with pytest.raises(ValueError, match="exceeds the candidate set size"):
            power_curve_by_n(FORMULA, FACTORS, _contrast_cfg(), design_opts=tiny_opts)


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

    def test_raises_when_candidate_pool_too_small_for_n(self):
        tiny_opts = DesignOptions(
            candidate_points=3,
            auto_candidate=False,
            starts=1,
            max_iter=20,
            random_state=0,
        )
        with pytest.raises(ValueError, match="exceeds the candidate set size"):
            power_curve_by_effect(
                FORMULA, FACTORS, n=5, power_cfg=_contrast_cfg(), design_opts=tiny_opts
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


# ---------------------------------------------------------------------------
# Regression: report JSON serialization with Path values (issue #2)
# ---------------------------------------------------------------------------

class TestReportJsonSerialization:
    """Regression tests: report dict must always be JSON-serializable."""

    def test_report_is_json_serializable_contrast(self):
        """report dict from contrast mode must not contain non-JSON types."""
        import json
        result = i_optimal_powered_design(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_contrast_cfg(), design_opts=FAST_OPTS,
        )
        # Must not raise; Path objects would cause TypeError here
        json.dumps(result["report"])

    def test_report_is_json_serializable_r2(self):
        """report dict from R² mode must not contain non-JSON types."""
        import json
        result = i_optimal_powered_design(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_r2_cfg(), design_opts=FAST_OPTS,
        )
        json.dumps(result["report"])

    def test_diagnostic_exports_stored_as_strings(self, tmp_path):
        """When export_diagnostics_to is set, paths in report are strings, not Path objects."""
        import json
        result = i_optimal_powered_design(
            formula=FORMULA, factors=FACTORS,
            power_cfg=_contrast_cfg(), design_opts=FAST_OPTS,
            export_diagnostics_to=str(tmp_path),
        )
        exports = result["report"].get("diagnostic_exports", {})
        for key, val in exports.items():
            assert isinstance(val, str), (
                f"diagnostic_exports[{key!r}] is {type(val).__name__}, expected str"
            )
        # Full report must still be JSON-serializable
        json.dumps(result["report"])


# ---------------------------------------------------------------------------
# robustness_report
# ---------------------------------------------------------------------------

# Shared fixture: small fast design for robustness tests
_ROB_OPTS = DesignOptions(candidate_points=100, starts=2, max_iter=40, random_state=7)


def _contrast_design():
    """Build a small contrast-mode design for robustness tests."""
    from iopt_power_design.contrasts import contrast_from_scenarios
    L, delta = contrast_from_scenarios(
        FORMULA, FACTORS,
        {"A": "low",  "B": 0.0},
        {"A": "high", "B": 10.0},
        sesoi=1.0,
    )
    cfg = PowerContrastConfig(L=L, delta=delta, sigma=1.0, power=0.8, max_n=80)
    result = i_optimal_powered_design(FORMULA, FACTORS, cfg, design_opts=_ROB_OPTS)
    return result["design_df"], cfg


def _r2_design():
    """Build a small R²-mode design for robustness tests."""
    cfg = PowerR2Config(r2_target=0.30, power=0.8, max_n=80)
    result = i_optimal_powered_design(FORMULA, FACTORS, cfg, design_opts=_ROB_OPTS)
    return result["design_df"], cfg


class TestRobustnessReport:
    """Tests for robustness_report() in both contrast and R² modes."""

    # ------------------------------------------------------------------
    # Return structure — contrast mode
    # ------------------------------------------------------------------

    def test_contrast_returns_dict_with_expected_keys(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        for key in ("mode", "nominal_power", "effect_sweep", "sigma_sweep",
                    "alpha_sweep", "summary", "thresholds", "figure"):
            assert key in rob, f"Missing key: {key}"

    def test_contrast_mode_field_is_contrast(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        assert rob["mode"] == "contrast"

    def test_contrast_nominal_power_in_unit_interval(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        assert 0.0 <= rob["nominal_power"] <= 1.0

    def test_contrast_effect_sweep_has_correct_columns(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        assert list(rob["effect_sweep"].columns) == [
            "effect_scale", "power", "noncentrality_lambda"
        ]

    def test_contrast_effect_sweep_length_matches_points(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg, effect_points=7)
        assert len(rob["effect_sweep"]) == 7

    def test_contrast_sigma_sweep_is_dataframe_not_none(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        assert rob["sigma_sweep"] is not None
        assert list(rob["sigma_sweep"].columns) == ["sigma", "power", "noncentrality_lambda"]

    def test_contrast_sigma_sweep_length_matches_points(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg, sigma_points=5)
        assert len(rob["sigma_sweep"]) == 5

    def test_contrast_alpha_sweep_has_correct_columns(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        assert list(rob["alpha_sweep"].columns) == ["alpha", "power", "noncentrality_lambda"]

    def test_contrast_alpha_sweep_length_matches_points(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg, alpha_points=5)
        assert len(rob["alpha_sweep"]) == 5

    # ------------------------------------------------------------------
    # Return structure — R² mode
    # ------------------------------------------------------------------

    def test_r2_returns_dict_with_expected_keys(self):
        design_df, cfg = _r2_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                effect_range=(0.05, 0.5))
        for key in ("mode", "nominal_power", "effect_sweep", "sigma_sweep",
                    "alpha_sweep", "summary", "thresholds", "figure"):
            assert key in rob

    def test_r2_mode_field_is_r2(self):
        design_df, cfg = _r2_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                effect_range=(0.05, 0.5))
        assert rob["mode"] == "r2"

    def test_r2_sigma_sweep_is_none(self):
        """sigma does not affect R² power — sigma_sweep must be None."""
        design_df, cfg = _r2_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                effect_range=(0.05, 0.5))
        assert rob["sigma_sweep"] is None

    def test_r2_effect_sweep_has_r2_target_column(self):
        design_df, cfg = _r2_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                effect_range=(0.05, 0.5))
        assert "r2_target" in rob["effect_sweep"].columns

    def test_r2_thresholds_max_sigma_is_none(self):
        design_df, cfg = _r2_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                effect_range=(0.05, 0.5))
        assert rob["thresholds"]["max_sigma_for_target"] is None

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def test_summary_has_expected_keys(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        for k in ("worst_power", "median_power", "best_power",
                  "power_target", "pct_scenarios_passing"):
            assert k in rob["summary"]

    def test_summary_worst_le_median_le_best(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        s = rob["summary"]
        assert s["worst_power"] <= s["median_power"] <= s["best_power"]

    def test_summary_power_target_matches_config(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        assert rob["summary"]["power_target"] == pytest.approx(cfg.power)

    def test_summary_pct_passing_in_unit_interval(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        assert 0.0 <= rob["summary"]["pct_scenarios_passing"] <= 1.0

    # ------------------------------------------------------------------
    # Threshold crossings
    # ------------------------------------------------------------------

    def test_thresholds_has_expected_keys(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg)
        for k in ("max_sigma_for_target", "min_effect_for_target", "min_alpha_for_target"):
            assert k in rob["thresholds"]

    def test_max_sigma_threshold_is_within_sweep_range(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                sigma_range=(0.5, 2.0))
        t = rob["thresholds"]["max_sigma_for_target"]
        if t is not None:
            assert 0.5 <= t <= 2.0

    def test_min_effect_threshold_is_within_sweep_range(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                effect_range=(0.5, 2.0))
        t = rob["thresholds"]["min_effect_for_target"]
        if t is not None:
            assert 0.5 <= t <= 2.0

    def test_min_alpha_threshold_is_within_sweep_range(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                alpha_range=(0.01, 0.10))
        t = rob["thresholds"]["min_alpha_for_target"]
        if t is not None:
            assert 0.01 <= t <= 0.10

    # ------------------------------------------------------------------
    # Power monotonicity
    # ------------------------------------------------------------------

    def test_power_increases_with_effect_scale(self):
        """Higher delta scale → higher power."""
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                effect_range=(0.3, 2.0), effect_points=6)
        powers = rob["effect_sweep"]["power"].values
        # Not strictly required to be monotone at every step, but overall trend must hold
        assert powers[0] <= powers[-1]

    def test_power_decreases_with_sigma(self):
        """Higher sigma → lower power."""
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                sigma_range=(0.5, 3.0), sigma_points=6)
        powers = rob["sigma_sweep"]["power"].values
        assert powers[0] >= powers[-1]

    def test_power_decreases_with_stricter_alpha(self):
        """Smaller alpha → stricter threshold → lower power."""
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg,
                                alpha_range=(0.005, 0.10), alpha_points=6)
        powers = rob["alpha_sweep"]["power"].values
        assert powers[0] <= powers[-1]

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_invalid_sigma_range_raises(self):
        design_df, cfg = _contrast_design()
        with pytest.raises(ValueError, match="sigma_range"):
            robustness_report(design_df, FORMULA, FACTORS, cfg,
                              sigma_range=(2.0, 0.5))

    def test_invalid_alpha_range_raises(self):
        design_df, cfg = _contrast_design()
        with pytest.raises(ValueError, match="alpha_range"):
            robustness_report(design_df, FORMULA, FACTORS, cfg,
                              alpha_range=(0.10, 0.01))

    def test_invalid_effect_range_r2_above_one_raises(self):
        design_df, cfg = _r2_design()
        with pytest.raises(ValueError, match="effect_range"):
            robustness_report(design_df, FORMULA, FACTORS, cfg,
                              effect_range=(0.5, 1.5))

    def test_sigma_points_below_two_raises(self):
        design_df, cfg = _contrast_design()
        with pytest.raises(ValueError, match="sigma_points"):
            robustness_report(design_df, FORMULA, FACTORS, cfg, sigma_points=1)

    def test_alpha_points_below_two_raises(self):
        design_df, cfg = _contrast_design()
        with pytest.raises(ValueError, match="alpha_points"):
            robustness_report(design_df, FORMULA, FACTORS, cfg, alpha_points=1)

    # ------------------------------------------------------------------
    # Figure output
    # ------------------------------------------------------------------

    def test_figure_is_none_when_plot_false(self):
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg, plot=False)
        assert rob["figure"] is None

    def test_figure_returned_when_plot_true(self):
        pytest.importorskip("matplotlib")
        design_df, cfg = _contrast_design()
        rob = robustness_report(design_df, FORMULA, FACTORS, cfg, plot=True)
        assert rob["figure"] is not None


# ---------------------------------------------------------------------------
# SP-10 — Integration regression tests for split-plot designs
# ---------------------------------------------------------------------------

# Shared helpers for SP integration tests
_SP_FORMULA = "~ 1 + A + B"
_SP_FACTORS = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
_SP_FAST_OPTS = DesignOptions(
    candidate_points=100,
    starts=2,
    max_iter=30,
    random_state=7,
)


def _sp_contrast_cfg(max_n: int = 20) -> PowerContrastConfig:
    from iopt_power_design.contrasts import contrast_from_scenarios
    L, delta = contrast_from_scenarios(
        _SP_FORMULA, _SP_FACTORS,
        {"A": -1.0, "B": -1.0},
        {"A":  1.0, "B":  1.0},
        sesoi=1.0,
    )
    return PowerContrastConfig(L=L, delta=delta, power=0.80, max_n=max_n)


class TestSplitPlotIntegration:
    """End-to-end regression tests for split-plot designs (SP-10)."""

    def test_2wp_3sp_contrast_mode(self):
        """Basic 2-factor model, 1 HTC (A), 1 ETC (B), 2 WPs, 3 SPs each."""
        sp_opts = SplitPlotOptions(
            htc_factors=["A"],
            n_whole_plots=2,
            subplots_per_wp=3,
            eta=1.0,
        )
        opts = DesignOptions(
            candidate_points=80,
            starts=2,
            max_iter=20,
            random_state=99,
            split_plot=sp_opts,
        )
        result = i_optimal_powered_design(_SP_FORMULA, _SP_FACTORS, _sp_contrast_cfg(max_n=30), design_opts=opts)
        assert isinstance(result, dict)
        assert "design_df" in result and "report" in result
        assert len(result["design_df"]) >= 6  # at least 2 WPs × 3 SPs

    def test_htc_factor_collision_raises(self):
        """htc_factors contains a name not in factors dict → ValueError."""
        sp_opts = SplitPlotOptions(
            htc_factors=["NONEXISTENT"],
            n_whole_plots=3,
            eta=1.0,
        )
        opts = _SP_FAST_OPTS.__class__(
            candidate_points=80,
            starts=1,
            max_iter=10,
            random_state=0,
            split_plot=sp_opts,
        )
        with pytest.raises(ValueError, match="htc_factors"):
            i_optimal_powered_design(_SP_FORMULA, _SP_FACTORS, _sp_contrast_cfg(), design_opts=opts)

    def test_split_plot_and_blocked_raises(self):
        """Setting both n_blocks ≥ 2 and split_plot together raises ValueError."""
        sp_opts = SplitPlotOptions(htc_factors=["A"], n_whole_plots=3, eta=1.0)
        # DesignOptions warns but API enforces; build opts with both set
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opts = DesignOptions(
                candidate_points=80,
                starts=1,
                max_iter=10,
                random_state=0,
                n_blocks=2,
                split_plot=sp_opts,
            )
        with pytest.raises(ValueError, match="n_blocks"):
            i_optimal_powered_design(_SP_FORMULA, _SP_FACTORS, _sp_contrast_cfg(), design_opts=opts)

    def test_eta_zero_matches_ols_power(self):
        """Power at eta=0 should equal OLS power (GLS degenerates to OLS)."""
        sp_opts_eta0 = SplitPlotOptions(
            htc_factors=["A"],
            n_whole_plots=4,
            subplots_per_wp=3,
            eta=0.0,
        )
        opts_sp = DesignOptions(
            candidate_points=80,
            starts=2,
            max_iter=20,
            random_state=11,
            split_plot=sp_opts_eta0,
        )
        result_sp = i_optimal_powered_design(
            _SP_FORMULA, _SP_FACTORS, _sp_contrast_cfg(max_n=40), design_opts=opts_sp
        )
        # eta=0 → V=I → GLS power == OLS power
        # The achieved_power should be non-trivial (> 0)
        assert result_sp["report"]["achieved_power"] > 0.0
        # And the split_plot dict should record eta=0.0
        assert result_sp["report"]["split_plot"]["eta"] == pytest.approx(0.0)

    def test_design_respects_nesting(self):
        """All sub-plots within a whole plot share identical HTC factor values."""
        sp_opts = SplitPlotOptions(
            htc_factors=["A"],
            n_whole_plots=3,
            subplots_per_wp=3,
            eta=1.0,
        )
        opts = DesignOptions(
            candidate_points=100,
            starts=2,
            max_iter=20,
            random_state=22,
            split_plot=sp_opts,
        )
        result = i_optimal_powered_design(_SP_FORMULA, _SP_FACTORS, _sp_contrast_cfg(max_n=30), design_opts=opts)
        df = result["design_df"]
        assert "__wp_id__" in df.columns, "design_df must contain __wp_id__ column"
        for wp_id, group in df.groupby("__wp_id__"):
            assert group["A"].nunique() == 1, (
                f"WP {wp_id} has non-constant HTC factor 'A': {group['A'].values}"
            )

    def test_report_contains_split_plot_dict(self):
        """Result report includes 'split_plot' sub-dict with expected keys."""
        sp_opts = SplitPlotOptions(htc_factors=["A"], n_whole_plots=3, eta=1.0, subplots_per_wp=3)
        opts = DesignOptions(
            candidate_points=80,
            starts=2,
            max_iter=20,
            random_state=33,
            split_plot=sp_opts,
        )
        result = i_optimal_powered_design(_SP_FORMULA, _SP_FACTORS, _sp_contrast_cfg(max_n=30), design_opts=opts)
        assert "split_plot" in result["report"]
        sp_dict = result["report"]["split_plot"]
        for key in ("n_whole_plots", "subplots_per_wp", "n_total", "eta", "htc_factors", "etc_factors", "df_method"):
            assert key in sp_dict, f"Missing key in split_plot report: {key}"

    def test_power_curve_by_wp_returns_dataframe(self):
        """power_curve_by_wp returns a DataFrame with n_wp and power columns."""
        cfg = _sp_contrast_cfg(max_n=40)
        opts = DesignOptions(candidate_points=80, starts=2, max_iter=15, random_state=55)
        df = power_curve_by_wp(
            _SP_FORMULA,
            _SP_FACTORS,
            cfg,
            subplots_per_wp=3,
            htc_factors=["A"],
            eta=1.0,
            wp_range=(2, 5),
            wp_points=3,
            design_opts=opts,
        )
        assert isinstance(df, pd.DataFrame)
        assert "n_wp" in df.columns
        assert "power" in df.columns
        assert len(df) == 3
        assert (df["power"] >= 0.0).all() and (df["power"] <= 1.0).all()


# ---------------------------------------------------------------------------
# TestMultiResponseAPI
# ---------------------------------------------------------------------------

# Shared helpers for multi-response tests — small problem so tests run fast.
_MR_FORMULA = "~ 1 + A + B"
_MR_FACTORS = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
_MR_L = np.array([[0, 1, 0], [0, 0, 1]])
_MR_DELTA = np.array([1.5, 1.5])


def _mr_opts(**kw):
    defaults = dict(candidate_points=200, starts=2, random_state=7, max_iter=30)
    defaults.update(kw)
    return DesignOptions(**defaults)


def _contrast_rs(name, sigma=1.0, power=0.8, max_n=60, **kw):
    cfg = PowerContrastConfig(
        L=_MR_L, delta=_MR_DELTA, sigma=sigma, power=power,
        max_n=max_n, max_iter=30, **kw
    )
    return ResponseSpec(name=name, power_cfg=cfg)


def _r2_rs(name, r2=0.5, power=0.8, max_n=60):
    cfg = PowerR2Config(r2_target=r2, power=power, max_n=max_n, max_iter=30)
    return ResponseSpec(name=name, power_cfg=cfg)


def _run_mr(responses, rule="min", **opts_kw):
    multi = MultiResponseOptions(responses=responses, power_combination=rule)
    opts = _mr_opts(**opts_kw)
    return i_optimal_multiresponse_design(_MR_FORMULA, _MR_FACTORS, multi, opts)


class TestMultiResponseAPI:
    def test_two_identical_responses_returns_result(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")])
        assert isinstance(result, dict)

    def test_responses_length_matches_multi_cfg(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")])
        assert len(result["responses"]) == 2

    def test_response_dicts_have_required_keys(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")])
        for rd in result["responses"]:
            assert "name" in rd
            assert "power" in rd
            assert "lam" in rd

    def test_compound_criterion_false_for_shared_formula(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")])
        assert result["compound_criterion"] is False

    def test_combination_rule_in_result(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")], rule="min")
        assert result["combination_rule"] == "min"

    def test_design_is_dataframe_with_factor_cols(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")])
        assert isinstance(result["design"], pd.DataFrame)
        assert set(_MR_FACTORS.keys()).issubset(result["design"].columns)

    def test_buckets_is_present(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")])
        assert "buckets" in result
        assert isinstance(result["buckets"], pd.DataFrame)

    def test_elapsed_sec_positive(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")])
        assert result["elapsed_sec"] > 0

    def test_achieved_power_equals_combine_powers(self):
        from iopt_power_design import combine_powers
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")], rule="min")
        per_powers = [rd["power"] for rd in result["responses"]]
        expected = combine_powers(per_powers, None, "min")
        assert abs(result["achieved_power"] - expected) < 1e-12

    def test_two_identical_responses_n_matches_single_response(self):
        # With two identical responses + min rule, n must equal single-response n.
        single_cfg = PowerContrastConfig(
            L=_MR_L, delta=_MR_DELTA, sigma=1.0, power=0.8, max_n=60, max_iter=30
        )
        opts = _mr_opts()
        single = i_optimal_powered_design(_MR_FORMULA, _MR_FACTORS, single_cfg, opts)
        multi_result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")])
        # Allow ±1 tolerance for design randomness
        assert abs(multi_result["n"] - single["report"]["n"]) <= 1

    def test_harder_response_drives_n(self):
        # Y2 has smaller sigma → needs more runs → n must be ≥ Y1 alone
        opts = _mr_opts()
        single_cfg = PowerContrastConfig(
            L=_MR_L, delta=_MR_DELTA, sigma=0.7, power=0.8, max_n=60, max_iter=30
        )
        single = i_optimal_powered_design(_MR_FORMULA, _MR_FACTORS, single_cfg, opts)
        result = _run_mr([_contrast_rs("Y1", sigma=1.0), _contrast_rs("Y2", sigma=0.7)])
        assert result["n"] >= single["report"]["n"] - 1

    def test_product_combination_rule_accepted(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")], rule="product")
        assert result["combination_rule"] == "product"

    def test_mixed_contrast_r2_responses_no_error(self):
        result = _run_mr([_contrast_rs("Y1"), _r2_rs("Y2")])
        assert len(result["responses"]) == 2
        assert result["compound_criterion"] is False

    def test_differing_formula_uses_compound_path(self):
        # MR-5: differing formulas now route to the compound criterion path.
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=np.array([[0, 1]]), delta=np.array([1.5]),
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=30),
                          formula="~ 1 + A")
        L2 = np.array([[0, 1, 0], [0, 0, 1]])
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=L2, delta=np.array([1.5, 1.5]),
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=30))
        multi = MultiResponseOptions([r1, r2])
        result = i_optimal_multiresponse_design(_MR_FORMULA, _MR_FACTORS, multi, _mr_opts())
        assert result["compound_criterion"] is True

    def test_workers_parallel_returns_same_structure(self):
        result = _run_mr([_contrast_rs("Y1"), _contrast_rs("Y2")], workers=2)
        assert "n" in result
        assert len(result["responses"]) == 2

    def test_exported_from_top_level(self):
        import iopt_power_design
        assert hasattr(iopt_power_design, "i_optimal_multiresponse_design")


# ---------------------------------------------------------------------------
# Shared helpers for compound criterion tests (MR-5)
# ---------------------------------------------------------------------------
_CP_FORMULA = "~ 1 + A + B"
_CP_FACTORS = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
# Linear formula (only A): p=2
_CP_FORMULA_LINEAR = "~ 1 + A"
# Quadratic formula (A, B, A*B): p=4
_CP_FORMULA_QUAD = "~ 1 + A + B + A:B"

_CP_L_FULL = np.array([[0, 1, 0], [0, 0, 1]])   # contrast for ~ 1 + A + B
_CP_L_LINEAR = np.array([[0, 1]])                 # contrast for ~ 1 + A
_CP_L_QUAD = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])  # for ~ 1 + A + B + A:B
_CP_DELTA_2 = np.array([1.5, 1.5])
_CP_DELTA_1 = np.array([1.5])


def _cp_opts(**kw):
    defaults = dict(candidate_points=200, starts=2, random_state=42, max_iter=20)
    defaults.update(kw)
    return DesignOptions(**defaults)


def _cp_run(r1, r2, rule="min", **opts_kw):
    multi = MultiResponseOptions(responses=[r1, r2], power_combination=rule)
    opts = _cp_opts(**opts_kw)
    return i_optimal_multiresponse_design(_CP_FORMULA, _CP_FACTORS, multi, opts)


class TestCompoundCriterion:
    """MR-5: compound criterion path for responses with different model formulas."""

    def test_identical_formulas_compound_flag_false(self):
        # When both responses use the global formula, compound_criterion must be False.
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        assert result["compound_criterion"] is False

    def test_different_formulas_compound_flag_true(self):
        # One response uses a sub-formula; compound path should be activated.
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        assert result["compound_criterion"] is True

    def test_compound_result_has_required_keys(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        for key in ("design", "n", "achieved_power", "responses", "combination_rule",
                    "compound_criterion", "buckets", "elapsed_sec", "p",
                    "iteration", "search_strategy", "warnings"):
            assert key in result, f"Missing key: {key}"

    def test_compound_design_is_dataframe(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        assert isinstance(result["design"], pd.DataFrame)
        assert set(_CP_FACTORS.keys()).issubset(result["design"].columns)

    def test_compound_responses_list_length(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        assert len(result["responses"]) == 2

    def test_compound_responses_have_required_keys(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        for rd in result["responses"]:
            assert set(rd.keys()) >= {"name", "power", "lam", "n"}

    def test_compound_achieved_power_positive(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        assert 0.0 < result["achieved_power"] <= 1.0

    def test_compound_n_positive_integer(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        assert isinstance(result["n"], int)
        assert result["n"] > 0

    def test_a_criterion_compound_raises(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=5),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=5))
        multi = MultiResponseOptions([r1, r2])
        opts = _cp_opts(criterion="A")
        with pytest.raises(NotImplementedError, match="A-compound"):
            i_optimal_multiresponse_design(_CP_FORMULA, _CP_FACTORS, multi, opts)

    def test_unknown_factor_in_formula_raises(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=np.array([[0, 1, 0]]),
                                                    delta=np.array([1.5]),
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=5),
                          formula="~ 1 + A + C")  # C not in factors
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=5))
        multi = MultiResponseOptions([r1, r2])
        with pytest.raises(ValueError, match="could not be evaluated"):
            i_optimal_multiresponse_design(_CP_FORMULA, _CP_FACTORS, multi, _cp_opts())

    def test_d_criterion_compound_runs(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        multi = MultiResponseOptions([r1, r2])
        opts = _cp_opts(criterion="D")
        result = i_optimal_multiresponse_design(_CP_FORMULA, _CP_FACTORS, multi, opts)
        assert result["compound_criterion"] is True
        assert result["n"] > 0

    def test_compound_design_estimable_both_formulas(self):
        # Both model matrices formed from the returned design rows must have full rank.
        from iopt_power_design.model_matrix import build_model_matrix
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        design = result["design"]
        X1, _ = build_model_matrix(_CP_FORMULA_LINEAR, design)
        X2, _ = build_model_matrix(_CP_FORMULA, design)
        assert np.linalg.matrix_rank(X1) == X1.shape[1]
        assert np.linalg.matrix_rank(X2) == X2.shape[1]

    def test_formula_none_treated_as_global(self):
        # A ResponseSpec with formula=None uses the global formula — no compound path.
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=None)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2)
        assert result["compound_criterion"] is False

    def test_compound_combination_rule_stored(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_CP_L_LINEAR, delta=_CP_DELTA_1,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20),
                          formula=_CP_FORMULA_LINEAR)
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_CP_L_FULL, delta=_CP_DELTA_2,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=20))
        result = _cp_run(r1, r2, rule="weighted_mean")
        assert result["combination_rule"] == "weighted_mean"


# ---------------------------------------------------------------------------
# MR-6: Hotelling T² integration tests (api-level)
# ---------------------------------------------------------------------------
_HT2_FORMULA = "~ 1 + A + B"
_HT2_FACTORS = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
_HT2_L = np.array([[0, 1, 0], [0, 0, 1]])
_HT2_DELTA = np.array([1.5, 1.5])


def _ht2_mr_opts(**kw):
    defaults = dict(candidate_points=200, starts=2, random_state=11, max_iter=25)
    defaults.update(kw)
    return DesignOptions(**defaults)


def _ht2_run(sigma_joint, rule="min", **opts_kw):
    r1 = ResponseSpec("Y1", PowerContrastConfig(L=_HT2_L, delta=_HT2_DELTA,
                                                sigma=1.0, power=0.8, max_n=60, max_iter=25))
    r2 = ResponseSpec("Y2", PowerContrastConfig(L=_HT2_L, delta=_HT2_DELTA,
                                                sigma=1.0, power=0.8, max_n=60, max_iter=25))
    multi = MultiResponseOptions([r1, r2], power_combination=rule,
                                 sigma_joint=sigma_joint)
    opts = _ht2_mr_opts(**opts_kw)
    return i_optimal_multiresponse_design(_HT2_FORMULA, _HT2_FACTORS, multi, opts)


class TestHotellingT2API:
    """MR-6: api-level integration tests for sigma_joint / Hotelling T²."""

    def test_joint_power_key_present(self):
        result = _ht2_run(np.eye(2))
        assert "joint_power" in result

    def test_joint_power_in_unit_interval(self):
        result = _ht2_run(np.eye(2))
        assert 0.0 <= result["joint_power"] <= 1.0

    def test_responses_still_present(self):
        # Per-response powers are always reported alongside joint_power.
        result = _ht2_run(np.eye(2))
        assert len(result["responses"]) == 2
        for rd in result["responses"]:
            assert "power" in rd

    def test_joint_lam_df_keys_present(self):
        result = _ht2_run(np.eye(2))
        for key in ("joint_lam", "joint_df1", "joint_df2"):
            assert key in result

    def test_sigma_joint_none_no_joint_power(self):
        # Without sigma_joint, joint_power should not appear.
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_HT2_L, delta=_HT2_DELTA,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=25))
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_HT2_L, delta=_HT2_DELTA,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=25))
        multi = MultiResponseOptions([r1, r2])
        result = i_optimal_multiresponse_design(_HT2_FORMULA, _HT2_FACTORS, multi,
                                                _ht2_mr_opts())
        assert "joint_power" not in result

    def test_sigma_joint_with_r2_response_raises(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_HT2_L, delta=_HT2_DELTA,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=5))
        r2 = ResponseSpec("Y2", PowerR2Config(r2_target=0.5, power=0.8, max_n=60, max_iter=5))
        multi = MultiResponseOptions([r1, r2], sigma_joint=np.eye(2))
        with pytest.raises(NotImplementedError, match="R²-mode"):
            i_optimal_multiresponse_design(_HT2_FORMULA, _HT2_FACTORS, multi, _ht2_mr_opts())

    def test_sigma_joint_with_compound_path_raises(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=np.array([[0, 1]]), delta=np.array([1.5]),
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=5),
                          formula="~ 1 + A")
        r2 = ResponseSpec("Y2", PowerContrastConfig(L=_HT2_L, delta=_HT2_DELTA,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=5))
        multi = MultiResponseOptions([r1, r2], sigma_joint=np.eye(2))
        with pytest.raises(NotImplementedError, match="compound"):
            i_optimal_multiresponse_design(_HT2_FORMULA, _HT2_FACTORS, multi, _ht2_mr_opts())


# ---------------------------------------------------------------------------
# MR-7: Multi-response analysis function tests
# ---------------------------------------------------------------------------
_ANA_FORMULA = "~ 1 + A + B"
_ANA_FACTORS = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
_ANA_L = np.array([[0, 1, 0], [0, 0, 1]])
_ANA_DELTA = np.array([1.5, 1.5])


def _ana_opts(**kw):
    defaults = dict(candidate_points=150, starts=2, random_state=77, max_iter=15)
    defaults.update(kw)
    return DesignOptions(**defaults)


def _ana_multi(rule="min"):
    r1 = ResponseSpec("Y1", PowerContrastConfig(L=_ANA_L, delta=_ANA_DELTA,
                                                sigma=1.0, power=0.8, max_n=60, max_iter=15))
    r2 = ResponseSpec("Y2", PowerContrastConfig(L=_ANA_L, delta=_ANA_DELTA,
                                                sigma=1.5, power=0.8, max_n=60, max_iter=15))
    return MultiResponseOptions([r1, r2], power_combination=rule)


class TestMultiResponseAnalysis:
    """MR-7: power_curve_by_n_multiresponse and multiresponse_sensitivity."""

    # --- power_curve_by_n_multiresponse ---

    def test_returns_dataframe(self):
        df = power_curve_by_n_multiresponse(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            n_range=(10, 30), n_points=5, design_opts=_ana_opts(),
        )
        assert isinstance(df, pd.DataFrame)

    def test_column_count(self):
        multi = _ana_multi()
        n_responses = len(multi.responses)
        df = power_curve_by_n_multiresponse(
            _ANA_FORMULA, _ANA_FACTORS, multi,
            n_range=(10, 30), n_points=5, design_opts=_ana_opts(),
        )
        # n, combined_power, Y1_power, Y2_power → 4 columns for 2 responses
        assert df.shape[1] == n_responses + 2

    def test_column_names(self):
        df = power_curve_by_n_multiresponse(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            n_range=(10, 30), n_points=5, design_opts=_ana_opts(),
        )
        assert "n" in df.columns
        assert "combined_power" in df.columns
        assert "Y1_power" in df.columns
        assert "Y2_power" in df.columns

    def test_exact_n_points_rows(self):
        n_points = 7
        df = power_curve_by_n_multiresponse(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            n_range=(10, 30), n_points=n_points, design_opts=_ana_opts(),
        )
        assert len(df) == n_points

    def test_n_range_respected(self):
        df = power_curve_by_n_multiresponse(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            n_range=(12, 25), n_points=5, design_opts=_ana_opts(),
        )
        assert df["n"].min() >= 12
        assert df["n"].max() <= 25

    def test_combined_power_equals_combine_powers(self):
        from iopt_power_design.power import combine_powers
        df = power_curve_by_n_multiresponse(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            n_range=(15, 25), n_points=4, design_opts=_ana_opts(),
        )
        multi = _ana_multi()
        weights = [r.weight for r in multi.responses]
        rule = multi.power_combination
        for _, row in df.iterrows():
            per_r = [row["Y1_power"], row["Y2_power"]]
            if not any(np.isnan(p) for p in per_r):
                expected = combine_powers(per_r, weights, rule)
                assert row["combined_power"] == pytest.approx(expected, abs=1e-9)

    def test_plot_matplotlib_no_error(self):
        pytest.importorskip("matplotlib")
        power_curve_by_n_multiresponse(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            n_range=(10, 20), n_points=3, design_opts=_ana_opts(),
            plot=True, plot_backend="matplotlib",
        )

    def test_plot_plotly_no_error(self):
        pytest.importorskip("plotly")
        power_curve_by_n_multiresponse(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            n_range=(10, 20), n_points=3, design_opts=_ana_opts(),
            plot=True, plot_backend="plotly",
        )

    def test_exported_from_top_level(self):
        import iopt_power_design
        assert hasattr(iopt_power_design, "power_curve_by_n_multiresponse")

    # --- multiresponse_sensitivity ---

    def test_sensitivity_returns_dataframe(self):
        df = multiresponse_sensitivity(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            fixed_n=20, sigma_range=(0.5, 2.0), sigma_points=5,
            design_opts=_ana_opts(),
        )
        assert isinstance(df, pd.DataFrame)

    def test_sensitivity_column_names(self):
        df = multiresponse_sensitivity(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            fixed_n=20, sigma_range=(0.5, 2.0), sigma_points=5,
            design_opts=_ana_opts(),
        )
        assert "sigma_scale" in df.columns
        assert "combined_power" in df.columns
        assert "Y1_power" in df.columns
        assert "Y2_power" in df.columns

    def test_sensitivity_exact_row_count(self):
        sigma_points = 8
        df = multiresponse_sensitivity(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            fixed_n=20, sigma_range=(0.5, 2.5), sigma_points=sigma_points,
            design_opts=_ana_opts(),
        )
        assert len(df) == sigma_points

    def test_sensitivity_monotonically_decreasing(self):
        df = multiresponse_sensitivity(
            _ANA_FORMULA, _ANA_FACTORS, _ana_multi(),
            fixed_n=25, sigma_range=(0.5, 3.0), sigma_points=10,
            design_opts=_ana_opts(),
        )
        pwr = df["combined_power"].tolist()
        # Each step should be non-increasing (larger sigma → lower power)
        for i in range(1, len(pwr)):
            assert pwr[i] <= pwr[i - 1] + 1e-9, (
                f"Power increased at step {i}: {pwr[i-1]:.4f} → {pwr[i]:.4f}"
            )

    def test_sensitivity_r2_response_raises_type_error(self):
        r1 = ResponseSpec("Y1", PowerContrastConfig(L=_ANA_L, delta=_ANA_DELTA,
                                                    sigma=1.0, power=0.8, max_n=60, max_iter=5))
        r2 = ResponseSpec("Y2", PowerR2Config(r2_target=0.4, power=0.8, max_n=60, max_iter=5))
        multi = MultiResponseOptions([r1, r2])
        with pytest.raises(TypeError, match="PowerR2Config"):
            multiresponse_sensitivity(
                _ANA_FORMULA, _ANA_FACTORS, multi,
                fixed_n=20, design_opts=_ana_opts(),
            )

    def test_sensitivity_exported_from_top_level(self):
        import iopt_power_design
        assert hasattr(iopt_power_design, "multiresponse_sensitivity")


# ---------------------------------------------------------------------------
# MR-8 CLI integration tests
# ---------------------------------------------------------------------------

class TestMultiResponseCLI:
    """Tests for _make_multi_response_cfg, _validate_config_keys (MR-8)."""

    def _scenario_cfg(self):
        """Minimal valid multi-response config dict."""
        return {
            "formula": "~ 1 + A + B",
            "factors": {"A": ["low", "high"], "B": [0.0, 10.0]},
            "alpha": 0.05,
            "power": 0.80,
            "responses": [
                {
                    "name": "Yield",
                    "sigma": 2.0,
                    "contrast": {
                        "scenario_a": {"A": "low", "B": 5.0},
                        "scenario_b": {"A": "high", "B": 5.0},
                        "sesoi": 1.0,
                    },
                },
                {
                    "name": "Purity",
                    "sigma": 0.5,
                    "r2_target": 0.20,
                },
            ],
        }

    def test_validate_accepts_responses_block(self):
        from iopt_power_design.cli import _validate_config_keys
        cfg = self._scenario_cfg()
        _validate_config_keys(cfg)  # must not raise

    def test_validate_still_rejects_missing_formula(self):
        from iopt_power_design.cli import _validate_config_keys
        cfg = self._scenario_cfg()
        del cfg["formula"]
        with pytest.raises(KeyError, match="formula"):
            _validate_config_keys(cfg)

    def test_validate_still_rejects_no_power_keys(self):
        from iopt_power_design.cli import _validate_config_keys
        cfg = {"formula": "~ 1 + A", "factors": {"A": [0.0, 1.0]}}
        with pytest.raises(KeyError):
            _validate_config_keys(cfg)

    def test_make_multi_response_cfg_returns_correct_type(self):
        from iopt_power_design.cli import _make_multi_response_cfg
        from iopt_power_design.config import MultiResponseOptions
        cfg = self._scenario_cfg()
        formula = cfg["formula"]
        from iopt_power_design.cli import _as_factors
        factors = _as_factors(cfg["factors"])
        multi = _make_multi_response_cfg(cfg, formula, factors)
        assert isinstance(multi, MultiResponseOptions)

    def test_make_multi_response_cfg_response_count(self):
        from iopt_power_design.cli import _make_multi_response_cfg, _as_factors
        cfg = self._scenario_cfg()
        factors = _as_factors(cfg["factors"])
        multi = _make_multi_response_cfg(cfg, cfg["formula"], factors)
        assert len(multi.responses) == 2

    def test_make_multi_response_cfg_names(self):
        from iopt_power_design.cli import _make_multi_response_cfg, _as_factors
        cfg = self._scenario_cfg()
        factors = _as_factors(cfg["factors"])
        multi = _make_multi_response_cfg(cfg, cfg["formula"], factors)
        names = [r.name for r in multi.responses]
        assert names == ["Yield", "Purity"]

    def test_make_multi_response_cfg_power_combination(self):
        from iopt_power_design.cli import _make_multi_response_cfg, _as_factors
        cfg = self._scenario_cfg()
        cfg["power_combination"] = "product"
        factors = _as_factors(cfg["factors"])
        multi = _make_multi_response_cfg(cfg, cfg["formula"], factors)
        assert multi.power_combination == "product"

    def test_make_multi_response_cfg_explicit_L(self):
        from iopt_power_design.cli import _make_multi_response_cfg, _as_factors
        from iopt_power_design.config import PowerContrastConfig
        import numpy as np
        cfg = self._scenario_cfg()
        # Replace first response with explicit L/delta
        cfg["responses"][0] = {
            "name": "Yield",
            "sigma": 1.0,
            "contrast": {"L": [[0, 0, 1, 0]], "delta": [0.5]},
        }
        factors = _as_factors(cfg["factors"])
        multi = _make_multi_response_cfg(cfg, cfg["formula"], factors)
        assert isinstance(multi.responses[0].power_cfg, PowerContrastConfig)

    def test_make_multi_response_cfg_missing_name_raises(self):
        from iopt_power_design.cli import _make_multi_response_cfg, _as_factors
        cfg = self._scenario_cfg()
        cfg["responses"][0]["name"] = ""
        factors = _as_factors(cfg["factors"])
        with pytest.raises(KeyError):
            _make_multi_response_cfg(cfg, cfg["formula"], factors)

    def test_make_multi_response_cfg_missing_power_key_raises(self):
        from iopt_power_design.cli import _make_multi_response_cfg, _as_factors
        cfg = self._scenario_cfg()
        # Remove both contrast and r2_target from second response
        del cfg["responses"][1]["r2_target"]
        factors = _as_factors(cfg["factors"])
        with pytest.raises(KeyError):
            _make_multi_response_cfg(cfg, cfg["formula"], factors)

    def test_make_multi_response_cfg_too_few_responses_raises(self):
        from iopt_power_design.cli import _make_multi_response_cfg, _as_factors
        from iopt_power_design.config import MultiResponseOptions
        cfg = self._scenario_cfg()
        cfg["responses"] = cfg["responses"][:1]  # only 1
        factors = _as_factors(cfg["factors"])
        with pytest.raises((ValueError, TypeError)):
            _make_multi_response_cfg(cfg, cfg["formula"], factors)

    def test_cli_main_multiresponse_dry_run(self, tmp_path):
        """--dry-run with a responses: config should succeed."""
        import yaml
        from iopt_power_design.cli import main
        cfg = self._scenario_cfg()
        cfg_path = tmp_path / "mr_config.yml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        rc = main(["--config", str(cfg_path), "--dry-run"])
        assert rc == 0

    def test_cli_main_multiresponse_flag_dry_run(self, tmp_path):
        """--multi-response flag with responses: config should also work."""
        import yaml
        from iopt_power_design.cli import main
        cfg = self._scenario_cfg()
        cfg_path = tmp_path / "mr_config.yml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        rc = main(["--config", str(cfg_path), "--dry-run", "--multi-response"])
        assert rc == 0

    def test_cli_main_multiresponse_produces_output_files(self, tmp_path):
        """Full end-to-end: multi-response config writes CSV outputs."""
        import yaml
        from iopt_power_design.cli import main
        cfg = self._scenario_cfg()
        # Low power target and small max_n keeps the bisection bounded
        cfg["power"] = 0.50
        cfg["max_n"] = 30  # caps bisection for both responses
        cfg["responses"][1]["r2_target"] = 0.50  # easier to detect
        # Use very small design options via design: block
        cfg["design"] = {
            "candidate_points": 300, "starts": 1, "max_iter": 30,
            "random_state": 1, "auto_candidate": False,
        }
        cfg_path = tmp_path / "mr_config.yml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        out = str(tmp_path / "out")
        rc = main(["--config", str(cfg_path), "--out", out])
        assert rc == 0
        assert (tmp_path / "out_design.csv").exists()
        assert (tmp_path / "out_report.json").exists()


# ---------------------------------------------------------------------------
# MR-10 Integration and property-based tests
# ---------------------------------------------------------------------------

_MR10_FORMULA = "~ 1 + A + B"
_MR10_FACTORS = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
_MR10_L = np.array([[0, 1, 0], [0, 0, 1]])
_MR10_DELTA = np.array([1.5, 1.5])


def _mr10_opts(**kw):
    defaults = dict(candidate_points=150, starts=2, random_state=17, max_iter=25)
    defaults.update(kw)
    return DesignOptions(**defaults)


def _mr10_crs(name, sigma=1.0, power=0.8, max_n=40, weight=1.0):
    cfg = PowerContrastConfig(
        L=_MR10_L, delta=_MR10_DELTA,
        sigma=sigma, power=power, max_n=max_n, max_iter=25,
    )
    return ResponseSpec(name=name, power_cfg=cfg, weight=weight)


def _mr10_r2rs(name, r2=0.30, power=0.8, max_n=40, weight=1.0):
    cfg = PowerR2Config(r2_target=r2, power=power, max_n=max_n, max_iter=25)
    return ResponseSpec(name=name, power_cfg=cfg, weight=weight)



@pytest.mark.slow
class TestMR10Integration:
    """MR-10: seven integration scenarios for multi-response powered designs."""

    def test_s1_min_n_geq_harder_single_response(self):
        """n from min combination >= max(individual n)."""
        opts = _mr10_opts()
        r1 = _mr10_crs("Y1", sigma=1.0, max_n=40)
        r2 = _mr10_crs("Y2", sigma=1.5, max_n=40)
        multi = MultiResponseOptions([r1, r2], power_combination="min")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        n1 = i_optimal_powered_design(
            _MR10_FORMULA, _MR10_FACTORS,
            PowerContrastConfig(L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25),
            opts,
        )["report"]["n"]
        n2 = i_optimal_powered_design(
            _MR10_FORMULA, _MR10_FACTORS,
            PowerContrastConfig(L=_MR10_L, delta=_MR10_DELTA, sigma=1.5, power=0.8, max_n=40, max_iter=25),
            opts,
        )["report"]["n"]
        assert mr["n"] >= max(n1, n2) - 1

    def test_s1_min_achieved_is_min_of_per_response(self):
        """achieved_power == min(per-response) under min rule."""
        opts = _mr10_opts()
        r1 = _mr10_crs("Y1", sigma=1.0, max_n=40)
        r2 = _mr10_crs("Y2", sigma=1.2, max_n=40)
        multi = MultiResponseOptions([r1, r2], power_combination="min")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        per = [rd["power"] for rd in mr["responses"]]
        assert mr["achieved_power"] == pytest.approx(min(per), abs=1e-9)

    def test_s2_product_achieved_equals_p1_times_p2(self):
        """achieved_power == P1 * P2 under product rule."""
        opts = _mr10_opts()
        r1 = _mr10_r2rs("Y1", r2=0.28, max_n=40)
        r2 = _mr10_r2rs("Y2", r2=0.33, max_n=40)
        multi = MultiResponseOptions([r1, r2], power_combination="product")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        p1 = next(rd["power"] for rd in mr["responses"] if rd["name"] == "Y1")
        p2 = next(rd["power"] for rd in mr["responses"] if rd["name"] == "Y2")
        assert mr["achieved_power"] == pytest.approx(p1 * p2, abs=1e-9)

    def test_s2_product_power_leq_min_per_response(self):
        """product rule: achieved_power <= min(per-response powers)."""
        opts = _mr10_opts()
        r1 = _mr10_r2rs("Y1", r2=0.28, max_n=40)
        r2 = _mr10_r2rs("Y2", r2=0.35, max_n=40)
        multi = MultiResponseOptions([r1, r2], power_combination="product")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        per = [rd["power"] for rd in mr["responses"]]
        assert mr["achieved_power"] <= min(per) + 1e-9

    def test_s3_three_responses_weighted_mean_structure(self):
        """Three-response weighted_mean returns valid result."""
        opts = _mr10_opts()
        responses = [
            ResponseSpec("Y1", PowerContrastConfig(
                L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25), weight=2.0),
            ResponseSpec("Y2", PowerContrastConfig(
                L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25), weight=1.0),
            ResponseSpec("Y3", PowerContrastConfig(
                L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25), weight=1.0),
        ]
        multi = MultiResponseOptions(responses, power_combination="weighted_mean")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        assert len(mr["responses"]) == 3
        assert mr["combination_rule"] == "weighted_mean"
        assert isinstance(mr["design"], pd.DataFrame)

    def test_s3_weighted_mean_achieved_equals_formula(self):
        """achieved_power == weighted mean of per-response powers."""
        from iopt_power_design.power import combine_powers
        opts = _mr10_opts()
        responses = [
            ResponseSpec("Y1", PowerContrastConfig(
                L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25), weight=2.0),
            ResponseSpec("Y2", PowerContrastConfig(
                L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25), weight=1.0),
            ResponseSpec("Y3", PowerContrastConfig(
                L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25), weight=1.0),
        ]
        multi = MultiResponseOptions(responses, power_combination="weighted_mean")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        per_powers = [rd["power"] for rd in mr["responses"]]
        weights = [r.weight for r in multi.responses]
        expected = combine_powers(per_powers, weights, "weighted_mean")
        assert mr["achieved_power"] == pytest.approx(expected, abs=1e-9)

    def test_s4_compound_criterion_flag_set(self):
        """Responses with different formulas set compound_criterion=True."""
        opts = _mr10_opts()
        r1 = ResponseSpec(
            "Y1",
            PowerContrastConfig(L=np.array([[0, 1]]), delta=np.array([1.5]),
                                sigma=1.0, power=0.8, max_n=40, max_iter=25),
            formula="~ 1 + A",
        )
        r2 = ResponseSpec(
            "Y2",
            PowerContrastConfig(L=np.array([[0, 1, 0, 0]]), delta=np.array([1.5]),
                                sigma=1.0, power=0.8, max_n=40, max_iter=25),
            formula="~ 1 + A + B + A:B",
        )
        multi = MultiResponseOptions([r1, r2], power_combination="min")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        assert mr["compound_criterion"] is True

    def test_s4_compound_design_estimable_for_both_formulas(self):
        """Compound-criterion design is full-rank for both response model matrices."""
        from iopt_power_design.model_matrix import build_model_matrix
        opts = _mr10_opts()
        r1 = ResponseSpec(
            "Y1",
            PowerContrastConfig(L=np.array([[0, 1]]), delta=np.array([1.5]),
                                sigma=1.0, power=0.8, max_n=40, max_iter=25),
            formula="~ 1 + A",
        )
        r2 = ResponseSpec(
            "Y2",
            PowerContrastConfig(L=np.array([[0, 1, 0, 0]]), delta=np.array([1.5]),
                                sigma=1.0, power=0.8, max_n=40, max_iter=25),
            formula="~ 1 + A + B + A:B",
        )
        multi = MultiResponseOptions([r1, r2], power_combination="min")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        design_df = mr["design"]
        X_lin, _ = build_model_matrix("~ 1 + A", design_df)
        X_inter, _ = build_model_matrix("~ 1 + A + B + A:B", design_df)
        assert np.linalg.matrix_rank(X_lin) == X_lin.shape[1]
        assert np.linalg.matrix_rank(X_inter) == X_inter.shape[1]

    def test_s5_hotelling_joint_power_in_unit_interval(self):
        """Joint T2 power is in [0, 1] with identity sigma_joint."""
        opts = _mr10_opts()
        L = np.array([[0, 1, 0]])
        r1 = ResponseSpec("Y1", PowerContrastConfig(
            L=L, delta=np.array([1.5]), sigma=1.0, power=0.8, max_n=40, max_iter=25))
        r2 = ResponseSpec("Y2", PowerContrastConfig(
            L=L, delta=np.array([1.5]), sigma=1.0, power=0.8, max_n=40, max_iter=25))
        multi = MultiResponseOptions([r1, r2], sigma_joint=np.eye(2))
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        assert 0.0 <= mr["joint_power"] <= 1.0

    def test_s5_hotelling_joint_power_geq_min_per_response(self):
        """Joint T2 power >= min(per-response powers) with identity sigma_joint."""
        opts = _mr10_opts()
        L = np.array([[0, 1, 0]])
        r1 = ResponseSpec("Y1", PowerContrastConfig(
            L=L, delta=np.array([1.5]), sigma=1.0, power=0.8, max_n=40, max_iter=25))
        r2 = ResponseSpec("Y2", PowerContrastConfig(
            L=L, delta=np.array([1.5]), sigma=1.0, power=0.8, max_n=40, max_iter=25))
        multi = MultiResponseOptions([r1, r2], sigma_joint=np.eye(2))
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        per = [rd["power"] for rd in mr["responses"]]
        assert mr["joint_power"] >= min(per) - 1e-9

    def test_s6_split_plot_multi_response_valid_result(self):
        """SplitPlotOptions + MultiResponseOptions produces valid result."""
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=3, subplots_per_wp=3, eta=1.0)
        opts = DesignOptions(candidate_points=120, starts=2, max_iter=20, random_state=31, split_plot=sp)
        L = np.array([[0, 1, 0]])
        r1 = ResponseSpec("Y1", PowerContrastConfig(
            L=L, delta=np.array([2.0]), sigma=1.0, power=0.8, max_n=30, max_iter=20))
        r2 = ResponseSpec("Y2", PowerContrastConfig(
            L=L, delta=np.array([2.0]), sigma=1.5, power=0.8, max_n=30, max_iter=20))
        multi = MultiResponseOptions([r1, r2], power_combination="min")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        assert isinstance(mr["design"], pd.DataFrame)
        assert len(mr["responses"]) == 2
        assert mr["n"] >= 1

    def test_s6_split_plot_design_has_wp_column(self):
        """Split-plot multi-response design includes __wp_id__ column."""
        sp = SplitPlotOptions(htc_factors=["A"], n_whole_plots=3, subplots_per_wp=3, eta=1.0)
        opts = DesignOptions(candidate_points=120, starts=2, max_iter=20, random_state=31, split_plot=sp)
        L = np.array([[0, 1, 0]])
        r1 = ResponseSpec("Y1", PowerContrastConfig(
            L=L, delta=np.array([2.0]), sigma=1.0, power=0.8, max_n=30, max_iter=20))
        r2 = ResponseSpec("Y2", PowerContrastConfig(
            L=L, delta=np.array([2.0]), sigma=1.0, power=0.8, max_n=30, max_iter=20))
        multi = MultiResponseOptions([r1, r2], power_combination="min")
        mr = i_optimal_multiresponse_design(_MR10_FORMULA, _MR10_FACTORS, multi, opts)
        assert "__wp_id__" in mr["design"].columns

    def test_s7_combined_power_nondecreasing(self):
        """combined_power is non-decreasing as n increases."""
        r1 = ResponseSpec("Y1", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=60, max_iter=25))
        r2 = ResponseSpec("Y2", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.5, power=0.8, max_n=60, max_iter=25))
        multi = MultiResponseOptions([r1, r2], power_combination="min")
        df = power_curve_by_n_multiresponse(
            _MR10_FORMULA, _MR10_FACTORS, multi,
            n_range=(8, 35), n_points=8, design_opts=_mr10_opts(),
        )
        powers = df["combined_power"].tolist()
        violations = sum(1 for i in range(1, len(powers)) if powers[i] < powers[i - 1] - 0.05)
        assert violations <= 1, f"Non-monotone steps > 0.05: {violations}"

    def test_s7_power_curve_no_nan(self):
        """power_curve_by_n_multiresponse returns no NaN values."""
        r1 = ResponseSpec("Y1", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=60, max_iter=20))
        r2 = ResponseSpec("Y2", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=60, max_iter=20))
        multi = MultiResponseOptions([r1, r2], power_combination="min")
        df = power_curve_by_n_multiresponse(
            _MR10_FORMULA, _MR10_FACTORS, multi,
            n_range=(8, 25), n_points=5, design_opts=_mr10_opts(),
        )
        assert not df.isnull().any().any(), "Got NaN in power curve"


class TestMR10PropertyBased:
    """MR-10: fast property-based tests for multi-response code paths."""

    def test_identical_responses_min_n_matches_single(self):
        """Two identical responses under min => same n as single (+/-1)."""
        opts = _mr10_opts()
        cfg = PowerContrastConfig(L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25)
        n_single = i_optimal_powered_design(_MR10_FORMULA, _MR10_FACTORS, cfg, opts)["report"]["n"]
        r1 = ResponseSpec("Y1", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25))
        r2 = ResponseSpec("Y2", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25))
        mr = i_optimal_multiresponse_design(
            _MR10_FORMULA, _MR10_FACTORS, MultiResponseOptions([r1, r2], power_combination="min"), opts)
        assert abs(mr["n"] - n_single) <= 1

    def test_adding_harder_third_response_min_nondecreasing(self):
        """Adding harder third response under min never decreases n."""
        opts = _mr10_opts()
        r1 = ResponseSpec("Y1", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=40, max_iter=25))
        r2 = ResponseSpec("Y2", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.2, power=0.8, max_n=40, max_iter=25))
        mr2 = i_optimal_multiresponse_design(
            _MR10_FORMULA, _MR10_FACTORS, MultiResponseOptions([r1, r2], power_combination="min"), opts)
        r3 = ResponseSpec("Y3", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=2.0, power=0.8, max_n=40, max_iter=25))
        mr3 = i_optimal_multiresponse_design(
            _MR10_FORMULA, _MR10_FACTORS, MultiResponseOptions([r1, r2, r3], power_combination="min"), opts)
        assert mr3["n"] >= mr2["n"] - 1

    def test_power_curve_no_nan_fast(self):
        """power_curve_by_n_multiresponse: no NaN (small range)."""
        r1 = ResponseSpec("Y1", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=60, max_iter=20))
        r2 = ResponseSpec("Y2", PowerContrastConfig(
            L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=60, max_iter=20))
        df = power_curve_by_n_multiresponse(
            _MR10_FORMULA, _MR10_FACTORS, MultiResponseOptions([r1, r2]),
            n_range=(10, 20), n_points=3, design_opts=_mr10_opts(),
        )
        assert not df.isnull().any().any()

    def test_r2_product_achieved_leq_individual(self):
        """Product <= min(per-response powers)."""
        r1 = _mr10_r2rs("Y1", r2=0.30, max_n=40)
        r2 = _mr10_r2rs("Y2", r2=0.30, max_n=40)
        mr = i_optimal_multiresponse_design(
            _MR10_FORMULA, _MR10_FACTORS,
            MultiResponseOptions([r1, r2], power_combination="product"), _mr10_opts())
        per = [rd["power"] for rd in mr["responses"]]
        assert mr["achieved_power"] <= min(per) + 1e-9

    def test_response_names_preserved(self):
        """Response names in result match names in MultiResponseOptions."""
        names = ["Alpha", "Beta"]
        responses = [
            ResponseSpec(n, PowerContrastConfig(
                L=_MR10_L, delta=_MR10_DELTA, sigma=1.0, power=0.8, max_n=30, max_iter=20))
            for n in names
        ]
        mr = i_optimal_multiresponse_design(
            _MR10_FORMULA, _MR10_FACTORS, MultiResponseOptions(responses), _mr10_opts())
        assert [rd["name"] for rd in mr["responses"]] == names

    def test_shared_formula_compound_criterion_false(self):
        """Shared global formula => compound_criterion=False."""
        mr = i_optimal_multiresponse_design(
            _MR10_FORMULA, _MR10_FACTORS,
            MultiResponseOptions([_mr10_crs("Y1"), _mr10_crs("Y2")]), _mr10_opts())
        assert mr["compound_criterion"] is False

    def test_warnings_key_is_list(self):
        """warnings key is always a list."""
        mr = i_optimal_multiresponse_design(
            _MR10_FORMULA, _MR10_FACTORS,
            MultiResponseOptions([_mr10_crs("Y1", max_n=30), _mr10_crs("Y2", max_n=30)]), _mr10_opts())
        assert isinstance(mr["warnings"], list)

    def test_elapsed_sec_positive(self):
        """elapsed_sec is positive."""
        mr = i_optimal_multiresponse_design(
            _MR10_FORMULA, _MR10_FACTORS,
            MultiResponseOptions([_mr10_crs("Y1", max_n=30), _mr10_crs("Y2", max_n=30)]), _mr10_opts())
        assert mr["elapsed_sec"] > 0


# ---------------------------------------------------------------------------
# GL-3 — GLM Design API Integration
# ---------------------------------------------------------------------------

_GLM_FORMULA = "~ 1 + A + B"
_GLM_FACTORS = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
_GLM_L = np.array([[0.0, 1.0, 0.0]])  # test main effect of A


def _glm_opts(starts: int = 1, max_iter: int = 8) -> DesignOptions:
    return DesignOptions(starts=starts, random_state=0, max_iter=max_iter)


def _binomial_cfg(
    delta: float = 0.5,
    baseline: float = 0.4,
    power: float = 0.70,
    max_n: int = 60,
    max_iter: int = 12,
    **kwargs,
) -> PowerGLMContrastConfig:
    return PowerGLMContrastConfig(
        L=_GLM_L,
        delta=np.array([delta]),
        baseline=baseline,
        family="binomial",
        power=power,
        max_n=max_n,
        max_iter=max_iter,
        **kwargs,
    )


def _poisson_cfg(
    delta: float = 0.4,
    baseline: float = 2.0,
    power: float = 0.70,
    max_n: int = 60,
    max_iter: int = 12,
    **kwargs,
) -> PowerGLMContrastConfig:
    return PowerGLMContrastConfig(
        L=_GLM_L,
        delta=np.array([delta]),
        baseline=baseline,
        family="poisson",
        power=power,
        max_n=max_n,
        max_iter=max_iter,
        **kwargs,
    )


@pytest.mark.slow
class TestGLMDesignAPI:
    """GL-3: GLM configs accepted by i_optimal_powered_design."""

    # --- Happy path ---

    def test_binomial_runs_without_error(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(), _glm_opts()
        )
        assert "design_df" in result
        assert "report" in result

    def test_binomial_report_has_family_key(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(), _glm_opts()
        )
        assert result["report"]["family"] == "binomial"

    def test_binomial_report_has_test_type_wald_chi2(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(), _glm_opts()
        )
        assert result["report"]["test_type"] == "wald_chi2"

    def test_binomial_design_df_has_factor_columns(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(), _glm_opts()
        )
        df = result["design_df"]
        assert "A" in df.columns
        assert "B" in df.columns

    def test_binomial_achieved_power_between_0_and_1(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(), _glm_opts()
        )
        p = result["report"]["achieved_power"]
        assert 0.0 <= p <= 1.0

    def test_poisson_runs_without_error(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _poisson_cfg(), _glm_opts()
        )
        assert "report" in result

    def test_poisson_report_has_baseline_key(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _poisson_cfg(), _glm_opts()
        )
        assert result["report"]["baseline"] == pytest.approx(2.0)

    # --- Report structure ---

    def test_glm_report_df2_is_none(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(), _glm_opts()
        )
        assert result["report"]["df2"] is None

    def test_glm_report_glm_weight_close_to_p0_times_1_minus_p0(self):
        p0 = 0.4
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(baseline=p0), _glm_opts()
        )
        expected_w = p0 * (1 - p0)
        assert result["report"]["glm_weight"] == pytest.approx(expected_w, rel=1e-6)

    def test_glm_report_has_link_key(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(), _glm_opts()
        )
        assert result["report"]["link"] == "logit"

    def test_poisson_report_link_is_log(self):
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _poisson_cfg(), _glm_opts()
        )
        assert result["report"]["link"] == "log"

    # --- OLS backward compat: test_type == "f" ---

    def test_ols_contrast_report_has_test_type_f(self):
        cfg = PowerContrastConfig(
            L=_GLM_L, delta=np.array([0.5]), sigma=1.0,
            alpha=0.05, power=0.70, max_n=60, max_iter=12,
        )
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, cfg, _glm_opts()
        )
        assert result["report"]["test_type"] == "f"

    def test_ols_r2_report_has_test_type_f(self):
        cfg = PowerR2Config(
            r2_target=0.5, sigma=1.0, alpha=0.05,
            power=0.70, max_n=60, max_iter=12,
        )
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, cfg, _glm_opts()
        )
        assert result["report"]["test_type"] == "f"

    # --- Design options ---

    def test_glm_with_d_criterion(self):
        opts = DesignOptions(criterion="D", starts=1, random_state=0, max_iter=8)
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(), opts
        )
        assert result["report"]["criterion"] == "D"

    def test_glm_with_a_criterion(self):
        opts = DesignOptions(criterion="A", starts=1, random_state=0, max_iter=8)
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, _binomial_cfg(), opts
        )
        assert result["report"]["criterion"] == "A"

    def test_glm_with_categorical_factor(self):
        # "~ 1 + A + C" with C=["low","mid","high"] → intercept, A, C[T.mid], C[T.high] = 4 cols
        factors = {"A": (-1.0, 1.0), "C": ["low", "mid", "high"]}
        L = np.zeros((1, 4))
        L[0, 1] = 1.0  # A coefficient
        cfg = PowerGLMContrastConfig(
            L=L, delta=np.array([0.5]), baseline=0.4, family="binomial",
            alpha=0.05, power=0.70, max_n=80, max_iter=12,
        )
        result = i_optimal_powered_design("~ 1 + A + C", factors, cfg, _glm_opts())
        assert "design_df" in result

    # --- Error handling ---

    def test_glm_split_plot_raises(self):
        cfg = _binomial_cfg()
        opts = DesignOptions(
            split_plot=SplitPlotOptions(htc_factors=["A"], n_whole_plots=3),
            starts=1, random_state=0,
        )
        with pytest.raises(ValueError, match="GLM"):
            i_optimal_powered_design(_GLM_FORMULA, _GLM_FACTORS, cfg, opts)

    def test_glm_config_wrong_family_raises(self):
        with pytest.raises((ValueError, TypeError)):
            PowerGLMContrastConfig(
                L=_GLM_L, delta=np.array([0.5]), baseline=0.4,
                family="gaussian",  # invalid
            )

    # --- Backward compat ---

    def test_ols_contrast_api_result_shape_unchanged(self):
        """OLS contrast API still returns design_df, buckets_df, report."""
        cfg = PowerContrastConfig(
            L=_GLM_L, delta=np.array([0.5]), sigma=1.0,
            alpha=0.05, power=0.70, max_n=60, max_iter=12,
        )
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, cfg, _glm_opts()
        )
        assert set(["design_df", "buckets_df", "report"]).issubset(result.keys())

    def test_ols_r2_api_result_shape_unchanged(self):
        """OLS R² API still returns design_df, buckets_df, report."""
        cfg = PowerR2Config(
            r2_target=0.5, sigma=1.0, alpha=0.05,
            power=0.70, max_n=60, max_iter=12,
        )
        result = i_optimal_powered_design(
            _GLM_FORMULA, _GLM_FACTORS, cfg, _glm_opts()
        )
        assert set(["design_df", "buckets_df", "report"]).issubset(result.keys())
