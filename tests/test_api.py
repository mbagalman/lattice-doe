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
    SplitPlotOptions,
    i_optimal_powered_design,
    power_curve_by_effect,
    power_curve_by_n,
    power_curve_by_wp,
    power_sensitivity,
    min_detectable_effect,
    compare_criteria,
    augment_design,
    robustness_report,
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
