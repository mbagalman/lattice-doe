# tests/test_factor_specs.py
"""Tests for discriminated factor specifications (UX-5)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lattice_doe.utils import (
    _CategoricalSpec,
    _ContinuousSpec,
    _spec_is_continuous,
    normalize_factors,
)


class TestNormalizeFactors:
    def test_categorical_dict_form(self):
        f = normalize_factors({"arm": {"type": "categorical", "levels": [0, 1]}})
        assert isinstance(f["arm"], _CategoricalSpec)
        assert not _spec_is_continuous(f["arm"])
        assert list(f["arm"]) == [0, 1]

    def test_continuous_dict_form(self):
        f = normalize_factors({"t": {"type": "continuous", "low": 0.0, "high": 5.0}})
        assert isinstance(f["t"], _ContinuousSpec)
        assert _spec_is_continuous(f["t"])
        assert tuple(f["t"]) == (0.0, 5.0)

    def test_legacy_forms_pass_through(self):
        f = normalize_factors({"a": [-1.0, 1.0], "b": ["x", "y", "z"]})
        assert _spec_is_continuous(f["a"])       # heuristic: two numerics
        assert not _spec_is_continuous(f["b"])   # three levels

    def test_numeric_binary_category_disambiguated(self):
        """The exact case the legacy heuristic cannot express."""
        legacy = normalize_factors({"arm": [0, 1]})
        assert _spec_is_continuous(legacy["arm"])          # legacy: continuous
        explicit = normalize_factors({"arm": {"type": "categorical", "levels": [0, 1]}})
        assert not _spec_is_continuous(explicit["arm"])    # explicit: categorical

    def test_bad_continuous_missing_bounds(self):
        with pytest.raises(ValueError, match="'low' and 'high'"):
            normalize_factors({"t": {"type": "continuous", "low": 0.0}})

    def test_bad_categorical_empty_levels(self):
        with pytest.raises(ValueError, match="non-empty 'levels'"):
            normalize_factors({"g": {"type": "categorical", "levels": []}})

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="unknown type"):
            normalize_factors({"g": {"type": "ordinal", "levels": [1, 2]}})


class TestTargetedDeprecationWarning:
    def test_warns_only_for_two_number_under_C(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_factors({"arm": [0, 1]}, formula="~ C(arm)")
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep) == 1
        assert "categorical" in str(dep[0].message)

    def test_no_warning_without_C_wrapper(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_factors({"t": [0.0, 10.0]}, formula="~ 1 + t")
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_explicit_form_never_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_factors(
                {"arm": {"type": "categorical", "levels": [0, 1]}},
                formula="~ C(arm)",
            )
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_three_level_numeric_list_does_not_warn(self):
        # A 3-element numeric list is already unambiguously categorical.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_factors({"g": [0, 1, 2]}, formula="~ C(g)")
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)


class TestDiscriminatedSpecsEndToEnd:
    def test_numeric_categorical_candidate_set(self):
        from lattice_doe.candidate import build_candidate
        f = normalize_factors({"arm": {"type": "categorical", "levels": [0, 1]}})
        cand = build_candidate(f, candidate_points=50, seed=0)
        assert sorted(cand["arm"].unique().tolist()) == [0, 1]
        assert len(cand) == 2  # two distinct cells, not a continuous sweep

    def test_numeric_categorical_through_find_optimal_design(self):
        from lattice_doe.api import find_optimal_design
        from lattice_doe.config import PowerContrastConfig, DesignOptions
        pc = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([1.5]),
            sigma=1.0, alpha=0.05, power=0.8, max_n=40,
        )
        out = find_optimal_design(
            "~ C(arm)", {"arm": {"type": "categorical", "levels": [0, 1]}},
            pc, DesignOptions(random_state=0, starts=1, candidate_points=50,
                              preallocate_categorical=True),
        )
        assert sorted(out["design_df"]["arm"].unique().tolist()) == [0, 1]

    def test_ambiguous_legacy_warns_through_api(self):
        """The ambiguous [0,1]-under-C() warning fires at the API boundary.

        (The legacy form then genuinely misbehaves — a continuous [0,1] under
        C() expands to one dummy per candidate point — which is exactly why
        the warning steers users to the explicit categorical form; we only
        assert the warning here and tolerate the downstream failure.)
        """
        from lattice_doe.api import find_optimal_design
        from lattice_doe.config import PowerR2Config, DesignOptions
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(Exception):
                find_optimal_design(
                    "~ C(g)", {"g": [0, 1]},
                    PowerR2Config(r2_target=0.3, alpha=0.05, power=0.6, max_n=20),
                    DesignOptions(random_state=0, starts=1, candidate_points=40),
                )
        assert any(
            issubclass(x.category, DeprecationWarning) and "g" in str(x.message)
            for x in w
        )


class TestDiscriminatedSpecsAnalysisBoundaries:
    """Contract tests: every exported factor-taking API must normalize the
    discriminated dict forms before candidate generation (P1). Regression for
    the bug where a continuous ``{"type","low","high"}`` dict was generated as
    a categorical iterable over its keys.
    """

    CONT = {"x": {"type": "continuous", "low": 0.0, "high": 1.0}}

    @staticmethod
    def _assert_numeric_x(series):
        vals = [float(v) for v in series.tolist()]  # raises if keys leaked in
        assert all(0.0 <= v <= 1.0 for v in vals)
        # A leaked categorical over {"type","low","high"} could never be numeric.
        assert len(set(vals)) > 2

    def test_build_candidate_safety_net(self):
        from lattice_doe.candidate import build_candidate
        cand = build_candidate(self.CONT, candidate_points=50, seed=1)
        self._assert_numeric_x(cand["x"])

    def test_estimate_candidate_size_accepts_dict_spec(self):
        from lattice_doe.candidate import estimate_candidate_size
        # Must not raise treating the dict as a categorical iterable.
        n = estimate_candidate_size("~ 1 + x", self.CONT)
        assert n > 0

    def _cfg(self):
        from lattice_doe.config import PowerContrastConfig
        return PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([1.0]),
            sigma=1.0, alpha=0.05, power=0.8, max_n=40,
        )

    def _opts(self):
        from lattice_doe.config import DesignOptions
        return DesignOptions(random_state=0, starts=1, candidate_points=60)

    def test_power_curve_by_n(self):
        from lattice_doe.analysis import power_curve_by_n
        df = power_curve_by_n("~ 1 + x", self.CONT, self._cfg(),
                              self._opts(), n_range=(10, 30), n_points=3)
        assert len(df) == 3

    def test_power_curve_by_effect(self):
        from lattice_doe.analysis import power_curve_by_effect
        df = power_curve_by_effect("~ 1 + x", self.CONT, 12, self._cfg(),
                                   self._opts())
        assert len(df) > 0

    def test_power_surface_2d(self):
        from lattice_doe.power_curves import power_surface_2d
        out = power_surface_2d(
            "~ 1 + x", self.CONT, self._cfg(),
            param1="n", param1_range=(10, 20),
            param2="effect", param2_range=(0.5, 1.5),
            grid_points=3, design_opts=self._opts(),
        )
        assert out["data"] is not None and len(out["data"]) == 9

    def test_power_curve_by_n_multiresponse(self):
        from lattice_doe.analysis import power_curve_by_n_multiresponse
        from lattice_doe.config import ResponseSpec, MultiResponseOptions
        mc = MultiResponseOptions(
            responses=[ResponseSpec(name="y1", power_cfg=self._cfg()),
                       ResponseSpec(name="y2", power_cfg=self._cfg())],
            power_combination="min",
        )
        df = power_curve_by_n_multiresponse(
            "~ 1 + x", self.CONT, mc, n_range=(10, 30), n_points=3,
            design_opts=self._opts(),
        )
        assert len(df) == 3

    def test_augment_design(self):
        from lattice_doe.iopt_search import augment_design
        from lattice_doe.candidate import build_candidate
        base = build_candidate(self.CONT, candidate_points=20, seed=0).head(8)
        aug, new = augment_design(base, m=4, formula="~ 1 + x",
                                  factors=self.CONT, design_opts=self._opts())
        assert len(new) == 4
        # Values are numeric in-range (not the dict keys "type"/"low"/"high").
        # I-optimal augmentation legitimately concentrates a 1-D continuous
        # factor at its extremes, so we do not require many distinct values.
        vals = [float(v) for v in new["x"].tolist()]
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_contrast_from_scenarios_continuous_dict(self):
        """P1 regression: explicit continuous dicts raised TypeError in
        scenario validation (validated before normalization)."""
        from lattice_doe.contrasts import contrast_from_scenarios
        L, delta = contrast_from_scenarios(
            "~ 1 + x", self.CONT,
            scenario_a={"x": 0.2}, scenario_b={"x": 0.8},
            sesoi=1.0,
        )
        assert L.shape == (1, 2)
        assert np.isclose(L[0, 1], 0.6)  # x_b - x_a on the x column
        assert delta.shape == (1,)

    def test_contrast_from_scenarios_numeric_categorical_dict(self):
        """P1 regression: an explicit binary numeric category must be treated
        as categorical in scenario validation (the old inline heuristic
        classified any two-numeric list as continuous)."""
        from lattice_doe.contrasts import contrast_from_scenarios
        factors = {"arm": {"type": "categorical", "levels": [0, 1]},
                   "x": {"type": "continuous", "low": 0.0, "high": 1.0}}
        L, delta = contrast_from_scenarios(
            "~ 1 + C(arm) + x", factors,
            scenario_a={"arm": 0, "x": 0.5},
            scenario_b={"arm": 1, "x": 0.5},
            sesoi=2.0,
        )
        assert L.shape == (1, 3)
        # The contrast isolates the arm dummy; x cancels.
        assert np.isclose(L[0, 2], 0.0)
        # Out-of-level scenario value must fail CATEGORICAL validation
        # (not a continuous range check).
        with pytest.raises(ValueError, match="categorical levels"):
            contrast_from_scenarios(
                "~ 1 + C(arm) + x", factors,
                scenario_a={"arm": 0.5, "x": 0.5},
                scenario_b={"arm": 1, "x": 0.5},
                sesoi=2.0,
            )

    def test_i_optimal_allocation_normalizes(self):
        """P1 regression (UX-35): i_optimal_allocation classified raw specs,
        so explicit dicts were treated as categoricals whose levels were the
        dictionary KEYS — returning plausible-looking allocations over cells
        like ('type', 'low')."""
        from lattice_doe.allocation import i_optimal_allocation
        factors = {
            "Material": {"type": "categorical",
                         "levels": ["Steel", "Aluminum", "Titanium"]},
            "Temp": {"type": "continuous", "low": -10.0, "high": 50.0},
        }
        alloc = i_optimal_allocation("1 + Material + Temp", factors, n=24)
        cells = set(alloc.keys())
        assert cells == {("Steel",), ("Aluminum",), ("Titanium",)}
        assert sum(alloc.values()) == 24

    def test_factor_spec_hints_resolvable_at_runtime(self):
        """P2 regression (UX-39): FactorSpec was imported under TYPE_CHECKING
        only, so typing.get_type_hints() raised NameError on the candidate
        APIs. The alias must resolve at runtime on every exported
        factor-taking function."""
        import typing
        from lattice_doe import (
            find_optimal_design, find_multiresponse_design, build_candidate,
            build_split_plot_candidate, augment_design, i_optimal_allocation,
            power_curve_by_n, power_curve_by_effect, power_surface_2d,
            power_curve_by_n_multiresponse, multiresponse_sensitivity,
        )
        from lattice_doe.candidate import estimate_candidate_size
        from lattice_doe.contrasts import contrast_from_scenarios
        from lattice_doe.utils import FactorSpec

        for fn in (find_optimal_design, find_multiresponse_design,
                   build_candidate, estimate_candidate_size,
                   build_split_plot_candidate, augment_design,
                   i_optimal_allocation, power_curve_by_n,
                   power_curve_by_effect, power_surface_2d,
                   power_curve_by_n_multiresponse, multiresponse_sensitivity,
                   contrast_from_scenarios):
            hints = typing.get_type_hints(fn)  # NameError before the fix
            assert hints["factors"] == FactorSpec, fn.__name__

    def test_factor_spec_aliases_exported_top_level(self):
        import lattice_doe
        for name in ("FactorSpec", "FactorSpecValue",
                     "ContinuousFactorSpec", "CategoricalFactorSpec"):
            assert hasattr(lattice_doe, name)
            assert name in lattice_doe.__all__

    def test_numeric_categorical_dict_through_analysis(self):
        """A numeric categorical dict spec is honored (not mis-read) at an
        analysis boundary."""
        from lattice_doe.candidate import build_candidate
        cand = build_candidate(
            {"arm": {"type": "categorical", "levels": [0, 1]}},
            candidate_points=30, seed=0,
        )
        assert sorted(set(cand["arm"].tolist())) == [0, 1]


class TestNoSpuriousParameterCountWarning:
    """P2: a multi-level categorical must count p correctly with no misleading
    'parameter count changed / levels had 0 candidates' warning."""

    def test_multilevel_categorical_no_pcount_warning(self):
        from lattice_doe.api import find_optimal_design
        from lattice_doe.config import PowerContrastConfig, DesignOptions
        factors = {
            "g": {"type": "categorical", "levels": ["a", "b", "c"]},
            "x": {"type": "continuous", "low": 0.0, "high": 1.0},
        }
        cfg = PowerContrastConfig(
            L=np.array([[0, 1, 0, 0]]), delta=np.array([1.0]),
            sigma=1.0, alpha=0.05, power=0.8, max_n=80,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = find_optimal_design(
                "~ 1 + C(g) + x", factors, cfg,
                DesignOptions(random_state=0, starts=1),
            )
        assert res["report"]["p"] == 4
        pcount = [x for x in w if "parameter count" in str(x.message)]
        assert pcount == [], f"unexpected parameter-count warning(s): " \
                             f"{[str(x.message) for x in pcount]}"


class TestSafeNameSlug:
    """UX-67: response names are free-form and end up in filenames, sheet
    titles and widget keys; the slug must neutralize every reserved
    character, resolve collisions deterministically, and never be empty."""

    def test_reserved_characters_neutralized(self):
        from lattice_doe.utils import safe_name_slug

        assert safe_name_slug("Yield/Day") == "Yield_Day"
        assert safe_name_slug(r"a\b:c*d?e") == "a_b_c_d_e"
        assert safe_name_slug('q"<>|[]') == "q______"
        assert safe_name_slug("tab\tname") == "tab_name"

    def test_collisions_resolved_deterministically(self):
        from lattice_doe.utils import safe_name_slug

        taken: set = set()
        assert safe_name_slug("Yield/Day", taken) == "Yield_Day"
        assert safe_name_slug("Yield:Day", taken) == "Yield_Day_2"
        assert safe_name_slug("Yield?Day", taken) == "Yield_Day_3"

    def test_truncation_and_empty_fallback(self):
        from lattice_doe.utils import safe_name_slug

        assert safe_name_slug("x" * 100, maxlen=10) == "x" * 10
        assert safe_name_slug("... ") == "response"
        # collision suffix respects maxlen
        taken = {"x" * 10}
        assert len(safe_name_slug("x" * 100, taken, maxlen=10)) <= 10

    def test_case_only_collisions_resolved(self):
        """UX-69: Windows filenames and Excel sheet titles are
        case-insensitive — Yield and yield are distinct valid response names
        and must not produce case-only-different slugs."""
        from lattice_doe.utils import safe_name_slug

        taken: set = set()
        assert safe_name_slug("Yield", taken) == "Yield"
        assert safe_name_slug("yield", taken) == "yield_2"
        assert safe_name_slug("YIELD", taken) == "YIELD_3"

    def test_mixed_case_seeds_respected(self):
        from lattice_doe.utils import safe_name_slug

        taken = {"MM_Taken"}
        assert safe_name_slug("mm_taken", taken) == "mm_taken_2"

    def test_prefix_collisions_checked_on_complete_name(self):
        """UX-73: when the slug is embedded in a longer sheet title
        (MM_<slug>), collisions exist between COMPLETE names — checking the
        bare slug against prefixed titles misses them, and the spreadsheet
        backend then renames the sheet behind the caller's back."""
        from lattice_doe.utils import safe_name_slug

        # bare-slug comparison does NOT collide ...
        assert safe_name_slug("yield", {"MM_Yield"}) == "yield"
        # ... prefixed comparison must (Excel titles ignore case)
        taken = {"MM_Yield"}
        assert safe_name_slug("yield", taken, prefix="MM_") == "yield_2"
        assert "MM_yield_2" in taken       # recorded as the complete title
        # repeat calls keep resolving against complete titles
        assert safe_name_slug("Yield", taken, prefix="MM_") == "Yield_3"

    def test_prefix_still_bounds_slug_length_alone(self):
        from lattice_doe.utils import safe_name_slug

        taken = {"MM_" + "a" * 10}
        slug = safe_name_slug("A" * 30, taken, maxlen=10, prefix="MM_")
        assert slug == "A" * 8 + "_2"      # suffix fits inside maxlen
        assert len(slug) == 10
