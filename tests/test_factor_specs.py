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
