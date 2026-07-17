# tests/test_contrasts.py
"""Coding-authority tests for contrasts.py (contrast_from_scenarios,
coding_is_data_dependent, the shared candidate authority and
coding-error remedies). Companion files split out for size:
test_result_contract.py (result envelope / analysis binding)
and test_matrix_wire_format.py (REST transport format)."""
import numpy as np
import pytest

from lattice_doe.contrasts import ContrastCodingError, contrast_from_scenarios


def _full_cross_truth(formula, factors, scenario_a, scenario_b):
    """Reference L: coding learned from the FULL level cross of every factor.

    Independent of the implementation under test — it enumerates everything
    rather than reasoning about which terms need joint enumeration."""
    import itertools

    import pandas as pd
    import patsy

    cols = {}
    for k, v in factors.items():
        if isinstance(v, dict) and v.get("type") == "categorical":
            cols[k] = list(v["levels"])
        else:
            cols[k] = list(v)
    keys = list(cols)
    full = pd.DataFrame(
        list(itertools.product(*(cols[k] for k in keys))), columns=keys
    )
    di = patsy.incr_dbuilder(formula, lambda: iter([full]))
    (M,) = patsy.build_design_matrices(
        [di], pd.DataFrame([scenario_a, scenario_b])
    )
    M = np.asarray(M)
    return (M[1] - M[0]).reshape(1, -1)


def _app_scenario_contrast():
    """The Streamlit layer's scenario_contrast, importable from the app dir."""
    import sys
    from pathlib import Path

    import lattice_doe

    app_dir = str(Path(lattice_doe.__file__).resolve().parent / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    pytest.importorskip("streamlit")
    from components.power_params import scenario_contrast

    return scenario_contrast


FORMULA = "~ 1 + A + B"
FACTORS = {
    "A": ["low", "high"],
    "B": (0.0, 10.0),
}


class TestContrastFromScenarios:
    def test_returns_two_arrays(self):
        result = contrast_from_scenarios(
            FORMULA, FACTORS,
            {"A": "low", "B": 0.0},
            {"A": "high", "B": 0.0},
            sesoi=1.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_L_is_single_row(self):
        L, delta = contrast_from_scenarios(
            FORMULA, FACTORS,
            {"A": "low", "B": 5.0},
            {"A": "high", "B": 5.0},
            sesoi=0.5,
        )
        assert L.ndim == 2
        assert L.shape[0] == 1

    def test_delta_is_scalar_array(self):
        L, delta = contrast_from_scenarios(
            FORMULA, FACTORS,
            {"A": "low", "B": 5.0},
            {"A": "high", "B": 5.0},
            sesoi=2.0,
        )
        assert delta.shape == (1,)
        assert delta[0] == 2.0

    def test_L_column_count_matches_p(self):
        """L must have p columns matching the compiled formula."""
        L, _ = contrast_from_scenarios(
            FORMULA, FACTORS,
            {"A": "low", "B": 0.0},
            {"A": "high", "B": 0.0},
            sesoi=1.0,
        )
        # ~ 1 + A + B encodes to: Intercept, A[T.high], B  (3 columns)
        assert L.shape[1] == 3

    def test_identical_scenarios_give_zero_L(self):
        L, _ = contrast_from_scenarios(
            FORMULA, FACTORS,
            {"A": "low", "B": 5.0},
            {"A": "low", "B": 5.0},
            sesoi=1.0,
        )
        assert np.allclose(L, 0.0)

    def test_antisymmetry(self):
        """Swapping A and B negates L."""
        L_ab, _ = contrast_from_scenarios(
            FORMULA, FACTORS,
            {"A": "low", "B": 5.0},
            {"A": "high", "B": 5.0},
            sesoi=1.0,
        )
        L_ba, _ = contrast_from_scenarios(
            FORMULA, FACTORS,
            {"A": "high", "B": 5.0},
            {"A": "low", "B": 5.0},
            sesoi=1.0,
        )
        assert np.allclose(L_ab, -L_ba)

    def test_raises_on_missing_factor_in_scenario(self):
        with pytest.raises(KeyError, match="Missing"):
            contrast_from_scenarios(
                FORMULA, FACTORS,
                {"A": "low"},           # missing B
                {"A": "high", "B": 5.0},
                sesoi=1.0,
            )

    def test_raises_on_extra_factor_in_scenario(self):
        with pytest.raises(KeyError, match="Unknown"):
            contrast_from_scenarios(
                FORMULA, FACTORS,
                {"A": "low", "B": 5.0, "C": 1.0},   # C not in factors
                {"A": "high", "B": 5.0},
                sesoi=1.0,
            )

    def test_raises_on_unknown_categorical_level(self):
        with pytest.raises(ValueError, match="not one of"):
            contrast_from_scenarios(
                FORMULA, FACTORS,
                {"A": "medium", "B": 5.0},   # "medium" not a level
                {"A": "high", "B": 5.0},
                sesoi=1.0,
            )

    def test_raises_on_continuous_out_of_range(self):
        with pytest.raises(ValueError, match="outside the allowed"):
            contrast_from_scenarios(
                FORMULA, FACTORS,
                {"A": "low", "B": 99.0},   # B > 10
                {"A": "high", "B": 5.0},
                sesoi=1.0,
            )

    def test_raises_on_non_positive_sesoi(self):
        with pytest.raises(ValueError, match="sesoi"):
            contrast_from_scenarios(
                FORMULA, FACTORS,
                {"A": "low", "B": 0.0},
                {"A": "high", "B": 0.0},
                sesoi=0.0,
            )
        with pytest.raises(ValueError, match="sesoi"):
            contrast_from_scenarios(
                FORMULA, FACTORS,
                {"A": "low", "B": 0.0},
                {"A": "high", "B": 0.0},
                sesoi=-1.0,
            )


class TestSharedCategoricalLevelAnchoring:
    """TD-7 / UX-25 regression: when both scenarios share a categorical level,
    the single anchor row could share it too, so Patsy silently dropped the
    unseen levels' dummy columns and L came back narrower than the design
    model. The anchor is now a level-covering frame."""

    def test_shared_level_full_width(self):
        factors = {"g": ["a", "b", "c"], "x": (0.0, 1.0)}
        L, delta = contrast_from_scenarios(
            "~ 1 + C(g) + x", factors,
            {"g": "a", "x": 0.2},   # same level of g in both scenarios
            {"g": "a", "x": 0.8},
            sesoi=1.0,
        )
        assert L.shape == (1, 4)  # intercept + 2 g-dummies + x
        # g is held constant, so its dummy columns must cancel exactly...
        assert np.allclose(L[0, 1:3], 0.0)
        # ...and the contrast lives entirely on x.
        assert np.isclose(L[0, 3], 0.6)

    def test_shared_level_other_categorical_changes(self):
        factors = {"g": ["a", "b", "c"], "h": ["p", "q"]}
        L, _ = contrast_from_scenarios(
            "~ 1 + C(g) + C(h)", factors,
            {"g": "b", "h": "p"},   # g constant, h changes
            {"g": "b", "h": "q"},
            sesoi=1.0,
        )
        assert L.shape == (1, 4)  # intercept + 2 g-dummies + 1 h-dummy
        assert np.count_nonzero(L) == 1  # only the h dummy differs

    def test_deterministic(self):
        """The level-covering anchor is deterministic — identical L on repeat."""
        factors = {"g": ["a", "b", "c"], "x": (0.0, 1.0)}
        args = ("~ 1 + C(g) + x", factors,
                {"g": "c", "x": 0.1}, {"g": "a", "x": 0.9})
        L1, _ = contrast_from_scenarios(*args, sesoi=1.0)
        L2, _ = contrast_from_scenarios(*args, sesoi=1.0)
        assert np.array_equal(L1, L2)


class TestIncrementalCodingAnchor:
    """UX-35..UX-38: the model coding is established by a chunked incremental
    Patsy scan — no anchor model matrix is materialized, only the referenced
    categorical cross is enumerated, and derived-from-continuous categorical
    terms require authoritative coding_data."""

    def test_unreferenced_factor_does_not_count_toward_cap(self):
        """A 400×400 factor space with `~ C(a)` needs only 400 anchor rows —
        the unused factor b must not trigger the cross cap (P2 regression)."""
        big = {"a": [str(i) for i in range(400)],
               "b": [str(i) for i in range(400)]}
        L, _ = contrast_from_scenarios(
            "~ C(a)", big,
            {"a": "0", "b": "0"}, {"a": "1", "b": "0"},
            sesoi=1.0,
        )
        assert L.shape == (1, 400)

    def test_additive_large_factors_use_level_cover(self):
        """UX-42: `~ C(a) + C(b)` with two 400-level factors needs only a
        400-row per-factor cover — main effects code per-factor, so the
        160 000-cell cross must NOT be required (the previous revision
        rejected this safe model)."""
        big = {"a": [str(i) for i in range(400)],
               "b": [str(i) for i in range(400)]}
        L, _ = contrast_from_scenarios(
            "~ C(a) + C(b)", big,
            {"a": "0", "b": "0"}, {"a": "1", "b": "0"},
            sesoi=1.0,
        )
        assert L.shape == (1, 1 + 399 + 399)

    def test_interaction_uses_level_cover(self):
        """`:`/`*` interactions build columns structurally from per-factor
        codings, so they too need only the per-factor cover."""
        big = {"a": [str(i) for i in range(400)],
               "b": [str(i) for i in range(400)]}
        L, _ = contrast_from_scenarios(
            "~ C(a) * C(b)", big,
            {"a": "0", "b": "0"}, {"a": "1", "b": "0"},
            sesoi=1.0,
        )
        assert L.shape[1] == 400 * 400  # full interaction model width

    def test_huge_combined_group_raises_actionable_error(self):
        """Rejection triggers only when factors COMBINED inside one derived
        expression (C(a + b)) have a level cross above the cap."""
        big = {"a": [str(i) for i in range(400)],
               "b": [str(i) for i in range(400)]}  # 160 000 cells > 100 000 cap
        with pytest.raises(ValueError, match="cannot anchor this model"):
            contrast_from_scenarios(
                "~ C(a + b)", big,
                {"a": "0", "b": "0"}, {"a": "1", "b": "0"},
                sesoi=1.0,
            )

    def test_derived_term_full_width(self):
        """A 101×101 cross (10 201 cells) is streamed in chunks; a derived
        C(a + b) contrast must match the realized full-cross model width."""
        import itertools
        import pandas as pd
        from lattice_doe.model_matrix import build_model_matrix

        factors = {"a": [str(i) for i in range(101)],
                   "b": [str(i) for i in range(101)]}
        L, _ = contrast_from_scenarios(
            "~ C(a + b)", factors,
            {"a": "0", "b": "0"}, {"a": "1", "b": "1"},
            sesoi=1.0,
        )
        combos = list(itertools.product(factors["a"], factors["b"]))
        full = pd.DataFrame({"a": [c[0] for c in combos],
                             "b": [c[1] for c in combos]})
        X_full, _ = build_model_matrix("~ C(a + b)", full)
        assert L.shape[1] == X_full.shape[1]

    def test_large_interaction_stays_flat_on_memory(self):
        """UX-36 regression: a 101×101 C(a)*C(b) interaction implies a
        10 201-column model; the old one-shot anchor matrix was ~832 MB.
        The incremental scan must stay far below that."""
        import tracemalloc

        factors = {"a": [str(i) for i in range(101)],
                   "b": [str(i) for i in range(101)]}
        tracemalloc.start()
        L, _ = contrast_from_scenarios(
            "~ C(a) * C(b)", factors,
            {"a": "0", "b": "0"}, {"a": "1", "b": "1"},
            sesoi=1.0,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert L.shape == (1, 101 * 101)
        assert peak < 100e6, f"peak {peak/1e6:.0f} MB — anchor matrix materialized?"

    def test_continuous_derived_categorical_requires_coding_data(self):
        """UX-37: C(I(x // 1)) derives its levels from realized continuous
        data; without authoritative coding the helper must refuse (the old
        midpoint anchor silently returned an undersized contrast)."""
        with pytest.raises(ValueError, match="coding_data"):
            contrast_from_scenarios(
                "~ C(I(x // 1))", {"x": (-2.0, 2.0)},
                {"x": 0.2}, {"x": 1.2},
                sesoi=1.0,
            )

    def test_continuous_derived_with_coding_data_matches_realized(self):
        from lattice_doe.candidate import build_candidate
        from lattice_doe.model_matrix import build_model_matrix

        cand = build_candidate({"x": (-2.0, 2.0)}, candidate_points=200, seed=0)
        L, _ = contrast_from_scenarios(
            "~ C(I(x // 1))", {"x": (-2.0, 2.0)},
            {"x": 0.2}, {"x": 1.2},
            sesoi=1.0, coding_data=cand,
        )
        X_c, _ = build_model_matrix("~ C(I(x // 1))", cand)
        assert L.shape[1] == X_c.shape[1]

    def test_coding_data_authoritative_for_categoricals(self):
        """coding_data path also anchors plain categorical models."""
        import pandas as pd

        cand = pd.DataFrame({"g": ["a", "b", "c"] * 4,
                             "x": [0.1, 0.5, 0.9] * 4})
        L, _ = contrast_from_scenarios(
            "~ 1 + C(g) + x", {"g": ["a", "b", "c"], "x": (0.0, 1.0)},
            {"g": "a", "x": 0.2}, {"g": "a", "x": 0.8},
            sesoi=1.0, coding_data=cand,
        )
        assert L.shape == (1, 4)
        assert np.allclose(L[0, 1:3], 0.0)

    def test_scenario_level_absent_from_coding_data_raises(self):
        """UX-40 regression: scenario rows previously participated in the
        coding scan, so a scenario level missing from coding_data silently
        WIDENED the model (L had a column the authoritative model lacks).
        The coding must be learned from coding_data alone."""
        import pandas as pd

        cd = pd.DataFrame({"g": ["a", "b"] * 3, "x": [0.1, 0.5, 0.9] * 2})
        with pytest.raises(ValueError, match="absent from the coding data"):
            contrast_from_scenarios(
                "~ 1 + C(g) + x", {"g": ["a", "b", "c"], "x": (0.0, 1.0)},
                {"g": "a", "x": 0.2}, {"g": "c", "x": 0.8},
                sesoi=1.0, coding_data=cd,
            )

    def test_stateful_transform_requires_coding_data(self):
        """UX-41 regression: bs/cr/center/… learn coding parameters from the
        data; the internal anchor produced a same-width but numerically
        different L (max coeff diff ~0.78 for bs(x, df=4)) — silently
        changing the estimand. Must refuse without coding_data."""
        with pytest.raises(ValueError, match="stateful"):
            contrast_from_scenarios(
                "~ bs(x, df=4)", {"x": (0.0, 1.0)},
                {"x": 0.2}, {"x": 0.8},
                sesoi=1.0,
            )

    def test_stateful_transform_with_coding_data_matches_authority(self):
        import pandas as pd
        import patsy
        from lattice_doe.candidate import build_candidate

        cand = build_candidate({"x": (0.0, 1.0)}, candidate_points=100, seed=0)
        L, _ = contrast_from_scenarios(
            "~ bs(x, df=4)", {"x": (0.0, 1.0)},
            {"x": 0.2}, {"x": 0.8},
            sesoi=1.0, coding_data=cand,
        )
        # Reference: coding learned from the candidate alone, scenarios
        # transformed against that fixed coding.
        di = patsy.incr_dbuilder("~ bs(x, df=4)", lambda: iter([cand]))
        (Xs,) = patsy.build_design_matrices(
            [di], pd.DataFrame({"x": [0.2, 0.8]})
        )
        Xs = np.asarray(Xs)
        assert np.allclose(L, (Xs[1] - Xs[0]).reshape(1, -1))

    def test_factor_named_like_stateful_transform_is_usable(self):
        """UX-44 regression: detection intersected every identifier with the
        transform names, so a factor merely NAMED `scale`/`center`/`bs` was
        rejected even though nothing stateful was called."""
        cases = [
            ("~ scale", {"scale": (0.0, 1.0)},
             {"scale": 0.0}, {"scale": 1.0}),
            ("~ C(bs)", {"bs": ["lo", "hi"]},
             {"bs": "lo"}, {"bs": "hi"}),
            ("~ center + x", {"center": ["p", "q"], "x": (0.0, 1.0)},
             {"center": "p", "x": 0.0}, {"center": "q", "x": 1.0}),
            ("~ standardize * te", {"standardize": ["u", "v"], "te": (0.0, 2.0)},
             {"standardize": "u", "te": 0.5}, {"standardize": "v", "te": 1.5}),
        ]
        for formula, factors, a, b in cases:
            L, _ = contrast_from_scenarios(formula, factors, a, b, sesoi=1.0)
            assert np.allclose(L, _full_cross_truth(formula, factors, a, b)), (
                f"wrong L for {formula!r}"
            )

    def test_called_stateful_transform_still_rejected(self):
        """The flip side of UX-44: an actual CALL must still be caught, and a
        name collision must not mask it."""
        for formula in ("~ bs(x, df=3)", "~ center(x)", "~ scale(x)"):
            with pytest.raises(ContrastCodingError, match="stateful"):
                contrast_from_scenarios(
                    formula, {"x": (0.0, 1.0)},
                    {"x": 0.0}, {"x": 1.0}, sesoi=1.0,
                )

    def test_numeric_derived_term_needs_no_cross(self):
        """UX-45 regression: `~ I(a + b)` over two 400-level numeric-coded
        categoricals is a two-column NUMERIC model, but every multi-factor
        derived term was treated as a categorical cross and rejected at the
        160 000-cell cap."""
        levels = list(range(400))
        factors = {"a": {"type": "categorical", "levels": levels},
                   "b": {"type": "categorical", "levels": levels}}
        L, _ = contrast_from_scenarios(
            "~ I(a + b)", factors, {"a": 0, "b": 0}, {"a": 1, "b": 1},
            sesoi=1.0,
        )
        assert L.shape == (1, 2)  # intercept + the single numeric column

    def test_numeric_derived_term_matches_full_cross(self):
        """Skipping the cross must not change L: at a size where the full
        cross is still computable, the two must agree exactly."""
        levels = list(range(20))
        factors = {"a": {"type": "categorical", "levels": levels},
                   "b": {"type": "categorical", "levels": levels}}
        a, b = {"a": 3, "b": 7}, {"a": 11, "b": 2}
        L, _ = contrast_from_scenarios("~ I(a + b)", factors, a, b, sesoi=1.0)
        assert np.allclose(L, _full_cross_truth("~ I(a + b)", factors, a, b))

    def test_numeric_derived_term_alongside_own_main_effect(self):
        """A factor inside a numeric derived term still needs its own level
        cover when it also appears as a categorical main effect."""
        factors = {"a": {"type": "categorical", "levels": [0, 1, 2]},
                   "b": {"type": "categorical", "levels": [0, 5, 10]}}
        a, b = {"a": 0, "b": 0}, {"a": 2, "b": 5}
        f = "~ I(a + b) + C(a)"
        L, _ = contrast_from_scenarios(f, factors, a, b, sesoi=1.0)
        assert np.allclose(L, _full_cross_truth(f, factors, a, b))

    def test_categorical_valued_derived_term_still_crossed(self):
        """The result TYPE decides, not the syntax: `I(a + b)` over STRING
        levels is a concatenation — categorical — so it must still be jointly
        enumerated even though it is not wrapped in C()."""
        factors = {"a": ["x", "y"], "b": ["p", "q"]}
        a, b = {"a": "x", "b": "p"}, {"a": "y", "b": "q"}
        L, _ = contrast_from_scenarios("~ I(a + b)", factors, a, b, sesoi=1.0)
        assert L.shape == (1, 4)  # 4 realized combinations -> 4 columns
        assert np.allclose(L, _full_cross_truth("~ I(a + b)", factors, a, b))

    def test_q_quoted_factor_name_referenced(self):
        """UX-40 regression (reference detection): a factor named 'dose%' is
        only referable as Q(\"dose%\") — word-boundary text matching missed
        it, so shared-level scenarios collapsed its dummy columns."""
        factors = {"dose%": ["lo", "mid", "hi"], "x": (0.0, 1.0)}
        L, _ = contrast_from_scenarios(
            '~ C(Q("dose%")) + x', factors,
            {"dose%": "lo", "x": 0.2}, {"dose%": "lo", "x": 0.8},
            sesoi=1.0,
        )
        assert L.shape == (1, 4)  # intercept + 2 dose% dummies + x
        assert np.allclose(L[0, 1:3], 0.0)


class TestCodingErrorRemedies:
    """UX-46/UX-48: each interface answers a data-dependent coding in the way
    ITS user can act on. The CLI and the Run page build the run's own candidate
    set and just work; only the preview (no design options) and the growth case
    report, and neither may echo the Python API's "pass coding_data=" advice."""

    # Scenario values sit INSIDE the range: a sampled candidate set does not
    # reach the declared bounds, and a spline cannot extrapolate past its
    # outermost knots (see test_scenario_outside_spline_knots_explains_why).
    _STATEFUL = ("~ bs(x, df=3)", {"x": (0.0, 1.0)}, {"x": 0.2}, {"x": 0.8}, 1.0)

    def test_error_splits_reason_from_remedy(self):
        with pytest.raises(ContrastCodingError) as ei:
            contrast_from_scenarios(*self._STATEFUL)
        exc = ei.value
        assert "stateful" in exc.reason
        assert "coding_data" not in exc.reason  # the reason is remedy-free
        assert "coding_data" in exc.remedy
        assert str(exc) == f"{exc.reason} {exc.remedy}"

    def test_error_is_valueerror_subclass(self):
        """Existing callers catching ValueError must keep working."""
        assert issubclass(ContrastCodingError, ValueError)
        with pytest.raises(ValueError):
            contrast_from_scenarios(*self._STATEFUL)

    def test_cli_codes_against_the_runs_own_candidate(self):
        """UX-48: the CLI no longer dead-ends on a stateful formula — it codes
        the contrast against the candidate set its own design_opts produce,
        which is the set find_optimal_design will select from."""
        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.cli import _scenario_contrast
        from lattice_doe.config import DesignOptions

        formula, factors, a, b, sesoi = self._STATEFUL
        opts = DesignOptions(random_state=7, candidate_points=400)
        L, _ = _scenario_contrast(formula, factors, a, b, sesoi, opts)

        cand, _ = build_search_candidate(formula, factors, opts)
        ref, _ = contrast_from_scenarios(
            formula, factors, a, b, sesoi, coding_data=cand,
        )
        assert np.array_equal(L, ref)

    def test_cli_remedy_when_growth_would_invalidate(self):
        """Growth rebuilds the candidate mid-search and re-derives the coding
        while L stays fixed, so it must be refused, not silently accepted."""
        from lattice_doe.cli import _scenario_contrast
        from lattice_doe.config import DesignOptions

        formula, factors, a, b, sesoi = self._STATEFUL
        opts = DesignOptions(allow_candidate_growth=True)
        with pytest.raises(ContrastCodingError) as ei:
            _scenario_contrast(formula, factors, a, b, sesoi, opts)
        exc = ei.value
        assert "stateful" in exc.reason                        # diagnosis kept
        assert "allow_candidate_growth" in exc.remedy          # names the flag
        assert "coding_data" not in exc.remedy                 # not YAML-able

    def test_ui_run_path_codes_against_the_runs_own_candidate(self):
        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.config import DesignOptions

        scenario_contrast = _app_scenario_contrast()
        formula, factors, a, b, sesoi = self._STATEFUL
        opts = DesignOptions(random_state=11, candidate_points=400)
        L, _ = scenario_contrast(
            design_opts=opts, formula=formula, factors=factors,
            scenario_a=a, scenario_b=b, sesoi=sesoi,
        )
        cand, _ = build_search_candidate(formula, factors, opts)
        ref, _ = contrast_from_scenarios(
            formula, factors, a, b, sesoi, coding_data=cand,
        )
        assert np.array_equal(L, ref)

    def test_ui_preview_defers_instead_of_guessing(self):
        """Without design options the preview cannot know the coding authority.
        It must say so — never guess a seed/size, which would show an L the run
        does not use."""
        scenario_contrast = _app_scenario_contrast()
        formula, factors, a, b, sesoi = self._STATEFUL
        with pytest.raises(ContrastCodingError) as ei:
            scenario_contrast(formula=formula, factors=factors,
                              scenario_a=a, scenario_b=b, sesoi=sesoi)
        exc = ei.value
        assert "stateful" in exc.reason
        assert "Run / Results" in exc.remedy
        assert "coding_data" not in exc.remedy

    def test_no_interface_remedy_hardcodes_a_candidate_recipe(self):
        """UX-48 regression: the remedies used to hand out a snippet pinning
        `candidate_points=2000, seed=42`. The CLI default seed is 123 and the
        app may size adaptively, so copying it silently produced an L for
        different spline knots — same width, no error."""
        from lattice_doe import cli

        _app_scenario_contrast()   # puts the app dir on sys.path
        import components.power_params as pp

        remedies = [
            cli._CLI_CODING_REMEDY, cli._CLI_GROWTH_REMEDY,
            pp.UI_CODING_REMEDY, pp.UI_PREVIEW_REMEDY, pp.UI_GROWTH_REMEDY,
        ]
        for text in remedies:
            assert "seed=42" not in text
            assert "candidate_points=2000" not in text
            assert "build_candidate(" not in text

    def test_power_params_importable_without_lattice_doe(self):
        """The pages import components.power_params unconditionally and must
        still render their 'not installed' notice, so the lattice_doe import
        has to stay lazy."""
        import ast
        from pathlib import Path

        import lattice_doe

        src = (Path(lattice_doe.__file__).resolve().parent
               / "app" / "components" / "power_params.py").read_text()
        tree = ast.parse(src)
        top_level = [n for n in tree.body
                     if isinstance(n, (ast.Import, ast.ImportFrom))]
        for node in top_level:
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in node.names]
            assert not mod.startswith("lattice_doe"), (
                f"lattice_doe imported at module level: {mod}"
            )
            assert not any(n.startswith("lattice_doe") for n in names)


class TestOverlappingDerivedTerms:
    """UX-47: derived terms are scanned one segment each. Overlapping terms
    must NOT be unioned into a single Cartesian cross."""

    def test_overlapping_terms_are_not_unioned(self):
        """`C(a+b) + C(b+c)` at 50 levels needs the a×b cross and the b×c
        cross (2 500 rows each), never the a×b×c cross (125 000 > cap). The
        previous revision merged {a,b} and {b,c} into {a,b,c} and rejected a
        model whose coding is perfectly establishable."""
        lv = list(range(50))
        factors = {k: {"type": "categorical", "levels": lv}
                   for k in ("a", "b", "c")}
        a, b = {"a": 0, "b": 0, "c": 0}, {"a": 1, "b": 1, "c": 1}
        L, _ = contrast_from_scenarios(
            "~ C(a + b) + C(b + c)", factors, a, b, sesoi=1.0,
        )
        # 1 intercept + 98 dummies per derived term (sums span 0..98).
        assert L.shape == (1, 197)

    def test_overlapping_terms_match_full_cross(self):
        """Segmented scanning must not change L: at a size where the full
        3-way cross is computable, the two must agree exactly."""
        lv = list(range(6))
        factors = {k: {"type": "categorical", "levels": lv}
                   for k in ("a", "b", "c")}
        a, b = {"a": 0, "b": 0, "c": 0}, {"a": 1, "b": 2, "c": 3}
        f = "~ C(a + b) + C(b + c)"
        L, _ = contrast_from_scenarios(f, factors, a, b, sesoi=1.0)
        assert np.allclose(L, _full_cross_truth(f, factors, a, b))

    def test_three_way_single_term_still_needs_its_own_cross(self):
        """A genuine 3-factor term is one group: its cross is unavoidable and
        the cap must still apply — and name the offending term."""
        lv = [str(i) for i in range(50)]  # 125 000 combinations > cap
        factors = {k: lv for k in ("a", "b", "c")}
        with pytest.raises(ContrastCodingError, match=r"C\(a \+ b \+ c\)"):
            contrast_from_scenarios(
                "~ C(a + b + c)", factors,
                {"a": "0", "b": "0", "c": "0"},
                {"a": "1", "b": "1", "c": "1"},
                sesoi=1.0,
            )

    def test_duplicate_factor_sets_scanned_once(self):
        """Two derived terms over the same factors need only one segment."""
        factors = {"a": ["p", "q"], "b": ["r", "s"]}
        a, b = {"a": "p", "b": "r"}, {"a": "q", "b": "s"}
        f = "~ C(a + b) + I(a + b)"
        L, _ = contrast_from_scenarios(f, factors, a, b, sesoi=1.0)
        assert np.allclose(L, _full_cross_truth(f, factors, a, b))


class TestCodingIsDataDependent:
    """UX-48: interfaces ask up front whether they must supply coding_data."""

    def test_none_for_ordinary_formulas(self):
        from lattice_doe.contrasts import coding_is_data_dependent

        for formula, factors in [
            ("~ 1 + C(g) + x", {"g": ["a", "b"], "x": (0.0, 1.0)}),
            ("~ C(a) * C(b)", {"a": ["p", "q"], "b": ["r", "s"]}),
            ("~ scale", {"scale": (0.0, 1.0)}),          # name collision only
            ("~ I(a + b)", {"a": {"type": "categorical", "levels": [0, 1]},
                            "b": {"type": "categorical", "levels": [0, 5]}}),
        ]:
            assert coding_is_data_dependent(formula, factors) is None, formula

    def test_reason_for_learned_codings(self):
        from lattice_doe.contrasts import coding_is_data_dependent

        assert "stateful" in coding_is_data_dependent(
            "~ bs(x, df=3)", {"x": (0.0, 1.0)}
        )
        assert "derives categorical levels" in coding_is_data_dependent(
            "~ C(I(x // 1))", {"x": (0.0, 5.0)}
        )

    def test_agrees_with_contrast_from_scenarios(self):
        """The predicate and the builder must not disagree: whenever a reason
        is reported, the spec-only path must refuse, and vice versa."""
        from lattice_doe.contrasts import coding_is_data_dependent

        cases = [
            ("~ 1 + C(g)", {"g": ["a", "b"]}, {"g": "a"}, {"g": "b"}),
            ("~ bs(x, df=3)", {"x": (0.0, 1.0)}, {"x": 0.2}, {"x": 0.8}),
            ("~ C(I(x // 1))", {"x": (0.0, 5.0)}, {"x": 0.5}, {"x": 4.5}),
            ("~ center", {"center": ["u", "v"]}, {"center": "u"},
             {"center": "v"}),
        ]
        for formula, factors, a, b in cases:
            reason = coding_is_data_dependent(formula, factors)
            try:
                contrast_from_scenarios(formula, factors, a, b, sesoi=1.0)
                refused = False
            except ContrastCodingError:
                refused = True
            assert refused == (reason is not None), formula

    def test_scenario_outside_spline_knots_explains_why(self):
        """A scenario sitting exactly on a factor's declared bound can fall
        outside a spline's outermost knots, because a sampled candidate set
        never quite reaches the bound. The error must not blame a missing
        categorical level — the cause here is continuous and different."""
        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.config import DesignOptions

        formula, factors = "~ bs(x, df=3)", {"x": (0.0, 1.0)}
        opts = DesignOptions(random_state=7, candidate_points=400)
        cand, _ = build_search_candidate(formula, factors, opts)
        assert cand["x"].min() > 0.0 and cand["x"].max() < 1.0  # premise

        with pytest.raises(ValueError, match="outside the range"):
            contrast_from_scenarios(
                formula, factors, {"x": 0.0}, {"x": 1.0}, 1.0,
                coding_data=cand,
            )
        # Just inside the sampled range is fine.
        L, _ = contrast_from_scenarios(
            formula, factors, {"x": 0.2}, {"x": 0.8}, 1.0, coding_data=cand,
        )
        assert L.shape == (1, 4)


class TestScenarioContrastForRun:
    """The shared decision tree behind every interface's scenario builder
    (TD refactor): the CLI and app wrappers only rewrite remedies, so the
    Python-facing defaults and the authority logic are pinned here once."""

    _SPLINE = "~ 1 + bs(x, df=3)"
    _FACTORS = {"x": (0.0, 1.0)}
    _A, _B = {"x": 0.2}, {"x": 0.8}

    def test_python_default_remedies_per_refusal(self):
        from lattice_doe.config import DesignOptions, SplitPlotOptions
        from lattice_doe.contrasts import (
            PY_GROWTH_REMEDY, PY_NO_DESIGN_OPTS_REMEDY, PY_SPLIT_PLOT_REMEDY,
            scenario_contrast_for_run,
        )

        with pytest.raises(ContrastCodingError) as ei:
            scenario_contrast_for_run(
                self._SPLINE, self._FACTORS, self._A, self._B, 0.5,
            )
        assert ei.value.remedy == PY_NO_DESIGN_OPTS_REMEDY
        # the advice must fit THIS function's signature: it names an
        # accepted argument, and any coding_data= mention is explicitly
        # scoped to contrast_from_scenarios (passing it here is TypeError)
        assert "design_opts" in ei.value.remedy
        assert "contrast_from_scenarios(..., coding_data=" in ei.value.remedy

        sp_opts = DesignOptions(split_plot=SplitPlotOptions(
            htc_factors=["x"], n_whole_plots=2,
        ))
        with pytest.raises(ContrastCodingError) as ei:
            scenario_contrast_for_run(
                self._SPLINE, self._FACTORS, self._A, self._B, 0.5,
                design_opts=sp_opts,
            )
        assert ei.value.remedy == PY_SPLIT_PLOT_REMEDY

        grow_opts = DesignOptions(allow_candidate_growth=True)
        with pytest.raises(ContrastCodingError) as ei:
            scenario_contrast_for_run(
                self._SPLINE, self._FACTORS, self._A, self._B, 0.5,
                design_opts=grow_opts,
            )
        assert ei.value.remedy == PY_GROWTH_REMEDY

    def test_matches_manual_candidate_coding_exactly(self):
        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.config import DesignOptions
        from lattice_doe.contrasts import scenario_contrast_for_run

        opts = DesignOptions(random_state=7, candidate_points=120)
        L, delta = scenario_contrast_for_run(
            self._SPLINE, self._FACTORS, self._A, self._B, 0.5,
            design_opts=opts,
        )
        cand, _ = build_search_candidate(self._SPLINE, self._FACTORS, opts)
        L_ref, delta_ref = contrast_from_scenarios(
            self._SPLINE, self._FACTORS, self._A, self._B, 0.5,
            coding_data=cand,
        )
        assert np.array_equal(L, L_ref)
        assert np.array_equal(delta, delta_ref)

    def test_remedy_overrides_replace_only_their_key(self):
        from lattice_doe.contrasts import (
            PY_SPLIT_PLOT_REMEDY, scenario_contrast_for_run,
        )

        with pytest.raises(ContrastCodingError) as ei:
            scenario_contrast_for_run(
                self._SPLINE, self._FACTORS, self._A, self._B, 0.5,
                remedies={"preview": "ask the run page"},
            )
        assert ei.value.remedy == "ask the run page"
        assert PY_SPLIT_PLOT_REMEDY  # defaults still exist for other keys

    def test_coding_fallthrough_remedy_fits_this_api(self, monkeypatch):
        """A coding failure raised past the refusals (e.g. the anchoring
        cap) must not propagate contrast_from_scenarios' own advice to pass
        coding_data= — this function has no such parameter, so following
        that advice is a TypeError. Stubbed: the cap needs a >100k joint
        level cross, and only the rewrap plumbing is under test here."""
        from lattice_doe import contrasts as contrasts_module
        from lattice_doe.contrasts import (
            PY_CODING_REMEDY, PY_RUN_CODING_REMEDY, scenario_contrast_for_run,
        )

        def _cap_error(*args, **kwargs):
            raise ContrastCodingError("level cross too large",
                                      PY_CODING_REMEDY)

        monkeypatch.setattr(
            contrasts_module, "contrast_from_scenarios", _cap_error,
        )
        with pytest.raises(ContrastCodingError) as ei:
            scenario_contrast_for_run(
                FORMULA, FACTORS,
                {"A": "low", "B": 0.0}, {"A": "high", "B": 10.0}, 0.5,
            )
        assert ei.value.reason == "level cross too large"
        assert ei.value.remedy == PY_RUN_CODING_REMEDY


class TestSearchCandidateIsTheSharedAuthority:
    """UX-48: the contrast's coding authority must be the very candidate set
    the search selects from — not a look-alike rebuilt from guessed args."""

    def test_candidate_is_spec_form_invariant(self):
        """find_optimal_design normalizes factors before building its
        candidate; the CLI/app pass the raw spec. All spec forms must yield
        the identical candidate, or the two would code differently."""
        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.config import DesignOptions
        from lattice_doe.utils import normalize_factors

        f = "~ 1 + bs(x, df=3) + C(g)"
        opts = DesignOptions(random_state=123, candidate_points=300)
        plain = {"x": (0.0, 1.0), "g": ["a", "b", "c"]}
        typed = {"x": {"type": "continuous", "low": 0.0, "high": 1.0},
                 "g": {"type": "categorical", "levels": ["a", "b", "c"]}}

        cands = [
            build_search_candidate(f, plain, opts)[0],
            build_search_candidate(f, normalize_factors(plain, f), opts)[0],
            build_search_candidate(f, typed, opts)[0],
            build_search_candidate(f, normalize_factors(typed, f), opts)[0],
        ]
        for other in cands[1:]:
            assert cands[0].equals(other)

    def test_api_builds_its_candidate_through_the_shared_helper(self):
        """If find_optimal_design ever stops routing through
        build_search_candidate, the interfaces' authority silently becomes a
        look-alike. Pin the wiring."""
        import inspect

        from lattice_doe import api

        src = inspect.getsource(api.find_optimal_design)
        assert "build_search_candidate(" in src
        src_mr = inspect.getsource(api.find_multiresponse_design)
        assert "build_search_candidate(" in src_mr

    @pytest.mark.slow
    def test_cli_contrast_width_matches_realized_design(self):
        """End to end: L built by the CLI for a stateful formula must have
        exactly as many columns as the design the same options produce."""
        from lattice_doe.api import find_optimal_design
        from lattice_doe.cli import _scenario_contrast
        from lattice_doe.config import DesignOptions, PowerContrastConfig

        formula, factors = "~ 1 + bs(x, df=3)", {"x": (0.0, 1.0)}
        opts = DesignOptions(candidate_points=200, random_state=5, starts=1)
        L, delta = _scenario_contrast(
            formula, factors, {"x": 0.2}, {"x": 0.8}, 0.6, opts,
        )
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.8, sigma=1.0, max_n=120,
        )
        result = find_optimal_design(formula, factors, power_cfg, opts)
        assert L.shape[1] == result["report"]["p"]

    def test_cli_multiresponse_sizes_candidate_by_the_global_formula(self):
        """In multi-response mode a response may carry its own formula, but
        find_multiresponse_design builds ONE candidate sized by the GLOBAL
        formula. The per-response contrast must be coded against that
        candidate — sizing by the response's formula would silently diverge
        the moment candidate sizing starts reading the formula (today it does
        not, so this pins the wiring, not a live behavior difference)."""
        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.cli import _scenario_contrast
        from lattice_doe.config import DesignOptions

        factors = {"x": (0.0, 1.0)}
        global_formula = "~ 1 + x"
        resp_formula = "~ 1 + bs(x, df=3)"
        opts = DesignOptions(auto_candidate=True, random_state=3)

        L, _ = _scenario_contrast(
            resp_formula, factors, {"x": 0.2}, {"x": 0.8}, 0.5, opts,
            sizing_formula=global_formula,
        )
        run_cand, _ = build_search_candidate(global_formula, factors, opts)
        ref, _ = contrast_from_scenarios(
            resp_formula, factors, {"x": 0.2}, {"x": 0.8}, 0.5,
            coding_data=run_cand,
        )
        assert np.array_equal(L, ref)


class TestImplicitCategoricalDetection:
    """UX-51: Patsy treats object-valued expression results as categorical
    without any C(...) call. Detection must go by the EVALUATED result type,
    not by C-call syntax — the old check let a thresholding expression through
    to the spec-only anchor, which realized only one branch and produced a
    same-shaped but wrong (or confusingly failing) contrast."""

    _F = '~ I(np.where(x < 0.5, "lo", "hi"))'
    _FACTORS = {"x": (0.0, 1.0)}

    def test_predicate_flags_implicit_categorical(self):
        from lattice_doe.contrasts import coding_is_data_dependent

        reason = coding_is_data_dependent(self._F, self._FACTORS)
        assert reason is not None and "categorical" in reason

    def test_spec_only_path_refuses(self):
        with pytest.raises(ContrastCodingError):
            contrast_from_scenarios(
                self._F, self._FACTORS, {"x": 0.2}, {"x": 0.8}, sesoi=1.0,
            )

    def test_coding_data_path_matches_candidate_reference(self):
        import pandas as pd
        import patsy

        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.config import DesignOptions

        cand, _ = build_search_candidate(
            self._F, self._FACTORS, DesignOptions()
        )
        a, b = {"x": 0.2}, {"x": 0.8}
        L, _ = contrast_from_scenarios(
            self._F, self._FACTORS, a, b, sesoi=1.0, coding_data=cand,
        )
        di = patsy.incr_dbuilder(self._F, lambda: iter([cand]))
        (M,) = patsy.build_design_matrices([di], pd.DataFrame([a, b]))
        M = np.asarray(M)
        assert np.array_equal(L, (M[1] - M[0]).reshape(1, -1))
        assert np.array_equal(L, np.array([[0.0, -1.0]]))  # reviewer's truth

    def test_explicit_c_cases_still_flagged(self):
        from lattice_doe.contrasts import coding_is_data_dependent

        for f in ("~ C(I(x // 1))", "~ C(x)"):
            assert coding_is_data_dependent(f, {"x": (0.0, 1.0)}) is not None

    def test_numeric_derived_terms_not_flagged(self):
        from lattice_doe.contrasts import coding_is_data_dependent

        cases = [
            ("~ 1 + x", {"x": (0.0, 1.0)}),
            ("~ I(x * 2) + C(g)", {"x": (0.0, 1.0), "g": ["a", "b"]}),
            ("~ np.log(x + 1)", {"x": (0.0, 1.0)}),
        ]
        for f, fac in cases:
            assert coding_is_data_dependent(f, fac) is None, f


class TestSplitPlotCodingGuard:
    """UX-50: a split-plot search learns its coding from separately built
    whole-plot/sub-plot pools, not from build_search_candidate's ordinary
    candidate — so for data-dependent codings the interfaces must refuse the
    scenario form rather than hand over a non-authoritative candidate."""

    _STATEFUL = ("~ bs(x, df=3)", {"x": (0.0, 1.0), "w": ["a", "b"]},
                 {"x": 0.2, "w": "a"}, {"x": 0.8, "w": "b"}, 1.0)

    @staticmethod
    def _sp_opts():
        from lattice_doe.config import DesignOptions, SplitPlotOptions

        return DesignOptions(
            split_plot=SplitPlotOptions(htc_factors=["w"], n_whole_plots=4),
        )

    def test_cli_refuses_stateful_scenario_with_split_plot(self):
        from lattice_doe.cli import _scenario_contrast

        formula, factors, a, b, sesoi = self._STATEFUL
        with pytest.raises(ContrastCodingError) as ei:
            _scenario_contrast(formula, factors, a, b, sesoi, self._sp_opts())
        assert "split-plot" in ei.value.remedy
        assert "'L' and 'delta'" in ei.value.remedy

    def test_ui_refuses_stateful_scenario_with_split_plot(self):
        scenario_contrast = _app_scenario_contrast()
        formula, factors, a, b, sesoi = self._STATEFUL
        with pytest.raises(ContrastCodingError) as ei:
            scenario_contrast(design_opts=self._sp_opts(), formula=formula,
                              factors=factors, scenario_a=a, scenario_b=b,
                              sesoi=sesoi)
        assert "split-plot" in ei.value.remedy
        assert "Matrix" in ei.value.remedy

    def test_ordinary_formula_with_split_plot_unaffected(self):
        """The guard is scoped to data-dependent codings: a plain formula's
        coding is spec-derivable regardless of how the search selects rows."""
        from lattice_doe.cli import _scenario_contrast

        factors = {"x": (0.0, 1.0), "w": ["a", "b"]}
        L, _ = _scenario_contrast(
            "~ 1 + x + C(w)", factors,
            {"x": 0.2, "w": "a"}, {"x": 0.8, "w": "b"}, 1.0, self._sp_opts(),
        )
        assert L.shape == (1, 3)

    def test_boundary_error_no_longer_advises_extending_coding_data(self):
        """UX-52: extending coding_data re-derives spline knots while the
        search still uses its own sampled candidate — the old advice broke
        the exact-authority invariant this series established."""
        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.config import DesignOptions

        formula, factors = "~ bs(x, df=3)", {"x": (0.0, 1.0)}
        cand, _ = build_search_candidate(
            formula, factors, DesignOptions(candidate_points=400,
                                            random_state=7),
        )
        with pytest.raises(ValueError, match="inside the realized range") as ei:
            contrast_from_scenarios(
                formula, factors, {"x": 0.0}, {"x": 1.0}, 1.0,
                coding_data=cand,
            )
        assert "extend coding_data to cover it" not in str(ei.value)


class TestExplicitCodingParameters:
    """UX-55/UX-56: a transform whose parameters are all supplied literally is
    NOT data-dependent — its coding is identical on any input data, so the
    spec-only path is exact and no interface guard should fire."""

    _FACTORS = {"x": (0.0, 1.0)}
    _FULL_BS = "~ bs(x, knots=[0.3, 0.6], lower_bound=0.0, upper_bound=1.0)"
    _C_LEVELS = "~ C(I(x > 0.5), levels=[False, True])"

    def _candidate_ref(self, formula, a, b):
        import pandas as pd
        import patsy

        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.config import DesignOptions

        cand, _ = build_search_candidate(
            formula, self._FACTORS, DesignOptions()
        )
        di = patsy.incr_dbuilder(formula, lambda: iter([cand]))
        (M,) = patsy.build_design_matrices([di], pd.DataFrame([a, b]))
        M = np.asarray(M)
        return (M[1] - M[0]).reshape(1, -1)

    @pytest.mark.parametrize("formula", [_FULL_BS, _C_LEVELS])
    def test_not_flagged_and_spec_only_L_is_exact(self, formula):
        from lattice_doe.contrasts import coding_is_data_dependent

        assert coding_is_data_dependent(formula, self._FACTORS) is None
        a, b = {"x": 0.2}, {"x": 0.8}
        L, _ = contrast_from_scenarios(formula, self._FACTORS, a, b, sesoi=1.0)
        assert np.allclose(L, self._candidate_ref(formula, a, b))

    def test_fully_specified_spline_works_with_split_plot(self):
        """UX-55's practical payoff: split-plot scenario mode no longer
        blocks a spline whose coding is explicit."""
        from lattice_doe.cli import _scenario_contrast
        from lattice_doe.config import DesignOptions, SplitPlotOptions

        factors = {"x": (0.0, 1.0), "w": ["a", "b"]}
        opts = DesignOptions(
            split_plot=SplitPlotOptions(htc_factors=["w"], n_whole_plots=4),
        )
        L, _ = _scenario_contrast(
            "~ bs(x, knots=[0.3, 0.6], lower_bound=0.0, upper_bound=1.0) + C(w)",
            factors,
            {"x": 0.2, "w": "a"}, {"x": 0.8, "w": "b"}, 1.0, opts,
        )
        assert L.shape[0] == 1 and L.shape[1] >= 5

    @pytest.mark.parametrize("formula", [
        "~ bs(x, df=5)",                                   # knots learned
        "~ bs(x, knots=[0.3, 0.6])",                       # bounds learned
        "~ bs(x, knots=[0.3], lower_bound=0.0, upper_bound=ub)",  # non-literal
        "~ cr(x, knots=[0.3, 0.6], lower_bound=0.0, upper_bound=1.0,"
        " constraints='center')",                          # data-computed constraint
        "~ center(x)",
        "~ C(I(x > 0.5))",                                 # no explicit levels
        "~ C(I(x > 0.5), levels=lv)",                      # non-literal levels
        '~ I(np.where(x < 0.5, "lo", "hi"))',              # no C at all
    ])
    def test_learned_variants_still_flagged(self, formula):
        from lattice_doe.contrasts import coding_is_data_dependent

        assert coding_is_data_dependent(formula, self._FACTORS) is not None


class TestAugmentUsesOneCoding:
    """UX-53 (augment instance): X_current and X_cand must share ONE
    DesignInfo learned from the candidate. Coding each frame separately
    vstacked two different bases for stateful formulas — and crashed outright
    when the seed design realized only a subset of a categorical's levels."""

    def test_seed_design_missing_levels_augments_cleanly(self):
        import pandas as pd

        from lattice_doe.config import DesignOptions
        from lattice_doe.iopt_search import augment_design

        factors = {"g": ["a", "b", "c"]}
        seed = pd.DataFrame({"g": ["a", "a", "a", "a"]})  # one level only
        combined, added = augment_design(
            seed, 4, "~ C(g)", factors,
            DesignOptions(candidate_points=60, random_state=1),
        )
        assert len(combined) == 8 and len(added) == 4

    def test_stateful_augment_scores_one_basis(self):
        from lattice_doe.candidate import build_search_candidate
        from lattice_doe.config import DesignOptions
        from lattice_doe.iopt_search import augment_design

        f, factors = "~ 1 + bs(x, df=4)", {"x": (0.0, 1.0)}
        opts = DesignOptions(candidate_points=120, random_state=3)
        cand, _ = build_search_candidate(f, factors, opts)
        seed = cand.iloc[:8][["x"]].reset_index(drop=True)
        combined, added = augment_design(seed, 4, f, factors, opts)
        assert len(combined) == 12 and len(added) == 4


class TestNoneIsNotExplicit:
    """UX-59: an explicit ``None`` means 'learn from the data' in every
    parameter the fixed-coding checks inspect — it must not qualify."""

    @pytest.mark.parametrize("formula", [
        "~ bs(x, knots=None, lower_bound=0.0, upper_bound=1.0)",
        "~ bs(x, knots=[0.3], lower_bound=None, upper_bound=1.0)",
        "~ bs(x, knots=[0.3], lower_bound=0.0, upper_bound=None)",
        "~ C(I(x > 0.5), levels=None)",
        "~ bs(x, knots=[0.3, None], lower_bound=0.0, upper_bound=1.0)",
    ])
    def test_none_valued_parameters_stay_flagged(self, formula):
        from lattice_doe.contrasts import coding_is_data_dependent

        assert coding_is_data_dependent(formula, {"x": (0.0, 1.0)}) is not None


class TestAugmentBoundaryRows:
    """UX-58: a normal existing design contains the factor bounds, which a
    candidate-only coding scan cannot transform for spline formulas (sampled
    candidates never reach the bounds). The augmentation coding is learned
    from the existing rows AND the candidate together."""

    def test_boundary_seed_design_augments(self):
        import pandas as pd

        from lattice_doe.config import DesignOptions
        from lattice_doe.iopt_search import augment_design

        seed = pd.DataFrame({"x": [0.0, 0.25, 0.5, 0.75, 1.0]})
        combined, added = augment_design(
            seed, 3, "~ 1 + bs(x, df=4)", {"x": (0.0, 1.0)},
            DesignOptions(candidate_points=120, random_state=3),
        )
        assert len(combined) == 8 and len(added) == 3


class TestAnalysisHonorsCodingAuthority:
    """UX-57: analysis functions on a FIXED design must evaluate the exact
    basis the design run powered. Rebuilding from design_df re-learns
    data-dependent codings, so without model_matrix they refuse."""

    _F, _FACTORS = "~ 1 + bs(x, df=4)", {"x": (0.0, 1.0)}

    def _run(self):
        import numpy as np

        from lattice_doe import DesignOptions, find_optimal_design
        from lattice_doe.config import PowerContrastConfig

        opts = DesignOptions(candidate_points=150, random_state=5, starts=1)
        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0, 0.0, 0.0, -1.0]]),
            delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=30,
        )
        return find_optimal_design(self._F, self._FACTORS, cfg, opts), cfg, opts

    def test_sensitivity_refuses_then_accepts_the_authority(self):
        from lattice_doe.analysis import power_sensitivity

        res, cfg, opts = self._run()
        with pytest.raises(ValueError, match="model_matrix"):
            power_sensitivity(
                formula=self._F, factors=self._FACTORS, power_cfg=cfg,
                design_df=res["design_df"], design_opts=opts,
            )
        sens = power_sensitivity(
            formula=self._F, factors=self._FACTORS, power_cfg=cfg,
            design_df=res["design_df"], design_opts=opts,
            model_matrix=res["model_matrix"],
        )
        assert "nominal_power" in sens

    def test_mde_refuses_then_accepts_the_authority(self):
        from lattice_doe.analysis import min_detectable_effect

        res, cfg, opts = self._run()
        with pytest.raises(ValueError, match="model_matrix"):
            min_detectable_effect(
                design_df=res["design_df"], formula=self._F,
                factors=self._FACTORS, power_cfg=cfg, design_opts=opts,
            )
        mde = min_detectable_effect(
            design_df=res["design_df"], formula=self._F,
            factors=self._FACTORS, power_cfg=cfg, design_opts=opts,
            model_matrix=res["model_matrix"],
        )
        assert mde["mde"] > 0

    def test_plain_formulas_unchanged_without_model_matrix(self):
        import numpy as np

        from lattice_doe import DesignOptions, find_optimal_design
        from lattice_doe.analysis import power_sensitivity
        from lattice_doe.config import PowerContrastConfig

        opts = DesignOptions(candidate_points=80, random_state=3, starts=1)
        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=25,
        )
        res = find_optimal_design("~ 1 + x", {"x": (0.0, 1.0)}, cfg, opts)
        sens = power_sensitivity(
            formula="~ 1 + x", factors={"x": (0.0, 1.0)}, power_cfg=cfg,
            design_df=res["design_df"], design_opts=opts,
        )
        assert "nominal_power" in sens
