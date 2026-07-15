# tests/test_contrasts.py
"""Unit tests for contrasts.py — contrast_from_scenarios."""
import numpy as np
import pytest

from lattice_doe.contrasts import contrast_from_scenarios

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
        400-row cycling cover — main effects code per-factor, so the
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
        codings, so they too need only the cycling cover."""
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
