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
