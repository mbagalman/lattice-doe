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
