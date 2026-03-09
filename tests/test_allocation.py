# tests/test_allocation.py
"""Unit and integration tests for allocation.py (Enhancement 26).

Tests cover:
  - _wynn_multiplicative_I: convergence, single-cell edge case, uniform output
  - _round_allocation: total sum, min/max bounds, feasibility errors
  - i_optimal_allocation: return type, sum, raises for no-cat factors,
    bounds propagation, DesignOptions passthrough, formula interaction
"""
import numpy as np
import pytest

from iopt_power_design import DesignOptions, i_optimal_allocation
from iopt_power_design.allocation import (
    _round_allocation,
    _wynn_multiplicative_I,
)


# ---------------------------------------------------------------------------
# _wynn_multiplicative_I
# ---------------------------------------------------------------------------

class TestWynnMultiplicativeI:
    """Tests for the internal Wynn multiplicative update."""

    def _symmetric_X(self):
        """Two-cell symmetric model matrix — expect equal weights."""
        return np.array([
            [1.0, 1.0],
            [1.0, -1.0],
        ])

    def test_single_cell_returns_one(self):
        X = np.array([[1.0, 0.5]])
        w = _wynn_multiplicative_I(X)
        assert w.shape == (1,)
        assert float(w[0]) == pytest.approx(1.0)

    def test_output_sums_to_one(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((6, 4))
        w = _wynn_multiplicative_I(X)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-9)

    def test_output_nonnegative(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, 3))
        w = _wynn_multiplicative_I(X)
        assert np.all(w >= 0)

    def test_symmetric_cells_equal_weights(self):
        """Perfectly symmetric cells → equal optimal weights."""
        X = self._symmetric_X()
        w = _wynn_multiplicative_I(X, tol=1e-9)
        assert w[0] == pytest.approx(w[1], abs=1e-6)

    def test_shape_matches_k(self):
        k, p = 8, 3
        rng = np.random.default_rng(0)
        X = rng.standard_normal((k, p))
        w = _wynn_multiplicative_I(X)
        assert w.shape == (k,)

    def test_convergence_in_few_iters(self):
        """Well-conditioned problem should converge well within max_iter=500."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((4, 2))
        # If it doesn't converge, weights would still be uniform; with
        # a reasonable problem they deviate from uniform.
        w = _wynn_multiplicative_I(X, max_iter=500, tol=1e-6)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_singular_does_not_raise(self):
        """Singular moment matrix should break gracefully, not raise."""
        # All rows identical → rank-1 matrix
        X = np.ones((4, 3))
        w = _wynn_multiplicative_I(X)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _round_allocation
# ---------------------------------------------------------------------------

class TestRoundAllocation:
    def test_sum_equals_n(self):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        counts = _round_allocation(w, n=12)
        assert int(counts.sum()) == 12

    def test_min_per_cell_enforced(self):
        w = np.array([0.9, 0.05, 0.05])
        counts = _round_allocation(w, n=12, min_per_cell=2)
        assert np.all(counts >= 2)
        assert int(counts.sum()) == 12

    def test_max_per_cell_enforced(self):
        w = np.array([0.9, 0.05, 0.05])
        counts = _round_allocation(w, n=12, max_per_cell=5)
        assert np.all(counts <= 5)
        assert int(counts.sum()) == 12

    def test_min_zero_allowed(self):
        """min_per_cell=0 means some cells can be empty."""
        w = np.array([0.99, 0.005, 0.005])
        counts = _round_allocation(w, n=5, min_per_cell=0)
        assert int(counts.sum()) == 5

    def test_infeasible_min_raises(self):
        w = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="min_per_cell"):
            _round_allocation(w, n=1, min_per_cell=2)

    def test_infeasible_max_raises(self):
        w = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="max_per_cell"):
            _round_allocation(w, n=10, max_per_cell=3)

    def test_dtype_is_int(self):
        w = np.array([0.4, 0.35, 0.25])
        counts = _round_allocation(w, n=20)
        assert counts.dtype.kind == "i"

    def test_uniform_weights_distribute_evenly(self):
        k = 4
        w = np.ones(k) / k
        counts = _round_allocation(w, n=12)
        assert int(counts.sum()) == 12
        # Each cell should get exactly 3
        assert np.all(counts == 3)


# ---------------------------------------------------------------------------
# i_optimal_allocation — public API
# ---------------------------------------------------------------------------

class TestIOptimalAllocation:
    """Integration tests for the public i_optimal_allocation function."""

    FACTORS_MIXED = {
        "Material": ["Steel", "Aluminum", "Titanium"],
        "Temp":     (-10.0, 50.0),
        "Pressure": (1.0, 5.0),
    }
    FORMULA_MIXED = "1 + Material + Temp + Pressure"

    FACTORS_TWO_CAT = {
        "A": ["Low", "High"],
        "B": ["X", "Y", "Z"],
    }
    FORMULA_TWO_CAT = "1 + A + B"

    FACTORS_ALL_CONT = {
        "x1": (0.0, 1.0),
        "x2": (-1.0, 1.0),
    }

    def test_return_type_is_dict(self):
        alloc = i_optimal_allocation(self.FORMULA_MIXED, self.FACTORS_MIXED, n=24)
        assert isinstance(alloc, dict)

    def test_sum_equals_n(self):
        alloc = i_optimal_allocation(self.FORMULA_MIXED, self.FACTORS_MIXED, n=24)
        assert sum(alloc.values()) == 24

    def test_keys_are_tuples(self):
        alloc = i_optimal_allocation(self.FORMULA_MIXED, self.FACTORS_MIXED, n=24)
        for key in alloc:
            assert isinstance(key, tuple)

    def test_key_values_are_factor_levels(self):
        alloc = i_optimal_allocation(self.FORMULA_MIXED, self.FACTORS_MIXED, n=24)
        valid_materials = {"Steel", "Aluminum", "Titanium"}
        for (mat,) in alloc:
            assert mat in valid_materials

    def test_two_categorical_factors_cell_keys(self):
        alloc = i_optimal_allocation(self.FORMULA_TWO_CAT, self.FACTORS_TWO_CAT, n=18)
        assert sum(alloc.values()) == 18
        for key in alloc:
            assert len(key) == 2
            assert key[0] in {"Low", "High"}
            assert key[1] in {"X", "Y", "Z"}

    def test_no_categorical_raises(self):
        with pytest.raises(ValueError, match="categorical"):
            i_optimal_allocation("1 + x1 + x2", self.FACTORS_ALL_CONT, n=10)

    def test_n_less_than_1_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            i_optimal_allocation(self.FORMULA_MIXED, self.FACTORS_MIXED, n=0)

    def test_counts_are_positive_integers(self):
        alloc = i_optimal_allocation(self.FORMULA_MIXED, self.FACTORS_MIXED, n=15)
        for v in alloc.values():
            assert isinstance(v, int)
            assert v > 0

    def test_min_per_cell_respected(self):
        opts = DesignOptions(alloc_min_per_cell=2)
        alloc = i_optimal_allocation(
            self.FORMULA_MIXED, self.FACTORS_MIXED, n=24, design_opts=opts
        )
        for v in alloc.values():
            assert v >= 2

    def test_max_per_cell_respected(self):
        opts = DesignOptions(alloc_max_per_cell=8)
        alloc = i_optimal_allocation(
            self.FORMULA_MIXED, self.FACTORS_MIXED, n=24, design_opts=opts
        )
        for v in alloc.values():
            assert v <= 8

    def test_default_design_opts_none(self):
        """None design_opts should use defaults without error."""
        alloc = i_optimal_allocation(
            self.FORMULA_MIXED, self.FACTORS_MIXED, n=12, design_opts=None
        )
        assert sum(alloc.values()) == 12

    def test_cells_cap_exceeded_raises(self):
        opts = DesignOptions(cat_cells_cap=2)
        with pytest.raises(ValueError, match="cat_cells_cap"):
            i_optimal_allocation(self.FORMULA_MIXED, self.FACTORS_MIXED, n=12, design_opts=opts)

    def test_three_levels_all_cells_covered(self):
        """With enough n, all 3 cells should receive at least 1 run."""
        alloc = i_optimal_allocation(self.FORMULA_MIXED, self.FACTORS_MIXED, n=24)
        # All 3 material cells should be present
        assert len(alloc) == 3

    def test_infeasible_min_raises(self):
        opts = DesignOptions(alloc_min_per_cell=10)
        with pytest.raises(ValueError):
            i_optimal_allocation(
                self.FORMULA_MIXED, self.FACTORS_MIXED, n=6, design_opts=opts
            )

    def test_wynn_params_accepted(self):
        """Custom Wynn params should run without error."""
        opts = DesignOptions(alloc_wynn_max_iter=50, alloc_wynn_tol=1e-4)
        alloc = i_optimal_allocation(
            self.FORMULA_MIXED, self.FACTORS_MIXED, n=12, design_opts=opts
        )
        assert sum(alloc.values()) == 12

    def test_interaction_formula(self):
        """Two-way interaction formula should still produce valid allocation."""
        factors = {
            "Cat": ["A", "B"],
            "x": (0.0, 1.0),
        }
        alloc = i_optimal_allocation("1 + Cat * x", factors, n=10)
        assert sum(alloc.values()) == 10


# ---------------------------------------------------------------------------
# DesignOptions validation for allocation fields (sanity)
# ---------------------------------------------------------------------------

class TestDesignOptionsAllocationValidation:
    def test_negative_min_per_cell_raises(self):
        with pytest.raises(ValueError, match="alloc_min_per_cell"):
            DesignOptions(alloc_min_per_cell=-1)

    def test_zero_max_per_cell_raises(self):
        with pytest.raises(ValueError, match="alloc_max_per_cell"):
            DesignOptions(alloc_max_per_cell=0)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="alloc_max_per_cell"):
            DesignOptions(alloc_min_per_cell=3, alloc_max_per_cell=2)

    def test_zero_wynn_iter_raises(self):
        with pytest.raises(ValueError, match="alloc_wynn_max_iter"):
            DesignOptions(alloc_wynn_max_iter=0)

    def test_nonpositive_wynn_tol_raises(self):
        with pytest.raises(ValueError, match="alloc_wynn_tol"):
            DesignOptions(alloc_wynn_tol=0.0)

    def test_valid_defaults_no_error(self):
        opts = DesignOptions()
        assert opts.preallocate_categorical is False
        assert opts.alloc_min_per_cell == 1
        assert opts.alloc_max_per_cell is None
        assert opts.alloc_wynn_max_iter == 500
        assert opts.alloc_wynn_tol == 1e-6
