# tests/test_blocked.py
# License: MIT
"""
Comprehensive tests for Enhancement 20 — Blocked Designs.

Covers:
- balanced_block_sizes utility
- blocked_formula utility
- DesignOptions blocked validation
- build_blocked_design
- i_optimal_powered_design with blocked designs (contrast and R² modes)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iopt_power_design import (
    DesignOptions,
    PowerContrastConfig,
    PowerR2Config,
    balanced_block_sizes,
    blocked_formula,
    build_blocked_design,
    i_optimal_powered_design,
)


# ---------------------------------------------------------------------------
# balanced_block_sizes
# ---------------------------------------------------------------------------

class TestBalancedBlockSizes:
    def test_even_split(self):
        assert balanced_block_sizes(12, 4) == [3, 3, 3, 3]

    def test_uneven_remainder_first_blocks_get_extra(self):
        result = balanced_block_sizes(10, 3)
        assert result == [4, 3, 3]
        assert sum(result) == 10
        assert len(result) == 3

    def test_two_blocks_even(self):
        result = balanced_block_sizes(6, 2)
        assert result == [3, 3]

    def test_two_blocks_uneven(self):
        result = balanced_block_sizes(7, 2)
        assert result == [4, 3]
        assert sum(result) == 7

    def test_n_equals_n_blocks(self):
        result = balanced_block_sizes(4, 4)
        assert result == [1, 1, 1, 1]

    def test_large_n(self):
        n, nb = 100, 7
        result = balanced_block_sizes(n, nb)
        assert sum(result) == n
        assert len(result) == nb
        # All sizes are either floor or ceil
        base = n // nb
        for s in result:
            assert s in (base, base + 1)

    def test_n_blocks_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_blocks must be >= 2"):
            balanced_block_sizes(10, 1)

    def test_n_less_than_n_blocks_raises(self):
        with pytest.raises(ValueError, match="n .* must be >= n_blocks"):
            balanced_block_sizes(2, 5)

    def test_sum_is_n(self):
        for n in range(4, 30):
            for nb in range(2, min(n + 1, 8)):
                result = balanced_block_sizes(n, nb)
                assert sum(result) == n, f"n={n}, nb={nb}"
                assert len(result) == nb


# ---------------------------------------------------------------------------
# blocked_formula
# ---------------------------------------------------------------------------

class TestBlockedFormula:
    def test_default_block_name(self):
        result = blocked_formula("1 + A + B")
        assert result == "1 + A + B + C(Block)"

    def test_custom_block_name(self):
        result = blocked_formula("1 + X1 + X2", block_factor_name="Day")
        assert result == "1 + X1 + X2 + C(Day)"

    def test_formula_unchanged_except_appended(self):
        formula = "1 + A + B + A:B"
        result = blocked_formula(formula)
        assert result.startswith(formula)
        assert "C(Block)" in result


# ---------------------------------------------------------------------------
# DesignOptions blocked validation
# ---------------------------------------------------------------------------

class TestDesignOptionsBlockedValidation:
    def test_valid_n_blocks(self):
        opts = DesignOptions(n_blocks=3)
        assert opts.n_blocks == 3

    def test_n_blocks_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_blocks must be >= 2"):
            DesignOptions(n_blocks=1)

    def test_n_blocks_zero_raises(self):
        with pytest.raises(ValueError, match="n_blocks must be >= 2"):
            DesignOptions(n_blocks=0)

    def test_block_sizes_without_n_blocks_raises(self):
        with pytest.raises(ValueError, match="block_sizes requires n_blocks"):
            DesignOptions(block_sizes=[3, 3, 4])

    def test_mismatched_block_sizes_length_raises(self):
        with pytest.raises(ValueError, match="len\\(block_sizes\\)"):
            DesignOptions(n_blocks=3, block_sizes=[5, 5])

    def test_zero_block_size_raises(self):
        with pytest.raises(ValueError, match="All block_sizes must be >= 1"):
            DesignOptions(n_blocks=3, block_sizes=[0, 5, 5])

    def test_valid_n_blocks_with_block_sizes(self):
        opts = DesignOptions(n_blocks=3, block_sizes=[4, 3, 3])
        assert opts.block_sizes == [4, 3, 3]

    def test_default_block_factor_name(self):
        opts = DesignOptions()
        assert opts.block_factor_name == "Block"

    def test_custom_block_factor_name(self):
        opts = DesignOptions(n_blocks=2, block_factor_name="Day")
        assert opts.block_factor_name == "Day"

    def test_empty_block_factor_name_raises(self):
        with pytest.raises(ValueError, match="block_factor_name must be a non-empty string"):
            DesignOptions(block_factor_name="")

    def test_n_blocks_none_is_default(self):
        opts = DesignOptions()
        assert opts.n_blocks is None


# ---------------------------------------------------------------------------
# build_blocked_design
# ---------------------------------------------------------------------------

class TestBuildBlockedDesign:
    @pytest.fixture
    def simple_cand(self):
        """Simple 2-factor continuous candidate set."""
        rng = np.random.default_rng(42)
        n_cand = 200
        A = rng.uniform(-1, 1, n_cand)
        B = rng.uniform(-1, 1, n_cand)
        return pd.DataFrame({"A": A, "B": B})

    def test_returns_dataframe_with_block_column(self, simple_cand):
        formula = "1 + A + B"
        aug_formula = "1 + A + B + C(Block)"
        design_df, X_full = build_blocked_design(
            cand=simple_cand,
            formula=formula,
            n=12,
            n_blocks=3,
            block_sizes=None,
            block_factor_name="Block",
            aug_formula=aug_formula,
            criterion="I",
            n_start=2,
            algo="fedorov",
            max_iter=100,
            random_state=42,
            workers=None,
            parallel_seed_stride=10000,
            jitter=1e-8,
            preallocate_categorical=False,
            alloc_min_per_cell=1,
            alloc_max_per_cell=None,
            alloc_wynn_max_iter=500,
            alloc_wynn_tol=1e-6,
            cat_cells_cap=10000,
        )
        assert isinstance(design_df, pd.DataFrame)
        assert "Block" in design_df.columns

    def test_correct_n_rows(self, simple_cand):
        formula = "1 + A + B"
        aug_formula = "1 + A + B + C(Block)"
        design_df, X_full = build_blocked_design(
            cand=simple_cand,
            formula=formula,
            n=15,
            n_blocks=3,
            block_sizes=None,
            block_factor_name="Block",
            aug_formula=aug_formula,
            criterion="I",
            n_start=2,
            algo="fedorov",
            max_iter=100,
            random_state=42,
            workers=None,
            parallel_seed_stride=10000,
            jitter=1e-8,
            preallocate_categorical=False,
            alloc_min_per_cell=1,
            alloc_max_per_cell=None,
            alloc_wynn_max_iter=500,
            alloc_wynn_tol=1e-6,
            cat_cells_cap=10000,
        )
        assert len(design_df) == 15

    def test_x_full_shape(self, simple_cand):
        """X_full should have n rows and p_treat + (n_blocks-1) columns."""
        formula = "1 + A + B"
        n_blocks = 3
        aug_formula = f"1 + A + B + C(Block)"
        n = 12
        design_df, X_full = build_blocked_design(
            cand=simple_cand,
            formula=formula,
            n=n,
            n_blocks=n_blocks,
            block_sizes=None,
            block_factor_name="Block",
            aug_formula=aug_formula,
            criterion="I",
            n_start=2,
            algo="fedorov",
            max_iter=100,
            random_state=42,
            workers=None,
            parallel_seed_stride=10000,
            jitter=1e-8,
            preallocate_categorical=False,
            alloc_min_per_cell=1,
            alloc_max_per_cell=None,
            alloc_wynn_max_iter=500,
            alloc_wynn_tol=1e-6,
            cat_cells_cap=10000,
        )
        assert X_full.shape[0] == n
        # formula "1 + A + B" gives 3 columns; 3 blocks gives 2 dummy columns → 5 total
        assert X_full.shape[1] == 3 + (n_blocks - 1)

    def test_block_labels_are_correct(self, simple_cand):
        formula = "1 + A + B"
        aug_formula = "1 + A + B + C(Block)"
        n_blocks = 4
        design_df, _ = build_blocked_design(
            cand=simple_cand,
            formula=formula,
            n=12,
            n_blocks=n_blocks,
            block_sizes=None,
            block_factor_name="Block",
            aug_formula=aug_formula,
            criterion="I",
            n_start=2,
            algo="fedorov",
            max_iter=100,
            random_state=0,
            workers=None,
            parallel_seed_stride=10000,
            jitter=1e-8,
            preallocate_categorical=False,
            alloc_min_per_cell=1,
            alloc_max_per_cell=None,
            alloc_wynn_max_iter=500,
            alloc_wynn_tol=1e-6,
            cat_cells_cap=10000,
        )
        unique_blocks = set(design_df["Block"].unique())
        expected = {f"B{i+1}" for i in range(n_blocks)}
        assert unique_blocks == expected

    def test_mismatched_block_sizes_sum_raises(self, simple_cand):
        formula = "1 + A + B"
        aug_formula = "1 + A + B + C(Block)"
        with pytest.raises(ValueError, match="sum\\(block_sizes\\)"):
            build_blocked_design(
                cand=simple_cand,
                formula=formula,
                n=12,
                n_blocks=3,
                block_sizes=[3, 3, 5],  # sums to 11, not 12
                block_factor_name="Block",
                aug_formula=aug_formula,
                criterion="I",
                n_start=2,
                algo="fedorov",
                max_iter=100,
                random_state=42,
                workers=None,
                parallel_seed_stride=10000,
                jitter=1e-8,
                preallocate_categorical=False,
                alloc_min_per_cell=1,
                alloc_max_per_cell=None,
                alloc_wynn_max_iter=500,
                alloc_wynn_tol=1e-6,
                cat_cells_cap=10000,
            )


# ---------------------------------------------------------------------------
# i_optimal_powered_design with blocked designs
# ---------------------------------------------------------------------------

@pytest.fixture
def two_factor_setup():
    """Shared factors and formula for integration tests."""
    factors = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
    formula = "1 + A + B"
    return factors, formula


class TestIOptimalPoweredDesignBlocked:
    """Integration tests for blocked design via i_optimal_powered_design."""

    def test_contrast_mode_returns_block_column(self, two_factor_setup):
        factors, formula = two_factor_setup
        # L: test main effect of A (3rd col = B), p_treat=3 (intercept, A, B)
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=7, starts=2, candidate_points=200,
        )
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        assert "Block" in result["design_df"].columns

    def test_contrast_mode_n_blocks_distinct_values(self, two_factor_setup):
        factors, formula = two_factor_setup
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=7, starts=2, candidate_points=200,
        )
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        n_distinct = result["design_df"]["Block"].nunique()
        assert n_distinct == 3

    def test_contrast_mode_power_is_float_in_0_1(self, two_factor_setup):
        factors, formula = two_factor_setup
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=7, starts=2, candidate_points=200,
        )
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        power = result["report"]["achieved_power"]
        assert isinstance(power, float)
        assert 0.0 <= power <= 1.0

    def test_contrast_mode_report_contains_block_structure(self, two_factor_setup):
        factors, formula = two_factor_setup
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=7, starts=2, candidate_points=200,
        )
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        bs = result["report"]["block_structure"]
        assert bs is not None
        assert bs["n_blocks"] == 3
        assert bs["block_factor_name"] == "Block"

    def test_r2_mode_returns_block_column(self, two_factor_setup):
        factors, formula = two_factor_setup
        power_cfg = PowerR2Config(
            r2_target=0.3, alpha=0.05, power=0.7,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=11, starts=2, candidate_points=200,
        )
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        assert "Block" in result["design_df"].columns

    def test_r2_mode_n_blocks_distinct_values(self, two_factor_setup):
        factors, formula = two_factor_setup
        power_cfg = PowerR2Config(
            r2_target=0.3, alpha=0.05, power=0.7,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=11, starts=2, candidate_points=200,
        )
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        n_distinct = result["design_df"]["Block"].nunique()
        assert n_distinct == 3

    def test_r2_mode_block_structure_in_report(self, two_factor_setup):
        factors, formula = two_factor_setup
        power_cfg = PowerR2Config(
            r2_target=0.3, alpha=0.05, power=0.7,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=11, starts=2, candidate_points=200,
        )
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        bs = result["report"]["block_structure"]
        assert bs is not None
        assert bs["n_blocks"] == 3

    def test_unblocked_design_block_structure_is_none(self, two_factor_setup):
        """Unblocked designs should have block_structure=None in report."""
        factors, formula = two_factor_setup
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=50, max_iter=20,
        )
        design_opts = DesignOptions(random_state=42, starts=2, candidate_points=200)
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        assert result["report"]["block_structure"] is None

    def test_contrast_L_validated_against_p_treat(self, two_factor_setup):
        """L with wrong number of columns (p_full instead of p_treat) should raise."""
        factors, formula = two_factor_setup
        # p_treat=3 (intercept, A, B); with 3 blocks p_full=5
        # Passing L with 5 columns should raise
        L_wrong = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L_wrong, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=7, starts=2, candidate_points=200,
        )
        with pytest.raises(ValueError, match="p_treat"):
            i_optimal_powered_design(
                formula=formula,
                factors=factors,
                power_cfg=power_cfg,
                design_opts=design_opts,
            )

    def test_custom_block_factor_name(self, two_factor_setup):
        factors, formula = two_factor_setup
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, block_factor_name="Day",
            random_state=7, starts=2, candidate_points=200,
        )
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        assert "Day" in result["design_df"].columns
        assert result["report"]["block_structure"]["block_factor_name"] == "Day"

    def test_p_treat_in_report(self, two_factor_setup):
        """Report should include p_treat for blocked designs."""
        factors, formula = two_factor_setup
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=80, max_iter=30,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=7, starts=2, candidate_points=200,
        )
        result = i_optimal_powered_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        assert "p_treat" in result["report"]
        assert result["report"]["p_treat"] == 3  # intercept, A, B
