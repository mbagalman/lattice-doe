# tests/test_blocked.py
# License: MIT
"""
Comprehensive tests for Enhancement 20 — Blocked Designs.

Covers:
- balanced_block_sizes utility
- blocked_formula utility
- DesignOptions blocked validation
- build_blocked_design
- find_optimal_design with blocked designs (contrast and R² modes)
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
    find_optimal_design,
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
# find_optimal_design with blocked designs
# ---------------------------------------------------------------------------

@pytest.fixture
def two_factor_setup():
    """Shared factors and formula for integration tests."""
    factors = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
    formula = "1 + A + B"
    return factors, formula


class TestIOptimalPoweredDesignBlocked:
    """Integration tests for blocked design via find_optimal_design."""

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
        result = find_optimal_design(
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
        result = find_optimal_design(
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
        result = find_optimal_design(
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
        result = find_optimal_design(
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
        result = find_optimal_design(
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
        result = find_optimal_design(
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
        result = find_optimal_design(
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
        result = find_optimal_design(
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
            find_optimal_design(
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
        result = find_optimal_design(
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
        result = find_optimal_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        assert "p_treat" in result["report"]
        assert result["report"]["p_treat"] == 3  # intercept, A, B


# ---------------------------------------------------------------------------
# CR-17: block_factor_name collision validation
# ---------------------------------------------------------------------------

class TestCR17BlockFactorNameCollision:
    """CR-17: block_factor_name that matches a treatment factor must be rejected."""

    @pytest.fixture
    def two_factor_setup(self):
        factors = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
        formula = "1 + A + B"
        return factors, formula

    def test_api_raises_when_block_name_collides_with_factor(self, two_factor_setup):
        """block_factor_name='A' should raise before any search is attempted."""
        factors, formula = two_factor_setup
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=50, max_iter=10,
        )
        design_opts = DesignOptions(
            n_blocks=3, block_factor_name="A",
            random_state=1, starts=1, candidate_points=100,
        )
        with pytest.raises(ValueError, match="block_factor_name.*collides"):
            find_optimal_design(
                formula=formula,
                factors=factors,
                power_cfg=power_cfg,
                design_opts=design_opts,
            )

    def test_api_raises_for_second_factor_collision(self, two_factor_setup):
        """block_factor_name='B' should also raise."""
        factors, formula = two_factor_setup
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=50, max_iter=10,
        )
        design_opts = DesignOptions(
            n_blocks=3, block_factor_name="B",
            random_state=1, starts=1, candidate_points=100,
        )
        with pytest.raises(ValueError, match="block_factor_name.*collides"):
            find_optimal_design(
                formula=formula,
                factors=factors,
                power_cfg=power_cfg,
                design_opts=design_opts,
            )

    def test_api_default_block_name_does_not_collide(self, two_factor_setup):
        """Default 'Block' name is fine when no factor is named 'Block'."""
        factors, formula = two_factor_setup
        power_cfg = PowerR2Config(
            r2_target=0.3, alpha=0.05, power=0.7, max_n=50, max_iter=10,
        )
        design_opts = DesignOptions(
            n_blocks=2, random_state=1, starts=1, candidate_points=100,
        )
        # Should not raise
        result = find_optimal_design(
            formula=formula, factors=factors, power_cfg=power_cfg,
            design_opts=design_opts,
        )
        assert "Block" in result["design_df"].columns

    def test_build_blocked_design_raises_when_column_exists(self):
        """build_blocked_design raises if block_factor_name already in cand.columns."""
        rng = np.random.default_rng(0)
        # 'Block' is already a column in the candidate
        cand = pd.DataFrame({
            "A": rng.uniform(-1, 1, 100),
            "Block": ["X"] * 100,  # collision
        })
        with pytest.raises(ValueError, match="block_factor_name.*already a column"):
            build_blocked_design(
                cand=cand,
                formula="1 + A",
                n=6,
                n_blocks=3,
                block_sizes=None,
                block_factor_name="Block",
                aug_formula="1 + A + C(Block)",
                criterion="I",
                n_start=1,
                algo="fedorov",
                max_iter=50,
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

    def test_build_blocked_design_raises_for_arbitrary_collision(self):
        """Any existing column name — not just 'Block' — triggers the guard."""
        rng = np.random.default_rng(0)
        cand = pd.DataFrame({
            "A": rng.uniform(-1, 1, 100),
            "Day": rng.uniform(-1, 1, 100),
        })
        with pytest.raises(ValueError, match="block_factor_name.*already a column"):
            build_blocked_design(
                cand=cand,
                formula="1 + A",
                n=4,
                n_blocks=2,
                block_sizes=None,
                block_factor_name="Day",  # collides with 'Day' column
                aug_formula="1 + A + C(Day)",
                criterion="I",
                n_start=1,
                algo="fedorov",
                max_iter=50,
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


# ---------------------------------------------------------------------------
# CR-18: blocked designs with categorical treatment factors
# ---------------------------------------------------------------------------

class TestCR18BlockedWithCategoricalTreatment:
    """CR-18: p_full must account for all categorical treatment levels.

    The old _sample_rows approach pinned treatment categoricals to one level,
    so Patsy emitted fewer dummy columns and p_full < p_treat, triggering a
    false 'no block dummy columns' ValueError.
    """

    def test_blocked_with_categorical_treatment_does_not_raise(self):
        """Blocked design with a 3-level categorical treatment factor must work."""
        factors = {
            "Material": ["Steel", "Aluminum", "Titanium"],
            "Temp":     (-10.0, 50.0),
        }
        formula = "1 + Material + Temp"
        # p_treat = intercept + 2 Material dummies + Temp = 4
        L = np.array([[0.0, 1.0, 0.0, 0.0]])  # test Material[Aluminum]
        delta = np.array([0.5])
        power_cfg = PowerContrastConfig(
            L=L, delta=delta, alpha=0.05, power=0.7, sigma=1.0,
            max_n=60, max_iter=15,
        )
        design_opts = DesignOptions(
            n_blocks=3, random_state=42, starts=1, candidate_points=150,
        )
        result = find_optimal_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        assert "Block" in result["design_df"].columns
        assert result["report"]["p_treat"] == 4
        assert result["report"]["block_structure"]["n_blocks"] == 3

    def test_blocked_p_block_cols_equals_n_blocks_minus_one_with_cat(self):
        """p_block_cols = n_blocks - 1 even when treatment has multiple cat levels."""
        factors = {
            "Cat": ["A", "B", "C"],
            "x":   (0.0, 1.0),
        }
        formula = "1 + Cat + x"
        power_cfg = PowerR2Config(
            r2_target=0.25, alpha=0.05, power=0.7, max_n=60, max_iter=15,
        )
        design_opts = DesignOptions(
            n_blocks=4, random_state=7, starts=1, candidate_points=120,
        )
        result = find_optimal_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        # p_treat = 1 (intercept) + 2 (Cat dummies) + 1 (x) = 4
        assert result["report"]["p_treat"] == 4
        assert result["design_df"]["Block"].nunique() == 4

    def test_blocked_pure_categorical_treatment_does_not_raise(self):
        """All-categorical treatment with blocked design must not raise.

        Pure 2×3 categorical → 6-row candidate; cap max_n=6 so bisection
        never requests more runs than the candidate pool.
        """
        factors = {
            "A": ["Low", "High"],
            "B": ["X", "Y", "Z"],
        }
        formula = "1 + A + B"
        power_cfg = PowerR2Config(
            r2_target=0.25, alpha=0.05, power=0.7, max_n=6, max_iter=15,
        )
        design_opts = DesignOptions(
            n_blocks=2, random_state=5, starts=1, candidate_points=100,
        )
        result = find_optimal_design(
            formula=formula,
            factors=factors,
            power_cfg=power_cfg,
            design_opts=design_opts,
        )
        assert "Block" in result["design_df"].columns
        # p_treat: intercept + 1 (A) + 2 (B) = 4
        assert result["report"]["p_treat"] == 4
