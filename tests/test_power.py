# tests/test_power.py
"""Unit tests for power.py — contrast_power and global_r2_power."""
import numpy as np
import pytest

from iopt_power_design.power import contrast_power, global_r2_power


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_rank_X(n: int, p: int, seed: int = 0) -> np.ndarray:
    """Return a full-rank (n x p) design matrix (intercept + random predictors)."""
    rng = np.random.default_rng(seed)
    X = np.ones((n, p))
    X[:, 1:] = rng.standard_normal((n, p - 1))
    return X


# ---------------------------------------------------------------------------
# contrast_power
# ---------------------------------------------------------------------------

class TestContrastPower:
    def test_returns_named_tuple(self):
        X = _full_rank_X(30, 3)
        result = contrast_power(np.array([[0, 1, 0]]), np.array([1.0]), X)
        assert hasattr(result, "power")
        assert hasattr(result, "lam")

    def test_power_in_unit_interval(self):
        X = _full_rank_X(30, 3)
        result = contrast_power(np.array([[0, 1, 0]]), np.array([1.0]), X)
        assert 0.0 <= result.power <= 1.0

    def test_lambda_non_negative(self):
        X = _full_rank_X(30, 3)
        result = contrast_power(np.array([[0, 1, 0]]), np.array([1.0]), X)
        assert result.lam >= 0.0

    def test_power_increases_with_n(self):
        """More observations → more power (all else fixed)."""
        L = np.array([[0, 1, 0]])
        delta = np.array([0.5])
        powers = [
            contrast_power(L, delta, _full_rank_X(n, 3)).power
            for n in [10, 40, 150]
        ]
        assert powers[0] < powers[1] < powers[2]

    def test_power_increases_with_delta(self):
        """Larger effect size → more power."""
        X = _full_rank_X(40, 3)
        L = np.array([[0, 1, 0]])
        powers = [
            contrast_power(L, np.array([d]), X).power
            for d in [0.2, 0.6, 1.5]
        ]
        assert powers[0] < powers[1] < powers[2]

    def test_power_decreases_with_sigma(self):
        """Higher noise → less power."""
        X = _full_rank_X(40, 3)
        L = np.array([[0, 1, 0]])
        delta = np.array([1.0])
        powers = [
            contrast_power(L, delta, X, sigma=s).power
            for s in [0.5, 1.0, 2.0]
        ]
        assert powers[0] > powers[1] > powers[2]

    def test_1d_L_accepted(self):
        """L can be passed as a flat 1D array and is promoted to (1, p)."""
        X = _full_rank_X(20, 3)
        result = contrast_power(np.array([0, 1, 0]), np.array([1.0]), X)
        assert 0.0 <= result.power <= 1.0

    def test_multi_row_L(self):
        """q > 1 contrasts are supported."""
        X = _full_rank_X(40, 3)
        L = np.array([[0, 1, 0], [0, 0, 1]])
        delta = np.array([0.5, 0.5])
        result = contrast_power(L, delta, X)
        assert 0.0 <= result.power <= 1.0

    def test_raises_on_non_positive_sigma(self):
        X = _full_rank_X(20, 2)
        with pytest.raises(ValueError, match="sigma"):
            contrast_power(np.array([[0, 1]]), np.array([1.0]), X, sigma=0.0)

    def test_raises_on_L_column_mismatch(self):
        X = _full_rank_X(20, 3)
        L_bad = np.array([[0, 1, 0, 0]])  # 4 cols, X has 3
        with pytest.raises(ValueError):
            contrast_power(L_bad, np.array([1.0]), X)

    def test_raises_on_delta_length_mismatch(self):
        X = _full_rank_X(20, 3)
        L = np.array([[0, 1, 0], [0, 0, 1]])
        with pytest.raises(ValueError):
            contrast_power(L, np.array([1.0]), X)  # 2-row L, 1-element delta

    def test_raises_when_df_denom_zero(self):
        """n <= rank(X) → df_denom <= 0 → should raise."""
        X = _full_rank_X(3, 3)  # n == p, df_denom = 0
        with pytest.raises(ValueError):
            contrast_power(np.array([[0, 1, 0]]), np.array([1.0]), X)


# ---------------------------------------------------------------------------
# global_r2_power
# ---------------------------------------------------------------------------

class TestGlobalR2Power:
    def test_returns_named_tuple(self):
        X = _full_rank_X(30, 3)
        result = global_r2_power(r2_target=0.2, X=X, alpha=0.05)
        assert hasattr(result, "power")
        assert hasattr(result, "lam")

    def test_power_in_unit_interval(self):
        X = _full_rank_X(30, 3)
        result = global_r2_power(r2_target=0.2, X=X, alpha=0.05)
        assert 0.0 <= result.power <= 1.0

    def test_power_increases_with_n(self):
        powers = [
            global_r2_power(0.1, _full_rank_X(n, 3), alpha=0.05).power
            for n in [10, 50, 200]
        ]
        assert powers[0] < powers[1] < powers[2]

    def test_power_increases_with_r2(self):
        X = _full_rank_X(50, 3)
        powers = [
            global_r2_power(r2, X, alpha=0.05).power
            for r2 in [0.05, 0.20, 0.50]
        ]
        assert powers[0] < powers[1] < powers[2]

    def test_n_minus_p_lambda_formula(self):
        """lambda_mode='n_minus_p' must give f2 * (n - p), not f2 * p."""
        n, p = 30, 3
        r2 = 0.2
        f2 = r2 / (1.0 - r2)
        X = _full_rank_X(n, p)
        result = global_r2_power(r2, X, alpha=0.05, lambda_mode="n_minus_p")
        expected_lam = f2 * (n - p)  # f2 * 27 = 6.75
        assert abs(result.lam - expected_lam) < 1e-6

    def test_n_minus_p_more_conservative_than_n(self):
        """n_minus_p should produce a smaller lambda and ≤ power vs 'n' mode."""
        X = _full_rank_X(30, 3)
        r_n = global_r2_power(0.2, X, alpha=0.05, lambda_mode="n")
        r_nm = global_r2_power(0.2, X, alpha=0.05, lambda_mode="n_minus_p")
        assert r_nm.lam < r_n.lam
        assert r_nm.power <= r_n.power

    def test_raises_on_r2_zero(self):
        X = _full_rank_X(20, 2)
        with pytest.raises(ValueError):
            global_r2_power(r2_target=0.0, X=X, alpha=0.05)

    def test_raises_on_r2_one(self):
        X = _full_rank_X(20, 2)
        with pytest.raises(ValueError):
            global_r2_power(r2_target=1.0, X=X, alpha=0.05)

    def test_df_num_excludes_intercept(self):
        """With an intercept column, df_num must equal rank(X)-1, not rank(X).

        Verified by computing expected power directly with scipy using df1 = p-1
        and confirming it matches global_r2_power's output.
        """
        from scipy.stats import ncf, f as scipy_f
        n, p = 30, 3
        X = _full_rank_X(n, p)   # col 0 = all-ones intercept; rank = p
        r2, alpha = 0.2, 0.05
        f2 = r2 / (1.0 - r2)

        result = global_r2_power(r2, X, alpha=alpha, lambda_mode="n")

        # Expected: df1 = p-1 (slopes only), df2 = n-p, lam = f2*n
        df1_expected = p - 1          # 2
        df2_expected = n - p          # 27
        lam_expected = f2 * n
        fcrit = scipy_f.isf(alpha, df1_expected, df2_expected)
        power_expected = float(1.0 - ncf.cdf(fcrit, df1_expected, df2_expected, lam_expected))

        assert abs(result.lam - lam_expected) < 1e-9
        assert abs(result.power - power_expected) < 1e-9

    def test_raises_when_df_denom_zero(self):
        X = _full_rank_X(3, 3)  # df_denom = 0
        with pytest.raises(ValueError):
            global_r2_power(r2_target=0.2, X=X, alpha=0.05)
