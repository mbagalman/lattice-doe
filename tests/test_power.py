# tests/test_power.py
"""Unit tests for power.py — contrast_power and global_r2_power."""
import numpy as np
import pytest

from lattice_doe.power import contrast_power, global_r2_power, eval_response_power, combine_powers, hotelling_t2_power, glm_contrast_power
from lattice_doe.config import (
    PowerContrastConfig, PowerR2Config, ResponseSpec, SplitPlotOptions,
    PowerGLMContrastConfig, glm_fisher_weight,
)


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


# ---------------------------------------------------------------------------
# eval_response_power
# ---------------------------------------------------------------------------

def _make_X(n=30, p=3, seed=7):
    rng = np.random.default_rng(seed)
    X = np.ones((n, p))
    X[:, 1:] = rng.standard_normal((n, p - 1))
    return X


def _contrast_rs(name="Y1", **kw):
    defaults = dict(L=[[0, 1, 0]], delta=[1.0], sigma=1.0, alpha=0.05, power=0.8)
    defaults.update(kw)
    cfg = PowerContrastConfig(**defaults)
    return ResponseSpec(name=name, power_cfg=cfg)


def _r2_rs(name="Y2", **kw):
    defaults = dict(r2_target=0.4, alpha=0.05, power=0.8)
    defaults.update(kw)
    cfg = PowerR2Config(**defaults)
    return ResponseSpec(name=name, power_cfg=cfg)


class TestEvalResponsePower:
    def test_contrast_returns_dict(self):
        X = _make_X()
        p_names = ["Intercept", "A", "B"]
        result = eval_response_power(_contrast_rs(), X, p_names)
        assert isinstance(result, dict)

    def test_contrast_dict_keys(self):
        X = _make_X()
        p_names = ["Intercept", "A", "B"]
        result = eval_response_power(_contrast_rs(), X, p_names)
        assert set(result.keys()) == {"name", "power", "lam", "df1", "df2"}

    def test_contrast_name_matches_response(self):
        X = _make_X()
        result = eval_response_power(_contrast_rs(name="Yield"), X, ["Intercept", "A", "B"])
        assert result["name"] == "Yield"

    def test_contrast_power_matches_contrast_power_direct(self):
        X = _make_X()
        p_names = ["Intercept", "A", "B"]
        rs = _contrast_rs()
        cfg = rs.power_cfg
        expected = contrast_power(cfg.L, cfg.delta, X, cfg.sigma, cfg.alpha)
        result = eval_response_power(rs, X, p_names)
        assert abs(result["power"] - expected.power) < 1e-12
        assert abs(result["lam"] - expected.lam) < 1e-12

    def test_contrast_lam_matches_direct(self):
        X = _make_X()
        p_names = ["Intercept", "A", "B"]
        rs = _contrast_rs()
        cfg = rs.power_cfg
        expected = contrast_power(cfg.L, cfg.delta, X, cfg.sigma, cfg.alpha)
        result = eval_response_power(rs, X, p_names)
        assert abs(result["lam"] - expected.lam) < 1e-12

    def test_contrast_df1_equals_L_rows(self):
        X = _make_X()
        rs = _contrast_rs()
        result = eval_response_power(rs, X, ["Intercept", "A", "B"])
        assert result["df1"] == rs.power_cfg.L.shape[0]

    def test_contrast_df2_equals_n_minus_rank(self):
        X = _make_X(n=30, p=3)
        rs = _contrast_rs()
        result = eval_response_power(rs, X, ["Intercept", "A", "B"])
        expected_df2 = int(X.shape[0] - np.linalg.matrix_rank(X))
        assert result["df2"] == expected_df2

    def test_r2_returns_dict(self):
        X = _make_X()
        result = eval_response_power(_r2_rs(), X, ["Intercept", "A", "B"])
        assert isinstance(result, dict)

    def test_r2_dict_keys_no_df1(self):
        X = _make_X()
        result = eval_response_power(_r2_rs(), X, ["Intercept", "A", "B"])
        assert "df1" not in result
        assert "df2" in result

    def test_r2_power_matches_global_r2_power_direct(self):
        X = _make_X()
        rs = _r2_rs()
        cfg = rs.power_cfg
        from lattice_doe.power import global_r2_power
        expected = global_r2_power(cfg.r2_target, X, cfg.alpha, lambda_mode=cfg.lambda_mode)
        result = eval_response_power(rs, X, ["Intercept", "A", "B"])
        assert abs(result["power"] - expected.power) < 1e-12
        assert abs(result["lam"] - expected.lam) < 1e-12

    def test_r2_name_in_result(self):
        X = _make_X()
        result = eval_response_power(_r2_rs(name="Purity"), X, ["Intercept", "A", "B"])
        assert result["name"] == "Purity"

    def test_power_value_in_zero_one(self):
        X = _make_X()
        result = eval_response_power(_contrast_rs(), X, ["Intercept", "A", "B"])
        assert 0.0 <= result["power"] <= 1.0

    def test_lam_nonnegative(self):
        X = _make_X()
        result = eval_response_power(_contrast_rs(), X, ["Intercept", "A", "B"])
        assert result["lam"] >= 0.0

    def test_split_plot_missing_Z_raises(self):
        X = _make_X()
        sp_opts = SplitPlotOptions(htc_factors=["A"], n_whole_plots=5)
        with pytest.raises(ValueError, match="Z"):
            eval_response_power(
                _contrast_rs(), X, ["Intercept", "A", "B"],
                split_plot_opts=sp_opts, Z=None,
            )

    def test_exported_from_top_level(self):
        import lattice_doe
        assert hasattr(lattice_doe, "eval_response_power")


# ---------------------------------------------------------------------------
# combine_powers
# ---------------------------------------------------------------------------

class TestCombinePowers:
    def test_min_returns_minimum(self):
        assert combine_powers([0.8, 0.9], None, "min") == 0.8

    def test_min_single_value(self):
        assert combine_powers([0.75], None, "min") == 0.75

    def test_min_three_values(self):
        assert combine_powers([0.9, 0.7, 0.85], None, "min") == 0.7

    def test_product_two_values(self):
        assert combine_powers([0.8, 0.9], None, "product") == pytest.approx(0.72)

    def test_product_three_values(self):
        assert combine_powers([0.8, 0.9, 0.5], None, "product") == pytest.approx(0.36)

    def test_product_single_value(self):
        assert combine_powers([0.65], None, "product") == pytest.approx(0.65)

    def test_weighted_mean_equal_weights(self):
        # equal weights → arithmetic mean
        result = combine_powers([0.8, 0.9], [1.0, 1.0], "weighted_mean")
        assert result == pytest.approx(0.85)

    def test_weighted_mean_unequal_weights(self):
        # 2×0.8 + 1×0.9 / 3 = (1.6 + 0.9)/3 = 2.5/3
        result = combine_powers([0.8, 0.9], [2.0, 1.0], "weighted_mean")
        assert result == pytest.approx(2.5 / 3.0)

    def test_weighted_mean_none_weights_uses_equal(self):
        # None → equal weights → mean
        result = combine_powers([0.8, 0.9], None, "weighted_mean")
        assert result == pytest.approx(0.85)

    def test_weighted_mean_weights_normalised(self):
        # weights [4, 2] same ratio as [2, 1]; result identical to above
        result = combine_powers([0.8, 0.9], [4.0, 2.0], "weighted_mean")
        assert result == pytest.approx(2.5 / 3.0)

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="unknown combination rule"):
            combine_powers([0.8, 0.9], None, "geometric")

    def test_empty_powers_raises(self):
        with pytest.raises(ValueError, match="empty"):
            combine_powers([], None, "min")

    def test_min_ignores_weights(self):
        # weights are irrelevant for "min"
        assert combine_powers([0.8, 0.9], [99.0, 1.0], "min") == 0.8

    def test_product_ignores_weights(self):
        assert combine_powers([0.8, 0.9], [99.0, 1.0], "product") == pytest.approx(0.72)

    def test_result_in_zero_one_for_valid_inputs(self):
        for rule in ("min", "product", "weighted_mean"):
            result = combine_powers([0.6, 0.7, 0.8], None, rule)
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestHotellingT2Power  (MR-6)
# ---------------------------------------------------------------------------

def _ht2_X(n: int = 30, p: int = 3, seed: int = 42) -> np.ndarray:
    """Small full-rank design matrix for Hotelling T² tests."""
    rng = np.random.default_rng(seed)
    X = np.ones((n, p))
    X[:, 1:] = rng.standard_normal((n, p - 1))
    return X


class TestHotellingT2Power:
    """MR-6: Hotelling T² joint power for multi-response linear contrasts."""

    def test_returns_named_tuple(self):
        X = _ht2_X()
        L = np.array([[0, 1, 0]])
        Delta = np.array([[1.5, 1.5]])
        Sigma = np.eye(2)
        result = hotelling_t2_power(L, Delta, X, Sigma)
        assert hasattr(result, "power")
        assert hasattr(result, "lam")
        assert hasattr(result, "df1")
        assert hasattr(result, "df2")

    def test_power_in_unit_interval(self):
        X = _ht2_X()
        L = np.array([[0, 1, 0]])
        Delta = np.array([[1.5, 1.5]])
        result = hotelling_t2_power(L, Delta, X, np.eye(2))
        assert 0.0 <= result.power <= 1.0

    def test_k1_sigma_identity_matches_contrast_power(self):
        # For k=1, sigma_joint=[[sigma²]], should match contrast_power exactly.
        X = _ht2_X(n=30, p=3)
        L = np.array([[0, 1, 0]])
        delta = np.array([1.5])
        sigma = 1.0
        # contrast_power with q=1, sigma=1
        cp = contrast_power(L, delta, X, sigma=sigma, alpha=0.05)
        # hotelling_t2_power with k=1, sigma_joint=[[sigma²]]
        Delta = delta.reshape(-1, 1)
        ht2 = hotelling_t2_power(L, Delta, X, np.array([[sigma ** 2]]))
        assert ht2.power == pytest.approx(cp.power, abs=1e-6)

    def test_k1_sigma_not_1_matches_contrast_power(self):
        # For k=1, sigma_joint=[[σ²]], any sigma value.
        X = _ht2_X(n=40, p=3)
        L = np.array([[0, 1, 0], [0, 0, 1]])
        delta = np.array([1.2, 0.8])
        sigma = 1.5
        cp = contrast_power(L, delta, X, sigma=sigma, alpha=0.05)
        Delta = delta.reshape(-1, 1)
        ht2 = hotelling_t2_power(L, Delta, X, np.array([[sigma ** 2]]))
        assert ht2.power == pytest.approx(cp.power, abs=1e-6)

    def test_identity_sigma_k2_geq_individual_min(self):
        # With sigma_joint=I_2, Hotelling T² power should be at least the individual min.
        X = _ht2_X(n=40, p=3)
        L = np.array([[0, 1, 0]])
        delta1 = np.array([1.5])
        delta2 = np.array([1.5])
        cp1 = contrast_power(L, delta1, X, sigma=1.0)
        Delta = np.column_stack([delta1, delta2])
        ht2 = hotelling_t2_power(L, Delta, X, np.eye(2))
        assert ht2.power >= min(cp1.power, cp1.power) - 1e-6

    def test_lam_positive(self):
        X = _ht2_X()
        L = np.array([[0, 1, 0]])
        Delta = np.array([[1.0, 1.0]])
        result = hotelling_t2_power(L, Delta, X, np.eye(2))
        assert result.lam >= 0.0

    def test_df1_equals_q_times_k(self):
        X = _ht2_X(n=50, p=3)
        L = np.array([[0, 1, 0], [0, 0, 1]])  # q=2
        Delta = np.column_stack([np.array([1.0, 0.8]), np.array([1.0, 0.8]), np.array([1.0, 0.8])])  # k=3
        result = hotelling_t2_power(L, Delta, X, np.eye(3))
        assert result.df1 == 2 * 3  # q * k

    def test_df2_correct(self):
        n, p, k = 30, 3, 2
        X = _ht2_X(n=n, p=p)
        L = np.array([[0, 1, 0]])
        Delta = np.array([[1.0, 1.0]])
        result = hotelling_t2_power(L, Delta, X, np.eye(k))
        # df2 = n - rank(X) - k + 1 = 30 - 3 - 2 + 1 = 26
        assert result.df2 == n - p - k + 1

    def test_singular_sigma_raises(self):
        X = _ht2_X()
        L = np.array([[0, 1, 0]])
        Delta = np.array([[1.0, 1.0]])
        # Singular: row 1 = row 2
        Sigma_sing = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="singular"):
            hotelling_t2_power(L, Delta, X, Sigma_sing)

    def test_non_symmetric_sigma_raises(self):
        X = _ht2_X()
        L = np.array([[0, 1, 0]])
        Delta = np.array([[1.0, 1.0]])
        Sigma_asym = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            hotelling_t2_power(L, Delta, X, Sigma_asym)

    def test_delta_1d_treated_as_single_response(self):
        # 1-D delta is treated as q×1 Delta.
        X = _ht2_X()
        L = np.array([[0, 1, 0]])
        delta_1d = np.array([1.5])
        sigma_sq = np.array([[1.0]])
        result = hotelling_t2_power(L, delta_1d, X, sigma_sq)
        assert 0.0 <= result.power <= 1.0

    def test_higher_effect_size_gives_higher_power(self):
        X = _ht2_X(n=30, p=3)
        L = np.array([[0, 1, 0]])
        Sigma = np.eye(2)
        low = hotelling_t2_power(L, np.array([[0.5, 0.5]]), X, Sigma)
        high = hotelling_t2_power(L, np.array([[2.0, 2.0]]), X, Sigma)
        assert high.power > low.power

    def test_exported_from_top_level(self):
        import lattice_doe
        assert hasattr(lattice_doe, "hotelling_t2_power")


# ---------------------------------------------------------------------------
# MR-10 — combine_powers and hotelling_t2_power property-based tests
# ---------------------------------------------------------------------------


class TestMR10CombinePowersProperties:
    """MR-10: parametrized properties of combine_powers."""

    @pytest.mark.parametrize("powers,weights", [
        ([0.8, 0.9], [1.0, 1.0]),
        ([0.5, 0.7, 0.6], [1.0, 2.0, 1.0]),
        ([0.3, 0.95], [3.0, 1.0]),
        ([0.85, 0.85], None),
        ([0.1, 0.99, 0.5], [1.0, 1.0, 1.0]),
        ([0.8, 0.8, 0.8], [2.0, 1.0, 3.0]),
    ])
    def test_weighted_mean_between_min_and_max(self, powers, weights):
        """min(powers) <= weighted_mean <= max(powers) for any input."""
        result = combine_powers(powers, weights, "weighted_mean")
        assert min(powers) <= result <= max(powers) + 1e-12

    @pytest.mark.parametrize("powers,weights", [
        ([0.8, 0.9], [1.0, 2.0]),
        ([0.6, 0.4, 0.7], [2.0, 1.0, 3.0]),
        ([0.5, 0.5], [10.0, 10.0]),
    ])
    def test_product_leq_weighted_mean(self, powers, weights):
        """product rule <= weighted_mean for powers in (0,1)."""
        r_prod = combine_powers(powers, weights, "product")
        r_wm = combine_powers(powers, weights, "weighted_mean")
        assert r_prod <= r_wm + 1e-12

    @pytest.mark.parametrize("powers", [
        [0.8, 0.9],
        [0.5, 0.7, 0.6],
        [0.3, 0.95],
    ])
    def test_min_leq_weighted_mean_leq_max(self, powers):
        """min <= weighted_mean <= max for equal weights."""
        r = combine_powers(powers, None, "weighted_mean")
        assert min(powers) - 1e-12 <= r <= max(powers) + 1e-12

    def test_three_equal_powers_all_rules_same(self):
        """Equal powers p: min=product^(1/k)=wm=p."""
        powers = [0.8, 0.8, 0.8]
        r_min = combine_powers(powers, None, "min")
        r_wm = combine_powers(powers, None, "weighted_mean")
        assert r_min == pytest.approx(0.8)
        assert r_wm == pytest.approx(0.8)

    def test_min_rule_returns_exact_minimum(self):
        """min rule returns exactly the smallest value."""
        assert combine_powers([0.9, 0.3, 0.7], None, "min") == pytest.approx(0.3)

    def test_product_decreases_with_more_responses(self):
        """Adding more sub-unit powers under product gives lower combined."""
        p2 = combine_powers([0.8, 0.9], None, "product")
        p3 = combine_powers([0.8, 0.9, 0.85], None, "product")
        assert p3 < p2

    def test_weighted_mean_higher_weight_on_higher_power(self):
        """Placing higher weight on larger power raises weighted mean above equal-weight mean."""
        powers = [0.6, 0.9]
        r_equal = combine_powers(powers, [1.0, 1.0], "weighted_mean")
        r_heavier = combine_powers(powers, [1.0, 3.0], "weighted_mean")  # more weight on 0.9
        assert r_heavier > r_equal

    def test_combine_powers_all_rules_output_in_unit_interval(self):
        """All combination rules produce values in [0, 1] for valid inputs."""
        powers = [0.7, 0.85, 0.6]
        for rule in ("min", "product", "weighted_mean"):
            r = combine_powers(powers, [1.0, 2.0, 1.0], rule)
            assert 0.0 <= r <= 1.0, f"Rule {rule} gave {r} outside [0,1]"


class TestMR10HotellingT2Properties:
    """MR-10: property-based tests for hotelling_t2_power."""

    def test_large_diagonal_sigma_power_near_zero(self):
        """Very large diagonal sigma_joint => joint power approaches 0."""
        X = _ht2_X(n=40, p=3)
        L = np.array([[0, 1, 0]])
        Delta = np.array([[1.0, 1.0]])
        sigma_large = np.diag([1e8, 1e8])
        result = hotelling_t2_power(L, Delta, X, sigma_large)
        assert result.power < 0.10

    def test_scaling_sigma_joint_monotone_in_power(self):
        """Increasing scale of sigma_joint decreases power monotonically."""
        X = _ht2_X(n=50, p=3)
        L = np.array([[0, 1, 0]])
        Delta = np.array([[2.0, 2.0]])
        r_small = hotelling_t2_power(L, Delta, X, np.eye(2))
        r_large = hotelling_t2_power(L, Delta, X, 100.0 * np.eye(2))
        assert r_small.power >= r_large.power

    def test_larger_n_gives_higher_power(self):
        """Larger design matrix (more runs) gives higher joint power."""
        L = np.array([[0, 1, 0]])
        Delta = np.array([[1.0, 1.0]])
        r_small = hotelling_t2_power(L, Delta, _ht2_X(n=15, p=3), np.eye(2))
        r_large = hotelling_t2_power(L, Delta, _ht2_X(n=80, p=3), np.eye(2))
        assert r_large.power >= r_small.power

    def test_lam_nonneg_for_any_nonzero_delta(self):
        """Noncentrality lambda >= 0 for any valid input."""
        X = _ht2_X(n=30, p=3)
        L = np.array([[0, 1, 0]])
        Delta = np.array([[0.5, 0.5]])
        result = hotelling_t2_power(L, Delta, X, np.eye(2))
        assert result.lam >= 0.0

    def test_zero_delta_gives_alpha_level_power(self):
        """Zero effect (delta=0) should give power == alpha (no effect)."""
        X = _ht2_X(n=30, p=3)
        L = np.array([[0, 1, 0]])
        Delta = np.array([[0.0, 0.0]])
        result = hotelling_t2_power(L, Delta, X, np.eye(2), alpha=0.05)
        assert result.power == pytest.approx(0.05, abs=0.01)

    def test_identity_sigma_power_between_0_and_1(self):
        """Identity sigma_joint gives power in [0, 1]."""
        X = _ht2_X(n=30, p=3)
        L = np.array([[0, 1, 0]])
        Delta = np.array([[1.5, 1.5]])
        result = hotelling_t2_power(L, Delta, X, np.eye(2))
        assert 0.0 <= result.power <= 1.0

    @pytest.mark.parametrize("k", [2, 3])
    def test_df1_equals_q_times_k_parametrized(self, k):
        """df1 == q * k for any valid q and k."""
        q = 2
        X = _ht2_X(n=40, p=3)
        L = np.array([[0, 1, 0], [0, 0, 1]])  # q=2
        Delta = np.ones((q, k))
        result = hotelling_t2_power(L, Delta, X, np.eye(k))
        assert result.df1 == q * k


# ---------------------------------------------------------------------------
# GLM contrast power (GL-2)
# ---------------------------------------------------------------------------

def _glm_X(n: int = 40, p: int = 3, seed: int = 0) -> np.ndarray:
    """Full-rank design matrix for GLM tests."""
    rng = np.random.default_rng(seed)
    X = np.ones((n, p))
    X[:, 1:] = rng.standard_normal((n, p - 1))
    return X


def _binomial_cfg(**kwargs) -> PowerGLMContrastConfig:
    defaults = dict(
        L=np.array([[0.0, 1.0, 0.0]]),
        delta=np.array([0.3]),
        baseline=0.5,
        family="binomial",
        alpha=0.05,
    )
    defaults.update(kwargs)
    return PowerGLMContrastConfig(**defaults)


def _poisson_cfg(**kwargs) -> PowerGLMContrastConfig:
    defaults = dict(
        L=np.array([[0.0, 1.0, 0.0]]),
        delta=np.array([0.5]),
        baseline=2.0,
        family="poisson",
        alpha=0.05,
    )
    defaults.update(kwargs)
    return PowerGLMContrastConfig(**defaults)


class TestGLMContrastPower:
    # --- Return type and basic properties ---

    def test_returns_contrast_power_result(self):
        cfg = _binomial_cfg()
        result = glm_contrast_power(cfg, _glm_X())
        assert hasattr(result, "power")
        assert hasattr(result, "lam")

    def test_power_in_unit_interval_binomial(self):
        result = glm_contrast_power(_binomial_cfg(), _glm_X())
        assert 0.0 <= result.power <= 1.0

    def test_power_in_unit_interval_poisson(self):
        result = glm_contrast_power(_poisson_cfg(), _glm_X())
        assert 0.0 <= result.power <= 1.0

    def test_lam_nonneg_binomial(self):
        result = glm_contrast_power(_binomial_cfg(), _glm_X())
        assert result.lam >= 0.0

    def test_lam_nonneg_poisson(self):
        result = glm_contrast_power(_poisson_cfg(), _glm_X())
        assert result.lam >= 0.0

    # --- Power increases with n ---

    def test_power_increases_with_n_binomial(self):
        cfg = _binomial_cfg(delta=np.array([0.4]))
        L_row = cfg.L
        powers = [
            glm_contrast_power(
                PowerGLMContrastConfig(L=L_row, delta=np.array([0.4]), baseline=0.5,
                                       family="binomial"),
                _glm_X(n=n),
            ).power
            for n in [10, 40, 120]
        ]
        assert powers[0] < powers[1] < powers[2]

    def test_power_increases_with_n_poisson(self):
        powers = [
            glm_contrast_power(
                PowerGLMContrastConfig(L=np.array([[0.0, 1.0, 0.0]]),
                                       delta=np.array([0.2]), baseline=1.0,
                                       family="poisson"),
                _glm_X(n=n),
            ).power
            for n in [10, 40, 120]
        ]
        assert powers[0] < powers[1] < powers[2]

    # --- Power increases with delta ---

    def test_power_increases_with_delta_binomial(self):
        X = _glm_X(n=50)
        L = np.array([[0.0, 1.0, 0.0]])
        powers = [
            glm_contrast_power(
                PowerGLMContrastConfig(L=L, delta=np.array([d]), baseline=0.5,
                                       family="binomial"),
                X,
            ).power
            for d in [0.1, 0.5, 1.5]
        ]
        assert powers[0] < powers[1] < powers[2]

    def test_power_increases_with_delta_poisson(self):
        X = _glm_X(n=50)
        L = np.array([[0.0, 1.0, 0.0]])
        powers = [
            glm_contrast_power(
                PowerGLMContrastConfig(L=L, delta=np.array([d]), baseline=2.0,
                                       family="poisson"),
                X,
            ).power
            for d in [0.1, 0.5, 2.0]
        ]
        assert powers[0] < powers[1] < powers[2]

    # --- Equivalence to OLS with sigma_eff = 1/sqrt(w) ---

    def test_glm_binomial_matches_ols_sigma_eff(self):
        """GLM binomial power == OLS contrast power with sigma = 1/sqrt(w)."""
        X = _glm_X(n=60)
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        cfg = PowerGLMContrastConfig(L=L, delta=delta, baseline=0.4, family="binomial")
        w = glm_fisher_weight(cfg)
        sigma_eff = 1.0 / np.sqrt(w)
        glm_result = glm_contrast_power(cfg, X)
        ols_result = contrast_power(L, delta, X, sigma=sigma_eff, alpha=cfg.alpha)
        # Power should match (noncentral chi-square vs noncentral F differ slightly
        # in small samples but are asymptotically equivalent; check lambda instead)
        assert glm_result.lam == pytest.approx(ols_result.lam, rel=1e-6)

    def test_glm_poisson_matches_ols_sigma_eff(self):
        """GLM Poisson power == OLS contrast power with sigma = 1/sqrt(mu0)."""
        X = _glm_X(n=60)
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        cfg = PowerGLMContrastConfig(L=L, delta=delta, baseline=3.0, family="poisson")
        w = glm_fisher_weight(cfg)
        sigma_eff = 1.0 / np.sqrt(w)
        glm_result = glm_contrast_power(cfg, X)
        ols_result = contrast_power(L, delta, X, sigma=sigma_eff, alpha=cfg.alpha)
        assert glm_result.lam == pytest.approx(ols_result.lam, rel=1e-6)

    # --- Zero delta → power ≈ alpha ---
    # PowerGLMContrastConfig rejects delta=0 at construction, so we build a valid
    # config then overwrite delta to exercise the power function directly.

    def test_zero_delta_power_equals_alpha_binomial(self):
        cfg = _binomial_cfg()
        object.__setattr__(cfg, "delta", np.array([0.0]))
        result = glm_contrast_power(cfg, _glm_X(n=100))
        assert result.power == pytest.approx(0.05, abs=0.01)

    def test_zero_delta_power_equals_alpha_poisson(self):
        cfg = _poisson_cfg()
        object.__setattr__(cfg, "delta", np.array([0.0]))
        result = glm_contrast_power(cfg, _glm_X(n=100))
        assert result.power == pytest.approx(0.05, abs=0.01)

    # --- 1-D L treated as single-row ---

    def test_1d_L_treated_as_single_row(self):
        X = _glm_X()
        cfg_2d = _binomial_cfg(L=np.array([[0.0, 1.0, 0.0]]))
        cfg_1d = _binomial_cfg(L=np.array([0.0, 1.0, 0.0]))
        r_2d = glm_contrast_power(cfg_2d, X)
        r_1d = glm_contrast_power(cfg_1d, X)
        assert r_2d.lam == pytest.approx(r_1d.lam, rel=1e-10)

    # --- Multi-row contrast (q > 1) ---

    def test_multirow_contrast_binomial(self):
        X = _glm_X(n=60, p=4)
        L = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        delta = np.array([0.4, 0.4])
        cfg = PowerGLMContrastConfig(L=L, delta=delta, baseline=0.5, family="binomial")
        result = glm_contrast_power(cfg, X)
        assert 0.0 <= result.power <= 1.0
        assert result.lam >= 0.0

    # --- Higher baseline Poisson → higher weight → higher power ---

    def test_higher_baseline_poisson_increases_power(self):
        X = _glm_X(n=40)
        L = np.array([[0.0, 1.0, 0.0]])
        delta = np.array([0.5])
        r_low = glm_contrast_power(
            PowerGLMContrastConfig(L=L, delta=delta, baseline=1.0, family="poisson"), X
        )
        r_high = glm_contrast_power(
            PowerGLMContrastConfig(L=L, delta=delta, baseline=10.0, family="poisson"), X
        )
        assert r_high.power >= r_low.power

    # --- Stricter alpha → less power ---

    def test_stricter_alpha_decreases_power(self):
        X = _glm_X(n=50)
        cfg_lenient = _binomial_cfg(alpha=0.10)
        cfg_strict = _binomial_cfg(alpha=0.01)
        r_lenient = glm_contrast_power(cfg_lenient, X)
        r_strict = glm_contrast_power(cfg_strict, X)
        assert r_lenient.power >= r_strict.power

    # --- eval_response_power dispatches GLM correctly ---

    def test_eval_response_power_dispatches_glm(self):
        X = _glm_X(n=50)
        p_names = ["(Intercept)", "x1", "x2"]
        cfg = _binomial_cfg()
        spec = ResponseSpec(name="y", formula="~x1+x2", power_cfg=cfg)
        result = eval_response_power(spec, X, p_names)
        assert result["name"] == "y"
        assert 0.0 <= result["power"] <= 1.0
        assert result["lam"] >= 0.0
        assert result["df2"] is None
        assert result["family"] == "binomial"
        assert result["baseline"] == pytest.approx(0.5)

    def test_eval_response_power_glm_poisson(self):
        X = _glm_X(n=50)
        p_names = ["(Intercept)", "x1", "x2"]
        cfg = _poisson_cfg()
        spec = ResponseSpec(name="counts", formula="~x1+x2", power_cfg=cfg)
        result = eval_response_power(spec, X, p_names)
        assert result["family"] == "poisson"
        assert result["df2"] is None

    # --- Bad input errors ---

    def test_bad_X_ndim_raises(self):
        cfg = _binomial_cfg()
        with pytest.raises(ValueError, match="2D"):
            glm_contrast_power(cfg, np.ones(10))

    def test_zero_rank_L_raises(self):
        """A zero L matrix has rank 0 — should raise."""
        L = np.zeros((1, 3))
        # PowerGLMContrastConfig validates L at construction; bypass by patching
        cfg = _binomial_cfg()
        object.__setattr__(cfg, "L", L)
        # V_unscaled will be zero-matrix → rank 0
        with pytest.raises(ValueError):
            glm_contrast_power(cfg, _glm_X())


# ---------------------------------------------------------------------------
# SR-8: the X'X ridge must be scale-relative, not absolute
# ---------------------------------------------------------------------------

class TestSR8ScaleInvariantJitter:
    """SR-8 regression: an absolute ridge (jitter * I) dominated columns in
    small physical units and silently inflated the noncentrality parameter
    (power 0.81 -> 1.000 at column scale 1e-5 before the fix). The ridge is
    now relative to each diagonal entry, so the same physical effect gives
    the same power regardless of the units the factor columns are stated in."""

    def _design(self):
        rng = np.random.default_rng(0)
        n = 12
        x = rng.uniform(-1, 1, n)
        X = np.column_stack([np.ones(n), x])
        return X, np.array([[0.0, 1.0]]), np.array([1.2])

    @pytest.mark.parametrize("scale", [1e-3, 1e-5, 1e-8, 1e3, 1e6])
    def test_contrast_power_unit_invariant(self, scale):
        from lattice_doe.power import contrast_power
        X, L, delta = self._design()
        base = contrast_power(L, delta, X, sigma=1.0, alpha=0.05)
        Xs = X.copy()
        Xs[:, 1] *= scale
        # Same physical effect: the slope coefficient scales by 1/scale.
        res = contrast_power(L, delta / scale, Xs, sigma=1.0, alpha=0.05)
        assert res.lam == pytest.approx(base.lam, rel=1e-6)
        assert res.power == pytest.approx(base.power, abs=1e-9)

    def test_contrast_power_sp_unit_invariant(self):
        from lattice_doe.power import contrast_power_sp
        from lattice_doe.split_plot import build_whole_plot_indicator
        X, L, delta = self._design()
        Z = build_whole_plot_indicator(12, 4, 3)
        base = contrast_power_sp(L, delta, X, Z, sigma_sp=1.0, eta=1.0, alpha=0.05)
        Xs = X.copy()
        Xs[:, 1] *= 1e-5
        res = contrast_power_sp(L, delta / 1e-5, Xs, Z, sigma_sp=1.0, eta=1.0,
                                alpha=0.05)
        assert res.lam == pytest.approx(base.lam, rel=1e-6)
        assert res.power == pytest.approx(base.power, abs=1e-9)

    def test_glm_contrast_power_unit_invariant(self):
        from lattice_doe.power import glm_contrast_power
        from lattice_doe.config import PowerGLMContrastConfig
        X, L, delta = self._design()
        cfg = PowerGLMContrastConfig(L=L, delta=delta, baseline=0.3,
                                     family="binomial")
        base = glm_contrast_power(cfg, X)
        Xs = X.copy()
        Xs[:, 1] *= 1e-5
        cfg_s = PowerGLMContrastConfig(L=L, delta=delta / 1e-5, baseline=0.3,
                                       family="binomial")
        res = glm_contrast_power(cfg_s, Xs)
        assert res.power == pytest.approx(base.power, abs=1e-9)

    def test_unit_scale_results_unchanged(self):
        """At ordinary scales the relative ridge is as negligible as the old
        absolute one: lambda matches the ridgeless computation closely."""
        from lattice_doe.power import contrast_power
        X, L, delta = self._design()
        res = contrast_power(L, delta, X, sigma=1.0, alpha=0.05)
        XtX_inv = np.linalg.inv(X.T @ X)
        lam_exact = float(delta @ np.linalg.inv(L @ XtX_inv @ L.T) @ delta)
        assert res.lam == pytest.approx(lam_exact, rel=1e-6)


# ---------------------------------------------------------------------------
# SR-10: implicit intercepts must be detected for the global R2 numerator df
# ---------------------------------------------------------------------------

class TestSR10ImplicitIntercept:
    """SR-10 regression: _r2_df_num scanned for a literal all-ones column, so
    cell-means coding (0 + C(group)) got df1 = k instead of k - 1 -- a 7.5 pp
    power error in the audit's k=3 example. The intercept is now detected as
    the constant vector lying in the column span (rank([X, 1]) == rank(X))."""

    def _group_designs(self, n=30, k=3):
        g = np.repeat(np.arange(k), n // k)
        X_cm = np.zeros((n, k))
        X_cm[np.arange(n), g] = 1.0
        X_int = np.column_stack(
            [np.ones(n)] + [(g == j).astype(float) for j in range(1, k)]
        )
        return X_cm, X_int

    def test_cell_means_df_matches_intercept_coding(self):
        from lattice_doe.power import _r2_df_num
        X_cm, X_int = self._group_designs()
        assert _r2_df_num(X_cm) == _r2_df_num(X_int) == 2

    def test_cell_means_power_matches_intercept_coding(self):
        """Same model space must give the same global R2 power under either
        coding (MC-verified at 0.767 for this configuration)."""
        X_cm, X_int = self._group_designs()
        p_cm = global_r2_power(0.25, X_cm, alpha=0.05).power
        p_int = global_r2_power(0.25, X_int, alpha=0.05).power
        assert p_cm == pytest.approx(p_int, abs=1e-12)
        assert p_cm == pytest.approx(0.7674, abs=5e-4)

    def test_constant_non_unit_column_detected(self):
        from lattice_doe.power import _r2_df_num
        _, X_int = self._group_designs()
        X_c2 = X_int.copy()
        X_c2[:, 0] = 2.0  # constant column that is not all-ones
        assert _r2_df_num(X_c2) == 2

    def test_true_no_intercept_model_unchanged(self):
        """A genuine through-the-origin model keeps df1 = rank(X)."""
        from lattice_doe.power import _r2_df_num
        rng = np.random.default_rng(3)
        X = np.column_stack(
            [rng.uniform(0.5, 1.5, 24), rng.uniform(-1.0, -0.2, 24)]
        )
        assert _r2_df_num(X) == 2

    def test_explicit_intercept_unchanged(self):
        from lattice_doe.power import _r2_df_num
        _, X_int = self._group_designs()
        assert _r2_df_num(X_int) == 2


# ---------------------------------------------------------------------------
# SR-9: infeasible hypotheses (rank-deficient L, inconsistent delta) must raise
# ---------------------------------------------------------------------------

class TestSR9DeltaConsistency:
    """SR-9 regression: with linearly dependent contrast rows, delta must
    satisfy the same dependencies for L*beta = delta to be feasible. The
    pseudo-inverse used to silently project an infeasible delta onto
    range(V) -- a sign-flipped duplicate gave lambda = 0 (power = alpha) and
    a zeroed duplicate gave lambda/4, both without warning. All four power
    functions now raise a ValueError instead."""

    def _fixture(self):
        rng = np.random.default_rng(0)
        n = 16
        x = rng.uniform(-1, 1, n)
        X = np.column_stack([np.ones(n), x])
        l = np.array([0.0, 1.0])
        return X, l, np.vstack([l, l])

    def test_consistent_duplicate_matches_single_row(self):
        X, l, L2 = self._fixture()
        ref = contrast_power(l, np.array([1.2]), X, sigma=1.0, alpha=0.05)
        dup = contrast_power(L2, np.array([1.2, 1.2]), X, sigma=1.0, alpha=0.05)
        assert dup.lam == pytest.approx(ref.lam, abs=1e-8)
        assert dup.power == pytest.approx(ref.power, abs=1e-10)

    @pytest.mark.parametrize("bad_delta", [[1.2, -1.2], [1.2, 0.0]])
    def test_contrast_power_inconsistent_delta_raises(self, bad_delta):
        X, _, L2 = self._fixture()
        with pytest.raises(ValueError, match="inconsistent with the linear"):
            contrast_power(L2, np.array(bad_delta), X, sigma=1.0, alpha=0.05)

    def test_glm_contrast_power_inconsistent_delta_raises(self):
        from lattice_doe.power import glm_contrast_power
        from lattice_doe.config import PowerGLMContrastConfig
        X, _, L2 = self._fixture()
        cfg = PowerGLMContrastConfig(L=L2, delta=np.array([0.9, -0.9]),
                                     baseline=0.3, family="binomial")
        with pytest.raises(ValueError, match="inconsistent with the linear"):
            glm_contrast_power(cfg, X)

    def test_contrast_power_sp_inconsistent_delta_raises(self):
        from lattice_doe.power import contrast_power_sp
        from lattice_doe.split_plot import build_whole_plot_indicator
        X, _, L2 = self._fixture()
        Z = build_whole_plot_indicator(16, 4, 4)
        with pytest.raises(ValueError, match="inconsistent with the linear"):
            contrast_power_sp(L2, np.array([1.2, -1.2]), X, Z,
                              sigma_sp=1.0, eta=1.0, alpha=0.05)

    def test_hotelling_inconsistent_delta_raises(self):
        from lattice_doe.power import hotelling_t2_power
        X, _, L2 = self._fixture()
        Sig = np.array([[1.0, 0.3], [0.3, 2.0]])
        with pytest.raises(ValueError, match="inconsistent with the linear"):
            hotelling_t2_power(L2, np.array([[1.0, 0.5], [-1.0, 0.5]]), X,
                               Sig, alpha=0.05)

    def test_full_rank_L_unaffected(self):
        X, _, _ = self._fixture()
        L = np.array([[0.0, 1.0], [1.0, 0.0]])
        res = contrast_power(L, np.array([1.0, 0.5]), X, sigma=1.0, alpha=0.05)
        assert 0.0 < res.power < 1.0
