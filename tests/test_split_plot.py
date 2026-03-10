# tests/test_split_plot.py
"""Unit tests for iopt_power_design.split_plot (SP-2 covariance utilities)."""
from __future__ import annotations

import numpy as np
import pytest

from iopt_power_design.split_plot import (
    build_whole_plot_indicator,
    build_split_plot_covariance_inv,
    gls_information_matrix,
    classify_contrasts,
    split_plot_df_denom,
)


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_balanced(n_wp: int, s: int):
    """Return (Z, n_total, V, V_inv) for a balanced layout with eta=1."""
    n_total = n_wp * s
    Z = build_whole_plot_indicator(n_total, n_wp, s)
    V_inv = build_split_plot_covariance_inv(Z, eta=1.0)
    V = np.eye(n_total) + np.ones((n_total, n_total)) * 0.0  # placeholder
    # Build true V = I + eta * Z Z'
    V = np.eye(n_total) + Z @ Z.T
    return Z, n_total, V, V_inv


# ===========================================================================
# TestBuildWholePlotIndicator
# ===========================================================================

class TestBuildWholePlotIndicator:

    def test_shape(self):
        Z = build_whole_plot_indicator(12, 3, 4)
        assert Z.shape == (12, 3)

    def test_dtype(self):
        Z = build_whole_plot_indicator(6, 2, 3)
        assert Z.dtype == np.float64

    def test_entries_are_zero_or_one(self):
        Z = build_whole_plot_indicator(8, 4, 2)
        assert set(np.unique(Z)) == {0.0, 1.0}

    def test_each_row_sums_to_one(self):
        """Each observation belongs to exactly one WP."""
        Z = build_whole_plot_indicator(12, 4, 3)
        np.testing.assert_array_equal(Z.sum(axis=1), np.ones(12))

    def test_each_column_sums_to_subplots_per_wp(self):
        """Each WP column has exactly subplots_per_wp ones."""
        n_wp, s = 5, 3
        Z = build_whole_plot_indicator(n_wp * s, n_wp, s)
        np.testing.assert_array_equal(Z.sum(axis=0), np.full(n_wp, s))

    def test_correct_block_structure(self):
        """Rows 0..s-1 → WP 0, rows s..2s-1 → WP 1, etc."""
        n_wp, s = 3, 2
        Z = build_whole_plot_indicator(6, n_wp, s)
        for k in range(n_wp):
            for j in range(s):
                row = k * s + j
                assert Z[row, k] == 1.0
                for other_k in range(n_wp):
                    if other_k != k:
                        assert Z[row, other_k] == 0.0

    def test_ztZ_is_scalar_multiple_of_identity_for_balanced(self):
        """For balanced layout, Z'Z = s * I_{n_wp}."""
        n_wp, s = 4, 3
        Z = build_whole_plot_indicator(n_wp * s, n_wp, s)
        ZtZ = Z.T @ Z
        expected = s * np.eye(n_wp)
        np.testing.assert_allclose(ZtZ, expected)

    def test_mismatched_n_total_raises(self):
        with pytest.raises(ValueError, match="n_total"):
            build_whole_plot_indicator(7, 2, 3)  # 7 != 2*3

    def test_single_subplot_per_wp(self):
        """Edge case: one sub-plot per WP reduces to independent runs."""
        n_wp = 5
        Z = build_whole_plot_indicator(5, n_wp, 1)
        np.testing.assert_array_equal(Z, np.eye(n_wp))


# ===========================================================================
# TestBuildSplitPlotCovarianceInv
# ===========================================================================

class TestBuildSplitPlotCovarianceInv:

    def test_eta_zero_returns_identity(self):
        Z = build_whole_plot_indicator(6, 2, 3)
        V_inv = build_split_plot_covariance_inv(Z, eta=0.0)
        np.testing.assert_allclose(V_inv, np.eye(6))

    def test_shape(self):
        Z = build_whole_plot_indicator(12, 3, 4)
        V_inv = build_split_plot_covariance_inv(Z, eta=2.0)
        assert V_inv.shape == (12, 12)

    def test_dtype(self):
        Z = build_whole_plot_indicator(6, 2, 3)
        V_inv = build_split_plot_covariance_inv(Z, eta=1.0)
        assert V_inv.dtype == np.float64

    def test_V_times_V_inv_is_identity(self):
        """V · V⁻¹ ≈ I for various eta values."""
        n_wp, s = 3, 4
        Z = build_whole_plot_indicator(n_wp * s, n_wp, s)
        n = n_wp * s
        for eta in (0.5, 1.0, 2.0, 5.0, 10.0):
            V = np.eye(n) + eta * (Z @ Z.T)
            V_inv = build_split_plot_covariance_inv(Z, eta=eta)
            product = V @ V_inv
            np.testing.assert_allclose(product, np.eye(n), atol=1e-10,
                                       err_msg=f"V @ V_inv ≠ I at eta={eta}")

    def test_symmetric(self):
        Z = build_whole_plot_indicator(8, 2, 4)
        for eta in (0.1, 1.0, 3.0):
            V_inv = build_split_plot_covariance_inv(Z, eta=eta)
            np.testing.assert_allclose(V_inv, V_inv.T, atol=1e-14,
                                       err_msg=f"V_inv not symmetric at eta={eta}")

    def test_balanced_closed_form_diagonal(self):
        """V⁻¹[i,i] = 1 − η/(1 + η·s) for balanced layout."""
        n_wp, s = 2, 3
        eta = 1.0
        Z = build_whole_plot_indicator(n_wp * s, n_wp, s)
        V_inv = build_split_plot_covariance_inv(Z, eta=eta)
        expected_diag = 1.0 - eta / (1.0 + eta * s)
        np.testing.assert_allclose(np.diag(V_inv), expected_diag, atol=1e-12)

    def test_balanced_closed_form_same_wp_off_diagonal(self):
        """V⁻¹[i,j] = −η/(1 + η·s) for i≠j in the same WP."""
        n_wp, s = 2, 3
        eta = 1.0
        Z = build_whole_plot_indicator(n_wp * s, n_wp, s)
        V_inv = build_split_plot_covariance_inv(Z, eta=eta)
        expected_off = -eta / (1.0 + eta * s)
        # Check all same-WP off-diagonal entries (rows 0,1,2 are WP 0)
        for i in range(s):
            for j in range(s):
                if i != j:
                    np.testing.assert_allclose(
                        V_inv[i, j], expected_off, atol=1e-12,
                        err_msg=f"Off-diagonal same-WP mismatch at ({i},{j})"
                    )

    def test_different_wp_entries_are_zero(self):
        """V⁻¹[i,j] = 0 for observations in different WPs."""
        n_wp, s = 3, 2
        Z = build_whole_plot_indicator(n_wp * s, n_wp, s)
        V_inv = build_split_plot_covariance_inv(Z, eta=2.0)
        # WP 0: rows 0,1   WP 1: rows 2,3   WP 2: rows 4,5
        cross_pairs = [(0, 2), (0, 3), (1, 4), (2, 5)]
        for i, j in cross_pairs:
            np.testing.assert_allclose(
                V_inv[i, j], 0.0, atol=1e-14,
                err_msg=f"Cross-WP entry ({i},{j}) should be 0"
            )

    def test_negative_eta_raises(self):
        Z = build_whole_plot_indicator(6, 2, 3)
        with pytest.raises(ValueError, match="eta"):
            build_split_plot_covariance_inv(Z, eta=-0.5)

    def test_large_eta_row_sums_approach_zero(self):
        """For very large η, V⁻¹ row sums approach 0 (WP means absorbed)."""
        n_wp, s = 2, 4
        Z = build_whole_plot_indicator(n_wp * s, n_wp, s)
        V_inv = build_split_plot_covariance_inv(Z, eta=1e6)
        # Each block row sum = 1 - η/(1+ηs) + (s-1)*(-η/(1+ηs)) = 1 - sη/(1+ηs) → 0
        row_sums = V_inv.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-4)

    def test_known_2wp_3sp_example(self):
        """Manually verify a 2-WP, 3-SP, η=1 example against explicit inverse."""
        # V (first 3×3 block) = [[2,1,1],[1,2,1],[1,1,2]]
        # V_inv (first block) = 1/4 * [[3,-1,-1],[-1,3,-1],[-1,-1,3]]
        n_wp, s, eta = 2, 3, 1.0
        Z = build_whole_plot_indicator(n_wp * s, n_wp, s)
        V_inv = build_split_plot_covariance_inv(Z, eta=eta)
        expected_block = np.array([[3, -1, -1], [-1, 3, -1], [-1, -1, 3]]) / 4.0
        np.testing.assert_allclose(V_inv[:3, :3], expected_block, atol=1e-12)

    def test_importable_from_top_level(self):
        import iopt_power_design
        assert hasattr(iopt_power_design, "build_split_plot_covariance_inv")


# ===========================================================================
# TestGlsInformationMatrix
# ===========================================================================

class TestGlsInformationMatrix:

    def _simple_X(self):
        """Simple 6×3 model matrix for testing."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((6, 3))
        X[:, 0] = 1.0  # intercept
        return X

    def test_shape(self):
        X = self._simple_X()
        M = gls_information_matrix(X, np.eye(6))
        assert M.shape == (3, 3)

    def test_identity_V_inv_matches_XtX_plus_jitter(self):
        """With V_inv = I, M = X'X + jitter·I."""
        X = self._simple_X()
        jitter = 1e-8
        M = gls_information_matrix(X, np.eye(6), jitter=jitter)
        expected = X.T @ X + jitter * np.eye(3)
        np.testing.assert_allclose(M, expected, atol=1e-12)

    def test_symmetric(self):
        X = self._simple_X()
        n_wp, s = 2, 3
        Z = build_whole_plot_indicator(6, n_wp, s)
        V_inv = build_split_plot_covariance_inv(Z, eta=1.5)
        M = gls_information_matrix(X, V_inv)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_positive_definite(self):
        """M must be positive definite (all eigenvalues > 0)."""
        rng = np.random.default_rng(7)
        n_wp, s = 3, 4
        n = n_wp * s
        X = rng.standard_normal((n, 4))
        X[:, 0] = 1.0
        Z = build_whole_plot_indicator(n, n_wp, s)
        V_inv = build_split_plot_covariance_inv(Z, eta=2.0)
        M = gls_information_matrix(X, V_inv, jitter=1e-6)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"Non-positive eigenvalue: {eigvals.min()}"

    def test_jitter_adds_to_diagonal(self):
        """Increasing jitter increases diagonal entries by exactly that amount."""
        X = self._simple_X()
        M1 = gls_information_matrix(X, np.eye(6), jitter=1e-8)
        M2 = gls_information_matrix(X, np.eye(6), jitter=1e-4)
        diff = np.diag(M2) - np.diag(M1)
        np.testing.assert_allclose(diff, np.full(3, 1e-4 - 1e-8), atol=1e-12)

    def test_gls_differs_from_ols_when_eta_nonzero(self):
        """GLS information matrix is not equal to OLS (X'X) when η > 0."""
        rng = np.random.default_rng(99)
        n_wp, s = 3, 4
        n = n_wp * s
        X = rng.standard_normal((n, 3))
        Z = build_whole_plot_indicator(n, n_wp, s)
        V_inv = build_split_plot_covariance_inv(Z, eta=2.0)
        M_gls = gls_information_matrix(X, V_inv, jitter=0)
        M_ols = gls_information_matrix(X, np.eye(n), jitter=0)
        assert not np.allclose(M_gls, M_ols)

    def test_importable_from_top_level(self):
        import iopt_power_design
        assert hasattr(iopt_power_design, "gls_information_matrix")


# ===========================================================================
# TestClassifyContrasts
# ===========================================================================

class TestClassifyContrasts:
    """
    Model: intercept + A (HTC) + B (ETC) + C (ETC) → 4 columns (p=4)
    htc_factor_cols = [0, 1]  (intercept + A)
    """

    HTC_COLS = [0, 1]
    P = 4

    def test_all_htc_contrast_is_wp(self):
        L = np.array([[1.0, 1.0, 0.0, 0.0]])  # only cols 0,1
        is_wp = classify_contrasts(L, self.HTC_COLS, self.P)
        assert is_wp[0] is np.bool_(True)

    def test_intercept_only_contrast_is_wp(self):
        L = np.array([[1.0, 0.0, 0.0, 0.0]])  # only col 0
        is_wp = classify_contrasts(L, self.HTC_COLS, self.P)
        assert is_wp[0] is np.bool_(True)

    def test_htc_only_no_intercept_is_wp(self):
        L = np.array([[0.0, 1.0, 0.0, 0.0]])  # col 1 (A) only
        is_wp = classify_contrasts(L, self.HTC_COLS, self.P)
        assert is_wp[0] is np.bool_(True)

    def test_etc_contrast_is_sp(self):
        L = np.array([[0.0, 0.0, 1.0, 0.0]])  # col 2 (B)
        is_wp = classify_contrasts(L, self.HTC_COLS, self.P)
        assert is_wp[0] is np.bool_(False)

    def test_mixed_htc_etc_is_sp(self):
        L = np.array([[0.0, 1.0, 1.0, 0.0]])  # cols 1 (A, HTC) + 2 (B, ETC)
        is_wp = classify_contrasts(L, self.HTC_COLS, self.P)
        assert is_wp[0] is np.bool_(False)

    def test_multiple_rows(self):
        L = np.array([
            [1.0, 0.0, 0.0, 0.0],  # WP (intercept)
            [0.0, 1.0, 0.0, 0.0],  # WP (A)
            [0.0, 0.0, 1.0, 0.0],  # SP (B)
            [0.0, 0.0, 0.0, 1.0],  # SP (C)
            [0.0, 1.0, 1.0, 0.0],  # SP (mixed)
        ])
        is_wp = classify_contrasts(L, self.HTC_COLS, self.P)
        expected = np.array([True, True, False, False, False])
        np.testing.assert_array_equal(is_wp, expected)

    def test_all_zeros_row_is_wp(self):
        """All-zero row has no nonzero entries → subset of any set → WP."""
        L = np.array([[0.0, 0.0, 0.0, 0.0]])
        is_wp = classify_contrasts(L, self.HTC_COLS, self.P)
        assert is_wp[0] is np.bool_(True)

    def test_empty_htc_cols_all_are_sp(self):
        """When htc_factor_cols is empty, any non-zero contrast is SP."""
        L = np.array([[1.0, 0.0, 0.0, 0.0]])
        is_wp = classify_contrasts(L, [], self.P)
        # non-zero entry not in empty set → SP
        assert is_wp[0] is np.bool_(False)

    def test_1d_L_treated_as_single_row(self):
        L = np.array([1.0, 0.0, 0.0, 0.0])  # 1D
        is_wp = classify_contrasts(L, self.HTC_COLS, self.P)
        assert is_wp.shape == (1,)
        assert is_wp[0] is np.bool_(True)

    def test_output_dtype_is_bool(self):
        L = np.array([[0.0, 0.0, 1.0, 0.0]])
        is_wp = classify_contrasts(L, self.HTC_COLS, self.P)
        assert is_wp.dtype == bool


# ===========================================================================
# TestSplitPlotDfDenom
# ===========================================================================

class TestSplitPlotDfDenom:
    """
    Fixture: 2 WPs, 4 SP each → n_total=8, n_wp=2.
    Model: intercept + A (HTC) + B (ETC) → p=3.
    True df_wp = n_wp - rank(X_wp) = 2 - 2 = 0... need bigger example.
    Use: 4 WPs, 3 SP each → n_total=12, n_wp=4.
    Model: intercept + A + B → p=3, A=HTC (col 1), B=ETC (col 2).
    """

    def _make_fixture(self):
        n_wp, s = 4, 3
        n_total = n_wp * s
        rng = np.random.default_rng(11)
        # WP factor A varies by whole plot (4 distinct values)
        A_vals = rng.standard_normal(n_wp)
        A = np.repeat(A_vals, s)
        # SP factor B varies freely
        B = rng.standard_normal(n_total)
        X = np.column_stack([np.ones(n_total), A, B])
        Z = build_whole_plot_indicator(n_total, n_wp, s)
        htc_cols = [0, 1]  # intercept + A
        return X, Z, htc_cols, n_wp, n_total

    def test_conservative_all_wp_df(self):
        X, Z, htc_cols, n_wp, n_total = self._make_fixture()
        # Contrast rows: one WP, one SP
        is_wp = np.array([True, False])
        df = split_plot_df_denom(X, Z, is_wp, "conservative", htc_cols)
        # All df should be df_wp
        assert df[0] == df[1]

    def test_sp_only_all_sp_df(self):
        X, Z, htc_cols, n_wp, n_total = self._make_fixture()
        is_wp = np.array([True, False])
        df = split_plot_df_denom(X, Z, is_wp, "sp_only", htc_cols)
        assert df[0] == df[1]

    def test_auto_wp_contrast_gets_wp_df(self):
        X, Z, htc_cols, n_wp, n_total = self._make_fixture()
        is_wp_wp = np.array([True])
        is_wp_sp = np.array([False])
        df_wp_row = split_plot_df_denom(X, Z, is_wp_wp, "auto", htc_cols)
        df_sp_row = split_plot_df_denom(X, Z, is_wp_sp, "auto", htc_cols)
        # WP contrast → df_wp; SP contrast → df_sp; df_wp <= df_sp typically
        assert df_wp_row[0] <= df_sp_row[0]

    def test_auto_assigns_different_df_for_wp_vs_sp(self):
        X, Z, htc_cols, n_wp, n_total = self._make_fixture()
        is_wp = np.array([True, False])
        df = split_plot_df_denom(X, Z, is_wp, "auto", htc_cols)
        # df[0] (WP) and df[1] (SP) should differ
        assert df[0] != df[1], "auto mode should assign different df to WP vs SP contrasts"

    def test_conservative_returns_same_as_conservative_for_wp(self):
        """Conservative df for a WP contrast = same as auto WP df."""
        X, Z, htc_cols, n_wp, n_total = self._make_fixture()
        is_wp = np.array([True])
        df_auto = split_plot_df_denom(X, Z, is_wp, "auto", htc_cols)
        df_cons = split_plot_df_denom(X, Z, is_wp, "conservative", htc_cols)
        np.testing.assert_array_equal(df_auto, df_cons)

    def test_df_values_are_positive(self):
        X, Z, htc_cols, n_wp, n_total = self._make_fixture()
        is_wp = np.array([True, False, True, False])
        for method in ("auto", "conservative", "sp_only"):
            df = split_plot_df_denom(X, Z, is_wp, method, htc_cols)
            assert np.all(df >= 1), f"Non-positive df found for method={method!r}"

    def test_output_dtype_is_int(self):
        X, Z, htc_cols, n_wp, n_total = self._make_fixture()
        is_wp = np.array([True, False])
        df = split_plot_df_denom(X, Z, is_wp, "auto", htc_cols)
        assert df.dtype == int or np.issubdtype(df.dtype, np.integer)

    def test_output_length_matches_is_wp(self):
        X, Z, htc_cols, n_wp, n_total = self._make_fixture()
        is_wp = np.array([True, False, True])
        df = split_plot_df_denom(X, Z, is_wp, "auto", htc_cols)
        assert len(df) == 3

    def test_without_htc_cols_still_returns_valid_df(self):
        """Fallback (htc_factor_cols=None) returns positive df."""
        X, Z, _, n_wp, n_total = self._make_fixture()
        is_wp = np.array([True, False])
        df = split_plot_df_denom(X, Z, is_wp, "auto", htc_factor_cols=None)
        assert np.all(df >= 1)

    def test_invalid_df_method_raises(self):
        X, Z, htc_cols, _, _ = self._make_fixture()
        with pytest.raises(ValueError, match="df_method"):
            split_plot_df_denom(X, Z, np.array([True]), "kenward_roger", htc_cols)
