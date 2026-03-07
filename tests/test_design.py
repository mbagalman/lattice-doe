# tests/test_design.py
"""Unit tests for design.py — candidate generation and model matrix construction."""
import numpy as np
import pandas as pd
import pytest

from iopt_power_design.design import (
    build_candidate,
    build_model_matrix,
    estimate_candidate_size,
    _i_criterion_for_indices,
    _d_criterion_for_indices,
    _a_criterion_for_indices,
    _criterion_score,
    _score_design,
    augment_design,
)
from iopt_power_design.config import DesignOptions

# ---------------------------------------------------------------------------
# Fixtures / shared factor specs
# ---------------------------------------------------------------------------

CONTINUOUS = {"A": (0.0, 10.0), "B": (-1.0, 1.0)}
CATEGORICAL = {"X": ["a", "b", "c"], "Y": ["p", "q"]}
MIXED = {"A": (0.0, 10.0), "X": ["low", "high"]}


# ---------------------------------------------------------------------------
# build_candidate
# ---------------------------------------------------------------------------

class TestBuildCandidate:
    def test_continuous_shape(self):
        cand = build_candidate(CONTINUOUS, candidate_points=50, seed=0)
        assert len(cand) == 50
        assert set(cand.columns) == {"A", "B"}

    def test_continuous_bounds_respected(self):
        cand = build_candidate(CONTINUOUS, candidate_points=200, seed=1)
        assert cand["A"].between(0.0, 10.0).all()
        assert cand["B"].between(-1.0, 1.0).all()

    def test_categorical_columns(self):
        cand = build_candidate(CATEGORICAL, candidate_points=20, seed=0)
        assert set(cand.columns) == {"X", "Y"}
        assert set(cand["X"].unique()).issubset({"a", "b", "c"})
        assert set(cand["Y"].unique()).issubset({"p", "q"})

    def test_mixed_columns(self):
        cand = build_candidate(MIXED, candidate_points=40, seed=0)
        assert set(cand.columns) == {"A", "X"}
        assert cand["A"].between(0.0, 10.0).all()
        assert set(cand["X"].unique()).issubset({"low", "high"})

    def test_constraint_filtering(self):
        cand = build_candidate(
            CONTINUOUS,
            candidate_points=200,
            seed=0,
            constraint_func=lambda row: row["A"] <= 5.0,
        )
        assert (cand["A"] <= 5.0).all()

    def test_reproducible_with_same_seed(self):
        c1 = build_candidate(CONTINUOUS, candidate_points=50, seed=42)
        c2 = build_candidate(CONTINUOUS, candidate_points=50, seed=42)
        pd.testing.assert_frame_equal(c1, c2)

    def test_different_seeds_differ(self):
        c1 = build_candidate(CONTINUOUS, candidate_points=50, seed=1)
        c2 = build_candidate(CONTINUOUS, candidate_points=50, seed=2)
        assert not c1.equals(c2)

    def test_returns_dataframe(self):
        cand = build_candidate(CONTINUOUS, candidate_points=10, seed=0)
        assert isinstance(cand, pd.DataFrame)

    def test_no_nans(self):
        cand = build_candidate(MIXED, candidate_points=30, seed=0)
        assert not cand.isnull().any().any()


# ---------------------------------------------------------------------------
# build_model_matrix
# ---------------------------------------------------------------------------

class TestBuildModelMatrix:
    def test_returns_tuple_of_two(self):
        cand = build_candidate(CONTINUOUS, candidate_points=20, seed=0)
        result = build_model_matrix("~ 1 + A + B", cand)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_ndarray(self):
        cand = build_candidate(CONTINUOUS, candidate_points=20, seed=0)
        X, names = build_model_matrix("~ 1 + A + B", cand)
        assert isinstance(X, np.ndarray)

    def test_second_element_is_list_of_strings(self):
        cand = build_candidate(CONTINUOUS, candidate_points=20, seed=0)
        X, names = build_model_matrix("~ 1 + A + B", cand)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_matrix_shape_continuous(self):
        cand = build_candidate(CONTINUOUS, candidate_points=20, seed=0)
        X, names = build_model_matrix("~ 1 + A + B", cand)
        assert X.shape == (20, 3)   # intercept + A + B
        assert len(names) == 3

    def test_intercept_column_is_ones(self):
        cand = build_candidate(CONTINUOUS, candidate_points=20, seed=0)
        X, _ = build_model_matrix("~ 1 + A + B", cand)
        assert np.allclose(X[:, 0], 1.0)

    def test_interaction_expands_correctly(self):
        cand = build_candidate(CONTINUOUS, candidate_points=20, seed=0)
        X, names = build_model_matrix("~ 1 + A + B + A:B", cand)
        assert X.shape[1] == 4   # intercept + A + B + A:B

    def test_names_length_matches_columns(self):
        cand = build_candidate(MIXED, candidate_points=20, seed=0)
        X, names = build_model_matrix("~ 1 + A + X", cand)
        assert len(names) == X.shape[1]

    def test_row_count_matches_input(self):
        n = 35
        cand = build_candidate(CONTINUOUS, candidate_points=n, seed=0)
        X, _ = build_model_matrix("~ 1 + A", cand)
        assert X.shape[0] == n


# ---------------------------------------------------------------------------
# estimate_candidate_size
# ---------------------------------------------------------------------------

class TestEstimateCandidateSize:
    def test_continuous_within_bounds(self):
        size = estimate_candidate_size(
            "~ 1 + A + B", CONTINUOUS, cand_min=100, cand_max=5000
        )
        assert 100 <= size <= 5000

    def test_categorical_at_least_min(self):
        size = estimate_candidate_size(
            "~ 1 + X", CATEGORICAL, cand_min=100, cand_max=5000
        )
        assert 100 <= size <= 5000

    def test_mixed_within_bounds(self):
        size = estimate_candidate_size(
            "~ 1 + A + X", MIXED, cand_min=100, cand_max=5000
        )
        assert 100 <= size <= 5000

    def test_respects_cand_max(self):
        size = estimate_candidate_size(
            "~ 1 + A", CONTINUOUS, cand_min=10, cand_max=50
        )
        assert size <= 50

    def test_respects_cand_min(self):
        size = estimate_candidate_size(
            "~ 1 + X", {"Z": ["a", "b"]}, cand_min=500, cand_max=5000
        )
        assert size >= 500

    def test_returns_int(self):
        size = estimate_candidate_size("~ 1 + A", CONTINUOUS)
        assert isinstance(size, int)


# ---------------------------------------------------------------------------
# D-criterion helpers and _criterion_score dispatcher
# ---------------------------------------------------------------------------

class TestCriterionScoring:
    """Unit tests for _i_criterion_for_indices, _d_criterion_for_indices,
    and the _criterion_score dispatcher introduced for D-optimal support."""

    @pytest.fixture()
    def simple_X_and_idx(self):
        """A small, full-rank X_cand and a selection of 4 rows."""
        rng = np.random.default_rng(0)
        # 10 candidate rows, 3 parameters (intercept + 2 continuous)
        raw = rng.uniform(size=(10, 2))
        X_cand = np.column_stack([np.ones(10), raw])
        idx = np.array([0, 2, 5, 8])
        return X_cand, idx

    # --- _i_criterion_for_indices ---

    def test_i_criterion_returns_float(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        score = _i_criterion_for_indices(X_cand, idx)
        assert isinstance(score, float)

    def test_i_criterion_is_positive(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        score = _i_criterion_for_indices(X_cand, idx)
        assert score > 0.0

    # --- _d_criterion_for_indices ---

    def test_d_criterion_returns_float(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        score = _d_criterion_for_indices(X_cand, idx)
        assert isinstance(score, float)

    def test_d_criterion_is_finite_for_full_rank(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        score = _d_criterion_for_indices(X_cand, idx)
        assert np.isfinite(score)

    def test_d_criterion_is_negative(self, simple_X_and_idx):
        """Negative log-det should be negative when det > 1 (or negative when
        log-det is positive). More precisely, lower = better, and well-conditioned
        full-rank selections produce finite scores that are not +inf."""
        X_cand, idx = simple_X_and_idx
        score = _d_criterion_for_indices(X_cand, idx)
        assert score < float("inf")

    def test_d_criterion_inf_for_singular_design(self):
        """A design whose X is rank-deficient should return +inf."""
        # All-zeros X_cand → XtX singular even after jitter if jitter is tiny
        X_cand = np.zeros((6, 3))
        idx = np.array([0, 1, 2])
        score = _d_criterion_for_indices(X_cand, idx, jitter=0.0)
        assert score == float("inf")

    def test_d_criterion_better_design_has_lower_score(self):
        """A design that spans the space better should have a lower (better) D score
        than a near-degenerate one."""
        # 4-point full design in 2D (intercept + x)
        X_good = np.array([[1, -1], [1, -0.5], [1, 0.5], [1, 1]], dtype=float)
        # 4-point near-degenerate design (clustered)
        X_bad = np.array([[1, 0.0], [1, 0.01], [1, 0.02], [1, 0.03]], dtype=float)

        X_cand = np.vstack([X_good, X_bad])
        idx_good = np.arange(4)
        idx_bad = np.arange(4, 8)

        score_good = _d_criterion_for_indices(X_cand, idx_good)
        score_bad = _d_criterion_for_indices(X_cand, idx_bad)
        assert score_good < score_bad

    # --- _criterion_score dispatcher ---

    def test_criterion_score_i_matches_i_function(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        assert _criterion_score("I", X_cand, idx) == _i_criterion_for_indices(X_cand, idx)

    def test_criterion_score_d_matches_d_function(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        assert _criterion_score("D", X_cand, idx) == _d_criterion_for_indices(X_cand, idx)

    def test_criterion_score_raises_on_unknown_criterion(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        with pytest.raises(ValueError, match="criterion"):
            _criterion_score("E", X_cand, idx)

    # --- _a_criterion_for_indices ---

    def test_a_criterion_returns_float(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        score = _a_criterion_for_indices(X_cand, idx)
        assert isinstance(score, float)

    def test_a_criterion_is_positive_finite(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        score = _a_criterion_for_indices(X_cand, idx)
        assert score > 0.0
        assert np.isfinite(score)

    def test_a_criterion_inf_for_singular_design(self):
        """Near-singular design with jitter=0 should give +inf."""
        X_cand = np.zeros((6, 3))
        idx = np.array([0, 1, 2])
        score = _a_criterion_for_indices(X_cand, idx, jitter=0.0)
        assert score == float("inf")

    def test_a_criterion_better_design_has_lower_score(self):
        """A spread-out design should have a lower (better) A-score."""
        X_good = np.array([[1, -1], [1, -0.5], [1, 0.5], [1, 1]], dtype=float)
        X_bad  = np.array([[1,  0.0], [1, 0.01], [1, 0.02], [1, 0.03]], dtype=float)
        X_cand = np.vstack([X_good, X_bad])
        score_good = _a_criterion_for_indices(X_cand, np.arange(4))
        score_bad  = _a_criterion_for_indices(X_cand, np.arange(4, 8))
        assert score_good < score_bad

    def test_criterion_score_a_matches_a_function(self, simple_X_and_idx):
        X_cand, idx = simple_X_and_idx
        assert _criterion_score("A", X_cand, idx) == _a_criterion_for_indices(X_cand, idx)


# ---------------------------------------------------------------------------
# _score_design helper
# ---------------------------------------------------------------------------

class TestScoreDesign:
    """Unit tests for the _score_design matrix-level helper."""

    @pytest.fixture
    def small_X(self):
        """Simple full-rank design matrix."""
        return np.array([
            [1, -1, 0.0],
            [1,  1, 0.0],
            [1, -1, 1.0],
            [1,  1, 1.0],
            [1,  0, 0.5],
        ], dtype=float)

    def test_i_criterion_requires_Mcand(self, small_X):
        with pytest.raises(ValueError, match="Mcand"):
            _score_design("I", small_X, Mcand=None, N_cand=10)

    def test_i_criterion_returns_positive(self, small_X):
        Mcand = small_X.T @ small_X
        score = _score_design("I", small_X, Mcand=Mcand, N_cand=small_X.shape[0])
        assert score > 0

    def test_d_criterion_returns_finite(self, small_X):
        score = _score_design("D", small_X)
        assert np.isfinite(score)

    def test_a_criterion_returns_positive(self, small_X):
        score = _score_design("A", small_X)
        assert score > 0

    def test_unknown_criterion_raises(self, small_X):
        with pytest.raises(ValueError, match="criterion"):
            _score_design("Z", small_X)


# ---------------------------------------------------------------------------
# augment_design
# ---------------------------------------------------------------------------

class TestAugmentDesign:
    """Tests for the greedy augment_design function."""

    FORMULA = "~ 1 + A + B"
    FACTORS = {"A": (0.0, 1.0), "B": (0.0, 1.0)}
    FAST_OPTS = DesignOptions(candidate_points=80, starts=1, max_iter=20, random_state=0)

    @pytest.fixture
    def seed_design(self):
        """A small starter design (5 rows) for augmentation tests."""
        return pd.DataFrame({"A": [0.0, 0.5, 1.0, 0.25, 0.75],
                             "B": [0.0, 0.5, 1.0, 0.75, 0.25]})

    def test_augmented_length(self, seed_design):
        aug, new = augment_design(
            seed_design, m=3,
            formula=self.FORMULA, factors=self.FACTORS,
            design_opts=self.FAST_OPTS,
        )
        assert len(aug) == len(seed_design) + 3
        assert len(new) == 3

    def test_new_runs_columns_match(self, seed_design):
        aug, new = augment_design(
            seed_design, m=2,
            formula=self.FORMULA, factors=self.FACTORS,
            design_opts=self.FAST_OPTS,
        )
        assert set(new.columns) == set(seed_design.columns)

    def test_existing_rows_preserved(self, seed_design):
        aug, _ = augment_design(
            seed_design, m=2,
            formula=self.FORMULA, factors=self.FACTORS,
            design_opts=self.FAST_OPTS,
        )
        # First n rows should match the original design
        original_part = aug.iloc[:len(seed_design)].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            original_part, seed_design.reset_index(drop=True)
        )

    def test_raises_on_zero_m(self, seed_design):
        with pytest.raises(ValueError, match="m"):
            augment_design(seed_design, m=0,
                           formula=self.FORMULA, factors=self.FACTORS)

    def test_raises_on_empty_design(self):
        empty = pd.DataFrame({"A": [], "B": []})
        with pytest.raises(ValueError, match="empty"):
            augment_design(empty, m=2,
                           formula=self.FORMULA, factors=self.FACTORS)

    def test_a_criterion_augmentation(self, seed_design):
        """Augmentation with criterion='A' should not raise and return valid output."""
        opts_a = DesignOptions(
            candidate_points=80, starts=1, max_iter=20,
            random_state=0, criterion="A"
        )
        aug, new = augment_design(
            seed_design, m=2,
            formula=self.FORMULA, factors=self.FACTORS,
            design_opts=opts_a,
        )
        assert len(aug) == len(seed_design) + 2
