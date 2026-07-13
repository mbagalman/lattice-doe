# tests/test_design.py
"""Unit tests for design.py — candidate generation and model matrix construction."""
import numpy as np
import pandas as pd
import pytest

from lattice_doe.candidate import (
    build_candidate,
    estimate_candidate_size,
)
from lattice_doe.model_matrix import build_model_matrix
from lattice_doe.iopt_search import (
    build_i_opt_design,
    build_i_opt_design_with_idx,
    _i_criterion_for_indices,
    _d_criterion_for_indices,
    _a_criterion_for_indices,
    _criterion_score,
    _one_start_worker,
    _optimal_indices_from_X,
    _score_design,
    augment_design,
)
from lattice_doe.config import DesignOptions

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


class TestBuildIOptDesignGuards:
    """Regression tests for n > n_cand behavior (Issue #1)."""

    def test_build_i_opt_design_with_idx_raises_when_n_exceeds_candidates(self):
        cand = pd.DataFrame({"A": [0.1, 0.2, 0.3]})
        with pytest.raises(ValueError, match="exceeds the candidate set size"):
            build_i_opt_design_with_idx(
                cand=cand,
                formula="~ 1 + A",
                n=5,
                n_start=1,
                max_iter=10,
                random_state=0,
            )

    def test_build_i_opt_design_raises_when_n_exceeds_candidates(self):
        cand = pd.DataFrame({"A": [0.1, 0.2, 0.3]})
        with pytest.raises(ValueError, match="exceeds the candidate set size"):
            build_i_opt_design(
                cand=cand,
                formula="~ 1 + A",
                n=5,
                n_start=1,
                max_iter=10,
                random_state=0,
            )


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


# ---------------------------------------------------------------------------
# CR-19: jitter forwarded to Fedorov exchange in parallel worker (_one_start_worker)
# ---------------------------------------------------------------------------

class TestCR19JitterInParallelWorker:
    """Verify _one_start_worker forwards jitter to the Fedorov exchange."""

    FORMULA = "1 + A + B"
    FACTORS = {"A": (0.0, 1.0), "B": (0.0, 1.0)}

    def _make_X_cand(self, seed=0):
        from lattice_doe.candidate import build_candidate
        cand = build_candidate(self.FACTORS, candidate_points=50, seed=seed)
        X, _ = build_model_matrix(self.FORMULA, cand)
        return X

    def test_one_start_worker_accepts_custom_jitter(self):
        """_one_start_worker runs without error for a non-default jitter value."""
        X_cand = self._make_X_cand()
        score, idx = _one_start_worker(
            X_cand, n=8, seed=42,
            algo="fedorov", criterion="I", max_iter=20,
            jitter=1e-4,  # non-default
        )
        assert np.isfinite(score)
        assert len(idx) == 8

    def test_one_start_worker_default_jitter_matches_explicit(self):
        """Default jitter=1e-8 and explicit jitter=1e-8 yield the same result."""
        X_cand = self._make_X_cand(seed=1)
        score_default, idx_default = _one_start_worker(
            X_cand, n=8, seed=7,
            algo="fedorov", criterion="I", max_iter=20,
        )
        score_explicit, idx_explicit = _one_start_worker(
            X_cand, n=8, seed=7,
            algo="fedorov", criterion="I", max_iter=20,
            jitter=1e-8,
        )
        assert score_default == pytest.approx(score_explicit)
        np.testing.assert_array_equal(idx_default, idx_explicit)

    def test_build_i_opt_design_with_idx_parallel_custom_jitter(self):
        """build_i_opt_design_with_idx with workers=2 and custom jitter runs cleanly."""
        from lattice_doe.candidate import build_candidate
        cand = build_candidate(self.FACTORS, candidate_points=80, seed=3)
        design_df, sel_idx, _ = build_i_opt_design_with_idx(
            cand=cand, formula=self.FORMULA, n=10,
            criterion="I", n_start=4, algo="fedorov", max_iter=30,
            random_state=0, workers=2, jitter=1e-4,
        )
        assert len(design_df) == 10
        assert len(sel_idx) == 10


# ---------------------------------------------------------------------------
# SR-12: compound-D exchange gain must be exact (log det-ratio, not ratio-1)
# ---------------------------------------------------------------------------

class TestSR12CompoundDGain:
    """SR-12 regression: the compound-D exchange summed w_k*(det_ratio_k - 1)
    while minimizing sum(w_k * -logdet_k). The surrogate can disagree in sign
    with the true compound delta (2,647 of 25,000 scanned swaps in the audit),
    so the exchange accepted score-worsening swaps and oscillated between two
    states forever. The gain is now sum(w_k * log(det_ratio_k)) -- exact."""

    def _fixture(self):
        rng = np.random.default_rng(0)
        n_cand = 40
        base = rng.uniform(-1, 1, (n_cand, 3))
        X1 = np.column_stack([np.ones(n_cand), base])                       # p=4
        X2 = np.column_stack([np.ones(n_cand), base[:, :2], base[:, :2]**2])
        return [X1, X2], [0.6, 0.4], rng

    @staticmethod
    def _compound_d_score(cands, weights, idx, jitter=1e-8):
        s = 0.0
        for Xc, w in zip(cands, weights):
            M = Xc[idx].T @ Xc[idx] + jitter * np.eye(Xc.shape[1])
            sign, ld = np.linalg.slogdet(M)
            s += w * (-ld if sign > 0 else np.inf)
        return s / sum(weights)

    def test_gains_match_true_compound_deltas(self):
        """For every possible swap of a random design, the update-formula gain
        must equal the brute-force compound-score improvement (and therefore
        agree in sign)."""
        cands, weights, rng = self._fixture()
        n = 10
        jitter = 1e-8
        w_norm = [w / sum(weights) for w in weights]
        idx = rng.choice(cands[0].shape[0], size=n, replace=False)
        non = np.setdiff1d(np.arange(cands[0].shape[0]), idx)
        base_score = self._compound_d_score(cands, weights, idx)
        for s_pos, s in enumerate(idx):
            for t in non:
                idx2 = idx.copy()
                idx2[s_pos] = t
                true_delta = base_score - self._compound_d_score(cands, weights, idx2)
                gain = 0.0
                for Xc, w in zip(cands, w_norm):
                    M = Xc[idx].T @ Xc[idx] + jitter * np.eye(Xc.shape[1])
                    Mi = np.linalg.inv(M)
                    xs, xt = Xc[s], Xc[t]
                    denom = 1.0 - xs @ Mi @ xs
                    wst = xt @ Mi @ xs
                    vt = xt @ Mi @ xt + wst**2 / denom
                    gain += w * np.log(max(denom * (1.0 + vt), 1e-300))
                assert gain == pytest.approx(true_delta, abs=1e-10)

    def test_convergence_is_max_iter_parity_independent(self):
        """Before the fix the exchange oscillated between two states and the
        result depended on max_iter parity."""
        from lattice_doe.iopt_search import _compound_fedorov_single
        cands, weights, _ = self._fixture()
        i_a = _compound_fedorov_single(cands, weights, 10, criterion="D",
                                       max_iter=200, seed=7)
        i_b = _compound_fedorov_single(cands, weights, 10, criterion="D",
                                       max_iter=201, seed=7)
        assert sorted(i_a.tolist()) == sorted(i_b.tolist())

    def test_compound_d_beats_random_designs(self):
        from lattice_doe.iopt_search import build_compound_design
        cands, weights, rng = self._fixture()
        idx = build_compound_design(cands, weights, 10, criterion="D",
                                    n_start=3, random_state=1)
        score = self._compound_d_score(cands, weights, np.asarray(idx))
        rand_best = min(
            self._compound_d_score(
                cands, weights,
                rng.choice(cands[0].shape[0], size=10, replace=False),
            )
            for _ in range(100)
        )
        assert score <= rand_best + 1e-9

    def test_compound_i_regression(self):
        """Compound-I gains were already exact; the path must keep working."""
        from lattice_doe.iopt_search import build_compound_design
        cands, weights, _ = self._fixture()
        idx = build_compound_design(cands, weights, 10, criterion="I",
                                    n_start=3, random_state=1)
        assert np.asarray(idx).shape == (10,)


# ---------------------------------------------------------------------------
# SR-6: pre-allocation counts are replication counts, never a silent shrink
# ---------------------------------------------------------------------------

class TestSR6PreallocationReplication:
    """SR-6 regression: with preallocate_categorical=True, a cell whose
    allocation exceeded its distinct candidate rows was silently clamped --
    a pure-categorical 2x2 request for n=12 returned a 4-row design. Surplus
    allocation is now fulfilled by replicating the cell's selected rows
    (exact optimal designs replicate design points)."""

    def _cat_cand(self):
        from lattice_doe.candidate import build_candidate
        return build_candidate({"A": ["a1", "a2"], "B": ["b1", "b2"]}, 100, seed=0)

    def test_pure_categorical_returns_requested_n(self):
        from lattice_doe.iopt_search import build_i_opt_design_with_idx
        cand = self._cat_cand()
        df, idx, _ = build_i_opt_design_with_idx(
            cand, "~ A + B", n=12, preallocate_categorical=True, random_state=0,
        )
        assert len(df) == 12 and len(idx) == 12
        counts = df.groupby(["A", "B"]).size()
        assert counts.sum() == 12
        # Balanced allocation is I-optimal for a 2x2 main-effects model.
        assert counts.min() >= 3 and counts.max() <= 3

    def test_small_n_falls_back_to_subset_search(self):
        """n below cells*min_per_cell must select a subset of cells instead of
        raising from the allocation feasibility guard (keeps n-search probes
        at small n working)."""
        from lattice_doe.iopt_search import build_i_opt_design_with_idx
        cand = self._cat_cand()
        df, idx, _ = build_i_opt_design_with_idx(
            cand, "~ A + B", n=3, preallocate_categorical=True, random_state=0,
        )
        assert len(df) == 3
        assert len(df.drop_duplicates()) == 3  # distinct cells, no replication

    def test_alloc_max_per_cell_respected_under_replication(self):
        from lattice_doe.iopt_search import build_i_opt_design_with_idx
        cand = self._cat_cand()
        df, _, _ = build_i_opt_design_with_idx(
            cand, "~ A + B", n=12, preallocate_categorical=True,
            alloc_max_per_cell=4, random_state=0,
        )
        assert len(df) == 12
        assert df.groupby(["A", "B"]).size().max() <= 4

    def test_mixed_design_replicates_beyond_cell_pool(self):
        """A mixed factor space whose per-cell candidate pools are smaller
        than the allocation must replicate rather than shrink."""
        from lattice_doe.candidate import build_candidate
        from lattice_doe.iopt_search import build_i_opt_design_with_idx
        cand = build_candidate({"C": ["x", "y", "z"], "t": (0.0, 1.0)}, 9, seed=1)
        assert len(cand) == 9  # 3 rows per cell
        df, _, _ = build_i_opt_design_with_idx(
            cand, "~ C + t", n=15, preallocate_categorical=True, random_state=0,
        )
        assert len(df) == 15
        assert df.groupby("C").size().min() >= 4  # every cell got replicates

    def test_non_prealloc_error_hints_preallocation(self):
        from lattice_doe.iopt_search import build_i_opt_design_with_idx
        cand = self._cat_cand()
        with pytest.raises(ValueError, match="preallocate_categorical=True"):
            build_i_opt_design_with_idx(cand, "~ A + B", n=12, random_state=0)


# ---------------------------------------------------------------------------
# SR-13: candidate construction must never drop categorical cells or levels
# ---------------------------------------------------------------------------

class TestSR13CellCoverage:
    """SR-13 regression: the mixed-design path subsampled the cell list to
    candidate_points // 2 (10 levels with candidate_points=8 kept only 4
    levels) and the final size trim could drop cells again; the
    cap-exceeded branch could miss entire levels. Dropped cells make those
    treatments unreachable in any design and their model columns
    unestimable. Cells are now always retained (budget is a soft target,
    with a warning) and the cap branch repairs level coverage."""

    def test_small_budget_keeps_all_levels_and_warns(self):
        import warnings as _w
        from lattice_doe.candidate import build_candidate
        factors = {"G": [f"g{i}" for i in range(10)], "x": (0.0, 1.0)}
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            cand = build_candidate(factors, candidate_points=8, seed=0)
        assert cand["G"].nunique() == 10
        assert any("fewer than 2 continuous samples" in str(c.message)
                   for c in caught)

    def test_no_permanently_unestimable_columns(self):
        """The audit's downstream failure: dropped levels left all-zero dummy
        columns — unestimable at ANY design size. Every level column must now
        be populated, the rows must be linearly independent (rank = n_rows),
        and one continuous sample per cell must vary across cells (a constant
        midpoint would be collinear with the intercept)."""
        from lattice_doe.candidate import build_candidate
        from lattice_doe.model_matrix import build_model_matrix
        factors = {"G": [f"g{i}" for i in range(10)], "x": (0.0, 1.0)}
        cand = build_candidate(factors, candidate_points=8, seed=0)
        X, names = build_model_matrix("~ G + x", cand)
        dummy_cols = [j for j, nm in enumerate(names) if nm.startswith("G[")]
        assert all(np.abs(X[:, j]).max() > 0 for j in dummy_cols), (
            "a level's dummy column is all-zero — that treatment is "
            "permanently unestimable"
        )
        # No redundant rows: the candidate spans as much as its size allows.
        assert np.linalg.matrix_rank(X) == X.shape[0]
        # The single continuous sample varies across cells.
        assert cand["x"].nunique() > 1

    def test_ample_budget_respected_without_warning(self):
        import warnings as _w
        from lattice_doe.candidate import build_candidate
        factors = {"G": [f"g{i}" for i in range(10)], "x": (0.0, 1.0)}
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            cand = build_candidate(factors, candidate_points=100, seed=0)
        assert cand["G"].nunique() == 10
        assert len(cand) <= 100
        assert not any("fewer than 2" in str(c.message) for c in caught)

    def test_cap_branch_repairs_level_coverage(self):
        from lattice_doe.candidate import build_candidate
        factors = {"F1": ["1a", "1b", "1c"], "F2": ["2a", "2b", "2c"],
                   "F3": ["3a", "3b", "3c"]}
        cand = build_candidate(factors, candidate_points=6, seed=0,
                               cat_cells_cap=5)
        for f in factors:
            assert cand[f].nunique() == 3, f"level(s) of {f} missing"

    def test_pure_continuous_unchanged(self):
        from lattice_doe.candidate import build_candidate
        cand = build_candidate({"x": (0.0, 1.0), "y": (-1.0, 1.0)},
                               candidate_points=50, seed=0)
        assert len(cand) == 50


# ---------------------------------------------------------------------------
# SR-21: condition number convention must match its interpretation thresholds
# ---------------------------------------------------------------------------

class TestSR21ConditionNumberConvention:
    """SR-21 regression: compute_design_metrics returned cond(X'X) = k(X)^2
    while the export/report layers applied Belsley thresholds stated for
    k(X) ("<30 well-conditioned / >1000 ill-conditioned") -- a fine design
    with k(X)=10 was reported as 100 -> 'Moderate'. The metric is now k(X)."""

    def test_condition_number_is_kappa_of_X(self):
        from lattice_doe.diag_metrics import compute_design_metrics
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(20), rng.uniform(-1, 1, 20),
                             rng.uniform(-1, 1, 20)])
        m = compute_design_metrics(X)
        assert m["condition_number"] == pytest.approx(
            float(np.linalg.cond(X)), rel=1e-10
        )
        # And explicitly NOT the squared convention.
        assert m["condition_number"] != pytest.approx(
            float(np.linalg.cond(X.T @ X)), rel=1e-3
        )

    def test_orthonormal_design_reports_kappa_one(self):
        """A perfectly conditioned design must sit deep in the 'Good' band
        of the Belsley thresholds the display layers apply."""
        from lattice_doe.diag_metrics import compute_design_metrics
        n = 16
        X = np.linalg.qr(np.random.default_rng(1).standard_normal((n, 3)))[0]
        m = compute_design_metrics(X)
        assert m["condition_number"] == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# SR-28: numeric-coded categorical factors must reach the preallocation path
# ---------------------------------------------------------------------------

class TestSR28NumericCodedCategoricals:
    """SR-28 regression (review of SR-6): categorical columns were inferred
    from pandas dtype, so numeric-coded categories like {"g": [0, 1, 2]}
    were treated as continuous -- preallocation/replication was bypassed and
    n=6 over 3 cells raised. Factor-type metadata is now threaded from the
    original specification via the cat_cols parameter."""

    def test_numeric_categories_replicate_with_cat_cols(self):
        cand = build_candidate({"g": [0, 1, 2]}, 50, seed=0)
        assert len(cand) == 3  # numeric dtype, 3 distinct cells
        df, idx, _ = build_i_opt_design_with_idx(
            cand, "~ C(g)", n=6, preallocate_categorical=True,
            random_state=0, cat_cols=["g"],
        )
        assert len(df) == 6
        assert df.groupby("g").size().tolist() == [2, 2, 2]

    def test_dtype_inference_fallback_unchanged(self):
        """Without cat_cols, string-typed categories still work as before."""
        cand = build_candidate({"g": ["a", "b", "c"]}, 50, seed=0)
        df, _, _ = build_i_opt_design_with_idx(
            cand, "~ g", n=6, preallocate_categorical=True, random_state=0,
        )
        assert len(df) == 6
