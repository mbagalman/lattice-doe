# tests/test_search_bounds.py
"""Regression tests for TICKET-039: the n-search must never request a design
larger than the candidate set.

Before the fix, the bisection upper bound was ``power_cfg.max_n`` regardless of
the candidate pool, so the very first midpoint could exceed ``n_cand`` and the
Fedorov path raised an opaque ValueError (e.g. default ``max_n=2000`` with a
200-point pool requested n=1002 immediately). The search bound is now capped at
what the candidate set can support; if the capped search then fails to reach
the target power, an actionable warning names both ``max_n`` and
``candidate_points``.

Problems are kept tiny (2 continuous factors, 2 starts) so these run in the
fast suite.
"""
import warnings

import numpy as np
import pytest

from lattice_doe import (
    DesignOptions,
    MultiResponseOptions,
    PowerContrastConfig,
    ResponseSpec,
    find_multiresponse_design,
    find_optimal_design,
)

FACTORS = {"A": (0.0, 1.0), "B": (0.0, 1.0)}
FORMULA = "~ 1 + A + B"


def _cfg(delta: float = 1.0, **kw) -> PowerContrastConfig:
    """Contrast on the B slope; max_n defaults to 2000 unless overridden."""
    return PowerContrastConfig(
        L=np.array([[0.0, 0.0, 1.0]]),
        delta=np.array([delta]),
        sigma=1.0,
        alpha=0.05,
        power=0.90,
        **kw,
    )


class TestSearchCappedByCandidateSet:
    def test_small_pool_with_default_max_n_does_not_crash(self):
        """The original TICKET-039 crash: pool of 200, default max_n=2000.

        The first bisection midpoint used to be n=1002 > n_cand=200 and raised
        ValueError before any design was built. The optimum here is well under
        200 runs, so the search must simply succeed.
        """
        opts = DesignOptions(candidate_points=200, starts=2, random_state=42)
        res = find_optimal_design(FORMULA, FACTORS, _cfg(), opts)
        report = res["report"]
        assert report["n"] <= 200
        assert report["achieved_power"] + 1e-3 >= 0.90

    def test_capped_and_unreachable_warns_with_both_knobs(self):
        """When the cap binds AND the target is unreachable within the pool,
        the warning must name both max_n and the candidate-set remedy."""
        # delta=0.02 needs n in the thousands; the pool allows only 200.
        opts = DesignOptions(candidate_points=200, starts=2, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = find_optimal_design(FORMULA, FACTORS, _cfg(delta=0.02), opts)
        report = res["report"]
        # Search stopped at the pool size, not at max_n.
        assert report["n"] <= 200
        cap_msgs = [m for m in report["warnings"] if "capped" in m]
        assert cap_msgs, f"no cap warning in report warnings: {report['warnings']}"
        assert "max_n" in cap_msgs[0]
        assert "candidate_points" in cap_msgs[0]
        # The same message must also have been issued as a RuntimeWarning.
        assert any(
            issubclass(w.category, RuntimeWarning) and "capped" in str(w.message)
            for w in caught
        )

    def test_pool_smaller_than_model_raises_actionable_error(self):
        """A pool that cannot even hold the smallest estimable design must
        raise immediately with a remediation hint, not an opaque crash."""
        opts = DesignOptions(candidate_points=3, starts=2, random_state=42)
        with pytest.raises(ValueError, match="candidate_points"):
            find_optimal_design(FORMULA, FACTORS, _cfg(), opts)

    def test_constraints_filtering_all_candidates_raises_actionable_error(self):
        """Constraint filtering that empties the pool must surface the
        constraint_expr remedy rather than crash downstream."""
        opts = DesignOptions(
            candidate_points=100,
            starts=2,
            random_state=42,
            constraint_expr="A > 0.9999",  # eliminates essentially everything
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # candidate.py warns about elimination
            with pytest.raises(ValueError, match="constraint_expr|candidate_points"):
                find_optimal_design(FORMULA, FACTORS, _cfg(), opts)

    def test_achievable_target_ignores_cap_silently(self):
        """When the optimum fits comfortably in the pool, no cap warning
        should be emitted even though n_cand < max_n."""
        opts = DesignOptions(candidate_points=200, starts=2, random_state=42)
        res = find_optimal_design(FORMULA, FACTORS, _cfg(), opts)
        assert not any("capped" in m for m in res["report"]["warnings"])


class TestSearchCappedBlocked:
    def test_blocked_small_pool_does_not_crash(self):
        """Blocked designs draw per block, so their ceiling is
        n_blocks * n_cand; the capped search must still run end-to-end."""
        opts = DesignOptions(
            candidate_points=150, starts=2, random_state=42, n_blocks=2
        )
        res = find_optimal_design(FORMULA, FACTORS, _cfg(), opts)
        report = res["report"]
        assert report["n"] <= 2 * 150
        assert report["achieved_power"] + 1e-3 >= 0.90


class TestSearchCappedMultiResponse:
    def _multi_cfg(self, delta: float = 1.0) -> MultiResponseOptions:
        return MultiResponseOptions(
            responses=[
                ResponseSpec(name="y1", power_cfg=_cfg(delta)),
                ResponseSpec(name="y2", power_cfg=_cfg(delta)),
            ],
            power_combination="min",
        )

    def test_multiresponse_small_pool_does_not_crash(self):
        """Shared-formula MR path: same cap applies to the joint search."""
        opts = DesignOptions(candidate_points=200, starts=2, random_state=42)
        res = find_multiresponse_design(FORMULA, FACTORS, self._multi_cfg(), opts)
        assert res["n"] <= 200
        assert res["achieved_power"] + 1e-3 >= 0.90

    def test_multiresponse_capped_and_unreachable_warns(self):
        opts = DesignOptions(candidate_points=150, starts=2, random_state=42)
        with pytest.warns(RuntimeWarning):
            res = find_multiresponse_design(
                FORMULA, FACTORS, self._multi_cfg(delta=0.02), opts
            )
        assert res["n"] <= 150
        cap_msgs = [m for m in res["warnings"] if "capped" in m]
        assert cap_msgs, f"no cap warning in MR warnings: {res['warnings']}"
        assert "max_n" in cap_msgs[0]
        assert "candidate_points" in cap_msgs[0]

    def test_compound_path_small_pool_does_not_crash(self):
        """Different per-response formulas activate the compound-criterion
        path, which has its own bisection; it must be capped too."""
        y2_cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]),  # A slope under the reduced "~ 1 + A"
            delta=np.array([1.0]),
            sigma=1.0,
            alpha=0.05,
            power=0.90,
        )
        multi = MultiResponseOptions(
            responses=[
                ResponseSpec(name="y1", power_cfg=_cfg(), formula="~ 1 + A + B"),
                ResponseSpec(name="y2", power_cfg=y2_cfg, formula="~ 1 + A"),
            ],
            power_combination="min",
        )
        opts = DesignOptions(candidate_points=200, starts=2, random_state=42)
        res = find_multiresponse_design(FORMULA, FACTORS, multi, opts)
        assert res["compound_criterion"] is True
        assert res["n"] <= 200
        assert res["achieved_power"] + 1e-3 >= 0.90


# ---------------------------------------------------------------------------
# SR-7: power_surface_2d must not evaluate at a hidden, arbitrary n
# ---------------------------------------------------------------------------

class TestSR7PowerSurfaceFixedN:
    """SR-7 regression: analytic sweeps (neither axis is 'n') previously used
    an undisclosed n ~ (p+1+max_n)/2 ~ 1000 -- crashing on small candidate
    pools or returning a meaningless ~1.0 surface. An explicit n is now
    required, validated, and disclosed; 'n' axes are capped at the candidate
    size (TICKET-039 parity)."""

    def _opts(self, **kw):
        base = dict(candidate_points=100, starts=1, max_iter=10, random_state=0)
        base.update(kw)
        return DesignOptions(**base)

    def test_analytic_sweep_without_n_raises(self):
        from lattice_doe import power_surface_2d
        with pytest.raises(ValueError, match="pass n="):
            power_surface_2d(
                FORMULA, FACTORS, _cfg(), "effect", (0.5, 2.0),
                "sigma", (0.5, 2.0), grid_points=3, design_opts=self._opts(),
            )

    def test_analytic_sweep_with_n_discloses_and_varies(self):
        from lattice_doe import power_surface_2d
        res = power_surface_2d(
            FORMULA, FACTORS, _cfg(), "effect", (0.5, 2.0),
            "sigma", (0.5, 2.0), grid_points=4, design_opts=self._opts(), n=12,
        )
        assert res["fixed_n"] == 12
        grid = res["power_grid"]
        # A real sensitivity surface varies; the old hidden n ~ 1000 gave ~1.0
        # everywhere.
        assert grid.min() < 0.9 < grid.max() + 0.2
        assert grid.min() < grid.max()

    @pytest.mark.parametrize("bad_n", [2, 101])
    def test_n_out_of_range_raises(self, bad_n):
        from lattice_doe import power_surface_2d
        with pytest.raises(ValueError, match="not evaluable"):
            power_surface_2d(
                FORMULA, FACTORS, _cfg(), "effect", (0.5, 2.0),
                "sigma", (0.5, 2.0), grid_points=3,
                design_opts=self._opts(), n=bad_n,
            )

    def test_n_alongside_n_axis_raises(self):
        from lattice_doe import power_surface_2d
        with pytest.raises(ValueError, match="drop the n argument"):
            power_surface_2d(
                FORMULA, FACTORS, _cfg(), "n", (5, 20),
                "effect", (0.5, 2.0), grid_points=3,
                design_opts=self._opts(), n=12,
            )

    def test_n_axis_clipped_to_candidate_size_warns(self):
        from lattice_doe import power_surface_2d
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = power_surface_2d(
                FORMULA, FACTORS, _cfg(), "n", (5, 500),
                "effect", (0.5, 2.0), grid_points=4, design_opts=self._opts(),
            )
        assert res["param1_values"].max() <= 100
        assert any("clipping" in str(w.message) for w in caught)

    def test_n_axis_sweep_unchanged(self):
        from lattice_doe import power_surface_2d
        res = power_surface_2d(
            FORMULA, FACTORS, _cfg(), "n", (5, 20),
            "effect", (0.5, 2.0), grid_points=3, design_opts=self._opts(),
        )
        assert res["fixed_n"] is None
        assert res["power_grid"].shape[1] == 3
