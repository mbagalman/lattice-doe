# tests/test_result_contract.py
"""Result-contract tests, split from test_contrasts.py.

Every public return path carries the coding authority (the four-key
envelope with ``model_matrix`` / per-response ``model_matrices``), and the
fixed-design analyses consume that matrix — blocked, split-plot and
compound multi-response modes included."""
import pytest

class TestEveryResultModeCarriesTheBasis:
    """UX-57: the four-key contract holds on every public return path —
    split-plot and multi-response included, not just the ordinary mode."""

    def test_split_plot_result_has_model_matrix_and_warning(self):
        import warnings as W

        import numpy as np

        from lattice_doe import DesignOptions, find_optimal_design
        from lattice_doe.config import PowerContrastConfig, SplitPlotOptions

        f = "~ 1 + bs(x, knots=[0.5], lower_bound=0.0, upper_bound=1.0) + C(w)"
        factors = {"x": (0.0, 1.0), "w": ["a", "b"]}
        opts = DesignOptions(
            candidate_points=60, random_state=2, starts=1,
            split_plot=SplitPlotOptions(htc_factors=["w"], n_whole_plots=4),
        )
        cfg = PowerContrastConfig(
            L=np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
            delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=40,
        )
        with W.catch_warnings():
            W.simplefilter("ignore")
            res = find_optimal_design(f, factors, cfg, opts)
        mm = res["model_matrix"]
        assert mm.shape[0] == len(res["design_df"])
        # fully-specified spline -> fixed coding -> no basis warning
        assert not any("model_matrix" in w
                       for w in res["report"].get("warnings", []))

    def test_multiresponse_result_has_model_matrix(self):
        import warnings as W

        import numpy as np

        from lattice_doe import DesignOptions, find_multiresponse_design
        from lattice_doe.config import (
            MultiResponseOptions, PowerContrastConfig, ResponseSpec,
        )

        f, factors = "~ 1 + bs(x, df=4)", {"x": (0.0, 1.0)}
        opts = DesignOptions(candidate_points=100, random_state=4, starts=1)
        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0, 0.0, 0.0, -1.0]]),
            delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=25,
        )
        multi = MultiResponseOptions(responses=[
            ResponseSpec(name="y1", power_cfg=cfg),
            ResponseSpec(name="y2", power_cfg=cfg),
        ])
        with W.catch_warnings(record=True) as caught:
            W.simplefilter("always")
            res = find_multiresponse_design(f, factors, multi, opts)
        mm = res["model_matrix"]
        assert mm.shape == (len(res["design_df"]), 5)
        # MR warnings consolidate per response and point at model_matrices
        # (note: "model_matrix" is NOT a substring of "model_matrices").
        assert any("model_matrices" in str(w.message) for w in caught)
        assert any("model_matrices" in w
                   for w in res["report"].get("warnings", []))
        # and the per-response authorities are present for every response
        mms = res["model_matrices"]
        assert list(mms) == ["y1", "y2"]
        assert all(v.shape == mm.shape for v in mms.values())

    def test_blocked_result_has_augmented_parameter_names(self):
        """UX-60: blocked runs power the augmented model (treatment + block
        dummies); their model_matrix must carry the augmented parameter
        names, not the x0.. fallback."""
        import numpy as np

        from lattice_doe import DesignOptions, find_optimal_design
        from lattice_doe.config import PowerContrastConfig

        opts = DesignOptions(
            candidate_points=80, random_state=6, starts=1, n_blocks=2,
        )
        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=24,
        )
        res = find_optimal_design("~ 1 + x", {"x": (0.0, 1.0)}, cfg, opts)
        mm = res["model_matrix"]
        cols = list(mm.columns)
        assert not any(c.startswith("x0") or c == "x1" for c in cols), cols
        assert any("Block" in c for c in cols), cols   # block dummy named
        assert "x" in cols                             # treatment named
        assert mm.shape[0] == len(res["design_df"])


class TestBlockedAnalysisAlignment:
    """UX-62: a blocked run's authoritative matrix is the AUGMENTED model
    (treatment + block dummies), but power_cfg.L addresses the treatment
    columns and R²'s tested predictors are the treatment slopes only. The
    analysis functions share the API's name-based alignment, so their nominal
    power must EQUAL the run's achieved power — contrast mode used to crash
    on shape, and R² mode silently counted the block dummy as tested."""

    @staticmethod
    def _blocked_run(power_cfg):
        from lattice_doe import DesignOptions, find_optimal_design

        opts = DesignOptions(
            candidate_points=80, random_state=6, starts=1, n_blocks=2,
        )
        res = find_optimal_design("~ 1 + x", {"x": (0.0, 1.0)}, power_cfg, opts)
        return res, opts

    def test_contrast_sensitivity_matches_run_power(self):
        import numpy as np

        from lattice_doe.analysis import power_sensitivity
        from lattice_doe.config import PowerContrastConfig

        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=24,
        )
        res, opts = self._blocked_run(cfg)
        sens = power_sensitivity(
            formula="~ 1 + x", factors={"x": (0.0, 1.0)}, power_cfg=cfg,
            design_df=res["design_df"], design_opts=opts,
            model_matrix=res["model_matrix"],
        )
        assert np.isclose(sens["nominal_power"],
                          res["report"]["achieved_power"], atol=1e-9)

    def test_r2_sensitivity_matches_run_power(self):
        import numpy as np

        from lattice_doe.analysis import power_sensitivity
        from lattice_doe.config import PowerR2Config

        cfg = PowerR2Config(r2_target=0.3, alpha=0.05, power=0.8, max_n=30)
        res, opts = self._blocked_run(cfg)
        sens = power_sensitivity(
            formula="~ 1 + x", factors={"x": (0.0, 1.0)}, power_cfg=cfg,
            design_df=res["design_df"], design_opts=opts,
            model_matrix=res["model_matrix"],
        )
        # The silent failure mode: without treatment-only df_num this was
        # 0.69 vs 0.81 — same shapes, wrong tested-predictor count.
        assert np.isclose(sens["nominal_power"],
                          res["report"]["achieved_power"], atol=1e-9)

    def test_mde_and_robustness_accept_blocked_matrix(self):
        import numpy as np

        from lattice_doe.analysis import min_detectable_effect, robustness_report
        from lattice_doe.config import PowerContrastConfig

        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=24,
        )
        res, opts = self._blocked_run(cfg)
        kw = dict(design_df=res["design_df"], formula="~ 1 + x",
                  factors={"x": (0.0, 1.0)}, power_cfg=cfg, design_opts=opts,
                  model_matrix=res["model_matrix"])
        mde = min_detectable_effect(**kw)
        assert mde["mde"] > 0
        rob = robustness_report(**kw)
        assert np.isclose(rob["nominal_power"],
                          res["report"]["achieved_power"], atol=1e-9)

    def test_glm_baseline_curve_accepts_blocked_matrix(self):
        import numpy as np

        from lattice_doe.analysis import power_curve_by_baseline
        from lattice_doe.config import PowerGLMContrastConfig

        cfg = PowerGLMContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.8]),
            baseline=0.2, family="binomial",
            alpha=0.05, power=0.8, max_n=40,
        )
        res, opts = self._blocked_run(cfg)
        df = power_curve_by_baseline(
            formula="~ 1 + x", factors={"x": (0.0, 1.0)}, cfg=cfg,
            design_df=res["design_df"], design_opts=opts,
            model_matrix=res["model_matrix"], baseline_points=3,
        )
        assert len(df) == 3 and (df["power"] > 0).all()

    def test_unblocked_analysis_identity_preserved(self):
        """The alignment is a no-op for ordinary matrices: nominal power from
        the authoritative matrix equals the run's achieved power exactly."""
        import numpy as np

        from lattice_doe import DesignOptions, find_optimal_design
        from lattice_doe.analysis import power_sensitivity
        from lattice_doe.config import PowerContrastConfig

        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=24,
        )
        opts = DesignOptions(candidate_points=80, random_state=6, starts=1)
        res = find_optimal_design("~ 1 + x", {"x": (0.0, 1.0)}, cfg, opts)
        sens = power_sensitivity(
            formula="~ 1 + x", factors={"x": (0.0, 1.0)}, power_cfg=cfg,
            design_df=res["design_df"], design_opts=opts,
            model_matrix=res["model_matrix"],
        )
        assert np.isclose(sens["nominal_power"],
                          res["report"]["achieved_power"], atol=1e-9)

    def test_blocked_without_matrix_keeps_prior_treatment_behavior(self):
        """Without model_matrix a plain formula rebuilds the treatment-only
        matrix from design_df (blocks ignored) — the long-standing behavior,
        which must not crash or change."""
        import numpy as np

        from lattice_doe.analysis import power_sensitivity
        from lattice_doe.config import PowerContrastConfig

        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=24,
        )
        res, opts = self._blocked_run(cfg)
        sens = power_sensitivity(
            formula="~ 1 + x", factors={"x": (0.0, 1.0)}, power_cfg=cfg,
            design_df=res["design_df"], design_opts=opts,
        )
        assert 0.0 < sens["nominal_power"] < 1.0

    def test_blocked_report_carries_nuisance_metadata(self):
        import numpy as np

        from lattice_doe.config import PowerContrastConfig

        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=24,
        )
        res, _ = self._blocked_run(cfg)
        nuis = res["report"]["nuisance_columns"]
        assert nuis and all("Block" in c for c in nuis)
        assert set(nuis) < set(res["model_matrix"].columns)


class TestCompoundResponseMatrices:
    """UX-63: in compound multi-response mode each response's power is
    computed on its OWN formula's matrix over the shared design rows — the
    global-formula model_matrix is not that response's authority. Results
    must carry ordered per-response matrices and run the data-dependence
    check over every response formula."""

    _FACTORS = {"x": (0.0, 1.0)}

    @staticmethod
    def _compound_run(y2_formula="~ 1 + bs(x, df=4)"):
        import warnings as W

        import numpy as np

        from lattice_doe import DesignOptions, find_multiresponse_design
        from lattice_doe.config import (
            MultiResponseOptions, PowerContrastConfig, ResponseSpec,
        )

        opts = DesignOptions(candidate_points=100, random_state=4, starts=1)
        cfg_lin = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=18,
        )
        q2 = 5 if "bs(" in y2_formula else 3
        L2 = np.zeros((1, q2)); L2[0, 1] = 1.0
        cfg_2 = PowerContrastConfig(
            L=L2, delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=18,
        )
        multi = MultiResponseOptions(responses=[
            ResponseSpec(name="y1", power_cfg=cfg_lin),
            ResponseSpec(name="y2", power_cfg=cfg_2, formula=y2_formula),
        ])
        with W.catch_warnings(record=True) as caught:
            W.simplefilter("always")
            res = find_multiresponse_design(
                "~ 1 + x", {"x": (0.0, 1.0)}, multi, opts,
            )
        return res, opts, caught

    def test_per_response_matrices_are_each_responses_authority(self):
        import numpy as np
        import patsy

        from lattice_doe.candidate import build_search_candidate

        res, opts, caught = self._compound_run()
        assert res["report"]["compound_criterion"] is True
        mms = res["model_matrices"]
        assert list(mms) == ["y1", "y2"]           # configured order
        assert mms["y1"].shape[1] == 2
        assert mms["y2"].shape[1] == 5             # the 18x5 basis y2 used
        assert list(mms["y2"].columns)[1].startswith("bs(x")

        # y2's matrix equals its formula coded from the RUN's candidate,
        # evaluated at the design rows — the exact powered basis.
        cand, _ = build_search_candidate("~ 1 + x", self._FACTORS, opts)
        di = patsy.incr_dbuilder("~ 1 + bs(x, df=4)", lambda: iter([cand]))
        (ref,) = patsy.build_design_matrices([di], res["design_df"])
        assert np.allclose(np.asarray(mms["y2"]), np.asarray(ref))

    def test_dependency_warning_covers_response_formulas(self):
        res, _, caught = self._compound_run()
        hits = [str(w.message) for w in caught
                if "model_matrices" in str(w.message)]
        assert hits and "y2" in hits[0] and "bs(x, df=4)" in hits[0]
        assert any("y2" in w for w in res["report"].get("warnings", []))

    def test_plain_compound_carries_matrices_without_warning(self):
        res, _, caught = self._compound_run(y2_formula="~ 1 + x + I(x**2)")
        assert list(res["model_matrices"]) == ["y1", "y2"]
        assert res["model_matrices"]["y2"].shape[1] == 3
        assert not any("model_matrices" in str(w.message) for w in caught)

    def test_rest_response_carries_model_matrices(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from api_server.main import create_app

        client = TestClient(create_app())
        req = {
            "formula": "~ 1 + x",
            "factors": {"x": {"type": "continuous", "low": 0.0, "high": 1.0}},
            "multi_cfg": {
                "responses": [
                    {"name": "y1", "power_cfg": {
                        "type": "contrast", "L": [[0.0, 1.0]], "delta": [0.5],
                        "alpha": 0.05, "power": 0.8, "sigma": 1.0, "max_n": 18}},
                    {"name": "y2", "formula": "~ 1 + bs(x, df=4)",
                     "power_cfg": {
                        "type": "contrast",
                        "L": [[0.0, 1.0, 0.0, 0.0, -1.0]], "delta": [0.5],
                        "alpha": 0.05, "power": 0.8, "sigma": 1.0, "max_n": 18}},
                ],
            },
            "design_opts": {"candidate_points": 100, "random_state": 4,
                            "starts": 1},
        }
        r = client.post("/multiresponse_design", json=req)
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        mms = body.get("model_matrices")
        assert mms is not None and set(mms) == {"y1", "y2"}
        # Split orientation (UX-64): explicit ordered columns + row arrays.
        assert len(mms["y2"]["columns"]) == 5
        assert mms["y2"]["columns"][1].startswith("bs(x")
        assert len(mms["y2"]["data"][0]) == 5

    def test_split_plot_mr_also_carries_per_response_matrices(self):
        """Mode-matrix cell: the SP-MR return path routes through the same
        envelope; every response shares the SP GLS matrix there."""
        import numpy as np

        from lattice_doe import DesignOptions, find_multiresponse_design
        from lattice_doe.config import (
            MultiResponseOptions, PowerContrastConfig, ResponseSpec,
            SplitPlotOptions,
        )

        cfg = PowerContrastConfig(
            L=np.array([[0.0, 0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=24,
        )
        multi = MultiResponseOptions(responses=[
            ResponseSpec(name="y1", power_cfg=cfg),
            ResponseSpec(name="y2", power_cfg=cfg),
        ])
        opts = DesignOptions(
            candidate_points=60, random_state=2, starts=1,
            split_plot=SplitPlotOptions(htc_factors=["w"], n_whole_plots=4),
        )
        res = find_multiresponse_design(
            "~ 1 + C(w) + x", {"x": (0.0, 1.0), "w": ["a", "b"]}, multi, opts,
        )
        mms = res["model_matrices"]
        assert list(mms) == ["y1", "y2"]
        assert all(v.shape == res["model_matrix"].shape for v in mms.values())
