# tests/test_request_builder.py
"""Unit tests for iopt_power_design._request_builder."""
import numpy as np
import pytest

from iopt_power_design._request_builder import (
    build_design_opts,
    build_multi_response,
    build_power_cfg,
    build_response_spec,
    build_split_plot_opts,
)
from iopt_power_design.config import (
    DesignOptions,
    MultiResponseOptions,
    PowerContrastConfig,
    PowerGLMContrastConfig,
    PowerR2Config,
    ResponseSpec,
    SplitPlotOptions,
)


# ---------------------------------------------------------------------------
# Custom error class for testing error_cls forwarding
# ---------------------------------------------------------------------------
class _MyError(Exception):
    pass


# ---------------------------------------------------------------------------
# build_power_cfg — contrast mode
# ---------------------------------------------------------------------------
class TestBuildPowerCfgContrast:
    def _make(self, **overrides):
        d = dict(
            power_mode="contrast",
            L=[[1.0, -1.0]],
            delta=[0.5],
            alpha=0.05,
            power=0.80,
            sigma=1.0,
            max_n=200,
        )
        d.update(overrides)
        return build_power_cfg(d)

    def test_returns_contrast_config(self):
        cfg = self._make()
        assert isinstance(cfg, PowerContrastConfig)

    def test_L_shape(self):
        cfg = self._make()
        assert cfg.L.shape == (1, 2)

    def test_delta_shape(self):
        cfg = self._make()
        assert cfg.delta.shape == (1,)

    def test_alpha_forwarded(self):
        cfg = self._make(alpha=0.01)
        assert cfg.alpha == pytest.approx(0.01)

    def test_sigma_forwarded(self):
        cfg = self._make(sigma=2.5)
        assert cfg.sigma == pytest.approx(2.5)

    def test_max_n_forwarded(self):
        cfg = self._make(max_n=300)
        assert cfg.max_n == 300

    def test_tol_power_forwarded(self):
        cfg = self._make(tol_power=0.01)
        assert cfg.tol_power == pytest.approx(0.01)

    def test_max_iter_forwarded(self):
        cfg = self._make(max_iter=50)
        assert cfg.max_iter == 50

    def test_default_power_mode_is_contrast(self):
        # No power_mode key → defaults to contrast
        cfg = build_power_cfg({"L": [[1.0, 0.0]], "delta": [0.3]})
        assert isinstance(cfg, PowerContrastConfig)

    def test_missing_L_raises(self):
        with pytest.raises(ValueError, match="'L' is required"):
            build_power_cfg({"power_mode": "contrast", "delta": [0.3]})

    def test_missing_delta_raises(self):
        with pytest.raises(ValueError, match="'delta' is required"):
            build_power_cfg({"power_mode": "contrast", "L": [[1.0, 0.0]]})

    def test_custom_error_cls(self):
        with pytest.raises(_MyError):
            build_power_cfg(
                {"power_mode": "contrast", "delta": [0.3]},
                error_cls=_MyError,
            )

    def test_context_in_error_message(self):
        with pytest.raises(ValueError, match="my context"):
            build_power_cfg(
                {"power_mode": "contrast", "delta": [0.3]},
                context="my context",
            )

    def test_list_of_lists_L(self):
        cfg = build_power_cfg({"L": [[1.0, 0.0], [0.0, 1.0]], "delta": [0.3, 0.3]})
        assert cfg.L.shape == (2, 2)

    def test_numpy_array_L(self):
        cfg = build_power_cfg({"L": np.array([[1.0, -1.0]]), "delta": np.array([0.5])})
        assert isinstance(cfg.L, np.ndarray)


# ---------------------------------------------------------------------------
# build_power_cfg — r2 mode
# ---------------------------------------------------------------------------
class TestBuildPowerCfgR2:
    def _make(self, **overrides):
        d = dict(power_mode="r2", r2_target=0.20, alpha=0.05, power=0.80, max_n=200)
        d.update(overrides)
        return build_power_cfg(d)

    def test_returns_r2_config(self):
        assert isinstance(self._make(), PowerR2Config)

    def test_r2_target_forwarded(self):
        cfg = self._make(r2_target=0.30)
        assert cfg.r2_target == pytest.approx(0.30)

    def test_lambda_mode_forwarded(self):
        cfg = self._make(lambda_mode="n_minus_p")
        assert cfg.lambda_mode == "n_minus_p"

    def test_lambda_mode_default_is_n(self):
        cfg = self._make()
        assert cfg.lambda_mode == "n"

    def test_missing_r2_target_raises(self):
        with pytest.raises(ValueError, match="r2_target"):
            build_power_cfg({"power_mode": "r2"})

    def test_sigma_forwarded(self):
        cfg = self._make(sigma=1.5)
        assert cfg.sigma == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# build_power_cfg — glm mode
# ---------------------------------------------------------------------------
class TestBuildPowerCfgGLM:
    def _make(self, **overrides):
        d = dict(
            power_mode="glm",
            L=[[1.0, 0.0]],
            delta=[0.5],
            baseline=0.20,
            family="binomial",
            alpha=0.05,
            power=0.80,
            max_n=200,
        )
        d.update(overrides)
        return build_power_cfg(d)

    def test_returns_glm_config(self):
        assert isinstance(self._make(), PowerGLMContrastConfig)

    def test_baseline_forwarded(self):
        cfg = self._make(baseline=0.35)
        assert cfg.baseline == pytest.approx(0.35)

    def test_family_forwarded(self):
        cfg = self._make(family="poisson")
        assert cfg.family == "poisson"

    def test_link_forwarded(self):
        cfg = self._make(link="logit")
        assert cfg.link == "logit"

    def test_link_none_uses_canonical(self):
        # When link is omitted, PowerGLMContrastConfig.__post_init__ sets the
        # canonical link for the family (binomial → "logit", poisson → "log").
        cfg = self._make()
        assert cfg.link == "logit"

    def test_missing_baseline_raises(self):
        with pytest.raises(ValueError, match="baseline"):
            build_power_cfg({"power_mode": "glm", "L": [[1.0]], "delta": [0.5]})

    def test_missing_L_raises(self):
        with pytest.raises(ValueError, match="'L' is required"):
            build_power_cfg({"power_mode": "glm", "delta": [0.5], "baseline": 0.2})

    def test_tol_power_forwarded(self):
        cfg = self._make(tol_power=0.005)
        assert cfg.tol_power == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# build_split_plot_opts
# ---------------------------------------------------------------------------
class TestBuildSplitPlotOpts:
    def _make(self, **overrides):
        d = dict(htc_factors=["A"], n_whole_plots=4, eta=1.0)
        d.update(overrides)
        return build_split_plot_opts(d)

    def test_returns_split_plot_options(self):
        assert isinstance(self._make(), SplitPlotOptions)

    def test_htc_factors_forwarded(self):
        sp = self._make(htc_factors=["X", "Y"])
        assert sp.htc_factors == ["X", "Y"]

    def test_n_whole_plots_forwarded(self):
        sp = self._make(n_whole_plots=6)
        assert sp.n_whole_plots == 6

    def test_eta_forwarded(self):
        sp = self._make(eta=2.0)
        assert sp.eta == pytest.approx(2.0)

    def test_subplots_per_wp_forwarded(self):
        sp = self._make(subplots_per_wp=3)
        assert sp.subplots_per_wp == 3

    def test_subplots_per_wp_zero_becomes_none(self):
        sp = self._make(subplots_per_wp=0)
        assert sp.subplots_per_wp is None

    def test_subplots_per_wp_none_stays_none(self):
        sp = self._make(subplots_per_wp=None)
        assert sp.subplots_per_wp is None

    def test_df_method_forwarded(self):
        sp = self._make(df_method="conservative")
        assert sp.df_method == "conservative"

    def test_df_method_default(self):
        sp = self._make()
        assert sp.df_method == "auto"

    def test_custom_error_cls(self):
        with pytest.raises(_MyError):
            build_split_plot_opts(
                {"htc_factors": [], "n_whole_plots": 4},
                error_cls=_MyError,
            )


# ---------------------------------------------------------------------------
# build_design_opts
# ---------------------------------------------------------------------------
class TestBuildDesignOpts:
    def test_minimal_dict_returns_design_options(self):
        opts = build_design_opts({})
        assert isinstance(opts, DesignOptions)

    def test_criterion_forwarded(self):
        opts = build_design_opts({"criterion": "D"})
        assert opts.criterion == "D"

    def test_starts_forwarded(self):
        opts = build_design_opts({"starts": 10})
        assert opts.starts == 10

    def test_random_state_forwarded(self):
        opts = build_design_opts({"random_state": 42})
        assert opts.random_state == 42

    def test_n_blocks_forwarded_when_gte_2(self):
        opts = build_design_opts({"n_blocks": 3})
        assert opts.n_blocks == 3

    def test_n_blocks_not_forwarded_when_lt_2(self):
        opts = build_design_opts({"n_blocks": 1})
        assert opts.n_blocks is None

    def test_n_blocks_not_forwarded_when_zero(self):
        opts = build_design_opts({"n_blocks": 0})
        assert opts.n_blocks is None

    def test_n_blocks_not_forwarded_when_none(self):
        opts = build_design_opts({"n_blocks": None})
        assert opts.n_blocks is None

    def test_alloc_max_forwarded_when_gt_zero(self):
        opts = build_design_opts({"alloc_max_per_cell": 10})
        assert opts.alloc_max_per_cell == 10

    def test_alloc_max_not_forwarded_when_zero(self):
        opts = build_design_opts({"alloc_max_per_cell": 0})
        assert opts.alloc_max_per_cell is None

    def test_constraint_expr_forwarded(self):
        opts = build_design_opts({"constraint_expr": "A <= 5"})
        assert opts.constraint_expr == "A <= 5"

    def test_constraint_expr_not_forwarded_when_empty(self):
        opts = build_design_opts({"constraint_expr": ""})
        assert opts.constraint_expr is None

    def test_split_plot_dict_is_built(self):
        opts = build_design_opts({
            "split_plot": {
                "htc_factors": ["HTC1"],
                "n_whole_plots": 4,
            }
        })
        assert isinstance(opts.split_plot, SplitPlotOptions)
        assert opts.split_plot.htc_factors == ["HTC1"]

    def test_no_split_plot_when_absent(self):
        opts = build_design_opts({})
        assert opts.split_plot is None

    def test_block_sizes_forwarded(self):
        opts = build_design_opts({"n_blocks": 2, "block_sizes": [4, 4]})
        assert opts.block_sizes == [4, 4]

    def test_workers_none_forwarded(self):
        opts = build_design_opts({"workers": None})
        assert opts.workers is None

    def test_workers_int_forwarded(self):
        opts = build_design_opts({"workers": 4})
        assert opts.workers == 4

    def test_preallocate_categorical_forwarded(self):
        opts = build_design_opts({"preallocate_categorical": True})
        assert opts.preallocate_categorical is True

    def test_custom_error_cls_on_bad_criterion(self):
        with pytest.raises(_MyError):
            build_design_opts({"criterion": "Z"}, error_cls=_MyError)

    def test_context_in_error_message(self):
        with pytest.raises(ValueError, match="my sheet"):
            build_design_opts({"criterion": "Z"}, context="my sheet")


# ---------------------------------------------------------------------------
# build_response_spec
# ---------------------------------------------------------------------------
class TestBuildResponseSpec:
    def _contrast_spec(self, **overrides):
        d = dict(
            name="Y1",
            power_cfg=dict(power_mode="contrast", L=[[1.0, 0.0]], delta=[0.3]),
            weight=1.0,
        )
        d.update(overrides)
        return build_response_spec(d)

    def test_returns_response_spec(self):
        assert isinstance(self._contrast_spec(), ResponseSpec)

    def test_name_forwarded(self):
        spec = self._contrast_spec(name="Yield")
        assert spec.name == "Yield"

    def test_weight_forwarded(self):
        spec = self._contrast_spec(weight=2.0)
        assert spec.weight == pytest.approx(2.0)

    def test_formula_forwarded(self):
        spec = self._contrast_spec(formula="~ A + B")
        assert spec.formula == "~ A + B"

    def test_formula_none_default(self):
        spec = self._contrast_spec()
        assert spec.formula is None

    def test_power_cfg_built_correctly(self):
        spec = self._contrast_spec()
        assert isinstance(spec.power_cfg, PowerContrastConfig)

    def test_r2_response_spec(self):
        spec = build_response_spec({
            "name": "R",
            "power_cfg": {"power_mode": "r2", "r2_target": 0.25},
        })
        assert isinstance(spec.power_cfg, PowerR2Config)

    def test_glm_response_spec(self):
        spec = build_response_spec({
            "name": "G",
            "power_cfg": {
                "power_mode": "glm",
                "L": [[1.0, 0.0]],
                "delta": [0.5],
                "baseline": 0.20,
            },
        })
        assert isinstance(spec.power_cfg, PowerGLMContrastConfig)

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            build_response_spec({"power_cfg": {"power_mode": "r2", "r2_target": 0.2}})

    def test_power_cfg_not_dict_raises(self):
        with pytest.raises(ValueError, match="power_cfg"):
            build_response_spec({"name": "Y", "power_cfg": "contrast"})

    def test_custom_error_cls(self):
        with pytest.raises(_MyError):
            build_response_spec(
                {"name": "Y", "power_cfg": {"power_mode": "r2"}},
                error_cls=_MyError,
            )


# ---------------------------------------------------------------------------
# build_multi_response
# ---------------------------------------------------------------------------
class TestBuildMultiResponse:
    def _two_contrast_responses(self):
        return [
            {"name": "Y1", "power_cfg": {"power_mode": "contrast", "L": [[1.0, 0.0]], "delta": [0.3]}},
            {"name": "Y2", "power_cfg": {"power_mode": "contrast", "L": [[0.0, 1.0]], "delta": [0.4]}},
        ]

    def test_returns_multi_response_options(self):
        mr = build_multi_response({"responses": self._two_contrast_responses()})
        assert isinstance(mr, MultiResponseOptions)

    def test_responses_count(self):
        mr = build_multi_response({"responses": self._two_contrast_responses()})
        assert len(mr.responses) == 2

    def test_power_combination_default(self):
        mr = build_multi_response({"responses": self._two_contrast_responses()})
        assert mr.power_combination == "min"

    def test_power_combination_forwarded(self):
        mr = build_multi_response({
            "responses": self._two_contrast_responses(),
            "power_combination": "product",
        })
        assert mr.power_combination == "product"

    def test_sigma_joint_none_default(self):
        mr = build_multi_response({"responses": self._two_contrast_responses()})
        assert mr.sigma_joint is None

    def test_sigma_joint_forwarded(self):
        mr = build_multi_response({
            "responses": self._two_contrast_responses(),
            "sigma_joint": [[1.0, 0.3], [0.3, 1.0]],
        })
        assert mr.sigma_joint is not None
        assert mr.sigma_joint.shape == (2, 2)
        assert mr.sigma_joint[0, 1] == pytest.approx(0.3)

    def test_too_few_responses_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            build_multi_response({
                "responses": [
                    {"name": "Y1", "power_cfg": {"power_mode": "r2", "r2_target": 0.2}}
                ]
            })

    def test_empty_responses_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            build_multi_response({"responses": []})

    def test_mixed_modes(self):
        mr = build_multi_response({
            "responses": [
                {"name": "Y1", "power_cfg": {"power_mode": "contrast", "L": [[1.0, 0.0]], "delta": [0.3]}},
                {"name": "Y2", "power_cfg": {"power_mode": "r2", "r2_target": 0.25}},
            ]
        })
        assert isinstance(mr.responses[0].power_cfg, PowerContrastConfig)
        assert isinstance(mr.responses[1].power_cfg, PowerR2Config)

    def test_custom_error_cls(self):
        with pytest.raises(_MyError):
            build_multi_response({"responses": []}, error_cls=_MyError)

    def test_context_in_error_message(self):
        with pytest.raises(ValueError, match="sheet parse"):
            build_multi_response({"responses": []}, context="sheet parse")

    def test_invalid_sigma_joint_raises(self):
        with pytest.raises(ValueError, match="sigma_joint"):
            build_multi_response({
                "responses": self._two_contrast_responses(),
                "sigma_joint": "not_a_matrix",
            })
