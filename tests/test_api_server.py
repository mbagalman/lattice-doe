# tests/test_api_server.py
# License: MIT
"""
Tests for the lattice-doe FastAPI REST API server.

Test layers
-----------
Layer 1 — Unit (no compute, no HTTP)
    Serialization helpers, Pydantic model validation.

Layer 2 — HTTP unit (ASGI test client, mock or trivial compute)
    Health check, OpenAPI schema, 404, 422 validation errors.
    Skipped if FastAPI / httpx not installed.

Layer 3 — HTTP integration (real compute, small FAST_OPTS)
    One round-trip per endpoint with a two-factor R² problem.
    Marked @pytest.mark.slow; skipped if FastAPI / httpx not installed.
"""
from __future__ import annotations

import importlib.util
import math
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Availability checks (no hard dependency at module level)
# ---------------------------------------------------------------------------

_HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
_HAS_HTTPX = importlib.util.find_spec("httpx") is not None
_HAS_SERVER = _HAS_FASTAPI and _HAS_HTTPX

# Layer 1 needs no HTTP stack, but the api_server request/response models it
# converts are Pydantic v2 models, and pydantic ships only with the [server]
# extra. Skip the whole module on a core install (same pattern as
# test_report.py's jinja2 guard) so `pytest` passes without extras.
pytest.importorskip("pydantic", reason="server extra not installed")

# ---------------------------------------------------------------------------
# Layer 1 — pure unit tests (no HTTP, no FastAPI needed)
# ---------------------------------------------------------------------------

from api_server.serialization import (
    sanitize_float,
    sanitize_value,
    df_to_records,
    records_to_df,
    pydantic_power_cfg_to_dataclass,
    pydantic_design_opts_to_dataclass,
    serialize_design_result,
)
from lattice_doe.config import DesignOptions, PowerContrastConfig, PowerR2Config


class TestSanitizeFloat:
    def test_normal_float(self):
        assert sanitize_float(3.14) == pytest.approx(3.14)

    def test_numpy_float64(self):
        assert sanitize_float(np.float64(2.5)) == pytest.approx(2.5)

    def test_nan_returns_none(self):
        assert sanitize_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert sanitize_float(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert sanitize_float(float("-inf")) is None

    def test_none_returns_none(self):
        assert sanitize_float(None) is None

    def test_zero(self):
        assert sanitize_float(0.0) == 0.0


class TestSanitizeValue:
    def test_numpy_integer(self):
        assert sanitize_value(np.int64(42)) == 42
        assert isinstance(sanitize_value(np.int64(42)), int)

    def test_numpy_float(self):
        result = sanitize_value(np.float64(1.5))
        assert result == pytest.approx(1.5)

    def test_numpy_array(self):
        result = sanitize_value(np.array([1.0, 2.0]))
        assert result == [1.0, 2.0]

    def test_nested_dict(self):
        d = {"a": np.int64(3), "b": {"c": np.float64(0.5)}}
        result = sanitize_value(d)
        assert result == {"a": 3, "b": {"c": pytest.approx(0.5)}}

    def test_list_with_numpy(self):
        result = sanitize_value([np.int64(1), np.float64(2.5)])
        assert result == [1, pytest.approx(2.5)]

    def test_nan_float_in_dict(self):
        result = sanitize_value({"x": float("nan")})
        assert result["x"] is None


class TestDfToRecords:
    def test_basic_roundtrip(self):
        df = pd.DataFrame({"A": [1.0, 2.0], "B": ["x", "y"]})
        records = df_to_records(df)
        assert len(records) == 2
        assert records[0]["A"] == pytest.approx(1.0)
        assert records[0]["B"] == "x"

    def test_nan_becomes_none(self):
        df = pd.DataFrame({"A": [1.0, float("nan")]})
        records = df_to_records(df)
        assert records[1]["A"] is None

    def test_numpy_scalars_converted(self):
        df = pd.DataFrame({"x": np.array([1, 2], dtype=np.int64)})
        records = df_to_records(df)
        assert isinstance(records[0]["x"], int)


class TestRecordsToDf:
    def test_basic(self):
        records = [{"A": 1.0, "B": "x"}, {"A": 2.0, "B": "y"}]
        df = records_to_df(records)
        assert list(df.columns) == ["A", "B"]
        assert len(df) == 2

    def test_empty_returns_empty_df(self):
        df = records_to_df([])
        assert len(df) == 0


class TestPydanticPowerCfgToDataclass:
    def _r2_model(self):
        from api_server.models.common import PowerR2ConfigModel
        return PowerR2ConfigModel(type="r2", r2_target=0.15, alpha=0.05, power=0.8)

    def _contrast_model(self):
        from api_server.models.common import PowerContrastConfigModel
        return PowerContrastConfigModel(
            type="contrast",
            L=[[0, 1, 0]],
            delta=[0.5],
            sigma=1.0,
        )

    def test_r2_returns_PowerR2Config(self):
        cfg = pydantic_power_cfg_to_dataclass(self._r2_model())
        assert isinstance(cfg, PowerR2Config)

    def test_r2_fields_propagated(self):
        cfg = pydantic_power_cfg_to_dataclass(self._r2_model())
        assert cfg.r2_target == pytest.approx(0.15)
        assert cfg.alpha == pytest.approx(0.05)
        assert cfg.power == pytest.approx(0.8)

    def test_contrast_returns_PowerContrastConfig(self):
        cfg = pydantic_power_cfg_to_dataclass(self._contrast_model())
        assert isinstance(cfg, PowerContrastConfig)

    def test_contrast_L_is_ndarray(self):
        cfg = pydantic_power_cfg_to_dataclass(self._contrast_model())
        assert isinstance(cfg.L, np.ndarray)
        assert cfg.L.shape == (1, 3)

    def test_contrast_delta_is_ndarray(self):
        cfg = pydantic_power_cfg_to_dataclass(self._contrast_model())
        assert isinstance(cfg.delta, np.ndarray)
        assert cfg.delta.shape == (1,)


class TestPydanticDesignOptsToDataclass:
    def test_none_returns_default_DesignOptions(self):
        opts = pydantic_design_opts_to_dataclass(None)
        assert isinstance(opts, DesignOptions)

    def test_fields_propagated(self):
        from api_server.models.common import DesignOptionsModel
        model = DesignOptionsModel(criterion="D", starts=10, random_state=42)
        opts = pydantic_design_opts_to_dataclass(model)
        assert opts.criterion == "D"
        assert opts.starts == 10
        assert opts.random_state == 42

    def test_workers_always_none(self):
        """workers is forced to None inside ASGI."""
        from api_server.models.common import DesignOptionsModel
        model = DesignOptionsModel()
        opts = pydantic_design_opts_to_dataclass(model)
        assert opts.workers is None

    def test_constraint_expr_forwarded(self):
        from api_server.models.common import DesignOptionsModel
        model = DesignOptionsModel(constraint_expr="A + B <= 1")
        opts = pydantic_design_opts_to_dataclass(model)
        assert opts.constraint_expr == "A + B <= 1"

    def test_n_blocks_forwarded(self):
        from api_server.models.common import DesignOptionsModel
        model = DesignOptionsModel(n_blocks=3)
        opts = pydantic_design_opts_to_dataclass(model)
        assert opts.n_blocks == 3

    def test_alloc_max_per_cell_forwarded(self):
        from api_server.models.common import DesignOptionsModel
        model = DesignOptionsModel(alloc_max_per_cell=5)
        opts = pydantic_design_opts_to_dataclass(model)
        assert opts.alloc_max_per_cell == 5

    def test_alloc_max_per_cell_none_not_passed(self):
        """alloc_max_per_cell=None should not appear in DesignOptions kwargs."""
        from api_server.models.common import DesignOptionsModel
        model = DesignOptionsModel(alloc_max_per_cell=None)
        opts = pydantic_design_opts_to_dataclass(model)
        # Should use DesignOptions default (None)
        assert opts.alloc_max_per_cell is None

    # ------------------------------------------------------------------ #
    # CR-28 regression: split_plot field round-trips through serialization #
    # ------------------------------------------------------------------ #

    def test_split_plot_none_by_default(self):
        """CR-28: split_plot defaults to None — no split-plot mode unless set."""
        from api_server.models.common import DesignOptionsModel
        model = DesignOptionsModel()
        opts = pydantic_design_opts_to_dataclass(model)
        assert opts.split_plot is None

    def test_split_plot_fields_forwarded(self):
        """CR-28: all SplitPlotOptionsModel fields are mapped to SplitPlotOptions."""
        from api_server.models.common import DesignOptionsModel, SplitPlotOptionsModel
        from lattice_doe.config import SplitPlotOptions
        sp_model = SplitPlotOptionsModel(
            htc_factors=["Temp", "Press"],
            n_whole_plots=6,
            eta=2.5,
            subplots_per_wp=4,
            df_method="conservative",
        )
        model = DesignOptionsModel(split_plot=sp_model)
        opts = pydantic_design_opts_to_dataclass(model)
        assert opts.split_plot is not None
        assert isinstance(opts.split_plot, SplitPlotOptions)
        sp = opts.split_plot
        assert sp.htc_factors == ["Temp", "Press"]
        assert sp.n_whole_plots == 6
        assert sp.eta == pytest.approx(2.5)
        assert sp.subplots_per_wp == 4
        assert sp.df_method == "conservative"

    def test_split_plot_subplots_per_wp_none(self):
        """CR-28: omitting subplots_per_wp passes None to SplitPlotOptions (auto)."""
        from api_server.models.common import DesignOptionsModel, SplitPlotOptionsModel
        sp_model = SplitPlotOptionsModel(htc_factors=["A"], n_whole_plots=3)
        model = DesignOptionsModel(split_plot=sp_model)
        opts = pydantic_design_opts_to_dataclass(model)
        assert opts.split_plot is not None
        assert opts.split_plot.subplots_per_wp is None

    def test_split_plot_defaults_eta_and_df_method(self):
        """CR-28: SplitPlotOptionsModel defaults (eta=1.0, df_method='auto') are preserved."""
        from api_server.models.common import DesignOptionsModel, SplitPlotOptionsModel
        sp_model = SplitPlotOptionsModel(htc_factors=["A"], n_whole_plots=4)
        model = DesignOptionsModel(split_plot=sp_model)
        opts = pydantic_design_opts_to_dataclass(model)
        assert opts.split_plot.eta == pytest.approx(1.0)
        assert opts.split_plot.df_method == "auto"

    def test_split_plot_validation_n_whole_plots_lt2_raises(self):
        """CR-28: n_whole_plots < 2 is rejected by Pydantic validation."""
        from api_server.models.common import SplitPlotOptionsModel
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SplitPlotOptionsModel(htc_factors=["A"], n_whole_plots=1)

    def test_split_plot_validation_empty_htc_factors_raises(self):
        """CR-28: empty htc_factors list is rejected by Pydantic validation."""
        from api_server.models.common import SplitPlotOptionsModel
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SplitPlotOptionsModel(htc_factors=[], n_whole_plots=3)

    def test_split_plot_validation_negative_eta_raises(self):
        """CR-28: eta < 0 is rejected by Pydantic validation."""
        from api_server.models.common import SplitPlotOptionsModel
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SplitPlotOptionsModel(htc_factors=["A"], n_whole_plots=3, eta=-0.1)


class TestSerializeDesignResult:
    def _make_result(self):
        return {
            "design_df": pd.DataFrame({"A": [0.1, -0.1], "B": [1.0, -1.0]}),
            "buckets_df": pd.DataFrame({"A": [0.1], "count": [1]}),
            "report": {
                "n": 10, "p": 3, "df_num": 2, "df_denom": 7,
                "alpha": 0.05, "target_power": 0.8, "achieved_power": 0.85,
                "noncentrality_lambda": 8.0, "criterion": "I",
                "elapsed_sec": 1.23, "warnings": [],
                "diagnostics": {"i_criterion": 0.4, "d_efficiency": 0.9},
                "_X": np.eye(3),  # internal key — must be stripped
            },
        }

    def test_design_df_is_list_of_dicts(self):
        result = serialize_design_result(self._make_result())
        assert isinstance(result["design_df"], list)
        assert isinstance(result["design_df"][0], dict)

    def test_internal_X_key_stripped(self):
        result = serialize_design_result(self._make_result())
        assert "_X" not in result["report"]

    def test_numpy_scalars_sanitized(self):
        r = self._make_result()
        r["report"]["noncentrality_lambda"] = np.float64(8.0)
        result = serialize_design_result(r)
        v = result["report"]["noncentrality_lambda"]
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Layer 2 — HTTP unit tests (mock or trivial, ASGI test client)
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    if not _HAS_SERVER:
        pytest.skip("fastapi/httpx not installed")
    from api_server.main import create_app
    return create_app()


@pytest.fixture
async def client(app):
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        r = await client.get("/health")
        assert r.status_code == 200

    async def test_health_body_has_status_ok(self, client):
        r = await client.get("/health")
        assert r.json()["status"] == "ok"

    async def test_health_body_has_version(self, client):
        r = await client.get("/health")
        assert "version" in r.json()


@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestOpenAPI:
    async def test_openapi_json_accessible(self, client):
        r = await client.get("/openapi.json")
        assert r.status_code == 200
        assert "paths" in r.json()

    async def test_docs_accessible(self, client):
        r = await client.get("/docs")
        assert r.status_code == 200

    async def test_redoc_accessible(self, client):
        r = await client.get("/redoc")
        assert r.status_code == 200


@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestErrorHandling:
    async def test_404_unknown_path(self, client):
        r = await client.get("/nonexistent")
        assert r.status_code == 404

    async def test_422_missing_formula(self, client):
        r = await client.post("/design", json={
            "factors": {"A": [-1, 1]},
            "power_cfg": {"type": "r2", "r2_target": 0.15},
        })
        assert r.status_code == 422

    async def test_422_missing_factors(self, client):
        r = await client.post("/design", json={
            "formula": "~ 1 + A",
            "power_cfg": {"type": "r2", "r2_target": 0.15},
        })
        assert r.status_code == 422

    async def test_422_bad_power_cfg_discriminator(self, client):
        r = await client.post("/design", json={
            "formula": "~ 1 + A",
            "factors": {"A": [-1, 1]},
            "power_cfg": {"type": "unknown"},
        })
        assert r.status_code == 422

    async def test_422_r2_target_out_of_range(self, client):
        r = await client.post("/design", json={
            "formula": "~ 1 + A",
            "factors": {"A": [-1, 1]},
            "power_cfg": {"type": "r2", "r2_target": 1.5},  # > 1
        })
        assert r.status_code == 422

    async def test_discriminator_selects_r2_model(self, client):
        """Sending type='r2' must not be confused with contrast."""
        body = {
            "formula": "~ 1 + A",
            "factors": {"A": [-1, 1]},
            "power_cfg": {"type": "r2", "r2_target": 0.15},
        }
        # We just check schema validation passes (422 only if bad schema)
        # Actual compute error (ValueError from library) → 422 is also acceptable here
        r = await client.post("/design", json=body)
        assert r.status_code in (200, 422)  # 422 only for compute/library errors


@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestDesignEndpointMocked:
    @patch("api_server.routers.design.find_optimal_design")
    async def test_design_returns_200_with_mock(self, mock_run, client):
        mock_run.return_value = {
            "design_df": pd.DataFrame({"A": [0.1, -0.1], "B": [1.0, -1.0]}),
            "buckets_df": pd.DataFrame({"A": [0.1], "count": [1]}),
            "report": {
                "n": 5, "p": 3, "df_num": 2, "df_denom": 2,
                "alpha": 0.05, "target_power": 0.8, "achieved_power": 0.85,
                "noncentrality_lambda": 8.0, "criterion": "I",
                "elapsed_sec": 0.5, "warnings": [],
                "diagnostics": {"i_criterion": 0.4, "d_efficiency": 0.9,
                                "condition_number": 2.0},
            },
        }
        r = await client.post("/design", json={
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "power_cfg": {"type": "r2", "r2_target": 0.15},
        })
        assert r.status_code == 200
        body = r.json()
        assert "design_df" in body
        assert "report" in body
        assert body["report"]["n"] == 5

    @patch("api_server.routers.design.find_optimal_design")
    async def test_design_ValueError_returns_422(self, mock_run, client):
        mock_run.side_effect = ValueError("bad formula")
        r = await client.post("/design", json={
            "formula": "~ 1 + A",
            "factors": {"A": [-1.0, 1.0]},
            "power_cfg": {"type": "r2", "r2_target": 0.15},
        })
        assert r.status_code == 422
        assert "bad formula" in r.json()["detail"]

    @patch("api_server.routers.design.find_optimal_design")
    async def test_design_contrast_mode(self, mock_run, client):
        mock_run.return_value = {
            "design_df": pd.DataFrame({"A": [0.1], "B": [-0.1]}),
            "buckets_df": pd.DataFrame({"A": [0.1], "count": [1]}),
            "report": {
                "n": 8, "p": 3, "df_num": 1, "df_denom": 5,
                "alpha": 0.05, "target_power": 0.8, "achieved_power": 0.82,
                "noncentrality_lambda": 6.0, "criterion": "I",
                "elapsed_sec": 0.5, "warnings": [],
            },
        }
        r = await client.post("/design", json={
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "power_cfg": {
                "type": "contrast",
                "L": [[0, 1, 0]],
                "delta": [0.5],
                "sigma": 1.0,
            },
        })
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Layer 3 — HTTP integration (real compute, small problems, @slow)
# ---------------------------------------------------------------------------

_FAST_OPTS: Dict[str, Any] = {
    "candidate_points": 100,
    "auto_candidate": False,
    "starts": 2,
    "max_iter": 50,
    "random_state": 42,
}

_SIMPLE_BODY: Dict[str, Any] = {
    "formula": "~ 1 + A + B",
    "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
    "power_cfg": {"type": "r2", "r2_target": 0.15, "max_n": 30},
    "design_opts": _FAST_OPTS,
}


@pytest.mark.slow
@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestDesignIntegration:
    async def test_post_design_r2_returns_valid_response(self, client):
        r = await client.post("/design", json=_SIMPLE_BODY)
        assert r.status_code == 200
        body = r.json()
        assert len(body["design_df"]) >= 1
        assert body["report"]["n"] >= 1
        assert 0 < body["report"]["achieved_power"] <= 1.0

    async def test_post_design_response_design_df_has_factor_columns(self, client):
        r = await client.post("/design", json=_SIMPLE_BODY)
        assert r.status_code == 200
        row = r.json()["design_df"][0]
        assert "A" in row and "B" in row

    async def test_post_design_constraint_expr(self, client):
        body = {**_SIMPLE_BODY, "design_opts": {**_FAST_OPTS, "constraint_expr": "A + B <= 1"}}
        r = await client.post("/design", json=body)
        assert r.status_code == 200

    async def test_post_design_report_no_nan_json(self, client):
        """Verify no JSON NaN/Inf in response (would be invalid JSON)."""
        r = await client.post("/design", json=_SIMPLE_BODY)
        # httpx parses JSON; if NaN were present, it would raise
        assert r.status_code == 200
        body = r.json()
        # spot-check key numeric fields
        assert math.isfinite(body["report"]["achieved_power"])
        assert math.isfinite(body["report"]["noncentrality_lambda"])


@pytest.mark.slow
@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestPowerCurveIntegration:
    async def test_post_power_curve_by_n_returns_rows(self, client):
        body = {**_SIMPLE_BODY, "n_points": 5, "n_range": [5, 25]}
        r = await client.post("/power_curve/by_n", json=body)
        assert r.status_code == 200
        body_json = r.json()
        assert len(body_json["rows"]) >= 1
        assert "columns" in body_json

    async def test_post_power_curve_by_effect_returns_rows(self, client):
        body = {
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "n": 15,
            "power_cfg": {"type": "r2", "r2_target": 0.15},
            "design_opts": _FAST_OPTS,
        }
        r = await client.post("/power_curve/by_effect", json=body)
        assert r.status_code == 200
        assert len(r.json()["rows"]) >= 1


@pytest.mark.slow
@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestAugmentIntegration:
    async def test_post_augment_increases_design_size(self, client):
        # First generate a small design
        r_design = await client.post("/design", json=_SIMPLE_BODY)
        assert r_design.status_code == 200
        design_df = r_design.json()["design_df"]

        # Then augment it
        r_aug = await client.post("/augment", json={
            "design_df": design_df,
            "m": 2,
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "design_opts": _FAST_OPTS,
        })
        assert r_aug.status_code == 200
        body = r_aug.json()
        assert body["n_added"] == 2
        assert body["n_total"] == body["n_original"] + 2
        assert len(body["augmented_df"]) == body["n_total"]


# ---------------------------------------------------------------------------
# MR-9 — Multi-response endpoint tests
# ---------------------------------------------------------------------------

_MR_FAST_OPTS: Dict[str, Any] = {
    "candidate_points": 100,
    "auto_candidate": False,
    "starts": 2,
    "max_iter": 50,
    "random_state": 42,
}

_MR_TWO_RESPONSES = [
    {"name": "Y1", "power_cfg": {"type": "r2", "r2_target": 0.15, "max_n": 30}},
    {"name": "Y2", "power_cfg": {"type": "r2", "r2_target": 0.20, "max_n": 30}},
]

_MR_SIMPLE_BODY: Dict[str, Any] = {
    "formula": "~ 1 + A + B",
    "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
    "multi_cfg": {
        "responses": _MR_TWO_RESPONSES,
        "power_combination": "min",
    },
    "design_opts": _MR_FAST_OPTS,
}


class TestMultiResponseModels:
    """Unit tests for MR-9 Pydantic models (no HTTP, no server needed)."""

    def test_response_spec_model_basic(self):
        from api_server.models.common import ResponseSpecModel, PowerR2ConfigModel
        r = ResponseSpecModel(
            name="Y1",
            power_cfg=PowerR2ConfigModel(type="r2", r2_target=0.15),
        )
        assert r.name == "Y1"
        assert r.weight == 1.0
        assert r.formula is None

    def test_response_spec_model_weight_gt0_required(self):
        from api_server.models.common import ResponseSpecModel
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ResponseSpecModel(
                name="Y1",
                power_cfg={"type": "r2", "r2_target": 0.15},
                weight=0.0,
            )

    def test_response_spec_model_empty_name_rejected(self):
        from api_server.models.common import ResponseSpecModel
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ResponseSpecModel(
                name="",
                power_cfg={"type": "r2", "r2_target": 0.15},
            )

    def test_multi_response_options_model_min_two_responses(self):
        from api_server.models.common import MultiResponseOptionsModel
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            MultiResponseOptionsModel(
                responses=[
                    {"name": "Y1", "power_cfg": {"type": "r2", "r2_target": 0.15}},
                ],
            )

    def test_multi_response_options_model_invalid_combination_rule(self):
        from api_server.models.common import MultiResponseOptionsModel
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            MultiResponseOptionsModel(
                responses=_MR_TWO_RESPONSES,
                power_combination="invalid_rule",
            )

    def test_multi_response_options_sigma_joint_accepted(self):
        from api_server.models.common import MultiResponseOptionsModel
        model = MultiResponseOptionsModel(
            responses=_MR_TWO_RESPONSES,
            sigma_joint=[[1.0, 0.3], [0.3, 1.0]],
        )
        assert model.sigma_joint == [[1.0, 0.3], [0.3, 1.0]]

    def test_pydantic_multi_cfg_to_dataclass_returns_MultiResponseOptions(self):
        from api_server.models.common import MultiResponseOptionsModel
        from api_server.serialization import pydantic_multi_cfg_to_dataclass
        from lattice_doe.config import MultiResponseOptions
        model = MultiResponseOptionsModel(responses=_MR_TWO_RESPONSES)
        result = pydantic_multi_cfg_to_dataclass(model)
        assert isinstance(result, MultiResponseOptions)
        assert len(result.responses) == 2

    def test_pydantic_multi_cfg_response_names_propagated(self):
        from api_server.models.common import MultiResponseOptionsModel
        from api_server.serialization import pydantic_multi_cfg_to_dataclass
        model = MultiResponseOptionsModel(responses=_MR_TWO_RESPONSES)
        result = pydantic_multi_cfg_to_dataclass(model)
        assert result.responses[0].name == "Y1"
        assert result.responses[1].name == "Y2"

    def test_pydantic_multi_cfg_sigma_joint_is_ndarray(self):
        from api_server.models.common import MultiResponseOptionsModel
        from api_server.serialization import pydantic_multi_cfg_to_dataclass
        model = MultiResponseOptionsModel(
            responses=_MR_TWO_RESPONSES,
            sigma_joint=[[1.0, 0.5], [0.5, 1.0]],
        )
        result = pydantic_multi_cfg_to_dataclass(model)
        assert isinstance(result.sigma_joint, np.ndarray)
        assert result.sigma_joint.shape == (2, 2)

    def test_serialize_multiresponse_result_keys(self):
        from api_server.serialization import serialize_multiresponse_result
        fake_result = {
            "design_df": pd.DataFrame({"A": [0.1, -0.1], "B": [1.0, -1.0]}),
            "buckets_df": pd.DataFrame({"A": [0.1], "count": [2]}),
            "report": {
                "n": 10,
                "achieved_power": 0.85,
                "responses": [{"name": "Y1", "power": 0.87}, {"name": "Y2", "power": 0.85}],
                "combination_rule": "min",
                "compound_criterion": False,
                "elapsed_sec": 1.23,
                "warnings": [],
            },
        }
        out = serialize_multiresponse_result(fake_result)
        assert isinstance(out["design_df"], list)
        assert isinstance(out["buckets_df"], list)
        assert out["report"]["n"] == 10
        assert out["report"]["combination_rule"] == "min"
        assert not out["report"]["compound_criterion"]
        assert len(out["report"]["responses"]) == 2

    def test_serialize_multiresponse_result_numpy_sanitized(self):
        from api_server.serialization import serialize_multiresponse_result
        fake_result = {
            "design_df": pd.DataFrame({"A": [0.1]}),
            "buckets_df": pd.DataFrame({"A": [0.1]}),
            "report": {
                "n": np.int64(8),
                "achieved_power": np.float64(0.82),
                "responses": [{"name": "Y1", "power": np.float64(0.82)}],
                "combination_rule": "min",
                "compound_criterion": False,
                "elapsed_sec": None,
                "warnings": [],
            },
        }
        out = serialize_multiresponse_result(fake_result)
        assert isinstance(out["report"]["n"], int)
        # achieved_power sanitized to plain float
        ap = out["report"]["achieved_power"]
        assert ap is None or isinstance(ap, float)


@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestMultiResponseEndpoint:
    """MR-9 HTTP tests for POST /multiresponse_design."""

    @patch("api_server.routers.design.find_multiresponse_design")
    async def test_multiresponse_returns_200_with_mock(self, mock_run, client):
        mock_run.return_value = {
            "design_df": pd.DataFrame({"A": [0.1, -0.1], "B": [1.0, -1.0]}),
            "buckets_df": pd.DataFrame({"A": [0.1], "count": [2]}),
            "report": {
                "n": 10,
                "achieved_power": 0.85,
                "responses": [
                    {"name": "Y1", "power": 0.87},
                    {"name": "Y2", "power": 0.85},
                ],
                "combination_rule": "min",
                "compound_criterion": False,
                "elapsed_sec": 0.5,
                "warnings": [],
            },
        }
        r = await client.post("/multiresponse_design", json=_MR_SIMPLE_BODY)
        assert r.status_code == 200
        body = r.json()
        assert "design_df" in body
        assert body["report"]["n"] == 10
        assert len(body["report"]["responses"]) == 2

    @patch("api_server.routers.design.find_multiresponse_design")
    async def test_multiresponse_responses_count_matches_request(self, mock_run, client):
        mock_run.return_value = {
            "design_df": pd.DataFrame({"A": [0.1]}),
            "buckets_df": pd.DataFrame({"A": [0.1]}),
            "report": {
                "n": 5,
                "achieved_power": 0.80,
                "responses": [
                    {"name": "Y1", "power": 0.82},
                    {"name": "Y2", "power": 0.80},
                ],
                "combination_rule": "min",
                "compound_criterion": False,
                "elapsed_sec": 0.2,
                "warnings": [],
            },
        }
        r = await client.post("/multiresponse_design", json=_MR_SIMPLE_BODY)
        assert r.status_code == 200
        body = r.json()
        assert len(body["report"]["responses"]) == len(_MR_TWO_RESPONSES)

    @patch("api_server.routers.design.find_multiresponse_design")
    async def test_multiresponse_combination_rule_in_response(self, mock_run, client):
        mock_run.return_value = {
            "design_df": pd.DataFrame({"A": [0.1]}),
            "buckets_df": pd.DataFrame({"A": [0.1]}),
            "report": {
                "n": 5,
                "achieved_power": 0.80,
                "responses": [{"name": "Y1", "power": 0.80}, {"name": "Y2", "power": 0.80}],
                "combination_rule": "min",
                "compound_criterion": False,
                "elapsed_sec": 0.1,
                "warnings": [],
            },
        }
        r = await client.post("/multiresponse_design", json=_MR_SIMPLE_BODY)
        assert r.status_code == 200
        assert r.json()["report"]["combination_rule"] == "min"

    async def test_422_missing_multi_cfg(self, client):
        r = await client.post("/multiresponse_design", json={
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
        })
        assert r.status_code == 422

    async def test_422_only_one_response(self, client):
        r = await client.post("/multiresponse_design", json={
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "multi_cfg": {
                "responses": [
                    {"name": "Y1", "power_cfg": {"type": "r2", "r2_target": 0.15}},
                ],
            },
        })
        assert r.status_code == 422

    async def test_422_invalid_power_combination(self, client):
        r = await client.post("/multiresponse_design", json={
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "multi_cfg": {
                "responses": _MR_TWO_RESPONSES,
                "power_combination": "nonsense",
            },
        })
        assert r.status_code == 422

    @patch("api_server.routers.design.find_multiresponse_design")
    async def test_422_ValueError_from_library(self, mock_run, client):
        mock_run.side_effect = ValueError("incompatible formulas")
        r = await client.post("/multiresponse_design", json=_MR_SIMPLE_BODY)
        assert r.status_code == 422
        assert "incompatible formulas" in r.json()["detail"]

    @patch("api_server.routers.design.find_multiresponse_design")
    async def test_sigma_joint_roundtrips(self, mock_run, client):
        """sigma_joint list-of-lists passes through serialization without error."""
        mock_run.return_value = {
            "design_df": pd.DataFrame({"A": [0.1]}),
            "buckets_df": pd.DataFrame({"A": [0.1]}),
            "report": {
                "n": 5,
                "achieved_power": 0.82,
                "responses": [{"name": "Y1", "power": 0.82}, {"name": "Y2", "power": 0.82}],
                "combination_rule": "min",
                "compound_criterion": False,
                "elapsed_sec": 0.1,
                "warnings": [],
            },
        }
        body = {
            **_MR_SIMPLE_BODY,
            "multi_cfg": {
                **_MR_SIMPLE_BODY["multi_cfg"],
                "sigma_joint": [[1.0, 0.3], [0.3, 1.0]],
            },
        }
        r = await client.post("/multiresponse_design", json=body)
        assert r.status_code == 200
        # sigma_joint is passed to library — verify it reached the mock as ndarray
        call_args = mock_run.call_args
        passed_multi_cfg = call_args.kwargs.get("multi_cfg") or call_args.args[2]
        assert passed_multi_cfg.sigma_joint is not None
        assert passed_multi_cfg.sigma_joint.shape == (2, 2)

    @patch("api_server.routers.design.find_multiresponse_design")
    async def test_contrast_response_accepted(self, mock_run, client):
        mock_run.return_value = {
            "design_df": pd.DataFrame({"A": [0.1], "B": [-0.1]}),
            "buckets_df": pd.DataFrame({"A": [0.1]}),
            "report": {
                "n": 8,
                "achieved_power": 0.81,
                "responses": [{"name": "Y1", "power": 0.84}, {"name": "Y2", "power": 0.81}],
                "combination_rule": "min",
                "compound_criterion": False,
                "elapsed_sec": 0.3,
                "warnings": [],
            },
        }
        body = {
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "multi_cfg": {
                "responses": [
                    {
                        "name": "Y1",
                        "power_cfg": {
                            "type": "contrast",
                            "L": [[0, 1, 0]],
                            "delta": [0.5],
                            "sigma": 1.0,
                        },
                    },
                    {
                        "name": "Y2",
                        "power_cfg": {"type": "r2", "r2_target": 0.15, "max_n": 30},
                    },
                ],
                "power_combination": "min",
            },
            "design_opts": _MR_FAST_OPTS,
        }
        r = await client.post("/multiresponse_design", json=body)
        assert r.status_code == 200

    @patch("api_server.routers.design.find_multiresponse_design")
    async def test_health_not_broken_by_mr9(self, mock_run, client):
        """GET /health still returns 200 after MR-9 is registered."""
        r = await client.get("/health")
        assert r.status_code == 200


@pytest.mark.slow
@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestMultiResponseIntegration:
    """MR-9 real-compute integration test (marked @slow)."""

    async def test_post_multiresponse_design_returns_valid_response(self, client):
        r = await client.post("/multiresponse_design", json=_MR_SIMPLE_BODY)
        assert r.status_code == 200
        body = r.json()
        assert len(body["design_df"]) >= 1
        assert body["report"]["n"] >= 1
        assert 0 < body["report"]["achieved_power"] <= 1.0
        assert len(body["report"]["responses"]) == 2
        assert body["report"]["combination_rule"] == "min"
        assert "compound_criterion" in body["report"]

    async def test_post_multiresponse_design_has_factor_columns(self, client):
        r = await client.post("/multiresponse_design", json=_MR_SIMPLE_BODY)
        assert r.status_code == 200
        row = r.json()["design_df"][0]
        assert "A" in row and "B" in row

    async def test_post_multiresponse_design_no_nan_json(self, client):
        r = await client.post("/multiresponse_design", json=_MR_SIMPLE_BODY)
        assert r.status_code == 200
        body = r.json()
        assert math.isfinite(body["report"]["achieved_power"])
        assert isinstance(body["report"]["n"], int)


# ---------------------------------------------------------------------------
# GL-7 — REST API GLM support
# ---------------------------------------------------------------------------

_GLM_FAST_OPTS: Dict[str, Any] = {
    "candidate_points": 100,
    "auto_candidate": False,
    "starts": 2,
    "max_iter": 50,
    "random_state": 42,
}

_GLM_BINOMIAL_CFG: Dict[str, Any] = {
    "type": "glm_contrast",
    "L": [[0, 1]],
    "delta": [0.4],
    "family": "binomial",
    "baseline": 0.30,
}

_GLM_POISSON_CFG: Dict[str, Any] = {
    "type": "glm_contrast",
    "L": [[0, 1]],
    "delta": [0.5],
    "family": "poisson",
    "baseline": 2.0,
}

_GLM_MOCK_REPORT: Dict[str, Any] = {
    "n": 12, "p": 2, "df_num": 1, "df_denom": 10,
    "alpha": 0.05, "target_power": 0.8, "achieved_power": 0.83,
    "noncentrality_lambda": 7.5, "criterion": "I",
    "elapsed_sec": 0.4, "warnings": [],
    "test_type": "wald_chi2",
    "family": "binomial", "link": "logit", "baseline": 0.30,
    "glm_weight": 0.21, "df2": None,
}


# --- Layer 1: pure unit tests (no HTTP) ---

class TestGLMPydanticModels:
    """GL-7 Layer 1: PowerGLMContrastModel construction and validation."""

    def test_glm_model_construction_binomial(self):
        from api_server.models.common import PowerGLMContrastModel
        m = PowerGLMContrastModel(L=[[0, 1]], delta=[0.4], baseline=0.3, family="binomial")
        assert m.type == "glm_contrast"
        assert m.family == "binomial"
        assert m.baseline == pytest.approx(0.3)

    def test_glm_model_construction_poisson(self):
        from api_server.models.common import PowerGLMContrastModel
        m = PowerGLMContrastModel(L=[[0, 1]], delta=[0.5], baseline=2.0, family="poisson")
        assert m.family == "poisson"
        assert m.baseline == pytest.approx(2.0)

    def test_glm_baseline_out_of_range_binomial(self):
        from api_server.models.common import PowerGLMContrastModel
        import pydantic
        with pytest.raises(pydantic.ValidationError, match="baseline"):
            PowerGLMContrastModel(L=[[0, 1]], delta=[0.4], baseline=1.5, family="binomial")

    def test_glm_baseline_zero_poisson_rejected(self):
        from api_server.models.common import PowerGLMContrastModel
        import pydantic
        with pytest.raises(pydantic.ValidationError, match="baseline"):
            PowerGLMContrastModel(L=[[0, 1]], delta=[0.5], baseline=0.0, family="poisson")

    def test_glm_wrong_family_rejected(self):
        from api_server.models.common import PowerGLMContrastModel
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            PowerGLMContrastModel(L=[[0, 1]], delta=[0.4], baseline=0.3, family="gaussian")

    def test_glm_missing_baseline_rejected(self):
        from api_server.models.common import PowerGLMContrastModel
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            PowerGLMContrastModel(L=[[0, 1]], delta=[0.4], family="binomial")

    def test_pydantic_power_cfg_to_dataclass_glm_branch(self):
        from api_server.models.common import PowerGLMContrastModel
        from lattice_doe.config import PowerGLMContrastConfig
        m = PowerGLMContrastModel(L=[[0, 1]], delta=[0.4], baseline=0.3, family="binomial")
        cfg = pydantic_power_cfg_to_dataclass(m)
        assert isinstance(cfg, PowerGLMContrastConfig)
        assert cfg.family == "binomial"
        assert cfg.baseline == pytest.approx(0.3)
        assert cfg.L.shape == (1, 2)

    def test_pydantic_power_cfg_to_dataclass_glm_link_forwarded(self):
        from api_server.models.common import PowerGLMContrastModel
        m = PowerGLMContrastModel(L=[[0, 1]], delta=[0.4], baseline=0.3,
                                   family="binomial", link="logit")
        cfg = pydantic_power_cfg_to_dataclass(m)
        assert cfg.link == "logit"

    def test_glm_type_discriminator_in_union(self):
        """PowerCfgModel discriminates glm_contrast correctly."""
        from api_server.models.common import PowerGLMContrastModel
        from pydantic import TypeAdapter
        from api_server.models.common import PowerCfgModel
        ta = TypeAdapter(PowerCfgModel)
        m = ta.validate_python({
            "type": "glm_contrast",
            "L": [[0, 1]], "delta": [0.3], "baseline": 0.25, "family": "binomial",
        })
        assert isinstance(m, PowerGLMContrastModel)


# --- Layer 2: HTTP unit tests (mocked, ASGI test client) ---

@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestGLMDesignEndpointMocked:
    """GL-7 Layer 2: mocked HTTP tests for /design with GLM power config."""

    def _glm_body(self, cfg=None):
        return {
            "formula": "~ 1 + A",
            "factors": {"A": [-1.0, 1.0]},
            "power_cfg": cfg or _GLM_BINOMIAL_CFG,
            "design_opts": _GLM_FAST_OPTS,
        }

    def _mock_return(self, extra=None):
        r = {
            "design_df": pd.DataFrame({"A": [0.1, -0.1]}),
            "buckets_df": pd.DataFrame({"A": [0.1], "count": [2]}),
            "report": dict(_GLM_MOCK_REPORT),
        }
        if extra:
            r["report"].update(extra)
        return r

    @patch("api_server.routers.design.find_optimal_design")
    async def test_glm_binomial_returns_200(self, mock_run, client):
        mock_run.return_value = self._mock_return()
        r = await client.post("/design", json=self._glm_body())
        assert r.status_code == 200

    @patch("api_server.routers.design.find_optimal_design")
    async def test_glm_binomial_response_has_design_df(self, mock_run, client):
        mock_run.return_value = self._mock_return()
        r = await client.post("/design", json=self._glm_body())
        assert "design_df" in r.json()
        assert len(r.json()["design_df"]) >= 1

    @patch("api_server.routers.design.find_optimal_design")
    async def test_glm_binomial_report_has_family(self, mock_run, client):
        mock_run.return_value = self._mock_return()
        r = await client.post("/design", json=self._glm_body())
        assert r.json()["report"]["family"] == "binomial"

    @patch("api_server.routers.design.find_optimal_design")
    async def test_glm_binomial_report_test_type_wald_chi2(self, mock_run, client):
        mock_run.return_value = self._mock_return()
        r = await client.post("/design", json=self._glm_body())
        assert r.json()["report"]["test_type"] == "wald_chi2"

    @patch("api_server.routers.design.find_optimal_design")
    async def test_glm_binomial_report_df2_is_none(self, mock_run, client):
        mock_run.return_value = self._mock_return()
        r = await client.post("/design", json=self._glm_body())
        assert r.json()["report"]["df2"] is None

    @patch("api_server.routers.design.find_optimal_design")
    async def test_glm_poisson_returns_200(self, mock_run, client):
        mock_run.return_value = self._mock_return({
            "family": "poisson", "link": "log", "baseline": 2.0, "glm_weight": 2.0,
        })
        r = await client.post("/design", json=self._glm_body(_GLM_POISSON_CFG))
        assert r.status_code == 200

    @patch("api_server.routers.design.find_optimal_design")
    async def test_glm_poisson_report_has_baseline(self, mock_run, client):
        mock_run.return_value = self._mock_return({
            "family": "poisson", "link": "log", "baseline": 2.0, "glm_weight": 2.0,
        })
        r = await client.post("/design", json=self._glm_body(_GLM_POISSON_CFG))
        assert r.json()["report"]["baseline"] == pytest.approx(2.0)

    async def test_glm_baseline_out_of_range_returns_422(self, client):
        body = self._glm_body({
            "type": "glm_contrast",
            "L": [[0, 1]], "delta": [0.4],
            "family": "binomial",
            "baseline": 1.5,  # invalid: > 1
        })
        r = await client.post("/design", json=body)
        assert r.status_code == 422

    async def test_glm_missing_baseline_returns_422(self, client):
        body = self._glm_body({
            "type": "glm_contrast",
            "L": [[0, 1]], "delta": [0.4],
            "family": "binomial",
            # baseline omitted — required field
        })
        r = await client.post("/design", json=body)
        assert r.status_code == 422

    @patch("api_server.routers.design.find_optimal_design")
    async def test_ols_contrast_endpoint_unchanged(self, mock_run, client):
        mock_run.return_value = {
            "design_df": pd.DataFrame({"A": [0.1, -0.1]}),
            "buckets_df": pd.DataFrame({"A": [0.1], "count": [1]}),
            "report": {
                "n": 10, "p": 2, "df_num": 1, "df_denom": 8,
                "alpha": 0.05, "target_power": 0.8, "achieved_power": 0.82,
                "noncentrality_lambda": 6.5, "criterion": "I",
                "elapsed_sec": 0.3, "warnings": [],
            },
        }
        r = await client.post("/design", json={
            "formula": "~ 1 + A",
            "factors": {"A": [-1.0, 1.0]},
            "power_cfg": {"type": "contrast", "L": [[0, 1]], "delta": [0.5], "sigma": 1.0},
            "design_opts": _GLM_FAST_OPTS,
        })
        assert r.status_code == 200


# --- Layer 3: HTTP integration tests (real compute, @slow) ---

_GLM_INTEGRATION_BODY: Dict[str, Any] = {
    "formula": "~ 1 + A",
    "factors": {"A": [-1.0, 1.0]},
    "power_cfg": {
        "type": "glm_contrast",
        "L": [[0, 1]],
        "delta": [0.6],
        "family": "binomial",
        "baseline": 0.30,
        "max_n": 40,
    },
    "design_opts": _GLM_FAST_OPTS,
}


@pytest.mark.slow
@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestGLMDesignIntegration:
    """GL-7 Layer 3: real compute round-trip for GLM designs."""

    async def test_glm_result_json_parseable(self, client):
        r = await client.post("/design", json=_GLM_INTEGRATION_BODY)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, dict)

    async def test_glm_design_has_factor_columns(self, client):
        r = await client.post("/design", json=_GLM_INTEGRATION_BODY)
        assert r.status_code == 200
        row = r.json()["design_df"][0]
        assert "A" in row

    async def test_glm_report_achieved_power_between_0_and_1(self, client):
        r = await client.post("/design", json=_GLM_INTEGRATION_BODY)
        assert r.status_code == 200
        ap = r.json()["report"]["achieved_power"]
        assert 0 < ap <= 1.0

    async def test_glm_no_nan_in_json(self, client):
        r = await client.post("/design", json=_GLM_INTEGRATION_BODY)
        assert r.status_code == 200
        body = r.json()
        assert math.isfinite(body["report"]["achieved_power"])
        assert math.isfinite(body["report"]["noncentrality_lambda"])

    async def test_ols_r2_endpoint_unchanged(self, client):
        """OLS R² path still works after GL-7 changes."""
        r = await client.post("/design", json={
            "formula": "~ 1 + A",
            "factors": {"A": [-1.0, 1.0]},
            "power_cfg": {"type": "r2", "r2_target": 0.15, "max_n": 30},
            "design_opts": _GLM_FAST_OPTS,
        })
        assert r.status_code == 200
        assert 0 < r.json()["report"]["achieved_power"] <= 1.0


@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestUX4StrictRequestModels:
    """UX-4 regression: request models silently discarded unknown fields
    (misspellings like 'strats') and the documented 'workers' option was not
    modeled at all. Request models now forbid extras and model workers
    explicitly."""

    _BODY = {
        "formula": "~ 1 + A + B",
        "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
        "power_cfg": {"type": "contrast", "L": [[0.0, 1.0, 0.0]],
                      "delta": [1.5], "sigma": 1.0, "alpha": 0.05,
                      "power": 0.8, "max_n": 40},
        "design_opts": {"random_state": 0, "starts": 1,
                        "candidate_points": 100},
    }

    async def test_misspelled_option_422(self, client):
        import copy
        body = copy.deepcopy(self._BODY)
        body["design_opts"]["strats"] = 3
        r = await client.post("/design", json=body)
        assert r.status_code == 422
        assert "strats" in str(r.json())

    async def test_unknown_top_level_422(self, client):
        import copy
        body = copy.deepcopy(self._BODY)
        body["bogus"] = True
        r = await client.post("/design", json=body)
        assert r.status_code == 422

    async def test_workers_gt1_422_with_guidance(self, client):
        import copy
        body = copy.deepcopy(self._BODY)
        body["design_opts"]["workers"] = 4
        r = await client.post("/design", json=body)
        assert r.status_code == 422
        assert "--workers" in str(r.json())

    async def test_workers_serial_accepted(self, client):
        import copy
        body = copy.deepcopy(self._BODY)
        body["design_opts"]["workers"] = 1
        r = await client.post("/design", json=body)
        assert r.status_code == 200

    @pytest.mark.parametrize("bad", [0, -1, -5])
    async def test_workers_zero_or_negative_422(self, client, bad):
        """Only null or 1 are valid; 0 and negatives must 422 rather than being
        silently coerced to None (P2)."""
        import copy
        body = copy.deepcopy(self._BODY)
        body["design_opts"]["workers"] = bad
        r = await client.post("/design", json=body)
        assert r.status_code == 422


@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestUX7RestStatusFields:
    """UX-7: the REST report must expose the structured search outcome."""

    async def test_partial_status_in_report(self, client):
        body = {
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "power_cfg": {"type": "contrast", "L": [[0.0, 1.0, 0.0]],
                          "delta": [0.3], "sigma": 1.0, "alpha": 0.05,
                          "power": 0.8, "max_n": 30},
            "design_opts": {"random_state": 0, "starts": 1,
                            "candidate_points": 100},
        }
        r = await client.post("/design", json=body)
        assert r.status_code == 200
        rep = r.json()["report"]
        assert rep["status"] == "partial"
        assert rep["target_met"] is False
        assert rep["termination_reason"] in ("max_n", "max_iter",
                                             "candidate_cap")


@pytest.mark.anyio
@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
class TestUX2JobsRouter:
    """UX-2: asynchronous design jobs — submit/poll/cancel/capacity/SSE."""

    _BODY = {
        "formula": "~ 1 + A + B",
        "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
        "power_cfg": {"type": "contrast", "L": [[0.0, 1.0, 0.0]],
                      "delta": [1.2], "sigma": 1.0, "alpha": 0.05,
                      "power": 0.8, "max_n": 60},
        "design_opts": {"random_state": 0, "starts": 1,
                        "candidate_points": 120},
    }

    async def _poll(self, client, jid, timeout=15.0):
        import time as _time

        import anyio
        deadline = _time.monotonic() + timeout
        terminal = {"done", "failed", "cancelled"}
        while _time.monotonic() < deadline:
            snap = (await client.get(f"/jobs/{jid}")).json()
            if snap["state"] in terminal:
                return snap
            await anyio.sleep(0.1)
        return (await client.get(f"/jobs/{jid}")).json()

    async def test_submit_poll_done(self, client):
        r = await client.post("/jobs/design", json=self._BODY)
        assert r.status_code == 202
        assert r.headers.get("Location") == f"/jobs/{r.json()['job_id']}"
        jid = r.json()["job_id"]
        snap = await self._poll(client, jid)
        assert snap["state"] == "done", snap.get("error")
        assert snap["result"]["report"]["n"] > 0
        assert snap["progress"]["phase"] == "done"

    async def test_multiresponse_submit_done(self, client):
        body = {
            "formula": "~ 1 + A + B",
            "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
            "multi_cfg": {
                "responses": [
                    {"name": "Y1", "power_cfg": {"type": "contrast",
                     "L": [[0.0, 1.0, 0.0]], "delta": [1.2], "sigma": 1.0,
                     "alpha": 0.05, "power": 0.8, "max_n": 60}},
                    {"name": "Y2", "power_cfg": {"type": "contrast",
                     "L": [[0.0, 1.0, 0.0]], "delta": [1.0], "sigma": 1.0,
                     "alpha": 0.05, "power": 0.8, "max_n": 60}},
                ],
                "power_combination": "min",
            },
            "design_opts": {"random_state": 0, "starts": 1,
                            "candidate_points": 120},
        }
        r = await client.post("/jobs/multiresponse_design", json=body)
        assert r.status_code == 202
        snap = await self._poll(client, r.json()["job_id"])
        assert snap["state"] == "done", snap.get("error")
        assert snap["result"]["report"]["n"] > 0

    async def test_unknown_job_404(self, client):
        assert (await client.get("/jobs/nope")).status_code == 404
        assert (await client.delete("/jobs/nope")).status_code == 404

    async def test_capacity_returns_503_with_retry_after(self, client, app):
        from api_server.jobs import JobsAtCapacity

        class _Full:
            max_concurrent = 2
            def submit(self, kind, runner):
                raise JobsAtCapacity(retry_after=9)

        original = app.state.job_manager
        app.state.job_manager = _Full()
        try:
            r = await client.post("/jobs/design", json=self._BODY)
            assert r.status_code == 503
            assert r.headers.get("Retry-After") == "9"
        finally:
            app.state.job_manager = original

    async def test_cancel_running_job(self, client):
        # A hard target with more work so the job stays running long enough
        # to be cancelled.
        body = dict(self._BODY)
        body["power_cfg"] = dict(self._BODY["power_cfg"], delta=[0.12],
                                 max_n=400)
        body["design_opts"] = dict(self._BODY["design_opts"],
                                   candidate_points=500, starts=3, max_iter=40)
        r = await client.post("/jobs/design", json=body)
        jid = r.json()["job_id"]
        # Wait until it is actually running.
        import anyio
        for _ in range(50):
            s = (await client.get(f"/jobs/{jid}")).json()
            if s["state"] == "running":
                break
            await anyio.sleep(0.05)
        assert (await client.delete(f"/jobs/{jid}")).status_code == 200
        snap = await self._poll(client, jid, timeout=20.0)
        assert snap["state"] == "cancelled", snap


@pytest.mark.skipif(not _HAS_SERVER, reason="fastapi/httpx not installed")
def test_ux2_sse_stream_reaches_terminal():
    """UX-2: GET /jobs/{id}/events streams SSE frames to a terminal state."""
    import json as _json

    from fastapi.testclient import TestClient
    from api_server.main import create_app

    client = TestClient(create_app())
    body = {
        "formula": "~ 1 + A + B",
        "factors": {"A": [-1.0, 1.0], "B": [-1.0, 1.0]},
        "power_cfg": {"type": "contrast", "L": [[0.0, 1.0, 0.0]],
                      "delta": [1.2], "sigma": 1.0, "alpha": 0.05,
                      "power": 0.8, "max_n": 60},
        "design_opts": {"random_state": 0, "starts": 1,
                        "candidate_points": 120},
    }
    jid = client.post("/jobs/design", json=body).json()["job_id"]
    frames = []
    with client.stream("GET", f"/jobs/{jid}/events") as r:
        assert "text/event-stream" in r.headers.get("content-type", "")
        for line in r.iter_lines():
            if line and line.startswith("data:"):
                frames.append(_json.loads(line[len("data: "):]))
            if len(frames) > 100:
                break
    assert frames
    assert frames[-1]["state"] in ("done", "failed", "cancelled")
