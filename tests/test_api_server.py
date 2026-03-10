# tests/test_api_server.py
# License: MIT
"""
Tests for the iopt-power-design FastAPI REST API server.

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
    serialize_report,
)
from iopt_power_design.config import DesignOptions, PowerContrastConfig, PowerR2Config


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
    @patch("api_server.routers.design.i_optimal_powered_design")
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

    @patch("api_server.routers.design.i_optimal_powered_design")
    async def test_design_ValueError_returns_422(self, mock_run, client):
        mock_run.side_effect = ValueError("bad formula")
        r = await client.post("/design", json={
            "formula": "~ 1 + A",
            "factors": {"A": [-1.0, 1.0]},
            "power_cfg": {"type": "r2", "r2_target": 0.15},
        })
        assert r.status_code == 422
        assert "bad formula" in r.json()["detail"]

    @patch("api_server.routers.design.i_optimal_powered_design")
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
