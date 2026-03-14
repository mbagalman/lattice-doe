# tests/test_cli.py
"""Unit tests for the CLI — GL-6: GLM YAML / CLI support."""
from __future__ import annotations

import json
import textwrap
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from iopt_power_design.cli import (
    _make_power_cfg,
    _validate_config_keys,
    _apply_glm_cli_args,
    _print_template,
    main,
)
from iopt_power_design.config import PowerGLMContrastConfig, PowerContrastConfig, PowerR2Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _glm_cfg_dict(family="binomial", baseline=0.3, link=None, sesoi=0.15):
    """Minimal in-memory config dict for a GLM design."""
    d = {
        "formula": "~ 1 + A",
        "factors": {"A": [0.0, 1.0]},
        "family": family,
        "baseline": baseline,
        "contrast": {
            "scenario_a": {"A": 0.0},
            "scenario_b": {"A": 1.0},
            "sesoi": sesoi,
        },
        "alpha": 0.05,
        "power": 0.80,
    }
    if link is not None:
        d["link"] = link
    return d


FORMULA = "~ 1 + A"
FACTORS = {"A": (0.0, 1.0)}


class TestGLMCLI:
    # ------------------------------------------------------------------
    # 1. _validate_config_keys accepts family
    # ------------------------------------------------------------------

    def test_validate_accepts_family_key(self):
        cfg = {
            "formula": "~ 1 + A",
            "factors": {"A": [0.0, 1.0]},
            "family": "binomial",
            "contrast": {"L": [[0, 1]], "delta": [0.2]},
        }
        # Should not raise
        _validate_config_keys(cfg)

    def test_validate_rejects_no_power_key(self):
        cfg = {"formula": "~ 1 + A", "factors": {"A": [0.0, 1.0]}}
        with pytest.raises(KeyError, match="family"):
            _validate_config_keys(cfg)

    # ------------------------------------------------------------------
    # 2. _make_power_cfg builds PowerGLMContrastConfig from YAML dict
    # ------------------------------------------------------------------

    def test_make_power_cfg_returns_glm_type_binomial(self):
        cfg = _glm_cfg_dict(family="binomial", baseline=0.3)
        result = _make_power_cfg(cfg, FORMULA, FACTORS)
        assert isinstance(result, PowerGLMContrastConfig)
        assert result.family == "binomial"

    def test_make_power_cfg_returns_glm_type_poisson(self):
        cfg = _glm_cfg_dict(family="poisson", baseline=2.0, sesoi=0.3)
        result = _make_power_cfg(cfg, FORMULA, FACTORS)
        assert isinstance(result, PowerGLMContrastConfig)
        assert result.family == "poisson"

    def test_make_power_cfg_glm_baseline_forwarded(self):
        cfg = _glm_cfg_dict(baseline=0.25)
        result = _make_power_cfg(cfg, FORMULA, FACTORS)
        assert result.baseline == pytest.approx(0.25)

    def test_make_power_cfg_glm_link_forwarded(self):
        cfg = _glm_cfg_dict(link="logit")
        result = _make_power_cfg(cfg, FORMULA, FACTORS)
        assert result.link == "logit"

    def test_make_power_cfg_glm_alpha_power_forwarded(self):
        cfg = _glm_cfg_dict()
        cfg["alpha"] = 0.01
        cfg["power"] = 0.90
        result = _make_power_cfg(cfg, FORMULA, FACTORS)
        assert result.alpha == pytest.approx(0.01)
        assert result.power == pytest.approx(0.90)

    def test_make_power_cfg_glm_explicit_L_delta(self):
        cfg = {
            "formula": FORMULA,
            "factors": {"A": [0.0, 1.0]},
            "family": "binomial",
            "baseline": 0.3,
            "contrast": {"L": [[0, 1]], "delta": [0.15]},
        }
        result = _make_power_cfg(cfg, FORMULA, FACTORS)
        assert isinstance(result, PowerGLMContrastConfig)
        assert result.L.shape == (1, 2)
        assert result.delta[0] == pytest.approx(0.15)

    def test_make_power_cfg_glm_missing_baseline_raises(self):
        cfg = {
            "formula": FORMULA,
            "factors": {"A": [0.0, 1.0]},
            "family": "binomial",
            "contrast": {"L": [[0, 1]], "delta": [0.15]},
        }
        with pytest.raises(ValueError, match="baseline"):
            _make_power_cfg(cfg, FORMULA, FACTORS)

    def test_make_power_cfg_glm_missing_contrast_raises(self):
        cfg = {
            "formula": FORMULA,
            "factors": {"A": [0.0, 1.0]},
            "family": "binomial",
            "baseline": 0.3,
            # no contrast block
        }
        with pytest.raises(ValueError, match="contrast"):
            _make_power_cfg(cfg, FORMULA, FACTORS)

    # ------------------------------------------------------------------
    # 3. _apply_glm_cli_args
    # ------------------------------------------------------------------

    def test_apply_glm_cli_args_no_flags_returns_same(self):
        cfg = {"formula": "x"}

        class _Args:
            family = None
            link = None
            baseline = None

        result = _apply_glm_cli_args(cfg, _Args())
        assert result is cfg  # unchanged reference

    def test_apply_glm_cli_args_family_overrides(self):
        cfg = {"family": "binomial"}

        class _Args:
            family = "poisson"
            link = None
            baseline = None

        result = _apply_glm_cli_args(cfg, _Args())
        assert result["family"] == "poisson"
        assert cfg["family"] == "binomial"  # original not mutated

    def test_apply_glm_cli_args_baseline_overrides(self):
        cfg = {"baseline": 0.2}

        class _Args:
            family = None
            link = None
            baseline = 0.5

        result = _apply_glm_cli_args(cfg, _Args())
        assert result["baseline"] == pytest.approx(0.5)

    # ------------------------------------------------------------------
    # 4. Templates
    # ------------------------------------------------------------------

    def test_print_template_glm_binomial(self, capsys):
        _print_template("glm-binomial")
        out = capsys.readouterr().out
        assert "family: binomial" in out
        assert "baseline:" in out

    def test_print_template_glm_poisson(self, capsys):
        _print_template("glm-poisson")
        out = capsys.readouterr().out
        assert "family: poisson" in out
        assert "baseline:" in out

    # ------------------------------------------------------------------
    # 5. OLS path still works (regression guard)
    # ------------------------------------------------------------------

    def test_make_power_cfg_ols_contrast_unchanged(self):
        cfg = {
            "formula": FORMULA,
            "factors": {"A": [0.0, 1.0]},
            "contrast": {
                "scenario_a": {"A": 0.0},
                "scenario_b": {"A": 1.0},
                "sesoi": 1.0,
            },
            "sigma": 1.0,
        }
        result = _make_power_cfg(cfg, FORMULA, FACTORS)
        assert isinstance(result, PowerContrastConfig)

    def test_make_power_cfg_r2_unchanged(self):
        cfg = {
            "formula": FORMULA,
            "factors": {"A": [0.0, 1.0]},
            "r2_target": 0.15,
        }
        result = _make_power_cfg(cfg, FORMULA, FACTORS)
        assert isinstance(result, PowerR2Config)
