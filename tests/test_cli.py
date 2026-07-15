# tests/test_cli.py
"""Unit tests for the CLI — GL-6: GLM YAML / CLI support."""
from __future__ import annotations


import pytest

from lattice_doe.cli import (
    _make_power_cfg,
    _validate_config_keys,
    _apply_glm_cli_args,
    _print_template,
    main,
)
from lattice_doe.config import (
    DesignOptions, PowerGLMContrastConfig, PowerContrastConfig, PowerR2Config,
)


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
        result = _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())
        assert isinstance(result, PowerGLMContrastConfig)
        assert result.family == "binomial"

    def test_make_power_cfg_returns_glm_type_poisson(self):
        cfg = _glm_cfg_dict(family="poisson", baseline=2.0, sesoi=0.3)
        result = _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())
        assert isinstance(result, PowerGLMContrastConfig)
        assert result.family == "poisson"

    def test_make_power_cfg_glm_baseline_forwarded(self):
        cfg = _glm_cfg_dict(baseline=0.25)
        result = _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())
        assert result.baseline == pytest.approx(0.25)

    def test_make_power_cfg_glm_link_forwarded(self):
        cfg = _glm_cfg_dict(link="logit")
        result = _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())
        assert result.link == "logit"

    def test_make_power_cfg_glm_alpha_power_forwarded(self):
        cfg = _glm_cfg_dict()
        cfg["alpha"] = 0.01
        cfg["power"] = 0.90
        result = _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())
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
        result = _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())
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
            _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())

    def test_make_power_cfg_glm_missing_contrast_raises(self):
        cfg = {
            "formula": FORMULA,
            "factors": {"A": [0.0, 1.0]},
            "family": "binomial",
            "baseline": 0.3,
            # no contrast block
        }
        with pytest.raises(ValueError, match="contrast"):
            _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())

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
        result = _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())
        assert isinstance(result, PowerContrastConfig)

    def test_make_power_cfg_r2_unchanged(self):
        cfg = {
            "formula": FORMULA,
            "factors": {"A": [0.0, 1.0]},
            "r2_target": 0.15,
        }
        result = _make_power_cfg(cfg, FORMULA, FACTORS, DesignOptions())
        assert isinstance(result, PowerR2Config)

    # ------------------------------------------------------------------
    # 6. CR-38: --link parser choices and template correctness
    # ------------------------------------------------------------------

    def test_link_identity_rejected_by_parser(self):
        """CR-38: --link identity must be rejected by argparse before reaching config."""
        with pytest.raises(SystemExit):
            main(["--link", "identity", "--dry-run"])

    def test_link_sqrt_rejected_by_parser(self):
        """CR-38: --link sqrt must be rejected by argparse before reaching config."""
        with pytest.raises(SystemExit):
            main(["--link", "sqrt", "--dry-run"])

    def test_glm_binomial_template_no_invalid_links(self, capsys):
        """CR-38: glm-binomial template must not advertise identity or sqrt."""
        _print_template("glm-binomial")
        out = capsys.readouterr().out
        assert "identity" not in out
        assert "sqrt" not in out

    def test_glm_poisson_template_no_invalid_links(self, capsys):
        """CR-38: glm-poisson template must not advertise identity or sqrt."""
        _print_template("glm-poisson")
        out = capsys.readouterr().out
        assert "identity" not in out
        assert "sqrt" not in out


class TestUX7CliExitCode:
    """UX-7 regression: the CLI exited 0 on a search that missed its target.
    A partial result now exits 3 unless --allow-partial is given."""

    _CFG = """
formula: "~ 1 + x1 + x2"
factors:
  x1: [-1.0, 1.0]
  x2: [-1.0, 1.0]
contrast:
  L: [[0.0, 1.0, 0.0]]
  delta: [0.3]
alpha: 0.05
power: 0.80
sigma: 1.0
max_n: 30
design:
  auto_candidate: false
  candidate_points: 100
  starts: 1
  random_state: 0
"""

    def _run(self, tmp_path, extra):
        cfg = tmp_path / "cfg.yml"
        cfg.write_text(self._CFG, encoding="utf-8")
        return main(["--config", str(cfg), "--out", str(tmp_path)] + extra)

    def test_partial_exits_3(self, tmp_path):
        assert self._run(tmp_path, []) == 3

    def test_allow_partial_exits_0(self, tmp_path):
        assert self._run(tmp_path, ["--allow-partial"]) == 0

    def test_partial_no_allow_logs_error(self, tmp_path, caplog):
        """Without --allow-partial the miss is an error (exit 3)."""
        import logging
        with caplog.at_level(logging.WARNING):
            assert self._run(tmp_path, []) == 3
        recs = [r for r in caplog.records if "WITHOUT reaching" in r.message]
        assert recs and all(r.levelno == logging.ERROR for r in recs)

    def test_allow_partial_logs_warning_not_error(self, tmp_path, caplog):
        """With --allow-partial the miss is informational (exit 0), so it must
        NOT be logged at ERROR and must not claim 'exiting 3' (P3)."""
        import logging
        with caplog.at_level(logging.WARNING):
            assert self._run(tmp_path, ["--allow-partial"]) == 0
        recs = [r for r in caplog.records if "WITHOUT reaching" in r.message]
        assert recs, "expected an informational partial-completion log"
        assert all(r.levelno == logging.WARNING for r in recs)
        assert not any("exiting 3" in r.message for r in recs)


class TestUX3CliProgress:
    """UX-3: --progress streams live search progress to stderr."""

    _CFG = """
formula: "~ 1 + x1 + x2"
factors:
  x1: [-1.0, 1.0]
  x2: [-1.0, 1.0]
contrast:
  L: [[0.0, 1.0, 0.0]]
  delta: [1.2]
alpha: 0.05
power: 0.80
sigma: 1.0
max_n: 60
design:
  auto_candidate: false
  candidate_points: 100
  starts: 1
  random_state: 0
"""

    def test_progress_flag_writes_phase_lines_to_stderr(self, tmp_path, capsys):
        cfg = tmp_path / "cfg.yml"
        cfg.write_text(self._CFG, encoding="utf-8")
        rc = main(["--config", str(cfg), "--out", str(tmp_path), "--progress"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "validating" in err
        assert "optimizing" in err
        assert "done" in err

    def test_no_progress_flag_stays_quiet(self, tmp_path, capsys):
        cfg = tmp_path / "cfg.yml"
        cfg.write_text(self._CFG, encoding="utf-8")
        main(["--config", str(cfg), "--out", str(tmp_path)])
        err = capsys.readouterr().err
        # Phase lines only appear with --progress (or --verbose).
        assert "optimizing" not in err
