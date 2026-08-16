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
    DesignOptions,
    PowerGLMContrastConfig,
    PowerContrastConfig,
    PowerR2Config,
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


class TestHostileResponseNames:
    """UX-67 regression: ResponseSpec accepts names like 'Yield/Day', which
    used to raise ValueError inside Path.with_name AFTER the design search
    completed. Files get slugged names; the report maps originals to files."""

    def test_compound_export_with_path_separator_name(self, tmp_path):
        import json

        cfg = tmp_path / "mr.yml"
        cfg.write_text(
            "formula: '~ 1 + x'\n"
            "factors:\n  x: [0.0, 1.0]\n"
            "responses:\n"
            "  - name: y1\n    sigma: 1.0\n"
            "    contrast: {L: [[0.0, 1.0]], delta: [0.5]}\n"
            "  - name: 'Yield/Day'\n    sigma: 1.0\n"
            "    formula: '~ 1 + x + I(x**2)'\n"
            "    contrast: {L: [[0.0, 1.0, 0.0]], delta: [0.5]}\n"
            "alpha: 0.05\npower: 0.8\nmax_n: 12\n"
            "design: {candidate_points: 40, starts: 1}\n",
            encoding="utf-8",
        )
        out = tmp_path / "run"
        rc = main(
            [
                "--config",
                str(cfg),
                "--out",
                str(out),
                "--allow-partial",
            ]
        )
        assert rc == 0
        assert (tmp_path / "run_model_matrix_Yield_Day.csv").exists()
        report = json.loads((tmp_path / "run_report.json").read_text(encoding="utf-8"))
        assert report["model_matrix_files"]["Yield/Day"] == ("run_model_matrix_Yield_Day.csv")

    def test_case_only_response_names_write_distinct_files(self, tmp_path):
        """UX-69: on a case-insensitive filesystem, Yield and yield would
        silently overwrite each other's CSV without casefolded collision
        tracking."""
        import json

        cfg = tmp_path / "mr.yml"
        cfg.write_text(
            "formula: '~ 1 + x'\n"
            "factors:\n  x: [0.0, 1.0]\n"
            "responses:\n"
            "  - name: Yield\n    sigma: 1.0\n"
            "    contrast: {L: [[0.0, 1.0]], delta: [0.5]}\n"
            "  - name: 'yield'\n    sigma: 1.0\n"
            "    formula: '~ 1 + x + I(x**2)'\n"
            "    contrast: {L: [[0.0, 1.0, 0.0]], delta: [0.5]}\n"
            "alpha: 0.05\npower: 0.8\nmax_n: 12\n"
            "design: {candidate_points: 40, starts: 1}\n",
            encoding="utf-8",
        )
        rc = main(
            [
                "--config",
                str(cfg),
                "--out",
                str(tmp_path / "run"),
                "--allow-partial",
            ]
        )
        assert rc == 0
        report = json.loads((tmp_path / "run_report.json").read_text(encoding="utf-8"))
        files = report["model_matrix_files"]
        assert files["Yield"] != files["yield"]
        assert files["Yield"].casefold() != files["yield"].casefold()
        for f in files.values():
            assert (tmp_path / f).exists()


# ---------------------------------------------------------------------------
# TD-13 phase 2: main() dispatch, output flags, and error-exit coverage
# ---------------------------------------------------------------------------

_TD13_CFG = """
formula: "~ 1 + x1 + x2"
factors:
  x1: [-1.0, 1.0]
  x2: [-1.0, 1.0]
contrast:
  L: [[0.0, 1.0, 0.0]]
  delta: [1.5]
alpha: 0.05
power: 0.80
sigma: 1.0
max_n: 40
design:
  auto_candidate: false
  candidate_points: 100
  starts: 1
  random_state: 0
"""


def _write_cfg(tmp_path, text=_TD13_CFG, name="cfg.yml"):
    cfg = tmp_path / name
    cfg.write_text(text, encoding="utf-8")
    return cfg


class TestMainErrorExits:
    def test_missing_config_file_exits_2(self, tmp_path):
        assert main(["--config", str(tmp_path / "nope.yml"), "--out", str(tmp_path)]) == 2

    def test_invalid_yaml_exits_2(self, tmp_path):
        cfg = _write_cfg(tmp_path, "formula: [unclosed")
        assert main(["--config", str(cfg), "--out", str(tmp_path)]) == 2

    def test_missing_required_keys_exits_2(self, tmp_path):
        cfg = _write_cfg(tmp_path, 'formula: "~ 1 + x1"\n')
        assert main(["--config", str(cfg), "--out", str(tmp_path)]) == 2

    def test_no_entry_path_errors_with_alternatives(self, capsys):
        # CR-16: the message must list every non-config entry path.
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        for alt in ("--template", "--sheets", "--excel-template", "--excel-run"):
            assert alt in err


class TestMainDryRun:
    def test_dry_run_exits_0_and_writes_nothing(self, tmp_path, caplog):
        import logging

        cfg = _write_cfg(tmp_path)
        out = tmp_path / "out"
        with caplog.at_level(logging.INFO):
            assert main(["--config", str(cfg), "--out", str(out), "--dry-run"]) == 0
        assert any("Dry Run Validation Successful" in r.message for r in caplog.records)
        assert not list(out.glob("*.csv"))


class TestMainOutputFlags:
    def _run(self, tmp_path, extra=(), cfg_text=_TD13_CFG):
        cfg = _write_cfg(tmp_path, cfg_text)
        out = tmp_path / "out"
        rc = main(["--config", str(cfg), "--out", str(out)] + list(extra))
        return rc, out

    def test_html_report_flag_writes_html(self, tmp_path):
        pytest.importorskip("jinja2")
        pytest.importorskip("PIL")
        rc, out = self._run(tmp_path, ["--html-report"])
        assert rc == 0
        assert list(out.parent.glob("**/*.html")) or list(out.glob("**/*.html"))

    def test_excel_flag_writes_workbook(self, tmp_path):
        pytest.importorskip("xlsxwriter")
        rc, out = self._run(tmp_path, ["--excel"])
        assert rc == 0
        assert list(out.glob("**/*.xlsx")) or list(out.parent.glob("**/*.xlsx"))

    def test_robustness_report_prints_summary(self, tmp_path, capsys):
        rc, _ = self._run(tmp_path, ["--robustness-report"])
        assert rc == 0
        stdout = capsys.readouterr().out
        assert "=== Robustness Report ===" in stdout
        assert "pct_scenarios_passing" in stdout
        assert "worst_power" in stdout


class TestMainSheetsDispatch:
    def test_sheets_flag_dispatches_to_sheets_run(self, tmp_path, monkeypatch):
        # The sheets path must not need --config; sheets_run is stubbed so no
        # network is touched — this pins the dispatch wiring only.
        sheets = pytest.importorskip("lattice_doe.sheets")
        calls = {}

        def _fake_sheets_run(url, credentials=None, **kw):
            calls["url"] = url
            return {
                "report": {"n": 12, "p": 3, "achieved_power": 0.9, "elapsed_sec": 0.1},
                "spreadsheet_url": "https://sheets.example/abc/view",
            }

        monkeypatch.setattr(sheets, "sheets_run", _fake_sheets_run)
        rc = main(["--sheets", "https://sheets.example/abc"])
        assert rc == 0
        assert calls["url"] == "https://sheets.example/abc"


class TestMainExcelTemplateRoundTrip:
    def test_excel_template_then_run(self, tmp_path):
        pytest.importorskip("openpyxl")
        wb = tmp_path / "starter.xlsx"
        assert main(["--excel-template", str(wb), "--template-mode", "r2"]) == 0
        assert wb.exists()
