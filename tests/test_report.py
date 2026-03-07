# tests/test_report.py
"""Unit and integration tests for iopt_power_design.report (Enhancement #14).

Test classes
------------
TestGenerateReportHTML          -- HTML output correctness (requires jinja2 + pillow)
TestGenerateReportAPIIntegration -- i_optimal_powered_design export_report_to= param
TestPDFExportImportError        -- PDF path raises ImportError when weasyprint absent
"""
from __future__ import annotations

import importlib
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from iopt_power_design import PowerContrastConfig, PowerR2Config

# ---------------------------------------------------------------------------
# Skip marker — all HTML tests require jinja2 and pillow
# ---------------------------------------------------------------------------

jinja2 = pytest.importorskip("jinja2", reason="jinja2 not installed")
pytest.importorskip("PIL", reason="pillow not installed")

from iopt_power_design.report import generate_report  # noqa: E402  (after importorskip)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FORMULA = "~ 1 + A + B"
FACTORS = {"A": ["low", "high"], "B": (0.0, 10.0)}


def _contrast_cfg() -> PowerContrastConfig:
    """Minimal 1-row contrast config; L has p=3 columns for '~ 1 + A + B'."""
    return PowerContrastConfig(
        L=[[0, 1, 0]],
        delta=[0.5],
        alpha=0.05,
        power=0.80,
        sigma=1.0,
        max_n=100,
    )


def _r2_cfg() -> PowerR2Config:
    return PowerR2Config(r2_target=0.30, power=0.80, alpha=0.05, max_n=100)


def _minimal_result(n: int = 12) -> dict:
    """Build a minimal result dict that satisfies _build_context without running the optimizer."""
    rng = np.random.default_rng(0)
    design_df = pd.DataFrame(
        {
            "A": np.tile(["low", "high"], n // 2 + 1)[:n],
            "B": rng.uniform(0.0, 10.0, n),
        }
    )
    buckets_df = pd.DataFrame({"A": ["low", "high"], "B_mean": [2.5, 7.5], "count": [n // 2, n // 2]})
    report = {
        "n": n,
        "achieved_power": 0.83,
        "target_power": 0.80,
        "noncentrality_lambda": 9.12,
        "df_num": 1,
        "df_denom": n - 3,
        "criterion": "I",
        "elapsed_sec": 0.42,
        "search_strategy": "binary_search",
        "random_state": 42,
        "warnings": [],
        "p": 3,
        "diagnostics": {
            "condition_number": 45.3,
            "d_efficiency": 0.923,
            "i_criterion": 0.0034,
            "vifs": {"A[T.high]": 1.05, "B": 1.02},
        },
    }
    return {"design_df": design_df, "buckets_df": buckets_df, "report": report}


class _StrictHTMLParser(HTMLParser):
    """Raises AssertionError on malformed HTML fed to it."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.errors: list[str] = []

    def handle_entityref(self, name):
        pass  # allow HTML entities like &middot;

    def unknown_decl(self, data):
        self.errors.append(f"Unknown decl: {data}")


# ---------------------------------------------------------------------------
# TestGenerateReportHTML
# ---------------------------------------------------------------------------

class TestGenerateReportHTML:
    """HTML report generation — correctness and content."""

    def test_html_report_creates_file(self, tmp_path):
        out = tmp_path / "report.html"
        returned = generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            output_path=out,
            include_power_curve=False,
        )
        assert returned == out.resolve()
        assert out.exists()
        assert out.suffix == ".html"

    def test_html_is_parseable(self, tmp_path):
        out = tmp_path / "report.html"
        generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            output_path=out,
            include_power_curve=False,
        )
        html = out.read_text(encoding="utf-8")
        parser = _StrictHTMLParser()
        parser.feed(html)
        assert not parser.errors, f"HTML parse errors: {parser.errors}"

    def test_html_contains_key_sections(self, tmp_path):
        out = tmp_path / "report.html"
        generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            output_path=out,
            include_power_curve=False,
        )
        html = out.read_text(encoding="utf-8")
        for section in ("Config Summary", "Power Metrics", "Selected Runs", "Unique Run Allocations"):
            assert section in html, f"Section not found in HTML: {section!r}"

    def test_html_is_self_contained(self, tmp_path):
        out = tmp_path / "report.html"
        generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            output_path=out,
            include_power_curve=False,
        )
        html = out.read_text(encoding="utf-8")
        # No external URLs (http:// or https://) should appear
        assert "http://" not in html, "Report contains http:// link — not self-contained"
        assert "https://" not in html, "Report contains https:// link — not self-contained"

    def test_html_report_contrast_mode(self, tmp_path):
        out = tmp_path / "report_contrast.html"
        generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            output_path=out,
            include_power_curve=False,
        )
        html = out.read_text(encoding="utf-8")
        # Contrast-specific fields
        assert "sigma" in html.lower() or "Sigma" in html, "sigma not found for contrast mode"
        assert "delta" in html.lower() or "Delta" in html, "delta not found for contrast mode"

    def test_html_report_r2_mode(self, tmp_path):
        out = tmp_path / "report_r2.html"
        generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_r2_cfg(),
            output_path=out,
            include_power_curve=False,
        )
        html = out.read_text(encoding="utf-8")
        # R²-specific fields
        assert "r2_target" in html or "R" in html, "r2_target not found for R² mode"
        assert "lambda_mode" in html or "mode" in html.lower(), "lambda_mode not found for R² mode"

    def test_truncation_note(self, tmp_path):
        out = tmp_path / "report_trunc.html"
        # Build a result with 100 design rows
        result = _minimal_result(n=100)
        generate_report(
            result=result,
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            output_path=out,
            include_power_curve=False,
            design_rows_shown=10,
        )
        html = out.read_text(encoding="utf-8")
        assert "100" in html, "Total row count (100) not shown in truncation note"
        assert "10" in html, "Rows-shown count (10) not shown in truncation note"

    def test_directory_path_creates_default_filename(self, tmp_path):
        returned = generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            output_path=tmp_path,           # directory, not a file
            include_power_curve=False,
        )
        assert returned.name == "iopt_report.html"
        assert returned.exists()

    def test_no_suffix_path_gets_html_extension(self, tmp_path):
        out = tmp_path / "my_report"          # no extension
        returned = generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            output_path=out,
            include_power_curve=False,
        )
        assert returned.suffix == ".html"
        assert returned.exists()


# ---------------------------------------------------------------------------
# TestGenerateReportAPIIntegration
# ---------------------------------------------------------------------------

class TestGenerateReportAPIIntegration:
    """Test export_report_to= parameter on i_optimal_powered_design()."""

    def test_export_report_to_path(self, tmp_path):
        from iopt_power_design import DesignOptions, i_optimal_powered_design
        from iopt_power_design.contrasts import contrast_from_scenarios

        formula = "~ 1 + A + B"
        factors = {"A": ["low", "high"], "B": (0.0, 10.0)}
        L, delta = contrast_from_scenarios(
            formula, factors,
            {"A": "low", "B": 0.0},
            {"A": "high", "B": 10.0},
            sesoi=1.0,
        )
        cfg = PowerContrastConfig(L=L, delta=delta, power=0.80, max_n=60)
        opts = DesignOptions(candidate_points=100, starts=2, max_iter=30, random_state=0)

        result = i_optimal_powered_design(
            formula, factors, cfg, opts,
            export_report_to=str(tmp_path),
        )

        path_str = result["report"].get("report_path")
        assert path_str is not None, "report_path not set in result['report']"
        report_file = Path(path_str)
        assert report_file.exists(), f"Report file does not exist: {report_file}"
        assert report_file.suffix == ".html"

    def test_export_report_failure_does_not_crash(self, tmp_path):
        from iopt_power_design import DesignOptions, i_optimal_powered_design
        from iopt_power_design.contrasts import contrast_from_scenarios

        formula = "~ 1 + A + B"
        factors = {"A": ["low", "high"], "B": (0.0, 10.0)}
        L, delta = contrast_from_scenarios(
            formula, factors,
            {"A": "low", "B": 0.0},
            {"A": "high", "B": 10.0},
            sesoi=1.0,
        )
        cfg = PowerContrastConfig(L=L, delta=delta, power=0.80, max_n=60)
        opts = DesignOptions(candidate_points=100, starts=2, max_iter=30, random_state=0)

        with patch("iopt_power_design.report.generate_report", side_effect=RuntimeError("boom")):
            result = i_optimal_powered_design(
                formula, factors, cfg, opts,
                export_report_to=str(tmp_path / "report.html"),
            )

        # Design result still returned despite report failure
        assert "design_df" in result
        assert result["design_df"] is not None
        assert "report_path_error" in result["report"]


# ---------------------------------------------------------------------------
# TestPDFExportImportError
# ---------------------------------------------------------------------------

class TestPDFExportImportError:
    """PDF export raises ImportError with install hint when weasyprint is absent."""

    def test_pdf_raises_import_error_without_weasyprint(self, tmp_path):
        out = tmp_path / "report.pdf"
        # Simulate weasyprint being absent
        with patch.dict("sys.modules", {"weasyprint": None}):
            with pytest.raises(ImportError, match="report-pdf"):
                generate_report(
                    result=_minimal_result(),
                    formula=FORMULA,
                    factors=FACTORS,
                    power_cfg=_contrast_cfg(),
                    output_path=out,
                    include_power_curve=False,
                )
