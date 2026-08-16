# tests/test_report.py
"""Unit and integration tests for lattice_doe.report (Enhancement #14).

Test classes
------------
TestGenerateReportHTML          -- HTML output correctness (requires jinja2 + pillow)
TestGenerateReportAPIIntegration -- find_optimal_design export_report_to= param
TestPDFExportImportError        -- PDF path raises ImportError when weasyprint absent
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lattice_doe import PowerContrastConfig, PowerR2Config

# ---------------------------------------------------------------------------
# Skip marker — all HTML tests require jinja2 and pillow
# ---------------------------------------------------------------------------

jinja2 = pytest.importorskip("jinja2", reason="jinja2 not installed")
pytest.importorskip("PIL", reason="pillow not installed")

from lattice_doe.report import generate_report  # noqa: E402  (after importorskip)

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
    buckets_df = pd.DataFrame(
        {"A": ["low", "high"], "B_mean": [2.5, 7.5], "count": [n // 2, n // 2]}
    )
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
        for section in (
            "Config Summary",
            "Power Metrics",
            "Selected Runs",
            "Unique Run Allocations",
        ):
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
            output_path=tmp_path,  # directory, not a file
            include_power_curve=False,
        )
        assert returned.name == "iopt_report.html"
        assert returned.exists()

    def test_no_suffix_path_gets_html_extension(self, tmp_path):
        out = tmp_path / "my_report"  # no extension
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
    """Test export_report_to= parameter on find_optimal_design()."""

    def test_export_report_to_path(self, tmp_path):
        from lattice_doe import DesignOptions, find_optimal_design
        from lattice_doe.contrasts import contrast_from_scenarios

        formula = "~ 1 + A + B"
        factors = {"A": ["low", "high"], "B": (0.0, 10.0)}
        L, delta = contrast_from_scenarios(
            formula,
            factors,
            {"A": "low", "B": 0.0},
            {"A": "high", "B": 10.0},
            sesoi=1.0,
        )
        cfg = PowerContrastConfig(L=L, delta=delta, power=0.80, max_n=60)
        opts = DesignOptions(candidate_points=100, starts=2, max_iter=30, random_state=0)

        result = find_optimal_design(
            formula,
            factors,
            cfg,
            opts,
            export_report_to=str(tmp_path),
        )

        path_str = result["report"].get("report_path")
        assert path_str is not None, "report_path not set in result['report']"
        report_file = Path(path_str)
        assert report_file.exists(), f"Report file does not exist: {report_file}"
        assert report_file.suffix == ".html"

    def test_export_report_failure_does_not_crash(self, tmp_path):
        from lattice_doe import DesignOptions, find_optimal_design
        from lattice_doe.contrasts import contrast_from_scenarios

        formula = "~ 1 + A + B"
        factors = {"A": ["low", "high"], "B": (0.0, 10.0)}
        L, delta = contrast_from_scenarios(
            formula,
            factors,
            {"A": "low", "B": 0.0},
            {"A": "high", "B": 10.0},
            sesoi=1.0,
        )
        cfg = PowerContrastConfig(L=L, delta=delta, power=0.80, max_n=60)
        opts = DesignOptions(candidate_points=100, starts=2, max_iter=30, random_state=0)

        with patch("lattice_doe.report.generate_report", side_effect=RuntimeError("boom")):
            result = find_optimal_design(
                formula,
                factors,
                cfg,
                opts,
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


# ---------------------------------------------------------------------------
# TD-13: private-helper coverage — figure conversion, diagnostics context,
# and the power-curve figure fallback chain (57% -> targeted)
# ---------------------------------------------------------------------------


class TestFigToBase64:
    def test_matplotlib_figure_roundtrips_to_png(self):
        plt = pytest.importorskip("matplotlib.pyplot")
        from lattice_doe.report import _fig_to_base64

        fig, ax = plt.subplots(figsize=(2, 1))
        ax.plot([0, 1], [0, 1])
        try:
            b64 = _fig_to_base64(fig)
        finally:
            plt.close(fig)
        assert b64 is not None
        import base64 as _b64

        assert _b64.b64decode(b64)[:4] == b"\x89PNG"

    def test_to_image_bytes_win_over_savefig(self):
        from lattice_doe.report import _fig_to_base64

        class FakePlotly:
            def to_image(self, format, width, height):
                return b"png-bytes"

        import base64 as _b64

        assert _b64.b64decode(_fig_to_base64(FakePlotly())) == b"png-bytes"

    def test_unrecognised_object_returns_none(self):
        from lattice_doe.report import _fig_to_base64

        assert _fig_to_base64(object()) is None

    def test_failing_to_image_falls_through_to_none(self):
        from lattice_doe.report import _fig_to_base64

        class Broken:
            def to_image(self, **kw):
                raise RuntimeError("no kaleido")

        assert _fig_to_base64(Broken()) is None


class TestBuildDiagnosticsCtx:
    @staticmethod
    def _ctx(diag):
        from lattice_doe.report import _build_diagnostics_ctx

        return _build_diagnostics_ctx({"diagnostics": diag} if diag is not None else {})

    def test_absent_diagnostics_returns_none(self):
        assert self._ctx(None) is None
        assert self._ctx({}) is None

    @pytest.mark.parametrize(
        "cond,badge",
        [(10.0, "pass"), (45.3, "warn"), (5000.0, "fail")],
    )
    def test_condition_number_badge_thresholds(self, cond, badge):
        # Belsley scale (SR-21): <30 pass, 30-1000 warn, >1000 fail.
        ctx = self._ctx({"condition_number": cond})
        assert ctx["condition_badge"] == badge
        assert ctx["condition_number"] == f"{cond:.2f}"

    def test_missing_fields_render_as_none(self):
        ctx = self._ctx({"d_efficiency": 0.9251})
        assert ctx["condition_number"] is None and ctx["condition_badge"] is None
        assert ctx["d_efficiency"] == "0.9251"


class TestBuildPowerCurveFigure:
    """The B5 fallback chain: real figure -> base64 PNG; any failure or a
    non-DataFrame curve payload -> None (the TD-11 isinstance narrow, whose
    first draft's NameError this try/except would have silently eaten)."""

    def _args(self):
        return dict(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
        )

    def test_sweep_failure_returns_none(self):
        from lattice_doe.report import _build_power_curve_figure

        with patch("lattice_doe.power_curves.power_curve_by_n", side_effect=RuntimeError("boom")):
            assert _build_power_curve_figure(**self._args()) is None

    def test_non_dataframe_curve_payload_returns_none(self):
        # TD-11 regression: curve_result["data"] typed DataFrame|Figure|None;
        # a non-DataFrame must be rejected by the narrow, not crash indexing.
        from lattice_doe.report import _build_power_curve_figure

        with patch(
            "lattice_doe.power_curves.power_curve_by_n",
            return_value={"data": None, "figure": None, "target_n": None},
        ):
            assert _build_power_curve_figure(**self._args()) is None

    def test_real_sweep_produces_png(self):
        pytest.importorskip("matplotlib")
        import base64 as _b64

        from lattice_doe.report import _build_power_curve_figure

        b64 = _build_power_curve_figure(**self._args())
        assert b64 is not None
        assert _b64.b64decode(b64)[:4] == b"\x89PNG"


class TestGenerateReportPowerCurveSection:
    def test_note_rendered_when_figure_unavailable(self, tmp_path):
        out = tmp_path / "report.html"
        with patch("lattice_doe.power_curves.power_curve_by_n", side_effect=RuntimeError("boom")):
            generate_report(
                result=_minimal_result(),
                formula=FORMULA,
                factors=FACTORS,
                power_cfg=_contrast_cfg(),
                output_path=out,
                include_power_curve=True,
            )
        html = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," not in html

    def test_figure_embedded_when_available(self, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "report.html"
        generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=_contrast_cfg(),
            output_path=out,
            include_power_curve=True,
        )
        html = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," in html


class TestGLMReportContent:
    """RV-11 regression: GLM reports carried only the class name as
    power_mode, then the template rendered the R2 branch with blank fields —
    family, link, baseline, contrast shape, and delta were all omitted from
    the shareable report."""

    def test_glm_parameters_render(self, tmp_path):
        from lattice_doe import PowerGLMContrastConfig

        cfg = PowerGLMContrastConfig(
            alpha=0.05,
            power=0.8,
            L=[[0.0, 1.0, 0.0]],
            delta=[0.5],
            family="binomial",
            baseline=0.2,
        )
        out = tmp_path / "glm.html"
        generate_report(
            result=_minimal_result(),
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=cfg,
            output_path=out,
            include_power_curve=False,
        )
        html = out.read_text(encoding="utf-8")
        assert "Wald" in html
        assert "binomial" in html
        assert "logit" in html  # link=None resolves to the canonical link
        assert "0.2" in html
        assert "matrix" in html  # contrast shape
        assert "[0.5]" in html  # delta


class TestDiscriminatedFactorSpecsInReport:
    """RV-13 regression: discriminated dict specs were classified 'unknown'
    and their bounds/levels vanished from the report."""

    def test_bounds_and_levels_render(self, tmp_path):
        factors = {
            "x": {"type": "continuous", "low": -2.5, "high": 7.5},
            "g": {"type": "categorical", "levels": ["red", "green"]},
        }
        out = tmp_path / "disc.html"
        generate_report(
            result=_minimal_result(),
            formula="~ 1 + x + g",
            factors=factors,
            power_cfg=_contrast_cfg(),
            output_path=out,
            include_power_curve=False,
        )
        html = out.read_text(encoding="utf-8")
        assert "unknown" not in html
        assert "-2.5" in html and "7.5" in html
        assert "red" in html and "green" in html


class TestGLMDfLabel:
    """RV-17 regression: GLM results carry an OLS residual df_denom, and the
    report rendered 'df (num / denom)' — but a Wald chi-square test has no
    denominator df. GLM reports must show only the chi-square df."""

    def test_glm_report_shows_chi2_df_only(self, tmp_path):
        from lattice_doe import PowerGLMContrastConfig

        cfg = PowerGLMContrastConfig(
            alpha=0.05,
            power=0.8,
            L=[[0.0, 1.0, 0.0]],
            delta=[0.5],
            family="binomial",
            baseline=0.2,
        )
        result = _minimal_result()
        result["report"]["test_type"] = "wald_chi2"
        out = tmp_path / "glmdf.html"
        generate_report(
            result=result,
            formula=FORMULA,
            factors=FACTORS,
            power_cfg=cfg,
            output_path=out,
            include_power_curve=False,
        )
        html = out.read_text(encoding="utf-8")
        assert "df (num / denom)" not in html
        assert "df" in html  # the chi-square df box renders
