# tests/test_diag_export.py
"""Unit tests for diag_export.export_diagnostics (TD-13).

The module previously had NO direct tests. export_diagnostics is its sole
public function and is re-exported from the documented
`lattice_doe.diagnostics` compat wrapper.

Soft-dependency layout mirrors the module: CSV paths need only pandas;
pdf/png/html need matplotlib; xlsx needs xlsxwriter (write) and the tests
read workbooks back with openpyxl. Each layer skips cleanly when its
dependency is absent, matching the CI no-extras install.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lattice_doe import diag_export
from lattice_doe.diag_export import export_diagnostics
from lattice_doe.diag_metrics import compute_design_metrics, compute_leverages

requires_matplotlib = pytest.mark.skipif(
    not diag_export._HAS_MATPLOTLIB, reason="matplotlib not installed"
)


@pytest.fixture
def design():
    n = 12
    x = np.linspace(-1.0, 1.0, n)
    X = np.column_stack([np.ones(n), x, x**2])
    df = pd.DataFrame({"x": x, "x2": x**2})
    return X, df


class TestDataExports:
    def test_csv_summary_round_trips_exact_metrics(self, design, tmp_path):
        X, df = design
        out = export_diagnostics(X, df, tmp_path / "diag", formats=["csv"])
        metrics = compute_design_metrics(X, include_vif=True, X_cand=None, feature_names=None)

        summary = pd.read_csv(out["summary_csv"])
        by_name = dict(zip(summary["Metric"], summary["Value"]))
        assert by_name["Condition Number"] == pytest.approx(metrics["condition_number"], rel=1e-12)
        assert by_name["D-Efficiency"] == pytest.approx(metrics["d_efficiency"], rel=1e-12)
        assert by_name["Mean Leverage"] == pytest.approx(metrics["leverage_mean"], rel=1e-12)
        assert by_name["Max Leverage"] == pytest.approx(metrics["leverage_max"], rel=1e-12)

    def test_csv_leverages_match_compute_leverages_exactly(self, design, tmp_path):
        X, df = design
        out = export_diagnostics(X, df, tmp_path / "diag", formats=["csv"])
        lev = pd.read_csv(out["leverages_csv"])
        expected = compute_leverages(X)
        assert lev["run"].tolist() == list(range(1, len(expected) + 1))
        np.testing.assert_allclose(lev["leverage"].to_numpy(), expected, rtol=1e-12)

    def test_csv_vif_written_for_multi_predictor(self, design, tmp_path):
        X, df = design
        out = export_diagnostics(X, df, tmp_path / "diag", formats=["csv"])
        assert out["vif_csv"].exists()
        vif = pd.read_csv(out["vif_csv"])
        assert {"feature", "vif"} <= set(vif.columns)

    def test_include_data_false_writes_no_tables(self, design, tmp_path):
        X, df = design
        out = export_diagnostics(X, df, tmp_path / "diag", formats=["csv"], include_data=False)
        assert "summary_csv" not in out
        assert "leverages_csv" not in out

    def test_xlsx_workbook_has_three_sheets(self, design, tmp_path):
        pytest.importorskip("xlsxwriter")
        openpyxl = pytest.importorskip("openpyxl")
        X, df = design
        out = export_diagnostics(X, df, tmp_path / "diag", formats=["xlsx"])
        wb = openpyxl.load_workbook(out["xlsx"])
        assert set(wb.sheetnames) == {"Summary", "VIF", "Leverages"}

    def test_parent_directories_are_created(self, design, tmp_path):
        X, df = design
        nested = tmp_path / "a" / "b" / "diag"
        out = export_diagnostics(X, df, nested, formats=["csv"])
        assert out["summary_csv"].exists()


@requires_matplotlib
class TestPlotExports:
    def test_default_formats_are_html_and_pdf(self, design, tmp_path):
        X, df = design
        out = export_diagnostics(X, df, tmp_path / "diag")  # formats=None
        assert set(out) == {"html", "pdf"}
        assert out["html"].exists() and out["pdf"].exists()

    def test_html_is_self_contained_with_exact_metrics(self, design, tmp_path):
        X, df = design
        out = export_diagnostics(X, df, tmp_path / "diag", formats=["html"])
        html = out["html"].read_text(encoding="utf-8")
        # The Windows-default-encoding regression (TD-13): the κ that used to
        # crash write_text under cp1252 must round-trip, with the charset
        # declared so browsers decode it the same way.
        assert '<meta charset="utf-8">' in html
        assert "κ(X)" in html
        metrics = compute_design_metrics(X, include_vif=True, X_cand=None, feature_names=None)
        # The figure is embedded, not referenced.
        assert "data:image/png;base64," in html
        # Metric values appear exactly as the module formats them.
        assert f"{metrics['condition_number']:.2f}" in html
        assert f"{metrics['d_efficiency']:.4f}" in html

    def test_png_written(self, design, tmp_path):
        X, df = design
        out = export_diagnostics(X, df, tmp_path / "diag", formats=["png"])
        assert out["png"].exists()
        assert out["png"].stat().st_size > 1000  # a real rasterised figure

    def test_pdf_written(self, design, tmp_path):
        X, df = design
        out = export_diagnostics(X, df, tmp_path / "diag", formats=["pdf"])
        assert out["pdf"].exists()
        assert out["pdf"].read_bytes()[:5] == b"%PDF-"


class TestWithoutMatplotlib:
    def test_plot_formats_silently_skipped_csv_still_written(self, design, tmp_path):
        X, df = design
        with patch.object(diag_export, "_HAS_MATPLOTLIB", False):
            out = export_diagnostics(X, df, tmp_path / "diag", formats=["csv", "html", "pdf"])
        assert "summary_csv" in out
        assert "html" not in out and "pdf" not in out
