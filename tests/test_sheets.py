# tests/test_sheets.py
# License: MIT
"""Unit tests for lattice_doe.sheets — all gspread calls mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import lattice_doe.sheets as sheets_module
from lattice_doe.sheets import (
    SheetsError,
    _df_to_rows,
    _parse_config_sheet,
    _TEMPLATE_ROWS,
)
from lattice_doe.config import DesignOptions, PowerContrastConfig, PowerR2Config


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_worksheet(rows):
    """Mock gspread Worksheet whose get_all_values() returns *rows*."""
    ws = MagicMock()
    ws.get_all_values.return_value = rows
    return ws


def _make_mock_gspread():
    """Minimal mock of the gspread module, including a real exception class."""
    mg = MagicMock()
    mg.exceptions.WorksheetNotFound = type("WorksheetNotFound", (Exception,), {})
    return mg


class _FakeWorksheet:
    """In-memory worksheet: a title plus the last-written rectangle."""

    def __init__(self, title):
        self.title = title
        self.rows: list = []

    def update(self, _addr, rows):
        self.rows = rows

    def clear(self):
        self.rows = []

    def get_all_values(self):
        return self.rows


class _FakeSpreadsheet:
    """In-memory spreadsheet enforcing Google's title rules (UX-73).

    Lookups are exact, but creating a title that differs from an existing
    one only by case fails hard — the real API rejects it, so a writer that
    only compares bare slugs (not complete titles) errors here instead of
    silently desyncing the index."""

    def __init__(self, mock_gspread):
        self._mg = mock_gspread
        self._sheets: list = []

    def worksheets(self):
        return list(self._sheets)

    def worksheet(self, title):
        for w in self._sheets:
            if w.title == title:
                return w
        raise self._mg.exceptions.WorksheetNotFound(title)

    def add_worksheet(self, title, rows, cols):
        if any(w.title.casefold() == title.casefold() for w in self._sheets):
            raise AssertionError(
                f"Google rejects duplicate worksheet title: {title!r}"
            )
        w = _FakeWorksheet(title)
        self._sheets.append(w)
        return w

    def del_worksheet(self, ws):
        self._sheets.remove(ws)

    def titles(self):
        return [w.title for w in self._sheets]


def _minimal_result(design_df=None, buckets_df=None):
    """Minimal result dict matching find_optimal_design() output."""
    if design_df is None:
        design_df = pd.DataFrame({"x1": [0.1, 0.5], "x2": [-1.0, 1.0]})
    if buckets_df is None:
        buckets_df = pd.DataFrame({"x1": [0.1], "count": [2]})
    report = {
        "n": 10,
        "p": 3,
        "df_num": 2,
        "df_denom": 7,
        "alpha": 0.05,
        "target_power": 0.80,
        "achieved_power": 0.85,
        "noncentrality_lambda": 8.0,
        "criterion": "I",
        "elapsed_sec": 1.23,
        "warnings": [],
        "diagnostics": {
            "i_criterion": 0.45,
            "d_efficiency": 0.91,
            "condition_number": 3.2,
        },
    }
    return {"design_df": design_df, "buckets_df": buckets_df, "report": report}


def _simple_parse_return():
    """Return value for a patched _parse_config_sheet (r2 mode, 1 factor)."""
    return (
        "x1",
        {"x1": (-1.0, 1.0)},
        PowerR2Config(r2_target=0.25),
        DesignOptions(),
        None,  # multi_cfg (single-response path)
    )


# ---------------------------------------------------------------------------
# TestSheetsImportGuard
# ---------------------------------------------------------------------------

class TestSheetsImportGuard:
    def test_sheets_run_raises_import_error_when_no_gspread(self):
        with patch.object(sheets_module, "_HAS_GSPREAD", False):
            with pytest.raises(ImportError, match="gspread"):
                sheets_module.sheets_run("fake-id")

    def test_create_template_raises_import_error_when_no_gspread(self):
        with patch.object(sheets_module, "_HAS_GSPREAD", False):
            with pytest.raises(ImportError, match="gspread"):
                sheets_module.create_sheet_template()


# ---------------------------------------------------------------------------
# TestGetClient
# ---------------------------------------------------------------------------

class TestGetClient:
    def test_service_account_path_calls_gspread_service_account(self):
        mg = _make_mock_gspread()
        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._get_client("path/to/sa.json")

        mg.service_account.assert_called_once_with(filename="path/to/sa.json")

    def test_none_credentials_calls_gspread_oauth(self):
        mg = _make_mock_gspread()
        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._get_client(None)

        mg.oauth.assert_called_once()


# ---------------------------------------------------------------------------
# TestParseConfigSheet
# ---------------------------------------------------------------------------

class TestParseConfigSheet:
    def test_r2_mode_parses_formula_and_factors(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1 + x2"],
            ["power_mode", "r2"],
            ["r2_target", "0.30"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
            ["x2", "continuous", "-1.0", "1.0"],
        ]
        formula, factors, power_cfg, design_opts, _ = _parse_config_sheet(
            _make_mock_worksheet(rows)
        )
        assert formula == "x1 + x2"
        assert factors["x1"] == (-1.0, 1.0)
        assert factors["x2"] == (-1.0, 1.0)

    def test_r2_mode_returns_power_r2_config(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1 + x2"],
            ["power_mode", "r2"],
            ["r2_target", "0.25"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        _, _, power_cfg, _, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert isinstance(power_cfg, PowerR2Config)
        assert power_cfg.r2_target == pytest.approx(0.25)

    def test_r2_mode_defaults_applied_when_keys_absent(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "r2"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        _, _, power_cfg, design_opts, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert isinstance(power_cfg, PowerR2Config)
        assert power_cfg.alpha == pytest.approx(0.05)
        assert power_cfg.power == pytest.approx(0.80)
        assert design_opts.starts == 5
        assert design_opts.random_state == 123

    def test_contrast_mode_single_contrast(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1 + x2"],
            ["power_mode", "contrast"],
            ["[CONTRAST]", ""],
            ["L_row", "0,1,0"],
            ["delta", "1.0"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
            ["x2", "continuous", "-1.0", "1.0"],
        ]
        _, _, power_cfg, _, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert isinstance(power_cfg, PowerContrastConfig)
        assert power_cfg.L.shape == (1, 3)
        np.testing.assert_array_equal(power_cfg.L[0], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(power_cfg.delta, [1.0])

    def test_contrast_mode_multi_row_L_matrix(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1 + x2"],
            ["power_mode", "contrast"],
            ["[CONTRAST]", ""],
            ["L_row", "0,1,0"],
            ["L_row", "0,0,1"],
            ["delta", "1.0,0.5"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
            ["x2", "continuous", "-1.0", "1.0"],
        ]
        _, _, power_cfg, _, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert isinstance(power_cfg, PowerContrastConfig)
        assert power_cfg.L.shape == (2, 3)
        np.testing.assert_array_equal(power_cfg.delta, [1.0, 0.5])

    def test_contrast_mode_delta_length_mismatch_raises(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "contrast"],
            ["[CONTRAST]", ""],
            ["L_row", "0,1"],
            ["L_row", "1,0"],
            ["delta", "1.0"],   # 1 value but 2 L_rows
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        with pytest.raises(SheetsError, match="delta"):
            _parse_config_sheet(_make_mock_worksheet(rows))

    def test_missing_settings_sentinel_raises(self):
        rows = [
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        with pytest.raises(SheetsError, match=r"\[SETTINGS\]"):
            _parse_config_sheet(_make_mock_worksheet(rows))

    def test_missing_factors_sentinel_raises(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "r2"],
        ]
        with pytest.raises(SheetsError, match=r"\[FACTORS\]"):
            _parse_config_sheet(_make_mock_worksheet(rows))

    def test_missing_formula_key_raises(self):
        rows = [
            ["[SETTINGS]", ""],
            ["power_mode", "r2"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        with pytest.raises(SheetsError, match="formula"):
            _parse_config_sheet(_make_mock_worksheet(rows))

    def test_unknown_power_mode_raises(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "bayes"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        with pytest.raises(SheetsError, match="power_mode"):
            _parse_config_sheet(_make_mock_worksheet(rows))

    def test_contrast_mode_missing_contrast_sentinel_raises(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "contrast"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        with pytest.raises(SheetsError, match=r"\[CONTRAST\]"):
            _parse_config_sheet(_make_mock_worksheet(rows))

    def test_continuous_factor_parses_to_tuple(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "temp"],
            ["power_mode", "r2"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["temp", "continuous", "20.0", "80.0"],
        ]
        _, factors, _, _, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert factors["temp"] == (20.0, 80.0)
        assert isinstance(factors["temp"], tuple)

    def test_categorical_factor_parses_to_list(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "material"],
            ["power_mode", "r2"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2", "value3"],
            ["material", "categorical", "Steel", "Aluminum", "Titanium"],
        ]
        _, factors, _, _, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert factors["material"] == ["Steel", "Aluminum", "Titanium"]

    def test_unknown_factor_type_raises(self):
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "r2"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1"],
            ["x1", "ordinal", "low"],
        ]
        with pytest.raises(SheetsError, match="ordinal"):
            _parse_config_sheet(_make_mock_worksheet(rows))


# ---------------------------------------------------------------------------
# TestWriteResults
# ---------------------------------------------------------------------------

class TestWriteResults:
    def test_write_creates_sheets_when_absent(self):
        mg = _make_mock_gspread()
        sh = MagicMock()
        ws = MagicMock()
        sh.worksheet.side_effect = mg.exceptions.WorksheetNotFound("missing")
        sh.add_worksheet.return_value = ws

        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sh, _minimal_result())

        assert sh.add_worksheet.call_count == 3

    def test_write_clears_sheets_when_clear_results_true(self):
        mg = _make_mock_gspread()
        sh = MagicMock()
        ws = MagicMock()
        ws.get_all_values.return_value = []   # "existing" index is empty
        sh.worksheet.return_value = ws

        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sh, _minimal_result(), clear_results=True)

        assert ws.clear.call_count == 3

    def test_write_skips_clear_when_clear_results_false(self):
        mg = _make_mock_gspread()
        sh = MagicMock()
        ws = MagicMock()
        ws.get_all_values.return_value = []
        sh.worksheet.return_value = ws

        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sh, _minimal_result(), clear_results=False)

        ws.clear.assert_not_called()
        sh.del_worksheet.assert_not_called()   # UX-73: False must not delete

    def test_design_df_written_with_headers(self):
        mg = _make_mock_gspread()
        sh = MagicMock()
        ws = MagicMock()
        ws.get_all_values.return_value = []
        sh.worksheet.return_value = ws

        result = _minimal_result(
            design_df=pd.DataFrame({"x1": [0.1, 0.5], "x2": [-1.0, 1.0]})
        )
        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sh, result)

        # ws.update("A1", data) is called three times (Results, Design, Buckets).
        # Find the call whose first-row contains the design column names.
        written_header_rows = [c.args[1][0] for c in ws.update.call_args_list]
        assert ["x1", "x2"] in written_header_rows

    def test_numpy_scalars_converted_to_python_types(self):
        df = pd.DataFrame({
            "a": np.array([1, 2], dtype=np.int64),
            "b": np.array([0.1, 0.2], dtype=np.float64),
        })
        rows = _df_to_rows(df)
        assert len(rows) == 3  # header + 2 data rows
        for row in rows[1:]:
            for val in row:
                assert not isinstance(val, (np.integer, np.floating)), (
                    f"Expected Python-native scalar, got {type(val)}"
                )


# ---------------------------------------------------------------------------
# TestSheetsRun
# ---------------------------------------------------------------------------

class TestSheetsRun:
    """sheets_run() with all gspread and design calls mocked."""

    def _common_patches(self, *, auth_exc=None, open_exc=None, ws_exc=None):
        """Return (patches_dict, mock_sh) for common setup."""
        mg = _make_mock_gspread()
        mock_sh = MagicMock()
        mock_sh.url = "https://docs.google.com/spreadsheets/d/FAKE"
        mock_client = MagicMock()

        get_client = MagicMock(
            side_effect=auth_exc if auth_exc else None,
            return_value=mock_client,
        )
        if open_exc:
            mock_client.open_by_url.side_effect = open_exc
            mock_client.open_by_key.side_effect = open_exc
        else:
            mock_client.open_by_url.return_value = mock_sh

        if ws_exc:
            mock_sh.worksheet.side_effect = ws_exc
        else:
            mock_sh.worksheet.return_value = MagicMock()

        return mg, get_client, mock_sh

    def test_auth_failure_raises_sheets_error(self):
        mg, get_client, mock_sh = self._common_patches(
            auth_exc=Exception("bad credentials")
        )
        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True), \
             patch.object(sheets_module, "_get_client", get_client):
            with pytest.raises(SheetsError, match="Authentication"):
                sheets_module.sheets_run("fake-id")

    def test_spreadsheet_not_found_raises_sheets_error(self):
        mg, get_client, mock_sh = self._common_patches(
            open_exc=Exception("not found")
        )
        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True), \
             patch.object(sheets_module, "_get_client", get_client):
            with pytest.raises(SheetsError, match="Could not open"):
                sheets_module.sheets_run("bad-id")

    def test_config_sheet_not_found_raises_sheets_error(self):
        mg, get_client, mock_sh = self._common_patches(
            ws_exc=Exception("worksheet not found")
        )
        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True), \
             patch.object(sheets_module, "_get_client", get_client):
            with pytest.raises(SheetsError, match="not found"):
                sheets_module.sheets_run("fake-id")

    def test_design_failure_raises_sheets_error(self):
        mg, get_client, mock_sh = self._common_patches()
        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True), \
             patch.object(sheets_module, "_get_client", get_client), \
             patch.object(sheets_module, "_parse_config_sheet",
                          return_value=_simple_parse_return()), \
             patch("lattice_doe.api.find_optimal_design",
                   side_effect=RuntimeError("solver failed")):
            with pytest.raises(SheetsError, match="Design optimisation"):
                sheets_module.sheets_run("fake-id")

    def test_successful_run_returns_spreadsheet_url(self):
        mg, get_client, mock_sh = self._common_patches()
        dr = _minimal_result()
        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True), \
             patch.object(sheets_module, "_get_client", get_client), \
             patch.object(sheets_module, "_parse_config_sheet",
                          return_value=_simple_parse_return()), \
             patch("lattice_doe.api.find_optimal_design",
                   return_value=dr), \
             patch.object(sheets_module, "_write_results"):
            result = sheets_module.sheets_run("fake-id")

        assert result["spreadsheet_url"] == "https://docs.google.com/spreadsheets/d/FAKE"

    def test_result_dict_has_all_expected_keys(self):
        mg, get_client, mock_sh = self._common_patches()
        dr = _minimal_result()
        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True), \
             patch.object(sheets_module, "_get_client", get_client), \
             patch.object(sheets_module, "_parse_config_sheet",
                          return_value=_simple_parse_return()), \
             patch("lattice_doe.api.find_optimal_design",
                   return_value=dr), \
             patch.object(sheets_module, "_write_results"):
            result = sheets_module.sheets_run("fake-id")

        assert "design_df" in result
        assert "buckets_df" in result
        assert "report" in result
        assert "spreadsheet_url" in result

    def test_write_failure_raises_sheets_error(self):
        mg, get_client, mock_sh = self._common_patches()
        dr = _minimal_result()
        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True), \
             patch.object(sheets_module, "_get_client", get_client), \
             patch.object(sheets_module, "_parse_config_sheet",
                          return_value=_simple_parse_return()), \
             patch("lattice_doe.api.find_optimal_design",
                   return_value=dr), \
             patch.object(sheets_module, "_write_results",
                          side_effect=Exception("API quota exceeded")):
            with pytest.raises(SheetsError, match="Failed to write"):
                sheets_module.sheets_run("fake-id")


# ---------------------------------------------------------------------------
# TestCreateSheetTemplate
# ---------------------------------------------------------------------------

class TestCreateSheetTemplate:
    def test_creates_spreadsheet_with_title(self):
        mg = _make_mock_gspread()
        mock_sh = MagicMock()
        mock_sh.url = "https://docs.google.com/spreadsheets/d/NEW"
        mock_sh.sheet1 = MagicMock()
        mock_client = MagicMock()
        mock_client.create.return_value = mock_sh

        with patch.object(sheets_module, "_HAS_GSPREAD", True), \
             patch.object(sheets_module, "gspread", mg, create=True), \
             patch.object(sheets_module, "_get_client",
                          MagicMock(return_value=mock_client)):
            url = sheets_module.create_sheet_template(title="My DOE")

        mock_client.create.assert_called_once_with("My DOE")
        assert url == "https://docs.google.com/spreadsheets/d/NEW"

    def test_r2_example_produces_parseable_config(self):
        """_TEMPLATE_ROWS['r2'] must round-trip through _parse_config_sheet."""
        ws = _make_mock_worksheet(_TEMPLATE_ROWS["r2"])
        formula, factors, power_cfg, design_opts, _ = _parse_config_sheet(ws)
        assert formula == "x1 + x2"
        assert isinstance(power_cfg, PowerR2Config)
        assert "x1" in factors
        assert "x2" in factors

    def test_contrast_example_produces_parseable_config(self):
        """_TEMPLATE_ROWS['contrast'] must round-trip through _parse_config_sheet."""
        ws = _make_mock_worksheet(_TEMPLATE_ROWS["contrast"])
        formula, factors, power_cfg, design_opts, _ = _parse_config_sheet(ws)
        assert formula == "x1 + x2"
        assert isinstance(power_cfg, PowerContrastConfig)
        assert "x1" in factors

    def test_unknown_example_raises_value_error(self):
        with patch.object(sheets_module, "_HAS_GSPREAD", True):
            with pytest.raises(ValueError, match="Unknown example"):
                sheets_module.create_sheet_template(example="excel")


# ---------------------------------------------------------------------------
# CR-21: blocked/pre-allocation fields in parser and template
# ---------------------------------------------------------------------------

class TestCR21BlockedPreAllocFields:
    """Verify that n_blocks, block_factor_name, preallocate_categorical,
    alloc_min_per_cell, and alloc_max_per_cell are parsed and propagated."""

    def _base_rows(self, extra_settings):
        return [
            ["[SETTINGS]", ""],
            ["formula", "x1 + x2"],
            ["power_mode", "r2"],
            ["r2_target", "0.25"],
        ] + extra_settings + [
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]

    def test_n_blocks_zero_leaves_unblocked(self):
        rows = self._base_rows([["n_blocks", "0"]])
        _, _, _, design_opts, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.n_blocks is None

    def test_n_blocks_2_enables_blocking(self):
        rows = self._base_rows([
            ["n_blocks", "2"],
            ["block_factor_name", "Batch"],
        ])
        _, _, _, design_opts, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.n_blocks == 2
        assert design_opts.block_factor_name == "Batch"

    def test_block_factor_name_default_is_block(self):
        rows = self._base_rows([["n_blocks", "3"]])
        _, _, _, design_opts, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.block_factor_name == "Block"

    def test_preallocate_categorical_true(self):
        rows = self._base_rows([
            ["preallocate_categorical", "true"],
            ["alloc_min_per_cell", "2"],
        ])
        _, _, _, design_opts, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.preallocate_categorical is True
        assert design_opts.alloc_min_per_cell == 2

    def test_preallocate_categorical_false_by_default(self):
        rows = self._base_rows([])
        _, _, _, design_opts, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.preallocate_categorical is False

    def test_alloc_max_per_cell_zero_maps_to_none(self):
        rows = self._base_rows([
            ["preallocate_categorical", "true"],
            ["alloc_max_per_cell", "0"],
        ])
        _, _, _, design_opts, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.alloc_max_per_cell is None

    def test_alloc_max_per_cell_positive_is_forwarded(self):
        rows = self._base_rows([
            ["preallocate_categorical", "true"],
            ["alloc_max_per_cell", "5"],
        ])
        _, _, _, design_opts, _ = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.alloc_max_per_cell == 5

    def test_invalid_bool_raises(self):
        rows = self._base_rows([["preallocate_categorical", "maybe"]])
        with pytest.raises(SheetsError, match="true/false"):
            _parse_config_sheet(_make_mock_worksheet(rows))

    def test_template_rows_contain_new_keys(self):
        for template_name, template_rows in _TEMPLATE_ROWS.items():
            keys = [r[0] for r in template_rows]
            assert "n_blocks" in keys, f"{template_name!r} template missing n_blocks"
            assert "block_factor_name" in keys, f"{template_name!r} missing block_factor_name"
            assert "preallocate_categorical" in keys, f"{template_name!r} missing preallocate_categorical"
            assert "alloc_min_per_cell" in keys, f"{template_name!r} missing alloc_min_per_cell"
            assert "alloc_max_per_cell" in keys, f"{template_name!r} missing alloc_max_per_cell"

    def test_template_is_still_parseable_after_additions(self):
        """All templates must still parse cleanly end-to-end."""
        for example, rows in _TEMPLATE_ROWS.items():
            ws = _make_mock_worksheet(rows)
            # Should not raise
            formula, factors, power_cfg, design_opts, multi_cfg = _parse_config_sheet(ws)
            assert formula
            assert factors
            # multiresponse template has multi_cfg, not power_cfg
            if example == "multiresponse":
                assert multi_cfg is not None
            else:
                assert power_cfg is not None


# ---------------------------------------------------------------------------
# TestCR34SheetsMultiResponseAdvancedFields
# ---------------------------------------------------------------------------

class TestCR34SheetsMultiResponseAdvancedFields:
    """CR-34: [RESPONSES] parser must accept sigma_joint and advanced per-response knobs."""

    def _base_rows(self, response_rows: list) -> list:
        """Minimal [SETTINGS]+[FACTORS]+[RESPONSES] config."""
        return [
            ["[SETTINGS]", ""],
            ["formula", "x1 + x2"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
            ["x2", "continuous", "-1.0", "1.0"],
            ["[RESPONSES]", ""],
            ["name", "power_mode", "sigma", "alpha", "power", "weight",
             "L_row", "delta", "r2_target", "formula",
             "lambda_mode", "max_n", "max_iter", "tol_power"],
        ] + response_rows

    def test_basic_r2_responses_parse(self):
        rows = self._base_rows([
            ["power_combination", "min"],
            ["Y1", "r2", "", "", "", "1.0", "", "", "0.15", "", "", "", "", ""],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "", "", "", ""],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert multi_cfg is not None
        assert len(multi_cfg.responses) == 2
        assert multi_cfg.power_combination == "min"

    def test_lambda_mode_forwarded_to_r2_config(self):
        rows = self._base_rows([
            ["Y1", "r2", "", "", "", "1.0", "", "", "0.15", "", "n_minus_p", "", "", ""],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "n", "", "", ""],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert multi_cfg.responses[0].power_cfg.lambda_mode == "n_minus_p"
        assert multi_cfg.responses[1].power_cfg.lambda_mode == "n"

    def test_max_n_forwarded(self):
        rows = self._base_rows([
            ["Y1", "r2", "", "", "", "1.0", "", "", "0.15", "", "", "300", "", ""],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "", "400", "", ""],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert multi_cfg.responses[0].power_cfg.max_n == 300
        assert multi_cfg.responses[1].power_cfg.max_n == 400

    def test_max_iter_forwarded(self):
        rows = self._base_rows([
            ["Y1", "r2", "", "", "", "1.0", "", "", "0.15", "", "", "", "50", ""],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "", "", "100", ""],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert multi_cfg.responses[0].power_cfg.max_iter == 50
        assert multi_cfg.responses[1].power_cfg.max_iter == 100

    def test_tol_power_forwarded(self):
        rows = self._base_rows([
            ["Y1", "r2", "", "", "", "1.0", "", "", "0.15", "", "", "", "", "0.005"],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "", "", "", "0.001"],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert multi_cfg.responses[0].power_cfg.tol_power == pytest.approx(0.005)
        assert multi_cfg.responses[1].power_cfg.tol_power == pytest.approx(0.001)

    def test_contrast_response_advanced_knobs(self):
        rows = self._base_rows([
            ["Y1", "contrast", "1.0", "", "", "1.0", "0,1,0", "1.0", "", "", "", "200", "50", "0.002"],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "", "", "", ""],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert isinstance(multi_cfg.responses[0].power_cfg, PowerContrastConfig)
        assert multi_cfg.responses[0].power_cfg.max_n == 200
        assert multi_cfg.responses[0].power_cfg.max_iter == 50
        assert multi_cfg.responses[0].power_cfg.tol_power == pytest.approx(0.002)

    def test_sigma_joint_parsed_2x2(self):
        rows = self._base_rows([
            ["power_combination", "min"],
            ["sigma_joint", "1.0,0.3; 0.3,1.0"],
            ["Y1", "r2", "", "", "", "1.0", "", "", "0.15", "", "", "", "", ""],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "", "", "", ""],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert multi_cfg.sigma_joint is not None
        assert multi_cfg.sigma_joint.shape == (2, 2)
        assert multi_cfg.sigma_joint[0, 1] == pytest.approx(0.3)

    def test_sigma_joint_blank_gives_none(self):
        rows = self._base_rows([
            ["sigma_joint", ""],
            ["Y1", "r2", "", "", "", "1.0", "", "", "0.15", "", "", "", "", ""],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "", "", "", ""],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert multi_cfg.sigma_joint is None

    def test_sigma_joint_invalid_raises(self):
        rows = self._base_rows([
            ["sigma_joint", "1.0,abc; 0.3,1.0"],
            ["Y1", "r2", "", "", "", "1.0", "", "", "0.15", "", "", "", "", ""],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "", "", "", ""],
        ])
        with pytest.raises(SheetsError, match="sigma_joint"):
            _parse_config_sheet(_make_mock_worksheet(rows))

    def test_power_combination_weighted_mean(self):
        rows = self._base_rows([
            ["power_combination", "weighted_mean"],
            ["Y1", "r2", "", "", "", "2.0", "", "", "0.15", "", "", "", "", ""],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "", "", "", ""],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert multi_cfg.power_combination == "weighted_mean"
        assert multi_cfg.responses[0].weight == pytest.approx(2.0)

    def test_defaults_when_advanced_cols_absent(self):
        """Rows with only basic columns still produce default advanced knob values."""
        rows = self._base_rows([
            # Only 9 columns — no formula, lambda_mode, max_n, max_iter, tol_power
            ["Y1", "r2", "", "", "", "1.0", "", "", "0.15"],
            ["Y2", "r2", "", "", "", "1.0", "", "", "0.20"],
        ])
        _, _, _, _, multi_cfg = _parse_config_sheet(_make_mock_worksheet(rows))
        assert multi_cfg.responses[0].power_cfg.lambda_mode == "n"
        assert multi_cfg.responses[0].power_cfg.max_n == 2000
        assert multi_cfg.responses[0].power_cfg.max_iter == 200
        assert multi_cfg.responses[0].power_cfg.tol_power == pytest.approx(1e-3)

    def test_multiresponse_template_parses(self):
        """The built-in 'multiresponse' template must parse without errors."""
        ws = _make_mock_worksheet(_TEMPLATE_ROWS["multiresponse"])
        _, _, power_cfg, _, multi_cfg = _parse_config_sheet(ws)
        assert power_cfg is None
        assert multi_cfg is not None
        assert len(multi_cfg.responses) == 2
        assert multi_cfg.power_combination == "min"

    def test_multiresponse_template_sigma_joint_is_none(self):
        """Built-in template has blank sigma_joint — should parse to None."""
        ws = _make_mock_worksheet(_TEMPLATE_ROWS["multiresponse"])
        _, _, _, _, multi_cfg = _parse_config_sheet(ws)
        assert multi_cfg.sigma_joint is None


# ---------------------------------------------------------------------------
# TestCR34ExtGLM — GL-9: GLM support in sheets connector
# ---------------------------------------------------------------------------

class TestCR34ExtGLM:
    """Tests for GLM power mode in the Google Sheets connector (GL-9)."""

    # ------------------------------------------------------------------
    # [SETTINGS] parsing
    # ------------------------------------------------------------------

    def test_glm_binomial_settings_parses(self):
        """power_mode='glm' with family/baseline is accepted without error."""
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1 + x2"],
            ["power_mode", "glm"],
            ["family", "binomial"],
            ["link", ""],
            ["baseline", "0.20"],
            ["[CONTRAST]", ""],
            ["L_row", "0,1,0"],
            ["delta", "0.5"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
            ["x2", "continuous", "-1.0", "1.0"],
        ]
        ws = _make_mock_worksheet(rows)
        _, _, power_cfg, _, multi_cfg = _parse_config_sheet(ws)
        from lattice_doe.config import PowerGLMContrastConfig
        assert isinstance(power_cfg, PowerGLMContrastConfig)
        assert multi_cfg is None

    def test_glm_poisson_settings_parses(self):
        """power_mode='glm' with Poisson family is accepted."""
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "glm"],
            ["family", "poisson"],
            ["baseline", "2.0"],
            ["[CONTRAST]", ""],
            ["L_row", "0,1"],
            ["delta", "0.3"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        ws = _make_mock_worksheet(rows)
        _, _, power_cfg, _, _ = _parse_config_sheet(ws)
        from lattice_doe.config import PowerGLMContrastConfig
        assert isinstance(power_cfg, PowerGLMContrastConfig)
        assert power_cfg.family == "poisson"
        assert power_cfg.baseline == pytest.approx(2.0)

    def test_glm_family_builds_glm_config(self):
        """Parsed GLM config has correct family, baseline, L, and delta."""
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1 + x2"],
            ["power_mode", "glm"],
            ["family", "binomial"],
            ["baseline", "0.30"],
            ["[CONTRAST]", ""],
            ["L_row", "0,1,0"],
            ["delta", "0.6"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
            ["x2", "continuous", "-1.0", "1.0"],
        ]
        ws = _make_mock_worksheet(rows)
        _, _, power_cfg, _, _ = _parse_config_sheet(ws)
        from lattice_doe.config import PowerGLMContrastConfig
        assert isinstance(power_cfg, PowerGLMContrastConfig)
        assert power_cfg.family == "binomial"
        assert power_cfg.baseline == pytest.approx(0.30)
        np.testing.assert_array_equal(power_cfg.L, [[0, 1, 0]])
        np.testing.assert_array_equal(power_cfg.delta, [0.6])

    def test_glm_missing_baseline_raises(self):
        """Omitting 'baseline' when power_mode='glm' raises SheetsError."""
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "glm"],
            ["family", "binomial"],
            # baseline intentionally omitted
            ["[CONTRAST]", ""],
            ["L_row", "0,1"],
            ["delta", "0.5"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        ws = _make_mock_worksheet(rows)
        with pytest.raises(SheetsError, match="baseline"):
            _parse_config_sheet(ws)

    def test_glm_with_invalid_mode_raises(self):
        """An unrecognised power_mode raises SheetsError with the value shown."""
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "wald"],   # not a valid mode
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        ws = _make_mock_worksheet(rows)
        with pytest.raises(SheetsError, match="wald"):
            _parse_config_sheet(ws)

    def test_glm_sigma_ignored_gracefully(self):
        """sigma key is silently ignored when power_mode='glm'."""
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["power_mode", "glm"],
            ["family", "binomial"],
            ["baseline", "0.25"],
            ["sigma", "1.0"],   # present but should be ignored
            ["[CONTRAST]", ""],
            ["L_row", "0,1"],
            ["delta", "0.5"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
        ]
        ws = _make_mock_worksheet(rows)
        # Should not raise; sigma is irrelevant for GLM
        _, _, power_cfg, _, _ = _parse_config_sheet(ws)
        from lattice_doe.config import PowerGLMContrastConfig
        assert isinstance(power_cfg, PowerGLMContrastConfig)

    def test_glm_template_binomial_parseable(self):
        """Built-in 'glm-binomial' template parses without error."""
        ws = _make_mock_worksheet(_TEMPLATE_ROWS["glm-binomial"])
        _, _, power_cfg, _, multi_cfg = _parse_config_sheet(ws)
        from lattice_doe.config import PowerGLMContrastConfig
        assert isinstance(power_cfg, PowerGLMContrastConfig)
        assert multi_cfg is None
        assert power_cfg.family == "binomial"

    def test_glm_template_poisson_parseable(self):
        """Built-in 'glm-poisson' template parses without error."""
        ws = _make_mock_worksheet(_TEMPLATE_ROWS["glm-poisson"])
        _, _, power_cfg, _, _ = _parse_config_sheet(ws)
        from lattice_doe.config import PowerGLMContrastConfig
        assert isinstance(power_cfg, PowerGLMContrastConfig)
        assert power_cfg.family == "poisson"

    def test_responses_glm_per_response_family(self):
        """[RESPONSES] rows with power_mode='glm' build PowerGLMContrastConfig."""
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1 + x2"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
            ["x2", "continuous", "-1.0", "1.0"],
            ["[RESPONSES]", ""],
            # header
            ["name", "power_mode", "sigma", "alpha", "power", "weight",
             "L_row", "delta", "r2_target", "formula", "lambda_mode",
             "max_n", "max_iter", "tol_power", "family", "baseline"],
            # special rows
            ["power_combination", "min"],
            ["sigma_joint", ""],
            # data rows — first GLM, second R²
            ["Y1", "glm", "", "", "", "1.0",
             "0,1,0", "0.5", "", "", "", "", "", "",
             "binomial", "0.20"],
            ["Y2", "r2", "", "", "", "1.0",
             "", "", "0.15", "", "n", "", "", ""],
        ]
        ws = _make_mock_worksheet(rows)
        _, _, power_cfg, _, multi_cfg = _parse_config_sheet(ws)
        assert power_cfg is None
        assert multi_cfg is not None
        from lattice_doe.config import PowerGLMContrastConfig, PowerR2Config
        assert isinstance(multi_cfg.responses[0].power_cfg, PowerGLMContrastConfig)
        assert isinstance(multi_cfg.responses[1].power_cfg, PowerR2Config)

    def test_responses_glm_baseline_forwarded(self):
        """Baseline value from col 16 is correctly forwarded to PowerGLMContrastConfig."""
        rows = [
            ["[SETTINGS]", ""],
            ["formula", "x1"],
            ["[FACTORS]", ""],
            ["factor_name", "type", "value1", "value2"],
            ["x1", "continuous", "-1.0", "1.0"],
            ["[RESPONSES]", ""],
            ["name", "power_mode", "sigma", "alpha", "power", "weight",
             "L_row", "delta", "r2_target", "formula", "lambda_mode",
             "max_n", "max_iter", "tol_power", "family", "baseline"],
            ["power_combination", "min"],
            ["sigma_joint", ""],
            ["Y1", "glm", "", "", "", "1.0",
             "0,1", "0.4", "", "", "", "", "", "",
             "poisson", "3.0"],
            ["Y2", "r2", "", "", "", "1.0",
             "", "", "0.20", "", "n", "", "", ""],
        ]
        ws = _make_mock_worksheet(rows)
        _, _, _, _, multi_cfg = _parse_config_sheet(ws)
        from lattice_doe.config import PowerGLMContrastConfig
        y1_cfg = multi_cfg.responses[0].power_cfg
        assert isinstance(y1_cfg, PowerGLMContrastConfig)
        assert y1_cfg.baseline == pytest.approx(3.0)
        assert y1_cfg.family == "poisson"


class TestCompoundMatrixExport:
    """UX-66: compound multi-response runs must export every per-response
    matrix — a data-dependent response may not be reproducible from the
    Design or global ModelMatrix sheets — plus a name-to-worksheet index
    (response names are free-form and get slugged for worksheet titles)."""

    @staticmethod
    def _compound_result():
        res = _minimal_result()
        res["report"]["compound_criterion"] = True
        res["model_matrix"] = pd.DataFrame({"Intercept": [1.0, 1.0],
                                            "x1": [0.1, 0.5]})
        res["model_matrices"] = {
            "y1": pd.DataFrame({"Intercept": [1.0, 1.0], "x1": [0.1, 0.5]}),
            "Yield/Day": pd.DataFrame({"Intercept": [1.0, 1.0],
                                       "bs(x1)[0]": [0.2, 0.3]}),
        }
        return res

    def test_writes_per_response_worksheets_and_index(self):
        mg = _make_mock_gspread()
        sh = MagicMock()
        ws = MagicMock()
        ws.get_all_values.return_value = []
        sh.worksheet.return_value = ws
        sh.worksheets.return_value = []

        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sh, self._compound_result())

        requested = [c.args[0] for c in sh.worksheet.call_args_list]
        assert "ModelMatrix" in requested
        assert "MM_y1" in requested
        assert "MM_Yield_Day" in requested        # slugged, no separator
        assert "ModelMatrixIndex" in requested

        # the index carries the ORIGINAL names alongside the worksheet titles
        idx_updates = [
            c.args[1] for c in ws.update.call_args_list
            if c.args and isinstance(c.args[1], list)
            and c.args[1] and c.args[1][0] == ["response", "worksheet"]
        ]
        assert idx_updates, "ModelMatrixIndex content missing"
        assert ["Yield/Day", "MM_Yield_Day"] in idx_updates[0]

    def test_non_compound_result_writes_no_extra_sheets(self):
        # Empty spreadsheet: every lookup misses, so every write creates.
        mg = _make_mock_gspread()
        sh = MagicMock()
        sh.worksheet.side_effect = mg.exceptions.WorksheetNotFound("missing")
        sh.add_worksheet.return_value = MagicMock()

        res = _minimal_result()
        res["model_matrix"] = pd.DataFrame({"Intercept": [1.0, 1.0]})
        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sh, res)

        created = [c.kwargs["title"] for c in sh.add_worksheet.call_args_list]
        assert "ModelMatrixIndex" not in created
        assert not any(t.startswith("MM_") for t in created)

    def test_case_only_response_names_get_distinct_worksheets(self):
        """UX-69: case-only response names must map to casefold-distinct
        worksheet titles."""
        mg = _make_mock_gspread()
        sh = MagicMock()
        ws = MagicMock()
        ws.get_all_values.return_value = []
        sh.worksheet.return_value = ws
        sh.worksheets.return_value = []

        res = self._compound_result()
        res["model_matrices"] = {
            "Yield": pd.DataFrame({"Intercept": [1.0, 1.0]}),
            "yield": pd.DataFrame({"Intercept": [1.0, 1.0]}),
        }
        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sh, res)

        requested = [c.args[0] for c in sh.worksheet.call_args_list]
        mm = [t for t in requested if t.startswith("MM_")]
        assert len({t.casefold() for t in mm}) == len(mm), mm
        assert "MM_yield_2" in mm


class TestRepeatExportReconciliation:
    """UX-73: repeat exports into the same spreadsheet must replace the
    per-response worksheets the previous export recorded in its index.
    Google rejects titles differing only by case, so a case-changed
    response name would otherwise fail to create its worksheet — or leave
    the index pointing at the previous export's output. The fake
    spreadsheet enforces that rejection, so a bare-slug collision check
    fails these tests loudly instead of desyncing silently."""

    @staticmethod
    def _compound_result(name_to_cell):
        res = _minimal_result()
        res["report"]["compound_criterion"] = True
        res["model_matrix"] = pd.DataFrame({"Intercept": [1.0, 1.0]})
        res["model_matrices"] = {
            name: pd.DataFrame({"Intercept": [v, v]})
            for name, v in name_to_cell.items()
        }
        return res

    @staticmethod
    def _export(sp, mg, res, **kw):
        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sp, res, **kw)

    def test_case_changed_rerun_replaces_previous_worksheets(self):
        mg = _make_mock_gspread()
        sp = _FakeSpreadsheet(mg)
        self._export(sp, mg, self._compound_result({"Yield": 1.0, "Other": 1.0}))
        assert "MM_Yield" in sp.titles()

        self._export(sp, mg, self._compound_result({"yield": 2.0, "Other": 1.0}))
        mm = sorted(t for t in sp.titles() if t.startswith("MM_"))
        assert mm == ["MM_Other", "MM_yield"], mm
        idx = sp.worksheet("ModelMatrixIndex").get_all_values()
        assert idx == [["response", "worksheet"],
                       ["yield", "MM_yield"], ["Other", "MM_Other"]]
        # the replaced worksheet carries the SECOND export's matrix
        assert sp.worksheet("MM_yield").rows == [["Intercept"], [2.0], [2.0]]

    def test_clear_results_false_reuses_previous_worksheet_in_place(self):
        mg = _make_mock_gspread()
        sp = _FakeSpreadsheet(mg)
        self._export(sp, mg, self._compound_result({"Yield": 1.0, "Other": 1.0}))

        self._export(sp, mg, self._compound_result({"yield": 2.0, "Other": 3.0}),
                     clear_results=False)
        # nothing deleted; the case-variant worksheet is reused in place and
        # the index records the ACTUAL title the output lives under
        assert "MM_Yield" in sp.titles()
        assert "MM_yield" not in sp.titles()
        idx = sp.worksheet("ModelMatrixIndex").get_all_values()
        assert ["yield", "MM_Yield"] in idx
        assert sp.worksheet("MM_Yield").rows == [["Intercept"], [2.0], [2.0]]
        assert sp.worksheet("MM_Other").rows == [["Intercept"], [3.0], [3.0]]

    def test_users_own_worksheet_never_deleted_and_forces_suffix(self):
        mg = _make_mock_gspread()
        sp = _FakeSpreadsheet(mg)
        sp.add_worksheet(title="mm_yield", rows=10, cols=5)  # user's, unindexed

        self._export(sp, mg, self._compound_result({"Yield": 1.0, "Other": 1.0}))
        assert "mm_yield" in sp.titles()               # survived reconcile
        idx = sp.worksheet("ModelMatrixIndex").get_all_values()
        assert ["Yield", "MM_Yield_2"] in idx          # dodged the collision
        for _name, title in idx[1:]:
            assert title in sp.titles()

    def test_non_compound_rerun_removes_stale_compound_output(self):
        mg = _make_mock_gspread()
        sp = _FakeSpreadsheet(mg)
        self._export(sp, mg, self._compound_result({"Yield": 1.0, "Other": 1.0}))
        assert any(t.startswith("MM_") for t in sp.titles())

        self._export(sp, mg, _minimal_result())   # no compound, no basis
        assert not any(t.startswith("MM_") for t in sp.titles())
        assert "ModelMatrixIndex" not in sp.titles()
        assert "ModelMatrix" not in sp.titles()
