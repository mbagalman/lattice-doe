# tests/test_sheets.py
# License: MIT
"""Unit tests for iopt_power_design.sheets — all gspread calls mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import iopt_power_design.sheets as sheets_module
from iopt_power_design.sheets import (
    SheetsError,
    _df_to_rows,
    _parse_config_sheet,
    _TEMPLATE_ROWS,
)
from iopt_power_design.config import DesignOptions, PowerContrastConfig, PowerR2Config


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


def _minimal_result(design_df=None, buckets_df=None):
    """Minimal result dict matching i_optimal_powered_design() output."""
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
        formula, factors, power_cfg, design_opts = _parse_config_sheet(
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
        _, _, power_cfg, _ = _parse_config_sheet(_make_mock_worksheet(rows))
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
        _, _, power_cfg, design_opts = _parse_config_sheet(_make_mock_worksheet(rows))
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
        _, _, power_cfg, _ = _parse_config_sheet(_make_mock_worksheet(rows))
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
        _, _, power_cfg, _ = _parse_config_sheet(_make_mock_worksheet(rows))
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
        _, factors, _, _ = _parse_config_sheet(_make_mock_worksheet(rows))
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
        _, factors, _, _ = _parse_config_sheet(_make_mock_worksheet(rows))
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
        sh.worksheet.return_value = ws

        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sh, _minimal_result(), clear_results=True)

        assert ws.clear.call_count == 3

    def test_write_skips_clear_when_clear_results_false(self):
        mg = _make_mock_gspread()
        sh = MagicMock()
        ws = MagicMock()
        sh.worksheet.return_value = ws

        with patch.object(sheets_module, "gspread", mg, create=True):
            sheets_module._write_results(sh, _minimal_result(), clear_results=False)

        ws.clear.assert_not_called()

    def test_design_df_written_with_headers(self):
        mg = _make_mock_gspread()
        sh = MagicMock()
        ws = MagicMock()
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
             patch("iopt_power_design.api.i_optimal_powered_design",
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
             patch("iopt_power_design.api.i_optimal_powered_design",
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
             patch("iopt_power_design.api.i_optimal_powered_design",
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
             patch("iopt_power_design.api.i_optimal_powered_design",
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
        formula, factors, power_cfg, design_opts = _parse_config_sheet(ws)
        assert formula == "x1 + x2"
        assert isinstance(power_cfg, PowerR2Config)
        assert "x1" in factors
        assert "x2" in factors

    def test_contrast_example_produces_parseable_config(self):
        """_TEMPLATE_ROWS['contrast'] must round-trip through _parse_config_sheet."""
        ws = _make_mock_worksheet(_TEMPLATE_ROWS["contrast"])
        formula, factors, power_cfg, design_opts = _parse_config_sheet(ws)
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
        _, _, _, design_opts = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.n_blocks is None

    def test_n_blocks_2_enables_blocking(self):
        rows = self._base_rows([
            ["n_blocks", "2"],
            ["block_factor_name", "Batch"],
        ])
        _, _, _, design_opts = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.n_blocks == 2
        assert design_opts.block_factor_name == "Batch"

    def test_block_factor_name_default_is_block(self):
        rows = self._base_rows([["n_blocks", "3"]])
        _, _, _, design_opts = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.block_factor_name == "Block"

    def test_preallocate_categorical_true(self):
        rows = self._base_rows([
            ["preallocate_categorical", "true"],
            ["alloc_min_per_cell", "2"],
        ])
        _, _, _, design_opts = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.preallocate_categorical is True
        assert design_opts.alloc_min_per_cell == 2

    def test_preallocate_categorical_false_by_default(self):
        rows = self._base_rows([])
        _, _, _, design_opts = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.preallocate_categorical is False

    def test_alloc_max_per_cell_zero_maps_to_none(self):
        rows = self._base_rows([
            ["preallocate_categorical", "true"],
            ["alloc_max_per_cell", "0"],
        ])
        _, _, _, design_opts = _parse_config_sheet(_make_mock_worksheet(rows))
        assert design_opts.alloc_max_per_cell is None

    def test_alloc_max_per_cell_positive_is_forwarded(self):
        rows = self._base_rows([
            ["preallocate_categorical", "true"],
            ["alloc_max_per_cell", "5"],
        ])
        _, _, _, design_opts = _parse_config_sheet(_make_mock_worksheet(rows))
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
        """Both templates must still parse cleanly end-to-end."""
        for example, rows in _TEMPLATE_ROWS.items():
            ws = _make_mock_worksheet(rows)
            # Should not raise
            formula, factors, power_cfg, design_opts = _parse_config_sheet(ws)
            assert formula
            assert factors
