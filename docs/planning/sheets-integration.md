# Enhancement 16: Google Sheets Integration — Development Plan & Ticket Pack

Tracks all work for **Enhancement 16** (bidirectional Google Sheets connector).

**Rules for contributors:**
1. Before starting a ticket, set its `Status` to `Claimed` and fill in `Claimed by`.
2. When done, check the box in the Dashboard and set `Status` to `Done`.
3. Never start work on a ticket marked `Claimed` by someone else.
4. If you hit a usage limit mid-ticket, leave a `Progress note` so the next session can continue.

---

## Dashboard

| ID | Title | Epic | Status | Claimed by |
|----|-------|------|--------|------------|
| [A1](#a1-scaffolding--soft-dependency-guard) | Scaffolding + soft-dependency guard | Foundation | Done | Claude |
| [B1](#b1-config-sheet-parser) | Config sheet parser | Input parsing | Done | Claude |
| [C1](#c1-results-writer) | Results writer | Output | Done | Claude |
| [D1](#d1-create_sheet_template) | `create_sheet_template()` | Template | Done | Claude |
| [E1](#e1-sheets_run-orchestrator) | `sheets_run()` orchestrator | Orchestration | Done | Claude |
| [F1](#f1-wiring--cli-flag) | Wiring + CLI flag | Integration | Done | Claude |
| [G1](#g1-unit-tests) | Unit tests | Tests | Done | Claude |
| [G2](#g2-documentation-updates) | Documentation updates | Docs | Done | Claude |

**Progress:** 8 / 8 tickets done. ✅

---

## Design Decisions

### New file: `iopt_power_design/sheets.py`

Single new module. Keeps all gspread logic isolated — the rest of the package
never imports from `sheets.py`. Same pattern as `plot_backends.py` (Plotly) and
`report.py` (Jinja2 HTML).

### Soft dependency — same pattern as plotly

```python
try:
    import gspread
    from google.oauth2.service_account import Credentials
    _HAS_GSPREAD = True
except ImportError:
    _HAS_GSPREAD = False

_INSTALL_HINT = 'pip install "iopt-power-design[sheets]"'
```

Every public function raises `ImportError` with `_INSTALL_HINT` when
`_HAS_GSPREAD = False`. No hard import of gspread anywhere else in the package.

### Authentication modes

Two modes, selected by the `credentials` parameter:

| Value | Auth mode |
|-------|-----------|
| `"path/to/service_account.json"` | Service account — for CI/automation/servers |
| `None` | OAuth2 browser flow via `gspread.oauth()` — for interactive/desktop use |

The service account requires the spreadsheet to be shared with the service
account's email. The OAuth flow opens a browser tab the first time and caches
the token in `~/.config/gspread/`.

### Config sheet layout

The `Config` sheet uses **sentinel headers** in column A to delimit sections.
The parser scans column A for sentinels and reads key-value pairs (or tables)
between them. This makes the layout robust to adding rows within a section
without breaking offsets.

```
Column A            Column B         Column C …
─────────────────────────────────────────────────
[SETTINGS]
formula             x1 + x2
power_mode          r2               # or "contrast"
alpha               0.05
power               0.80
sigma               1.0
r2_target           0.3              # ignored when power_mode=contrast
max_n               500
criterion           I                # I, D, or A
starts              5
max_iter            1000
random_state        123

[CONTRAST]                           # only parsed when power_mode=contrast
# One spreadsheet row per contrast (row of L matrix).
# Values in column B are comma-separated floats for that L row.
# delta row: comma-separated floats (one per L row).
L_row               0,1,0
delta               1.0

[FACTORS]
# Header row is ignored; columns are: name | type | value1 | value2 | ...
# type = "continuous" → value1=low, value2=high
# type = "categorical" → value1, value2, … = levels (as many as needed)
factor_name         type             value1    value2    value3
x1                  continuous       -1.0      1.0
x2                  continuous       -1.0      1.0
material            categorical      Steel     Aluminum  Titanium
```

**Key parsing rules:**
- Section sentinels are exact strings `[SETTINGS]`, `[CONTRAST]`, `[FACTORS]`.
- A row with an empty column A (and non-empty column B) after a settings row
  is ignored (allows blank separator rows).
- Unknown keys in `[SETTINGS]` are silently ignored (forward compatibility).
- `[CONTRAST]` section is optional; required only when `power_mode=contrast`.
- In `[CONTRAST]`, all `L_row` rows are collected in order to form the L matrix.
  There must be exactly one `delta` row; its comma-separated length must match
  the number of `L_row` rows.
- In `[FACTORS]`, the first non-sentinel row is treated as a header (ignored);
  subsequent rows are factor definitions.

### Output sheet layout

Three output sheets, all written by `_write_results()`:

**`Results` sheet** — summary key-value table starting at A1:
```
Key                 Value
n                   12
p                   3
df_num              2
df_denom            9
alpha               0.05
target_power        0.80
achieved_power      0.852
noncentrality_λ     8.34
i_criterion         1.23
d_efficiency        0.91
condition_number    4.52
criterion           I
elapsed_sec         2.34
generated_at        2026-03-07 10:30:00
warnings            (blank or newline-joined warning strings)
```

**`Design` sheet** — full design DataFrame, with column headers in row 1.

**`Buckets` sheet** — buckets DataFrame (unique run combinations + counts).

If the output sheets do not exist, `_write_results()` creates them. If they
exist and `clear_results=True` (default), they are cleared before writing.

### `sheets_run()` return value

Returns the same dict as `i_optimal_powered_design()`:
```python
{
    "design_df": pd.DataFrame,
    "buckets_df": pd.DataFrame,
    "report": dict,           # same report dict as i_optimal_powered_design
    "spreadsheet_url": str,   # URL of the spreadsheet (added by sheets_run)
}
```

### Error handling

A custom `SheetsError(RuntimeError)` is raised for all sheets-specific failures
(auth errors, missing sentinel, malformed factor table, missing required key).
Underlying `gspread.exceptions.APIError` and network errors are caught and
re-raised as `SheetsError` with a descriptive message.

### `create_sheet_template()`

Creates a new Google Spreadsheet with the full Config/Results/Design/Buckets
sheet structure pre-populated with a working example (2 continuous factors,
R² mode, reasonable defaults). Returns the URL of the created spreadsheet.
This is the "starter kit" for new users.

```python
url = create_sheet_template(
    title="My DOE — iopt template",
    credentials="service_account.json",   # or None for OAuth
    example="r2",     # "r2" or "contrast" — which example to populate
)
```

### `[sheets]` extras in `pyproject.toml`

```toml
[project.optional-dependencies.sheets]
gspread = ">=6.0"
google-auth = ">=2.0"
```

Also add both packages to the `all` extras group.

### CLI integration

New `--sheets URL` flag on `iopt-design`. The credentials path is read from
`--sheets-credentials PATH` or, if absent, from the `GOOGLE_APPLICATION_CREDENTIALS`
environment variable (standard GCP convention). Falls back to OAuth browser
flow if neither is set.

```
iopt-design --sheets "https://docs.google.com/spreadsheets/d/…"
iopt-design --sheets "SPREADSHEET_ID" --sheets-credentials /path/to/sa.json
```

When `--sheets` is used, `--config` is not required (config is read from the
spreadsheet). If both are provided, the spreadsheet config takes precedence.

### Callers table

| File | Change | Reason |
|------|--------|--------|
| `sheets.py` | New file | All sheets logic |
| `pyproject.toml` | Add `[sheets]` extras | Dependency declaration |
| `__init__.py` | Export `sheets_run`, `create_sheet_template` | Public API |
| `cli.py` | Add `--sheets`, `--sheets-credentials` flags | CLI integration |
| `ENHANCEMENTS.md` | Mark #16 Done | Completion tracking |

---

## Ticket details

---

## Epic A — Foundation

---

### A1 Scaffolding + soft-dependency guard

**Status:** Open
**Claimed by:**
**Est.:** 30 minutes
**Depends on:** nothing

**What to do:**

Create `iopt_power_design/sheets.py` with:

1. Module docstring explaining the bidirectional Sheets connector, auth modes,
   and sheet layout overview.

2. Soft-dependency guard:
   ```python
   try:
       import gspread
       from google.oauth2.service_account import Credentials
       _HAS_GSPREAD = True
   except ImportError:
       _HAS_GSPREAD = False

   _INSTALL_HINT = 'pip install "iopt-power-design[sheets]"'
   ```

3. Custom exception:
   ```python
   class SheetsError(RuntimeError):
       """Raised for all Google Sheets integration failures."""
   ```

4. Section sentinel constants (used by parser and template builder):
   ```python
   _SENTINEL_SETTINGS  = "[SETTINGS]"
   _SENTINEL_CONTRAST  = "[CONTRAST]"
   _SENTINEL_FACTORS   = "[FACTORS]"
   ```

5. Auth helper:
   ```python
   def _get_client(credentials: Optional[str]) -> "gspread.Client":
       """Return an authenticated gspread Client.

       Parameters
       ----------
       credentials : str or None
           Path to a service account JSON file.  Pass None to use the
           OAuth2 browser flow (``gspread.oauth()``).
       """
       if not _HAS_GSPREAD:
           raise ImportError(f"gspread is required. {_INSTALL_HINT}")
       if credentials is not None:
           scope = [
               "https://spreadsheets.google.com/feeds",
               "https://www.googleapis.com/auth/drive",
           ]
           creds = Credentials.from_service_account_file(credentials, scopes=scope)
           return gspread.authorize(creds)
       return gspread.oauth()
   ```

6. `__all__` stub (will be filled in E1 and D1):
   ```python
   __all__: list[str] = []
   ```

**Invariants:**
- No imports from `design`, `iopt_search`, `candidate` — pure gspread + standard library + numpy/pandas.
- `_get_client` is module-private.
- The file is importable even when gspread is not installed (guard works correctly).

**Verification:**
```python
# With gspread absent:
import sys
sys.modules['gspread'] = None          # simulate missing package
import importlib
import iopt_power_design.sheets as s
assert not s._HAS_GSPREAD
assert "sheets" in s._INSTALL_HINT
```

**Acceptance criteria:**
- [ ] File is importable with and without gspread installed.
- [ ] `SheetsError` is a subclass of `RuntimeError`.
- [ ] `_get_client` raises `ImportError` (not `SheetsError`) when gspread absent.

---

## Epic B — Input parsing

---

### B1 Config sheet parser

**Status:** Open
**Claimed by:**
**Est.:** 2 hours
**Depends on:** A1

**What to do:**

Implement two functions in `sheets.py`:

#### `_read_all_rows(worksheet) -> list[list[str]]`

Read all values from the worksheet as a list of rows (each row is a list of
cell values as strings). Trailing empty cells within a row are stripped.
Completely empty rows are kept (they may be separators).

```python
def _read_all_rows(worksheet: "gspread.Worksheet") -> list[list[str]]:
    return worksheet.get_all_values()
```

#### `_parse_config_sheet(worksheet) -> tuple[str, dict, Union[PowerContrastConfig, PowerR2Config], DesignOptions]`

Parse the Config worksheet and return `(formula, factors, power_cfg, design_opts)`.

**Algorithm:**

1. Call `_read_all_rows(worksheet)`.
2. Find row indices of the three sentinel headers.
   `_SENTINEL_SETTINGS` is required — raise `SheetsError` if absent.
   `_SENTINEL_CONTRAST` is optional (required only when `power_mode="contrast"`).
   `_SENTINEL_FACTORS` is required — raise `SheetsError` if absent.
3. Parse `[SETTINGS]` section (rows between `_SENTINEL_SETTINGS` and the next sentinel):
   - Build a `dict[str, str]` of `{key: value}` from col A / col B pairs.
   - Skip rows where col A is empty.
   - Extract required keys: `formula`, `power_mode`.
   - Extract optional keys with defaults (see table below).
4. Build `DesignOptions` from settings keys.
5. If `power_mode == "r2"`:
   Build `PowerR2Config(r2_target=..., alpha=..., power=..., max_n=...)`.
6. If `power_mode == "contrast"`:
   Parse `[CONTRAST]` section (required):
   - Collect all rows where col A == `"L_row"` → each col B is one row of L
     (comma-separated floats).
   - Find the row where col A == `"delta"` → col B is comma-separated floats.
   - Build `np.ndarray` L (shape: n_contrasts × p_guess; validated later by
     `i_optimal_powered_design`) and delta.
   - Build `PowerContrastConfig(L=L, delta=delta, alpha=..., power=..., sigma=..., max_n=...)`.
7. Parse `[FACTORS]` section:
   - Skip the header row (first non-sentinel row).
   - For each subsequent non-empty row:
     - col A = factor name
     - col B = type (`"continuous"` or `"categorical"`)
     - cols C+ = values
   - `continuous` → `factors[name] = (float(val1), float(val2))`
   - `categorical` → `factors[name] = [v for v in values if v != ""]`
   - Raise `SheetsError` for unrecognised type or malformed values.
8. Return `(formula, factors, power_cfg, design_opts)`.

**Settings keys and defaults:**

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `formula` | str | required | |
| `power_mode` | str | required | `"r2"` or `"contrast"` |
| `alpha` | float | 0.05 | |
| `power` | float | 0.80 | |
| `sigma` | float | 1.0 | Used for contrast mode and stored in PowerR2Config for consistency |
| `r2_target` | float | 0.25 | Only used in R² mode |
| `max_n` | int | 500 | |
| `criterion` | str | `"I"` | |
| `starts` | int | 5 | → `DesignOptions.starts` |
| `max_iter` | int | 1000 | → `DesignOptions.max_iter` |
| `random_state` | int | 123 | → `DesignOptions.random_state` |

**Error cases to test:**
- `[SETTINGS]` sentinel missing → `SheetsError`
- `[FACTORS]` sentinel missing → `SheetsError`
- `formula` key missing → `SheetsError`
- `power_mode` not `"r2"` or `"contrast"` → `SheetsError`
- `power_mode="contrast"` but `[CONTRAST]` sentinel missing → `SheetsError`
- `delta` length ≠ number of `L_row` rows → `SheetsError`
- Factor type not `"continuous"` or `"categorical"` → `SheetsError`
- Continuous factor with non-numeric value → `SheetsError`
- Continuous factor with fewer than 2 values → `SheetsError`
- Categorical factor with fewer than 2 levels → `SheetsError` (degenerate)

**Imports needed at top of sheets.py:**
```python
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from .config import PowerContrastConfig, PowerR2Config, DesignOptions
```

**Acceptance criteria:**
- [ ] `_parse_config_sheet` returns correct types for both `r2` and `contrast` modes.
- [ ] Mixed continuous + categorical factors parse correctly.
- [ ] All error cases raise `SheetsError` (not bare `ValueError` or `KeyError`).
- [ ] `[CONTRAST]` section parses multi-row L matrices correctly.

---

## Epic C — Output writing

---

### C1 Results writer

**Status:** Open
**Claimed by:**
**Est.:** 1.5 hours
**Depends on:** A1

**What to do:**

Implement `_write_results()` in `sheets.py`.

```python
def _write_results(
    spreadsheet: "gspread.Spreadsheet",
    result: Dict[str, Any],
    results_sheet: str = "Results",
    design_sheet: str = "Design",
    buckets_sheet: str = "Buckets",
    clear_results: bool = True,
) -> None:
```

**Algorithm:**

Helper: `_get_or_create_sheet(spreadsheet, title) -> gspread.Worksheet`:
Try `spreadsheet.worksheet(title)`; if `gspread.exceptions.WorksheetNotFound`
is raised, call `spreadsheet.add_worksheet(title=title, rows=1000, cols=20)`.

1. **Results sheet** (`results_sheet`):
   - Get or create the sheet.
   - If `clear_results`, clear it with `ws.clear()`.
   - Build a list of `[key, value]` rows from `result["report"]`:
     ```
     ["n",                  result["report"]["n"]],
     ["p",                  result["report"]["p"]],
     ["df_num",             result["report"]["df_num"]],
     ["df_denom",           result["report"]["df_denom"]],
     ["alpha",              result["report"]["alpha"]],
     ["target_power",       result["report"]["target_power"]],
     ["achieved_power",     result["report"]["achieved_power"]],
     ["noncentrality_λ",    result["report"]["noncentrality_lambda"]],
     ["i_criterion",        result["report"]["diagnostics"].get("i_criterion", "")],
     ["d_efficiency",       result["report"]["diagnostics"].get("d_efficiency", "")],
     ["condition_number",   result["report"]["diagnostics"].get("condition_number", "")],
     ["criterion",          result["report"]["criterion"]],
     ["elapsed_sec",        result["report"]["elapsed_sec"]],
     ["generated_at",       datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")],
     ["warnings",           "\n".join(result["report"].get("warnings", []))],
     ```
   - Write with `ws.update("A1", rows)`.

2. **Design sheet** (`design_sheet`):
   - Get or create the sheet.
   - If `clear_results`, clear it.
   - Convert `result["design_df"]` to a list-of-lists (header row + data rows),
     coercing all values to Python native types (int/float/str) with `.tolist()`.
   - Write with `ws.update("A1", [headers] + data_rows)`.

3. **Buckets sheet** (`buckets_sheet`):
   - Same pattern as Design sheet but for `result["buckets_df"]`.

**Important:** All numeric values must be Python `float` or `int` (not numpy
scalars). Use `float(x)` / `int(x)` conversions before passing to `ws.update()`.
gspread will reject numpy scalars with a JSON serialization error.

**Import needed:** `from datetime import datetime`

**Acceptance criteria:**
- [ ] All three sheets are written correctly.
- [ ] `_get_or_create_sheet` creates a sheet if absent.
- [ ] `clear_results=False` does not call `ws.clear()`.
- [ ] NumPy scalar values are converted to Python native types.

---

## Epic D — Template

---

### D1 `create_sheet_template()`

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** A1

**What to do:**

Implement `create_sheet_template()` in `sheets.py`.

```python
def create_sheet_template(
    title: str = "iopt-power-design template",
    credentials: Optional[str] = None,
    example: str = "r2",
) -> str:
    """Create a new Google Spreadsheet pre-populated with a working example.

    Parameters
    ----------
    title : str
        Title of the new spreadsheet.
    credentials : str or None
        Path to service account JSON, or None for OAuth browser flow.
    example : {"r2", "contrast"}
        Which example to populate in the Config sheet.

    Returns
    -------
    str
        URL of the created spreadsheet.
    """
```

**Algorithm:**

1. Guard: if not `_HAS_GSPREAD`, raise `ImportError`.
2. Validate `example` ∈ `{"r2", "contrast"}`.
3. `client = _get_client(credentials)`.
4. `sh = client.create(title)`.
   Make it accessible: `sh.share(None, perm_type="anyone", role="writer")`.
   *(Note: for service accounts, the spreadsheet is owned by the SA email;
   for OAuth, owned by the authenticated user. The `share` call makes it
   accessible by link — appropriate for a personal template but callers
   can skip sharing by passing their SA email instead.)*
5. Rename the default "Sheet1" to "Config":
   `sh.sheet1.update_title("Config")`.
6. Populate the Config sheet with the example data:
   - For `example="r2"`:
     ```
     [SETTINGS]
     formula          x1 + x2
     power_mode       r2
     alpha            0.05
     power            0.80
     sigma            1.0
     r2_target        0.30
     max_n            500
     criterion        I
     starts           5
     max_iter         1000
     random_state     123

     [FACTORS]
     factor_name      type         value1   value2
     x1               continuous   -1.0     1.0
     x2               continuous   -1.0     1.0
     ```
   - For `example="contrast"`:
     ```
     [SETTINGS]
     formula          x1 + x2
     power_mode       contrast
     alpha            0.05
     power            0.80
     sigma            1.0
     max_n            500
     criterion        I
     starts           5
     max_iter         1000
     random_state     123

     [CONTRAST]
     L_row            0,1,0
     delta            1.0

     [FACTORS]
     factor_name      type         value1   value2
     x1               continuous   -1.0     1.0
     x2               continuous   -1.0     1.0
     ```
7. Add empty Results, Design, Buckets sheets:
   ```python
   for name in ("Results", "Design", "Buckets"):
       sh.add_worksheet(title=name, rows=200, cols=20)
   ```
8. Return `sh.url`.

**Invariants:**
- Do not call `i_optimal_powered_design` — this is a pure template setup function.
- Use `ws.update("A1", rows)` where `rows` is a list of `[col_A, col_B]` pairs.

**Acceptance criteria:**
- [ ] Raises `ImportError` (not `SheetsError`) when gspread absent.
- [ ] Returns a URL string.
- [ ] Both `"r2"` and `"contrast"` examples produce parseable Config sheets.
- [ ] Raises `ValueError` for unknown `example` value.

---

## Epic E — Orchestration

---

### E1 `sheets_run()` orchestrator

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** A1, B1, C1

**What to do:**

Implement `sheets_run()` in `sheets.py` and update `__all__`.

```python
def sheets_run(
    spreadsheet_url_or_id: str,
    *,
    credentials: Optional[str] = None,
    config_sheet: str = "Config",
    results_sheet: str = "Results",
    design_sheet: str = "Design",
    buckets_sheet: str = "Buckets",
    clear_results: bool = True,
) -> Dict[str, Any]:
    """Read config from a Google Sheet, run the design, and write results back.

    Parameters
    ----------
    spreadsheet_url_or_id : str
        Full URL or spreadsheet ID of the target Google Spreadsheet.
    credentials : str or None
        Path to a service account JSON file.  Pass None for OAuth browser flow.
    config_sheet : str, default "Config"
        Name of the worksheet containing the design configuration.
    results_sheet : str, default "Results"
        Name of the worksheet to write the summary results to.
    design_sheet : str, default "Design"
        Name of the worksheet to write the design table to.
    buckets_sheet : str, default "Buckets"
        Name of the worksheet to write the run-frequency buckets to.
    clear_results : bool, default True
        If True, clear the output sheets before writing.

    Returns
    -------
    dict
        Same keys as ``i_optimal_powered_design()`` plus:
        - ``"spreadsheet_url"`` (str): URL of the spreadsheet.

    Raises
    ------
    ImportError
        If gspread / google-auth are not installed.
    SheetsError
        If the config sheet cannot be parsed or the API call fails.
    """
```

**Algorithm:**

1. Guard: if not `_HAS_GSPREAD`, raise `ImportError(f"gspread required. {_INSTALL_HINT}")`.
2. `client = _get_client(credentials)`.
   Wrap any `Exception` from `_get_client` in `SheetsError(f"Authentication failed: {e}")`.
3. Open the spreadsheet:
   ```python
   try:
       sh = client.open_by_url(spreadsheet_url_or_id)
   except gspread.exceptions.SpreadsheetNotFound:
       try:
           sh = client.open_by_key(spreadsheet_url_or_id)
       except gspread.exceptions.SpreadsheetNotFound:
           raise SheetsError(f"Spreadsheet not found: {spreadsheet_url_or_id!r}")
   ```
4. Get the Config worksheet:
   ```python
   try:
       ws_config = sh.worksheet(config_sheet)
   except gspread.exceptions.WorksheetNotFound:
       raise SheetsError(f"Config sheet {config_sheet!r} not found in spreadsheet.")
   ```
5. Parse config:
   ```python
   try:
       formula, factors, power_cfg, design_opts = _parse_config_sheet(ws_config)
   except SheetsError:
       raise  # already wrapped
   except Exception as e:
       raise SheetsError(f"Config sheet parsing failed: {e}") from e
   ```
6. Run the design:
   ```python
   from .api import i_optimal_powered_design
   try:
       result = i_optimal_powered_design(
           formula=formula,
           factors=factors,
           power_cfg=power_cfg,
           design_opts=design_opts,
       )
   except Exception as e:
       raise SheetsError(f"Design generation failed: {e}") from e
   ```
7. Write results:
   ```python
   try:
       _write_results(
           spreadsheet=sh,
           result=result,
           results_sheet=results_sheet,
           design_sheet=design_sheet,
           buckets_sheet=buckets_sheet,
           clear_results=clear_results,
       )
   except SheetsError:
       raise
   except Exception as e:
       raise SheetsError(f"Failed to write results to spreadsheet: {e}") from e
   ```
8. Add `"spreadsheet_url"` to result and return:
   ```python
   result["spreadsheet_url"] = sh.url
   return result
   ```
9. Update `__all__`:
   ```python
   __all__ = ["SheetsError", "sheets_run", "create_sheet_template"]
   ```

**Lazy import of `api.py`:**
Import `i_optimal_powered_design` inside `sheets_run()` (step 6 above), not at
module level. This prevents a circular import: `api.py` does not import
`sheets.py`, and `sheets.py` should not be in `api.py`'s import chain.

**Acceptance criteria:**
- [ ] `sheets_run` raises `ImportError` when gspread absent.
- [ ] Auth failure → `SheetsError`.
- [ ] Missing spreadsheet → `SheetsError`.
- [ ] Missing Config sheet → `SheetsError`.
- [ ] Parse error → `SheetsError`.
- [ ] Design error → `SheetsError`.
- [ ] Write error → `SheetsError`.
- [ ] Returns `"spreadsheet_url"` key.
- [ ] `__all__` updated.

---

## Epic F — Wiring + CLI

---

### F1 Wiring + CLI flag

**Status:** Open
**Claimed by:**
**Est.:** 45 minutes
**Depends on:** E1

**What to do:**

**`pyproject.toml`** — add `[sheets]` extras and update `all`:
```toml
[project.optional-dependencies.sheets]
gspread = ">=6.0"
google-auth = ">=2.0"
```

In the `all` group, add:
```toml
  # sheets
  "gspread>=6.0",
  "google-auth>=2.0",
```

**`iopt_power_design/__init__.py`** — add exports:
```python
from .sheets import SheetsError, sheets_run, create_sheet_template  # noqa: F401
```

And add to `__all__`:
```python
"SheetsError",
"sheets_run",
"create_sheet_template",
```

**`iopt_power_design/cli.py`** — add two new flags to the argument parser:

```python
parser.add_argument(
    "--sheets",
    metavar="URL_OR_ID",
    default=None,
    help=(
        "Google Spreadsheet URL or ID. Reads config from the 'Config' sheet "
        "and writes design results back to Results/Design/Buckets sheets. "
        "Requires: pip install 'iopt-power-design[sheets]'"
    ),
)
parser.add_argument(
    "--sheets-credentials",
    metavar="PATH",
    default=None,
    help=(
        "Path to a service account JSON credentials file for Google Sheets. "
        "If omitted, falls back to the GOOGLE_APPLICATION_CREDENTIALS env var, "
        "then to OAuth2 browser flow."
    ),
)
```

In the `main()` function, after parsing args:
```python
if args.sheets:
    try:
        from iopt_power_design.sheets import sheets_run, SheetsError
    except ImportError:
        print(
            "Error: Google Sheets support requires gspread.\n"
            "  pip install 'iopt-power-design[sheets]'",
            file=sys.stderr,
        )
        sys.exit(1)

    creds = (
        args.sheets_credentials
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    try:
        result = sheets_run(args.sheets, credentials=creds)
    except SheetsError as e:
        print(f"Sheets error: {e}", file=sys.stderr)
        sys.exit(1)

    r = result["report"]
    print(
        f"Design written to spreadsheet.\n"
        f"  n={r['n']}, p={r['p']}, "
        f"achieved_power={r['achieved_power']:.3f}, "
        f"elapsed={r['elapsed_sec']:.1f}s\n"
        f"  {result['spreadsheet_url']}"
    )
    sys.exit(0)
```

This block runs **before** the YAML config loading block, so `--sheets` is an
independent execution path.

**Acceptance criteria:**
- [ ] `pyproject.toml` has `[sheets]` extras.
- [ ] `gspread` and `google-auth` are in the `all` group.
- [ ] `sheets_run` and `create_sheet_template` are in `iopt_power_design.__all__`.
- [ ] `--sheets` and `--sheets-credentials` flags exist in CLI.
- [ ] CLI exits cleanly with a message when gspread not installed.
- [ ] `GOOGLE_APPLICATION_CREDENTIALS` env var is honoured as a fallback.

---

## Epic G — Tests & documentation

---

### G1 Unit tests

**Status:** Open
**Claimed by:**
**Est.:** 2 hours
**Depends on:** E1, F1

**What to do:**

Create `tests/test_sheets.py`. All tests mock gspread — no live API calls.

**Test strategy:**
- Use `unittest.mock.patch` and `MagicMock` throughout.
- The `_HAS_GSPREAD = False` branch is tested by patching the module flag.
- A reusable `_make_mock_worksheet(rows)` helper builds a mock worksheet whose
  `get_all_values()` returns the given list-of-lists.

**Test classes:**

```
TestSheetsImportGuard         (2 tests)
  test_sheets_run_raises_import_error_when_no_gspread
  test_create_template_raises_import_error_when_no_gspread

TestGetClient                 (2 tests)
  test_service_account_path_calls_from_service_account_file
  test_none_credentials_calls_gspread_oauth

TestParseConfigSheet          (14 tests)
  test_r2_mode_parses_formula_and_factors
  test_r2_mode_returns_power_r2_config
  test_r2_mode_defaults_applied_when_keys_absent
  test_contrast_mode_single_contrast
  test_contrast_mode_multi_row_L_matrix
  test_contrast_mode_delta_length_mismatch_raises
  test_missing_settings_sentinel_raises
  test_missing_factors_sentinel_raises
  test_missing_formula_key_raises
  test_unknown_power_mode_raises
  test_contrast_mode_missing_contrast_sentinel_raises
  test_continuous_factor_parses_to_tuple
  test_categorical_factor_parses_to_list
  test_unknown_factor_type_raises

TestWriteResults              (5 tests)
  test_write_creates_sheets_when_absent
  test_write_clears_sheets_when_clear_results_true
  test_write_skips_clear_when_clear_results_false
  test_design_df_written_with_headers
  test_numpy_scalars_converted_to_python_types

TestSheetsRun                 (7 tests)
  test_auth_failure_raises_sheets_error
  test_spreadsheet_not_found_raises_sheets_error
  test_config_sheet_not_found_raises_sheets_error
  test_design_failure_raises_sheets_error
  test_successful_run_returns_spreadsheet_url
  test_result_dict_has_all_expected_keys
  test_write_failure_raises_sheets_error

TestCreateSheetTemplate       (4 tests)
  test_creates_spreadsheet_with_title
  test_r2_example_produces_parseable_config
  test_contrast_example_produces_parseable_config
  test_unknown_example_raises_value_error
```

**Total: ~34 tests.**

**Sample test pattern:**
```python
from unittest.mock import patch, MagicMock, call
import pytest

def _make_mock_worksheet(rows):
    ws = MagicMock()
    ws.get_all_values.return_value = rows
    return ws

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
        from iopt_power_design.sheets import _parse_config_sheet
        ws = _make_mock_worksheet(rows)
        formula, factors, power_cfg, design_opts = _parse_config_sheet(ws)
        assert formula == "x1 + x2"
        assert factors["x1"] == (-1.0, 1.0)
        assert factors["x2"] == (-1.0, 1.0)
        from iopt_power_design.config import PowerR2Config
        assert isinstance(power_cfg, PowerR2Config)
```

**Acceptance criteria:**
- [ ] All ~34 tests pass.
- [ ] No live API calls — all gspread interactions are mocked.
- [ ] `_HAS_GSPREAD = False` path is explicitly tested.

---

### G2 Documentation updates

**Status:** Open
**Claimed by:**
**Est.:** 20 minutes
**Depends on:** G1

**What to do:**

1. **`ENHANCEMENTS.md`** — move Enhancement #16 from backlog to Completed table.
   Add a row similar to the Plotly entry:
   ```
   | 16 | Google Sheets integration | `sheets.py` (new), `__init__.py`, `pyproject.toml`, `cli.py`, `tests/test_sheets.py` | Bidirectional connector: `sheets_run()` reads Config sheet (factors, formula, power config) and writes Design/Results/Buckets back. `create_sheet_template()` creates a starter spreadsheet. Soft gspread dep (`[sheets]` extras). `--sheets URL` CLI flag. ~34 new tests. |
   ```

2. **`docs/planning/sheets-integration.md`** (this file) — update all ticket
   statuses to Done and set `Progress: 8 / 8 tickets done. ✅`.

3. **`README.md`** — add a brief "Google Sheets" subsection under the
   "Export & Integration" section (or equivalent), showing:
   ```python
   from iopt_power_design import sheets_run, create_sheet_template

   # Create a template spreadsheet (once)
   url = create_sheet_template(title="My DOE", credentials="sa.json")

   # Fill in Config sheet, then run:
   result = sheets_run(url, credentials="sa.json")
   print(f"Optimal n = {result['report']['n']}")
   ```
   And the CLI one-liner:
   ```
   iopt-design --sheets "SPREADSHEET_URL" --sheets-credentials sa.json
   ```

**Acceptance criteria:**
- [ ] ENHANCEMENTS.md #16 entry marked Done.
- [ ] Ticket pack progress 8/8.
- [ ] README has a minimal Sheets usage example.

---

## Summary: inputs and outputs

| Direction | What | Where in spreadsheet |
|-----------|------|---------------------|
| Read | Formula, power mode, power params | `Config` — `[SETTINGS]` section |
| Read | L matrix, delta | `Config` — `[CONTRAST]` section |
| Read | Factor specifications | `Config` — `[FACTORS]` section |
| Write | Summary stats (n, power, elapsed, …) | `Results` sheet |
| Write | Full design DataFrame | `Design` sheet |
| Write | Run-frequency buckets | `Buckets` sheet |

## Dependency graph

```
A1 (scaffolding)
  └── B1 (parser) ──┐
  └── C1 (writer) ──┤
  └── D1 (template) ┤
                    ├── E1 (sheets_run) ── F1 (wiring) ── G1 (tests) ── G2 (docs)
```

B1 and C1 and D1 can all be worked in parallel after A1.
E1 requires B1 + C1 (but not D1).
F1 requires E1 (and D1 for `__init__.py` exports).
