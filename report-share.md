# Shareable Report — Development Plan & Ticket Pack

Tracks all work for **Enhancement #14** (PDF / HTML shareable report).

**Rules for contributors:**
1. Before starting a ticket, set its `Status` to `Claimed` and fill in `Claimed by`.
2. When done, check the box in the Dashboard and set `Status` to `Done`.
3. Never start work on a ticket marked `Claimed` by someone else — pick a different `Open` ticket or coordinate first.
4. If you hit a usage limit mid-ticket, leave a `Progress note` in the ticket card so the next session can continue without re-reading the whole codebase.

---

## Dashboard

| ID | Title | Epic | Status | Claimed by |
|----|-------|------|--------|------------|
| [A1](#a1-report-module-skeleton--packaging) | Report module skeleton & packaging | Infrastructure | Done | Claude |
| [A2](#a2-jinja2-html-template-skeleton) | Jinja2 HTML template skeleton | Infrastructure | Done | Claude |
| [B1](#b1-config-summary-section) | Config summary section | Report Content | Done | Claude |
| [B2](#b2-power-metrics-section) | Power metrics section | Report Content | Done | Claude |
| [B3](#b3-design--buckets-tables-section) | Design & buckets tables section | Report Content | Done | Claude |
| [B4](#b4-diagnostics-section) | Diagnostics section | Report Content | Done | Claude |
| [B5](#b5-embedded-power-curve-figure) | Embedded power curve figure | Report Content | Done | Claude |
| [C1](#c1-standalone-html-output) | Standalone HTML output | Export Backends | Done | Claude |
| [C2](#c2-optional-pdf-export) | Optional PDF export (weasyprint) | Export Backends | Done | Claude |
| [D1](#d1-api-integration) | API integration (`export_report_to=`) | Integration | Done | Claude |
| [D2](#d2-cli-integration) | CLI `--html-report` flag | Integration | Done | Claude |
| [D3](#d3-streamlit-download-button) | Streamlit "Download report" button | Integration | Done | Claude |
| [E1](#e1-unit-tests) | Unit tests | Tests & Docs | Open | |
| [E2](#e2-documentation-updates) | Documentation updates | Tests & Docs | Open | |

**Progress:** 12 / 14 tickets done.

---

## Design Decisions

### HTML-first, PDF optional

`weasyprint` requires system-level packages (`libpango`, `libcairo`, `libgdk-pixbuf`) that are absent on Streamlit Community Cloud and many CI environments. Therefore:

- **HTML** is the primary output format — always available, self-contained (inline CSS + base64 figures).
- **PDF** is an optional extra (`pip install -e ".[report-pdf]"`) that fails gracefully with a clear `ImportError` message if `weasyprint` is not installed.

### Jinja2 as new dependency

Jinja2 is small (~200 KB) and widely available. It goes in a new `[report]` extras group rather than core so that users who only use the Python API don't pull it in automatically.

### Self-contained HTML

The output `.html` file must open correctly offline and be email-attachable:
- All CSS is inlined in a `<style>` block inside the template.
- Figures are embedded as `data:image/png;base64,...` `<img>` tags.
- No external fonts, CDN links, or JavaScript.

### Figure generation

The power curve figure is generated using whichever plotting backend is available, in priority order: `plotly` → `matplotlib` → omitted (section shows a grey placeholder note). The figure is rasterised to PNG and base64-encoded before embedding.

### New file: `iopt_power_design/report.py`

Public API:

```python
def generate_report(
    result: dict,
    formula: str,
    factors: dict,
    power_cfg,                      # PowerContrastConfig | PowerR2Config
    output_path: str | Path,        # .html or .pdf; extension determines format
    title: str = "I-Optimal Design Report",
    include_power_curve: bool = True,
    design_rows_shown: int = 30,    # cap table at N rows to keep file size down
) -> Path:
    """Render and write the report. Returns the path written."""
```

### New template: `iopt_power_design/templates/report_template.html`

Jinja2 template. Sections (all conditional on data availability):
1. Header (title, generated timestamp, iopt-power-design version)
2. Config summary (formula, factors, power mode, key params)
3. Power metrics (n, achieved power, target power, λ, df, criterion, elapsed)
4. Design table (up to `design_rows_shown` rows)
5. Bucket allocations table
6. Diagnostics (condition number, D-efficiency, warnings list)
7. Power curve figure (embedded PNG)
8. Footer (reproducibility note: random_state, package version)

---

## Epic A — Infrastructure

Dependencies: none. Start here first; all other tickets depend on A1–A2.

---

### A1 Report module skeleton & packaging

**Status:** Done
**Claimed by:** Claude
**Est.:** 2–3 hours
**Progress note:** Complete. `iopt_power_design/report.py` created with `generate_report()` stub, `_get_jinja_env()`, and `_fig_to_base64()`. `iopt_power_design/templates/` directory created. `pyproject.toml` updated with `[report]` (jinja2, pillow) and `[report-pdf]` (+ weasyprint) extras and `[tool.setuptools.package-data]`. `__init__.py` exports `generate_report`.

**What to do:**
1. Create `iopt_power_design/templates/` directory.
2. Create `iopt_power_design/report.py` with:
   - `generate_report(...)` function stub that raises `NotImplementedError`.
   - `_get_jinja_env()` helper that loads the template from the `templates/` package directory using `importlib.resources` (Python ≥ 3.9) or `pkg_resources` as fallback.
   - `_fig_to_base64(fig) -> str` helper stub.
   - Module-level `__all__ = ["generate_report"]`.
3. Export `generate_report` from `iopt_power_design/__init__.py`.
4. Add to `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   report = ["jinja2>=3.0", "pillow>=9.0"]
   report-pdf = ["jinja2>=3.0", "pillow>=9.0", "weasyprint>=60.0"]
   ```
   Add both to the `all` meta-group.
5. Add `iopt_power_design/templates/` to `package_data` (or use `include_package_data = true` with a `MANIFEST.in` entry `recursive-include iopt_power_design/templates *.html`).
6. Verify: `pip install -e ".[report]"` succeeds and `from iopt_power_design import generate_report` works.

**Acceptance criteria:**
- [ ] `iopt_power_design/report.py` exists and is importable.
- [ ] `pip install -e ".[report]"` installs `jinja2` and `pillow`.
- [ ] `from iopt_power_design import generate_report` does not raise.
- [ ] `pyproject.toml` `[all]` includes report deps.

---

### A2 Jinja2 HTML template skeleton

**Status:** Done
**Claimed by:** Claude
**Est.:** 2–3 hours
**Depends on:** A1
**Progress note:** Complete. `iopt_power_design/templates/report_template.html` created with full inline CSS (metric-grid, tables, badges, alerts, print styles), and all section stubs guarded by `{% if ... %}`: header, config summary, power metrics grid, design table, buckets table, diagnostics (with VIF sub-table), power curve figure (with placeholder fallback), and footer. HTML structure verified clean.

**What to do:**
Create `iopt_power_design/templates/report_template.html` — a complete, valid Jinja2 template with:

1. **Document structure:** `<!DOCTYPE html>`, `<html lang="en">`, `<head>` (charset, viewport, title), `<body>`.
2. **Inline CSS `<style>` block** covering:
   - Clean sans-serif font stack, max-width 900px centered, print-friendly margins.
   - `.metric-grid`: CSS grid for the power metrics card (3 columns).
   - `.metric-box`: card style for each metric value.
   - `table`: full-width, alternating row colors, bordered.
   - `.section`: margin/padding between report sections.
   - `.badge-pass` / `.badge-warn` / `.badge-fail`: coloured inline badges for diagnostics.
   - `@media print`: remove box shadows, keep tables intact.
3. **Section stubs** (all wrapped in `{% if ... %}` guards so missing data skips the section):
   - `<!-- HEADER -->` — title, timestamp, version
   - `<!-- CONFIG SUMMARY -->` — placeholder
   - `<!-- POWER METRICS -->` — placeholder
   - `<!-- DESIGN TABLE -->` — placeholder
   - `<!-- BUCKETS TABLE -->` — placeholder
   - `<!-- DIAGNOSTICS -->` — placeholder
   - `<!-- POWER CURVE -->` — placeholder
   - `<!-- FOOTER -->` — placeholder
4. Pass a minimal context dict `{"title": "Test", "generated_at": "..."}` from `generate_report()` stub and confirm the template renders without error.

**Acceptance criteria:**
- [ ] Template file exists at `iopt_power_design/templates/report_template.html`.
- [ ] `jinja2.Environment(...).get_template("report_template.html").render({"title": "Test", "generated_at": "now"})` produces valid HTML with no Jinja2 errors.
- [ ] Rendered HTML passes the W3C Nu validator (or `html.parser` parse without errors) for the stub sections.

---

## Epic B — Report Content

Build out each section of the template and the corresponding Python context-building code. Work on B1–B5 in any order after A2 is done.

---

### B1 Config summary section

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** A2
**Progress note:** Complete. `_build_config_ctx(formula, factors, power_cfg)` in `report.py`. Detects categorical (list of strings) vs continuous (2-element numeric tuple/list). Handles both PowerContrastConfig (adds sigma, L_shape, delta) and PowerR2Config (adds r2_target, lambda_mode).

**What to do:**
In `report.py`, write `_build_config_ctx(formula, factors, power_cfg) -> dict` that returns a context dict for the config summary section:

```python
{
    "formula": formula,
    "factors": [
        {"name": "A", "type": "categorical", "levels": ["low", "high"]},
        {"name": "B", "type": "continuous", "low": 0.0, "high": 10.0},
    ],
    "power_mode": "contrast" | "r2",
    # Contrast mode:
    "alpha": 0.05, "power_target": 0.80, "sigma": 1.0, "max_n": 500,
    "L_shape": "1 × 4", "delta": "[0.5]",
    # R² mode:
    "r2_target": 0.15, "lambda_mode": "n",
}
```

In the template, render this as a two-column definition list (`<dl>`) inside the `<!-- CONFIG SUMMARY -->` section.

**Acceptance criteria:**
- [ ] Continuous and categorical factors both display correctly.
- [ ] Contrast mode shows L shape and delta; R² mode shows r2_target and lambda_mode.
- [ ] Missing / None fields are omitted, not shown as "None".

---

### B2 Power metrics section

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** A2
**Progress note:** Complete. `_build_metrics_ctx(report)` in `report.py`. Formats all 8 metric-box values; computes `power_class` (pass/warn/fail based on diff from target: ≥0/≥-5pp/else); formats elapsed_sec, noncentrality_lambda, df_num/df_denom, search_strategy, warnings list.

**What to do:**
In `report.py`, write `_build_metrics_ctx(report: dict) -> dict` that extracts and formats:

```python
{
    "n": 24,
    "achieved_power": "81.4 %",
    "target_power": "80.0 %",
    "power_ok": True,          # achieved >= target
    "noncentrality_lambda": "12.34",
    "df_num": 1,
    "df_denom": 20,
    "criterion": "I",
    "elapsed_sec": "1.4 s",
    "search_strategy": "bisection+verification",
    "random_state": 42,
    "warnings": [],            # list of warning strings
}
```

In the template, render the first six values as a `metric-grid` of `.metric-box` cards (matching the style from the CSS in A2). Show `achieved_power` with a green/amber/red badge depending on whether it meets the target. List any warnings below the grid in amber alert boxes.

**Acceptance criteria:**
- [ ] All seven metrics render as cards in a grid.
- [ ] Green badge when achieved ≥ target; amber when within 5 pp below; red otherwise.
- [ ] Warnings list renders as styled alert boxes (or is absent when empty).

---

### B3 Design & buckets tables section

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** A2
**Progress note:** Complete. `_df_to_html(df, max_rows)` in `report.py`. Returns `(html_str, was_truncated, total_rows)` tuple. Uses `df.to_html(classes="report-table", float_format=":.4g")`; truncation metadata passed to template which renders note.

**What to do:**
In `report.py`, write `_df_to_html(df: pd.DataFrame, max_rows: int) -> str` that converts a DataFrame to an HTML table string:
- Uses `df.head(max_rows).to_html(index=False, border=0, classes="report-table")`.
- If `len(df) > max_rows`, appends a `<p class="table-note">Showing {max_rows} of {len(df)} rows.</p>`.

In the template, render:
- **Design table** section with heading "Selected Runs (`n` = {{ n }})".
- **Bucket allocations** section with heading "Unique Run Allocations".

Pass `design_html` and `buckets_html` strings (pre-rendered) into the template context so the template just does `{{ design_html | safe }}`.

**Acceptance criteria:**
- [ ] Design table renders with correct column names and row count.
- [ ] Truncation note appears when rows > `design_rows_shown`.
- [ ] Buckets table always shows all rows (bucket count is always small).

---

### B4 Diagnostics section

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** A2
**Progress note:** Complete. `_build_diagnostics_ctx(report)` in `report.py`. Returns None when `report["diagnostics"]` is absent/empty (template skips section). Condition-number badge: pass <100, warn <1000, fail ≥1000. Formats d_efficiency, i_criterion, VIFs dict.

**What to do:**
In `report.py`, write `_build_diagnostics_ctx(report: dict) -> dict | None`. If `report.get("diagnostics")` is empty or absent, return `None` (section is skipped).

Otherwise return:

```python
{
    "condition_number": "12.5",
    "condition_badge": "pass",     # "pass" <100, "warn" <1000, "fail" >=1000
    "d_efficiency": "0.81",
    "vifs": {"A[T.high]": 1.0, "B": 1.2, ...},   # may be absent
    "i_criterion": "0.034",        # may be absent
}
```

In the template, render as a table with coloured `.badge-pass` / `.badge-warn` / `.badge-fail` badges next to condition number. VIFs (if present) render as a nested sub-table.

**Acceptance criteria:**
- [ ] Section is skipped (not rendered) when diagnostics are absent.
- [ ] Condition number badge colour is correct for all three ranges.
- [ ] VIF sub-table renders if present, omitted if absent.

---

### B5 Embedded power curve figure

**Status:** Done
**Claimed by:** Claude
**Est.:** 2–3 hours
**Depends on:** A2, B2
**Progress note:** Complete. `_build_power_curve_figure(result, formula, factors, power_cfg)` in `report.py`. Calls `power_curve_by_n` with starts=2, auto_candidate=True, random_state from report. Tries Plotly (with kaleido rasterisation) then matplotlib (Agg backend) for PNG → base64. Returns None on any failure (template shows placeholder note). Target-power hline and chosen-n vline included in both backends.

**What to do:**
In `report.py`, implement `_build_power_curve_figure(result, formula, factors, power_cfg, design_opts) -> str | None`:

1. Attempt to import `plotly` → if available, call `power_curve_by_n(...)` with a tight n range (from `p+1` to `min(result["n"] * 2, max_n)`, 30 points), build a Plotly figure with reference lines (target power, chosen n), write to PNG bytes using `fig.to_image(format="png", width=800, height=350)`, base64-encode.
2. If `plotly` is absent, attempt `matplotlib` similarly.
3. If neither is available, return `None` (section shows a grey note: "Install `iopt-power-design[viz]` or `[app]` to include the power curve.").

Return the base64 PNG string on success.

In the template, render as `<img src="data:image/png;base64,{{ power_curve_b64 }}" alt="Power curve" style="max-width:100%">`.

**Note:** `power_curve_by_n` builds new designs — use a low `starts` value (2–3) and `auto_candidate=True` to keep generation time under ~10 seconds.

**Acceptance criteria:**
- [ ] Figure embeds correctly in the HTML (viewable offline).
- [ ] Graceful degradation when neither plotly nor matplotlib is installed.
- [ ] Reference lines (target power, chosen n) are visible in the figure.

---

## Epic C — Export Backends

---

### C1 Standalone HTML output

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** A1, A2, B1–B5
**Progress note:** Complete. `_build_context()` private helper assembles the full Jinja2 context by calling all B helpers (B1–B5). `generate_report()` resolves the output path (directory → `iopt_report.html`; missing/unknown suffix → `.html`), creates parent dirs, renders the template, and writes UTF-8 HTML. All edge cases handled.

**What to do:**
Implement the HTML export path in `generate_report()`:

1. Call `_build_config_ctx`, `_build_metrics_ctx`, `_build_diagnostics_ctx`, `_df_to_html` (for design and buckets), and optionally `_build_power_curve_figure` to assemble the full template context.
2. Render the Jinja2 template with `env.get_template("report_template.html").render(ctx)`.
3. Write the rendered string to `output_path` with `encoding="utf-8"`.
4. Return the resolved `Path`.

Edge cases:
- Create parent directories if they don't exist (`output_path.parent.mkdir(parents=True, exist_ok=True)`).
- If `output_path` has no suffix or suffix is not `.html` / `.pdf`, default to `.html`.
- If `include_power_curve=False`, skip `_build_power_curve_figure` entirely.

**Acceptance criteria:**
- [ ] `generate_report(result, ..., output_path="./out/report.html")` writes a valid HTML file.
- [ ] File opens correctly in Chrome, Firefox, and Safari (or equivalent).
- [ ] All sections from B1–B5 are present when data is available.
- [ ] File is self-contained — works offline with no internet connection.

---

### C2 Optional PDF export

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** C1
**Progress note:** Complete. PDF branch in `generate_report()`: when `output_path.suffix == ".pdf"`, imports `weasyprint.HTML` and calls `.write_pdf()` on the already-rendered HTML string. Clear `ImportError` with `[report-pdf]` install hint when weasyprint is absent. Template already contains `@media print` styles (page margins, page-break-inside, figure max-height).

**What to do:**
Add PDF support as a conditional backend inside `generate_report()`:

```python
if output_path.suffix == ".pdf":
    try:
        from weasyprint import HTML as WeasyprintHTML
    except ImportError:
        raise ImportError(
            "PDF export requires weasyprint. "
            "Install it with: pip install \"iopt-power-design[report-pdf]\""
        ) from None
    html_str = env.get_template("report_template.html").render(ctx)
    WeasyprintHTML(string=html_str).write_pdf(str(output_path))
    return output_path
```

When `weasyprint` is present, also add `@media print` CSS adjustments in the template (already stubbed in A2):
- Page margins: `@page { margin: 15mm; }`
- Force page breaks before H2 sections.
- Hide the power curve if it makes the page overflow (set `max-height: 200px`).

**Acceptance criteria:**
- [ ] `generate_report(..., output_path="report.pdf")` writes a valid PDF when `weasyprint` is installed.
- [ ] Clear `ImportError` with install instructions when `weasyprint` is absent.
- [ ] PDF contains all the same sections as the HTML (verifiable by opening the PDF).

---

## Epic D — Integration

---

### D1 API integration

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** C1
**Progress note:** Complete. `export_report_to: Optional[str] = None` added to `i_optimal_powered_design()` signature in `api.py`. After the diagnostics export block, calls `generate_report()` with `include_power_curve=False` (keeps API call fast). Stores written path in `result["report"]["report_path"]`; stores error string in `report_path_error` on failure — design result always returned.

**What to do:**
Add `export_report_to=` parameter to `i_optimal_powered_design()` in `iopt_power_design/api.py`:

```python
def i_optimal_powered_design(
    formula: str,
    factors: dict,
    power_cfg,
    design_opts: DesignOptions | None = None,
    export_diagnostics_to: str | Path | None = None,
    export_report_to: str | Path | None = None,   # NEW
) -> dict:
```

Behaviour:
- If `export_report_to` is not `None`, call `generate_report(result, formula, factors, power_cfg, export_report_to)` after the design is built.
- Store the written path as a string in `result["report"]["report_path"]`.
- Wrap in a `try/except` with a warning (not a crash) if report generation fails — the design result is always returned regardless.

**Acceptance criteria:**
- [ ] `export_report_to="./out/"` writes `./out/report.html` and stores the path in `result["report"]["report_path"]`.
- [ ] A failure in report generation does not prevent the design result from being returned.
- [ ] `export_report_to=None` (default) leaves no new keys in the report dict.

---

### D2 CLI integration

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** D1
**Progress note:** Complete. `--html-report` argparse flag added to `cli.py`. Writes `<basename>_report.html` directly (does not go through `export_report_to=` on api — uses `generate_report()` directly so the path is deterministic). Also honoured via `output.html_report: true` in YAML. Report path printed in summary. `ImportError` and other failures emit `logger.warning()` and continue.

**What to do:**
In `iopt_power_design/cli.py`, add a `--html-report` flag:

```
iopt-design --config config.yml --out ./output/design --html-report
```

Behaviour:
- When `--html-report` is passed, set `export_report_to` to the same directory as `--out`.
- The report file is named `<basename>_report.html` (alongside `<basename>_design.csv` etc.).
- Add the report path to the verbose summary printed at the end.

Also support setting it in the YAML config:
```yaml
output:
  basename: my_design
  html_report: true
```

**Acceptance criteria:**
- [ ] `iopt-design --config config.yml --out ./out/design --html-report` writes `./out/design_report.html`.
- [ ] `output.html_report: true` in YAML config has the same effect.
- [ ] The report path is printed in the CLI summary output.

---

### D3 Streamlit download button

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** C1
**Progress note:** Complete. `_HAS_JINJA2` flag added via `importlib.util.find_spec`. `ss["_last_power_cfg"] = power_cfg` stored immediately after each successful run. Download button added in `exp_cols[3]` using `tempfile.NamedTemporaryFile(suffix=".html", delete=False)` — writes report to temp file, reads bytes back, offers `st.download_button("⬇ HTML report", ...)`. Falls back to `st.info(...)` when jinja2 is absent.

**What to do:**
In `app/pages/3_Run_Results.py`, add an HTML report download button in the export section (alongside the existing Design CSV, Excel, and JSON download buttons).

Implementation:
1. Check if `jinja2` is importable (`importlib.util.find_spec("jinja2")`). If not, show a grey info box: "Install `iopt-power-design[report]` to enable HTML report download."
2. If available, add a "Generate HTML report" button. On click:
   a. Call `generate_report(result, formula, factors, power_cfg, output_path=io.BytesIO() or tmp file)`.
   b. Read the bytes and pass to `st.download_button("Download HTML report", data=html_bytes, file_name="iopt_report.html", mime="text/html", use_container_width=True)`.
3. Since `generate_report` writes to a file path, use a `tempfile.NamedTemporaryFile` approach: write to a temp `.html` file, read it back as bytes, offer the download.

**Note:** Do not attempt PDF download in the Streamlit app — weasyprint's system dependencies make this unreliable in hosted environments.

**Acceptance criteria:**
- [ ] Download button appears in the export section after a successful run.
- [ ] Downloaded HTML file opens correctly offline in a browser.
- [ ] Graceful info box when `jinja2` is not installed.

---

## Epic E — Tests & Docs

---

### E1 Unit tests

**Status:** Open
**Claimed by:**
**Est.:** 2–3 hours
**Depends on:** C1, D1
**Progress note:**

**What to do:**
Create `tests/test_report.py` with the following test classes:

**`TestGenerateReportHTML`** (requires `jinja2` and `pillow`; mark with `pytest.importorskip`):
- `test_html_report_creates_file`: call `generate_report(...)` with a minimal result, assert the file exists and ends in `.html`.
- `test_html_is_parseable`: parse the output with `html.parser.HTMLParser` — no `HTMLParseError`.
- `test_html_contains_key_sections`: assert the substrings "Config Summary", "Power Metrics", "Selected Runs", "Unique Run Allocations" are all in the rendered HTML.
- `test_html_is_self_contained`: assert no `http://` or `https://` URLs appear in the output (all assets are inline).
- `test_html_report_contrast_mode`: render for `PowerContrastConfig` — assert `sigma` and `delta` appear.
- `test_html_report_r2_mode`: render for `PowerR2Config` — assert `r2_target` and `lambda_mode` appear.
- `test_truncation_note`: pass a result with 100 design rows and `design_rows_shown=10` — assert truncation note appears.

**`TestGenerateReportAPIIntegration`**:
- `test_export_report_to_path`: call `i_optimal_powered_design(..., export_report_to=tmp_path)` — assert `result["report"]["report_path"]` is set and the file exists.
- `test_export_report_failure_does_not_crash`: monkey-patch `generate_report` to raise `RuntimeError`; assert the design result is still returned.

**`TestPDFExportImportError`**:
- Mock `weasyprint` as absent; assert `generate_report(..., output_path="report.pdf")` raises `ImportError` with "report-pdf" in the message.

**Acceptance criteria:**
- [ ] All tests pass with `pip install -e ".[report,dev]"`.
- [ ] Tests are skipped (not failed) when `jinja2` / `pillow` are absent.
- [ ] No test writes files outside `tmp_path`.

---

### E2 Documentation updates

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** D1, D2
**Progress note:**

**What to do:**

**`README.md`:**
1. Add `pip install -e ".[report]"` and `pip install -e ".[report-pdf]"` to the Installation section with descriptions.
2. Add a new `## Shareable Reports` section (after the Diagnostics section) covering:
   - `generate_report()` Python API example.
   - `export_report_to=` in `i_optimal_powered_design`.
   - CLI `--html-report` flag.
   - Note on PDF support and its `weasyprint` requirement.

**`docs/quickstart.md`:**
Add a step "5b) Export a shareable HTML report" between the existing step 4 (Common next steps) and step 5 (Streamlit web UI), with a one-paragraph explanation and a short code snippet.

**`docs/recipes.md`:**
Add Recipe 7 "Generate a shareable HTML report for a team member":
```python
from iopt_power_design import generate_report

generate_report(
    result=result,
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    output_path="./reports/my_design_report.html",
)
```

**Acceptance criteria:**
- [ ] `README.md` Installation section lists `[report]` and `[report-pdf]`.
- [ ] `README.md` has a `## Shareable Reports` section with working code examples.
- [ ] `docs/recipes.md` has Recipe 7.
- [ ] All links and cross-references are correct.

---

## Suggested Build Order

For a single contributor:

```
A1 → A2
      ↓
B1 → B2 → B3 → B4 → B5   (any order after A2)
      ↓
C1 → C2
      ↓
D1 → D2
D1 → D3
      ↓
E1 → E2
```

A good **MVP cut** (HTML report, API + Streamlit, no PDF) is: **A1, A2, B1–B4, C1, D1, D3, E1 (partial), E2** — skipping B5 (power curve figure) and C2 (PDF) saves ~4–5 hours and still delivers a fully usable shareable report. Add B5 and C2 in a follow-up.
