# Streamlit Front-End — Development Plan & Ticket Pack

Tracks all work for **Enhancement #15** (Streamlit front-end).

**Rules for contributors:**
1. Before starting a ticket, set its `Status` to `Claimed` and fill in `Claimed by`.
2. When done, check the box in the Dashboard and set `Status` to `Done`.
3. Never start work on a ticket marked `Claimed` by someone else — pick a different `Open` ticket or coordinate first.
4. If you hit a usage limit mid-ticket, leave a `Progress note` in the ticket card so the next session can continue without re-reading the whole codebase.

---

## Dashboard

| ID | Title | Epic | Status | Claimed by |
|----|-------|------|--------|------------|
| [A1](#a1-project-scaffold) | Project scaffold | Setup | Done | Claude |
| [A2](#a2-session-state-schema) | Session state schema | Setup | Done | Claude |
| [A3](#a3-navigation--sidebar) | Navigation & sidebar | Setup | Done | Claude |
| [B1](#b1-continuous-factor-rows) | Continuous factor rows | Factors | Done | Claude |
| [B2](#b2-categorical-factor-rows) | Categorical factor rows | Factors | Done | Claude |
| [B3](#b3-formula-input--patsy-validation) | Formula input & Patsy validation | Factors | Done | Claude |
| [B4](#b4-factor--formula-persistence) | Factor & formula persistence | Factors | Done | Claude |
| [C1](#c1-power-mode-toggle) | Power mode toggle | Power Config | Done | Claude |
| [C2](#c2-contrast-mode--l-matrix--delta) | Contrast mode — L matrix & delta | Power Config | Done | Claude |
| [C3](#c3-contrast-mode--scenario-builder) | Contrast mode — scenario builder | Power Config | Done | Claude |
| [C4](#c4-r-mode-ui) | R² mode UI | Power Config | Done | Claude |
| [C5](#c5-shared-power-parameters) | Shared power parameters | Power Config | Done | Claude |
| [D1](#d1-criterion-selector) | Criterion selector | Design Options | Done | Claude |
| [D2](#d2-search-options) | Search options | Design Options | Done | Claude |
| [D3](#d3-constraint-expression-input) | Constraint expression input | Design Options | Done | Claude |
| [E1](#e1-run-button--error-handling) | Run button & error handling | Results | Done | Claude |
| [E2](#e2-report-metrics-card) | Report metrics card | Results | Done | Claude |
| [E3](#e3-design-table--csv-download) | Design table & CSV download | Results | Done | Claude |
| [E4](#e4-buckets-table--csv-download) | Buckets table & CSV download | Results | Done | Claude |
| [E5](#e5-power-curve-chart) | Power curve chart | Results | Done | Claude |
| [E6](#e6-excel-download) | Excel download | Results | Done | Claude |
| [F1](#f1-sensitivity-analysis) | Sensitivity analysis | Analysis | Done | Claude |
| [F2](#f2-minimum-detectable-effect) | Minimum detectable effect | Analysis | Done | Claude |
| [F3](#f3-compare-criteria) | Compare criteria | Analysis | Done | Claude |
| [G1](#g1-yaml-config-export) | YAML config export | Export | Done | Claude |
| [G2](#g2-json-report-download) | JSON report download | Export | Done | Claude |
| [H1](#h1-dependencies--packaging) | Dependencies & packaging | Deploy | Open | |
| [H2](#h2-streamlit-config) | Streamlit config | Deploy | Open | |
| [H3](#h3-deployment-guide) | Deployment guide | Deploy | Open | |

**Progress:** 26 / 29 tickets done.

---

## Proposed App Structure

```
app/
  app.py                  # entry point; sets page config & renders nav
  pages/
    1_Factors.py          # factor & formula builder
    2_Power_Config.py     # power mode, contrast/R² params, design options
    3_Run_Results.py      # run button, report, design table, power curve
    4_Analysis.py         # sensitivity, MDE, compare criteria
  components/
    factor_table.py       # reusable factor entry widget
    power_params.py       # reusable alpha/power/sigma/max_n widget
    charts.py             # Plotly chart helpers
  state.py                # session state schema & reset helper
.streamlit/
  config.toml             # theme & server settings
```

**Key dependencies to add:**
- `streamlit>=1.30`
- `plotly>=5.0` (charts)
- `pyyaml>=6.0` (YAML config export — already in `[cli]` extras)
- `xlsxwriter>=3.0` (Excel download — already in `[extras]`)

---

## Epic A — Setup & Infrastructure

Dependencies: none. Start here first; all other tickets depend on A1–A3.

---

### A1 Project scaffold

**Status:** Done
**Claimed by:** Claude
**Est.:** 2–3 hours
**Progress note:** Complete.

**What to do:**
1. Create the directory tree above (`app/`, `app/pages/`, `app/components/`, `.streamlit/`).
2. Add stub `app.py` that runs `st.set_page_config(layout="wide")` and a "welcome" message.
3. Add empty `__init__.py` files where needed.
4. Update `pyproject.toml`: add a new `[project.optional-dependencies]` group `app` with `streamlit>=1.30` and `plotly>=5.0`; add `app` to the `all` meta-group.
5. Add `app` to the `[project.scripts]` entry: `iopt-app = "app.app:main"` (or document `streamlit run app/app.py`).
6. Verify: `pip install -e ".[app]"` succeeds and `streamlit run app/app.py` opens a blank page.

**Acceptance criteria:**
- [x] `app/` directory and all stub files exist and are committed.
- [x] `pip install -e ".[app]"` installs without errors.
- [x] `streamlit run app/app.py` opens without a crash.

---

### A2 Session state schema

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** A1
**Progress note:** Complete. `app/state.py` contains `init_state()` (idempotent via `if key not in`) and `render_sidebar()`.

**What to do:**
Create `app/state.py` with a single function `init_state()` that populates `st.session_state` with default values for every key used across all pages. Call `init_state()` at the top of every page script.

Keys to define (with defaults):

```python
# Factors
"factors": []        # list of dicts: {name, type, low, high} or {name, type, levels}
"formula": "~ 1 + A + B"

# Power config
"power_mode": "contrast"   # "contrast" or "r2"
"contrast_input_mode": "matrix"  # "matrix" or "scenario"
"L_text": ""               # raw text for L matrix
"delta_text": ""           # raw text for delta
"scenario_a": {}
"scenario_b": {}
"sesoi": 1.0
"r2_target": 0.15
"lambda_mode": "n"
"alpha": 0.05
"power_target": 0.80
"sigma": 1.0
"max_n": 500

# Design options
"criterion": "I"
"starts": 8
"auto_candidate": True
"random_state": 42
"constraint_expr": ""

# Results (populated after run)
"result": None        # the full dict from i_optimal_powered_design
"run_error": None     # string error message if run failed
```

**Acceptance criteria:**
- [x] `state.py` exists with `init_state()`.
- [x] Calling `init_state()` twice is idempotent (uses `setdefault` or `if key not in`).

---

### A3 Navigation & sidebar

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** A1, A2
**Progress note:** Complete. Sidebar rendered via `render_sidebar()` in `state.py`; called from every page stub. Reset button clears all state keys and calls `st.rerun()`.

**What to do:**
1. Configure `app.py` as the Streamlit multipage entry point (Streamlit auto-discovers `pages/` directory).
2. Add a sidebar section showing a live summary of current config:
   - Number of factors defined
   - Formula
   - Power mode and key params (alpha, power, sigma)
   - Whether a result exists ("Ready to run" / "Result available")
3. Add a sidebar "Reset all" button that clears `st.session_state` and reruns.

**Acceptance criteria:**
- [x] Sidebar summary renders correctly.
- [x] "Reset all" clears state and returns to defaults.
- [x] All 4 pages are reachable from the nav.

---

## Epic B — Factor & Formula Builder (Page 1)

---

### B1 Continuous factor rows

**Status:** Done
**Claimed by:** Claude
**Est.:** 2–3 hours
**Depends on:** A1, A2
**Progress note:** Complete. UUID-keyed rows in `factor_table.py`; name text_input, type selectbox, Low/High number_inputs, delete button, "+ Continuous" button.

**What to do:**
In `app/components/factor_table.py`, build a `render_factor_table()` function that renders the current `st.session_state["factors"]` list as an editable table of rows. Each row shows:
- Factor name (text input)
- Type selector (`Continuous` / `Categorical`)
- For continuous: Low value (number input) + High value (number input)
- Delete button (removes the row from state)

Below the table, an "Add continuous factor" button appends a new default row.

In `app/pages/1_Factors.py`, call `render_factor_table()`.

**Acceptance criteria:**
- [x] Can add multiple continuous factors.
- [x] Deleting a row removes it immediately.
- [x] Values persist when navigating away and back.

---

### B2 Categorical factor rows

**Status:** Done
**Claimed by:** Claude
**Est.:** 2 hours
**Depends on:** B1
**Progress note:** Complete. Levels text_input (comma-separated) in same `render_factor_table()`; `_sync_factors()` strips whitespace and drops empties; "+ Categorical" button added.

**What to do:**
Extend `render_factor_table()` to handle categorical rows. When type = `Categorical`:
- Replace Low/High with a single "Levels" text input (comma-separated, e.g. `low, med, high`).
- Parse and store as a list of strings.

Add an "Add categorical factor" button alongside the continuous one.

**Acceptance criteria:**
- [x] Can mix continuous and categorical factors in the same table.
- [x] Levels are parsed correctly (stripped whitespace, no empty strings).

---

### B3 Formula input & Patsy validation

**Status:** Done
**Claimed by:** Claude
**Est.:** 2–3 hours
**Depends on:** B1, B2
**Progress note:** Complete. `1_Factors.py` builds a 1-row candidate from current factors, calls `build_model_matrix()`, shows p + column names on success or `st.error()` on failure.

**What to do:**
In `app/pages/1_Factors.py`, below the factor table:
1. Text input for `formula` (pre-filled from session state).
2. On change, attempt to build a small candidate (using `iopt_power_design.design.build_model_matrix` or a dummy candidate) and display:
   - Number of model parameters `p`
   - Column names from the Patsy model matrix
   - A green "Valid formula" or red error message.
3. Store formula in `st.session_state["formula"]`.

**Acceptance criteria:**
- [x] Valid formula shows `p` and column names.
- [x] Invalid formula shows a clear red error (does not crash the page).
- [x] Formula updates are reflected immediately.

---

### B4 Factor & formula persistence

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** B1, B2, B3
**Progress note:** Complete. Session state via UUID widget keys persists across navigation automatically. "Clear all factors" button added; resets only factors/formula, leaves power config and results intact. Manual checklist in `1_Factors.py` docstring.

**What to do:**
Verify that all factor and formula state survives page navigation. Write a manual test checklist in a comment at the top of `1_Factors.py`.

Also add a "Clear factors" button that resets only the factors/formula keys in session state.

**Acceptance criteria:**
- [x] Navigating to Page 2 and back preserves all factor entries.
- [x] "Clear factors" resets only factors, not power config or results.

---

## Epic C — Power Configuration (Page 2)

---

### C1 Power mode toggle

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** A2
**Progress note:** Complete. Radio synced to `power_mode` ("contrast"/"r2"); manual index sync handles persistence across navigation.

**What to do:**
At the top of `app/pages/2_Power_Config.py`, render a `st.radio` toggle:
```
Power mode:  ( ) Contrast-based   ( ) Global R²
```
Store selection in `st.session_state["power_mode"]`. Conditionally render C2/C3 or C4 below.

**Acceptance criteria:**
- [x] Toggle switches between contrast and R² sections.
- [x] Mode persists across navigation.

---

### C2 Contrast mode — L matrix & delta

**Status:** Done
**Claimed by:** Claude
**Est.:** 3–4 hours
**Depends on:** C1, B3
**Progress note:** Complete. Two text areas (L_text, delta_text) with `_parse_matrix`/`_parse_vector`; live shape validation against p; "What is a contrast matrix?" expander.

**What to do:**
When `contrast_input_mode == "matrix"`:
1. Text area for `L` matrix. Hint: one row per line, values space- or comma-separated. Example shown below the widget.
2. Text area for `delta` (one value per row of L).
3. On input, validate shape: number of columns must equal `p` from the formula; number of rows in L must equal len(delta). Show green/red validation badge.
4. Store raw text in `st.session_state["L_text"]` and `st.session_state["delta_text"]`.
5. Parse to numpy arrays on run (in E1), not here.

**Acceptance criteria:**
- [x] Shape validation fires immediately on change.
- [x] Mismatch between L columns and p shows a clear error.
- [x] Valid configuration shows green badge.

---

### C3 Contrast mode — scenario builder

**Status:** Done
**Claimed by:** Claude
**Est.:** 2–3 hours
**Depends on:** C1, B1, B2
**Progress note:** Complete. Per-factor A/B inputs (number_input for continuous, selectbox for categorical); stale-level guard; SESOI input; "Preview L and δ" expander calls `contrast_from_scenarios`.

**What to do:**
When `contrast_input_mode == "scenario"`:
1. For each factor defined in session state, render two columns of inputs: Scenario A value and Scenario B value.
2. A "SESOI" number input (smallest effect of interest).
3. A "Preview L and delta" expander that calls `contrast_from_scenarios` and displays the resulting L and delta arrays.

Add a toggle above the contrast section: `Input method: (•) Matrix  ( ) Scenario builder`

**Acceptance criteria:**
- [x] Scenario inputs update dynamically as factors change.
- [x] Preview shows correct L/delta or a clear error.

---

### C4 R² mode UI

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** C1
**Progress note:** Complete. r2_target slider (key="r2_target"); lambda_mode radio with manual index sync to "n"/"n_minus_p".

**What to do:**
When `power_mode == "r2"`:
1. Number slider for `r2_target` (0.01 – 0.99, step 0.01).
2. Radio for `lambda_mode`: `"n"` (default, matches G*Power) or `"n_minus_p"` (conservative).
3. Store in session state.

**Acceptance criteria:**
- [x] r2_target slider renders with current value.
- [x] lambda_mode radio persists.

---

### C5 Shared power parameters

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** A2
**Progress note:** Complete. `render_power_params()` in `power_params.py`; reads `power_mode` from session state to hide/show sigma; all four inputs use direct session-state keys.

**What to do:**
In `app/components/power_params.py`, create `render_power_params()` that renders:
- `alpha` — number input (0.001 – 0.20, default 0.05)
- `power` — number input (0.50 – 0.99, default 0.80)
- `sigma` — number input (positive float, default 1.0; hidden/greyed out in R² mode)
- `max_n` — integer input (10 – 5000, default 500)

Call this at the bottom of Page 2.

**Acceptance criteria:**
- [x] All four inputs render and persist.
- [x] `sigma` is hidden (or marked N/A) in R² mode.

---

## Epic D — Design Options

Render as a collapsible `st.expander("Advanced design options")` on Page 2, below power params.

---

### D1 Criterion selector

**Status:** Done
**Claimed by:** Claude
**Est.:** 30 min
**Depends on:** A2
**Progress note:** Complete. selectbox with key="criterion" and detailed help tooltip for I/D/A inside the Advanced expander.

**What to do:**
`st.selectbox("Optimality criterion", ["I", "D", "A"])` with tooltip explaining each option. Store in `st.session_state["criterion"]`.

**Acceptance criteria:**
- [x] Selectbox renders with tooltip.
- [x] Selection persists.

---

### D2 Search options

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** A2
**Progress note:** Complete. starts slider, random_state input, auto_candidate checkbox, candidate_points (conditionally shown). All in Advanced expander.

**What to do:**
Inside the advanced expander:
- `starts` — integer slider (1–50, default 8)
- `random_state` — integer input (default 42)
- `auto_candidate` — checkbox (default True)
- `candidate_points` — integer input, shown only if `auto_candidate` is False (default 2000)

**Acceptance criteria:**
- [x] `candidate_points` input is hidden when `auto_candidate` is checked.
- [x] All values persist.

---

### D3 Constraint expression input

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** A2, B1, B2
**Progress note:** Complete. text_input with key="constraint_expr"; live `compile()` syntax check; green/red feedback.

**What to do:**
Text input for `constraint_expr`. Show a help tooltip listing the allowed functions (`abs`, `min`, `max`, `sqrt`, `log`, etc.). On input, attempt a syntax check (use `compile()` — same as the library does) and show green/red status. Store in session state.

**Acceptance criteria:**
- [x] Syntax error shows red message immediately.
- [x] Valid expression shows green status.

---

## Epic E — Run & Results (Page 3)

---

### E1 Run button & error handling

**Status:** Open
**Claimed by:**
**Est.:** 3–4 hours
**Depends on:** A2, B3, C1–C5, D1–D3
**Progress note:**

**What to do:**
At the top of `app/pages/3_Run_Results.py`:
1. Show a summary of the current config (formula, factors, mode, key params).
2. A prominent "Generate design" button.
3. On click:
   a. Parse `L_text`/`delta_text` or call `contrast_from_scenarios` to build `PowerContrastConfig`; or build `PowerR2Config`.
   b. Build `DesignOptions` from session state.
   c. Call `i_optimal_powered_design(...)` inside `st.spinner("Searching for optimal design...")`.
   d. Store result in `st.session_state["result"]` on success; store error string in `st.session_state["run_error"]` on failure.
4. Display any warnings from `result["report"]["warnings"]` as `st.warning(...)`.
5. Display `run_error` as `st.error(...)` with guidance.

**Acceptance criteria:**
- [ ] Spinner shows during run.
- [ ] Success stores result and renders subsequent E2–E6 sections.
- [ ] `ValueError` / `RuntimeWarning` shows a user-friendly error message, not a traceback.

---

### E2 Report metrics card

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** E1
**Progress note:**

**What to do:**
Using `st.metric` columns, display:
- Sample size `n`
- Achieved power (formatted as %)
- Target power (formatted as %)
- Elapsed time (formatted as e.g. "1.4 s")
- Criterion used
- Search strategy

Below, an expander "Full report" that renders the entire `report` dict as `st.json(...)`.

**Acceptance criteria:**
- [ ] All six metrics render in a single row of columns.
- [ ] Full report expander shows complete JSON.

---

### E3 Design table & CSV download

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** E1
**Progress note:**

**What to do:**
1. `st.dataframe(result["design_df"], use_container_width=True)`
2. `st.download_button("Download design CSV", data=result["design_df"].to_csv(index=False), file_name="design.csv", mime="text/csv")`

**Acceptance criteria:**
- [ ] Table renders with correct columns.
- [ ] Download button produces a valid CSV.

---

### E4 Buckets table & CSV download

**Status:** Open
**Claimed by:**
**Est.:** 30 min
**Depends on:** E1
**Progress note:**

**What to do:**
Same pattern as E3 but for `result["buckets_df"]`. Label the section "Unique run allocations".

**Acceptance criteria:**
- [ ] Buckets table renders with `count` column.
- [ ] Download button produces a valid CSV.

---

### E5 Power curve chart

**Status:** Open
**Claimed by:**
**Est.:** 2–3 hours
**Depends on:** E1
**Progress note:**

**What to do:**
Below the tables, an expander "Power curve (n sweep)":
1. A slider to select `n_max` for the sweep (default: 1.5 × result n, capped at max_n).
2. A "Plot curve" button that calls `power_curve_by_n(...)` with a spinner.
3. Render result as a Plotly line chart (`plotly.express.line`) with:
   - x = n, y = power
   - Horizontal dashed line at target power
   - Vertical dashed line at the chosen n
4. `st.plotly_chart(fig, use_container_width=True)`.

**Acceptance criteria:**
- [ ] Chart renders after clicking "Plot curve".
- [ ] Reference lines for target power and chosen n are visible.
- [ ] Chart is interactive (hover shows n and power values).

---

### E6 Excel download

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** E1
**Progress note:**

**What to do:**
If `xlsxwriter` is installed (use `importlib.util.find_spec`):
1. Build an `.xlsx` workbook in memory using `pandas.ExcelWriter` with the `xlsxwriter` engine, writing design, buckets, and report sheets.
2. Offer a `st.download_button("Download Excel workbook", ...)`.

If not installed, show a grey info box: "Install `iopt-power-design[extras]` to enable Excel export."

**Acceptance criteria:**
- [ ] Excel download produces a valid `.xlsx` with three sheets.
- [ ] Graceful degradation when `xlsxwriter` is absent.

---

## Epic F — Advanced Analysis (Page 4)

---

### F1 Sensitivity analysis

**Status:** Done
**Claimed by:** Claude
**Est.:** 2–3 hours
**Depends on:** E1
**Progress note:** Complete. `power_sensitivity()` sweep in `4_Analysis.py`; Plotly line chart with nominal-power and target-power reference lines; contrast mode sweeps σ, R² mode sweeps r2_target.

**What to do:**
In `app/pages/4_Analysis.py`, if no result exists show "Run a design first (Page 3)".

Otherwise, section "Sensitivity analysis":
1. For contrast mode: sigma sweep inputs (`sigma_min`, `sigma_max`, `sigma_points`), "Run sensitivity" button, Plotly line chart of power vs sigma.
2. For R² mode: r2 sweep inputs (`r2_min`, `r2_max`, `r2_points`), same pattern.
3. Show nominal power as a horizontal reference line.

**Acceptance criteria:**
- [x] Sensitivity chart renders for both modes.
- [x] Nominal power reference line is visible.

---

### F2 Minimum detectable effect

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** E1
**Progress note:** Complete. `min_detectable_effect()` called on fixed design; contrast mode shows scale factor and absolute δ; R² mode shows min R² and Cohen's f²; achieved power shown as st.metric.

**What to do:**
Section "Minimum detectable effect (MDE)":
1. Target power input (default: same as configured power).
2. "Compute MDE" button that calls `min_detectable_effect(...)`.
3. Display MDE value and achieved power as `st.metric`.

For contrast mode display as "Scale factor on delta" and the absolute delta value.
For R² mode display as "Minimum detectable R²".

**Acceptance criteria:**
- [x] MDE renders for both modes.
- [x] Achieved power at MDE is shown.

---

### F3 Compare criteria

**Status:** Done
**Claimed by:** Claude
**Est.:** 2 hours
**Depends on:** E1
**Progress note:** Complete. `compare_criteria()` called with selected criteria; summary dataframe with formatting; Plotly grouped bar chart (power bars + n diamond markers on secondary axis); per-criterion CSV download buttons.

**What to do:**
Section "Compare optimality criteria":
1. Multi-select for criteria to compare (default: all three).
2. "Run comparison" button with spinner (can take a while — warn the user).
3. Display `comparison["summary"]` as `st.dataframe`.
4. Render Plotly grouped bar chart of `achieved_power` and `n` by criterion.

**Acceptance criteria:**
- [x] Summary table renders with all columns.
- [x] Bar chart shows both metrics side by side.

---

## Epic G — Export & Reproducibility

---

### G1 YAML config export

**Status:** Done
**Claimed by:** Claude
**Est.:** 2 hours
**Depends on:** A2
**Progress note:** Complete. `_build_yaml()` in `4_Analysis.py` serializes session state to CLI-compatible YAML (factors, formula, alpha, power, contrast/R² block, design options); preview expander + download button at top of page 4.

**What to do:**
In the sidebar (or on Page 3 after a run), a "Download YAML config" button.

Build a YAML string from the current session state that is a valid input for the CLI (`iopt-design --config`). Use `pyyaml` to serialize. Include all power config, factor, and design option fields.

This allows users to reproduce any UI-generated run from the CLI or Python API.

**Acceptance criteria:**
- [x] Downloaded YAML passes `iopt-design --config <file> --dry-run` without errors.
- [x] YAML is available before a run (reflects current config, not results).

---

### G2 JSON report download

**Status:** Done
**Claimed by:** Claude
**Est.:** 30 min
**Depends on:** E1
**Progress note:** Complete. "Report JSON" download button added to page 3 export section (alongside Design CSV and Excel) and also on page 4. Uses `_jsonify()` for numpy serialization.

**What to do:**
On Page 3 after a successful run, a "Download report JSON" button that serializes `result["report"]` to JSON (handle numpy types with a custom encoder) and offers as a download.

**Acceptance criteria:**
- [x] Downloaded JSON is valid and contains all report keys.
- [x] numpy `float64` / `int64` values are correctly serialized (not `NaN` or type errors).

---

## Epic H — Deployment

---

### H1 Dependencies & packaging

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** A1
**Progress note:**

**What to do:**
1. Add `[project.optional-dependencies] app` group in `pyproject.toml`:
   ```toml
   app = ["streamlit>=1.30", "plotly>=5.0", "pyyaml>=6.0"]
   ```
2. Add `app` packages to the `all` meta-group.
3. Add `app/` to `.gitignore` exceptions (it should be tracked, not ignored).
4. Confirm `pip install -e ".[app]"` installs cleanly in a fresh venv.

**Acceptance criteria:**
- [ ] `pyproject.toml` updated with `app` group.
- [ ] Clean install succeeds.

---

### H2 Streamlit config

**Status:** Open
**Claimed by:**
**Est.:** 30 min
**Depends on:** A1
**Progress note:**

**What to do:**
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
```

**Acceptance criteria:**
- [ ] `streamlit run app/app.py` uses the configured theme.

---

### H3 Deployment guide

**Status:** Open
**Claimed by:**
**Est.:** 1–2 hours
**Depends on:** H1, H2
**Progress note:**

**What to do:**
Add a `## Streamlit Deployment` section to `docs/quickstart.md` covering:
1. Local run: `pip install -e ".[app]"` + `streamlit run app/app.py`
2. Streamlit Community Cloud: connect GitHub repo, set main file to `app/app.py`, no secrets needed.
3. Docker: a minimal `Dockerfile` that runs the Streamlit app (add to repo root).

**Acceptance criteria:**
- [ ] Local run instructions work on a fresh clone.
- [ ] `Dockerfile` builds and runs the app at `localhost:8501`.

---

## Suggested Build Order

For a single contributor working top to bottom:

```
A1 → A2 → A3
         ↓
B1 → B2 → B3 → B4
         ↓
C1 → C5 → D1 → D2 → D3
C2 → C3 → C4
         ↓
E1 → E2 → E3 → E4 → E5 → E6
         ↓
F1 → F2 → F3
G1 → G2
H1 → H2 → H3
```

A good **MVP cut** (working end-to-end demo with no advanced analysis) is: **A1–A3, B1–B4, C1, C2, C4, C5, D1, D2, E1–E5, H1–H2** (18 tickets, ~4–5 days).
