# Enhancement Roadmap — iopt-power-design

Tracks planned, in-progress, and completed enhancements.
Update this file as work is completed or new ideas are added.

**Status key:** ✅ Done · 🔄 In progress · ⬜ Not started

---

## Completed

| # | Enhancement | Files changed | Notes |
|---|---|---|---|
| 1 | **`power_surface_2d`** — 2D power grid over any two of `{n, effect, sigma, alpha}` | `power_curves.py` | Replaced `NotImplementedError` stub; caches X per n |
| 2 | **`--template` CLI command** — print a fully-commented YAML scaffold to stdout | `cli.py` | `--template contrast` and `--template r2`; `--config` made optional |
| 3 | **Binary search for minimum `n`** — replaces O(max_n) linear scan with O(log max_n) bisection + linear verification window | `api.py` | Phase 1 bisection + Phase 2 downward scan handles non-monotone power |
| 4 | **`power_sensitivity`** — analytical sigma sweep on a fixed design | `api.py`, `__init__.py` | Contrast mode only; no new DOE builds |
| 5 | **D-optimal criterion** (`criterion="D"`) — maximises `det(X'X)` | `design.py`, `config.py`, `cli.py`, `README.md` | `_d_criterion_for_indices`, `_criterion_score` dispatcher; validated in `DesignOptions`; tests in `test_config.py`, `test_design.py`, `test_api.py` |
| 6 | **A-optimal criterion** (`criterion="A"`) — minimises `trace((X'X)⁻¹)` | `design.py`, `config.py`, `cli.py`, `README.md` | `_a_criterion_for_indices` + `_score_design` helper; criterion validation extended to `{"I","D","A"}`; tests in `test_config.py`, `test_design.py`, `test_api.py` |
| 7 | **`min_detectable_effect()`** — inverts the power curve analytically | `api.py`, `__init__.py`, `README.md` | Bisection over delta scale factor (contrast) or `r2_target` (R²); no new DOE calls; tests in `test_api.py` |
| 8 | **`power_sensitivity` for R² mode** — sweeps `r2_target` over a fixed design | `api.py`, `README.md` | Replaces the `ValueError` guard; adds `r2_range` / `r2_points` parameters; tests in `test_api.py` |
| 9 | **`augment_design()`** — greedy criterion-based augmentation | `design.py`, `__init__.py`, `README.md` | Fixes existing rows in X, greedily adds `m` new runs using `_score_design`; supports all three criteria; tests in `test_design.py` and `test_api.py` |
| 10 | **Richer run metadata in `report`** — adds `elapsed_sec`, `search_strategy`, `verify_window`, `random_state`, `warnings` | `api.py`, `cli.py`, `README.md`, `tests/test_api.py` | `time.perf_counter()` wall-clock; strategy string built incrementally (`"bisection"` ± `"+growth"` ± `"+verification"`); warnings captured alongside each `warnings.warn()` call; CLI summary updated; 17 new tests in `TestRunMetadata` |
| 11 | **`compare_criteria()` helper** — runs I/D/A in one call and returns a side-by-side `summary` DataFrame | `api.py`, `__init__.py`, `README.md`, `tests/test_api.py` | `dataclasses.replace` for safe per-criterion opts copy; `summary` cols: `criterion, n, achieved_power, elapsed_sec, condition_number, d_efficiency`; optional 2-panel bar chart; 22 new tests in `TestCompareCriteria` |
| 12 | **Declarative constraints in config** — `constraint_expr` string alternative to `constraint_func` callable | `config.py`, `cli.py`, `README.md`, `tests/test_config.py` | `_compile_constraint_expr()` pre-compiles with `compile()` + restricted `__builtins__={}`; safe globals: `abs, min, max, round, sqrt, log, log2, log10, exp, floor, ceil, pi`; `DesignOptions.constraint_expr` auto-overwrites `constraint_func`; `dataclasses.replace`-safe; YAML templates updated; 17 new tests in `TestCompileConstraintExpr` |
| 14 | **PDF / HTML shareable report** | `iopt_power_design/report.py`, `templates/report_template.html`, `api.py`, `cli.py`, `app/pages/3_Run_Results.py`, `pyproject.toml`, docs | Self-contained HTML (Jinja2 + inline CSS + base64 figures); optional PDF via weasyprint; `export_report_to=` API param; `--html-report` CLI flag; Streamlit download button with session-state caching; 12 unit tests |
| 15 | **Streamlit front-end** | `app/app.py`, `app/pages/`, `app/components/`, `app/state.py`, `pyproject.toml`, docs | Delivered multi-page app (factors, power config, run/results, analysis/export) with recent hardening fixes for syntax, YAML export validity, n-sweep scaling semantics, and MDE display |
| 13 | **Plotly interactive power charts** | `power_curves.py`, `api.py`, `plot_backends.py` (new), `__init__.py`, `pyproject.toml`, `tests/test_plot_backends.py` (new) | Opt-in `plot_backend="plotly"` on `power_curve_by_n`, `power_curve_by_effect`, `power_surface_2d`, and `power_sensitivity`. All four Plotly builders in `plot_backends.py` with soft plotly dependency (try/except guard). `power_surface_2d` exported from `__init__.py`. 19 new tests; `plot_backends.py` coverage 95%. `plotly>=5.0` added to `[viz]` extras. |

---

## Backlog

Items are loosely ordered by effort × value.
Move an item to **In Progress** or **Completed** when work starts/finishes.

### Low effort · High value

| # | Enhancement | Description | Est. LOE | Value | Key files |
|---|---|---|---|---|---|
### Medium effort · High value

| # | Enhancement | Description | Est. LOE | Value | Key files |
|---|---|---|---|---|---|
| 16 | **Google Sheets integration** | Bidirectional connector using `gspread` + OAuth2. Reads a structured template Sheet (factor table, power config cells) and writes design, buckets, and report back to a Results tab. Includes a ready-to-copy Sheet template and a `sheets_run()` helper. High value for teams that live in Sheets and share experiments collaboratively. | 4–6 days | High | New: `iopt_power_design/sheets.py`, `pyproject.toml` (new dep: `gspread`, `google-auth`) |
| 17 | **Excel workbook template** | Structured `.xlsx` input/output workbook with (1) a Config sheet for factor entry, formula, and power parameters with dropdown validation, and (2) auto-populated Results, Design, and Buckets sheets on run. Uses `openpyxl`; complements the existing `--excel` CLI flag. High value for pharma / corporate users who share Excel files. | 3–5 days | High | New: `iopt_power_design/excel_template.py`, `templates/iopt_template.xlsx` |
| 18 | **Jupyter ipywidgets UI** | Interactive in-notebook UI: dynamic factor-entry table, sliders for `alpha`/`power`/`sigma`, formula text field, run button, and inline Plotly power curve. No server needed — runs inside any JupyterLab / VS Code notebook. Ideal for data scientists who prototype in notebooks. | 3–4 days | Medium-High | New: `iopt_power_design/widgets.py`, `pyproject.toml` (new dep: `ipywidgets`) |
| 19 | **Robustness summary report** | Compact uncertainty report over ranges of `sigma`, `alpha`, and effect assumptions with worst/median/best power and threshold crossings. | 3–5 days | High | `api.py`, `power_curves.py`, `README.md`, `tests/test_api.py` |
| 26 | **Categorical pre-allocation (I-optimal allocation)** | Optional pre-allocation stage that explicitly optimises integer run counts across categorical cells before the point-exchange search. Uses a multiplicative Wynn/Kiefer-Wolfowitz algorithm to minimise the I-criterion over a weight simplex (w_i ≥ 0, Σw_i = 1), with support for per-cell lower/upper bounds and group-level proportion or count constraints. Weights are rounded to integers via constraint-preserving rounding. In mixed factor spaces, the existing search then runs within each categorical stratum for the allocated count. Exposed via `preallocate_categorical=False` in `DesignOptions` and a standalone `i_optimal_allocation()` helper. NumPy-only; no new solver dependency. Most valuable for designs with many categorical levels, fairness/policy constraints (minimum runs per level, regional quotas), or when current emergent allocation is unbalanced. Not needed when categorical structure is simple and current allocations look reasonable. | 4–6 days | Medium-High | New: `iopt_power_design/allocation.py`; `config.py`, `api.py`, `design.py` |
| 20 | **Blocked designs** (nuisance factors / block structure) | Add a `blocks` parameter; encode block membership as a factor in the Patsy formula; treat block effects as nuisance. D-optimal is the natural criterion. Requires modified power calculation (block-adjusted degrees of freedom). | 8–12 days | High | `design.py`, `api.py`, `config.py`, `README.md` |

### High effort · High value

| # | Enhancement | Description | Est. LOE | Value | Key files |
|---|---|---|---|---|---|
| 21 | **REST API (FastAPI)** | HTTP endpoints for `/design`, `/power_curve`, `/sensitivity`, `/compare_criteria`, and `/augment`. Enables no-Python integration (R, Excel VBA, web dashboards). Includes OpenAPI docs, async support, and a Docker compose file for one-command deployment. | 7–10 days | High | New: `api_server/main.py`, `api_server/routers/`, `Dockerfile`, `docker-compose.yml` |
| 22 | **Split-plot / hard-to-change factors** | Two-stratum variance model (`σ²_whole + σ²_subplot`), two-level design search, and modified power calculations. Substantial architectural change. | 15–25 days | Very High | `design.py`, `power.py`, `api.py`, `config.py` |
| 23 | **Multi-response designs** | Joint power across `k` responses; noncentrality becomes a matrix; requires Hotelling T² or Roy's largest-root distribution. | 10–15 days | High | `power.py`, `api.py`, `config.py` |
| 24 | **Bayesian / robust optimal design** | Support local/Bayesian D-optimality with priors over coefficients and robust objective averaging over parameter uncertainty. | 12–20 days | Medium-High | `design.py`, `config.py`, `api.py`, `README.md` |
| 25 | **GLM support (logistic/Poisson)** | Extend candidate scoring and power calculations beyond Gaussian linear models for common classification/count use cases. | 10–15 days | High | `design.py`, `power.py`, `api.py`, `config.py` |

---

## Technical Debt / Refactoring

Internal improvements with no user-visible behavior change. Worthwhile for maintainability and testability.

| # | Item | Description | Est. LOE | Key files |
|---|---|---|---|---|
| TD-1 | **Split `design.py`** | Now **1,267 lines** (was 770 when originally assessed). Mixes candidate sizing/building (lines 89–417), Patsy model-matrix encoding (418–439), exchange algorithm + criterion scoring (440–847), and multi-start orchestration + augmentation (848–1267). Split into: `candidate.py` (sizing + build), `model_matrix.py` (Patsy helpers), `iopt_search.py` (criterion functions + multi-start worker + exchange loop). `augment_design` stays in `design.py` or moves to `iopt_search.py`. Isolates concerns; makes swapping search algorithms easier; tightens unit tests per module. **Higher priority than originally estimated.** | 3–4 days | `design.py` → `candidate.py`, `model_matrix.py`, `iopt_search.py` |
| ~~TD-2~~ | ~~**Split `diagnostics.py`**~~ | **Done.** `diagnostics.py` (663 lines) split into `diag_metrics.py` (204 lines, pure NumPy), `diag_plots.py` (209 lines, matplotlib), `diag_export.py` (228 lines, file I/O). `diagnostics.py` is now a 20-line backward-compat re-export wrapper. `api.py` and `power_curves.py` updated to import from the new modules directly. `matplotlib` no longer imported at `api.py` load time. | Done | `diagnostics.py`, `diag_metrics.py`, `diag_plots.py`, `diag_export.py`, `api.py`, `power_curves.py` |
| TD-3 | **Slim down `api.py` and fix power-curve duplication** | Now **1,194 lines** (was 600 when originally assessed). `api.py` contains stub wrappers for `power_curve_by_n`/`power_curve_by_effect` that shadow the full implementations in `power_curves.py` (risk of silent drift). Additionally, `power_sensitivity` (200+ lines), `min_detectable_effect` (150+ lines), and `compare_criteria` (170+ lines) are analysis utilities that belong in `power_curves.py` or a new `analysis.py`, not in the orchestration entry point. Goal: `api.py` should contain only `i_optimal_powered_design` and its direct helpers; everything else re-exported via `__init__.py`. **Higher priority than originally estimated.** | 2–3 days | `api.py`, `power_curves.py`, new `analysis.py` (optional), `__init__.py` |

---

## Ideas / Future Consideration

Add rough ideas here before they are fleshed out enough to promote to the backlog.

- Simulation-based (Monte Carlo) power for non-normal residuals
- Bayesian D-optimal (incorporate prior beliefs on coefficients)
- Design catalog — pre-computed designs for common factor structures (e.g., 2³ full factorial, Box-Behnken, CCD)
- Export designs to JMP / Minitab / SAS format
- GPU acceleration for parallel starts (large grid problems)
- Design sensitivity to model misspecification (e.g., wrong functional form)
