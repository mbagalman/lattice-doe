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

---

## Backlog

Items are loosely ordered by effort × value.
Move an item to **In Progress** or **Completed** when work starts/finishes.

### Low effort · High value

*(All low-effort items complete.)*

### Medium effort · High value

| # | Enhancement | Description | Key files |
|---|---|---|---|
| 13 | **Interactive web UI** (Streamlit or Dash) | Wraps the Python API in a point-and-click interface. High value for collaborators and clients who are not Python users. Separate file or `app/` sub-package. | New: `app/app.py` (or similar) |
| 14 | **Blocked designs** (nuisance factors / block structure) | Add a `blocks` parameter; encode block membership as a factor in the Patsy formula; treat block effects as nuisance. D-optimal is the natural criterion. Requires modified power calculation (block-adjusted degrees of freedom). | `design.py`, `api.py`, `config.py`, `README.md` |
| 15 | **Robustness summary report** | Add a compact uncertainty report over ranges of `sigma`, `alpha`, and effect assumptions with worst/median/best power and threshold crossings. | `api.py`, `power_curves.py`, `README.md`, `tests/test_api.py` |

### High effort · High value

| # | Enhancement | Description | Key files |
|---|---|---|---|
| 16 | **Split-plot / hard-to-change factors** | Two-stratum variance model (`σ²_whole + σ²_subplot`), two-level design search, and modified power calculations. Substantial architectural change. | `design.py`, `power.py`, `api.py`, `config.py` |
| 17 | **Multi-response designs** | Joint power across `k` responses; noncentrality becomes a matrix; requires Hotelling T² or Roy's largest-root distribution. | `power.py`, `api.py`, `config.py` |
| 18 | **Bayesian / robust optimal design** | Support local/Bayesian D-optimality with priors over coefficients and robust objective averaging over parameter uncertainty. | `design.py`, `config.py`, `api.py`, `README.md` |
| 19 | **GLM support (logistic/Poisson)** | Extend candidate scoring and power calculations beyond Gaussian linear models for common classification/count use cases. | `design.py`, `power.py`, `api.py`, `config.py` |

---

## Ideas / Future Consideration

Add rough ideas here before they are fleshed out enough to promote to the backlog.

- Simulation-based (Monte Carlo) power for non-normal residuals
- Bayesian D-optimal (incorporate prior beliefs on coefficients)
- Design catalog — pre-computed designs for common factor structures (e.g., 2³ full factorial, Box-Behnken, CCD)
- Export designs to JMP / Minitab / SAS format
- GPU acceleration for parallel starts (large grid problems)
- Design sensitivity to model misspecification (e.g., wrong functional form)
