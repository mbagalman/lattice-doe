# Lattice DOE

## Stop Guessing Your Way Through Experimental Design

Most experimental design advice assumes a world that doesn't exist.

Clean factors. Unlimited sample sizes. Neatly separable effects.

In the real world, you get:
- Too many variables
- Not enough runs
- Constraints nobody told you about until it was too late

So what happens? People either oversimplify the problem until it's wrong, or overcomplicate it until it's unusable. Sometimes they do both in the same meeting.

Lattice DOE is for that situation.

It helps you design experiments that are:
- statistically powered
- run-efficient
- reproducible
- realistic about constraints

In plain English: instead of doing a power analysis in one place, picking a design in another, and hoping they vaguely agree, Lattice DOE solves those decisions together.

---

## What This Is

`lattice-doe` is a Python toolkit for designing **powered, efficient, structured experiments under real-world constraints**.

It searches for the **smallest experiment that can still hit your target power**, then optimizes the run locations under your chosen criterion (`I`, `D`, or `A`).

It's built for the messy middle:
- When full factorial designs are impossible
- When fractional factorial feels like guesswork
- When "just randomize it" isn't good enough

This is about getting **maximum information from limited experiments** without pretending your situation is cleaner than it is.

It supports:
- linear contrast power
- global R² power
- GLM contrast power for binomial and Poisson responses
- multi-response design
- blocked and split-plot structures
- Python, CLI, app, API, and spreadsheet-driven workflows

---

## Who This Is For

- Data scientists running experiments with multiple interacting variables
- Analysts asked to "design an experiment" without a textbook setup
- Teams with **tight budgets on experimental runs**
- Anyone who has ever thought: *"There has to be a better way to structure this"*
- Anyone tired of hearing "just use a factorial" from someone who is not paying for the runs

---

## A Concrete Example

Imagine a real study with 8 continuous variables and a budget for 32 experimental runs. A full factorial design would require far more runs, and random sampling leaves coverage gaps and hidden correlations.

The same logic applies at any scale. Here is a minimal 2-factor example you can run immediately — the API is identical whether you have 2 factors or 20:

```python
from lattice_doe import find_optimal_design, PowerContrastConfig, DesignOptions
from lattice_doe.contrasts import contrast_from_scenarios

formula = "~ 1 + A + B + A:B"
factors = {
    "A": (-1.0, 1.0),
    "B": (-1.0, 1.0),
}

L, delta = contrast_from_scenarios(
    formula=formula,
    factors=factors,
    scenario_a={"A": -1.0, "B": 0.0},
    scenario_b={"A": 1.0, "B": 0.0},
    sesoi=2.0,  # smallest response-scale effect worth detecting (in sigma units)
)

result = find_optimal_design(
    formula=formula,
    factors=factors,
    power_cfg=PowerContrastConfig(L=L, delta=delta, power=0.80, sigma=1.0, max_n=50),
    design_opts=DesignOptions(criterion="I", auto_candidate=True),
)

print(result["design_df"])          # the optimal run matrix
print(result["report"]["n"])        # minimum n that achieves 80% power
```

Now you have a design you can execute and defend.

---

## Why This Exists

Experimental design isn't just a statistics problem. It's a **decision problem**.

The goal isn't elegance. The goal is making better decisions with limited information.

Most tools optimise for theoretical purity. This one optimises for practical constraints, interpretability, and real-world usability.

---

## What Comes Next

Below you'll find full documentation, examples, and implementation details. If you're here to go deep, keep reading. If you're here because your last experiment was a mess, start with the [Quick Start Guide](docs/quickstart.md).

---

**Power-assured optimal experimental designs for linear and GLM models.**

The package automatically searches for the minimum sample size `n` that achieves your target power, then selects the best design at that `n` under your chosen criterion (`"I"` by default, or `"D"` / `"A"`). If the search hits practical limits first, it returns the best design found and reports that clearly.

**Supported power modes:**

| Mode | Config class | Test | Use case |
|---|---|---|---|
| Linear contrast | `PowerContrastConfig` | F-test on Lβ = δ | Detecting a specific effect in a linear model |
| Global R² | `PowerR2Config` | Omnibus F-test | Testing whether the full model explains meaningful variance |
| GLM Wald χ² | `PowerGLMContrastConfig` | Wald chi-square | Binomial (logistic) or Poisson response variables |
| Multi-response | `MultiResponseOptions` | Per-response + combined | Simultaneously powering several responses |

**Supported optimality criteria** (set via `DesignOptions.criterion`):

- **I-optimality** (default) — minimises *average prediction variance* over the design region; preferred when prediction accuracy across the factor space matters most.
- **D-optimality** — maximises `det(X'X)`; preferred when precise coefficient estimation is the primary goal.
- **A-optimality** — minimises `trace((X'X)⁻¹)`; equalises coefficient-estimate variances.

New here? Start with the 10-minute guide: [Quick Start Guide](docs/quickstart.md).  
Looking for task-oriented examples? See [Recipes](docs/recipes.md).

---

## Table of Contents

- [Installation](#installation)
- [Quick Start Guide (10 minutes)](docs/quickstart.md)
- [Recipes](docs/recipes.md)
- [Quick Start — Python API](#quick-start--python-api)
- [Quick Start — CLI](#quick-start--cli)
- [Streamlit Web UI](#streamlit-web-ui)
- [Power Modes](#power-modes)
- [Configuration Reference](#configuration-reference)
- [Output Structure](#output-structure)
- [Power Curves](#power-curves)
- [Sensitivity Analysis & MDE](#sensitivity-analysis)
- [Comparing Criteria](#comparing-criteria)
- [Augmenting Designs](#augmenting-an-existing-design)
- [Split-Plot Designs (Hard-to-Change Factors)](#split-plot-designs-hard-to-change-factors)
- [Diagnostics](#diagnostics)
- [Shareable Reports](#shareable-reports)
- [Candidate Set & Algorithm Details](#candidate-set--algorithm-details)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Installation

Requires Python ≥ 3.9.

If you just want the core optimizer, install that. If you want YAML, plots, reports, or the app, add the extras you actually need.

```bash
# Core install (from source)
pip install -e .

# With CLI support (YAML configs)
pip install -e ".[cli]"

# With visualization (power curve plots — matplotlib + plotly)
pip install -e ".[viz]"

# With Streamlit web UI (interactive frontend)
pip install -e ".[app]"

# With shareable HTML report generation (Jinja2 + Pillow)
pip install -e ".[report]"

# With PDF export support (requires system-level weasyprint dependencies)
pip install -e ".[report-pdf]"

# With progress bars and Excel export
pip install -e ".[extras]"

# Everything at once
pip install -e ".[all]"
```

**Core dependencies:** `numpy`, `scipy`, `pandas`, `patsy`

---

## Quick Start — Python API

If you work in notebooks or scripts, this is the fastest path from "I need a design" to something you can actually run.

### Contrast-based power

Specify which linear combination of coefficients you want to detect and by how much.

```python
from lattice_doe import (
    find_optimal_design,
    PowerContrastConfig,
    DesignOptions,
)

formula = "~ 1 + A + B + A:B"
factors = {
    "A": ["low", "high"],   # 2-level categorical → Patsy encodes 1 dummy column
    "B": (0.0, 10.0),       # continuous: (low, high) tuple
}
# With these factors, Patsy encodes p = 4 columns:
#   [Intercept, A[T.high], B, A[T.high]:B]
# so L must have exactly 4 columns.

# Contrast: test that the B main effect equals 0.5
# L must be (q x p) where p = number of columns in the Patsy model matrix
power_cfg = PowerContrastConfig(
    L=[[0, 0, 1, 0]],   # one-row contrast selecting the B main-effect coefficient
    delta=[0.5],         # minimum detectable effect (same units as sigma)
    alpha=0.05,
    power=0.80,
    sigma=1.0,
    max_n=500,
)

opts = DesignOptions(
    auto_candidate=True,   # adaptive candidate sizing (recommended)
    starts=8,
    random_state=42,
)

result = find_optimal_design(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=opts,
)

design_df  = result["design_df"]    # DataFrame: n-run optimal design
buckets_df = result["buckets_df"]   # DataFrame: unique run allocations with counts
report     = result["report"]       # dict: power, n, lambda, df, timing, etc.

print(f"n = {report['n']},  achieved power = {report['achieved_power']:.3f}")
print(buckets_df)
```

### Building contrasts from scenarios

Use `contrast_from_scenarios` to construct `L` and `delta` automatically by comparing two factor settings:

```python
from lattice_doe.contrasts import contrast_from_scenarios

scenario_a = {"A": "low",  "B": 5.0}
scenario_b = {"A": "high", "B": 5.0}

L, delta = contrast_from_scenarios(
    formula=formula,
    factors=factors,
    scenario_a=scenario_a,
    scenario_b=scenario_b,
    sesoi=1.0,   # smallest effect of interest (in response units)
)

power_cfg = PowerContrastConfig(L=L, delta=delta, alpha=0.05, power=0.80, sigma=1.0)
```

### Global R² power

Test whether the full model explains a meaningful proportion of variance.

```python
from lattice_doe import PowerR2Config

power_cfg = PowerR2Config(
    r2_target=0.15,   # detect R² ≥ 0.15
    alpha=0.05,
    power=0.80,
    max_n=500,
    lambda_mode="n",  # "n" (default) or "n_minus_p" (more conservative)
)

result = find_optimal_design(formula, factors, power_cfg, opts)
```

---

## Quick Start — CLI

If you would rather keep the logic in a config file and the outputs on disk, the CLI is the cleaner option.

Install with `pip install -e ".[cli]"` for YAML support, then run:

```bash
# Generate a starter config (no installation of PyYAML needed for this step)
lattice --template contrast > config.yml   # contrast mode template
lattice --template r2      > config.yml   # global R² mode template

# Generate a design
lattice --config config.yml --out ./output/design

# With Excel output and verbose logging
lattice --config config.yml --out ./output/design --excel -v

# Validate config without running (dry run)
lattice --config config.yml --dry-run
```

**Contrast mode config (`config.yml`):**

```yaml
formula: "~ 1 + A + B + A:B"

factors:
  A: [low, high]           # 2-level categorical → Patsy encodes 1 dummy column
  B: [0.0, 10.0]           # continuous [low, high]
# With these factors, Patsy encodes p = 4 columns:
#   [Intercept, A[T.high], B, A[T.high]:B]
# so L must have exactly 4 columns.

# Option 1: explicit contrast matrix
contrast:
  L: [[0, 0, 1, 0]]        # selects the B main-effect coefficient (column index 2)
  delta: [0.5]

# Option 2: scenario-based (auto-builds L and delta — safer, formula-agnostic)
# contrast:
#   scenario_a: {A: low,  B: 5.0}
#   scenario_b: {A: high, B: 5.0}
#   sesoi: 1.0

alpha: 0.05
power: 0.80
sigma: 1.0

design:
  auto_candidate: true
  starts: 8
  algo: fedorov
  random_state: 42

output:
  basename: my_design
  excel: false
```

**Global R² mode config:**

```yaml
formula: "~ 1 + A + B + A:B"
factors:
  A: [low, med, high]
  B: [0.0, 10.0]

r2_target: 0.15
alpha: 0.05
power: 0.80

design:
  auto_candidate: true
  starts: 8
  random_state: 42
```

The CLI always writes `<basename>_design.csv`, `<basename>_buckets.csv`, and `<basename>_report.json`. Pass `--excel` (or set `output.excel: true`) to also produce an `.xlsx` workbook.

---

## Streamlit Web UI

An interactive browser-based frontend lets you configure and run designs, explore sensitivity, compare criteria, and download results. It is useful when you want something more guided than a script, or when not everyone on the team wants to touch Python.

### Local run

```bash
# Install the package with Streamlit and Plotly
pip install -e ".[app]"

# Launch (opens at http://localhost:8501)
streamlit run app/app.py
```

### Docker

```bash
docker build -t lattice-doe .
docker run -p 8501:8501 lattice-doe
# Open http://localhost:8501
```

### Streamlit Community Cloud (free hosting)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository and set **Main file path** to `app/app.py`.
4. Click **Deploy** — no secrets or environment variables required.

For a full walkthrough see [Quick Start Guide § 5](docs/quickstart.md#5-streamlit-web-ui).

---

## Power Modes

Pick the power model that matches the question you are actually asking. This library does not make you pretend every problem is the same kind of effect test.

### Contrast-based (`PowerContrastConfig`)

Tests H₀: Lβ = 0 against H₁: Lβ = δ using an F-test on the linear contrast.

**Noncentrality parameter:**

```
λ = δᵀ [L (X'X)⁻¹ Lᵀ]⁺ δ / σ²
```

- `df_num` = rank(L), `df_denom` = n − rank(X)
- Supports multiple simultaneous contrasts (q > 1 rows in L)
- L and delta are validated for shape consistency and non-zero content

### Global R² (`PowerR2Config`)

Tests H₀: R² = 0 (all slopes are zero) using the omnibus F-test.

**Noncentrality parameter** (via Cohen's f² = R²/(1−R²)):

| `lambda_mode` | Formula | Matches |
|---|---|---|
| `"n"` (default) | λ = f² · n | G\*Power, statsmodels |
| `"n_minus_p"` | λ = f² · (n − p) | More conservative |

- `df_num` = number of slope parameters (intercept excluded, per G\*Power convention)
- `df_denom` = n − rank(X)

### GLM Wald χ² (`PowerGLMContrastConfig`)

Tests H₀: Lβ = 0 using a Wald chi-square statistic for binomial (logistic) or Poisson GLM responses.

The design search uses a **null-based locally optimal** information matrix: `M = w · X'X` where `w = p₀(1 − p₀)` (binomial) or `w = μ₀` (Poisson). Because `w` is a positive scalar it cancels from I/D/A criteria, so the Fedorov exchange is structurally identical to OLS — only the power calculation changes.

> **Approximation scope.** The Fisher weight `w` is a single scalar evaluated at the null baseline and applied uniformly to every design point. This is accurate when the true operating point is close to the baseline. For designs with wide covariate ranges and substantial slope effects, the true per-point weights `wᵢ = p(xᵢ)(1−p(xᵢ))` will vary across the design, and the constant-weight approximation may over- or understate power. Validate results via simulation when slopes are large relative to the baseline.

```python
from lattice_doe import PowerGLMContrastConfig
from lattice_doe.contrasts import contrast_from_scenarios

L, delta = contrast_from_scenarios(formula, factors, scenario_a, scenario_b, sesoi=0.4)

power_cfg = PowerGLMContrastConfig(
    L=L,
    delta=delta,           # effect on the linear-predictor (log-odds / log-rate) scale
    baseline=0.30,         # p₀ for binomial (0 < p₀ < 1) or μ₀ for Poisson (> 0)
    family="binomial",     # "binomial" or "poisson"
    link=None,             # None → canonical link (logit / log); or explicit "logit" / "log"
    alpha=0.05,
    power=0.80,
    max_n=500,
)

result = find_optimal_design(formula, factors, power_cfg, opts)
```

Use the CLI template to get started:

```bash
lattice --template glm-binomial > glm_config.yml
lattice --template glm-poisson  > glm_config.yml
```

---

## Configuration Reference

### `PowerContrastConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `L` | array (q×p) | — | Contrast matrix; must match model matrix column count |
| `delta` | array (q,) | — | Minimum detectable effects (one per contrast row) |
| `alpha` | float | `0.05` | Significance level |
| `power` | float | `0.80` | Target power |
| `sigma` | float | `1.0` | Residual standard deviation |
| `max_n` | int | `2000` | Hard cap on sample size search |
| `tol_power` | float | `1e-3` | Convergence tolerance |
| `max_iter` | int | `200` | Max n-search iterations |

### `PowerR2Config`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `r2_target` | float | — | Target R² effect size (0, 1) |
| `alpha` | float | `0.05` | Significance level |
| `power` | float | `0.80` | Target power |
| `max_n` | int | `2000` | Hard cap on sample size search |
| `lambda_mode` | `"n"` \| `"n_minus_p"` | `"n"` | Noncentrality convention |
| `tol_power` | float | `1e-3` | Convergence tolerance |
| `max_iter` | int | `200` | Max n-search iterations |

### `PowerGLMContrastConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `L` | array (q×p) | — | Contrast matrix; must match model matrix column count |
| `delta` | array (q,) | — | Effect sizes on the linear-predictor scale (log-odds for binomial, log-rate for Poisson) |
| `baseline` | float | — | Baseline mean on the response scale: probability ∈ (0, 1) for binomial; expected count > 0 for Poisson |
| `family` | `"binomial"` \| `"poisson"` | `"binomial"` | Response distribution family |
| `link` | `"logit"` \| `"log"` \| `None` | `None` | Link function; `None` selects the canonical link for the family |
| `alpha` | float | `0.05` | Significance level |
| `power` | float | `0.80` | Target power |
| `max_n` | int | `2000` | Hard cap on sample size search |
| `tol_power` | float | `1e-3` | Convergence tolerance |
| `max_iter` | int | `200` | Max n-search iterations |

### `DesignOptions`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `random_state` | int | `123` | Global random seed (must be an integer; `None` is not allowed) |
| `algo` | `"fedorov"` \| `"coordinate"` | `"fedorov"` | API-compatibility selector; both map to the internal Fedorov exchange engine |
| `starts` | int | `5` | Number of random starts |
| `max_iter` | int | `1000` | Max iterations per start |
| `xtx_jitter` | float | `1e-8` | Ridge added to X'X for numerical stability |
| `criterion` | str | `"I"` | Optimality criterion: `"I"` (minimise average prediction variance), `"D"` (maximise `det(X'X)`), or `"A"` (minimise `trace((X'X)⁻¹)`) |
| `candidate_points` | int | `2000` | Fixed candidate size (when `auto_candidate=False`) |
| `auto_candidate` | bool | `False` | Adaptively size the candidate set |
| `cand_min` | int | `1000` | Minimum candidate points (auto mode) |
| `cand_max` | int | `10000` | Maximum candidate points (auto mode) |
| `cat_cells_cap` | int | `10000` | Cap on categorical cell enumeration |
| `per_cell_alpha` | float | `1.5` | Candidate multiplier per categorical cell |
| `per_cell_min` | int | `5` | Min points per cell (mixed designs) |
| `per_cell_max` | int | `20` | Max points per cell (mixed designs) |
| `allow_candidate_growth` | bool | `False` | Grow candidate once if conditioning is poor |
| `growth_factor` | float | `2.0` | Multiplier applied when growing candidate |
| `workers` | int \| None | `None` | Parallel workers for starts (None = serial) |
| `parallel_seed_stride` | int | `10000` | Seed offset between parallel workers |
| `constraint_func` | callable \| None | `None` | Row-level feasibility filter (Python callable) |
| `constraint_expr` | str \| None | `None` | Row-level feasibility filter as a string expression (YAML/JSON-friendly alternative to `constraint_func`; see [Feasibility constraints](#candidate-set--algorithm-details)) |
| `n_blocks` | int \| None | `None` | Number of blocks (≥ 2 enables blocking; `None` / `0` = unblocked) |
| `split_plot` | `SplitPlotOptions` \| None | `None` | Split-plot configuration for hard-to-change factors (see [Split-Plot Designs](#split-plot-designs-hard-to-change-factors)) |

**Parallel starts note:** On macOS and Windows, set `workers > 1` only inside `if __name__ == "__main__":` (standard `multiprocessing` requirement).

---

## Output Structure

`find_optimal_design(...)` returns a dict with three keys: the design, the replication structure, and the audit trail for how the optimizer got there. (For multi-response designs, see `find_multiresponse_design(...)` which returns a different structure with `design`, `buckets`, `responses`, and flat summary fields.)

### `result["design_df"]` — `DataFrame`

The n-run optimal design (I-, D-, or A-optimal depending on `criterion`). Each row is a selected point from the candidate set. Duplicate rows represent replicated runs.

### `result["buckets_df"]` — `DataFrame`

Unique run allocations with replication counts:

| A | B | count |
|---|---|---|
| low | 3.2 | 3 |
| high | 7.8 | 2 |
| ... | ... | ... |

### `result["report"]` — `dict`

Key metrics from the design search:

| Key | Description |
|---|---|
| `n` | Final sample size |
| `p` | Number of model parameters |
| `df_num` | Numerator degrees of freedom |
| `df_denom` | Denominator degrees of freedom |
| `alpha` | Significance level used |
| `target_power` | Requested power |
| `achieved_power` | Power of the returned design |
| `noncentrality_lambda` | Noncentrality parameter λ |
| `criterion` | Optimality criterion |
| `algo` | Search algorithm used |
| `starts` | Number of starts configured |
| `workers` | Number of parallel workers used |
| `candidate_points` | Candidate set size used |
| `elapsed_sec` | Wall-clock seconds for the full n-search (excludes file export) |
| `search_strategy` | Phases executed, e.g. `"bisection"`, `"bisection+growth"`, `"bisection+verification"` |
| `verify_window` | Number of n values checked in the Phase 2 downward scan (0 if only bisection ran) |
| `random_state` | The `random_state` seed that was used (from `DesignOptions`) |
| `warnings` | List of warning messages issued during search (empty list if none) |
| `diagnostics` | Dict of design quality metrics (condition number, D-efficiency, etc.) |

---

## Power Curves

Use power curves when you are not ready to commit to one design yet and want to see how much the answer moves as `n` or effect size changes.

```python
from lattice_doe import power_curve_by_n, power_curve_by_effect

# Power vs. n (sweeps a range of n values)
df_n = power_curve_by_n(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=opts,
)
print(df_n)   # columns: n, power

# Power vs. effect size (sweeps delta or r2_target at a fixed n)
df_eff = power_curve_by_effect(
    formula=formula,
    factors=factors,
    n=30,            # fixed sample size for the sweep
    power_cfg=power_cfg,
    design_opts=opts,
)
print(df_eff)   # columns: effect_scale,power (contrast) or r2_target,power (R²)
```

Both functions respect `auto_candidate` in `DesignOptions`.

### 2D power surface

Sweep two parameters simultaneously to produce a contour map — useful for understanding the joint sensitivity of your design to (n, effect), (effect, sigma), etc.

```python
from lattice_doe.power_curves import power_surface_2d

result = power_surface_2d(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    param1="n",           # y-axis: 'n', 'effect', 'sigma', or 'alpha'
    param1_range=(10, 80),
    param2="effect",      # x-axis: multiplier on delta (1.0 = nominal)
    param2_range=(0.3, 2.0),
    grid_points=20,
    design_opts=opts,
    plot=True,            # returns a filled contour figure
)

print(result["data"])          # DataFrame: param1, param2, power, noncentrality_lambda
print(result["power_grid"])    # 2D numpy array of power values
# result["figure"]             # matplotlib Figure with contour plot
```

**Notes on `param1` / `param2` semantics:**

| Parameter | `PowerContrastConfig` | `PowerR2Config` |
|---|---|---|
| `"n"` | Sample size (integer) | Sample size (integer) |
| `"effect"` | Scale multiplier on `delta` (1.0 = nominal) | Actual `r2_target` value |
| `"sigma"` | Absolute σ value | ❌ not applicable |
| `"alpha"` | Significance level | Significance level |

When neither axis is `"n"`, the function builds one optimal design at a representative n and sweeps analytically (fast). When `"n"` is an axis, one optimal design is built per unique n value (expensive but cached).

### Interactive charts (Plotly)

All four power-analysis functions accept an opt-in `plot_backend="plotly"` parameter that returns a `plotly.graph_objects.Figure` instead of a matplotlib Figure.  The default (`"matplotlib"`) is unchanged — no existing code breaks.

```bash
pip install -e ".[viz]"   # includes plotly>=5.0
```

```python
from lattice_doe.power_curves import power_curve_by_n

result = power_curve_by_n(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=opts,
    plot=True,
    plot_backend="plotly",
)
fig = result["figure"]   # plotly.graph_objects.Figure
fig.show()               # interactive in Jupyter / browser
```

Plotly charts support hover tooltips, zoom/pan, and one-click PNG export (camera icon in the toolbar).  They also work directly in Streamlit:

```python
import streamlit as st
st.plotly_chart(result["figure"])
```

The same `plot_backend` parameter is available on `power_curve_by_effect`, `power_surface_2d`, and `power_sensitivity`.  To access the figure from those functions call the implementation modules directly (the `lattice_doe` top-level wrappers discard the figure for backward compatibility):

```python
from lattice_doe.power_curves import power_curve_by_effect, power_surface_2d
from lattice_doe import power_sensitivity
```

### Sensitivity analysis

Reveal how much power changes if a key assumption is wrong — without rebuilding any designs.

```python
from lattice_doe import power_sensitivity

# Contrast mode: sweep sigma
sensitivity = power_sensitivity(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,          # PowerContrastConfig → sweeps sigma
    design_df=result["design_df"],
    sigma_range=(0.5, 2.0),       # sweep σ from 0.5 to 2.0
    sigma_points=30,
    plot=True,
)
print(sensitivity["data"])           # DataFrame: sigma, power, noncentrality_lambda
print(sensitivity["nominal_power"])  # power at the configured sigma
# sensitivity["figure"]              # matplotlib Figure

# R² mode: sweep r2_target (sigma does not enter the R² power formula)
sensitivity_r2 = power_sensitivity(
    formula=formula,
    factors=factors,
    power_cfg=r2_power_cfg,       # PowerR2Config → sweeps r2_target
    design_df=result["design_df"],
    r2_range=(0.05, 0.50),        # sweep R² from 5 % to 50 %
    r2_points=30,
    plot=True,
)
print(sensitivity_r2["data"])        # DataFrame: r2_target, power, noncentrality_lambda
print(sensitivity_r2["r2_nominal"])  # the nominal r2_target from power_cfg
```

### Minimum detectable effect

Find the smallest effect your design can detect at a given power — no new design needed.

```python
from lattice_doe import min_detectable_effect

# Contrast mode: MDE expressed as a scale factor on delta
mde = min_detectable_effect(
    design_df=result["design_df"],
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,       # PowerContrastConfig
    target_power=0.80,
)
print(mde["mde"])              # scale factor (1.0 = original delta is just detectable)
print(mde["achieved_power"])   # power at the MDE

# R² mode: MDE expressed as the minimum detectable r2_target
mde_r2 = min_detectable_effect(
    design_df=result["design_df"],
    formula=formula,
    factors=factors,
    power_cfg=r2_power_cfg,    # PowerR2Config
    target_power=0.80,
)
print(mde_r2["mde"])           # minimum r2_target detectable at 80 % power
```

### Comparing criteria

Not sure which optimality criterion is right for your study?  `compare_criteria` runs the full powered-design search under each of `"I"`, `"D"`, and `"A"` (or any subset) and returns a side-by-side summary in a single call.

```python
from lattice_doe import compare_criteria, DesignOptions

comparison = compare_criteria(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,          # shared across all criteria
    design_opts=DesignOptions(    # criterion field is overridden per run
        auto_candidate=True,
        starts=8,
        random_state=42,
    ),
    criteria=["I", "D", "A"],    # default; any non-empty subset is valid
    plot=True,                    # side-by-side bar charts (requires matplotlib)
)

print(comparison["summary"])
# criterion   n   achieved_power  elapsed_sec  condition_number  d_efficiency
# I          24        0.814         1.23           12.5             0.81
# D          22        0.823         1.17           10.2             1.00
# A          23        0.811         1.19           11.8             0.94

# Access the full result for a specific criterion
i_design = comparison["results"]["I"]["design_df"]
d_report  = comparison["results"]["D"]["report"]
```

The function never mutates *design_opts* — it uses `dataclasses.replace` to create a per-criterion copy.  When only two criteria are needed, pass `criteria=["I", "D"]` etc.

---

### Augmenting an existing design

Add runs to a design that already exists, fixing the original rows in place:

```python
from lattice_doe import augment_design, DesignOptions

# Suppose existing_design is a DataFrame with 20 runs
augmented, new_runs = augment_design(
    design_df=existing_design,
    m=5,                            # add 5 new runs
    formula=formula,
    factors=factors,
    design_opts=DesignOptions(criterion="I", random_state=42),
)

print(f"Original: {len(existing_design)} runs")
print(f"Augmented: {len(augmented)} runs")
print(new_runs)                     # the 5 newly added rows
```

`augment_design` uses a greedy one-point-at-a-time exchange that optimises the same criterion (`"I"`, `"D"`, or `"A"`) as the original design search. It is fast but does not guarantee global optimality.

---

## Split-Plot Designs (Hard-to-Change Factors)

When some factors are **hard to change** (HTC) between runs — oven temperature, batch composition, equipment configuration — a split-plot design groups runs into **whole plots** so that HTC factors are reset only once per group, while **easy-to-change (ETC)** sub-plot factors vary freely within each group.

Ignoring this structure and using standard OLS inflates the apparent precision of WP-factor estimates. This package uses a **GLS information matrix** (`X'V⁻¹X` where `V = η·ZZ' + I`) for both design search and power calculations, giving correct Type-I error and power for both WP and SP effects.

### `SplitPlotOptions` reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `htc_factors` | `List[str]` | — | Factor names that are hard-to-change (whole-plot factors). Must be a non-empty subset of the factor names passed to the API. |
| `n_whole_plots` | int | — | Number of whole plots (outer randomization units). Must be ≥ 2. |
| `eta` | float | `1.0` | Variance ratio σ²_wp / σ²_sp. Must be ≥ 0. `eta=0` degenerates to OLS. |
| `subplots_per_wp` | int \| None | `None` | Sub-plots per whole plot. `None` → auto-computed as `max(2, ceil(p / n_wp) + 1)`. |
| `df_method` | `"auto"` \| `"conservative"` \| `"sp_only"` | `"auto"` | Denominator df assignment. `"auto"` classifies each contrast by stratum; `"conservative"` always uses WP df; `"sp_only"` always uses SP df. |

### Python API

```python
from lattice_doe import (
    find_optimal_design,
    SplitPlotOptions,
    DesignOptions,
    PowerContrastConfig,
    power_curve_by_wp,
)
from lattice_doe.contrasts import contrast_from_scenarios

formula = "~ 1 + A + B + C"
factors = {
    "A": (-1.0, 1.0),   # HTC: whole-plot factor
    "B": (-1.0, 1.0),   # HTC: whole-plot factor
    "C": (-1.0, 1.0),   # ETC: sub-plot factor
}

L, delta = contrast_from_scenarios(
    formula, factors,
    {"A": -1.0, "B": -1.0, "C": 0.0},
    {"A":  1.0, "B":  1.0, "C": 0.0},
    sesoi=1.0,
)
power_cfg = PowerContrastConfig(L=L, delta=delta, power=0.80, sigma=1.0, max_n=200)

result = find_optimal_design(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=DesignOptions(
        split_plot=SplitPlotOptions(
            htc_factors=["A", "B"],
            n_whole_plots=6,
            eta=1.5,           # σ²_wp / σ²_sp
            subplots_per_wp=4, # optional; auto-computed if omitted
            df_method="auto",  # "auto" | "conservative" | "sp_only"
        ),
        starts=8,
        random_state=42,
    ),
)

print(f"n = {result['report']['n']},  power = {result['report']['achieved_power']:.3f}")
print(result["report"]["split_plot"])
# {'n_whole_plots': 6, 'subplots_per_wp': 4, 'n_total': 24,
#  'eta': 1.5, 'htc_factors': ['A', 'B'], 'etc_factors': ['C'], 'df_method': 'auto'}

# The design DataFrame includes a __wp_id__ column
design_df = result["design_df"]
print(design_df[["__wp_id__", "A", "B", "C"]].head(8))
```

### Power vs. number of whole plots

```python
df = power_curve_by_wp(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    subplots_per_wp=4,
    htc_factors=["A", "B"],
    eta=1.5,
    wp_range=(3, 12),   # sweep n_whole_plots from 3 to 12
    wp_points=10,
    design_opts=DesignOptions(starts=5, random_state=42),
)
print(df)  # columns: n_wp, n_total, power, noncentrality_lambda
```

### CLI (YAML config)

Add a `split_plot:` block inside the `design:` section:

```yaml
formula: "~ 1 + A + B + C"
factors:
  A: [-1.0, 1.0]
  B: [-1.0, 1.0]
  C: [-1.0, 1.0]

contrast:
  scenario_a: {A: -1.0, B: -1.0, C: 0.0}
  scenario_b: {A:  1.0, B:  1.0, C: 0.0}
  sesoi: 1.0

alpha: 0.05
power: 0.80
sigma: 1.0

design:
  starts: 8
  random_state: 42
  split_plot:
    htc_factors: [A, B]
    n_whole_plots: 6
    eta: 1.5
    subplots_per_wp: 4    # omit for auto
    df_method: auto       # auto | conservative | sp_only
```

### η-sensitivity sweep

Assess how power degrades as the variance ratio η grows:

```python
from lattice_doe import power_sensitivity

result = find_optimal_design(formula, factors, power_cfg,
    DesignOptions(split_plot=SplitPlotOptions(htc_factors=["A","B"], n_whole_plots=6, eta=1.5),
                  starts=5, random_state=42))

sens = power_sensitivity(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_df=result["design_df"],
    eta_range=(0.0, 5.0),  # sweep η from 0 (OLS) to 5
    eta_points=25,
)
print(sens["data"])        # columns: eta, power, noncentrality_lambda
```

### Notes

- `n_blocks` and `split_plot` cannot both be set (blocked split-plots are not yet supported).
- `eta=0` degenerates to OLS: the GLS power equals the OLS power for that design.
- The `"conservative"` df_method never produces anti-conservative power for WP-factor contrasts.
- Set `criterion_ignore_vr=True` inside `SplitPlotOptions` to use the standard OLS criterion during design search while keeping GLS power calculations (useful for benchmarking only).
- **Denominator df approximation.** df assignment uses a WP-vs-SP stratum classification heuristic, not a full Satterthwaite or Kenward-Roger small-sample correction. For balanced designs with a single variance component this gives exact df. For unbalanced designs or near-singular settings the heuristic can be conservative or anti-conservative; use `df_method="conservative"` when in doubt.

---

## Diagnostics

If you want more than "here is your design, good luck," export diagnostics alongside the main outputs:

```python
result = find_optimal_design(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=opts,
    export_diagnostics_to="./output/",   # folder path
)
```

Diagnostics written to the output folder include:

- **Condition number** — detects near-collinearity
- **D-efficiency** — relative efficiency vs. D-optimal reference
- **Leverage** — per-row hat values and summary statistics
- **VIFs** — variance inflation factors per regressor
- **I-criterion** — average prediction variance over the candidate region

Output formats: HTML tables, CSV, and optional plots.

---

## Shareable Reports

Generate a self-contained HTML file (no external dependencies, works offline) that summarises the design configuration, power metrics, design table, diagnostics, and an embedded power-curve figure. This is handy when you need to hand results to someone who does not want a notebook or CLI log.

### Install

```bash
pip install -e ".[report]"          # HTML reports (Jinja2 + Pillow)
pip install -e ".[report-pdf]"      # also enables PDF export via weasyprint
```

### Python API

```python
from lattice_doe import generate_report

generate_report(
    result=result,          # dict returned by find_optimal_design()
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    output_path="./reports/my_design.html",   # .html or .pdf
)
```

### Inline with the optimizer

Pass `export_report_to=` directly to `find_optimal_design()` to write the report immediately after the design is found:

```python
result = find_optimal_design(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=opts,
    export_report_to="./output/",   # writes the default HTML report into this folder
)

# Path stored in result for reference
print(result["report"]["report_path"])
```

If report generation fails (e.g. `jinja2` not installed), the error message is stored in `result["report"]["report_path_error"]` and the design result is still returned normally.

### CLI

```bash
lattice --config my_config.yaml --out results --html-report
# writes: results_report.html alongside results_design.csv, etc.
```

Or set it permanently in your YAML config:

```yaml
output:
  html_report: true
```

### PDF export

Replace the `.html` extension with `.pdf`:

```python
generate_report(..., output_path="report.pdf")
```

> **Note:** PDF export requires `weasyprint`, which depends on system-level libraries (`libpango`, `libcairo`, `libgdk-pixbuf`). These are unavailable on Streamlit Community Cloud and some CI environments. Install with `pip install -e ".[report-pdf]"` and follow the [weasyprint installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html) for your OS.

---

## Google Sheets Integration

Connect directly to a Google Spreadsheet to read your design config and write results back — no local YAML file needed. It is a practical option for teams who already live in spreadsheets.

### Install

```bash
pip install "lattice-doe[sheets]"
```

### Create a starter spreadsheet (once)

```python
from lattice_doe import create_sheet_template

# Creates a new spreadsheet with Config/Results/Design/Buckets sheets pre-filled
url = create_sheet_template(
    title="My DOE",
    credentials="service_account.json",   # or None for OAuth2 browser flow
    example="r2",                         # "r2" or "contrast"
)
print(url)  # open this URL, fill in your factors and formula, then run below
```

### Run from the spreadsheet

```python
from lattice_doe import sheets_run

result = sheets_run(url, credentials="service_account.json")
print(f"Optimal n = {result['report']['n']}")
print(f"Achieved power = {result['report']['achieved_power']:.3f}")
# Design/Results/Buckets sheets are now populated in the spreadsheet
```

### CLI

```bash
lattice --sheets "https://docs.google.com/spreadsheets/d/…" \
            --sheets-credentials service_account.json
```

If `--sheets-credentials` is omitted, the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is checked, then an OAuth2 browser flow is used as a fallback.

### Authentication

| `credentials` value | Auth mode |
|---------------------|-----------|
| `"path/to/sa.json"` | Service account — for CI/automation; share the spreadsheet with the SA email |
| `None` | OAuth2 browser flow — opens a tab on first use, caches token in `~/.config/gspread/` |

---

## Candidate Set & Algorithm Details

### Factor specifications

```python
factors = {
    "Temperature": (20.0, 80.0),       # continuous: 2-element numeric tuple/list
    "Catalyst":    ["A", "B", "C"],    # categorical: list of levels
    "Time":        (1.0, 5.0),         # continuous
}
```

Continuous factors with exactly two numeric elements are sampled via Latin Hypercube. Categorical factors are enumerated as a Cartesian product (capped at `cat_cells_cap`). Mixed designs combine both.

### Adaptive candidate sizing (`auto_candidate=True`)

Recommended for most use cases. The package sizes the candidate set based on:

- Number and type of factors
- Categorical cell count (capped at `cat_cells_cap`)
- `per_cell_alpha`, `per_cell_min`, `per_cell_max` multipliers
- Bounded by `cand_min` and `cand_max`

### Candidate growth

If `allow_candidate_growth=True`, the candidate set is grown by `growth_factor` once during the search if the design matrix condition number exceeds 10⁶. This is a safety net for difficult factor spaces.

### Search algorithm

Design search uses an internal Fedorov point-exchange optimizer that operates
directly on the Patsy model matrix.

The `algo` option (`"fedorov"` or `"coordinate"`) is currently retained for
API compatibility; both settings route to this internal exchange implementation.

The core design search uses an internal vectorised Fedorov exchange that operates
directly on the Patsy model matrix and has no dependency on external design-of-experiments libraries.

### Feasibility constraints

Two equivalent ways to exclude infeasible candidate points:

**Python callable** — full flexibility, for use in Python scripts:

```python
def no_high_temp_low_time(row):
    return not (row["Temperature"] > 70 and row["Time"] < 2)

opts = DesignOptions(constraint_func=no_high_temp_low_time)
```

**String expression** — YAML/JSON-friendly, reproduces in config files without Python code:

```python
# In Python:
opts = DesignOptions(constraint_expr="not (Temperature > 70 and Time < 2)")

# Compound and math expressions work too:
opts = DesignOptions(constraint_expr="sqrt(Temperature) + Pressure <= 20")
opts = DesignOptions(constraint_expr="Catalyst != 'C' or Time <= 3")
```

In a YAML config file:

```yaml
design:
  constraint_expr: "not (Temperature > 70 and Time < 2)"
```

Available functions inside `constraint_expr`: `abs`, `min`, `max`, `round`, `sqrt`, `log`, `log2`, `log10`, `exp`, `floor`, `ceil`, `pi`.  The expression is evaluated with each factor's column name as a local variable.  No imports or arbitrary code execution are permitted.

If both `constraint_func` and `constraint_expr` are set, `constraint_expr` takes precedence.

---

## Reproducibility

Fix `random_state` in `DesignOptions` to reproduce candidate generation, design search, and parallel start assignments exactly. If you need to defend why a design changed, start here.

---

## Troubleshooting

Most failures here are informative. The package is usually telling you that the model, the search limits, or the contrast definition needs another look. The three most common errors and how to fix them are below.

### `ValueError: power_cfg.max_n (N) must be greater than the number of model parameters p (M).`

**What it means.** The cap on the search range (`max_n`) is too small relative to the number of model parameters `p`. Patsy expanded the formula into `M` columns (intercept, main effects, interactions, dummy levels for categoricals), and any usable design needs more runs than parameters. The check fires before the search starts, so no design is returned.

**How to inspect `p` for your formula** without running the search:

```python
from lattice_doe.candidate import build_candidate
from lattice_doe.model_matrix import build_model_matrix

formula = "~ 1 + A + B + A:B + C"
factors = {"A": (-1.0, 1.0), "B": (-1.0, 1.0), "C": ["low", "med", "high"]}

cand = build_candidate(factors, candidate_points=20, seed=0)
X, names = build_model_matrix(formula, cand)
print(f"p = {X.shape[1]}")
print("columns:", names)
```

**Fixes.** Raise `max_n` in the power config (e.g. `PowerContrastConfig(..., max_n=200)`), or simplify the formula by dropping interactions or collapsing categorical levels.

### `ValueError: Contrast L has X columns but model has p_treat=Y treatment parameters.`

**What it means.** Your contrast matrix `L` has the wrong number of columns. `L` must have one column per parameter in the **Patsy-encoded** model matrix — including the intercept, dummy columns for categorical factors, and interaction terms. Hand-written `L` arrays are the usual culprit.

**How to inspect the column layout:**

```python
from lattice_doe.candidate import build_candidate
from lattice_doe.model_matrix import build_model_matrix

cand = build_candidate(factors, candidate_points=20, seed=0)
X, names = build_model_matrix(formula, cand)
for i, name in enumerate(names):
    print(f"  column {i}: {name}")
```

**Fix (recommended).** Don't write `L` by hand. Use `contrast_from_scenarios`, which always produces a correctly-shaped row for the encoded matrix:

```python
from lattice_doe.contrasts import contrast_from_scenarios

L, delta = contrast_from_scenarios(
    formula=formula,
    factors=factors,
    scenario_a={"A": -1.0, "B": 0.0},
    scenario_b={"A":  1.0, "B": 0.0},
    sesoi=2.0,
)
```

If you do need a hand-written `L`, rebuild it to match the printed `names` list — one entry per column.

### `RuntimeWarning: Design generation finished without converging to target power.`

**What it means.** The search stopped without the design hitting the target power. The full warning message names the limit that was hit (`max_iter` or `max_n`), the achieved power, and the final `n`. The best design found is still returned — `result["design_df"]` and `result["report"]` are both populated — but `report["achieved_power"]` is below `power_cfg.power` and the warning text lands in `report["warnings"]`.

**How to inspect what happened:**

```python
result = find_optimal_design(
    formula=formula,
    factors=factors,
    power_cfg=PowerContrastConfig(L=L, delta=delta, power=0.80, sigma=1.0, max_n=50),
    design_opts=DesignOptions(criterion="I"),
)

print("achieved :", result["report"]["achieved_power"])
print("target   :", result["report"]["target_power"])
print("strategy :", result["report"]["search_strategy"])
print("warnings :", result["report"]["warnings"])
```

**Fixes, in order of preference.**
- Raise `max_n` to give the search more room.
- Raise `starts` in `DesignOptions` to do more random restarts (helps when the search keeps landing in local optima).
- Relax `power_cfg.power` to a target you can actually hit.
- Increase the SESOI (`delta`) — if you're asking for an effect smaller than the noise can resolve at any reasonable `n`, no amount of design optimization will save you.

### Other common failures

**Poor conditioning / near-singular X'X.** Enable `allow_candidate_growth=True`, increase `candidate_points`, or bump `xtx_jitter` slightly (e.g. `1e-6`).

**Parallelism on macOS / Windows.** Guard `workers > 1` calls inside `if __name__ == "__main__":`.

---

## License

MIT
