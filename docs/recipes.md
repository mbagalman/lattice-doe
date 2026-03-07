# Recipes

Task-oriented examples for common workflows.

## 1) Compare I vs D vs A before committing

```python
from iopt_power_design import (
    compare_criteria,
    PowerContrastConfig,
    DesignOptions,
)

formula = "~ 1 + A + B + A:B"
factors = {"A": ["low", "high"], "B": (0.0, 10.0)}

power_cfg = PowerContrastConfig(
    L=[[0, 0, 1, 0]],
    delta=[0.5],
    power=0.80,
    sigma=1.0,
)

comparison = compare_criteria(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=DesignOptions(auto_candidate=True, starts=5, random_state=42),
)

print(comparison["summary"][["criterion", "n", "achieved_power", "elapsed_sec"]])
```

Use this when you want a fast side-by-side tradeoff across criteria.

## 2) Add runs to an existing design without rebuilding from scratch

```python
from iopt_power_design import augment_design, DesignOptions

# existing_design is your current design DataFrame
augmented_df, new_runs_df = augment_design(
    design_df=existing_design,
    m=4,
    formula="~ 1 + A + B + A:B",
    factors={"A": ["low", "high"], "B": (0.0, 10.0)},
    design_opts=DesignOptions(criterion="I", auto_candidate=True, random_state=42),
)

print("new runs:", len(new_runs_df))
print(new_runs_df)
```

Use this when budget unlocks a few extra runs and you need incremental improvement.

## 3) Check robustness to assumptions (sensitivity)

Assumes you already have a `result` from `i_optimal_powered_design(...)` (see Recipe 1 or the Quick Start Guide).

Contrast mode (vary sigma):

```python
from iopt_power_design import power_sensitivity

sens = power_sensitivity(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,          # PowerContrastConfig used to build the design
    design_df=result["design_df"],
    sigma_range=(0.6, 1.8),
    sigma_points=21,
)

print(sens["data"].head())        # columns: sigma, power, noncentrality_lambda
print(sens["nominal_power"])      # power at the original sigma
```

R² mode (vary assumed R²):

```python
from iopt_power_design import PowerR2Config, i_optimal_powered_design, power_sensitivity
from iopt_power_design import DesignOptions

r2_cfg = PowerR2Config(r2_target=0.15, power=0.80, alpha=0.05)
result_r2 = i_optimal_powered_design(
    formula, factors, r2_cfg,
    DesignOptions(auto_candidate=True, random_state=42),
)

sens_r2 = power_sensitivity(
    formula=formula,
    factors=factors,
    power_cfg=r2_cfg,
    design_df=result_r2["design_df"],
    r2_range=(0.05, 0.30),
    r2_points=21,
)

print(sens_r2["data"].head())     # columns: r2_target, power, noncentrality_lambda
```

## 4) Compute minimum detectable effect (MDE) for a fixed design

Assumes you already have a `result` from `i_optimal_powered_design(...)`.

```python
from iopt_power_design import min_detectable_effect

mde = min_detectable_effect(
    design_df=result["design_df"],
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    target_power=0.80,
)

# Contrast mode: mde["mde"] is a scale factor on delta (1.0 = delta is just detectable)
print(mde["mde"])
print(mde["achieved_power"])
```

Use this when your design is fixed and you want to quantify what it can detect.

## 5) Use declarative feasibility constraints (YAML-friendly)

```python
from iopt_power_design import DesignOptions

opts = DesignOptions(
    auto_candidate=True,
    constraint_expr="not (Temperature > 70 and Time < 2)",
    random_state=42,
)
```

You can keep the same expression in YAML configs (`constraint_expr`) for reproducible pipelines.

## 6) Reproducible runs across machines

```python
from iopt_power_design import DesignOptions

opts = DesignOptions(
    auto_candidate=True,
    starts=8,
    workers=4,
    random_state=123,  # must be int
)
```

Recommendations:
- Keep `formula`, `factors`, and `random_state` fixed.
- Keep `starts`/`workers` fixed when comparing alternatives.
- Persist `result["report"]` with each run to track diagnostics and metadata.

## 7) Generate a shareable HTML report for a team member

```bash
pip install -e ".[report]"
```

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

The output is a single `.html` file with all CSS inline and figures embedded as base64 — no internet connection required. Share it by email or drop it in a shared folder; recipients can open it in any browser without installing Python.

To write the report automatically when the design is found:

```python
result = i_optimal_powered_design(
    formula, factors, power_cfg, opts,
    export_report_to="./reports/",   # writes iopt_report.html into this folder
)
```

For PDF output, change the extension to `.pdf` (requires `pip install -e ".[report-pdf]"`):

```python
generate_report(..., output_path="./reports/my_design_report.pdf")
```

## 8) Interactive Plotly power charts

Requires `pip install -e ".[viz]"` (includes `plotly>=5.0`).

### Power vs. sample size — two-panel interactive figure

```python
from iopt_power_design.power_curves import power_curve_by_n
from iopt_power_design import PowerContrastConfig, DesignOptions

power_cfg = PowerContrastConfig(L=[[0,0,1,0]], delta=[0.5], sigma=1.0, power=0.80)
opts = DesignOptions(auto_candidate=True, starts=5, random_state=42)

result = power_curve_by_n(
    formula="~ 1 + A + B + A:B",
    factors={"A": ["low","high"], "B": (0.0, 10.0)},
    power_cfg=power_cfg,
    design_opts=opts,
    plot=True,
    plot_backend="plotly",
)

fig = result["figure"]   # plotly.graph_objects.Figure
fig.show()               # opens interactive chart in browser / Jupyter
```

The two-panel figure shows power vs. n (top) and I-criterion + D-efficiency (bottom). Hover over any point for exact values; zoom and pan with the toolbar; export a PNG with the camera icon.

### Sensitivity analysis — interactive sweep

```python
from iopt_power_design import power_sensitivity, i_optimal_powered_design

result = i_optimal_powered_design(formula, factors, power_cfg, opts)

sens = power_sensitivity(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_df=result["design_df"],
    sigma_range=(0.5, 2.0),
    sigma_points=30,
    plot=True,
    plot_backend="plotly",
)

fig = sens["figure"]   # single-panel: power vs. σ with reference lines
fig.show()
```

### Note on top-level wrappers

`iopt_power_design.power_curve_by_n` and `iopt_power_design.power_curve_by_effect` are backward-compat wrappers that return a DataFrame only and discard the figure.  To access the Plotly figure, call the implementation modules directly:

```python
from iopt_power_design.power_curves import power_curve_by_n, power_curve_by_effect, power_surface_2d
```

For Streamlit, pass the figure directly to `st.plotly_chart`:

```python
import streamlit as st
st.plotly_chart(result["figure"], use_container_width=True)
```
