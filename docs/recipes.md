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
