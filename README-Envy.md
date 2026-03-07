# iopt-power-design

I‑optimal experimental designs **with power assurance** for linear models. This package helps you:
- Build an **I‑optimal** design for a Patsy formula over a mixed factor space (continuous + categorical).
- **Search for the minimum sample size `n`** that meets a target statistical power (contrast-based or global R²).
- **Allocate runs across test cells** (the selected rows of the candidate set), and summarize allocation with bucket counts.
- Optionally **export diagnostics** (conditioning, D‑efficiency, leverage, I‑criterion over the candidate region, etc.).
- Explore **power curves** (by `n` or by effect size).

> I‑optimality minimizes the **average prediction variance** over the design (or candidate) region. It is often preferred when precise prediction across the factor space matters, not just precise estimation of coefficients (D‑optimality).

---

## Table of Contents
- [Installation](#installation)
- [Quick Start (Python API)](#quick-start-python-api)
- [Quick Start (CLI)](#quick-start-cli)
- [Concepts & Inputs](#concepts--inputs)
- [Outputs](#outputs)
- [Power Modes](#power-modes)
- [Candidate Set & Algorithms](#candidate-set--algorithms)
- [Diagnostics](#diagnostics)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Installation

Python ≥ 3.9 is required.

```bash
# From source (editable)
pip install -e .

# Or build & install
python -m build
pip install dist/iopt_power_design-*.whl
```

### Dependencies

The package relies on:
- **NumPy, SciPy, pandas, patsy**
- **pyDOE3** (for Fedorov/coordinate‑exchange search). We support both modern and legacy APIs:
  - `doe_optimal.build_optimal_design(...)`
  - `doe_optimal.optimal_design(...)`
- **matplotlib** (only if you export plots in diagnostics)

If `pyDOE3` is missing, I‑optimal search will fail early with a clear import error.

---

## Quick Start (Python API)

```python
from iopt_power_design import (
    i_optimal_powered_design,
    PowerContrastConfig, PowerR2Config,
    DesignOptions,
)

# 1) Model and factor space
formula = "~ 1 + A + B + A:B"     # Patsy formula
factors = {
    "A": ["low", "high"],         # categorical levels
    "B": (0.0, 10.0),             # continuous range [low, high]
}

# 2) Choose *one* power mode

# (a) Contrast-based power
pcfg = PowerContrastConfig(
    L=[[0, 0, 1, 0]],             # contrast on B main effect (example; must match p)
    delta=[0.5],                  # effect size for the contrast(s)
    alpha=0.05,
    power=0.80,
    sigma=1.0,
    max_n=500
)

# (b) Global R² power (full-model F-test)
# rcfg = PowerR2Config(
#     r2_target=0.10,             # target R² to detect
#     alpha=0.05,
#     power=0.80,
#     max_n=500,
#     lambda_mode="noncentral"    # or "approx"
# )

# 3) Design options
opts = DesignOptions(
    auto_candidate=True,          # adaptive candidate sizing
    candidate_points=4000,        # if not auto, use fixed size
    starts=8,                     # multi-start
    workers=None,                 # set >1 to parallelize starts (see OS note below)
    allow_candidate_growth=True,  # one-time candidate growth if conditioning is poor
    xtx_jitter=1e-10,             # small Tikhonov ridge for stability
    algo="fedorov",               # "fedorov" or "coordinate"
    seed=2025
)

# 4) Build an I-optimal design with power assurance
result = i_optimal_powered_design(
    formula=formula,
    factors=factors,
    power_cfg=pcfg,
    design_opts=opts,
    export_diagnostics_to=None   # or a folder path for reports
)

design_df  = result["design_df"]   # the n selected rows from the candidate set
buckets_df = result["buckets_df"]  # allocation: counts of identical rows (test cells)
report     = result["report"]      # narrative + metrics (power achieved, λ, df, etc.)

print(report["summary"])
print(buckets_df)
print(design_df.head())
```

**OS note for parallelism:** If you set `workers > 1` on macOS/Windows, call the API inside `if __name__ == "__main__":` (standard `multiprocessing` requirement).

---

## Quick Start (CLI)

The package registers a console script named **`iopt-design`**. Supply a YAML/JSON config file and an output directory.

```bash
iopt-design --config ./config.yml --out ./design_out -v
# dry-run (builds candidate, checks formula/inputs, reports p and initial n)
iopt-design --config ./config.yml --dry-run
```

**Minimal YAML example (contrast mode):**
```yaml
# config.yml
formula: "~ 1 + A + B + A:B"

factors:
  A: [low, high]
  B: [0.0, 10.0]        # [low, high] for continuous

contrast:
  # You can specify L, delta directly, *or* define two scenarios to generate L, delta.
  L: [[0, 0, 1, 0]]
  delta: [0.5]

alpha: 0.05
power: 0.8
sigma: 1.0
max_n: 500

design:
  auto_candidate: true
  candidate_points: 4000
  starts: 8
  workers: 1
  allow_candidate_growth: true
  algo: fedorov
  seed: 2025

diagnostics:
  export: false         # or path like "./design_out"
  plots: false
  tables: true
```

**Minimal YAML example (global R² mode):**
```yaml
formula: "~ 1 + A + B + A:B"
factors:
  A: [low, high]
  B: [0.0, 10.0]

r2_target: 0.10
alpha: 0.05
power: 0.8
max_n: 500

design:
  auto_candidate: true
  starts: 8
```

> Run `iopt-design --help` for all CLI switches. The CLI validates and maps your config into the same API used in Python.

---

## Concepts & Inputs

### Model formula
We use **Patsy** to parse a standard R‑style formula. Example: `~ 1 + A + B + A:B`. The code compiles a small sample to determine the number of regressors **p** and checks that `max_n > p` before search.

### Factors
Provide a dict:
- **Continuous factors**: `(low, high)` (tuple) or two‑element list.
- **Categorical factors**: list of levels (strings, ints, etc.).

Internally, the **candidate set** is generated from this dict. For continuous factors, we use level grids (adaptive or fixed). For categoricals, we enumerate levels and (optionally) cap the number of unique cells when adaptive sizing is enabled.

### Power targets
Choose **one** mode:
- **Contrast** (`PowerContrastConfig`): supply contrast matrix **L** (q×p) and effect vector **δ** (q) along with `alpha`, desired `power`, noise scale `sigma`, and `max_n`.
- **Global R²** (`PowerR2Config`): supply `r2_target`, `alpha`, desired `power`, `max_n`, and `lambda_mode` (how λ is mapped from R²).

---

## Outputs

`i_optimal_powered_design(...)` returns a dict with:
- **`design_df`** (`DataFrame`): the **n-run I‑optimal design**. Each row is a selected candidate. Duplicate rows imply replication.
- **`buckets_df`** (`DataFrame`): **allocation by unique test cell** (counts of identical rows). This is your explicit run allocation.
- **`report`** (`dict`): details of the search (per‑iteration power, `n`, λ, df, chosen start, conditioning, any candidate growth, seed, timing) and final **achieved power**.

When you use the CLI, these are written to CSV/JSON (and optional plots) in the output directory.

---

## Power Modes

### Contrast-based (Wald / F test on \(Lβ = δ\))
- Computes noncentrality parameter **λ** from the design matrix **X**, contrasts **L**, and effect sizes **δ** using a stable pseudo‑inverse with small ridge (`xtx_jitter`).
- Uses the noncentral F distribution with numerator/denominator degrees of freedom to compute power.
- Supports multiple contrasts jointly (q > 1). Shapes are validated with helpful errors.

### Global R² (full-model F test)
- Maps a requested **R²** to a λ consistent with the model and sample size, then computes power for the omnibus F test.
- Two modes for λ derivation are available via `lambda_mode`.

---

## Candidate Set & Algorithms

### Candidate sizing
- **Fixed**: set `candidate_points` directly.
- **Adaptive** (`auto_candidate=True`): sizes the candidate based on
  - number/type of factors (continuous grids vs categorical cells),
  - optional caps per categorical cell,
  - and total parameter count **p** from the compiled formula.

If early diagnostics show poor conditioning, the code can **grow the candidate set once** if `allow_candidate_growth=True` (controlled by `growth_factor`).

### I‑optimal search
- Uses **pyDOE3** (Fedorov/coordinate‑exchange). We support both API variants automatically.
- **Multi‑start**: set `starts`. Each start runs independently; the best design by the **I‑criterion** (average prediction variance over the *candidate region*) is selected.
- **Parallel starts**: set `workers > 1` to dispatch starts across processes (see OS note above).

---

## Diagnostics

Diagnostics (if enabled) compute common design‑quality metrics and can export:
- **Tables**: condition number, D‑efficiency, leverage stats, I‑criterion over the candidate region, VIFs, etc.
- **Plots**: residualized leverage vs leverage, Cook’s‑like influence, leverage histograms, etc.

Set `export_diagnostics_to="path/"` in the API, or configure `diagnostics.export` in the CLI to write HTML/CSV/PNG artifacts alongside the design CSVs.

---

## Reproducibility

- Use `DesignOptions.seed` to fix randomness in candidate grids and multi‑start choices.
- The report includes the chosen seed and, when applicable, any one‑time candidate growth and best‑start index.

---

## Troubleshooting

- **Import error for pyDOE3**: install `pyDOE3` (ensuring you have the `doe_optimal` submodule). We try both `build_optimal_design(...)` and `optimal_design(...)` at runtime.
- **`max_n` must be > p**: increase `max_n` or simplify the model. The code checks `p` by compiling a tiny sample through Patsy first.
- **Parallelism on macOS/Windows**: guard calls with `if __name__ == "__main__":` when `workers > 1`.
- **Rank/conditioning issues**: enable `allow_candidate_growth`, increase `candidate_points`, or add a small `xtx_jitter`.
- **Contrast shape errors**: ensure `L` is q×p and `delta` has length q **after** Patsy encodes the model (including intercepts and interaction columns).

---

## License

MIT
