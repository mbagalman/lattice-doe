# User Guide — iopt-power-design

> **How this guide relates to other docs**
>
> - **[Quick Start Guide](quickstart.md)** — get a working design in 10 minutes.
> - **[Recipes](recipes.md)** — copy-paste task-oriented snippets.
> - **[README](../README.md)** — concise feature reference with configuration tables.
>
> This guide is different. It explains *why* things work the way they do, walks through realistic examples from first principles, and covers every interface the package provides. It assumes no prior knowledge of optimal experimental design, but does assume you are comfortable writing Python.

---

## Part I — Foundations

### Chapter 1 — What this package does and why it matters

#### 1.1 The problem: choosing where to run experiments

Suppose you are studying how three process variables — temperature, pressure, and reaction time — affect the yield of a chemical synthesis. Each variable has a range of plausible values you could set it to. In principle, the complete "design space" is a three-dimensional box of infinitely many possible experimental conditions. You cannot run all of them. You can afford, say, 20 runs. Which 20 do you choose?

This is the core problem of experimental design, and the answer is not obvious. Spreading runs evenly across a grid sounds reasonable, but a full three-factor grid with even five levels per factor would require 125 runs. Choosing randomly feels safe, but random selections frequently cluster in some regions while leaving others sparse, producing designs that are worse than deliberate choices. Running at the factor extremes and centre — a common engineering heuristic — turns out to be optimal for some models but poor for others.

**What makes a design "good" depends on what you plan to do with the data.** If you plan to fit a regression model and use it to predict yield at arbitrary process conditions, a good design places points where they minimise prediction uncertainty across the whole region. If you plan to estimate model coefficients and test specific hypotheses about them, a good design places points where they maximise estimation precision. These two goals are related but not identical, and they lead to different optimal designs.

This package automates the search for designs that are simultaneously:

1. **Statistically optimal** — the points are chosen to minimise prediction variance, maximise coefficient precision, or balance the two, depending on which criterion you select.
2. **Power-assured** — the number of runs is chosen to guarantee that your planned hypothesis test has at least the statistical power you specify.

Most design software addresses optimality and power separately. This package treats them as a joint problem: it searches for the minimum number of runs `n` that achieves your power target, and at each `n` it selects the statistically optimal arrangement of those runs.

---

#### 1.2 Optimality criteria — what they measure and why it matters

Every design optimality criterion is built on the same foundation: the **model matrix** X and the **information matrix** X'X.

When you fit a linear regression model to your data, the matrix X is the *n* × *p* array in which each row encodes one experimental run (with the factor settings expressed in the coding your formula specifies), and each column corresponds to one model term — an intercept, a main effect, or an interaction. The information matrix M = X'X summarises how much statistical information your design contains. It controls both the precision of coefficient estimates (through (X'X)⁻¹) and the variance of predictions at any point in the design space.

The three optimality criteria measure different aspects of M.

**I-optimality** (also called *integrated* or *average prediction variance* optimality) minimises the average variance of model predictions across the entire design region:

```
I-criterion = (1/|R|) ∫_R Var[ŷ(x)] dx
            = (1/|R|) ∫_R f(x)ᵀ (X'X)⁻¹ f(x) dx
```

where f(x) is the vector of model-term values at point x and R is the design region. Geometrically, an I-optimal design spreads its points to keep the prediction surface uniformly accurate everywhere, not just at the observed points. **Use I-optimality when your model will be used for prediction — for example, when you want to map the response surface and identify process conditions that achieve a target yield.**

**D-optimality** maximises the determinant of the information matrix:

```
D-criterion = det(X'X)
```

Maximising `det(X'X)` is equivalent to minimising the volume of the joint confidence ellipsoid for all model coefficients simultaneously. A D-optimal design packs the most statistical information about the coefficients into the fewest runs. **Use D-optimality when precise estimation of individual model coefficients is the primary goal — for example, in confirmatory experiments where you need tight standard errors on specific effects.**

**A-optimality** minimises the trace of the inverse information matrix:

```
A-criterion = trace((X'X)⁻¹) = Σᵢ Var[β̂ᵢ]
```

This is the sum of the variances of all coefficient estimates. Where D-optimality minimises the *joint* volume of uncertainty, A-optimality minimises the *total* variance summed across coefficients. **Use A-optimality when you want balanced estimation precision across all model terms, with no single coefficient estimate dominating the uncertainty budget.**

**How often do the criteria disagree in practice?** Less often than you might expect. For continuous factors in an unconstrained box, I-, D-, and A-optimal designs at the same `n` are often nearly identical, or differ by only a few run placements. Meaningful differences emerge when:

- The model includes many categorical factors with multiple levels (the number of encoding columns can be asymmetric)
- The design space is constrained by feasibility constraints that rule out part of the box
- The model is strongly nonlinear in the factors (e.g. a quadratic or interaction-heavy formula)

The package's `compare_criteria` function runs all three criteria for your specific formula, factors, and power target and returns a side-by-side summary — `n`, achieved power, I-criterion value, and D-efficiency — so you can see the practical tradeoff before committing to a design. This is covered in detail in Chapter 7.

---

#### 1.3 Power assurance — what it means to "guarantee" power

**Statistical power** is the probability of correctly rejecting the null hypothesis when the null hypothesis is false and the true effect is at least as large as your minimum effect of interest. A design with 80% power at effect size δ will detect an effect of that size (or larger) in 80% of replications of the experiment, at significance level α, given the assumed residual standard deviation σ.

Power is not a property of the analysis alone. It depends on the design through the **noncentrality parameter** λ, which for a linear contrast test takes the form:

```
λ = δᵀ [L (X'X)⁻¹ Lᵀ]⁺ δ / σ²
```

Here L is the contrast matrix defining your hypothesis, δ is the vector of minimum detectable effects, and (X'X)⁻¹ is determined entirely by your design X. A larger information matrix (better design) produces a larger λ, which translates directly into higher power. The same optimality criterion that governs prediction quality or coefficient precision also governs detectability.

**This coupling is what motivates the package's design.** Choosing `n` by power calculations alone (as a standalone power analysis tool would) and then designing independently (as a standalone optimal design tool would) ignores the feedback: a more efficient design achieves the same power with fewer runs, and the power you actually achieve depends on the specific design chosen, not just its size.

This package performs both steps jointly through a three-level search:

1. **Outer loop — sample size search.** The package performs a binary search over `n` from a small starting value up to `max_n`. At each candidate `n`, it asks: can an optimal design of this size achieve the target power? The search finds the minimum `n` that answers yes.

2. **Middle loop — multi-start.** At each `n`, the Fedorov exchange is run from multiple random starting designs (controlled by `starts`). This reduces the risk of getting stuck in a poor local optimum, which the exchange algorithm can produce if started from an unlucky initial point.

3. **Inner loop — Fedorov exchange.** Given a starting design and `n` fixed, the Fedorov exchange iteratively swaps each current design point with the best available candidate point (according to the chosen criterion), continuing until no single swap improves the criterion. This is a well-established algorithm for discrete optimal design.

The result is a design that is both optimal (by the chosen criterion) *and* guaranteed to meet your power target at the minimum feasible sample size.

> **A note on "guarantee."** Power assurance is based on assumptions — about the residual standard deviation σ, the effect size δ, and the form of the model. If σ turns out to be larger than assumed, the achieved power will be lower. Chapter 20 covers sensitivity analysis tools that let you quantify how much power you retain if the σ assumption is off by 20%, 50%, or more, before you run a single experiment.

---

#### 1.4 The four power modes at a glance

The package supports four distinct ways of specifying what you want to detect. Each corresponds to a different statistical test and a different configuration class.

| Mode | Config class | Test | When to use |
|---|---|---|---|
| **Linear contrast** | `PowerContrastConfig` | F-test on Lβ = δ | You have a specific effect in mind — a main effect, interaction, or comparison between two scenarios — and you know roughly how large it needs to be to matter |
| **Global R²** | `PowerR2Config` | Omnibus F-test | You want to test whether the model as a whole explains meaningful variance; you don't have a specific contrast in mind |
| **GLM Wald χ²** | `PowerGLMContrastConfig` | Wald chi-square | Your response is binary (pass/fail, conversion, presence/absence) or a count (defects per unit, events per period) |
| **Multi-response** | `MultiResponseOptions` + `ResponseSpec` | Per-response + combined rule | You have two or more responses that must all be adequately powered, and you want the design to satisfy all of them simultaneously |

**Linear contrast mode** (Chapter 3) is the most commonly used and the most flexible. You specify a contrast matrix L that encodes exactly which linear combination of model coefficients you want to test, and δ gives the minimum effect you need to detect on that combination. If you are unsure how to construct L, the `contrast_from_scenarios` helper builds it automatically by comparing two sets of factor settings.

**Global R² mode** (Chapter 4) is appropriate when you cannot or do not want to specify a precise contrast but still want the experiment to be powered to detect a meaningful model fit. It uses Cohen's f² effect-size convention and aligns with the omnibus F-test reported by most regression software.

**GLM mode** (Chapter 5) handles the two most common non-Gaussian response types. For a binomial response (e.g. a product test that passes or fails), the effect is expressed as a difference in log-odds; for a Poisson response (e.g. defect counts), it is a difference in log-rates. The design search is structurally the same as for linear models, but the power calculation uses a Wald chi-square statistic and accounts for the response family through a Fisher weight.

**Multi-response mode** (Chapter 6) lets you specify several responses at once, each with its own formula, factors, and power mode. The design search maximises a combined power objective whose combination rule you control: `"min"` (the design is only as good as its weakest response), `"product"` (penalises any response that is under-powered), or `"weighted_mean"` (lets you assign business-priority weights across responses).

---

#### 1.5 The seven interfaces at a glance — a map for new users

The package provides seven ways to generate a design. They all call the same underlying algorithm; they differ in how you provide input and receive output.

| Interface | Best for | Requires Python? |
|---|---|---|
| Python API | Scripting, automation, custom post-processing | Yes |
| CLI | Reproducible file-based pipelines, CI/CD | For install only |
| Streamlit web UI | Interactive exploration, non-programmer collaborators | No |
| Excel | Teams with an Excel-first workflow | No |
| Google Sheets | Distributed teams, cloud-first organisations | No |
| Jupyter Widgets | Interactive exploration inside a notebook | Yes |
| REST API | Integration with non-Python systems or shared platforms | No |

**If you are new to the package** and comfortable with Python, start with the Python API (Chapter 8) for maximum transparency and control. Run the [Quick Start Guide](quickstart.md) first — it gets a working design in ten minutes and introduces the core objects.

**If you need a no-code interface**, the Streamlit app (Chapter 10) covers all major design types with a four-page UI that requires no programming. You can run it locally with `streamlit run app/app.py` or deploy it to Streamlit Community Cloud for free.

**If your team lives in spreadsheets**, the Excel interface (Chapter 11) and Google Sheets interface (Chapter 12) accept a filled-in template and write results back to the same file, so the entire workflow stays in the tool your collaborators already use.

**If you are embedding the package in a larger platform**, the REST API (Chapter 14) exposes all major functions as HTTP endpoints, making it straightforward to call from R, JavaScript, or any language with an HTTP client.

**Choosing between interfaces for the same task** is mostly a question of convenience and team workflow. All interfaces share the same configuration parameters; there is no capability penalty for using the spreadsheet interfaces over the Python API for the features they support. Appendix C has a full feature-by-interface comparison table.

> **Cross-reference:** The [README](../README.md) has concise parameter tables for every configuration class. The [Quick Start Guide](quickstart.md) covers the Python API and CLI from zero to a working design. The [Recipes](recipes.md) have copy-paste snippets for common tasks like criteria comparison, sensitivity analysis, augmentation, and split-plot designs. This guide builds on all three by explaining the reasoning behind the choices those documents ask you to make.

---

### Chapter 2 — Installation and project layout

#### 2.1 Python version requirements

The package requires **Python 3.9 or later**. It has been tested on Python 3.9, 3.10, 3.11, and 3.12. It runs on Linux, macOS, and Windows.

The four core runtime dependencies — `numpy`, `pandas`, `scipy`, and `patsy` — are installed automatically with the core package. All other dependencies are optional and installed only when you request a specific extras group (see section 2.3).

---

#### 2.2 Installing the core package

The package is installed from source using pip's editable install mode, which means Python reads the source files directly from the repository rather than copying them into your site-packages directory. This makes it straightforward to update by pulling new commits without reinstalling.

From the repository root:

```bash
pip install -e .
```

This installs the core package with its four required dependencies and registers two command-line entry points:

- `iopt-design` — the CLI for YAML-driven design generation (requires the `[cli]` extra for YAML parsing; see section 2.3)
- `iopt-api` — the REST API server entry point (requires the `[server]` extra)

If you are setting up a fresh environment, a virtual environment is strongly recommended to keep the package's dependencies isolated from your system Python:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -e .
```

---

#### 2.3 Optional extras and when you need them

The package uses pip extras groups to keep the core install lightweight. Install only the extras you actually need for your workflow. Each extra is independent; you can combine them in a single install command.

```bash
pip install -e ".[extra1,extra2,...]"
```

| Extra | What it adds | Install when |
|---|---|---|
| `cli` | `pyyaml` — YAML config parsing | You want to use `iopt-design --config config.yml` |
| `viz` | `matplotlib`, `seaborn`, `plotly` — power curve figures | You want to generate or display power curve plots |
| `app` | `streamlit`, `plotly`, `pyyaml` — the web UI | You want to run `streamlit run app/app.py` |
| `report` | `jinja2`, `pillow`, `kaleido` — HTML report generation | You want to call `generate_report(...)` to produce shareable HTML files |
| `report-pdf` | Everything in `[report]` plus `weasyprint` | You want PDF output from `generate_report(...)` |
| `extras` | `tqdm` (progress bars), `xlsxwriter`, `openpyxl` (Excel I/O) | You want progress bars during long runs, or you use the Excel interface |
| `sheets` | `gspread`, `google-auth` — Google Sheets client | You want to use `sheets_run(...)` or `create_sheet_template(...)` |
| `widgets` | `ipywidgets`, `plotly` — in-notebook interactive UI | You want to call `design_widget(...)` inside a Jupyter notebook |
| `server` | `fastapi`, `uvicorn`, `pydantic`, `httpx` — REST API | You want to run `iopt-api` to start the REST server |
| `all` | Everything above | You want every feature available |

**Common combinations:**

```bash
# Core Python API only (no extras needed for scripting)
pip install -e .

# Python API + plots + HTML reports
pip install -e ".[viz,report]"

# CLI-driven workflows
pip install -e ".[cli,extras]"

# Full Streamlit deployment
pip install -e ".[app,report,extras]"

# Jupyter notebook exploration
pip install -e ".[viz,widgets]"

# Google Sheets integration
pip install -e ".[sheets]"

# Everything
pip install -e ".[all]"
```

> **PDF export note.** The `weasyprint` library in `[report-pdf]` requires system-level libraries (cairo and pango) that are not installed by pip. On Ubuntu/Debian: `sudo apt-get install libcairo2 libpango-1.0-0 libpangocairo-1.0-0`. On macOS with Homebrew: `brew install cairo pango`. On Windows, see the WeasyPrint documentation. If you only need shareable output, the HTML format from `[report]` requires no system dependencies and can be opened in any browser.

---

#### 2.4 Verifying the install

After installing, run a one-line smoke test to confirm the core package and its dependencies are working:

```bash
python -c "import iopt_power_design; print(iopt_power_design.__version__)"
```

This should print the current version string (e.g. `0.1.0`) without errors. If it fails, the most common causes are a missing dependency or a Python version below 3.9.

To confirm the CLI is registered:

```bash
iopt-design --help
```

You should see the help text listing `--config`, `--template`, `--out`, `--dry-run`, and related flags. If `iopt-design: command not found` is returned, your virtual environment's `bin/` directory may not be on `PATH` — activate the environment and try again.

---

#### 2.5 Project layout

Understanding where things live helps when you want to inspect or extend the code, run the tests, or look up a function's implementation.

```
iopt_power_design/        # core Python package — importable as `iopt_power_design`
│
├── __init__.py           # public API surface: re-exports everything in __all__
├── config.py             # dataclasses: PowerContrastConfig, PowerR2Config,
│                         #   PowerGLMContrastConfig, DesignOptions, SplitPlotOptions,
│                         #   ResponseSpec, MultiResponseOptions
├── api.py                # primary entry points: i_optimal_powered_design,
│                         #   i_optimal_multiresponse_design
├── analysis.py           # analytical utilities: power_curve_by_n, power_curve_by_effect,
│                         #   power_sensitivity, min_detectable_effect, compare_criteria,
│                         #   robustness_report, multiresponse_sensitivity, ...
├── power.py              # per-mode power functions: contrast_power_sp, glm_contrast_power,
│                         #   global_r2_power_sp, eval_response_power, combine_powers, ...
├── power_curves.py       # power curve implementations + power_surface_2d
├── iopt_search.py        # Fedorov exchange engine, multi-start orchestration, augment_design
├── candidate.py          # candidate set construction: build_candidate, build_split_plot_candidate
├── model_matrix.py       # Patsy wrapper: build_model_matrix
├── allocation.py         # i_optimal_allocation
├── contrasts.py          # contrast_from_scenarios
├── split_plot.py         # GLS information matrix, whole-plot covariance utilities
├── blocked.py            # blocked design utilities: balanced_block_sizes, build_blocked_design
├── _request_builder.py   # internal shared config builder (not part of public API)
│
├── cli.py                # iopt-design command-line tool
├── sheets.py             # Google Sheets interface: sheets_run, create_sheet_template
├── excel_template.py     # Excel interface: excel_run, create_excel_template
├── widgets.py            # Jupyter widgets UI: design_widget, DesignWidget
├── report.py             # HTML/PDF report generation: generate_report
│
├── diag_metrics.py       # diagnostics: pure-NumPy metrics
├── diag_plots.py         # diagnostics: matplotlib figures
├── diag_export.py        # diagnostics: file export utilities
├── diagnostics.py        # backward-compat re-export wrapper for diag_* modules
├── design.py             # backward-compat re-export wrapper (split into candidate/iopt_search)
│
└── plot_backends.py      # matplotlib / plotly figure helpers

app/                      # Streamlit web application
├── app.py                # entry point: `streamlit run app/app.py`
├── state.py              # shared session-state helpers
├── components/           # reusable UI components (factor table, power params, charts)
└── pages/
    ├── 1_Factors.py      # Page 1: factor definition
    ├── 2_Power_Config.py # Page 2: power mode and parameters
    ├── 3_Run_Results.py  # Page 3: run the design, view results, download
    └── 4_Analysis.py     # Page 4: power curves, sensitivity, MDE, criteria comparison

api_server/               # FastAPI REST API server
├── main.py               # app factory: `uvicorn api_server.main:create_app --factory`
├── serialization.py      # Pydantic request/response models
├── errors.py             # exception handlers
└── routers/
    ├── design.py         # POST /design, POST /multiresponse_design
    ├── power_curve.py    # POST /power_curve/by_n, POST /power_curve/by_effect
    ├── sensitivity.py    # POST /sensitivity, POST /mde
    ├── compare.py        # POST /compare_criteria
    └── augment.py        # POST /augment

docs/                     # documentation
├── quickstart.md         # 10-minute getting-started guide
├── recipes.md            # task-oriented code snippets
├── user-guide.md         # this document
└── planning/             # internal design and review notes

tests/                    # test suite (pytest)
```

**The public API surface is everything exported from `iopt_power_design/__init__.py`.** You should never need to import from any submodule directly for ordinary use. The one exception noted in the recipes is `from iopt_power_design.power_curves import power_curve_by_n` when you need access to the Plotly figure object — the top-level wrapper discards it.

The backward-compat wrappers (`design.py`, `diagnostics.py`) exist because those modules were previously monolithic and were split during a refactoring pass. They continue to work exactly as before; you do not need to update existing code that imports from them.

---

## Part II — Power Modes

*Each chapter in this part covers one power mode: the statistical concept, the configuration class, and a realistic end-to-end example. Examples build in complexity from chapter to chapter.*

---

### Chapter 3 — Linear contrasts: detecting a specific effect

**Running example:** A polymer chemistry lab is optimising a synthesis reaction. Two factors are under investigation: catalyst type (categorical: A or B) and reagent concentration (continuous: 0.0–2.0 mol/L). Yield (%) is the response. The team's goal is 80% power to detect an effect of concentration on yield — specifically, a slope of at least 0.5 yield units per mol/L, with a residual standard deviation of σ = 1.0.

---

#### 3.1 What a contrast is: L, δ, and the F-test

A **linear contrast** is a specific linear combination of model coefficients that you want to test. The test asks: is this combination equal to zero (null hypothesis), or does it differ from zero by at least δ (the alternative)?

To make this concrete, start from the model. Fitting the linear model

```
yield ~ Intercept + Catalyst[T.B] + Concentration + Catalyst[T.B]:Concentration
```

produces a vector of coefficient estimates β̂ with four entries:

| Index | Column name | Meaning |
|---|---|---|
| 0 | `Intercept` | expected yield when Catalyst=A, Concentration=0 |
| 1 | `Catalyst[T.B]` | extra yield when switching to Catalyst B (at Concentration=0) |
| 2 | `Concentration` | slope: yield change per mol/L (with Catalyst=A) |
| 3 | `Catalyst[T.B]:Concentration` | how much the Concentration slope differs for Catalyst B |

The **contrast matrix L** is a *q* × *p* matrix where each row selects one linear combination of these *p* coefficients. To test whether the Concentration main effect is non-zero, you write:

```
L = [[0, 0, 1, 0]]
```

Row 0 picks out β₂ (the Concentration coefficient) and ignores the rest. The corresponding **minimum detectable effect δ** is the smallest value of β₂ you care to detect: if δ = 0.5, you are asking the design to be powerful enough to detect a Concentration slope of at least 0.5 yield units per mol/L.

The test statistic is an F-statistic based on the noncentrality parameter:

```
λ = δᵀ [L (X'X)⁻¹ Lᵀ]⁺ δ / σ²
```

where X is the *n* × *p* model matrix assembled from your design, and σ is the residual standard deviation. A few things are worth noting:

- λ depends on the **design** through (X'X)⁻¹. A better design (larger, better-placed) gives a larger λ, which gives higher power.
- λ scales with 1/σ². If σ is larger than expected, λ drops and power falls. This is why sensitivity analysis (Chapter 20) is important.
- The test has `df_num` = rank(L) numerator degrees of freedom and `df_denom` = n − rank(X) denominator degrees of freedom. With one contrast row and a four-parameter model, `df_num` = 1 and `df_denom` = n − 4.

The package finds the minimum n such that the F-test at significance level α achieves at least the target power — evaluated at an I-optimal (or D- or A-optimal) design of size n.

---

#### 3.2 Setting up `PowerContrastConfig`

**Step 1: Count the model-matrix columns.**

This is the most common source of errors. You must construct L with exactly *p* columns, where *p* is the number of columns Patsy will generate for your formula and factors. Patsy's encoding depends on both the formula and the factor levels. The rules are:

- `~ 1` contributes one column (Intercept).
- A continuous factor contributes one column.
- A categorical factor with *k* levels contributes *k* − 1 dummy columns (reference level = first level alphabetically by default).
- An interaction `A:B` contributes one column per combination of dummy columns from A and B.

For the polymer chemistry example:

```
formula = "~ 1 + Catalyst + Concentration + Catalyst:Concentration"
factors  = {"Catalyst": ["A", "B"], "Concentration": (0.0, 2.0)}
```

Working through the rules:
- `1` → 1 column (Intercept)
- `Catalyst` with levels A, B → 1 dummy column (`Catalyst[T.B]`, since A is the reference)
- `Concentration` → 1 column
- `Catalyst:Concentration` → 1 column (`Catalyst[T.B]:Concentration`)
- **Total: p = 4 columns**, indexed 0 through 3.

If you are unsure, you can confirm the column names directly with Patsy:

```python
import patsy, pandas as pd

sample = pd.DataFrame({"Catalyst": ["A", "A", "B", "B"],
                        "Concentration": [0.0, 2.0, 0.0, 2.0]})
dm = patsy.dmatrix("~ 1 + Catalyst + Concentration + Catalyst:Concentration", sample)
print(dm.design_info.column_names)
# ['Intercept', 'Catalyst[T.B]', 'Concentration', 'Catalyst[T.B]:Concentration']
```

> **Important:** the sample DataFrame passed to `patsy.dmatrix` must include all factor levels; otherwise Patsy may drop dummy columns that have no variation, giving you a column count that differs from the design-generation run.

**Step 2: Construct L.**

With the column names confirmed, writing L is mechanical: place a 1 in the column you want to test, and 0 everywhere else.

```python
# Test the Concentration main effect (column index 2):
L = [[0, 0, 1, 0]]
```

L is always a list of lists (or a 2D array). Even if you have a single-row contrast, the outer list is required.

**Step 3: Choose δ.**

δ must be in the same units as the corresponding coefficient. Here, the Concentration coefficient has units of (yield units) / (mol/L), so `delta = [0.5]` means "detect a slope of 0.5 yield units per mol/L."

> **Common mistake: mismatching scales.** If your continuous factor spans a large range (e.g., Temperature from 150 to 250 °C), the corresponding coefficient has units of "yield per degree C," which is typically a small number. Setting δ to a round number like 1.0 may be asking to detect an enormous effect, making the required n unrealistically small. The `contrast_from_scenarios` approach in section 3.3 avoids this by working in terms of total effect over a defined scenario shift rather than in terms of the raw coefficient.

**Step 4: Set the remaining parameters.**

```python
from iopt_power_design import PowerContrastConfig

power_cfg = PowerContrastConfig(
    L=[[0, 0, 1, 0]],   # contrast matrix: test the Concentration coefficient
    delta=[0.5],         # minimum detectable slope: 0.5 yield units per mol/L
    alpha=0.05,          # significance level
    power=0.80,          # target power
    sigma=1.0,           # assumed residual standard deviation
    max_n=200,           # hard cap on the sample-size search
)
```

`max_n` is a safety cap. If the binary search reaches `max_n` without achieving the target power, the function returns the best design it found at `max_n` rather than raising an error — the `achieved_power` in the report will be below the target. Set `max_n` large enough that the search is unlikely to hit it; 200–500 is a reasonable default for most problems.

---

#### 3.3 `contrast_from_scenarios`: building L and δ from two experimental scenarios

Constructing L manually requires you to know the exact column order in the model matrix and to express δ in units of the raw coefficient. An alternative that sidesteps both requirements is `contrast_from_scenarios`, which builds L and δ by comparing the model-matrix row for two named factor settings.

The idea is simple: if scenario A and scenario B differ in factor values, the vector x_B − x_A encodes exactly which coefficients change and by how much when you move from A to B. That vector becomes L. The corresponding δ is the `sesoi` — the smallest total effect on the response scale that you care to detect at that scenario shift.

```python
from iopt_power_design.contrasts import contrast_from_scenarios

# Compare Catalyst B against Catalyst A, holding Concentration fixed at 1.0 mol/L.
# sesoi=0.5 means: detect a yield difference of at least 0.5 between the two catalysts.
L, delta = contrast_from_scenarios(
    formula="~ 1 + Catalyst + Concentration + Catalyst:Concentration",
    factors={"Catalyst": ["A", "B"], "Concentration": (0.0, 2.0)},
    scenario_a={"Catalyst": "A", "Concentration": 1.0},
    scenario_b={"Catalyst": "B", "Concentration": 1.0},
    sesoi=0.5,
)
# L = [[0., 1., 0., 1.]]  — the difference x_B - x_A at Concentration=1.0
# delta = [0.5]
```

This L = [[0, 1, 0, 1]] says: the total effect being tested is `β₁ + β₃ × 1.0`, which is the difference in predicted yield between Catalyst B and Catalyst A at Concentration = 1.0 mol/L. The `sesoi=0.5` means: power the design to detect a yield difference of 0.5 at that operating point.

**When to use scenarios vs. manual L:**

- Use **scenarios** when thinking about the effect is natural in terms of "what happens when I change these settings from here to there?" This is the right mental model for most practitioners and avoids the coefficient-scale confusion.
- Use **manual L** when you need precise control over the mathematical contrast — for example, when testing a specific coefficient regardless of operating point, or constructing multi-contrast joint tests.

> **A practical note on coverage.** `contrast_from_scenarios` builds the model matrix from a small candidate set plus your two scenario rows. For the built L to have the correct number of columns, the candidate set must include all categorical factor levels. This works automatically when your scenarios together span multiple levels of each categorical factor, or when the auto-generated candidate set is large enough to include all levels. If you see a `ValueError` about column count mismatch, use manual L construction instead.

---

#### 3.4 Running the design and reading the result

With `power_cfg` and `DesignOptions` in hand, the call is:

```python
from iopt_power_design import DesignOptions, i_optimal_powered_design

opts = DesignOptions(
    auto_candidate=True,  # recommended: adaptive candidate sizing
    starts=8,             # number of multi-start runs (more = less likely to hit a local optimum)
    random_state=42,      # integer seed for reproducibility
)

result = i_optimal_powered_design(
    formula="~ 1 + Catalyst + Concentration + Catalyst:Concentration",
    factors={"Catalyst": ["A", "B"], "Concentration": (0.0, 2.0)},
    power_cfg=power_cfg,
    design_opts=opts,
)
```

The return value is a dict with three keys.

**`result["design_df"]`** is a DataFrame with `n` rows, one per experimental run. Each row gives the factor settings for that run:

```
     Concentration Catalyst
0         0.002416        B
1         0.003419        A
2         0.005392        A
...
68        1.997221        A
69        1.998796        B
```

For this problem the design has `n = 70` runs. You will notice that all runs are at concentrations very close to either 0.0 or 2.0 mol/L, with none in the middle range. This is not a coincidence: for a linear slope model, the I-optimal design maximises information about the slope by placing runs at the extreme ends of the range. A middle-of-the-range run contributes less information per run about the slope than an extreme-range run, so the exchange algorithm discards it.

**`result["buckets_df"]`** groups identical-or-near-identical run settings and shows replication counts. For continuous factors, floating-point values rarely match exactly even when the design intends repetition, so buckets often show count = 1 per row. The pattern in the factor values (concentrations clustered near 0 and near 2) is still clearly visible.

**`result["report"]`** is a dict of diagnostics:

```python
r = result["report"]
print(r["n"])                    # 70
print(r["achieved_power"])       # 0.8030
print(r["noncentrality_lambda"]) # 8.1453
print(r["df_num"])               # 1
print(r["df_denom"])             # 66
print(r["elapsed_sec"])          # wall time for the search
print(r["criterion"])            # "I"
print(r["random_state"])         # 42
```

Reading these in order:

- `n = 70`: the search found that 70 runs are needed to reach 80% power. Fewer runs at this σ and δ produce power below the target.
- `achieved_power = 0.8030`: the actual power at the returned design, which is slightly above the 0.80 target due to the binary-search step size.
- `noncentrality_lambda = 8.1453`: the value of λ at the returned design. You can use this to understand how close you are to the power boundary — a design with λ roughly 7.9 achieves just under 80% for `df_num=1` and `df_denom=66` at α=0.05.
- `df_num = 1`, `df_denom = 66`: the degrees of freedom of the F-test. `df_num = rank(L) = 1` because L has one row. `df_denom = n − rank(X) = 70 − 4 = 66`.

---

#### 3.5 Multi-contrast tests: testing several effects jointly

L can have more than one row. A contrast matrix with *q* rows tests the joint hypothesis that *all q* contrasts are simultaneously zero (H₀: Lβ = 0). The test uses an F-statistic with `df_num = rank(L) = q` numerator degrees of freedom.

```python
# Test Concentration slope AND Catalyst main effect simultaneously (joint F-test)
power_cfg_joint = PowerContrastConfig(
    L=[[0, 0, 1, 0],   # row 0: Concentration coefficient
       [0, 1, 0, 0]],  # row 1: Catalyst[T.B] main effect
    delta=[0.5, 0.5],  # detect at least 0.5 in each direction
    alpha=0.05,
    power=0.80,
    sigma=1.0,
    max_n=200,
)
```

With `delta = [0.5, 0.5]`, the power calculation asks: for what n does the design achieve 80% power to jointly detect that *both* the Concentration slope and the Catalyst main effect are at least 0.5? This is a more demanding test than either single-row test alone, which is why the required n increases:

```python
result_joint = i_optimal_powered_design(formula, factors, power_cfg_joint, opts)
print(result_joint["report"]["n"])           # 88
print(result_joint["report"]["df_num"])      # 2
print(result_joint["report"]["df_denom"])    # 84
print(result_joint["report"]["achieved_power"])  # 0.803
```

**When is a joint test appropriate?** Use it when you need to conclude that *all* tested effects are non-negligible — for example, when a regulatory review requires simultaneous evidence for both a treatment effect and a covariate effect. For most exploratory studies where you are interested in each effect independently, separate single-row contrasts are easier to interpret.

---

#### 3.6 Full worked example

The following script is self-contained and runs the complete contrast-mode workflow from formula definition through result interpretation. It is the reference example used in later chapters on power curves (Chapter 19), sensitivity analysis (Chapter 20), and minimum detectable effect (Chapter 21).

```python
# chapter3_example.py
from iopt_power_design import (
    i_optimal_powered_design,
    PowerContrastConfig,
    DesignOptions,
)
from iopt_power_design.contrasts import contrast_from_scenarios

# ── 1. Define the model ────────────────────────────────────────────────────
formula = "~ 1 + Catalyst + Concentration + Catalyst:Concentration"
factors = {
    "Catalyst":     ["A", "B"],      # categorical: 2 levels → 1 dummy column
    "Concentration": (0.0, 2.0),     # continuous: mol/L
}
# Patsy model-matrix columns (p = 4):
#   0: Intercept
#   1: Catalyst[T.B]
#   2: Concentration
#   3: Catalyst[T.B]:Concentration

# ── 2. Specify what to detect ──────────────────────────────────────────────
# Goal: detect a Concentration slope of at least 0.5 yield units per mol/L.
# L selects the Concentration coefficient (column index 2).
power_cfg = PowerContrastConfig(
    L=[[0, 0, 1, 0]],  # test H₀: β_Concentration = 0
    delta=[0.5],        # minimum effect: 0.5 yield units per mol/L
    alpha=0.05,
    power=0.80,
    sigma=1.0,          # assumed residual standard deviation (yield units)
    max_n=200,
)

# ── 3. Set design search options ───────────────────────────────────────────
opts = DesignOptions(
    auto_candidate=True,   # adaptive candidate sizing (recommended)
    starts=8,              # multi-start count: more starts → lower risk of local optimum
    random_state=42,       # integer seed for reproducibility
)

# ── 4. Run the design search ───────────────────────────────────────────────
result = i_optimal_powered_design(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=opts,
)

# ── 5. Inspect the results ─────────────────────────────────────────────────
r = result["report"]
print(f"Minimum n: {r['n']}")
print(f"Achieved power: {r['achieved_power']:.4f}")
print(f"Noncentrality λ: {r['noncentrality_lambda']:.4f}")
print(f"F-test df: ({r['df_num']}, {r['df_denom']})")
print(f"Criterion: {r['criterion']}")
print(f"Search time: {r['elapsed_sec']:.1f}s")
print()
print("Design (first 5 runs):")
print(result["design_df"].head())
print()
print("Run allocation summary:")
print(result["buckets_df"].head(8))
print("  ...")
print(result["buckets_df"].tail(8))
```

**Expected output:**

```
Minimum n: 70
Achieved power: 0.8030
Noncentrality λ: 8.1453
F-test df: (1, 66)
Criterion: I
Search time: 3.2s

Design (first 5 runs):
   Concentration Catalyst
0       0.002416        B
1       0.003419        A
2       0.005392        A
3       0.007839        B
4       0.009636        A

Run allocation summary:
   Concentration Catalyst  count
0       0.002416        B      1
1       0.003419        A      1
2       0.005392        A      1
3       0.007839        B      1
4       0.009636        A      1
5       0.010438        B      1
6       0.013741        B      1
7       0.015441        A      1
  ...
62      1.986864        A      1
63      1.987920        B      1
64      1.988179        B      1
65      1.989635        A      1
66      1.992647        B      1
67      1.993081        A      1
68      1.997221        A      1
69      1.998796        B      1
```

**Interpreting the design.** All 70 runs sit at concentrations very close to 0.0 or 2.0 mol/L, split roughly evenly between Catalyst A and B. This is the I-optimal solution for detecting a linear slope: runs at the extreme ends of the range give the most information about the slope, so the exchange algorithm discards every middle-range candidate. In practice, you might round the run-table concentrations to 0.0 and 2.0 and recheck power — the achieved power would remain essentially unchanged because the design was already effectively a two-point layout in Concentration.

> **Cross-reference:** To visualise how power changes as a function of sample size for this design, see Chapter 19 (`power_curve_by_n`). To quantify the risk if σ = 1.0 was underestimated from pilot data, see Chapter 20 (`power_sensitivity`). To determine the smallest effect this fixed design can detect, see Chapter 21 (`min_detectable_effect`).

---

### Chapter 4 — Global R²: testing whether the model explains variance

**Running example:** A consumer research team is running a survey study with four continuous predictors and wants to know: is there enough signal in the data to conclude the model is non-trivial? They target R² ≥ 0.15.

- 4.1 What the omnibus F-test measures and when it is the right power target
  - Contrast mode vs. R² mode: when you have a specific effect in mind vs. when you don't
  - Cohen's f²: the effect-size measure underlying R² power calculations
- 4.2 Setting up `PowerR2Config`
  - `r2_target`: the minimum R² worth detecting
  - `lambda_mode`: `"n"` (matches G\*Power / statsmodels) vs. `"n_minus_p"` (more conservative)
  - How `df_num` is derived from the formula (intercept excluded per G\*Power convention)
- 4.3 D-optimality for coefficient estimation: why the criterion choice matters more here
  - When the goal is testing the full model rather than predicting at arbitrary points, D-optimal designs are worth considering
- 4.4 **Full worked example** (Python API, D-optimal, R² mode, consumer survey)
  - Side-by-side comparison of I-optimal and D-optimal designs at the same n and power target
  - Reading `compare_criteria` output to quantify the tradeoff

---

### Chapter 5 — GLM power: binary and count responses

**Running examples:**
- *Binomial*: An e-commerce team is A/B testing a redesigned checkout flow. The baseline conversion rate is 12%. They want 80% power to detect an absolute 3-percentage-point lift.
- *Poisson*: A manufacturing quality team is studying defect counts. The baseline defect rate is 2.4 per batch. They want to detect a 50% reduction driven by process temperature and dwell time.

- 5.1 Why ordinary linear power calculations are wrong for binary and count data
  - The link function, the linear predictor scale, and the response scale
  - Logit link (binomial): effects expressed as log-odds differences
  - Log link (Poisson): effects expressed as log-rate differences
- 5.2 The Fisher-weight approximation used in this package
  - Constant weight w evaluated at the null baseline: w = p₀(1 − p₀) for binomial, w = μ₀ for Poisson
  - Why w cancels from I/D/A criteria (so the design search is structurally identical to OLS)
  - When this approximation is accurate vs. when it degrades: large slopes, wide covariate ranges
- 5.3 Setting up `PowerGLMContrastConfig`
  - `baseline`: probability ∈ (0, 1) for binomial; expected count > 0 for Poisson
  - `family`: `"binomial"` or `"poisson"`
  - `link`: canonical link defaults (logit / log); when to override
  - Expressing δ on the linear-predictor scale (log-odds for binomial; log-rate for Poisson)
  - Using `contrast_from_scenarios` for GLM: what sesoi means on the link scale
- 5.4 **Full worked example — binomial** (Python API + CLI template, e-commerce checkout)
  - Defining scenarios at the response scale and translating to the link scale
  - Generating a starter YAML with `iopt-design --template glm-binomial`
  - Interpreting achieved power and n
- 5.5 **Full worked example — Poisson** (Python API, manufacturing defect count)
  - Multi-factor setup with Poisson count response
  - Validating against a simple simulation (optional sidebar)

---

### Chapter 6 — Multi-response designs: powering several outcomes simultaneously

**Running example:** A chemical process engineer is optimising for three responses at once — yield (continuous, contrast mode), colour score (continuous, contrast mode), and particle size (continuous, R² mode). All three must be adequately powered before the run schedule is approved.

- 6.1 The multi-response problem: why joint power is harder than single-response power
  - Trade-offs: designs that are efficient for one response may be poor for another
  - The design that gets selected is the one that satisfies all responses simultaneously
- 6.2 Combination rules: how per-response power scores are folded into one objective
  - `"min"`: the pessimistic rule — overall power equals the weakest response
  - `"product"`: geometric combination — sensitive to all responses simultaneously
  - `"weighted_mean"`: assign relative importance weights per response; useful when responses have unequal business priority
  - Guidance on which rule to choose
- 6.3 `ResponseSpec` and `MultiResponseOptions`
  - Defining each response: name, formula, factors, power_cfg, weight
  - Setting `sigma_joint` for correlated responses (optional)
  - The `power_combination` parameter
- 6.4 Running `i_optimal_multiresponse_design` and reading the result
  - Output structure: `design`, `buckets`, `responses`, summary fields
  - Per-response power in `result["responses"]`
- 6.5 **Full worked example** (Python API, three-response chemical process)
  - Mixed power modes across responses (contrast + R²)
  - `power_curve_by_n_multiresponse` to visualise how each response's power grows with n
  - `multiresponse_sensitivity` to probe sensitivity to σ assumptions

---

## Part III — Optimality Criteria in Depth

### Chapter 7 — Choosing between I, D, and A

- 7.1 Mathematical definitions and geometric interpretations
  - I-optimality: integrated prediction variance over the design region
  - D-optimality: volume of the confidence ellipsoid in coefficient space
  - A-optimality: sum of coefficient-estimate variances
- 7.2 Practical guidance: when each criterion is the right choice
  - Use I when prediction quality across the factor space matters (response surface modelling, prediction-focused studies)
  - Use D when coefficient estimation precision is the primary goal (hypothesis-driven studies, mechanistic modelling)
  - Use A when you want balanced precision across all effects and interactions
  - How the three criteria behave differently for categorical-heavy designs vs. continuous designs
- 7.3 `compare_criteria` in practice
  - What the summary table contains: n, achieved_power, I-criterion, D-efficiency, elapsed time
  - Reading the power-efficiency tradeoff
- 7.4 **Full worked example** (Python API + Plotly figure, building on the Chapter 3 polymer example)
  - Running `compare_criteria` across all three criteria
  - Visualising the comparison with an interactive Plotly bar chart

---

## Part IV — The Interfaces

*Each chapter introduces one interface from basic setup through a complete example. Simple cases appear in earlier chapters; these chapters focus on interface-specific features and workflows.*

---

### Chapter 8 — Python API: full programmatic control

- 8.1 The primary entry points: `i_optimal_powered_design` and `i_optimal_multiresponse_design`
- 8.2 `DesignOptions` deep dive
  - Candidate set sizing: `auto_candidate`, `cand_min`, `cand_max`, and when to tune them
  - Multi-start and parallelism: `starts`, `workers`, `parallel_seed_stride`
  - Reproducibility: `random_state` (must be an int; `None` is not allowed)
  - Blocking: `n_blocks` and `block_factor_name`
  - Split-plot: `split_plot` (see Chapter 15 for full coverage)
  - Feasibility: `constraint_expr` and `constraint_func` (see Chapter 17 for full coverage)
- 8.3 The result dict: complete field reference
  - Single-response: `design_df`, `buckets_df`, `report`
  - Multi-response: `design`, `buckets`, `responses`, summary fields
- 8.4 Progress callbacks: monitoring long runs
- 8.5 Patterns for production scripts: logging, error handling, persisting results
- 8.6 **Full worked example** (end-to-end Python script, parallel multi-start, progress callback, auto-saved report)

---

### Chapter 9 — CLI: reproducible file-based pipelines

- 9.1 Installing CLI support: `pip install -e ".[cli]"`
- 9.2 Config file structure and all four YAML templates
  - `--template contrast` — linear contrast mode
  - `--template r2` — global R² mode
  - `--template glm-binomial` / `glm-poisson` — GLM modes
  - Annotated walkthrough of a full contrast YAML
- 9.3 Running a design: `iopt-design --config config.yml --out ./output/design`
- 9.4 Output files: `_design.csv`, `_buckets.csv`, `_report.json`, optional `_output.xlsx`
- 9.5 Dry-run validation: `--dry-run` for CI/CD pipelines
- 9.6 CLI flags reference: `--excel`, `--verbose`, `--template`, `--dry-run`
- 9.7 Defining factors in YAML: categorical lists, continuous `[low, high]`, and scenario-based contrasts
- 9.8 Feasibility constraints in YAML: `constraint_expr` (YAML-safe string)
- 9.9 **Full worked example** (reproducible pipeline: YAML config → CSV outputs → shell script)
  - Example YAML for a split-plot design (contrasting the Python API workflow)
  - Wiring the CLI into a Makefile or CI step

---

### Chapter 10 — Streamlit web UI: interactive design without coding

- 10.1 What the Streamlit app is and what it is not
  - Who it is for: domain experts, collaborators without Python access, rapid exploration
  - What it supports: contrast, R², GLM, multi-response, split-plot, sensitivity analysis, power curves, report export
- 10.2 Launching the app locally: `streamlit run app/app.py`
- 10.3 Page-by-page walkthrough
  - **Page 1 — Factors**: defining factor names, types (categorical / continuous), and levels
  - **Page 2 — Power Config**: selecting power mode, entering the contrast or R² target, setting α, power, σ
  - **Page 3 — Run & Results**: running the design, reading the design table and power summary, downloading CSV and the HTML report
  - **Page 4 — Analysis**: power curves (by n, by effect size), sensitivity sweeps, MDE, criteria comparison (contrast and R² modes; multi-response analysis and power-by-baseline curves require the Python API)
- 10.4 Deploying to Streamlit Community Cloud (free, no server required)
  - Step-by-step: push to GitHub → share.streamlit.io → deploy
  - No secrets or environment variables needed for the core app
- 10.5 Docker deployment
  - Building the image, running with `docker run -p 8501:8501`
  - Customising the Dockerfile for restricted environments
- 10.6 **Full worked example** (end-to-end Streamlit walkthrough with screenshots / annotated descriptions of each UI state)
  - Scenario: a non-programmer statistician designing a three-factor binomial study

---

### Chapter 11 — Excel: spreadsheet-driven workflows

- 11.1 When to use the Excel interface
  - Teams working in Excel-first environments
  - Sharing study configurations without sharing Python scripts
  - Capturing results in a structured, formatted workbook
- 11.2 Creating a template workbook: `create_excel_template`
  - The workbook contains one `Config` sheet with sentinel-delimited sections:
    - `[SETTINGS]` — key/value pairs: formula, power parameters, design options
    - `[CONTRAST]` — optional; required when `power_mode = contrast` or `glm`
    - `[FACTORS]` — factor table: Name | Type | Value 1 | Value 2 | …
    - `[RESPONSES]` — optional; required for multi-response mode
  - Output is written to separate sheets: `Results`, `Design`, `Buckets`
  - Sentinel values: blank cells, zero for absent numeric options
- 11.3 Running the design from a workbook: `excel_run`
  - What `excel_run` returns and what it writes back to the workbook
  - The Results and Buckets sheets written on completion
  - Error handling: `ExcelError` and how it surfaces to the user
- 11.4 **Full worked example** (Excel interface, contrast mode, three-factor industrial process)
  - Creating the template, filling it in, running `excel_run`, reading the output sheets
  - Comparison: the same design run via the Python API and via Excel

---

### Chapter 12 — Google Sheets: collaborative cloud-based workflows

- 12.1 When to use the Sheets interface
  - Distributed teams and stakeholder collaboration
  - Cloud-first organizations that avoid local file dependencies
  - Integration with Google Workspace (Docs, Data Studio, etc.)
- 12.2 Authentication: service account credentials and the `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- 12.3 Creating a template spreadsheet: `create_sheet_template`
  - How the sheet structure maps to the Excel template
  - Sharing the sheet and setting permissions
- 12.4 Running the design: `sheets_run`
  - Passing the spreadsheet URL or ID
  - What `sheets_run` writes back to the sheet
  - Error handling: `SheetsError` and how it surfaces
- 12.5 **Full worked example** (Google Sheets interface, R² mode, consumer research study)
  - Setting up credentials, creating the template, filling in the config, calling `sheets_run`
  - Reading results directly in the browser

---

### Chapter 13 — Jupyter Widgets: interactive in-notebook UI

- 13.1 When to use the widgets interface
  - Exploratory analysis in JupyterLab or VS Code notebooks
  - Teaching and demonstration contexts
  - Interactive power-curve exploration without leaving the notebook
- 13.2 Installing widget support: `pip install -e ".[widgets]"`
- 13.3 `design_widget` and `DesignWidget`
  - The `formula` and `factors` pre-fill parameters
  - Selecting `power_mode`: `"contrast"` or `"r2"` (GLM modes are not available in the widget; use the Python API for GLM designs)
  - The inline Plotly power curve (live-updates on factor or config changes)
  - Retrieving the result after running: `w.get_result()`
- 13.4 Embedding widget output in a report or notebook
- 13.5 **Full worked example** (Jupyter notebook, R² mode, consumer survey from Chapter 4)
  - Comparing widget-driven exploration vs. scripted API calls
  - Exporting the widget-selected design to CSV and HTML report

---

### Chapter 14 — REST API: programmatic access and microservice integration

- 14.1 When to use the REST API
  - Integrating with non-Python systems (R, JavaScript, Java, etc.)
  - Deploying as a shared microservice in a data science platform
  - Automating design generation from external scheduling tools
- 14.2 Starting the server: `uvicorn api_server.main:create_app --factory`
  - The `iopt-api` CLI entry point
  - Multi-worker deployment
- 14.3 Available endpoints
  - `POST /design` — generate a single-response optimal design
  - `POST /multiresponse_design` — generate a multi-response design
  - `POST /power_curve/by_n` — power curve over sample sizes
  - `POST /power_curve/by_effect` — power curve over effect sizes
  - `POST /sensitivity` — run a sensitivity sweep
  - `POST /mde` — compute minimum detectable effect
  - `POST /compare_criteria` — compare I, D, and A criteria
  - `POST /augment` — augment an existing design
- 14.4 Request/response schema overview
  - How Python dataclasses map to JSON in the API
  - Handling optional parameters and their defaults
- 14.5 **Full worked example** (curl / Python `httpx` client, contrast-mode design request)
  - Full JSON request body for a two-factor contrast design
  - Parsing the JSON response: design rows, buckets, report fields

---

## Part V — Advanced Design Features

### Chapter 15 — Split-plot designs: hard-to-change factors

- 15.1 The split-plot problem: why some factors are expensive to change
  - Whole-plot (HTC) factors: those reset between groups of runs — e.g., oven temperature, batch material, operator
  - Sub-plot (ETC) factors: those that vary freely within each group — e.g., reaction time, reagent concentration
  - The η parameter: the ratio of whole-plot variance to sub-plot variance
  - Why ordinary (CRD) designs are wrong for split-plot structures
- 15.2 The GLS information matrix for split-plot designs
  - How the covariance structure changes the effective information
  - Degrees of freedom: how the approximation method (`df_method`) affects power calculations
- 15.3 Setting up `SplitPlotOptions`
  - `htc_factors`: list of whole-plot factor names
  - `n_whole_plots`: how many WP groups to run
  - `subplots_per_wp`: sub-plot runs per whole-plot group
  - `eta`: the variance ratio assumption
  - `df_method`: `"auto"` (default), `"conservative"` (always use WP df), `"sp_only"` (always use SP df)
- 15.4 The whole-plot cost-power curve: `power_curve_by_wp`
  - Presenting the whole-plot count vs. power tradeoff to stakeholders
- 15.5 **Full worked example** (Python API + Plotly, industrial baking process)
  - Oven temperature and flour type as HTC factors; baking time and humidity as ETC factors
  - WP cost-power curve: minimum whole-plot resets to hit 80% power
  - CLI equivalent using a split-plot YAML config

---

### Chapter 16 — Blocked designs: accounting for nuisance variation

- 16.1 What blocking is and when it is necessary
  - Day-to-day variation, batch effects, operator differences, equipment lots
  - Block effects as nuisance parameters: estimated but not the focus of inference
  - The cost of blocking: (n_blocks − 1) denominator degrees of freedom lost
- 16.2 The `n_blocks` and `block_factor_name` parameters in `DesignOptions`
- 16.3 `blocked_formula`, `balanced_block_sizes`, and `build_blocked_design`
  - When these low-level utilities are useful vs. when `DesignOptions.n_blocks` is sufficient
- 16.4 **Full worked example** (Python API, clinical-style study across 4 operators)
  - Two-factor design, 4 blocks (one per operator), contrast-mode power
  - Showing the power cost from blocking vs. an unblocked design of the same n

---

### Chapter 17 — Feasibility constraints: excluding impossible factor combinations

- 17.1 When factor combinations are physically impossible or dangerous
  - High temperature + short time (fire risk), extreme pH combinations, equipment limits
- 17.2 String expression constraints: `constraint_expr`
  - Python-expression syntax evaluated per candidate row
  - YAML-safe: works in CLI configs, Sheets, and Excel templates
  - Example: `"not (Temperature > 200 and Time < 5)"`
- 17.3 Callable constraints: `constraint_func`
  - Accepts a DataFrame row and returns bool
  - For complex multi-factor logic that string expressions cannot capture
  - Note: Python callables only; not usable in YAML/Sheets/Excel configs
- 17.4 How constraints interact with candidate sizing
  - When a tight constraint reduces the effective candidate set: `allow_candidate_growth` and `growth_factor`
- 17.5 **Full worked example** (Python API, two-factor pharmaceutical synthesis)
  - A compound process constraint ruling out high-concentration / high-temperature combinations
  - Visualising the feasible vs. infeasible region

---

### Chapter 18 — Augmenting an existing design

- 18.1 When augmentation is appropriate
  - Budget unlocks additional runs after an initial study
  - Preliminary design lacks power; adding targeted points to fix it
  - Following up on a surprising early result
- 18.2 `augment_design`: the API
  - Inputs: existing `design_df`, `m` (new runs to add), formula, factors, `DesignOptions`
  - Outputs: `augmented_df` (full design), `new_runs_df` (added points only)
  - The new points are chosen to most improve the I/D/A criterion given the existing rows
- 18.3 Power after augmentation: re-running `i_optimal_powered_design` or evaluating directly
- 18.4 **Full worked example** (Python API, building on the Chapter 3 polymer example)
  - Initial under-powered design → augment by 4 runs → compare before/after power

---

## Part VI — Analysis, Visualisation, and Reports

### Chapter 19 — Power curves: visualising the design–power relationship

- 19.1 Why power curves matter: seeing the tradeoff, not just the answer
- 19.2 Power vs. sample size: `power_curve_by_n`
  - How the curve is computed: optimal design at each n, power evaluated
  - Interpreting the two-panel figure: power on top, I-criterion / D-efficiency on bottom
  - The `plot` and `plot_backend` parameters: `"matplotlib"` vs. `"plotly"`
  - Accessing the Plotly figure for Streamlit: `result["figure"]` and `st.plotly_chart`
- 19.3 Power vs. effect size: `power_curve_by_effect`
  - Using the curve to communicate MDE to non-statisticians
- 19.4 Power vs. baseline (GLM): `power_curve_by_baseline`
  - How power changes as the baseline probability or count rate varies
  - Critical for binomial designs where the baseline assumption is uncertain
- 19.5 The power surface: `power_surface_2d`
  - Two-dimensional heatmap of power as a function of (n, σ) or (n, δ)
  - Reading the surface: "safe" vs. "risky" parameter regions
- 19.6 Multi-response power curves: `power_curve_by_n_multiresponse`
  - Per-response traces on a single figure
  - Identifying the bottleneck response
- 19.7 **Full worked example** (Python API + Plotly, all five curve types from one design)

---

### Chapter 20 — Sensitivity analysis and robustness

- 20.1 Why assumptions about σ, δ, and the baseline are often wrong
  - Pilot estimates are noisy; the design that "just barely" meets power may fail in practice
  - Presenting a "power at risk" framing to stakeholders
- 20.2 `power_sensitivity`: sweeping σ or R²
  - `sigma_range` / `sigma_points` for contrast-mode designs
  - `r2_range` / `r2_points` for R²-mode designs
  - Interactive Plotly output with reference lines at the nominal assumption
- 20.3 `robustness_report`: structured sensitivity summary
  - Table of power at several σ or R² values
  - Identifying the "breakeven" point where power falls below the target
- 20.4 `multiresponse_sensitivity`: per-response sensitivity for multi-response designs
- 20.5 **Full worked example** (Python API + Plotly, polymer chemistry design from Chapter 3)
  - σ was estimated from a pilot study with n=12; quantifying the risk if σ is 30% higher than estimated

---

### Chapter 21 — Minimum detectable effect

- 21.1 The inverse question: given a fixed design, what can it detect?
  - Two uses: validating an inherited design, communicating design capability to reviewers
- 21.2 `min_detectable_effect`
  - The `target_power` parameter
  - What `mde["mde"]` means in contrast mode: a scale factor on δ (1.0 = the stated δ is just detectable)
  - Interpreting MDE in R² mode
- 21.3 **Full worked example** (Python API, inherited 24-run design from a previous study)
  - Establishing what the existing design can and cannot detect before proposing augmentation

---

### Chapter 22 — Shareable reports

- 22.1 What the report contains
  - Configuration summary: formula, factors, power_cfg
  - Power metrics: n, achieved power, noncentrality, degrees of freedom
  - Design table with run allocations
  - Embedded power-curve figure (base64-inline, no external dependencies)
- 22.2 Generating a report: `generate_report`
  - `output_path`: `.html` or `.pdf` extension controls the format
  - Installing extras: `[report]` for HTML; `[report-pdf]` for PDF
  - `export_report_to` shortcut: auto-generate the report at the end of `i_optimal_powered_design`
- 22.3 The self-contained HTML format
  - All CSS inline; no internet connection required; safe to email
  - Opening in any browser without Python installed
- 22.4 PDF output
  - WeasyPrint dependency and system requirements
  - When to use PDF vs. HTML
- 22.5 **Full worked example** (Python API, generating and sharing an HTML report)
  - Auto-report from `i_optimal_powered_design`, plus manual `generate_report` call with a custom path
  - Embedding the report link in a CLI pipeline

---

## Part VII — Reproducibility, Deployment, and Troubleshooting

### Chapter 23 — Reproducibility

- 23.1 How randomness enters the algorithm
  - Multi-start initialisation uses `random_state`
  - Parallel workers use per-worker seed offsets (`parallel_seed_stride`)
- 23.2 Achieving exact reproducibility
  - `random_state` must be an integer; `None` is not allowed
  - Keep `formula`, `factors`, `starts`, `workers`, and `random_state` fixed across re-runs
  - Store `result["report"]` alongside each output — it records the seed and timing metadata
- 23.3 Cross-machine reproducibility: NumPy version pinning
  - NumPy RNG output can change across major versions; pin `numpy` in `requirements.txt` / `pyproject.toml` for long-lived pipelines
- 23.4 Documenting a design run for regulatory or archival purposes
  - What fields from `report` to capture: `n`, `achieved_power`, `criterion`, `elapsed_sec`, `random_state`, `starts`, `workers`

---

### Chapter 24 — Deployment and scaling

- 24.1 Local deployment options summary
  - Script (Python API), CLI, Streamlit local, Jupyter Widgets
- 24.2 Cloud / server deployment options
  - Streamlit Community Cloud: free, no ops overhead, suitable for teams
  - Docker: portable, reproducible, suitable for IT-managed environments
  - REST API (FastAPI + Uvicorn): suitable for integration into larger platforms
- 24.3 REST API multi-worker deployment
  - `uvicorn api_server.main:create_app --factory --workers 4`
  - Stateless design: each request is independent, safe to distribute across workers
- 24.4 Resource considerations
  - CPU: multi-start parallelism with `workers > N` on a server; multiprocessing guard on macOS / Windows
  - Memory: large candidate sets and high `starts` counts; use `auto_candidate` to avoid over-provisioning
  - Typical run times: single-response contrast at modest n < 1 minute; split-plot or multi-response designs may take several minutes with high `starts`

---

### Chapter 25 — Troubleshooting

- 25.1 `ValueError: power_cfg.max_n must be greater than p`
  - Cause: `max_n` is too small for the formula complexity
  - Fix: increase `max_n` or simplify the formula
- 25.2 No convergence / low achieved power warnings
  - Cause: the search did not find an adequately powered design within `max_n`
  - Fixes: raise `max_n`; increase `starts`; enable `auto_candidate=True`
- 25.3 Parallel `workers > 1` fails on macOS / Windows
  - Cause: `multiprocessing` fork-safety requirement
  - Fix: put all `workers > 1` calls inside `if __name__ == "__main__":`
- 25.4 Contrast matrix shape mismatch errors
  - Cause: L has the wrong number of columns for the formula
  - Diagnosis: `build_model_matrix(formula, factors, pd.DataFrame([...]))` to inspect the column count
- 25.5 GLM baseline out of range
  - Cause: `baseline` ≥ 1 for binomial, or ≤ 0 for either family
  - Fix: pass a probability strictly between 0 and 1 for binomial; a positive count for Poisson
- 25.6 Sheets / Excel authentication or cell-parsing errors
  - Sheets: check `GOOGLE_APPLICATION_CREDENTIALS` environment variable and sheet sharing permissions
  - Excel: check file is not open in Excel; ensure `openpyxl` is installed (`[extras]`)
- 25.7 Report generation failures
  - Missing `[report]` or `[report-pdf]` extras
  - PDF: WeasyPrint system-level dependencies (cairo, pango) not installed

---

## Appendix A — Configuration quick reference

Summary tables for all configuration parameters: `PowerContrastConfig`, `PowerR2Config`, `PowerGLMContrastConfig`, `DesignOptions`, `SplitPlotOptions`, `ResponseSpec`, `MultiResponseOptions`. (Cross-reference to the full README tables.)

---

## Appendix B — Statistical background

- B.1 The Fedorov exchange algorithm: how designs are searched
- B.2 The noncentrality parameter for each power mode: derivation sketches
- B.3 The GLS information matrix for split-plot designs
- B.4 The Fisher-weight GLM approximation: assumptions and limitations
- B.5 References and further reading (Goos & Jones 2011; Atkinson, Donev & Tobias 2007; Cohen 1988; Lenth 2001)

---

## Appendix C — Interface comparison table

| Feature | Python API | CLI | Streamlit | Excel | Sheets | Widgets | REST API |
|---|---|---|---|---|---|---|---|
| Contrast mode | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| R² mode | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GLM mode | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Multi-response | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Split-plot | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Blocking | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Feasibility constraints | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Power curves | ✓ | — | ✓ | — | — | ✓ | ✓ |
| Sensitivity analysis | ✓ | — | ✓ | — | — | — | ✓ |
| HTML/PDF report | ✓ | ✓ | ✓ | — | — | — | — |
| No Python required | — | — | ✓ | ✓ | ✓ | — | ✓ |
| Collaborative/cloud | — | — | ✓ | — | ✓ | — | ✓ |
