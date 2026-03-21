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

**Running example:** A consumer research team is designing a conjoint-style survey experiment. Respondents rate product preferences on a 0–10 scale. Four attributes are under study — Price, Quality, Convenience, and Brand perception — each scored on a standardised −1 to +1 range. The team's question is not "which specific attribute matters most?" but the more preliminary one: "does this combination of attributes explain a meaningful proportion of preference variance at all?" They target R² ≥ 0.15.

---

#### 4.1 The omnibus F-test and when R² mode is the right target

The **omnibus F-test** tests the null hypothesis that every slope in the model is simultaneously zero — equivalently, that R² = 0 and the predictors collectively explain nothing. Rejecting the null means concluding that the model as a whole explains a non-trivial proportion of response variance; it says nothing about which individual predictors drive that explanation.

**When is this the right power target?**

Use R² mode when:

- You are in an early exploratory phase and want to confirm there is signal worth pursuing before committing to more specific hypotheses.
- Your question is genuinely global: "does this set of factors influence the outcome?" rather than "does this specific factor influence the outcome by this amount?"
- You are replicating a prior study and want your design to have the same statistical power to detect the same overall effect size, without constraining yourself to a particular contrast.
- Regulatory or protocol requirements specify a minimum detectable R² rather than a specific effect size.

**When contrast mode is preferable:**

If you already have a theory about which effects matter, or if the eventual analysis will focus on testing individual coefficients or pre-specified comparisons, contrast mode (Chapter 3) is more appropriate. It lets you define exactly what you need to detect and will typically require fewer runs because it targets a narrower, more specific hypothesis. The omnibus F-test is a weaker test — it can be powerful for detecting large effects spread across many predictors while missing a single sharp effect that a contrast would catch easily.

**Cohen's f²: the effect-size measure for R² power.**

The omnibus F-test power depends on the noncentrality parameter λ, which is related to R² through Cohen's f² effect size:

```
f² = R² / (1 − R²)
```

For the target R² = 0.15, f² = 0.15 / 0.85 ≈ 0.176. Cohen (1988) classified f² ≈ 0.02 as small, 0.15 as medium, and 0.35 as large. An R² target of 0.15 therefore sits at the medium-to-large boundary — a model that explains 15% of preference variance is a meaningful but not overwhelming effect in consumer research.

The noncentrality parameter is then:

```
λ = f² × n          (lambda_mode = "n",   the default)
λ = f² × (n − p)    (lambda_mode = "n_minus_p", more conservative)
```

with `df_num` = p − 1 (number of slope parameters; intercept excluded) and `df_denom` = n − p.

---

#### 4.2 Setting up `PowerR2Config`

```python
from iopt_power_design import PowerR2Config

power_cfg = PowerR2Config(
    r2_target=0.15,      # minimum R² worth detecting
    alpha=0.05,          # significance level
    power=0.80,          # target power
    max_n=300,           # hard cap on the sample-size search
    lambda_mode="n",     # noncentrality convention (see below)
)
```

**`r2_target`** is the minimum proportion of variance that the model must explain for the effect to be practically meaningful. Setting this too low (e.g. 0.02) will demand a very large n for a very weak signal; setting it too high (e.g. 0.50) will underpower you against realistic medium effects. A useful heuristic: use the R² from a comparable published study as a lower bound.

**`lambda_mode`** controls how the noncentrality parameter is computed:

| `lambda_mode` | Formula | Matches |
|---|---|---|
| `"n"` (default) | λ = f² × n | G\*Power, statsmodels `FTestAnovaPower` |
| `"n_minus_p"` | λ = f² × (n − p) | More conservative; closer to the exact non-central F |

The difference is small for large n but meaningful when n is close to p. In the consumer survey example with p = 5 and the target R² = 0.15:

- `lambda_mode="n"` requires **n = 73** (λ = 12.88 at n = 73)
- `lambda_mode="n_minus_p"` requires **n = 78** (λ = 12.88 at n = 78)

The λ value is the same in both cases; the difference is that `"n_minus_p"` requires 5 more runs to achieve the same λ because it attributes the noncentrality to fewer effective observations. Use `"n_minus_p"` when you want to align with a more conservative reference, or when n is small relative to p.

**`df_num` and how it is derived.** You do not set `df_num` directly — the package derives it from the formula. Specifically, `df_num` = (number of model-matrix columns) − 1. The intercept is excluded from the numerator degrees of freedom, following the G\*Power convention for the omnibus F-test:

```
formula = "~ 1 + Price + Quality + Convenience + Brand"
# Model-matrix columns: [Intercept, Price, Quality, Convenience, Brand]  → p = 5
# df_num = p - 1 = 4
# df_denom = n - p = n - 5
```

For the returned design at n = 73: `df_num = 4`, `df_denom = 68`.

---

#### 4.3 Criterion choice in R² mode

For R² mode with continuous factors in a symmetric design space, the three optimality criteria (I, D, A) frequently produce the same required sample size. In the consumer survey example, running `compare_criteria` returns:

```python
from iopt_power_design import compare_criteria, DesignOptions

comparison = compare_criteria(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=DesignOptions(auto_candidate=True, starts=8, random_state=42),
)
print(comparison["summary"][["criterion", "n", "achieved_power", "elapsed_sec"]])
```

```
  criterion   n  achieved_power  elapsed_sec
0         I  73        0.803100        ...
1         D  73        0.803100        ...
2         A  73        0.803100        ...
```

All three criteria find the same n = 73 and the same achieved power. This is a common outcome for main-effects-only models with continuous factors on a symmetric box: the factor space has no asymmetry to exploit, and the omnibus F-test power is insensitive to the fine-grained placement differences between I-, D-, and A-optimal designs at a given n.

**When does the criterion choice matter for R² mode?**

Meaningful differences emerge when:

- The model includes categorical factors (the dummy encoding introduces asymmetry into the design space that the three criteria resolve differently).
- The model includes interactions or polynomial terms (the information matrix becomes less uniform across the design region).
- There are feasibility constraints that remove part of the design space.

In those situations, D-optimal designs are generally preferable for R² mode because the omnibus F-test is directly related to the information matrix determinant: det(X'X) governs the joint precision of all coefficient estimates simultaneously, which is exactly what the R² F-test measures. If you are unsure, run `compare_criteria` first — if the n values agree across criteria, the choice is immaterial for required sample size.

---

#### 4.4 Full worked example

The following script is self-contained and runs the complete R²-mode workflow for the consumer survey.

```python
# chapter4_example.py
from iopt_power_design import (
    i_optimal_powered_design,
    PowerR2Config,
    DesignOptions,
    compare_criteria,
)

# ── 1. Define the model ────────────────────────────────────────────────────
formula = "~ 1 + Price + Quality + Convenience + Brand"
factors = {
    "Price":       (-1.0, 1.0),   # standardised: -1 = lowest tier, +1 = highest tier
    "Quality":     (-1.0, 1.0),
    "Convenience": (-1.0, 1.0),
    "Brand":       (-1.0, 1.0),
}
# Patsy model-matrix columns (p = 5):
#   Intercept, Price, Quality, Convenience, Brand
# df_num = p - 1 = 4  (intercept excluded from omnibus F-test)

# ── 2. Specify the power target ────────────────────────────────────────────
# Goal: detect R² ≥ 0.15 (Cohen's f² ≈ 0.176 — medium-to-large effect).
# Using the default lambda_mode="n" to match G*Power conventions.
power_cfg = PowerR2Config(
    r2_target=0.15,
    alpha=0.05,
    power=0.80,
    max_n=300,
    lambda_mode="n",
)

# ── 3. Run the design search ───────────────────────────────────────────────
opts = DesignOptions(
    auto_candidate=True,
    starts=8,
    random_state=42,
    criterion="D",   # D-optimal preferred for omnibus F-test goals
)

result = i_optimal_powered_design(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=opts,
)

# ── 4. Inspect results ─────────────────────────────────────────────────────
r = result["report"]
print(f"Minimum n: {r['n']}")
print(f"Achieved power: {r['achieved_power']:.4f}")
print(f"Noncentrality λ: {r['noncentrality_lambda']:.4f}")
print(f"F-test df: ({r['df_num']}, {r['df_denom']})")
print(f"Cohen's f²: {0.15 / 0.85:.4f}")
print()
print("Design (first 8 runs):")
print(result["design_df"].head(8).round(3).to_string())

# ── 5. Compare criteria (optional but recommended) ─────────────────────────
comparison = compare_criteria(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=DesignOptions(auto_candidate=True, starts=8, random_state=42),
)
print()
print("Criteria comparison:")
print(comparison["summary"][["criterion", "n", "achieved_power"]].to_string())

# ── 6. Check the conservative lambda convention ────────────────────────────
power_cfg_conservative = PowerR2Config(
    r2_target=0.15, alpha=0.05, power=0.80, max_n=300, lambda_mode="n_minus_p"
)
result_cons = i_optimal_powered_design(formula, factors, power_cfg_conservative, opts)
print()
print(f"Conservative (n_minus_p) n: {result_cons['report']['n']}")
```

**Expected output:**

```
Minimum n: 73
Achieved power: 0.8031
Noncentrality λ: 12.8824
F-test df: (4, 68)
Cohen's f²: 0.1765

Design (first 8 runs):
   Price  Quality  Convenience  Brand
0 -0.765   -0.967        0.026 -0.762
1  0.940    0.414       -0.917  0.923
2 -0.575    0.950        0.848 -0.729
3 -0.011   -0.907       -0.807  0.883
4 -0.704    0.602       -0.751  0.979
5  0.703    0.946        0.183 -0.994
6 -0.720    0.710       -0.768  0.873
7  0.587   -0.915        0.466 -0.928

Criteria comparison:
  criterion   n  achieved_power
0         I  73        0.803100
1         D  73        0.803100
2         A  73        0.803100

Conservative (n_minus_p) n: 78
```

**Interpreting the design.** The 73 runs do not cluster at any obvious extreme values. Unlike the Chapter 3 concentration-slope design — where every run sat at either 0.0 or 2.0 mol/L — the R²-mode design scatters runs across the full four-dimensional factor space. This reflects the different structure of the two tests: a slope test (contrast mode) extracts the most information from extreme values of the tested factor, while an omnibus F-test needs to estimate all coefficients jointly, which requires broader coverage of the design region.

Each run combination is unique (no replication in the 73-row design). Replication would appear if the optimal design discovered that a few specific factor combinations are more informative than others — as happens in categorical designs — but with four independent continuous factors and a main-effects model, the I/D/A-optimal solution spreads runs roughly evenly.

**The lambda_mode decision in practice.** The five-run difference between `"n"` (n = 73) and `"n_minus_p"` (n = 78) is small here — about 7% more runs for the conservative convention. If your analysis will use a standard regression F-test as computed by R, Python `statsmodels`, or SAS `PROC REG`, use `"n"` for consistency with those tools. Use `"n_minus_p"` if your protocol calls for a more conservative pre-specification, or if n is small enough (< 3p) that the two conventions diverge noticeably.

> **Cross-reference:** If you want to visualise how power changes as a function of the assumed R² target (sensitivity to the r2_target assumption), use `power_sensitivity` with `r2_range` and `r2_points` as described in Chapter 20.

---

### Chapter 5 — GLM power: binary and count responses

**Running examples:**
- *Binomial*: A clinical team is running a dose-response study. Patient outcome is binary (responded / did not respond). The baseline response rate is 25%. They want 80% power to detect that the effect of dose achieves a 50% response rate.
- *Poisson*: A manufacturing quality team is studying defect counts per batch. The baseline defect rate is 0.8 per batch. They want to detect a 50% reduction driven by process temperature and dwell time.

---

#### 5.1 Why ordinary linear power calculations are wrong for binary and count data

Chapters 3 and 4 assumed that the response is normally distributed with constant variance σ². That assumption breaks down for two common response types:

**Binary responses** (pass/fail, responded/not, purchased/not) take only values 0 and 1. Their variance is not constant — it depends on the probability p: `Var(Y) = p(1 − p)`. A patient cohort with a 50% response rate is far more variable per observation than one with a 5% response rate. Using a Gaussian power formula with a guessed σ ignores this, and the resulting sample size will be wrong.

**Count responses** (defects per batch, events per hour, errors per session) follow a Poisson distribution whose variance equals its mean: `Var(Y) = μ`. A process with a high defect rate is intrinsically noisier than one with a low rate, and again the Gaussian σ² model is a poor approximation.

The right framework for both is a **generalised linear model (GLM)**, which:
1. Models the mean on a transformed (link function) scale where the effect is additive and can be large or small without constraint.
2. Uses the distribution-appropriate variance function rather than a constant σ².

**The link functions.** The GLM encodes effects on the **linear predictor** scale η, related to the mean μ by the link function g:

```
η = g(μ) = Xβ
```

| Family | Link | Scale | Coefficient meaning |
|---|---|---|---|
| Binomial | Logit | Log-odds | β₁ = change in log-odds per unit change in x |
| Poisson | Log | Log-rate | β₁ = change in log-rate per unit change in x |

The practical consequence: when you specify a minimum detectable effect for a GLM design, you must express it on the **link scale** (log-odds or log-rate), not on the response scale (probability or count). The sections below show how to make that translation.

---

#### 5.2 The Fisher-weight approximation

Computing a fully local D- or I-optimal GLM design in complete generality requires knowing the true parameter vector β before the experiment — which you do not. The common practical resolution is a **locally optimal design** evaluated at a nominal operating point.

This package uses the **Fisher-weight (constant-weight) approximation**: the information contributed by run *i* is weighted by the GLM variance evaluated at the null baseline, giving a scalar weight w applied uniformly to every design point:

```
w = p₀(1 − p₀)    for binomial (evaluated at null baseline probability p₀)
w = μ₀             for Poisson  (evaluated at null baseline count rate μ₀)
```

The scaled information matrix is then:

```
M = w · X'X
```

Because w is a positive scalar, it cancels from all three optimality criteria (I, D, A) — the criterion ratios are unchanged by a constant scaling of M. This means **the Fedorov exchange that optimises I, D, or A for a GLM design is structurally identical to the corresponding OLS exchange**. Only the power calculation changes: the noncentrality parameter for the Wald chi-square test becomes:

```
λ = w · δᵀ [L (X'X)⁻¹ Lᵀ]⁺ δ
```

**When is this approximation accurate?** The constant-weight approximation is reliable when:
- The factor effects (slopes β) are small relative to the baseline: the true per-point weight `wᵢ = p(xᵢ)(1 − p(xᵢ))` does not vary much across the design region.
- The baseline is not at an extreme of the response scale (far from 0 or 1 for binomial; not near 0 for Poisson).

It becomes less accurate when slopes are large, because the response surface curves strongly and the operating weights `wᵢ` vary substantially across runs. For such cases, the package's power estimates are approximate; validation by simulation is advisable before committing to a run schedule.

> **A note on the approximation scope.** This limitation is documented in the `PowerGLMContrastConfig` docstring and in the README. Chapter 5 of Goos & Jones (2011) discusses alternative approaches for cases where the approximation is unsatisfactory.

---

#### 5.3 Setting up `PowerGLMContrastConfig`

`PowerGLMContrastConfig` extends the contrast-mode framework of Chapter 3 with three additional parameters: `baseline`, `family`, and `link`.

```python
from iopt_power_design import PowerGLMContrastConfig

power_cfg = PowerGLMContrastConfig(
    L=[[0, 1, 0]],           # contrast matrix: same as PowerContrastConfig
    delta=[1.099],            # effect on the link scale (log-odds or log-rate)
    baseline=0.25,            # null operating point on the response scale
    family="binomial",        # "binomial" or "poisson"
    link=None,                # None → canonical link (logit for binomial, log for Poisson)
    alpha=0.05,
    power=0.80,
    max_n=300,
)
```

**`baseline`** is the null operating point expressed on the **response scale** — not the link scale. For binomial, it is a probability strictly between 0 and 1. For Poisson, it is a positive expected count. The Fisher weight is computed from this value internally.

**`family`** selects the response distribution. `"binomial"` uses the Bernoulli variance function and defaults to the logit link; `"poisson"` uses the Poisson variance function and defaults to the log link.

**`link`** defaults to `None`, which selects the canonical link for the family. You can override this with `"logit"` or `"log"` explicitly, but in practice the canonical link is almost always the right choice unless you have a strong domain reason to prefer a different linearisation.

**Translating from the response scale to the link scale.**

This is the step most likely to trip up a first-time GLM user. The minimum detectable effect `delta` must be expressed in link-scale units, not response-scale units.

For **binomial**, use the logit function:

```python
import numpy as np

def logit(p):
    return np.log(p / (1 - p))

p0 = 0.25   # baseline probability
p1 = 0.50   # target probability you want to detect

delta = logit(p1) - logit(p0)
# logit(0.50) = 0.0
# logit(0.25) = -1.099
# delta = 0.0 − (−1.099) = 1.099
print(f"delta (log-odds) = {delta:.4f}")
```

For **Poisson**, use the log function:

```python
mu0 = 0.8    # baseline defect rate (per batch)
mu1 = 0.4    # target rate after process improvement

delta = abs(np.log(mu1) - np.log(mu0))
# log(0.4) − log(0.8) = log(0.5) = −0.693
# delta = |−0.693| = 0.693
print(f"delta (log-rate) = {delta:.4f}")
```

The L matrix is constructed exactly as in Chapter 3 — by identifying which model-matrix column corresponds to the effect of interest and placing a 1 in that position.

> **Connecting response-scale intuition to the link scale.** A binomial delta of 1.099 log-odds units corresponds to a doubling of the odds: odds(0.50)/odds(0.25) = (0.5/0.5)/(0.25/0.75) = 1/0.333 = 3.0. A Poisson delta of 0.693 log-rate units corresponds to a halving of the rate: e^{−0.693} = 0.5. On the log scale, multiplicative changes on the response scale become additive changes on the linear predictor — this is the interpretive convenience the log link provides.

---

#### 5.4 Full worked example — binomial

**Context.** A clinical research team is running a dose-response study to characterise a new drug candidate. The binary endpoint is whether a patient shows a measurable therapeutic response within 48 hours. Historical data puts the placebo-equivalent response rate at 25%. The team wants 80% power to detect that a one-unit increase in normalised dose shifts the response probability from 25% to 50% — a clinically meaningful doubling of the odds.

Two factors are under study: normalised dose (continuous, −1 to +1) and patient age group (normalised, −1 to +1 representing young to elderly). The main-effects model is:

```
~ 1 + Dose + PatientAge
```

which gives p = 3 model-matrix columns: `[Intercept, Dose, PatientAge]`.

```python
# chapter5_binomial.py
import numpy as np
from iopt_power_design import (
    i_optimal_powered_design,
    PowerGLMContrastConfig,
    DesignOptions,
)

# ── 1. Define the model ────────────────────────────────────────────────────
formula = "~ 1 + Dose + PatientAge"
factors = {
    "Dose":       (-1.0, 1.0),   # normalised: -1 = lowest dose, +1 = highest dose
    "PatientAge": (-1.0, 1.0),   # normalised: -1 = youngest, +1 = oldest
}
# Patsy columns (p = 3): [Intercept, Dose, PatientAge]

# ── 2. Translate the effect to the log-odds scale ──────────────────────────
p0, p1 = 0.25, 0.50
logit  = lambda p: np.log(p / (1 - p))
delta  = logit(p1) - logit(p0)

print(f"logit({p0}) = {logit(p0):.4f}")   # −1.0986
print(f"logit({p1}) = {logit(p1):.4f}")   # 0.0000
print(f"delta (log-odds) = {delta:.4f}")   # 1.0986
print(f"Fisher weight w  = {p0*(1-p0):.4f}")  # 0.1875

# ── 3. Set up GLM power config ─────────────────────────────────────────────
power_cfg = PowerGLMContrastConfig(
    L=[[0, 1, 0]],   # test H₀: β_Dose = 0  (column index 1)
    delta=[delta],   # log-odds effect of 1.099 ≈ doubling of odds
    baseline=p0,     # null operating point on the probability scale
    family="binomial",
    link=None,       # canonical link: logit
    alpha=0.05,
    power=0.80,
    max_n=300,
)

# ── 4. Run the design search ───────────────────────────────────────────────
opts = DesignOptions(
    auto_candidate=True,
    starts=8,
    random_state=42,
)

result = i_optimal_powered_design(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=opts,
)

# ── 5. Inspect results ─────────────────────────────────────────────────────
r = result["report"]
print(f"\nMinimum n: {r['n']}")
print(f"Achieved power: {r['achieved_power']:.4f}")
print(f"Noncentrality λ: {r['noncentrality_lambda']:.4f}")
print(f"F-test df: ({r['df_num']}, {r['df_denom']})")
print()
print("Design (first 6 runs):")
print(result["design_df"].head(6).round(3).to_string())
```

**Expected output:**

```
logit(0.25) = -1.0986
logit(0.50) =  0.0000
delta (log-odds) = 1.0986
Fisher weight w  = 0.1875

Minimum n: 44
Achieved power: 0.7997
Noncentrality λ: 7.8425
F-test df: (1, 41)

Design (first 6 runs):
   Dose  PatientAge
0 -0.998      0.018
1 -0.991     -0.062
2 -0.985      0.114
3 -0.979     -0.030
4 -0.963      0.084
5 -0.957     -0.119
```

**Interpreting the results.**

The design requires **44 runs**. Compare this to a naive back-of-envelope calculation using a Gaussian formula: if you incorrectly plugged in σ = √(p₀(1−p₀)) = √0.1875 ≈ 0.433 as the "standard deviation" and used a t-test power formula, you would get a different n — one that ignores the non-constant variance of the binomial and the nonlinearity of the logit link. The GLM calculation accounts for both.

The design places all runs near the extremes of the Dose range (−1 and +1), which is optimal for estimating the dose slope — the same pattern seen in Chapter 3 for the Concentration slope. PatientAge values scatter across the range because the design needs to estimate the PatientAge coefficient as well, even though it is not the primary contrast being tested.

**Using the CLI template.** The package includes a starter YAML for binomial GLM designs:

```bash
iopt-design --template glm-binomial > dose_response.yml
```

This writes a commented template covering `baseline`, `family`, `L`, `delta`, and all `design` options. Edit the placeholder values for your formula and factors, then run:

```bash
iopt-design --config dose_response.yml --out ./output/dose_response
```

A Poisson equivalent is available at `iopt-design --template glm-poisson`.

---

#### 5.5 Full worked example — Poisson

**Context.** A process engineering team is optimising a coating process to reduce surface defects. The response is defect count per batch, which follows a Poisson distribution. Historical data gives a baseline rate of μ₀ = 0.8 defects per batch. The team wants 80% power to detect that optimising temperature and dwell time together can reduce the defect rate by at least 50% (from 0.8 to 0.4).

Two factors are investigated in an interaction model: Temperature (normalised, −1 to +1) and DwellTime (normalised, −1 to +1).

```python
# chapter5_poisson.py
import numpy as np
from iopt_power_design import (
    i_optimal_powered_design,
    PowerGLMContrastConfig,
    DesignOptions,
)

# ── 1. Define the model ────────────────────────────────────────────────────
formula = "~ 1 + Temperature + DwellTime + Temperature:DwellTime"
factors = {
    "Temperature": (-1.0, 1.0),
    "DwellTime":   (-1.0, 1.0),
}
# Patsy columns (p = 4):
#   [Intercept, Temperature, DwellTime, Temperature:DwellTime]

# ── 2. Translate the effect to the log-rate scale ──────────────────────────
mu0, mu1 = 0.8, 0.4   # baseline and target defect rates
delta = abs(np.log(mu1) - np.log(mu0))

print(f"log({mu0}) = {np.log(mu0):.4f}")
print(f"log({mu1}) = {np.log(mu1):.4f}")
print(f"delta (log-rate) = {delta:.4f}")     # log(0.5) = 0.693
print(f"Fisher weight w  = {mu0:.4f}")       # for Poisson, w = mu0

# ── 3. Set up GLM power config ─────────────────────────────────────────────
power_cfg = PowerGLMContrastConfig(
    L=[[0, 1, 0, 0]],  # test H₀: β_Temperature = 0  (column index 1)
    delta=[delta],      # log-rate effect of 0.693 ≈ halving the defect rate
    baseline=mu0,       # null operating point on the count scale
    family="poisson",
    link=None,          # canonical link: log
    alpha=0.05,
    power=0.80,
    max_n=200,
)

# ── 4. Run the design search ───────────────────────────────────────────────
opts = DesignOptions(
    auto_candidate=True,
    starts=8,
    random_state=42,
)

result = i_optimal_powered_design(
    formula=formula,
    factors=factors,
    power_cfg=power_cfg,
    design_opts=opts,
)

# ── 5. Inspect results ─────────────────────────────────────────────────────
r = result["report"]
print(f"\nMinimum n: {r['n']}")
print(f"Achieved power: {r['achieved_power']:.4f}")
print(f"Noncentrality λ: {r['noncentrality_lambda']:.4f}")
print(f"F-test df: ({r['df_num']}, {r['df_denom']})")
print()
print("Design (all runs):")
print(result["design_df"].round(3).to_string())
```

**Expected output:**

```
log(0.8) =  0.8755
log(0.4) =  0.1823
delta (log-rate) = 0.6931
Fisher weight w  = 0.8000

Minimum n: 25
Achieved power: 0.8076
Noncentrality λ: 8.0036
F-test df: (1, 21)

Design (all runs):
    Temperature  DwellTime
0        -0.998     -0.922
1        -0.988     -0.998
2        -0.937      0.870
3        -0.920      0.924
4         0.886      0.935
5         0.901     -0.999
6         0.960     -0.916
7         0.983      0.975
...      (25 runs total, all at extreme Temperature values)
```

**Comparing the two GLM examples.** The binomial example needed 44 runs; the Poisson example needs only 25. The key difference is the Fisher weight:

| Example | Family | Baseline | Fisher weight w |
|---|---|---|---|
| Dose-response | Binomial | p₀ = 0.25 | w = 0.25 × 0.75 = **0.1875** |
| Defect count | Poisson | μ₀ = 0.8 | w = **0.800** |

The Poisson process has a substantially larger Fisher weight — each observation carries more information about the rate parameter — so fewer runs are needed to achieve the same power. This is the formal expression of the intuition that count data (which can exceed 1) is generally more informative per observation than binary data.

Both designs concentrate runs at the extreme values of the primary factor (Dose or Temperature), for the same reason established in Chapter 3: estimating a linear slope on the link scale is most efficient at the extremes of the factor range.

> **Cross-reference:** To check how sensitive the Poisson design's power is to the assumed baseline rate μ₀ (what if the true rate is 1.2 rather than 0.8?), use `power_curve_by_baseline` as described in Chapter 19. For the binomial design, `power_curve_by_baseline` sweeps the baseline probability p₀ and shows how power changes as that assumption moves.

---

### Chapter 6 — Multi-response designs: powering several outcomes simultaneously

**Running example.** A polymer reactor team is studying how two coded process factors — Temperature (Temp, −1 to +1) and Pressure (Press, −1 to +1) — affect both **yield** and **purity** in the same set of experimental runs. The two responses have different measurement noise: yield has σ = 1.0, purity has σ = 1.5. Both must achieve 80% power before the run schedule is approved.

---

#### 6.1 The multi-response problem: why joint power is harder than single-response power

In a single-response experiment, choosing how many runs to run is a straightforward trade-off between cost and power. You specify a minimum detectable effect, a noise level, and a target power, and the binary search finds the smallest n that clears the bar.

With multiple responses, two complications arise.

**First, different responses may require different numbers of runs.** A noisy response or a small target effect size will demand more runs than a precise measurement of a large effect. Suppose you calculate the minimum n for each response independently:

- Yield (σ = 1.0, δ = 0.5): the Fedorov search finds n = 42 runs.
- Purity (σ = 1.5, δ = 0.5): the Fedorov search finds n = 100 runs.

If you run only 42 experiments, yield is powered but purity is not. You must run enough experiments to satisfy every response simultaneously.

**Second, a single design matrix must serve all responses at once.** You cannot run one set of experiments for yield and a different set for purity — you have one physical experiment and one run table. The design that is I-optimal for yield may not be optimal for purity, and the joint design is a compromise that satisfies both. In practice, when responses share the same model formula and factor space, the I-optimal compromise is usually close to optimal for each individual response; the dominant cost is simply the run count, not the point arrangement.

The package addresses both complications with a joint binary search: it finds the minimum n such that all responses are powered under a user-chosen combination rule, and at each n it builds a single I-optimal design that serves all responses.

---

#### 6.2 Combination rules: how per-response power scores are folded into one objective

At each candidate n during the binary search, the optimiser evaluates the power for every response on the chosen design and must decide whether that n is "good enough." It does this by collapsing per-response powers into one combined scalar and comparing that scalar to a target.

The package offers three rules.

---

**`"min"` — the guaranteed-floor rule (recommended default)**

```
combined_power = min(power_1, power_2, ..., power_k)
```

The combined power equals the power of the weakest response. The binary search finds the smallest n where `min(power_i) ≥ max(target_i)` — that is, where every response meets its own individual target. This is the most conservative choice: it gives an unconditional guarantee that no response falls short.

Use `"min"` whenever you need to be able to say "this design has ≥ 80% power for every one of these hypotheses individually."

---

**`"product"` — joint-probability rule**

```
combined_power = power_1 × power_2 × ... × power_k
```

The combined power is interpreted as the probability that all hypothesis tests reject simultaneously, under the assumption that the responses are statistically independent. The binary search finds n where `product(power_i) ≥ product(target_i)`.

For two responses with target 80%, the product target is 0.80 × 0.80 = 0.64 — a combined 64% joint-detection probability. This means individual response powers at that n will typically be below 80%. In the running example, `"product"` returns n = 71, but purity power at n = 71 is only 0.68 — purity does not meet its individual 80% target.

Use `"product"` only when (a) responses are genuinely independent (uncorrelated error structure) and (b) you care specifically about the joint-rejection probability rather than individual response guarantees. For correlated responses, use the Hotelling T² path with `sigma_joint` instead (see Section 6.3).

---

**`"weighted_mean"` — importance-weighted rule**

```
combined_power = Σ (w_i × power_i) / Σ w_i
```

The combined power is the weighted average of individual powers. Weights are normalised internally, so only the ratios matter: a response with weight 2.0 counts twice as much as one with weight 1.0.

Use `"weighted_mean"` when responses have genuinely unequal business importance and you can accept that a lower-priority response may be underpowered in exchange for a smaller overall n. In the running example with equal weights, `"weighted_mean"` returns n = 68, but purity power is only 0.66.

**The bottom line for most users:** use `"min"`. The run savings from `"product"` or `"weighted_mean"` come directly at the expense of per-response power guarantees. The table below summarises the tradeoff for the running example.

| Rule | n | Combined power | Yield power | Purity power |
|------|---|----------------|-------------|--------------|
| `"min"` | 100 | 0.8014 | 0.9868 | 0.8014 |
| `"product"` | 71 | 0.6405 | 0.9478 | 0.6757 |
| `"weighted_mean"` | 68 | 0.7997 | 0.9391 | 0.6604 |

Only `"min"` guarantees both responses meet 80%.

---

#### 6.3 `ResponseSpec` and `MultiResponseOptions`

Each response is described by a `ResponseSpec` dataclass, and the collection of responses plus the combination rule are bundled into a `MultiResponseOptions` dataclass.

**`ResponseSpec` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Label for this response. Used as a key in the result dict. |
| `power_cfg` | `PowerContrastConfig` or `PowerR2Config` | Power requirements for this response. Each response may use a different mode and different parameters. |
| `formula` | `str` or `None` | Response-specific formula override. If `None`, the global formula passed to the design function is used. Set a different formula only when responses have different model structures (the compound-criterion path). |
| `weight` | `float` (default 1.0) | Relative importance weight for `"weighted_mean"` combination. Ignored for `"min"` and `"product"`. |

**`MultiResponseOptions` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `responses` | `list[ResponseSpec]` | At least two entries required. Names must be unique. |
| `power_combination` | `"min"`, `"product"`, or `"weighted_mean"` | Combination rule. Default `"min"`. |
| `sigma_joint` | `ndarray (k×k)` or `None` | Inter-response error covariance for Hotelling T² joint power. Must be symmetric positive definite. Only valid when all responses share the same formula and use contrast mode. Leave as `None` unless you have a well-estimated cross-response covariance matrix. |

> **Note on `sigma_joint`:** The Hotelling T² path replaces per-response scalar power with a multivariate test. This is theoretically appropriate for correlated responses (for example, yield and purity measured from the same physical sample), but it requires a reliable estimate of the full k×k error covariance matrix. In most practical settings, pilot data does not provide a reliable estimate of this matrix, and the independence assumption behind `"min"` is the more robust choice.

---

#### 6.4 Running `i_optimal_multiresponse_design` and reading the result

The function signature mirrors `i_optimal_powered_design`:

```python
result = i_optimal_multiresponse_design(
    formula,     # global Patsy formula (RHS)
    factors,     # factor definitions dict
    multi_cfg,   # MultiResponseOptions
    design_opts, # DesignOptions (optional; defaults to DesignOptions())
)
```

The result is a dict with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `"design"` | `DataFrame` | The run table (n rows × factor columns). |
| `"n"` | `int` | Number of runs selected. |
| `"achieved_power"` | `float` | Combined power at the selected n, under the chosen rule. |
| `"combination_rule"` | `str` | Which rule was used. |
| `"responses"` | `list[dict]` | One entry per response. Each dict contains `"name"`, `"power"`, `"lam"` (noncentrality), and `"n"`. |
| `"buckets"` | `DataFrame` | Run counts by factor-level bucket (same format as single-response). |
| `"compound_criterion"` | `bool` | `True` if any response has a formula override (different model structure per response). |
| `"elapsed_sec"` | `float` | Wall-clock time for the search. |
| `"search_strategy"` | `str` | `"bisection"` or `"bisection+verification"`. |
| `"warnings"` | `list[str]` | Non-fatal warnings from the search. |

**Reading per-response results.** The `"responses"` list contains one entry per `ResponseSpec`, in the same order you defined them. Each entry is a dict:

```python
for resp in result["responses"]:
    print(f"{resp['name']}: power={resp['power']:.4f}, lambda={resp['lam']:.4f}")
```

When using the `"min"` rule, the `"achieved_power"` at the top level equals the minimum of the per-response powers. The response with the lowest power is the one that determined n.

---

#### 6.5 Full worked example

**Scenario.** A polymer reactor team needs to characterise how Temperature (Temp) and Pressure (Press) affect both yield and purity in a continuous-flow process. Both factors are available in coded form on [−1, +1]. The team has preliminary estimates: yield noise σ = 1.0, purity noise σ = 1.5. Both responses should have ≥ 80% power to detect a main-effect coefficient of 0.5 at α = 0.05.

```python
from iopt_power_design import (
    i_optimal_multiresponse_design,
    PowerContrastConfig,
    DesignOptions,
    ResponseSpec,
    MultiResponseOptions,
)

formula = "~ 1 + Temp + Press"
factors = {
    "Temp":  (-1.0, 1.0),
    "Press": (-1.0, 1.0),
}
# p = 3 model columns: [Intercept, Temp, Press]
# L = [[0, 1, 0]] tests the Temp main effect
# L = [[0, 0, 1]] tests the Press main effect

# --- Per-response power configs ---
yield_cfg = PowerContrastConfig(
    L=[[0, 1, 0]],   # test Temperature main effect
    delta=[0.5],      # minimum detectable coefficient
    alpha=0.05,
    power=0.80,
    sigma=1.0,
    max_n=300,
)

purity_cfg = PowerContrastConfig(
    L=[[0, 0, 1]],   # test Pressure main effect
    delta=[0.5],
    alpha=0.05,
    power=0.80,
    sigma=1.5,        # purity measurements are noisier
    max_n=300,
)

# --- Collect responses and set combination rule ---
responses = [
    ResponseSpec(name="yield",  power_cfg=yield_cfg),
    ResponseSpec(name="purity", power_cfg=purity_cfg),
]

multi_cfg = MultiResponseOptions(
    responses=responses,
    power_combination="min",   # guarantee both responses meet target
)

opts = DesignOptions(auto_candidate=True, starts=5, random_state=42)

# --- Run the joint design search ---
result = i_optimal_multiresponse_design(formula, factors, multi_cfg, opts)
```

**Reading the result:**

```python
print("n:", result["n"])
print("achieved_power:", round(result["achieved_power"], 4))

for resp in result["responses"]:
    print(f"  {resp['name']}: power={resp['power']:.4f}, lambda={resp['lam']:.4f}")
```

Output:

```
n: 100
achieved_power: 0.8014
  yield:  power=0.9868, lambda=17.8171
  purity: power=0.8014, lambda=8.0354
```

The search found that n = 100 runs are needed. Purity is the **binding constraint**: it requires n = 100 to achieve 80% power given its higher noise (σ = 1.5). Yield, being less noisy (σ = 1.0), only needed n = 42 on its own; at n = 100 it achieves near-perfect power (0.9868). The combined power reported at the top level equals the minimum across responses (0.8014), which is purity's power.

This is the central feature of the `"min"` rule: the design guarantees at least 80% power for every response, and it is always the noisiest or smallest-effect response that determines the final run count.

---

**Power curve across n.**

To understand how each response's power changes with sample size, use `power_curve_by_n_multiresponse`:

```python
from iopt_power_design import power_curve_by_n_multiresponse

curve_df = power_curve_by_n_multiresponse(
    formula, factors, multi_cfg,
    n_range=(20, 120),
    n_points=6,
    design_opts=opts,
)
print(curve_df.to_string())
```

Output:

```
     n  combined_power  yield_power  purity_power
0   20        0.2563       0.4880        0.2563
1   40        0.4525       0.7885        0.4525
2   60        0.6148       0.9091        0.6148
3   80        0.7215       0.9657        0.7215
4  100        0.8014       0.9868        0.8014
5  120        0.8581       0.9949        0.8581
```

At n = 40, yield already exceeds 0.78 power but purity is still below 0.45. The combined (min) power curve is entirely driven by purity. This view is useful for communicating the run-count–power tradeoff to stakeholders: if the team can only afford 80 runs, the combined power drops to 0.72, and that shortfall lands entirely on purity.

Pass `plot=True` to get an automatic line chart, or capture the DataFrame and build a custom Plotly or matplotlib figure (see Chapter 20 for plotting patterns).

---

**Sigma sensitivity at fixed n.**

`multiresponse_sensitivity` builds one design at a fixed n and sweeps a common noise scale factor, asking: "How does our joint power change if our preliminary σ estimates turn out to be wrong?"

This function requires all responses to use `PowerContrastConfig` (sigma scaling is undefined for R²-mode responses).

```python
from iopt_power_design import multiresponse_sensitivity

sens_df = multiresponse_sensitivity(
    formula, factors, multi_cfg,
    fixed_n=100,
    sigma_range=(0.5, 2.5),
    sigma_points=6,
    design_opts=opts,
)
print(sens_df.to_string())
```

Output:

```
   sigma_scale  combined_power  yield_power  purity_power
0          0.5        0.9999       1.0000        0.9999
1          0.9        0.8766       0.9964        0.8766
2          1.3        0.5789       0.8952        0.5789
3          1.7        0.3788       0.6909        0.3788
4          2.1        0.2670       0.5121        0.2670
5          2.5        0.2022       0.3867        0.2022
```

Each response's σ is multiplied by the scale factor. A scale of 1.0 (not shown; it lies between rows 1 and 2) recovers the nominal design-point power. The table shows that the design is robust to modest noise overestimation: even if purity is 10% noisier than expected (scale = 0.9 × 1.0, scale = 0.9 × 1.5 for purity), combined power stays above 0.87. But if the noise is 30% higher than the pilot estimate (scale = 1.3), combined power falls to 0.58 — a meaningful shortfall. The conclusion is that the σ estimate matters most for purity, and it is worth investing in a tight noise estimate for that response before committing to n = 100.

---

**When responses have different model formulas.**

All of the above assumes that yield and purity share the same formula and factor space. In some experiments, responses have different model structures — for example, one response is fit with a main-effects model while another requires an interaction term. `ResponseSpec` supports this via its `formula` field:

```python
response_a = ResponseSpec(
    name="yield",
    power_cfg=yield_cfg,
    formula="~ 1 + Temp + Press",            # main effects only
)
response_b = ResponseSpec(
    name="purity",
    power_cfg=purity_cfg,
    formula="~ 1 + Temp + Press + Temp:Press",  # interaction added
)
```

When any response has a formula that differs from the global formula, the package activates the **compound criterion** path. In this mode, each response has its own model matrix and its own power evaluation, but the design is still chosen from a single shared candidate set and the physical run table is the same for all responses. The `result["compound_criterion"]` key will be `True` when this path is active.

For most studies, all responses share the same model formula, and the compound path is not needed.

---

## Part III — Optimality Criteria in Depth

### Chapter 7 — Choosing between I, D, and A

Chapters 3–5 used I-optimality throughout and mentioned the alternatives only in passing. This chapter explains what I, D, and A actually measure, when the choice between them matters, and how to use `compare_criteria` to let the data guide the decision.

---

#### 7.1 Mathematical definitions and geometric interpretations

All three criteria operate on the **information matrix** M = X'X, where X is the n × p model matrix. The criteria differ in which property of M they optimise.

---

**I-optimality — integrated prediction variance**

The I-criterion is the average variance of the predicted response ŷ(x) across the entire design region R:

```
I = (1 / |R|) ∫_R f(x)ᵀ (X'X)⁻¹ f(x) dx
```

where f(x) is the vector of model-term values at point x. Minimising I spreads prediction uncertainty uniformly across the region. An I-optimal design is "prediction-fair" — no part of the design space is much worse-predicted than any other.

Geometrically: I-optimal designs typically cluster runs near the boundaries of the design region (where I-criterion is highest without design support) with moderate support in the interior. For a continuous interval [a, b], I-optimal designs often place the bulk of their runs at or near the extreme values, with a smaller fraction at intermediate points.

**Use I when the model will be used to predict the response at arbitrary points in the design space** — for example, when you want to map a response surface and identify factor settings that achieve a target response, or when you plan to use the fitted model for interpolation.

---

**D-optimality — volume of the coefficient confidence ellipsoid**

The D-criterion is:

```
D = -log det(X'X)
```

Minimising D is equivalent to maximising `det(X'X)`, which minimises the volume of the joint confidence ellipsoid for all model coefficients. A smaller ellipsoid means the coefficients are estimated with greater joint precision.

Geometrically: D-optimal designs push almost all runs to the extremes of the design region — the vertices of the design space in the case of a box constraint. For a two-level factor, all runs are at the two levels; for a continuous factor on [a, b], runs cluster at a and b. There are no interior points. This extreme-placement property maximises the spread of the columns of X, which maximises det(X'X).

**Use D when coefficient estimation precision is the primary goal** — for example, in mechanistic modelling where the coefficient values themselves are the scientific output, or in screening studies where you want tight estimates of all effects simultaneously.

---

**A-optimality — sum of coefficient-estimate variances**

The A-criterion is:

```
A = trace((X'X)⁻¹)
```

Minimising A minimises the sum of the individual coefficient variances, which is the sum of the diagonal entries of the covariance matrix of the coefficient estimates. Unlike D, which treats all coefficients jointly, A treats each coefficient independently and tries to equalise their individual uncertainties.

Geometrically: A-optimal designs look similar to D-optimal in many standard settings, but they can diverge in models with many categorical factors or in designs where some coefficients are much harder to estimate precisely than others. Because A-optimality penalises each variance term independently, it is sensitive to the relative scaling of model terms.

**Use A when you want balanced individual precision across all effects and interactions** — for example, when the analysis will consist of a series of separate tests (one per coefficient) and you want none of them to be substantially underpowered due to poor leverage on that term.

---

**The Venn overlap.** For symmetric, fully continuous designs on a hypercube, I, D, and A often agree on the run count (they produce essentially the same design). Divergence appears most strongly when:

- the model includes categorical factors (which introduce asymmetries in X'X)
- the factor space is irregular or constrained
- the model has many interaction or higher-order terms
- some factors span very different numerical ranges

---

#### 7.2 How the criteria appear in this package

Every call to `i_optimal_powered_design` (or `compare_criteria`) reports two criterion-level diagnostics in the result:

| Diagnostic | Description |
|------------|-------------|
| `i_criterion` | The I-criterion value of the returned design, normalised to the candidate set size. Lower is better. |
| `d_efficiency` | D-efficiency relative to the D-optimal design at the same n. A value of 1.0 means the design is D-optimal; lower values indicate that the design sacrifices D-efficiency to improve I or A. |
| `condition_number` | Condition number of X'X. Very large values (> 1000) indicate near-collinearity in the model matrix, which makes individual coefficient estimates unreliable regardless of which criterion is used. |

When you use `criterion="I"` (the default), the package minimises I-criterion and reports the resulting d_efficiency as a secondary measure. When you use `criterion="D"` or `criterion="A"`, the package minimises the corresponding objective and reports d_efficiency as before.

---

#### 7.3 `compare_criteria` in practice

`compare_criteria` runs the full powered-design search independently under each criterion and returns a side-by-side summary. It is the recommended tool for understanding whether your criterion choice actually matters for a given problem.

```python
from iopt_power_design import compare_criteria, PowerContrastConfig, DesignOptions

comparison = compare_criteria(formula, factors, power_cfg, design_opts=opts)
print(comparison["summary"])
```

The `summary` DataFrame contains one row per criterion:

| Column | Description |
|--------|-------------|
| `criterion` | `"I"`, `"D"`, or `"A"` |
| `n` | Minimum n achieving the target power under that criterion |
| `achieved_power` | Statistical power of the returned design |
| `elapsed_sec` | Wall-clock time for that criterion's search |
| `condition_number` | Condition number of the returned design's X'X |
| `d_efficiency` | D-efficiency of the returned design (1.0 = D-optimal at that n) |

The full design and report for each criterion are available in `comparison["results"]["I"]`, `["D"]`, and `["A"]`.

**When criteria agree on n**, the choice is mostly cosmetic — you can use any of them and the run count is the same. Focus on which design property matters most for your analysis (prediction vs. coefficient estimation).

**When criteria disagree on n**, the criterion with the lowest n is the most efficient for your specific hypothesis test. Understanding *why* it is more efficient — which the design structure comparison will reveal — is worth the extra analysis time.

---

#### 7.4 Full worked example

**Scenario.** The Chapter 3 polymer study (Catalyst A/B + Concentration 0–2 mol/L, with interaction) is being prepared for a final report. The statistician wants to confirm that the criterion choice is appropriate and to show the principal investigator a comparison of all three criteria before committing to the run schedule.

```python
from iopt_power_design import (
    compare_criteria,
    PowerContrastConfig,
    DesignOptions,
)

formula = "~ 1 + Catalyst + Concentration + Catalyst:Concentration"
factors = {
    "Catalyst":      ["A", "B"],
    "Concentration": (0.0, 2.0),
}
# p = 4: [Intercept, Catalyst[T.B], Concentration, Catalyst[T.B]:Concentration]
# Test: Concentration main effect  →  L = [[0, 0, 1, 0]]
power_cfg = PowerContrastConfig(
    L=[[0, 0, 1, 0]],
    delta=[0.5],
    alpha=0.05,
    power=0.80,
    sigma=1.0,
    max_n=500,
)
opts = DesignOptions(auto_candidate=True, starts=5, random_state=42)

comparison = compare_criteria(formula, factors, power_cfg, design_opts=opts)
print(comparison["summary"].to_string())
```

Output:

```
  criterion   n  achieved_power  elapsed_sec  condition_number  d_efficiency
0         I  70        0.8030        42.60         51.0            0.4824
1         D  71        0.8031        12.03         50.4            0.4822
2         A  68        0.8000        33.77         24.6            0.4448
```

**Reading the table.**

The three criteria produce nearly the same run count (68–71) for this model and power target. Any of the three would be a defensible choice on the basis of run count alone.

The d_efficiency column tells a more nuanced story. All three designs have d_efficiency around 0.48, which means they are all about half as D-efficient as a pure D-optimal design at the same n would be. This is not a failure of the search — it is the expected consequence of the model structure. With an interaction term and a categorical factor, the I-optimal design already places runs at extreme Concentration values to support the interaction, and this extreme placement brings D-efficiency close to its maximum. There is little room left for the criteria to diverge.

---

**Inspecting the designs.** The geometric difference between criteria becomes clear when you look at the design structures directly.

```python
for crit in ["I", "D", "A"]:
    df = comparison["results"][crit]["design_df"]
    diag = comparison["results"][crit]["report"]["diagnostics"]
    cat = df["Catalyst"].value_counts().to_dict()
    print(f"\n--- {crit}-optimal (n={len(df)}) ---")
    print(f"  Concentration: min={df['Concentration'].min():.3f}, "
          f"median={df['Concentration'].median():.3f}, "
          f"max={df['Concentration'].max():.3f}")
    print(f"  Catalyst: {cat}")
    print(f"  I-criterion: {diag['i_criterion']:.6f}, "
          f"condition_number: {diag['condition_number']:.1f}")
```

Output:

```
--- I-optimal (n=70) ---
  Concentration: min=0.002, median=1.931, max=1.999
  Catalyst: {'A': 35, 'B': 35}
  I-criterion: 0.038824, condition_number: 51.0

--- D-optimal (n=71) ---
  Concentration: min=0.002, median=1.930, max=1.999
  Catalyst: {'A': 35, 'B': 36}
  I-criterion: 0.038281, condition_number: 50.4

--- A-optimal (n=68) ---
  Concentration: min=0.002, median=0.069, max=1.999
  Catalyst: {'A': 40, 'B': 28}
  I-criterion: 0.046790, condition_number: 24.6
```

The I-optimal and D-optimal designs are nearly identical: both cluster the majority of runs at high Concentration (median ≈ 1.93) with a small fraction at near-zero, and both split Catalyst evenly (35/35 and 35/36). This is expected — for a model with an interaction term, I and D both benefit from the same extreme-Concentration strategy.

The A-optimal design is strikingly different:

- **Reversed Concentration skew.** Median Concentration is 0.069 — most runs are at *low* Concentration. The design still uses the full range (min ≈ 0, max ≈ 2), but the balance is inverted relative to I and D.
- **Unequal Catalyst allocation.** A-optimality assigns 40 runs to Catalyst A and only 28 to Catalyst B. This is not a mistake; it reflects A-optimality's coefficient-by-coefficient optimisation. The [Intercept] and Concentration coefficients are estimated with level A as the reference, so A-optimal allocates more runs there to tighten those specific variances.
- **Lower condition number** (24.6 vs. 50–51). The A-optimal design produces a better-conditioned information matrix. This can be an advantage in settings where numerical precision or near-collinearity is a concern.
- **Higher I-criterion** (0.0468 vs. 0.0388). The A-optimal design is about 20% worse at minimising prediction variance. If the model will be used for response surface mapping, the A-optimal design will give noisier predictions in parts of the design region.

The practical implication: for this study, if the goal is to predict yield across the full Concentration range (a response surface goal), I-optimal is the right choice. If the goal is purely hypothesis testing with good numerical properties, A-optimal achieves the same power at n = 68 (two fewer runs) with a better-conditioned matrix. The difference is small in absolute terms, but the structural differences in the design are meaningful.

---

**Adding a Plotly visualisation.**

The `compare_criteria` function has a built-in `plot=True` option that produces a matplotlib bar chart. For an interactive Plotly figure suitable for a notebook or report, build it from the summary DataFrame:

```python
import plotly.graph_objects as go

summary = comparison["summary"]

fig = go.Figure()

# Bar: n (primary axis)
fig.add_trace(go.Bar(
    x=summary["criterion"],
    y=summary["n"],
    name="n (runs)",
    marker_color=["steelblue", "tomato", "seagreen"],
    text=summary["n"],
    textposition="outside",
))

fig.update_layout(
    title="Criterion comparison — polymer study",
    xaxis_title="Optimality criterion",
    yaxis_title="Minimum runs (n)",
    yaxis=dict(range=[0, summary["n"].max() * 1.15]),
    showlegend=False,
    template="plotly_white",
    width=600,
    height=400,
)

fig.show()  # interactive in Jupyter; use fig.write_html("comparison.html") to save
```

For a side-by-side comparison of both n and d_efficiency, add a second trace on a secondary y-axis (see Chapter 20 for the full dual-axis pattern).

---

**When criteria agree — the symmetric continuous case.**

In the Chapter 4 consumer survey (four continuous factors on [−1, +1], main-effects model), all three criteria returned exactly the same n = 73. This is the typical result for main-effects-only models on symmetric continuous designs: the information matrix is nearly spherical in factor space, and I, D, and A achieve essentially the same objective under different names.

The practical guidance: run `compare_criteria` whenever you have categorical factors, interactions, or an asymmetric design region. Skip it when you have a simple main-effects model with symmetric continuous factors — the criteria will agree, and I-optimal is the safest default.

---

## Part IV — The Interfaces

*Each chapter introduces one interface from basic setup through a complete example. Simple cases appear in earlier chapters; these chapters focus on interface-specific features and workflows.*

---

### Chapter 8 — Python API: full programmatic control

The Python API is the foundation that all other interfaces build on. Every capability the package provides is reachable from Python, and the results are plain Python dicts and pandas DataFrames — easy to inspect, save, and integrate into larger workflows. Chapters 3–7 used the API throughout; this chapter consolidates the full parameter reference and covers the advanced features that earlier chapters deferred.

---

#### 8.1 The primary entry points

Two functions cover all design generation needs:

```python
from iopt_power_design import (
    i_optimal_powered_design,           # single response
    i_optimal_multiresponse_design,     # two or more responses simultaneously
)
```

Both share the same basic call shape: formula → factors → power configuration → design options. Single-response returns `{design_df, buckets_df, report}`; multi-response returns `{design, buckets, responses, n, achieved_power, ...}`. The result dict keys are described in full in Section 8.3.

The optional keyword arguments on `i_optimal_powered_design` that earlier chapters skipped:

| Argument | Type | Description |
|----------|------|-------------|
| `export_diagnostics_to` | `str` or `None` | Path prefix; if provided, writes diagnostic HTML and CSV files alongside the result. |
| `export_report_to` | `str` or `None` | Path prefix; if provided, writes a self-contained HTML (or PDF) report. Requires `pip install -e "[report]"`. A write failure does not stop the design result from being returned. |
| `progress_callback` | `callable` or `None` | Function called with the current `report` dict after each binary-search iteration. Useful for logging and progress bars (see Section 8.4). |

---

#### 8.2 `DesignOptions` deep dive

`DesignOptions` is a dataclass. All fields have defaults, so `DesignOptions()` is always valid. You only need to supply the fields you want to override.

```python
from iopt_power_design import DesignOptions

opts = DesignOptions(
    auto_candidate=True,
    starts=10,
    workers=4,
    random_state=42,
    criterion="I",
)
```

The fields group into five functional areas.

---

**Candidate set sizing**

The candidate set is the pool of feasible points from which the Fedorov exchange algorithm selects the design. A larger candidate set gives the algorithm more to choose from and tends to produce better designs, at the cost of more memory and slightly slower matrix operations.

| Field | Default | Description |
|-------|---------|-------------|
| `auto_candidate` | `False` | If `True`, size the candidate set automatically based on factor complexity (see below). Recommended for most workflows. |
| `candidate_points` | 2000 | Fixed candidate size when `auto_candidate=False`. Ignored when `auto_candidate=True`. |
| `cand_min` | 1000 | Minimum candidate size when `auto_candidate=True`. Ensures adequate coverage even for simple problems. |
| `cand_max` | 10000 | Maximum candidate size when `auto_candidate=True`. Prevents memory overuse in complex problems. |
| `cat_cells_cap` | 10000 | Cap on categorical cell enumeration. For designs with many categorical factors, the Cartesian product of levels can be enormous; this limits the enumerated cells before random sampling takes over. |
| `per_cell_alpha` | 1.5 | Multiplier for categorical cells in adaptive sizing. For purely categorical designs: `candidate_points = min(cells × per_cell_alpha, cand_max)`. |
| `per_cell_min` | 5 | Minimum continuous samples per categorical cell in mixed designs. |
| `per_cell_max` | 20 | Maximum continuous samples per categorical cell in mixed designs. |

**When to use `auto_candidate=True`:** For most problems — especially those with mixed factor types — `auto_candidate=True` is the safe default. It sizes the candidate set relative to the problem's complexity, so a 2-factor design gets a smaller candidate set than a 10-factor design with interactions.

**When to set `candidate_points` manually:** If you are running many repeated design searches (for example, in a loop over parameter values), fixing `candidate_points` to a moderate value (500–2000) gives more predictable runtime. If you are working with a constrained region where many candidate points will be rejected by the constraint filter, increase `candidate_points` to ensure the surviving pool is large enough.

---

**Search algorithm and starts**

The Fedorov exchange algorithm is a local optimiser: it starts from a random initial design and iteratively swaps points to improve the optimality criterion. Because it can get stuck in local optima, the package runs multiple independent random starts and returns the best result.

| Field | Default | Description |
|-------|---------|-------------|
| `criterion` | `"I"` | Optimality criterion: `"I"`, `"D"`, or `"A"`. See Chapter 7 for guidance. |
| `algo` | `"fedorov"` | Algorithm: `"fedorov"` (classic exchange) or `"coordinate"` (coordinate exchange, which can be faster for large problems at the cost of some solution quality). |
| `starts` | 5 | Number of independent random starts. More starts reduce the risk of returning a local optimum, at the cost of proportionally longer runtime. |
| `max_iter` | 1000 | Maximum exchange iterations per start. Rarely needs to be changed. |
| `xtx_jitter` | 1e-8 | Diagonal regularisation added to X'X before inversion. Increase slightly (to 1e-6) if you see numerical warnings about near-singular matrices. |

**Guidance on `starts`:** The default of 5 is sufficient for problems with fewer than about 6 factors and no interactions. For problems with many factors, interactions, or categorical variables, increasing `starts` to 10–20 can meaningfully improve solution quality, particularly for the I and A criteria (which have more complex landscapes than D). The elapsed time scales linearly with `starts`, so the cost is predictable.

---

**Parallelism**

| Field | Default | Description |
|-------|---------|-------------|
| `workers` | `None` | Number of parallel processes. `None` or `<= 1` runs serially. When `> 1`, each start runs in a separate process; results are collected and the best is returned. |
| `parallel_seed_stride` | 10000 | Seed offset between parallel starts. Worker `i` gets seed = `random_state + i × parallel_seed_stride`, ensuring that parallel starts explore different parts of the candidate space. |

**Important on Windows and macOS:** Python's `multiprocessing` module uses `spawn` start method on these platforms. Calls with `workers > 1` must be guarded with `if __name__ == "__main__":` in script files to prevent recursive subprocess spawning. This guard is not needed in Jupyter notebooks (which run in a `__main__` context).

```python
# script.py — required guard on Windows and macOS
if __name__ == "__main__":
    result = i_optimal_powered_design(
        formula, factors, power_cfg,
        DesignOptions(starts=20, workers=4, random_state=42),
    )
```

---

**Reproducibility**

| Field | Default | Description |
|-------|---------|-------------|
| `random_state` | 123 | Integer seed for all random number generation: candidate sampling, initial design construction, and start selection. **Must be an integer.** Passing `None` raises `ValueError`. |

`random_state` controls the entire search. Two calls with the same `random_state`, `starts`, `workers`, and `parallel_seed_stride` — and the same package version — will produce identical results. If you change `workers`, the parallel seed assignment changes and the result may differ even with the same `random_state`.

---

**Blocked designs**

| Field | Default | Description |
|-------|---------|-------------|
| `n_blocks` | `None` | Number of blocks. Set to an integer ≥ 2 to activate blocked design mode. |
| `block_sizes` | `None` | Optional list of per-block run counts. Length must equal `n_blocks`. If `None`, blocks are sized as evenly as possible. |
| `block_factor_name` | `"Block"` | Name of the blocking factor column in the output design DataFrame. Must not collide with any treatment factor name. |

Blocked designs are covered in Chapter 16. The formula in a blocked call does not include the block factor — the API adds block indicators automatically based on `n_blocks` and `block_factor_name`.

---

**Split-plot designs**

| Field | Default | Description |
|-------|---------|-------------|
| `split_plot` | `None` | `SplitPlotOptions` instance for split-plot designs. See Chapter 15 for full coverage. Cannot be set at the same time as `n_blocks`. |

---

**Feasibility constraints**

| Field | Default | Description |
|-------|---------|-------------|
| `constraint_func` | `None` | Python callable: `(pd.Series) → bool`. Returns `True` to keep a candidate point, `False` to discard it. |
| `constraint_expr` | `None` | String expression alternative for YAML configs and non-Python interfaces. Factor names are available as variables. A restricted set of math functions is supported (`sqrt`, `log`, `exp`, `abs`, `min`, `max`, etc.). If both are provided, `constraint_expr` takes precedence. |

Examples:

```python
# Callable form — useful in scripts where you want Python logic
opts = DesignOptions(
    constraint_func=lambda row: row["Temperature"] + row["Pressure"] <= 150.0,
    auto_candidate=True,
    random_state=42,
)

# String form — portable to YAML configs and the CLI
opts = DesignOptions(
    constraint_expr="Temperature + Pressure <= 150.0",
    auto_candidate=True,
    random_state=42,
)
```

Constraints are applied during candidate set construction, before the Fedorov exchange. Points that fail the constraint are excluded from the pool that the algorithm draws from. Chapter 17 covers constraints in depth.

---

#### 8.3 The result dict: complete field reference

**Single-response result** (`i_optimal_powered_design`):

```python
result = i_optimal_powered_design(formula, factors, power_cfg, opts)
```

| Key | Type | Description |
|-----|------|-------------|
| `"design_df"` | `DataFrame` | The run table: n rows × factor columns. Factor columns use the original names from the `factors` dict. |
| `"buckets_df"` | `DataFrame` | Factor-level bucket counts. Each row is a unique combination of factor settings that appears at least once; the `count` column says how many times. |
| `"report"` | `dict` | Search metadata, power metrics, and diagnostics. See below. |

**`report` fields:**

| Key | Type | Description |
|-----|------|-------------|
| `"n"` | `int` | Final run count (minimum n achieving target power). |
| `"p"` | `int` | Total model parameter count (including block indicators if blocked). |
| `"p_treat"` | `int` | Treatment-only parameter count (excluding block indicators). |
| `"df_num"` | `int` | Numerator degrees of freedom for the F-test (number of rows in L). |
| `"df_denom"` | `int` | Denominator degrees of freedom (n − p for standard designs; adjusted for split-plot or blocked designs). |
| `"alpha"` | `float` | Significance level from the power config. |
| `"target_power"` | `float` | Target power from the power config. |
| `"achieved_power"` | `float` | Power of the returned design at the selected n. |
| `"noncentrality_lambda"` | `float` | Noncentrality parameter λ for the final design. |
| `"criterion"` | `str` | Optimality criterion used (`"I"`, `"D"`, or `"A"`). |
| `"algo"` | `str` | Search algorithm used (`"fedorov"` or `"coordinate"`). |
| `"starts"` | `int` | Number of starts used. |
| `"workers"` | `int` or `None` | Parallel workers used. |
| `"candidate_points"` | `int` | Candidate set size used. |
| `"elapsed_sec"` | `float` | Wall-clock seconds for the entire search. |
| `"search_strategy"` | `str` | `"bisection"` or `"bisection+verification"`. |
| `"verify_window"` | `int` | Size of the verification window used at convergence. |
| `"random_state"` | `int` | Seed used (for reproducibility audit). |
| `"warnings"` | `list[str]` | Non-fatal warnings from the search (empty list if none). |
| `"diagnostics"` | `dict` | Design matrix diagnostics: `condition_number`, `d_efficiency`, `i_criterion`, `leverage_mean`, `leverage_max`, `leverages` (list of per-run leverage values). |
| `"block_structure"` | `dict` or `None` | Block sizes and factor name if blocked; `None` otherwise. |
| `"split_plot"` | `dict` or `None` | Split-plot structure if applicable; `None` otherwise. |
| `"report_path"` | `str` | Path to the written HTML report (only present when `export_report_to` was set and succeeded). |

**Accessing diagnostics:**

```python
report = result["report"]
diag   = report["diagnostics"]

print(f"n={report['n']}, power={report['achieved_power']:.4f}")
print(f"λ={report['noncentrality_lambda']:.4f}, df=({report['df_num']},{report['df_denom']})")
print(f"condition_number={diag['condition_number']:.1f}")
print(f"d_efficiency={diag['d_efficiency']:.4f}")
print(f"i_criterion={diag['i_criterion']:.6f}")
```

---

#### 8.4 Progress callbacks: monitoring long runs

For designs with many factors, high `max_n`, or many starts, the binary search can take several minutes. A progress callback lets you log each iteration or display a live progress indicator without polling.

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

def log_progress(report: dict) -> None:
    log.info(
        "  iter=%d  n=%d  power=%.4f  elapsed=%.1fs",
        report.get("iteration", 0),
        report["n"],
        report["achieved_power"],
        report.get("elapsed_sec", 0.0),
    )

result = i_optimal_powered_design(
    formula, factors, power_cfg,
    DesignOptions(auto_candidate=True, starts=10, random_state=42),
    progress_callback=log_progress,
)
```

The callback receives the `report` dict as it looks after each binary-search step — the same structure as the final `result["report"]`, but without the final enrichment fields (`elapsed_sec`, `search_strategy`, `warnings`). If the callback raises an exception, the package catches it, emits a `RuntimeWarning`, and continues the search. The callback is called once per bisection step, not once per Fedorov-exchange iteration.

For a Jupyter notebook progress bar, the `tqdm` library integrates cleanly:

```python
from tqdm.auto import tqdm

class TqdmCallback:
    def __init__(self, max_n: int):
        self.bar = tqdm(total=max_n, desc="n search", unit="runs")
        self._last_n = 0

    def __call__(self, report: dict) -> None:
        n = report["n"]
        self.bar.update(n - self._last_n)
        self.bar.set_postfix(power=f"{report['achieved_power']:.3f}")
        self._last_n = n

    def close(self):
        self.bar.close()

cb = TqdmCallback(max_n=power_cfg.max_n)
result = i_optimal_powered_design(
    formula, factors, power_cfg,
    DesignOptions(auto_candidate=True, starts=10, random_state=42),
    progress_callback=cb,
)
cb.close()
```

---

#### 8.5 Patterns for production scripts

**Persisting results.** The design DataFrame and the report dict are straightforward to save:

```python
import json
import pandas as pd

# Save design to CSV
result["design_df"].to_csv("design.csv", index=False)

# Save buckets to CSV
result["buckets_df"].to_csv("buckets.csv", index=False)

# Save report to JSON
# Convert numpy scalars to Python-native types first
report_serialisable = {
    k: (float(v) if hasattr(v, "item") else v)
    for k, v in result["report"].items()
    if k != "diagnostics"
}
report_serialisable["diagnostics"] = {
    k: (float(v) if hasattr(v, "item") else v)
    for k, v in result["report"]["diagnostics"].items()
    if k != "leverages"   # omit per-run list for brevity
}
with open("report.json", "w") as f:
    json.dump(report_serialisable, f, indent=2)
```

**Auto-saving the HTML report.** Pass `export_report_to` to have the package write the report in one step:

```python
result = i_optimal_powered_design(
    formula, factors, power_cfg, opts,
    export_report_to="./output/design_report.html",
)
# result["report"]["report_path"] contains the actual path written
```

**Error handling.** The package raises `ValueError` for invalid inputs (bad formula, wrong L dimensions, `max_n` too small) and `RuntimeWarning` for non-convergence (bisection reaches `max_n` without hitting the power target). Catch them explicitly in scripts that run unattended:

```python
import warnings

try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = i_optimal_powered_design(formula, factors, power_cfg, opts)

    if w:
        for warning in w:
            logging.warning("Design warning: %s", warning.message)

    if result["report"]["warnings"]:
        for msg in result["report"]["warnings"]:
            logging.warning("Search warning: %s", msg)

except ValueError as e:
    logging.error("Invalid input: %s", e)
    raise
```

---

#### 8.6 Full worked example

The following script puts together the API features covered in this chapter: parallel multi-start, a logging progress callback, the full result inspection, and auto-saved output.

```python
"""
production_design.py
End-to-end design generation with parallel starts,
progress logging, and persisted output.

Run with: python production_design.py
(if __name__ == "__main__" guard required on Windows / macOS for workers > 1)
"""

import json
import logging
import math
from pathlib import Path

from iopt_power_design import (
    i_optimal_powered_design,
    PowerContrastConfig,
    DesignOptions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Experiment definition ────────────────────────────────────────────────────
formula = "~ 1 + Temp + Press + Catalyst + Temp:Catalyst + Press:Catalyst"
factors = {
    "Temp":     (-1.0, 1.0),
    "Press":    (-1.0, 1.0),
    "Catalyst": ["A", "B", "C"],
}
# p = 8: Intercept, Temp, Press, Catalyst[T.B], Catalyst[T.C],
#         Temp:Catalyst[T.B], Temp:Catalyst[T.C],
#         Press:Catalyst[T.B], Press:Catalyst[T.C]   — actually p=9 for 3-level cat
# Test: Temp main effect  →  L = [[0, 1, 0, 0, 0, 0, 0, 0, 0]]
power_cfg = PowerContrastConfig(
    L=[[0, 1, 0, 0, 0, 0, 0, 0, 0]],
    delta=[0.5],
    alpha=0.05,
    power=0.80,
    sigma=1.0,
    max_n=400,
)

OUTPUT_DIR = Path("output")

# ── Progress callback ─────────────────────────────────────────────────────────
def log_iteration(report: dict) -> None:
    log.info(
        "  bisect iter=%-3d  n=%-4d  power=%.4f",
        report.get("iteration", 0),
        report["n"],
        report["achieved_power"],
    )

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    opts = DesignOptions(
        auto_candidate=True,
        starts=20,
        workers=4,                  # parallel on 4 cores
        random_state=2025,
        criterion="I",
    )

    log.info("Starting design search (starts=%d, workers=%d)…", opts.starts, opts.workers)

    result = i_optimal_powered_design(
        formula, factors, power_cfg, opts,
        export_report_to=str(OUTPUT_DIR / "report.html"),
        progress_callback=log_iteration,
    )

    rep = result["report"]
    log.info(
        "Done: n=%d, power=%.4f, λ=%.4f, elapsed=%.1fs",
        rep["n"], rep["achieved_power"], rep["noncentrality_lambda"], rep["elapsed_sec"],
    )

    # Save artefacts
    result["design_df"].to_csv(OUTPUT_DIR / "design.csv", index=False)
    result["buckets_df"].to_csv(OUTPUT_DIR / "buckets.csv", index=False)
    log.info("Design written to %s", OUTPUT_DIR / "design.csv")

    if rep.get("warnings"):
        for msg in rep["warnings"]:
            log.warning("Search warning: %s", msg)
    if "report_path" in rep:
        log.info("HTML report written to %s", rep["report_path"])
```

This script produces four output files: `design.csv`, `buckets.csv`, `report.html`, and the console log. It is structured to be importable without triggering the search (the `if __name__ == "__main__":` guard) and uses `workers=4` safely on all platforms.

> **Note on L construction for 3-level categoricals.** The formula above includes `Catalyst` with three levels, which produces two dummy columns (`Catalyst[T.B]` and `Catalyst[T.C]`), plus two interaction columns each. Before using a hardcoded L in production, verify the column order by running `build_model_matrix` on a small candidate and checking the column names, or use `contrast_from_scenarios` as described in Chapter 3. The `L=[[0,1,0,0,0,0,0,0,0]]` in the example above targets the `Temp` main effect and is correct for the stated formula, but any formula change requires re-verifying the L indices.

---

### Chapter 9 — CLI: reproducible file-based pipelines

The CLI is the right tool when you want design generation to be a **reproducible, file-based step** — something you can commit to version control, re-run months later, and wire into a Makefile or CI pipeline without touching Python. Every design produced by the CLI is deterministic given the YAML config; the config file is the single source of truth.

---

#### 9.1 Installing CLI support

The CLI requires PyYAML, which is bundled in the `cli` extras group:

```bash
pip install -e ".[cli]"
```

After installation, the `iopt-design` entry point is available in your shell:

```bash
iopt-design --help
```

---

#### 9.2 Config file structure and the four YAML templates

The CLI is driven by a single YAML (or JSON) config file. The fastest way to get a correct config is to print one of the four commented templates and edit it:

```bash
iopt-design --template contrast      > contrast_config.yml
iopt-design --template r2            > r2_config.yml
iopt-design --template glm-binomial  > glm_binomial_config.yml
iopt-design --template glm-poisson   > glm_poisson_config.yml
```

The four templates correspond to the four power modes:

| Template | Power mode | Key YAML field |
|----------|------------|----------------|
| `contrast` | Linear contrast F-test | `contrast:` block with `scenario_a/b` or explicit `L`/`delta` |
| `r2` | Omnibus global R² F-test | `r2_target: <float>` |
| `glm-binomial` | Logistic (binomial) Wald test | `family: binomial`, `link: logit`, `baseline: <p0>` |
| `glm-poisson` | Log-linear (Poisson) Wald test | `family: poisson`, `link: log`, `baseline: <mu0>` |

---

**Annotated contrast template.** The output of `iopt-design --template contrast` is reproduced below with explanatory annotations. Every field shown here is also valid in the other three templates.

```yaml
# iopt-design config — contrast mode

formula: "~ 1 + A + B + A:B"   # Patsy RHS formula; same syntax as Python API

factors:
  A: [low, high]      # categorical: a YAML list of level names
  B: [0.0, 10.0]      # continuous:  a two-element list [low, high]

# ── Power specification ──────────────────────────────────────────────────────
# For contrast mode: define the hypothesis to test.
contrast:
  # Option 1 — scenario-based (recommended)
  # The CLI builds L and delta automatically from the two factor settings.
  scenario_a: {A: low,  B: 5.0}
  scenario_b: {A: high, B: 5.0}
  sesoi: 1.0           # smallest effect of interest in response units

  # Option 2 — explicit L matrix and delta vector (advanced users)
  # L: [[0, 0, 1, 0]]  # p columns; must match Patsy column order
  # delta: [0.5]

alpha: 0.05
power: 0.80
sigma: 1.0             # assumed residual standard deviation

# ── Design search options ────────────────────────────────────────────────────
design:
  auto_candidate: true     # adaptive candidate sizing (recommended)
  candidate_points: 2000   # fixed size when auto_candidate: false
  cand_min: 1000
  cand_max: 10000
  starts: 5
  algo: fedorov            # fedorov | coordinate
  criterion: I             # I | D | A
  max_iter: 1000
  random_state: 123
  xtx_jitter: 1.0e-8
  workers: null            # null = serial; integer > 1 for parallel starts
  # constraint_expr: "Temperature <= 2 * Pressure"

# ── Output options ───────────────────────────────────────────────────────────
output:
  basename: design         # prefix for output file names
  excel: false             # also write a .xlsx workbook

# ── Split-plot (optional; uncomment to activate) ─────────────────────────────
# split_plot:
#   htc_factors: [A]       # hard-to-change (whole-plot) factor names
#   n_whole_plots: 6
#   eta: 1.0               # variance ratio sigma2_wp / sigma2_sp
#   subplots_per_wp: 4     # omit for auto
#   df_method: auto        # auto | conservative | sp_only

# ── Multi-response (optional; replace contrast: block with responses:) ────────
# power_combination: min   # min | product | weighted_mean
# responses:
#   - name: Yield
#     sigma: 2.0
#     contrast:
#       scenario_a: {A: low,  B: 5.0}
#       scenario_b: {A: high, B: 5.0}
#       sesoi: 1.0
#   - name: Purity
#     sigma: 0.5
#     r2_target: 0.20
```

The `formula` and `factors` keys are required for all modes. The power specification (`contrast:`, `r2_target:`, or `family:`/`baseline:`) switches the power mode. Everything in `design:` mirrors the `DesignOptions` fields from Chapter 8.

---

#### 9.3 Running a design

```bash
iopt-design --config polymer.yml --out ./output/polymer
```

The `--out` value is a **basename prefix** — not a directory. All output files are written alongside each other with the prefix prepended:

```
./output/polymer_design.csv
./output/polymer_buckets.csv
./output/polymer_report.json
```

The output directory is created automatically if it does not exist.

**Optional output flags** (can be passed on the command line or set in the `output:` section of the YAML):

```bash
# Write an Excel workbook alongside the CSVs
iopt-design --config polymer.yml --out ./output/polymer --excel

# Write a self-contained HTML report (requires pip install -e "[report]")
iopt-design --config polymer.yml --out ./output/polymer --html-report

# Print a compact robustness summary after the design (single-response only)
iopt-design --config polymer.yml --out ./output/polymer --robustness-report

# Verbose logging (DEBUG level)
iopt-design --config polymer.yml --out ./output/polymer -v
```

---

#### 9.4 Output files

| File | Description |
|------|-------------|
| `<basename>_design.csv` | Run table — n rows × factor columns. |
| `<basename>_buckets.csv` | Factor-level bucket counts. |
| `<basename>_report.json` | Search metadata, power metrics, and diagnostics. Same fields as `result["report"]` from the Python API. |
| `<basename>_report.html` | Self-contained HTML report (only with `--html-report`). |
| `<basename>.xlsx` | Excel workbook with Design, Buckets, and Report sheets (only with `--excel`). |

All CSV and JSON outputs use UTF-8 encoding. The JSON report can be parsed directly in Python with `json.load()` or used as a build artefact in CI to record the design parameters alongside the code that generated them.

---

#### 9.5 Dry-run validation

Add `--dry-run` to validate the config and output path without running the design search:

```bash
iopt-design --config polymer.yml --out ./output/polymer --dry-run
```

The dry run:
- parses and validates the YAML
- checks that the formula is syntactically valid
- verifies that the output directory is writable
- prints a brief summary of what would be run

Exit code is 0 on success, non-zero on validation failure. This makes `--dry-run` useful as a pre-flight check in CI pipelines before committing to a potentially long design search.

---

#### 9.6 Full CLI flag reference

**Core flags:**

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to the YAML or JSON config file. Required for design generation. |
| `--out BASENAME` | Output file basename prefix (default: `design`). |
| `--template MODE` | Print a commented template to stdout and exit. Modes: `contrast`, `r2`, `glm-binomial`, `glm-poisson`. |
| `--dry-run` | Validate and exit without running the search. |
| `-v`, `--verbose` | Enable DEBUG-level logging. |

**Output flags:**

| Flag | Description |
|------|-------------|
| `--excel` | Write a `.xlsx` workbook alongside the CSVs. |
| `--html-report` | Write a self-contained HTML report (requires `[report]` extra). |
| `--robustness-report` | Print a sigma/effect/alpha sensitivity table after the run (single-response only). |

**GLM override flags** (override the corresponding YAML keys):

| Flag | Description |
|------|-------------|
| `--family {binomial,poisson}` | GLM response family. |
| `--link {logit,log}` | GLM link function. |
| `--baseline FLOAT` | Baseline probability (binomial) or count rate (Poisson). |

**Split-plot override flags** (override the `split_plot:` YAML block):

| Flag | Description |
|------|-------------|
| `--htc-factors A,B,...` | Comma-separated hard-to-change factor names. Activates split-plot mode. |
| `--n-whole-plots N` | Number of whole plots. Required with `--htc-factors`. |
| `--eta ETA` | Variance ratio σ²_wp / σ²_sp (default 1.0). |
| `--subplots-per-wp S` | Sub-plots per whole plot. Omit for automatic. |

**Other interface flags** (covered in their respective chapters):

| Flag | Description |
|------|-------------|
| `--sheets URL_OR_ID` | Read from / write to a Google Spreadsheet (Chapter 12). |
| `--sheets-credentials PATH` | Service account JSON for Google Sheets auth. |
| `--excel-template PATH` | Create a starter Excel workbook template (Chapter 11). |
| `--excel-run PATH` | Run from an existing Excel workbook (Chapter 11). |
| `--multi-response` | Treat the config as a multi-response design (inferred automatically when `responses:` is present). |

---

#### 9.7 Defining factors in YAML

**Continuous factors** — a two-element list of numbers:

```yaml
factors:
  Temperature: [50.0, 150.0]    # range [50, 150]
  Concentration: [0.0, 2.0]     # range [0, 2]
```

**Categorical factors** — a YAML list of strings:

```yaml
factors:
  Catalyst: [A, B, C]           # three-level categorical
  Solvent:  [Ethanol, Water]    # two-level categorical
```

When a factor has exactly two numeric elements they are treated as a continuous range. To define a two-level numeric categorical, use strings: `["0", "1"]` or `[low, high]`.

**Scenario-based contrasts.** The `contrast.scenario_a` and `contrast.scenario_b` keys specify two complete factor settings in YAML dict notation. The CLI calls `contrast_from_scenarios` internally and derives L and delta automatically. Use this form whenever possible — it avoids manual column-index counting and is easier to review.

```yaml
contrast:
  scenario_a: {Catalyst: A, Temperature: 80.0,  Concentration: 1.0}
  scenario_b: {Catalyst: A, Temperature: 120.0, Concentration: 1.0}
  sesoi: 0.5    # minimum detectable change in response units
```

The `sesoi` (smallest effect of interest) is expressed in response units. The CLI translates this to a coefficient-scale delta by evaluating the linear predictor difference between the two scenarios. If the two scenarios differ in a categorical factor, see the note in Chapter 3 (Section 3.3) about the categorical anchor limitation — the scenario-based approach requires that both scenarios agree on a reference level or that the contrast direction is unambiguous.

**Explicit L matrix.** For advanced users who need precise control over the contrast, the explicit form is available:

```yaml
contrast:
  L: [[0, 0, 1, 0]]    # one row per hypothesis; p columns
  delta: [0.5]          # one element per row
```

When using explicit L, verify the column order by running `build_model_matrix` in Python first (see Chapter 3, Section 3.2).

---

#### 9.8 Feasibility constraints in YAML

The `constraint_expr` key in the `design:` section applies a filter to the candidate set. It is evaluated as a Python expression with factor column names available as variables:

```yaml
design:
  auto_candidate: true
  random_state: 42
  constraint_expr: "Temperature + Pressure <= 150.0"
```

Available math helpers: `sqrt`, `log`, `log10`, `log2`, `exp`, `floor`, `ceil`, `pi`, `abs`, `min`, `max`, `round`. No imports are permitted — the expression must be self-contained.

For multi-factor linear constraints, compound expressions work naturally:

```yaml
constraint_expr: "Temperature >= 60.0 and Pressure <= Temperature / 2.0"
```

The constraint is applied once during candidate generation. It does not slow down the Fedorov exchange itself; the exchange only selects from points that already passed the filter.

---

#### 9.9 Full worked example

**Scenario.** A polymer team wants a fully reproducible YAML-driven pipeline they can commit to Git and re-run at any time. The study has two coded factors (Temperature and Pressure, both [−1, +1]) and a split-plot structure: Temperature is hard-to-change (set once per whole plot), Pressure is easy to change between sub-runs.

**Step 1 — Create the config.**

```bash
iopt-design --template contrast > polymer_sp.yml
```

Edit the generated file to match the study. The final `polymer_sp.yml`:

```yaml
# polymer_sp.yml — Split-plot powered design for polymer study

formula: "~ 1 + Temp + Press + Temp:Press"

factors:
  Temp:  [-1.0, 1.0]
  Press: [-1.0, 1.0]

# p = 4: [Intercept, Temp, Press, Temp:Press]
# Test: Temp main effect
contrast:
  L: [[0, 1, 0, 0]]
  delta: [0.5]

alpha: 0.05
power: 0.80
sigma: 1.0

design:
  auto_candidate: true
  starts: 5
  criterion: I
  random_state: 42
  workers: null

output:
  basename: polymer_sp

split_plot:
  htc_factors: [Temp]
  n_whole_plots: 8
  eta: 1.0             # equal whole-plot and sub-plot variance
  df_method: auto
```

**Step 2 — Validate before running.**

```bash
iopt-design --config polymer_sp.yml --out ./output/polymer_sp --dry-run
```

Output:

```
--- Dry Run Validation Successful ---
  Formula: ~ 1 + Temp + Press + Temp:Press
  Factors: ['Temp', 'Press']
  Power Config: PowerContrastConfig
  Design Algo: fedorov
   Output Dir: /path/to/output (writable)
--- Exiting without design generation ---
```

**Step 3 — Run the design.**

```bash
iopt-design --config polymer_sp.yml --out ./output/polymer_sp --html-report
```

This produces:

```
./output/polymer_sp_design.csv
./output/polymer_sp_buckets.csv
./output/polymer_sp_report.json
./output/polymer_sp_report.html
```

**Step 4 — Wire into a Makefile.**

A minimal `Makefile` target that validates before running and only re-runs when the config changes:

```makefile
# Makefile

OUTPUT_DIR := output
CONFIG     := polymer_sp.yml

.PHONY: design validate clean

validate:
	iopt-design --config $(CONFIG) --out $(OUTPUT_DIR)/polymer_sp --dry-run

design: $(OUTPUT_DIR)/polymer_sp_design.csv

$(OUTPUT_DIR)/polymer_sp_design.csv: $(CONFIG)
	@echo "Running design generation..."
	iopt-design --config $(CONFIG) \
	            --out $(OUTPUT_DIR)/polymer_sp \
	            --html-report \
	            --verbose
	@echo "Design written to $(OUTPUT_DIR)/"

clean:
	rm -f $(OUTPUT_DIR)/polymer_sp_*.csv \
	      $(OUTPUT_DIR)/polymer_sp_*.json \
	      $(OUTPUT_DIR)/polymer_sp_*.html
```

With this setup:

- `make validate` — checks the config is parseable and the output directory is writable, without spending time on the search.
- `make design` — runs only if `polymer_sp.yml` has changed since the last run (Make's dependency tracking).
- `make clean` — removes generated artefacts without touching the config.

**GitHub Actions equivalent.** For CI, the same logic works as a workflow step:

```yaml
# .github/workflows/design.yml
- name: Validate design config
  run: iopt-design --config polymer_sp.yml --out ./output/polymer_sp --dry-run

- name: Generate design
  run: |
    iopt-design --config polymer_sp.yml \
                --out ./output/polymer_sp \
                --html-report
  # Upload the design artefacts
- name: Upload design outputs
  uses: actions/upload-artifact@v4
  with:
    name: design-outputs
    path: output/polymer_sp_*
```

The dry-run step acts as a config linter, catching YAML errors and path issues before the potentially long design search runs.

---

### Chapter 10 — Streamlit web UI: interactive design without coding

The Streamlit app provides the same I-optimal, power-assured design capability as the Python API and CLI, but through a point-and-click web interface. No Python knowledge is required to use it.

---

#### 10.1 What the app is and what it is not

**Who it is for.** The Streamlit app is designed for:

- Domain experts and experimenters who need a correct powered design but do not write Python
- Collaborators and reviewers who want to explore assumptions interactively (change σ, adjust α, see the power curve update)
- Rapid prototyping before committing a design to a YAML config or Python script

**What it supports.** The app exposes the full range of standard design capabilities:

- All three power modes: contrast-based, global R², and GLM (binomial/Poisson)
- Multi-response designs (the `responses:` path from Chapter 6)
- Split-plot and blocked designs
- Sensitivity analysis, minimum detectable effect (MDE), and criteria comparison
- CSV, JSON, Excel, and HTML report downloads

**What it does not support.** A small number of capabilities are Python-API-only:

- Power curves **by baseline** (`power_curve_by_baseline`) and **by whole-plot variance** (`power_curve_by_wp`) — the UI exposes power curves by n and by effect size only
- **Multi-response power curves and sensitivity** — `power_curve_by_n_multiresponse` and `multiresponse_sensitivity` require the Python API (the UI notes this explicitly when a multi-response result is present)
- **Feasibility constraints** via `constraint_func` (the callable form) — use `constraint_expr` (YAML string) for CLI or restrict to Python for callable constraints

**The four-page flow.** The app is structured as a linear four-step wizard:

```
1 · Factors  →  2 · Power Config  →  3 · Run & Results  →  4 · Analysis
```

Session state is preserved as you navigate between pages — you can go back to Page 1, adjust a factor, and then return to Page 3 to re-run without re-entering everything. Use the sidebar to jump between pages at any point.

---

#### 10.2 Launching the app

**Local run.** Install the `app` extras group and start Streamlit:

```bash
pip install -e ".[app]"
streamlit run app/app.py
```

The app opens automatically in your browser at `http://localhost:8501`. The home page shows a four-column overview of the workflow and a quick-reference expander that summarises factor types, formula syntax, and criteria.

**From the project root.** The `app/` directory is added to `sys.path` by Streamlit, so all pages can import from `state.py` and `components/` without any path manipulation. Run only from the project root.

---

#### 10.3 Page-by-page walkthrough

---

**Page 1 — Factors & Formula**

This page defines the experimental factors and the Patsy model formula.

*Adding factors.* Click **Add factor** to insert a new row into the factor table. For each factor, specify:

- **Name** — the column name that will appear in the design table (e.g. `Temperature`, `Catalyst`)
- **Type** — `Continuous` or `Categorical`
- **Range / Levels** — for continuous factors, enter the numeric low and high bounds; for categorical factors, enter a comma-separated list of level names

Factors persist in session state. Navigating away and back does not clear them. The **Clear all factors** button resets all factors and the formula to defaults without affecting the power configuration on Page 2.

*Model formula.* The formula input below the factor table accepts standard Patsy notation. As you type, the page evaluates the formula against the current factors and displays the resulting model parameter count **p** live. This p value is the number you need when constructing a contrast matrix L on Page 2.

```
~ 1 + A + B + A:B   →  p = 4  (Intercept, A[T.high], B, A[T.high]:B)
```

The formula must be consistent with the factors defined above — Patsy will raise an error if a factor name in the formula does not appear in the factor table.

---

**Page 2 — Power Configuration & Design Options**

This page specifies what the design should be powered to detect and how the search should be run.

*Power mode.* A radio selector at the top switches between three modes:

| Mode | UI label | Underlying config |
|------|----------|-------------------|
| Contrast-based | "Contrast-based" | `PowerContrastConfig` |
| Global R² | "Global R²" | `PowerR2Config` |
| GLM | "GLM (logistic/Poisson)" | `PowerGLMContrastConfig` |

*Contrast-based mode.* Two sub-options appear:

- **Scenario-based** (recommended): enter values for all factors in Scenario A and Scenario B, then specify the SESOI (smallest effect of interest in response units). The app calls `contrast_from_scenarios` automatically and displays the derived L matrix and δ for inspection before running.
- **Matrix mode** (advanced): paste the L matrix and δ vector directly as space- or comma-separated text.

The current p (from Page 1) is shown as a reminder when entering L in matrix mode, reducing the risk of column-count mismatches.

*Global R² mode.* A single numeric input for `r2_target` (the minimum R² to detect), plus controls for `alpha`, `power`, and `lambda_mode` (`"n"` or `"n_minus_p"`).

*GLM mode.* Family (binomial / Poisson) and baseline (p₀ or μ₀) selectors appear. The contrast can again be entered in scenario or matrix form. The Fisher weight is computed and displayed so you can verify the effective effect size before running.

*Common power settings.* Below the mode-specific section, all modes share:

| Control | Default | Description |
|---------|---------|-------------|
| α | 0.05 | Significance level |
| Target power | 0.80 | Minimum acceptable power |
| σ | 1.0 | Residual standard deviation (contrast/GLM modes) |
| Max n | 500 | Upper bound for the binary search |

*Design options.* A collapsible section exposes the `DesignOptions` fields: criterion (I/D/A), starts, random state, auto-candidate toggle, and workers. Defaults are appropriate for most studies.

*Advanced design structures.* Two toggles appear below design options:

- **Multi-response mode** — enables the responses list. When active, Page 3 calls `i_optimal_multiresponse_design` instead of `i_optimal_powered_design`. Each response gets its own power mode, σ, and contrast specification.
- **Split-plot design** — exposes `htc_factors`, `n_whole_plots`, `eta`, `subplots_per_wp`, and `df_method`. When active, the split-plot exchange algorithm is used (Chapter 15).

A **Number of blocks** input beneath the advanced toggles activates blocked design mode (Chapter 16). Split-plot and blocking cannot be active simultaneously; the page warns if both are enabled.

---

**Page 3 — Run & Results**

*Running the design.* The large **Generate design** button at the top of the page triggers the search. A spinner is displayed during the run. The search runs synchronously in the Streamlit server process; for long runs (high `max_n`, many starts, complex models), expect to wait. The run time is reported after completion.

*Power summary.* After a successful run, a summary card displays:

- **n** — minimum run count found
- **Achieved power** — power of the returned design
- **λ** (noncentrality) — for single-response contrast and R² modes
- **df** — numerator and denominator degrees of freedom
- **Criterion** and **Search strategy** used
- **Elapsed time**

For multi-response results, per-response powers are shown alongside the combined power and combination rule.

*Design and buckets tables.* The design DataFrame and the factor-level bucket counts are displayed in scrollable tables directly on the page.

*Downloading results.* The Export section at the bottom of the page provides:

| Download | File | Notes |
|----------|------|-------|
| Design CSV | `design.csv` | Run table (n × factors) |
| Buckets CSV | `buckets.csv` | Factor-level bucket counts |
| Full report JSON | `report.json` | Complete `result["report"]` dict |
| HTML report | `design_report.html` | Self-contained; requires `[report]` extra |
| Excel workbook | `design.xlsx` | Design, Buckets, Report sheets; requires `xlsxwriter` |

The HTML report download button is present regardless of whether the `[report]` extra is installed; it shows an informational message if `jinja2` is missing.

*Power curve.* A collapsible expander on Page 3 shows an approximate power-vs-n curve (analytical noncentral-F approximation) for contrast-based and R² single-response designs. This is an approximation computed without re-running the design search at each n; for an accurate sweep use Page 4 or `power_curve_by_n` from the Python API.

For **GLM** results, the expander notes that the noncentral-F approximation is not valid for Wald χ² tests and directs users to Page 4.

For **multi-response** results, the expander notes that the per-response power curve requires `power_curve_by_n_multiresponse` from the Python API.

---

**Page 4 — Advanced Analysis & Export**

This page provides post-design analysis tools. It requires a completed run on Page 3.

*Export configuration (top of page).* A **Download YAML config** button generates a CLI-compatible `config.yml` representing the current session state. This is the recommended bridge between the Streamlit UI and reproducible file-based pipelines: explore interactively in the UI, then export the YAML and commit it to version control for reproducibility.

*F1 — Sensitivity analysis.* Sweeps σ (for contrast mode) or R² target (for R² mode) across a range, building a new design at each point and reporting achieved power. The resulting table is displayed and can be downloaded as CSV. The sweep runs the full design search at each point, so it can be slow for large `max_n`.

*F2 — Minimum detectable effect (MDE).* Given a fixed design (the one returned on Page 3), reports the smallest effect size detectable at a specified power level. For contrast mode, MDE is expressed as a multiplier of the δ vector; for R² mode, as a minimum detectable R².

*F3 — Compare optimality criteria.* Runs `compare_criteria` for the current formula, factors, and power config under all three criteria (I, D, A). Displays the summary table comparing n, achieved power, d_efficiency, and condition number. A bar chart is shown inline.

*F4 — Split-plot η sensitivity.* Appears only when the Page 3 result is a split-plot design. Sweeps the variance ratio η (σ²_wp / σ²_sp) and shows how the achieved power changes as the whole-plot variance assumption changes. Useful for understanding how sensitive the sample size is to the variance ratio estimate.

---

#### 10.4 Deploying to Streamlit Community Cloud

Streamlit Community Cloud (share.streamlit.io) provides free hosting for public GitHub repositories, with no server setup required.

**Steps:**

1. Push the repository to GitHub (or fork it if you are working from a public repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. Click **New app**.
4. Select your repository and branch.
5. Set **Main file path** to `app/app.py`.
6. Click **Deploy**.

The app will be live at `https://<your-app-name>.streamlit.app` within a few minutes.

**No secrets or environment variables are required** for the core app. The `[app]` dependencies (`streamlit`, `plotly`) are listed in `requirements.txt` (or `pyproject.toml`), which Streamlit Cloud reads automatically. The Google Sheets integration (Chapter 12) requires a service account credentials file, which should be stored as a Streamlit secret rather than committed to the repository.

**Resource limits.** Streamlit Community Cloud free tier has memory and CPU limits. For designs with many factors, high `max_n`, or many starts, the cloud run may time out or run slowly. For production use with resource-intensive designs, a self-hosted deployment (Docker or a cloud VM) is more reliable.

---

#### 10.5 Docker deployment

The repository includes a `Dockerfile` that builds a self-contained image:

```bash
# Build (from the project root)
docker build -t iopt-doe .

# Run — app available at http://localhost:8501
docker run -p 8501:8501 iopt-doe
```

The default image installs all extras (`[app]`, `[report]`, `[extras]`) so the full feature set including HTML reports and Excel download is available.

**Customising for restricted environments.** If your environment has no internet access at build time, copy the package wheel and install from the local copy:

```dockerfile
# Add to Dockerfile before the pip install step
COPY iopt_power_design-*.whl /tmp/
RUN pip install /tmp/iopt_power_design-*.whl[app,report,extras]
```

**Running with a service account for Sheets.** Mount the credentials file at runtime rather than baking it into the image:

```bash
docker run -p 8501:8501 \
  -v /path/to/credentials.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  iopt-doe
```

---

#### 10.6 Full worked example

**Scenario.** A regulatory statistician at a pharmaceutical company needs to design a dose-response study for a new formulation. The response is binary (patient responds: yes/no), so the study requires a GLM (binomial/logistic) powered design. The statistician does not write Python but is comfortable with a web interface.

The study has two factors:
- **Dose** — continuous, range [1, 10] mg/kg
- **Age** — continuous covariate, range [18, 65] years

The model includes both main effects: `~ 1 + Dose + Age`. The minimum detectable effect is an increase in log-odds of 1.0 (corresponding approximately to doubling the odds ratio), tested on the Dose main effect. Baseline probability p₀ = 0.20. Target: 80% power at α = 0.05.

---

**Step 1: Factors (Page 1)**

Open the app and navigate to **1 · Factors**.

Click **Add factor** twice to create two rows:

| Name | Type | Range |
|------|------|-------|
| Dose | Continuous | Low: 1.0 / High: 10.0 |
| Age | Continuous | Low: 18.0 / High: 65.0 |

In the formula box, enter:

```
~ 1 + Dose + Age
```

The page immediately shows **p = 3** (Intercept, Dose, Age), confirming the model matrix has three columns.

---

**Step 2: Power configuration (Page 2)**

Navigate to **2 · Power Config**.

Select **GLM (logistic/Poisson)** as the power mode.

In the GLM Specification section:
- Family: **Binomial (logistic)**
- Baseline: **0.20** (20% baseline response probability)

For the contrast, choose **Matrix mode** and enter:
- L: `0 1 0` (tests the Dose main effect, the second column)
- δ: `1.0` (minimum detectable log-odds change)

The page shows the effective Fisher weight w = p₀(1 − p₀) = 0.20 × 0.80 = **0.16** and the noncentrality contribution λ = w × δ² = 0.16 for a reminder of the scale.

Common power settings:
- α = 0.05, target power = 0.80, σ = 1.0 (placeholder — ignored for GLM), max n = 300

Design options: auto candidate, starts = 5, criterion = I, random state = 42.

---

**Step 3: Run & Results (Page 3)**

Navigate to **3 · Run & Results** and click **Generate design**.

After the search completes (~20–40 seconds), the summary shows:

```
n = 139    achieved power = 0.8007    λ = 7.8629    df = (1, 136)
Criterion: I    Search strategy: bisection+verification    Elapsed: ~30s
```

The design table shows 139 rows with Dose and Age columns. As expected for I-optimal GLM designs, runs cluster at the extreme Dose values (near 1 and near 10 mg/kg), since extreme values provide the most information about the dose-response slope.

Click **Download design CSV** to save the run table. Click **Download HTML report** to save a self-contained report suitable for sharing with collaborators.

In the Export section, click **Download YAML config** to capture the full study specification as a CLI-compatible `config.yml`. This file can be committed to the study's version-control repository for a reproducible record.

---

**Step 4: Analysis (Page 4)**

Navigate to **4 · Analysis**.

*Sensitivity sweep.* Under **F1 · Sensitivity Analysis**, run a σ sweep. For GLM mode, the app sweeps the effect size δ (as a scale factor on the baseline δ) rather than σ. Adjust the sweep range to 0.5×–2.0× and click **Run sensitivity**. The table shows:

| δ scale | Achieved power |
|---------|----------------|
| 0.5 | ~0.42 |
| 1.0 | ~0.80 (design point) |
| 1.5 | ~0.97 |
| 2.0 | ~1.00 |

This confirms that the design is appropriately powered at the specified δ = 1.0, and that power drops sharply if the actual effect is smaller than assumed.

*MDE.* Under **F2 · Minimum Detectable Effect**, set target power = 0.80 and click **Compute MDE**. The result reports the minimum detectable log-odds change on the current 139-run design — if the true effect is above this threshold, the study has ≥ 80% power to detect it.

*Criteria comparison.* Under **F3 · Compare Optimality Criteria**, click **Run comparison**. For this GLM main-effects model, all three criteria (I, D, A) return similar n values with d_efficiency close to 1.0, confirming that the criterion choice is not material for this simple model structure.

---

> **Tip — bridging UI and reproducible pipelines.** The YAML export on Page 4 produces a config that runs identically with `iopt-design --config study.yml --out ./output/study`. This makes the Streamlit UI a useful design-exploration tool even for users who ultimately prefer reproducible CLI pipelines: explore in the UI, lock in the assumptions, export the YAML, then run from the CLI for the production record.

---

### Chapter 11 — Excel: spreadsheet-driven workflows

The Excel interface lets a team member who works entirely in spreadsheets configure and run a powered design without writing any Python. The statistician creates a template workbook from Python (or the CLI), hands it to the experimenter to fill in, and then runs the design from Python (or the CLI) against the completed file. Results are written back into the same workbook as new sheets.

---

#### 11.1 When to use the Excel interface

**Use Excel when:**

- The experimenter who defines factors and power assumptions works in Excel, not Python
- The study configuration needs to be shared with collaborators or stakeholders as a self-contained file (no Python environment, no YAML editor)
- You want study inputs and outputs in one auditable `.xlsx` file for archiving or regulatory documentation

**Limitations.** The Excel interface supports all four power modes (contrast, R², GLM, multi-response) and basic design options. A few advanced features are Python-API-only:

- Feasibility constraints (`constraint_expr` / `constraint_func`) — not supported in the Config sheet
- Progress callbacks — not available from Excel
- Post-design analysis functions (`power_sensitivity`, `compare_criteria`, etc.) — run these from Python against the returned result dict

---

#### 11.2 Installing Excel support

The Excel interface requires `openpyxl` for reading and writing `.xlsx` files:

```bash
pip install -e ".[extras]"
```

The `extras` group includes `openpyxl`, `xlsxwriter`, and the Google Sheets dependencies. If you only need Excel, `pip install openpyxl` also works.

---

#### 11.3 The Config sheet structure

The workbook has one input sheet (`Config`) and up to three output sheets (`Results`, `Design`, `Buckets`). The `Config` sheet uses **sentinel headers** — special values in column A that mark the start of each section. The parser scans column A top-to-bottom for these markers:

| Sentinel | Section | Required? |
|----------|---------|-----------|
| `[SETTINGS]` | Key/value configuration pairs | Always |
| `[CONTRAST]` | L matrix and δ vector | When `power_mode` is `contrast` or `glm` |
| `[FACTORS]` | Factor definitions table | Always |
| `[RESPONSES]` | Per-response specs (multi-response) | When using multi-response mode |

The sections can appear in any order in the sheet. The parser finds them by scanning rather than assuming fixed row positions, so you can add blank rows, comments, or formatting between sections without breaking the parser.

---

**`[SETTINGS]` section.** Key/value pairs in columns A (key) and B (value). The keys below are recognised; any key not listed is silently ignored.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `formula` | string | required | Patsy formula (e.g. `~ 1 + A + B + A:B`) |
| `power_mode` | `r2`, `contrast`, or `glm` | required | Power mode. Omit when `[RESPONSES]` is present. |
| `alpha` | float | 0.05 | Significance level |
| `power` | float | 0.80 | Target power |
| `sigma` | float | 1.0 | Residual SD (contrast mode; ignored for R²) |
| `r2_target` | float | 0.25 | Minimum R² to detect (R² mode only) |
| `max_n` | int | 500 | Upper bound for binary search |
| `criterion` | `I`, `D`, or `A` | `I` | Optimality criterion |
| `starts` | int | 5 | Number of random starts |
| `max_iter` | int | 1000 | Maximum iterations per start |
| `random_state` | int | 123 | Random seed |
| `family` | `binomial` or `poisson` | `binomial` | GLM response family (GLM mode) |
| `link` | `logit` or `log` | (canonical) | GLM link function; blank = canonical default |
| `baseline` | float | required for GLM | Baseline p₀ (binomial) or μ₀ (Poisson) |
| `n_blocks` | int | 0 | Number of blocks (0 = unblocked; ≥ 2 to enable) |
| `block_factor_name` | string | `Block` | Column name for the block indicator |
| `htc_factors` | comma-separated strings | blank | Hard-to-change factor names (split-plot) |
| `n_whole_plots` | int | 0 | Number of whole plots (0 = disabled; ≥ 2 to enable) |
| `eta` | float | 1.0 | Variance ratio σ²_wp / σ²_sp (split-plot) |
| `subplots_per_wp` | int | 0 | Sub-plots per whole plot (0 = auto) |
| `df_method` | `auto`, `conservative`, or `sp_only` | `auto` | Denominator df method (split-plot) |

The template applies dropdown validation to the `power_mode` and `criterion` cells, so these fields show an in-cell picker in Excel.

---

**`[CONTRAST]` section.** Used when `power_mode` is `contrast` or `glm`. Rows in this section use the key format:

| Key | Value |
|-----|-------|
| `L_row` | Comma-separated float values for one row of the L matrix (one row per hypothesis) |
| `delta` | Comma-separated float values, one per L row |

Example for a 4-column model, testing the third coefficient:

```
[CONTRAST]
L_row    0, 0, 1, 0
delta    0.5
```

For a two-hypothesis joint test:

```
[CONTRAST]
L_row    0, 1, 0, 0
L_row    0, 0, 1, 0
delta    0.5, 0.5
```

---

**`[FACTORS]` section.** A table with a header row (`Name | Type | Value 1 | Value 2 | ...`) followed by one row per factor.

| Column | Description |
|--------|-------------|
| Name | Factor name (must match names used in the formula) |
| Type | `continuous` or `categorical` |
| Value 1, Value 2, ... | For continuous: low and high bounds; for categorical: level names (as many columns as levels) |

Example:

```
[FACTORS]
Name          Type          Value 1    Value 2    Value 3
Temperature   continuous    50         150
Pressure      continuous    1          5
Catalyst      categorical   A          B          C
```

---

**`[RESPONSES]` section.** Optional; activates multi-response mode. Contains a header row followed by one row per response. Key/value rows for `power_combination` and `sigma_joint` can also appear here. The column layout per response row:

| Col | Field | Description |
|-----|-------|-------------|
| 1 | name | Response name |
| 2 | power_mode | `contrast`, `r2`, or `glm` |
| 3 | sigma | Residual SD |
| 4 | alpha | Significance level |
| 5 | power | Target power |
| 6 | weight | Weight for `weighted_mean` combination |
| 7 | L_row | Comma-separated L row (contrast/GLM mode) |
| 8 | delta | Comma-separated δ values |
| 9 | r2_target | R² target (R² mode) |
| 10 | formula | Per-response formula override (optional) |

Special rows (in column A, value in column B):

| Key | Value |
|-----|-------|
| `power_combination` | `min`, `product`, or `weighted_mean` |
| `sigma_joint` | Semicolon-separated matrix rows, comma-separated values |

---

#### 11.4 Creating a template workbook: `create_excel_template`

`create_excel_template` writes a new `.xlsx` workbook pre-populated with a runnable example in the chosen power mode:

```python
from iopt_power_design import create_excel_template

# Contrast-mode template
path = create_excel_template("study_template.xlsx", example="contrast")
print(f"Template written to: {path}")
```

The `example` parameter chooses which pre-filled values to use:

| Value | Description |
|-------|-------------|
| `"r2"` | Global R² mode with two continuous factors |
| `"contrast"` | Contrast mode with L matrix and δ for two continuous factors |
| `"multiresponse"` | Two-response joint design with `[RESPONSES]` section |
| `"glm-binomial"` | GLM binomial (logistic) mode |
| `"glm-poisson"` | GLM Poisson (log-linear) mode |

The created workbook contains:
- A **Config** sheet filled with the example configuration, with dropdown validation on `power_mode` and `criterion`
- Empty placeholder **Results**, **Design**, and **Buckets** sheets that `excel_run` will populate

**Via the CLI.** `create_excel_template` is also accessible through the CLI without writing Python:

```bash
# Create a contrast-mode starter workbook
iopt-design --excel-template study_template.xlsx --template-mode contrast
```

---

#### 11.5 Running the design: `excel_run`

`excel_run` reads the `Config` sheet, runs the design search, and writes three output sheets back into the same workbook:

```python
from iopt_power_design import excel_run

result = excel_run("study_template.xlsx")
```

After the call, the workbook at `study_template.xlsx` now contains three new (or updated) sheets:

| Sheet | Contents |
|-------|----------|
| **Results** | Key/value summary: n, achieved power, λ, df, criterion, elapsed time, search strategy, warnings |
| **Design** | Full design DataFrame (n rows × factor columns) |
| **Buckets** | Factor-level bucket counts |

The function returns the same result dict as `i_optimal_powered_design` (or `i_optimal_multiresponse_design` for multi-response configs), with an additional `"excel_path"` key containing the absolute path of the updated workbook.

```python
rep = result["report"]
print(f"n={rep['n']}, power={rep['achieved_power']:.4f}")
print(f"Updated workbook: {result['excel_path']}")
```

**Via the CLI:**

```bash
iopt-design --excel-run study_template.xlsx
```

This is the fully no-Python workflow: create the template with `--excel-template`, fill it in Excel, run with `--excel-run`. No Python script required.

**Overriding design options.** For programmatic control without modifying the workbook file, pass `design_opts_override`:

```python
from iopt_power_design import excel_run, DesignOptions

result = excel_run(
    "study_template.xlsx",
    design_opts_override=DesignOptions(starts=20, workers=4, random_state=99),
)
```

The override replaces the design options read from the Config sheet entirely; power configuration and factors are still read from the file.

**Error handling.** All Excel integration errors raise `ExcelError` (a subclass of `RuntimeError`). The underlying cause is attached as `__cause__`. Common error categories:

| Situation | Error message |
|-----------|---------------|
| Missing `[SETTINGS]` sentinel | `"Config sheet is missing the '[SETTINGS]' sentinel."` |
| Missing `[FACTORS]` sentinel | `"Config sheet is missing the '[FACTORS]' sentinel."` |
| Missing `[CONTRAST]` when required | `"power_mode is 'contrast' but the '[CONTRAST]' sentinel is missing."` |
| Non-numeric bound for a continuous factor | `"Factor 'Temperature' continuous bounds must be numeric."` |
| Wrong `power_mode` value | `"power_mode must be 'r2', 'contrast', or 'glm'."` |
| Design search failure | `"Design search failed: <underlying error>"` |

Catch `ExcelError` explicitly in scripts that run unattended:

```python
from iopt_power_design import excel_run, ExcelError

try:
    result = excel_run("study_template.xlsx")
except ExcelError as e:
    print(f"Excel error: {e}")
    if e.__cause__:
        print(f"  Caused by: {e.__cause__}")
    raise
```

---

#### 11.6 Full worked example

**Scenario.** A process engineer at an industrial manufacturing site is planning a three-factor study on a chemical reactor: Temperature (50–150°C), Pressure (1–5 bar), and Feed Rate (0.5–2.0 L/min), all continuous. The model includes all three main effects and two interactions: Temperature:Pressure and Temperature:FeedRate. The statistician needs the engineer to sign off on the factor ranges and effect assumptions before the study is approved, so the configuration is shared as an Excel file.

**Step 1 — Create the template.**

```python
from iopt_power_design import create_excel_template

create_excel_template("reactor_study.xlsx", example="contrast")
```

Open `reactor_study.xlsx` in Excel. The Config sheet looks like this (key entries shown):

```
[SETTINGS]
formula         ~ 1 + A + B
power_mode      contrast        ← dropdown: r2 / contrast / glm
alpha           0.05
power           0.80
sigma           1.0
max_n           500
criterion       I               ← dropdown: I / D / A
starts          5
random_state    123

[CONTRAST]
L_row           0, 0, 1, 0
delta           0.5

[FACTORS]
Name    Type         Value 1   Value 2
A       continuous   0.0       10.0
B       continuous   0.0       10.0
```

**Step 2 — Edit the file in Excel.**

The engineer edits the file to match the reactor study:

```
[SETTINGS]
formula         ~ 1 + Temp + Press + FeedRate + Temp:Press + Temp:FeedRate
power_mode      contrast
alpha           0.05
power           0.80
sigma           1.0
max_n           400
criterion       I
starts          5
random_state    42

[CONTRAST]
L_row           0, 1, 0, 0, 0, 0
delta           0.5

[FACTORS]
Name        Type         Value 1   Value 2
Temp        continuous   50.0      150.0
Press       continuous   1.0       5.0
FeedRate    continuous   0.5       2.0
```

The formula `~ 1 + Temp + Press + FeedRate + Temp:Press + Temp:FeedRate` produces p = 6 model columns: `[Intercept, Temp, Press, FeedRate, Temp:Press, Temp:FeedRate]`. The L row `0, 1, 0, 0, 0, 0` tests the Temp main effect.

**Step 3 — Run the design.**

The statistician receives the file and runs it from Python:

```python
from iopt_power_design import excel_run

result = excel_run("reactor_study.xlsx")

rep = result["report"]
print(f"n = {rep['n']}")
print(f"achieved power = {rep['achieved_power']:.4f}")
print(f"λ = {rep['noncentrality_lambda']:.4f}")
print(f"Workbook updated: {result['excel_path']}")
```

Or equivalently from the CLI (fully no-Python):

```bash
iopt-design --excel-run reactor_study.xlsx
```

After the run, `reactor_study.xlsx` contains three new sheets:

- **Results** — `n`, `achieved_power`, `noncentrality_lambda`, `elapsed_sec`, `criterion`, and all other report fields
- **Design** — the run table with 84 rows and columns `Temp`, `Press`, `FeedRate`
- **Buckets** — run frequency by factor-level bucket

**Step 4 — The engineer reviews the workbook.**

The engineer opens `reactor_study.xlsx`, looks at the Design sheet to see the run schedule, and checks the Results sheet to confirm the study is powered as agreed. No Python installation is required on the engineer's machine — the output is self-contained in the workbook.

---

**Comparison: same design via Python API.**

For reference, the equivalent Python API call for the same study:

```python
from iopt_power_design import (
    i_optimal_powered_design,
    PowerContrastConfig,
    DesignOptions,
)

formula = "~ 1 + Temp + Press + FeedRate + Temp:Press + Temp:FeedRate"
factors = {
    "Temp":     (50.0, 150.0),
    "Press":    (1.0,  5.0),
    "FeedRate": (0.5,  2.0),
}
# p = 6: [Intercept, Temp, Press, FeedRate, Temp:Press, Temp:FeedRate]
power_cfg = PowerContrastConfig(
    L=[[0, 1, 0, 0, 0, 0]],   # Temp main effect
    delta=[0.5],
    alpha=0.05,
    power=0.80,
    sigma=1.0,
    max_n=400,
)
opts = DesignOptions(auto_candidate=True, starts=5, random_state=42)

result = i_optimal_powered_design(formula, factors, power_cfg, opts)
print(f"n={result['report']['n']}, power={result['report']['achieved_power']:.4f}")
```

The Excel and Python API paths produce the same design (given the same `random_state` and `starts`), because `excel_run` delegates to `i_optimal_powered_design` internally. The Excel interface is simply a structured input/output layer around the same search engine.

---

### Chapter 12 — Google Sheets: collaborative cloud-based workflows

The Google Sheets interface works like the Excel interface (Chapter 11) in its structure and configuration syntax, but operates in the cloud. The Config sheet lives in a Google Spreadsheet, multiple team members can edit it simultaneously from anywhere, and results are written back to the same spreadsheet — visible to everyone who has access, no file transfer required.

---

#### 12.1 When to use the Sheets interface

**Use Google Sheets when:**

- The team is geographically distributed and needs a shared, always-current configuration
- Stakeholders who approve study parameters prefer a browser-based tool over email attachments
- The organisation uses Google Workspace and the spreadsheet can be linked from Docs, Data Studio, or other G-Suite tools
- You want a live audit trail: Google Sheets tracks every cell edit with a timestamp and author

**Compared to Excel.** The Config sheet syntax is identical — the same four sentinels (`[SETTINGS]`, `[CONTRAST]`, `[FACTORS]`, `[RESPONSES]`), the same key names, the same factor table format. If you are already familiar with the Excel interface from Chapter 11, the only new concepts are authentication and the cloud-sharing model.

**Limitations.** The same Python-API-only features that are absent from Excel are also absent here: feasibility constraints, progress callbacks, and post-design analysis functions. Feasibility constraints can be added in a post-processing Python step if needed.

---

#### 12.2 Installing Sheets support

The Sheets interface requires `gspread` and `google-auth`:

```bash
pip install -e ".[sheets]"
```

The `sheets` extras group installs `gspread` and its Google authentication dependencies. If you use the combined `extras` group (`pip install -e ".[extras]"`), Sheets support is included.

---

#### 12.3 Authentication

Two authentication modes are supported. Choose based on whether you are running interactively or in an automated/CI context.

---

**Option A — OAuth2 browser flow (interactive use)**

Pass `credentials=None` (the default). On the first call, `gspread.oauth()` opens a browser tab and asks you to sign in to your Google account. After approval, the access token is cached in `~/.config/gspread/` and reused on subsequent calls — you only authenticate in the browser once per machine.

```python
from iopt_power_design import sheets_run

# credentials=None → OAuth2 browser flow on first use
result = sheets_run("https://docs.google.com/spreadsheets/d/YOUR_ID")
```

This mode is convenient for local development and one-off runs. It requires a browser and is not suitable for headless CI environments.

---

**Option B — Service account (automation and CI)**

Create a service account in the Google Cloud Console, download its JSON credentials file, and pass the file path to `credentials`. The spreadsheet must be **shared with the service account's email address** (the `client_email` field in the JSON file).

```python
result = sheets_run(
    "https://docs.google.com/spreadsheets/d/YOUR_ID",
    credentials="path/to/service_account.json",
)
```

**Setting credentials from an environment variable.** For CI pipelines, store the credentials path in an environment variable rather than hard-coding it:

```python
import os
from iopt_power_design import sheets_run

creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
result = sheets_run(spreadsheet_url, credentials=creds)
```

**Security notes.**

- Never commit a service account JSON file to a public repository. Add `*.json` (or the specific filename) to `.gitignore`.
- In GitHub Actions or other CI systems, store the JSON contents as a secret and write it to a temp file at runtime:

```yaml
# .github/workflows/design.yml
- name: Write credentials
  run: echo '${{ secrets.GCLOUD_CREDENTIALS }}' > /tmp/sa.json

- name: Run design
  run: python run_design.py
  env:
    GOOGLE_APPLICATION_CREDENTIALS: /tmp/sa.json
```

---

#### 12.4 Creating a template spreadsheet: `create_sheet_template`

`create_sheet_template` creates a new Google Spreadsheet pre-populated with a runnable example and returns its URL:

```python
from iopt_power_design import create_sheet_template

url = create_sheet_template(
    title="Consumer survey — powered design",
    credentials="path/to/service_account.json",
    example="r2",
)
print(f"Spreadsheet created: {url}")
```

The new spreadsheet contains four sheets:

| Sheet | Initial contents |
|-------|-----------------|
| **Config** | Pre-filled with the chosen example |
| **Results** | Empty — populated by `sheets_run` |
| **Design** | Empty — populated by `sheets_run` |
| **Buckets** | Empty — populated by `sheets_run` |

The `example` parameter accepts the same values as `create_excel_template`: `"r2"`, `"contrast"`, `"multiresponse"`, `"glm-binomial"`, `"glm-poisson"`.

**Sharing the spreadsheet.** By default the new spreadsheet is not shared publicly. To grant link-based write access to anyone (useful for quick prototyping but not for private data):

```python
url = create_sheet_template(
    title="Pilot study config",
    credentials=creds,
    example="contrast",
    share_anyone=True,     # grant writer access to anyone with the link
)
```

For production use, share the spreadsheet explicitly with specific Google accounts via the Sheets UI (Share → add people by email), rather than using `share_anyone=True`.

If you used a service account to create the spreadsheet, you must also share it with your own Google account if you want to edit it in the browser — the service account owns the spreadsheet by default.

---

#### 12.5 The Config sheet structure

The Config sheet uses the same sentinel-based structure as the Excel interface. The four sentinels and their meanings are identical:

| Sentinel | Section | Required? |
|----------|---------|-----------|
| `[SETTINGS]` | Key/value configuration pairs | Always |
| `[CONTRAST]` | L matrix and δ (contrast/GLM modes) | When `power_mode` is `contrast` or `glm` |
| `[FACTORS]` | Factor definitions table | Always |
| `[RESPONSES]` | Per-response specs (multi-response) | Optional |

All `[SETTINGS]` keys from Chapter 11 (Table 11.3) apply here without change. The `[CONTRAST]`, `[FACTORS]`, and `[RESPONSES]` formats are also identical.

**One difference from Excel.** Google Sheets does not support dropdown validation cells the same way Excel does — the `power_mode` and `criterion` cells do not have in-cell pickers. Simply type the value directly: `r2`, `contrast`, or `glm` for `power_mode`; `I`, `D`, or `A` for `criterion`.

---

#### 12.6 Running the design: `sheets_run`

`sheets_run` authenticates, reads the Config sheet, runs the design search, writes results back, and returns the result dict:

```python
from iopt_power_design import sheets_run

result = sheets_run(
    spreadsheet_url_or_id,          # full URL or bare spreadsheet ID
    credentials="service_account.json",  # or None for OAuth2
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spreadsheet_url_or_id` | required | Full Google Sheets URL or bare spreadsheet ID. |
| `credentials` | `None` | Path to a service account JSON, or `None` for OAuth2 browser flow. |
| `config_sheet` | `"Config"` | Name of the configuration worksheet. |
| `results_sheet` | `"Results"` | Name of the sheet to write the summary to. |
| `design_sheet` | `"Design"` | Name of the sheet to write the design DataFrame to. |
| `buckets_sheet` | `"Buckets"` | Name of the sheet to write the bucket counts to. |
| `clear_results` | `True` | Clear existing content in output sheets before writing. |

The function returns the same result dict as `i_optimal_powered_design`, with one extra key: `"spreadsheet_url"` — the URL of the spreadsheet where results were written.

```python
rep = result["report"]
print(f"n={rep['n']}, power={rep['achieved_power']:.4f}")
print(f"Results written to: {result['spreadsheet_url']}")
```

After the call, the Results, Design, and Buckets sheets in the spreadsheet are populated and immediately visible to anyone who has access.

**Via the CLI:**

```bash
iopt-design --sheets "https://docs.google.com/spreadsheets/d/YOUR_ID" \
            --sheets-credentials path/to/service_account.json
```

The `--sheets` flag accepts both a full URL and a bare spreadsheet ID. `--sheets-credentials` is optional; if omitted, the CLI falls back to the `GOOGLE_APPLICATION_CREDENTIALS` environment variable and then to the OAuth2 browser flow.

---

**Error handling.** All Sheets integration errors raise `SheetsError`. The most common failure modes:

| Situation | Error message |
|-----------|---------------|
| `gspread` not installed | `ImportError: gspread is required…` |
| Authentication failure | `SheetsError: Authentication failed: <gspread error>` |
| Spreadsheet not shared with service account | `SheetsError: Could not open spreadsheet … Check that the spreadsheet exists and is shared with your account.` |
| Config sheet not found | `SheetsError: Worksheet 'Config' not found in spreadsheet.` |
| Missing `[SETTINGS]` sentinel | `SheetsError: Config sheet is missing the '[SETTINGS]' sentinel.` |
| Design search failure | `SheetsError: Design optimisation failed: <underlying error>` |

```python
from iopt_power_design import sheets_run, SheetsError

try:
    result = sheets_run(url, credentials=creds)
except SheetsError as e:
    print(f"Sheets error: {e}")
    if e.__cause__:
        print(f"  Caused by: {e.__cause__}")
    raise
```

---

#### 12.7 Full worked example

**Scenario.** A consumer research team is running a preference study. Four continuous factors — Price (£5–£25), Quality (1–10), Convenience (1–10), and Brand (1–5) — are studied with a main-effects model. The statistician has determined that an R² of 0.15 or above is worth detecting (consistent with published benchmarks for preference studies). The study protocol is reviewed by a cross-functional team in weekly Sheets-based planning documents. The design configuration will live in a shared Google Spreadsheet and the results will be written back to the same file.

---

**Step 1 — Create the template spreadsheet.**

```python
import os
from iopt_power_design import create_sheet_template

url = create_sheet_template(
    title="Consumer survey — powered design",
    credentials=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
    example="r2",
)
print(url)
# https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms
```

Open the URL in a browser. You will see the Config sheet pre-filled with the R² example. Share the spreadsheet with the review team using the standard Sheets share button (Share → add email addresses, Editor role).

---

**Step 2 — Edit the Config sheet.**

The review team edits the Config sheet in the browser to match the consumer study. The relevant `[SETTINGS]` changes:

```
[SETTINGS]
formula         ~ 1 + Price + Quality + Convenience + Brand
power_mode      r2
alpha           0.05
power           0.80
r2_target       0.15
max_n           300
criterion       I
starts          5
random_state    42

[FACTORS]
Name          Type         Value 1   Value 2
Price         continuous   5.0       25.0
Quality       continuous   1.0       10.0
Convenience   continuous   1.0       10.0
Brand         continuous   1.0       5.0
```

No `[CONTRAST]` section is needed — R² mode does not use an L matrix.

---

**Step 3 — Run the design.**

Once the protocol review is complete and the Config sheet is signed off, the statistician runs the design from Python:

```python
import os
from iopt_power_design import sheets_run

result = sheets_run(
    "https://docs.google.com/spreadsheets/d/YOUR_SPREADSHEET_ID",
    credentials=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
)

rep = result["report"]
print(f"n = {rep['n']}")
print(f"achieved power = {rep['achieved_power']:.4f}")
print(f"Results written to: {result['spreadsheet_url']}")
```

The Results, Design, and Buckets sheets in the spreadsheet are now populated. Everyone on the review team can open the spreadsheet in their browser and see:

- **Results sheet:** n, achieved power, R² target, criterion, elapsed time, search strategy
- **Design sheet:** the full run table with 73 rows and columns Price, Quality, Convenience, Brand
- **Buckets sheet:** run frequency by factor-level bucket

No one on the team needs a local Python installation to view or share the results.

---

**Step 4 — Automated re-run via CLI.**

If the protocol is revised and the Config sheet is updated, re-running the design requires only one command. In a shared team playbook or Makefile:

```bash
iopt-design \
  --sheets "https://docs.google.com/spreadsheets/d/YOUR_SPREADSHEET_ID" \
  --sheets-credentials "${GOOGLE_APPLICATION_CREDENTIALS}"
```

The output sheets are cleared and rewritten with the updated results. The Config sheet is unchanged — the run does not modify the inputs.

---

> **Tip — combining Sheets config with Python post-processing.** `sheets_run` returns the full result dict, so you can pipe the result directly into Python analysis functions without re-running the design:
>
> ```python
> from iopt_power_design import sheets_run, power_sensitivity
>
> result = sheets_run(url, credentials=creds)
>
> # Post-design sensitivity analysis in Python
> sens = power_sensitivity(
>     formula, factors, result["report"],
>     sigma_range=(0.5, 2.0),
> )
> print(sens)
> ```
>
> This pattern — Sheets for collaborative configuration, Python for analysis — combines the accessibility of a cloud spreadsheet with the full analytical power of the Python API.

---

### Chapter 13 — Jupyter Widgets: interactive in-notebook UI

The Jupyter widgets interface puts an interactive power-design UI directly
inside a JupyterLab or VS Code notebook cell. Instead of editing Python code to
explore different power assumptions, you adjust sliders and text fields, click
**Generate design**, and see the result — including an inline Plotly power curve
— update in place, with no page reload and no context switch.

---

#### 13.1 When to use the widgets interface

The widgets interface is best when the analytical goal is **exploration rather
than production**:

| Situation | Widgets fit? |
|-----------|-------------|
| You want to try several R² targets and see how sample size changes | Yes |
| You are teaching a class or running a workshop | Yes |
| You are demonstrating the package to a client who is in the room | Yes |
| You want a reproducible, auditable analysis script | Better as Python API |
| You need GLM (binomial/Poisson) mode | No — use Python API |
| You need multi-response design | No — use Python API |
| You are automating runs in CI | No — use CLI or Python API |

The widget UI supports **R² mode** and **contrast mode** only. For GLM designs,
multi-response designs, split-plot designs with advanced options, or any
scenario where you need programmatic post-processing, use `i_optimal_powered_design`
directly.

The practical teaching workflow is:

1. Use the widget to explore the design space interactively.
2. Once the configuration looks right, read off the parameters from the widget
   output.
3. Write a short, reproducible Python script with those parameters for archiving
   or handoff.

---

#### 13.2 Installing widget support

The widget dependencies (`ipywidgets` ≥ 8.0 and `plotly` ≥ 5.0) are optional.
Install them with the `[widgets]` extras group:

```bash
pip install "iopt-power-design[widgets]"
```

If you are working in JupyterLab, you do not need to install any additional
JupyterLab extension — `ipywidgets` 8.x bundles its own lab extension.

If you are working in VS Code, the Jupyter extension handles `ipywidgets`
rendering automatically.

To verify that the installation worked:

```python
from iopt_power_design.widgets import design_widget
print("widgets OK")
```

If `ipywidgets` is missing, importing `design_widget` succeeds (the import is
lazy), but calling it raises `WidgetsError` with a clear install hint. This
keeps the package importable in environments where widgets are not needed.

---

#### 13.3 The `design_widget` factory function and `DesignWidget` class

**`design_widget`** is the one-liner entry point. It creates a `DesignWidget`
instance, renders it in the current cell, and returns the instance so you can
read the result later:

```python
from iopt_power_design.widgets import design_widget

w = design_widget(
    formula="~ 1 + Price + Quality + Convenience",
    factors={
        "Price":       (0, 50),
        "Quality":     (1, 5),
        "Convenience": (1, 3),
    },
    power_mode="r2",
    r2_target=0.15,
    power=0.80,
    alpha=0.05,
)
```

`design_widget` accepts the same pre-fill parameters as `DesignWidget.__init__`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | `"~ 1 + A + B"` | Patsy formula — pre-fills the formula field |
| `factors` | `dict` | `{}` | Factor spec — continuous `(lo, hi)`, categorical `[...]` |
| `power_mode` | `"r2"` or `"contrast"` | `"r2"` | Which power panel is shown first |
| `alpha` | `float` | `0.05` | Significance level slider initial value |
| `power` | `float` | `0.80` | Target power slider initial value |
| `sigma` | `float` | `1.0` | σ field initial value (contrast mode) |
| `r2_target` | `float` | `0.15` | R² target slider initial value |
| `max_n` | `int` | `500` | Maximum sample-size search bound |
| `design_opts` | `DesignOptions` | `None` | Seed design options — exposed fields are pre-filled, non-exposed fields are forwarded at run time |
| `show_advanced` | `bool` | `False` | Whether to expand the Advanced accordion on load |

**Pre-filling from an existing `DesignOptions`**

If you already have a `DesignOptions` object — perhaps from a previous scripted
run — pass it as `design_opts`. The widget exposes six `DesignOptions` fields
as interactive controls:

| Widget control | `DesignOptions` field |
|----------------|-----------------------|
| Criterion dropdown (I / D / A) | `criterion` |
| Starts slider | `starts` |
| Random seed text box | `random_state` |
| Auto-size candidate set checkbox | `auto_candidate` |
| Candidate points text box | `candidate_points` |
| Constraint expression text box | `constraint_expr` |

Non-exposed fields (such as `n_blocks`, `n_whole_plots`, `workers`) are stored
internally and forwarded unchanged at run time. So if you set up a blocked
design via code and then pass the `DesignOptions` to the widget, the block
structure is preserved even though you cannot see or change it in the UI.

---

#### 13.4 What happens when you click "Generate design"

When you click the blue **Generate design** button, the widget:

1. Validates all inputs. Errors are shown in red beneath the button.
2. Reads every widget value into a state dict.
3. Calls `i_optimal_powered_design` with the assembled `formula`, `factors`,
   `PowerR2Config` or `PowerContrastConfig`, and `DesignOptions`.
4. On success, displays a metrics summary table, the full design matrix, the
   run buckets table (if any), and an inline Plotly power-vs-n curve.
5. Stores the full result in `w._result`, accessible via `w.get_result()`.

The inline power curve is built by scaling the returned noncentrality parameter
(λ) across a range of n values around the optimal. It is an approximation (the
design structure is not re-optimised at each n), but it is fast and accurate
enough for interactive exploration.

**Retrieving results after a run**

```python
# After clicking Generate design in the cell above:
result = w.get_result()        # full API result dict
df     = w.get_design_df()     # result["design_df"]  — the design matrix
report = w.get_report()        # result["report"]      — metrics dict
```

All three methods return `None` until at least one run has succeeded.

**Resetting the widget**

```python
w.reset()   # restores all controls to constructor defaults and clears result
```

---

#### 13.5 Full worked example — consumer survey in R² mode

This example mirrors the Chapter 4 consumer survey study (Price, Quality,
Convenience) using the widget UI instead of a pure API call.

**Cell 1 — launch the widget**

```python
from iopt_power_design.widgets import design_widget

w = design_widget(
    formula="~ 1 + Price + Quality + Convenience",
    factors={
        "Price":       (0, 50),
        "Quality":     (1, 5),
        "Convenience": (1, 3),
    },
    power_mode="r2",
    r2_target=0.15,   # detect a model explaining 15% of variance
    power=0.80,
    alpha=0.05,
    max_n=300,
    show_advanced=False,
)
```

The widget renders immediately. The factor table shows three pre-filled rows;
the R² slider sits at 0.15; the power slider at 0.80.

**Exploring in the UI**

Click **Generate design**. After a few seconds the metrics panel appears:

```
n (runs)          66
Achieved power    0.8013
Target power      0.800
α                 0.050
Noncentrality λ   11.6471
Criterion         I
```

The inline power curve shows power rising from near 0 at small n through 0.80
at n = 66, and approaching 1.0 by n ≈ 200.

Now slide the **R² target** from 0.15 to 0.10 — a smaller effect to detect —
and click **Generate design** again. The required n increases (the curve shifts
right). Slide it to 0.25 and run again: n decreases. This kind of rapid
what-if exploration is what the widget is designed for.

**Switching to contrast mode**

Click the **Contrast (L·β = δ)** toggle. The R² panel collapses and the
L matrix / δ vector / σ fields appear. Enter:

- **L matrix** (one row, three columns — intercept + Price + Quality + Convenience): `0 1 0 0`
- **δ vector** (minimum detectable effect for Price): `0.5`
- **σ (residual std)**: `1.0`

Click **Generate design**. For this simple main-effect contrast in a three-factor
model the returned n is 39 with power 0.8016.

Toggle back to R² mode (the widget remembers your R² values) and you can
compare the two approaches side by side.

**Cell 2 — extract and save the design**

After you are satisfied with a configuration, extract the result and save:

```python
import pandas as pd

result = w.get_result()
df     = w.get_design_df()

# Save design to CSV
df.to_csv("consumer_survey_design.csv", index=False)
print(f"Saved {len(df)} runs to consumer_survey_design.csv")

# Show report summary
r = w.get_report()
print(f"n={r['n']}  power={r['achieved_power']:.4f}  λ={r['noncentrality_lambda']:.4f}")
```

**Cell 3 — export an HTML report**

The full result dict is identical to what `i_optimal_powered_design` returns,
so you can use `export_report_to` after the fact via the API if needed — or
simply re-run with `export_report_to` set:

```python
from iopt_power_design.api import i_optimal_powered_design
from iopt_power_design.config import PowerR2Config, DesignOptions

# Re-run programmatically with the parameters identified via the widget
result = i_optimal_powered_design(
    formula="~ 1 + Price + Quality + Convenience",
    factors={
        "Price":       (0, 50),
        "Quality":     (1, 5),
        "Convenience": (1, 3),
    },
    power_cfg=PowerR2Config(r2_target=0.15, power=0.80, alpha=0.05),
    design_opts=DesignOptions(random_state=42),
    export_report_to="consumer_survey_report.html",
)
print("Report saved to consumer_survey_report.html")
```

This makes the widget-to-script handoff seamless: you explore with the widget,
identify the right parameters, then lock them in a reproducible script for
archiving.

**Widget-driven vs scripted: when to choose which**

| Task | Widget | Script |
|------|--------|--------|
| Initial power/n exploration | Preferred | Verbose |
| Trying 5 different R² targets | Preferred | Manageable |
| Sharing a notebook with collaborators | Good | Also good |
| Archiving in version control | Awkward | Preferred |
| CI/CD or automated scheduling | Not supported | Required |
| GLM / multi-response designs | Not supported | Required |

The typical workflow is widget first, script second: explore visually, then
freeze the winning configuration as a Python call for reproducibility.

---

#### 13.6 Using `DesignWidget` directly

`design_widget` is a thin wrapper that calls `DesignWidget(...).display()` and
returns the widget. Use `DesignWidget` directly if you want to build the widget
object first and display it later, or if you want to embed it inside a larger
`ipywidgets` layout:

```python
from iopt_power_design.widgets import DesignWidget
import ipywidgets as widgets

w = DesignWidget(
    formula="~ 1 + Price + Quality + Convenience",
    factors={
        "Price":       (0, 50),
        "Quality":     (1, 5),
        "Convenience": (1, 3),
    },
    power_mode="r2",
    r2_target=0.15,
)

# Wrap in a Tab alongside another widget
tab = widgets.Tab(children=[w._layout, widgets.HTML("<p>Other content</p>")])
tab.set_title(0, "Design")
tab.set_title(1, "Notes")
display(tab)
```

#### 13.7 `WidgetsError`

If `ipywidgets` or `plotly` is not installed, the first call to `design_widget`
or `DesignWidget(...)` raises `WidgetsError`:

```python
from iopt_power_design.widgets import WidgetsError

try:
    w = design_widget()
except WidgetsError as e:
    print(e)
# ipywidgets is required for the Jupyter UI. Install with:
# pip install "iopt-power-design[widgets]"
```

You can also import `WidgetsError` from the package root:

```python
from iopt_power_design import WidgetsError
```

---

### Chapter 14 — REST API: programmatic access and microservice integration

The REST API server wraps the full iopt-power-design library behind a set of
HTTP endpoints. It is the right interface when you need to call the package from
a non-Python environment, share a single design service across a team, or
integrate design generation into a data pipeline or scheduling system.

---

#### 14.1 When to use the REST API

| Situation | REST API fit? |
|-----------|--------------|
| Your analysis code is in R, JavaScript, Java, or any other language | Yes |
| You want a single shared design service for a team or platform | Yes |
| You are integrating with an external scheduler (Airflow, Prefect, cron) | Yes |
| You want to call the package from the Python API directly | No — import directly |
| You want an interactive UI | No — use Streamlit or widgets |

The REST API supports all power modes (R², contrast, GLM), multi-response
designs, split-plot designs, blocking, constraints, power curves, sensitivity
sweeps, MDE, criteria comparison, and design augmentation.

---

#### 14.2 Installing server dependencies

The server dependencies (FastAPI, Uvicorn, httpx, Pydantic v2) are optional:

```bash
pip install "iopt-power-design[server]"
```

---

#### 14.3 Starting the server

**Option A — `iopt-api` CLI entry point (recommended)**

```bash
iopt-api
```

This starts Uvicorn on `0.0.0.0:8000` with a fresh app instance. Equivalent to:

```bash
uvicorn api_server.main:create_app --factory --host 0.0.0.0 --port 8000
```

**Option B — Uvicorn with custom settings**

```bash
uvicorn api_server.main:create_app \
    --factory \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
```

The `--factory` flag is required. It tells Uvicorn to call `create_app()` once
per worker process rather than sharing a single app instance, which is the
correct pattern for multi-worker deployments.

**Note on parallelism inside the server**

The `workers` field in `DesignOptionsModel` (see §14.5) is accepted but silently
set to `None` inside the ASGI server. Process-pool executors conflict with
Uvicorn's event loop. Use Uvicorn's own `--workers` flag for horizontal scaling
instead.

**Interactive API docs**

Once the server is running, the auto-generated interactive documentation is
available at:

- `/docs` — Swagger UI (try endpoints directly in the browser)
- `/redoc` — ReDoc (readable reference)
- `/health` — health check: `{"status": "ok", "version": "..."}`

---

#### 14.4 Available endpoints

All endpoints accept and return JSON. Compute-heavy endpoints run in a
background thread so the server remains responsive; wall-clock time for a
design run is typically 10–120 seconds depending on complexity.

| Method + path | Purpose |
|---------------|---------|
| `POST /design` | Generate a single-response I-optimal powered design |
| `POST /multiresponse_design` | Generate a multi-response powered design |
| `POST /power_curve/by_n` | Power vs sample size curve |
| `POST /power_curve/by_effect` | Power vs effect size curve (fixed n) |
| `POST /sensitivity` | Power sensitivity sweep on a fixed design |
| `POST /mde` | Minimum detectable effect for a fixed design |
| `POST /compare_criteria` | Compare I, D, and A optimality criteria |
| `POST /augment` | Augment an existing design with additional runs |
| `GET /health` | Health check |

---

#### 14.5 Request/response schema overview

Every request body follows the same structural pattern:

```
formula        string              Patsy formula
factors        object              {"Name": [lo, hi]} or {"Name": ["lvl1", "lvl2"]}
power_cfg      object              Power configuration (see below)
design_opts    object (optional)   Design generation options
```

**Factor specification**

Continuous factors are encoded as a two-element JSON array `[low, high]`.
Categorical factors are encoded as an array of strings `["level1", "level2"]`.
In JSON, both are arrays — the API distinguishes them by element type.

**`power_cfg` — discriminated union on `"type"`**

The `power_cfg` field is a discriminated union. The `"type"` key is required
and selects which power model is used:

| `"type"` value | Power model | Key required fields |
|----------------|-------------|---------------------|
| `"r2"` | Global R² F-test | `r2_target` |
| `"contrast"` | L·β = δ contrast test | `L`, `delta` |
| `"glm_contrast"` | GLM Wald χ² contrast | `L`, `delta`, `family`, `baseline` |

All three types accept optional `alpha` (default 0.05), `power` (default 0.80),
and `max_n` (default 2000).

**`design_opts` — optional, uses defaults when omitted**

The `design_opts` object mirrors `DesignOptions` from the Python API. Key
fields:

| Field | Default | Description |
|-------|---------|-------------|
| `criterion` | `"I"` | Optimality criterion: `"I"`, `"D"`, or `"A"` |
| `starts` | `5` | Number of random starts |
| `random_state` | `123` | Random seed for reproducibility |
| `auto_candidate` | `false` | Auto-size candidate set |
| `candidate_points` | `2000` | Candidate set size (when `auto_candidate` is false) |
| `constraint_expr` | `null` | Row-level constraint string (e.g. `"A + B <= 1"`) |
| `n_blocks` | `null` | Number of blocks (≥ 2 to enable blocking) |
| `split_plot` | `null` | Split-plot options object |

Note: `constraint_func` (Python callable) cannot travel over HTTP. Use
`constraint_expr` (a string) instead — it is compiled server-side using the
same sandboxed AST evaluator as the Python API.

**`POST /design` response structure**

```json
{
  "design_df": [
    {"A": -0.97, "B": 0.84},
    ...
  ],
  "buckets_df": [
    {"A": -1.0, "B": -1.0, "count": 3},
    ...
  ],
  "report": {
    "n": 39,
    "p": 3,
    "df_num": 1,
    "df_denom": 36,
    "alpha": 0.05,
    "target_power": 0.80,
    "achieved_power": 0.8016,
    "noncentrality_lambda": 8.2471,
    "criterion": "I",
    "elapsed_sec": 4.21,
    "diagnostics": {
      "i_criterion": 0.034,
      "d_efficiency": 0.951,
      "condition_number": 3.1
    }
  }
}
```

**Error responses**

| HTTP status | `"error"` key | Cause |
|-------------|---------------|-------|
| 422 | `"ValidationError"` | Malformed JSON or invalid field value |
| 422 | `"InvalidInput"` | `ValueError` from the core library (e.g. bad formula) |
| 422 | `"DesignError"` | `RuntimeError` from the design search (e.g. no feasible design) |
| 500 | `"InternalServerError"` | Unexpected exception (see server logs) |

---

#### 14.6 Full worked example — curl and Python httpx

This example uses the same two-factor Temp/Pressure model from earlier chapters
to demonstrate the full request/response cycle. We show both `curl` and the
Python `httpx` client for completeness.

**Start the server** (in a separate terminal):

```bash
iopt-api
```

Wait for the log line:

```
INFO:     iopt-api v<version> started.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Health check**

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"..."}
```

**Design request — curl**

```bash
curl -s -X POST http://localhost:8000/design \
  -H "Content-Type: application/json" \
  -d '{
    "formula": "~ 1 + Temp + Pressure",
    "factors": {
      "Temp":     [-1, 1],
      "Pressure": [-1, 1]
    },
    "power_cfg": {
      "type":  "contrast",
      "L":     [[0, 1, 0]],
      "delta": [0.5],
      "sigma": 1.0,
      "power": 0.80,
      "alpha": 0.05
    },
    "design_opts": {
      "random_state": 42,
      "starts": 5
    }
  }' | python -m json.tool
```

The response `report` section will show `"n": 39` and `"achieved_power"` near
0.8016, matching the Python API result from earlier chapters.

**Design request — Python httpx**

```python
import httpx

BASE = "http://localhost:8000"

payload = {
    "formula": "~ 1 + Temp + Pressure",
    "factors": {
        "Temp":     [-1, 1],
        "Pressure": [-1, 1],
    },
    "power_cfg": {
        "type":  "contrast",
        "L":     [[0, 1, 0]],
        "delta": [0.5],
        "sigma": 1.0,
        "power": 0.80,
        "alpha": 0.05,
    },
    "design_opts": {
        "random_state": 42,
        "starts": 5,
    },
}

resp = httpx.post(f"{BASE}/design", json=payload, timeout=120)
resp.raise_for_status()
data = resp.json()

report = data["report"]
print(f"n={report['n']}  power={report['achieved_power']:.4f}")
print(f"λ={report['noncentrality_lambda']:.4f}  criterion={report['criterion']}")

# Parse the design matrix into a DataFrame
import pandas as pd
design_df = pd.DataFrame(data["design_df"])
print(design_df.head())
```

**GLM contrast request — curl**

For a pharma bioavailability study (binomial, baseline event rate 0.30,
formulation effect δ = 1.10 on the log-odds scale):

```bash
curl -s -X POST http://localhost:8000/design \
  -H "Content-Type: application/json" \
  -d '{
    "formula": "~ 1 + Dose + Formulation",
    "factors": {
      "Dose":        [-1, 1],
      "Formulation": ["IR", "SR"]
    },
    "power_cfg": {
      "type":     "glm_contrast",
      "L":        [[0, 0, 1]],
      "delta":    [1.10],
      "family":   "binomial",
      "baseline": 0.30,
      "power":    0.80,
      "alpha":    0.05
    }
  }' | python -m json.tool
```

**R² request — Python httpx**

```python
payload_r2 = {
    "formula": "~ 1 + Price + Quality + Convenience",
    "factors": {
        "Price":       [0, 50],
        "Quality":     [1, 5],
        "Convenience": [1, 3],
    },
    "power_cfg": {
        "type":      "r2",
        "r2_target": 0.15,
        "power":     0.80,
        "alpha":     0.05,
    },
}

resp = httpx.post(f"{BASE}/design", json=payload_r2, timeout=120)
data = resp.json()
print(f"n={data['report']['n']}  power={data['report']['achieved_power']:.4f}")
```

**Power curve request**

```python
payload_curve = {
    "formula": "~ 1 + Temp + Pressure",
    "factors": {"Temp": [-1, 1], "Pressure": [-1, 1]},
    "power_cfg": {
        "type":  "contrast",
        "L":     [[0, 1, 0]],
        "delta": [0.5],
        "sigma": 1.0,
    },
    "n_points": 10,
}

resp = httpx.post(f"{BASE}/power_curve/by_n", json=payload_curve, timeout=300)
curve_df = pd.DataFrame(resp.json()["rows"])
print(curve_df[["n", "power"]].to_string(index=False))
```

**MDE request (given a fixed design)**

First generate a design, then ask what the minimum detectable effect is for it
at 80% power:

```python
# 1. Generate design
design_resp = httpx.post(f"{BASE}/design", json=payload, timeout=120).json()
design_rows = design_resp["design_df"]

# 2. Ask for MDE
mde_payload = {
    "formula": "~ 1 + Temp + Pressure",
    "factors": {"Temp": [-1, 1], "Pressure": [-1, 1]},
    "design_df": design_rows,
    "power_cfg": {
        "type":  "contrast",
        "L":     [[0, 1, 0]],
        "delta": [0.5],    # ignored for MDE; starting guess only
        "sigma": 1.0,
    },
    "target_power": 0.80,
}

mde_resp = httpx.post(f"{BASE}/mde", json=mde_payload, timeout=60).json()
print(f"MDE={mde_resp['mde']:.4f}  achieved_power={mde_resp['achieved_power']:.4f}")
```

---

#### 14.7 Deploying behind a reverse proxy

The server has no built-in authentication. For production use, place a reverse
proxy (nginx, Traefik, Caddy) in front to enforce:

- TLS termination
- Access controls (API key, OAuth2, network CIDR)
- Rate limiting

A minimal nginx location block:

```nginx
location /iopt/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_set_header Host $host;
    proxy_read_timeout 180s;   # allow up to 3 min for heavy design searches
}
```

For multi-worker deployment:

```bash
uvicorn api_server.main:create_app \
    --factory \
    --workers 4 \
    --host 127.0.0.1 \
    --port 8000
```

Each Uvicorn worker is an independent process with its own Python interpreter.
There is no shared state between workers — every request is stateless.

---

## Part V — Advanced Design Features

### Chapter 15 — Split-plot designs: hard-to-change factors

In most experimental designs, every run is treated as independent: you can set
any factor to any value for each run. This is a completely randomised design
(CRD). Split-plot designs arise when one or more factors are physically
expensive or slow to change — resetting an oven to a new temperature, changing
a batch of raw material, or switching an operator — and the experiment must be
organised around those constraints.

---

#### 15.1 The split-plot problem

Consider an industrial baking study. You want to understand how oven
temperature (HTC — costly to change), flour type (HTC — requires pre-mixing),
baking time (ETC — easy to adjust), and humidity (ETC — controlled per run)
affect bread quality.

Resetting oven temperature and flour type for every run is impractical. The
natural structure is to fix temperature and flour type for a group of runs
(a *whole plot*), then vary baking time and humidity freely within that group
(the *sub-plots*). When the group is done, you reset the HTC factors for the
next group.

**Whole-plot (WP) factors** — changed only between groups. Every run within
a group shares the same WP factor settings. Examples: oven temperature, batch
material, operator, day-of-week.

**Sub-plot (SP) factors** — changed freely within each group. Examples:
reaction time, reagent concentration, baking time, humidity.

**Why CRD designs are wrong for split-plot structures**

An ordinary CRD design does not account for the within-group correlation. Runs
in the same whole plot are more alike than runs in different whole plots, because
they share the same WP factor settings. If you analyse a split-plot experiment
with CRD software (ignoring the WP structure), you under-estimate the variance
of WP-factor effects and produce anti-conservative (falsely narrow) confidence
intervals and inflated power estimates for those effects.

The correct model has two error terms:

```
y_ij = Xβ + τ_i + ε_ij

τ_i  ~ N(0, σ²_wp)    — whole-plot error, shared within WP group i
ε_ij ~ N(0, σ²_sp)    — sub-plot error, independent across runs

η = σ²_wp / σ²_sp     — variance ratio
```

The full observation covariance matrix is:

```
V = σ²_sp · (η Z Z' + I_n)
```

where `Z` is the n × n_wp whole-plot indicator matrix (1 if run j belongs to
WP group i). The GLS information matrix replaces the OLS `X'X` with
`M = X' V⁻¹ X`.

---

#### 15.2 The GLS information matrix and degrees of freedom

**Why η matters for power**

When η is large (WP variance dominates), runs within the same whole plot are
nearly perfectly correlated. The WP factor effects are estimated from the
variation *between* groups, not within them — you only get `n_wp − 1`
effective denominator degrees of freedom for those effects. That is much
smaller than `n − p` from an OLS analysis.

Sub-plot factor effects, by contrast, are estimated from variation *within*
groups and have close to `n_total − n_wp − (rank_SP)` denominator df.

**df_method — how the package assigns denominator degrees of freedom**

Power calculations require a denominator df for the non-central F-test. The
package provides three options through `SplitPlotOptions.df_method`:

| `df_method` | Rule | When to use |
|-------------|------|-------------|
| `"auto"` | WP df for contrasts involving only WP factors; SP df for all others | Default — correct for most studies |
| `"conservative"` | Always WP df (most conservative) | When in doubt, or when the contrast mixes WP and SP factors ambiguously |
| `"sp_only"` | Always SP df | Rarely correct; may be anti-conservative for WP contrasts |

The classification uses the model-matrix column labels: a contrast row is
classified as a WP contrast if every non-zero entry falls within a column that
involves only HTC factors.

---

#### 15.3 Setting up `SplitPlotOptions`

`SplitPlotOptions` is a dataclass imported from `iopt_power_design.config`.
Pass it as `design_opts.split_plot`:

```python
from iopt_power_design.config import SplitPlotOptions, DesignOptions

sp_opts = SplitPlotOptions(
    htc_factors=["OvenTemp", "FlourType"],   # whole-plot (HTC) factors
    n_whole_plots=10,                         # minimum WP groups to start bisection
    eta=1.0,                                  # σ²_wp / σ²_sp = 1 (equal variance)
    subplots_per_wp=4,                        # sub-plots per WP (None = auto)
    df_method="auto",                         # auto | conservative | sp_only
)

design_opts = DesignOptions(
    split_plot=sp_opts,
    random_state=42,
    starts=5,
)
```

**`SplitPlotOptions` field reference**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `htc_factors` | `list[str]` | *(required)* | Names of the HTC (whole-plot) factors — must match factor names passed to the API |
| `n_whole_plots` | `int` | *(required)* | Minimum number of whole-plot groups (≥ 2). The API bisects upward from this value to find the minimum WP count achieving target power. |
| `eta` | `float` | `1.0` | Variance ratio σ²_wp / σ²_sp. `eta=0` recovers standard OLS (no WP random effect). |
| `subplots_per_wp` | `int` or `None` | `None` | Number of sub-plots per WP group. `None` auto-computes `max(2, ceil(p / n_whole_plots) + 1)`. |
| `df_method` | `str` | `"auto"` | Denominator-df rule: `"auto"`, `"conservative"`, or `"sp_only"` |
| `criterion_ignore_vr` | `bool` | `False` | If `True`, use standard OLS criterion during design search (ignores η). For comparison studies only — not recommended for production. |

**Important constraints**

- `split_plot` and `n_blocks` cannot both be set. Three-stratum (blocked
  split-plot) designs are not yet supported. Setting both raises `ValueError`.
- GLM mode (`PowerGLMContrastConfig`) cannot be used with split-plot designs.
- `workers > 1` is not supported in split-plot mode.

**What the API does in split-plot mode**

`i_optimal_powered_design` bisects over `n_whole_plots`, starting at
`sp_opts.n_whole_plots` as the lower bound. At each WP count it builds a new
split-plot design, evaluates GLS power, and continues until the minimum WP
count achieving the target power is found. The returned design matrix contains
a `__wp_id__` column identifying which whole-plot group each run belongs to.

The total number of runs in the returned design is always
`n_whole_plots × subplots_per_wp`.

---

#### 15.4 The whole-plot cost-power curve: `power_curve_by_wp`

In many split-plot studies, the number of whole plots is the primary cost
driver — each WP reset requires time, material, and setup. Before committing to
a design, it is useful to see the full tradeoff: how does power grow as the
number of WP groups increases?

`power_curve_by_wp` sweeps `n_whole_plots` over a range and evaluates GLS
power at each value, returning a DataFrame with columns `n_wp`, `n_total`,
`power`, and `noncentrality_lambda`:

```python
from iopt_power_design import power_curve_by_wp
from iopt_power_design.config import PowerContrastConfig, DesignOptions, SplitPlotOptions

sp_curve_opts = DesignOptions(
    split_plot=SplitPlotOptions(
        htc_factors=["OvenTemp", "FlourType"],
        n_whole_plots=4,    # only used internally; wp_range overrides the sweep
        eta=1.0,
        df_method="auto",
    ),
    random_state=42,
    starts=3,
)

curve_df = power_curve_by_wp(
    formula="~ 1 + OvenTemp + FlourType + BakeTime + OvenTemp:BakeTime",
    factors={
        "OvenTemp":  (160, 200),
        "FlourType": ["standard", "whole-wheat"],
        "BakeTime":  (18, 30),
    },
    power_cfg=PowerContrastConfig(
        L=[[0, 0, 0, 1, 0]],   # BakeTime main effect
        delta=[0.5],
        sigma=1.0,
        power=0.80,
        alpha=0.05,
    ),
    subplots_per_wp=4,
    htc_factors=["OvenTemp", "FlourType"],
    eta=1.0,
    wp_range=(4, 20),
    wp_points=9,
    design_opts=sp_curve_opts,
)

print(curve_df[["n_wp", "n_total", "power"]].to_string(index=False))
```

The curve shows how power grows with WP count for the BakeTime sub-plot
effect. A typical result for a two-factor split-plot model with η = 1
(BakeTime tested against the SP error):

```
 n_wp  n_total   power
    4       16   0.271
    6       24   0.403
    8       32   0.522
   10       40   0.623
   12       48   0.708
   14       56   0.776
   16       64   0.830
   18       72   0.873
   20       80   0.905
```

The 80% power threshold is crossed between 14 and 16 whole plots (56–64
total runs). The minimum WP count is the decision point you present to the
team: *we need at least 15–16 oven setups to detect a half-standard-deviation
BakeTime effect with 80% confidence.*

**Adding a Plotly visualisation**

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=curve_df["n_wp"], y=curve_df["power"],
    mode="lines+markers", name="Power",
    line=dict(color="#1f77b4", width=2),
))
fig.add_hline(y=0.80, line_dash="dash", line_color="orange",
              annotation_text="Target 0.80")
fig.update_layout(
    title="WP Cost-Power Curve (BakeTime effect)",
    xaxis_title="Number of whole plots (oven setups)",
    yaxis_title="Power",
    yaxis=dict(range=[0, 1.05]),
    height=350,
)
fig.show()
```

---

#### 15.5 Full worked example — industrial baking process

**Study design**

A food science team is designing an experiment to understand how oven
temperature, flour type, baking time, and their interaction affect bread
texture. The factors are:

| Factor | Type | HTC? | Range/Levels |
|--------|------|------|--------------|
| OvenTemp | Continuous | Yes (HTC) | 160–200 °C |
| FlourType | Categorical | Yes (HTC) | standard, whole-wheat |
| BakeTime | Continuous | No (ETC) | 18–30 min |

Oven temperature and flour type require a batch pre-mix — they cannot be
changed between individual loaves. BakeTime is set per loaf. The team plans
4 loaves per oven setup (subplots_per_wp = 4).

The primary objective is to detect a main effect of BakeTime (the most
controllable and actionable factor) at 80% power, assuming σ = 1.0 (a
half-standard-deviation effect, δ = 0.5).

**Step 1 — Check the formula's column layout**

With FlourType as a two-level categorical factor (reference = `"standard"`),
the model matrix for `~ 1 + OvenTemp + FlourType + BakeTime + OvenTemp:BakeTime`
has p = 5 columns:

```
0: Intercept
1: FlourType[T.whole-wheat]
2: OvenTemp
3: BakeTime
4: OvenTemp:BakeTime
```

The BakeTime main-effect contrast is therefore `L = [[0, 0, 0, 1, 0]]`.

**Step 2 — Run the power-assured design**

```python
from iopt_power_design import i_optimal_powered_design
from iopt_power_design.config import (
    PowerContrastConfig, DesignOptions, SplitPlotOptions
)

result = i_optimal_powered_design(
    formula="~ 1 + OvenTemp + FlourType + BakeTime + OvenTemp:BakeTime",
    factors={
        "OvenTemp":  (160, 200),
        "FlourType": ["standard", "whole-wheat"],
        "BakeTime":  (18, 30),
    },
    power_cfg=PowerContrastConfig(
        L=[[0, 0, 0, 1, 0]],   # BakeTime main effect
        delta=[0.5],
        sigma=1.0,
        power=0.80,
        alpha=0.05,
    ),
    design_opts=DesignOptions(
        split_plot=SplitPlotOptions(
            htc_factors=["OvenTemp", "FlourType"],
            n_whole_plots=4,     # minimum; bisection searches upward
            subplots_per_wp=4,
            eta=1.0,
            df_method="auto",
        ),
        random_state=42,
        starts=5,
    ),
)

r = result["report"]
sp = r["split_plot"]

print(f"n_whole_plots : {sp['n_whole_plots']}")
print(f"subplots_per_wp: {sp['subplots_per_wp']}")
print(f"n_total       : {r['n']}")
print(f"achieved_power: {r['achieved_power']:.4f}")
print(f"lambda        : {r['noncentrality_lambda']:.4f}")
print(f"df_denom      : {r['df_denom']}")
```

The returned design is structured so that each block of 4 rows belongs to one
oven setup. The `__wp_id__` column identifies which whole-plot group each run
belongs to:

```python
df = result["design_df"]
print(df[["__wp_id__", "OvenTemp", "FlourType", "BakeTime"]].to_string(index=False))
```

Within each `__wp_id__` group, `OvenTemp` and `FlourType` are constant; only
`BakeTime` varies.

**Step 3 — Inspect the report split_plot sub-dict**

The `result["report"]["split_plot"]` dictionary contains the design structure:

```python
# result["report"]["split_plot"] contains:
# {
#     "n_whole_plots": <int>,
#     "subplots_per_wp": <int>,
#     "eta": 1.0,
#     "htc_factors": ["OvenTemp", "FlourType"],
#     "df_method": "auto"
# }
```

**Step 4 — Compare df_method choices**

For this study, the BakeTime contrast is a pure SP contrast (BakeTime is an
ETC factor). Under `df_method="auto"`, it receives SP denominator df.
Switching to `"conservative"` would assign WP df instead, which is smaller and
gives lower (more conservative) power. The difference matters most when WP df
is small (few whole plots):

```python
# Compare power under different df_method assumptions at the same n
for method in ["auto", "conservative", "sp_only"]:
    r2 = i_optimal_powered_design(
        formula="~ 1 + OvenTemp + FlourType + BakeTime + OvenTemp:BakeTime",
        factors={
            "OvenTemp":  (160, 200),
            "FlourType": ["standard", "whole-wheat"],
            "BakeTime":  (18, 30),
        },
        power_cfg=PowerContrastConfig(
            L=[[0, 0, 0, 1, 0]], delta=[0.5],
            sigma=1.0, power=0.80, alpha=0.05,
        ),
        design_opts=DesignOptions(
            split_plot=SplitPlotOptions(
                htc_factors=["OvenTemp", "FlourType"],
                n_whole_plots=4,
                subplots_per_wp=4,
                eta=1.0,
                df_method=method,
            ),
            random_state=42, starts=3,
        ),
    )
    rr = r2["report"]
    print(f"df_method={method:12s}  n_wp={rr['split_plot']['n_whole_plots']}  "
          f"power={rr['achieved_power']:.4f}")
```

The `"conservative"` result will return a larger n_whole_plots because the
WP denominator df is small and the F-test critical value is higher.

**Step 5 — CLI equivalent**

The same study can be run from the CLI using a YAML config:

```yaml
# baking_sp.yml
formula: "~ 1 + OvenTemp + FlourType + BakeTime + OvenTemp:BakeTime"

factors:
  OvenTemp:  [160, 200]
  FlourType: ["standard", "whole-wheat"]
  BakeTime:  [18, 30]

contrast:
  L: [[0, 0, 0, 1, 0]]
  delta: [0.5]

alpha: 0.05
power: 0.80
sigma: 1.0

design:
  random_state: 42
  starts: 5
  criterion: I

split_plot:
  htc_factors: [OvenTemp, FlourType]
  n_whole_plots: 4
  eta: 1.0
  subplots_per_wp: 4
  df_method: auto

output:
  basename: baking_sp
```

```bash
iopt-design --config baking_sp.yml --out ./output/baking_sp --html-report
```

---

#### 15.6 η sensitivity — how the variance ratio assumption affects results

The η parameter (σ²_wp / σ²_sp) is rarely known precisely in advance. It is
common to try several values before running an experiment:

```python
print("eta sensitivity for BakeTime effect:")
for eta in [0.5, 1.0, 2.0, 5.0]:
    r3 = i_optimal_powered_design(
        formula="~ 1 + OvenTemp + FlourType + BakeTime + OvenTemp:BakeTime",
        factors={
            "OvenTemp":  (160, 200),
            "FlourType": ["standard", "whole-wheat"],
            "BakeTime":  (18, 30),
        },
        power_cfg=PowerContrastConfig(
            L=[[0, 0, 0, 1, 0]], delta=[0.5],
            sigma=1.0, power=0.80, alpha=0.05,
        ),
        design_opts=DesignOptions(
            split_plot=SplitPlotOptions(
                htc_factors=["OvenTemp", "FlourType"],
                n_whole_plots=4,
                subplots_per_wp=4,
                eta=eta,
                df_method="auto",
            ),
            random_state=42, starts=3,
        ),
    )
    rr = r3["report"]
    print(f"  eta={eta:4.1f}  n_wp={rr['split_plot']['n_whole_plots']:2d}  "
          f"n={rr['n']:3d}  power={rr['achieved_power']:.4f}")
```

As η increases, WP variance dominates more and you need more whole plots to
detect the same effect. The SP factor effects become cheaper to detect (more
within-group replication) while WP factor effects become more expensive. This
asymmetry is the fundamental design challenge of split-plot experiments.

**Practical guidance on η**

- **η = 0**: No whole-plot random effect — identical to CRD. Use only if you
  are certain there is no batch-to-batch variation.
- **η = 1**: Equal WP and SP variance. A common conservative default.
- **η > 1**: WP variance dominates — typical for batch processes or when
  operators differ substantially.
- **Unknown η**: Run the analysis at η = 1 and η = 2. If the required WP
  counts are similar, the design is robust. If they differ substantially,
  collect a pilot estimate of η before committing.

---

### Chapter 16 — Blocked designs: accounting for nuisance variation

Experimental runs are rarely performed under perfectly homogeneous conditions.
A multi-day study sees day-to-day instrument drift. A multi-operator study sees
operator-to-operator differences. A batch process has batch-to-batch raw
material variation. Blocking is the statistical tool for accounting for these
known, structured sources of nuisance variation.

---

#### 16.1 What blocking is and when it is necessary

A **block** is a group of experimental runs performed under similar conditions
— the same day, the same operator, the same batch of material. Variation
*between* blocks is real but irrelevant to the treatment question. Including
block effects in the model as nuisance parameters lets you remove that
between-block variation from the error term, leaving the within-block
residuals to estimate your treatment effects cleanly.

**When blocking matters**

| Source of nuisance variation | Block factor |
|------------------------------|-------------|
| Equipment calibration drifts day to day | Day of experiment |
| Multiple operators run the experiment | Operator |
| Different reagent batches | Batch number |
| Equipment lots (separate runs of an instrument) | Equipment lot |
| Split across multiple labs | Lab |

**When blocking is not needed**

If all runs are genuinely exchangeable — same day, same operator, same batch
— there is no need to block. Adding unnecessary blocks only costs degrees of
freedom without any compensating reduction in error variance.

**The cost of blocking**

Adding `n_blocks` blocks to the model introduces `n_blocks − 1` nuisance
parameters (block dummy columns). The denominator degrees of freedom for the
F-test decreases accordingly:

```
df_denom (unblocked) = n − p_treat
df_denom (blocked)   = n − p_treat − (n_blocks − 1)
                     = n − (p_treat + n_blocks − 1)
```

For a small number of large blocks this cost is minor. For many small blocks
it can be substantial, and each block must have enough runs to support the
within-block design.

**Block effects as nuisance parameters**

Block effects are estimated and included in the model but are not the focus of
inference. They absorb the between-block variation, leaving the residuals to
reflect only the within-block (sub-block) experimental error. If blocks
genuinely capture real variation, the effective residual σ² is lower after
blocking — which compensates for the df loss.

---

#### 16.2 The `n_blocks` and `block_factor_name` parameters in `DesignOptions`

Blocking is activated by setting `n_blocks` in `DesignOptions`:

```python
from iopt_power_design.config import DesignOptions

design_opts = DesignOptions(
    n_blocks=2,               # number of blocks (≥ 2 to activate blocking)
    block_factor_name="Day",  # column name in the output design (default: "Block")
    random_state=42,
    starts=5,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_blocks` | `int` or `None` | `None` | Number of blocks. `None` = unblocked. Must be ≥ 2 when set. |
| `block_sizes` | `list[int]` or `None` | `None` | Explicit sizes for each block (must sum to design n). `None` = balanced (equal sizes, with first blocks getting 1 extra run if n is not divisible). |
| `block_factor_name` | `str` | `"Block"` | Name of the block column in the output design. Labels are `"B1"`, `"B2"`, …, `"Bn_blocks"`. |

**Specifying the L matrix with blocking**

The contrast matrix `L` always uses **treatment parameters only** — the same
columns as if no blocking were present. Block dummy columns are added
internally by the API; you do not need to include them in `L`. The number of
columns in `L` must equal `p_treat` (the treatment model parameter count).

```python
# For formula ~ 1 + A + B (p_treat = 3: Intercept, A, B)
# L tests the B main effect — same regardless of whether blocking is used
L = [[0, 0, 1]]
```

**What the API adds internally**

When `n_blocks ≥ 2`, the API:

1. Augments the formula with block dummy columns: `blocked_formula(formula)` →
   `"~ 1 + A + B + C(Block)"`.
2. Searches for I-optimal designs independently within each block (each block
   gets a distinct random seed derived from `random_state`).
3. Combines the per-block designs into a single DataFrame with a `Block` column
   (or the custom `block_factor_name`).
4. Evaluates power using the augmented model matrix, so the F-test correctly
   accounts for the block parameters.

The returned design matrix always includes the block column. The `report`
dict includes a `"block_structure"` key:

```python
# result["report"]["block_structure"]
# {"n_blocks": 2, "block_factor_name": "Day"}
```

---

#### 16.3 Low-level blocking utilities

The `blocked` module exposes the utilities the API uses internally. You rarely
need them directly, but they are useful for custom pipelines:

**`balanced_block_sizes(n, n_blocks)`** — compute balanced block sizes:

```python
from iopt_power_design.blocked import balanced_block_sizes

balanced_block_sizes(38, 2)   # [19, 19]
balanced_block_sizes(10, 3)   # [4, 3, 3]  — first block gets extra run
balanced_block_sizes(44, 4)   # [11, 11, 11, 11]
```

**`blocked_formula(formula, block_factor_name)`** — append block term to a
Patsy formula:

```python
from iopt_power_design.blocked import blocked_formula

blocked_formula("~ 1 + A + B")
# "~ 1 + A + B + C(Block)"

blocked_formula("~ 1 + A + B", block_factor_name="Day")
# "~ 1 + A + B + C(Day)"
```

**`build_blocked_design`** — low-level function called by `i_optimal_powered_design`
when blocking is active. It runs independent I-optimal searches for each block
and assembles the full design. Use this only if you need direct control over
the within-block optimisation (e.g., if blocks have unequal candidate sets).

---

#### 16.4 Full worked example — two-day study

**Scenario**

A quality team is running a two-factor screening experiment (factors A and B,
both coded [−1, +1]). The experiment spans two days. Day-to-day instrument
drift is real but not the focus of study. The team wants to detect a
half-standard-deviation effect of B (δ = 0.5, σ = 1.0) at 80% power.

**Step 1 — Unblocked baseline**

First, compute the sample size without blocking, as a reference:

```python
from iopt_power_design import i_optimal_powered_design
from iopt_power_design.config import PowerContrastConfig, DesignOptions

result_unblocked = i_optimal_powered_design(
    formula="~ 1 + A + B",
    factors={"A": (-1.0, 1.0), "B": (-1.0, 1.0)},
    power_cfg=PowerContrastConfig(
        L=[[0, 0, 1]],   # B main effect; p_treat=3: Intercept, A, B
        delta=[0.5],
        sigma=1.0,
        power=0.80,
        alpha=0.05,
    ),
    design_opts=DesignOptions(random_state=42),
)
r = result_unblocked["report"]
print(f"Unblocked: n={r['n']}  power={r['achieved_power']:.4f}  "
      f"df_denom={r['df_denom']}  lambda={r['noncentrality_lambda']:.4f}")
# Unblocked: n=39  power=0.7960  df_denom=36  lambda=8.2051
```

**Step 2 — Blocked design (2 days)**

```python
result_blocked = i_optimal_powered_design(
    formula="~ 1 + A + B",
    factors={"A": (-1.0, 1.0), "B": (-1.0, 1.0)},
    power_cfg=PowerContrastConfig(
        L=[[0, 0, 1]],   # same L — treatment parameters only, block columns excluded
        delta=[0.5],
        sigma=1.0,
        power=0.80,
        alpha=0.05,
    ),
    design_opts=DesignOptions(
        n_blocks=2,
        block_factor_name="Day",
        random_state=42,
        starts=5,
    ),
)
r = result_blocked["report"]
print(f"Blocked (2 days): n={r['n']}  power={r['achieved_power']:.4f}  "
      f"df_denom={r['df_denom']}  lambda={r['noncentrality_lambda']:.4f}")
# Blocked (2 days): n=38  power=0.8041  df_denom=34  lambda=8.4050
print(f"block_structure: {r['block_structure']}")
# block_structure: {'n_blocks': 2, 'block_factor_name': 'Day'}
```

**Interpreting the comparison**

| Design | n | p_full | df_denom | λ | Power |
|--------|---|--------|----------|---|-------|
| Unblocked | 39 | 3 | 36 | 8.205 | 0.796 |
| 2-block (days) | 38 | 4 | 34 | 8.405 | 0.804 |

With 2 blocks, the model gains 1 extra parameter (1 day dummy), losing 1
denominator df (36 → 34 at the same n). The API finds that n = 38 is
sufficient because the within-block I-optimal design achieves a slightly
higher noncentrality (8.405 vs 8.205) that more than compensates for the lost
df. The net result is essentially identical sample sizes.

**Step 3 — Inspect the design**

```python
df = result_blocked["design_df"]
print(df.to_string(index=False))
#      A         B   Day
# -0.9875  -0.9621   B1
#  0.9832   0.9814   B1
# ...
# -0.9815   0.9881   B2
#  0.9901  -0.9792   B2
# ...
```

Runs are labelled `"B1"` and `"B2"` (the auto-generated block labels) even
though `block_factor_name="Day"`. The label prefix is always `"B"` followed by
the block index. If you want more descriptive labels you can rename the column
after retrieval:

```python
df["Day"] = df["Day"].map({"B1": "Monday", "B2": "Tuesday"})
```

**Step 4 — Unbalanced blocks**

If the team can run only 15 experiments on Day 1 and 23 on Day 2, pass
`block_sizes` explicitly:

```python
result_unbalanced = i_optimal_powered_design(
    formula="~ 1 + A + B",
    factors={"A": (-1.0, 1.0), "B": (-1.0, 1.0)},
    power_cfg=PowerContrastConfig(
        L=[[0, 0, 1]], delta=[0.5], sigma=1.0, power=0.80, alpha=0.05,
    ),
    design_opts=DesignOptions(
        n_blocks=2,
        block_sizes=[15, 23],   # explicit unequal sizes; must sum to n
        block_factor_name="Day",
        random_state=42,
        starts=5,
    ),
)
```

`block_sizes` must have length `n_blocks` and must sum to the total design
n. Mismatches raise `ValueError`.

---

#### 16.5 How many blocks is too many?

The practical limit on block count comes from **within-block degrees of
freedom**. Each block has `n_block_i − p_treat` error df. If a block has
fewer runs than treatment parameters, the within-block design is
under-determined and the API will raise an error.

A rough guide:

| Scenario | Guidance |
|----------|----------|
| 2–3 large blocks | Low df cost, generally safe |
| 4–6 medium blocks | Manageable if each block has ≥ 2×p_treat runs |
| Many small blocks | Each block must have ≥ p_treat + 2 runs; df cost is high |

The noncentrality parameter can fall substantially when blocks are small
relative to the treatment model, because each within-block design has limited
resolution. If you have many anticipated blocks, consider whether the nuisance
variation can instead be captured by a continuous covariate (ANCOVA) or whether
a split-plot structure (Chapter 15) is more appropriate.

---

#### 16.6 YAML equivalent

For a reproducible pipeline, the same design can be specified in YAML:

```yaml
# two_day_study.yml
formula: "~ 1 + A + B"

factors:
  A: [-1.0, 1.0]
  B: [-1.0, 1.0]

contrast:
  L: [[0, 0, 1]]
  delta: [0.5]

alpha: 0.05
power: 0.80
sigma: 1.0

design:
  n_blocks: 2
  block_factor_name: Day
  random_state: 42
  starts: 5

output:
  basename: two_day_study
```

```bash
iopt-design --config two_day_study.yml --out ./output/two_day_study
```

---

### Chapter 17 — Feasibility constraints: excluding impossible factor combinations

Every experiment lives inside a design space, but not every corner of that space is
accessible. Equipment ratings, safety regulations, material compatibility, and economic
limits all define regions where experiments cannot, or should not, be run.
This chapter explains how to encode those limits so that the design search
automatically respects them.

---

#### 17.1 When factor combinations are physically impossible or dangerous

Consider a process with two continuous factors: temperature and pressure.
Individually, high temperature and high pressure are each achievable.
Together, they may exceed a vessel's rated limit, create an explosion risk, or
trigger an unwanted side-reaction.
The constraint is **compound**: both factors must be at their extreme simultaneously
for the problem to arise.
Ruling out only the individual extremes would be unnecessarily conservative and
would reduce the design space far more than necessary.

Other common scenarios follow the same pattern:

- **Pharmaceutical stability**: high concentration + high temperature accelerates
  degradation. Individual extremes are needed for a full quadratic model;
  the joint extreme is avoided.
- **Fermentation**: extreme pH + extreme temperature together kill the culture,
  while either alone is tolerable briefly.
- **Chemical synthesis**: some solvent/catalyst combinations are incompatible
  independent of other factor levels, creating a purely categorical exclusion.
- **Equipment limits**: a linear combination of factor values (e.g., total
  electrical load) must not exceed a rated maximum.

The package represents all of these through a single mechanism: a **filter applied
to the candidate set** before the design search begins.
The Fedorov exchange never sees infeasible candidate points,
so every design it produces is guaranteed to contain only feasible runs.

---

#### 17.2 String expression constraints: `constraint_expr`

The `constraint_expr` field of `DesignOptions` accepts a Python boolean expression
that is evaluated once per candidate row.
The expression can reference any factor column by name; rows where the expression
evaluates to `True` are kept; rows where it evaluates to `False` are removed.

```python
from iopt_power_design.config import DesignOptions

# Keep candidates only where the joint extreme is avoided
opts = DesignOptions(
    constraint_expr="not (Temperature > 150 and Pressure > 4.0)",
)
```

**Operator support.** The expression supports:

| Feature | Examples |
|---|---|
| Comparison | `>`, `>=`, `<`, `<=`, `==`, `!=` |
| Boolean logic | `and`, `or`, `not` |
| Arithmetic | `+`, `-`, `*`, `/`, `**`, `%`, `//` |
| Membership | `in`, `not in` (for categorical levels) |
| Math helpers | `sqrt`, `log`, `log10`, `log2`, `exp`, `floor`, `ceil`, `abs`, `min`, `max`, `round`, `pi` |

**What you cannot use.** For security, the expression is **AST-validated** before
execution. Any of the following raises a `ValueError` at `DesignOptions` construction
time, not at run time:

- Attribute access (`row.__class__`, `x.items()`)
- Subscripts (`x[0]`)
- Comprehensions, lambdas, or generator expressions
- Import statements
- Names starting with `_`
- Function calls other than the whitelisted math helpers

This makes `constraint_expr` safe to store in YAML files, Sheets configurations,
and Excel templates — it is never passed to `eval()` without validation.

**YAML usage.** Exactly the same string goes in a YAML config:

```yaml
design:
  constraint_expr: "not (Temperature > 150 and Pressure > 4.0)"
```

**Categorical constraints.** The `in` and `not in` operators work on categorical
factor columns:

```python
# Exclude a catalyst that is incompatible with high pH
opts = DesignOptions(
    constraint_expr="not (Catalyst == 'Pd' and pH > 9)",
)
```

---

#### 17.3 Callable constraints: `constraint_func`

When the logic is too complex for a single expression — involving Python objects,
external lookup tables, or multi-step computation — use `constraint_func` instead.

```python
from iopt_power_design.config import DesignOptions

def feasible(row):
    """Keep rows that don't combine high temperature with high pressure."""
    return not (row["Temperature"] > 150 and row["Pressure"] > 4.0)

opts = DesignOptions(constraint_func=feasible)
```

The callable must accept a single `pandas.Series` (one candidate row) and return
a `bool`.
It is called once per candidate point via `DataFrame.apply(constraint_func, axis=1)`.

**When to use `constraint_func` instead of `constraint_expr`:**

| Use `constraint_expr` | Use `constraint_func` |
|---|---|
| Simple arithmetic or logical combinations of factor columns | Logic requiring Python objects, loops, or lookup tables |
| Configs stored in YAML / Sheets / Excel | Code-only workflows |
| The expression reads naturally in a single line | Multi-step conditionals that span many lines |

**Mutual exclusivity.** If both `constraint_expr` and `constraint_func` are provided,
`constraint_expr` takes precedence: it is compiled to a callable and overwrites
`constraint_func`.
This lets `dataclasses.replace` copies preserve the expression string
without creating ambiguity.

---

#### 17.4 How constraints interact with candidate sizing

The constraint is applied to the candidate set **after** it is generated.
If the constraint removes a large fraction of candidates, the search operates
on a smaller feasible pool, which can have two consequences.

**Consequence 1 — more runs may be needed.** If the excluded region contains the
most informative points for estimating the effect of interest
(typically factor-level extremes), the design search must compensate with
more runs at the remaining feasible points to achieve the same power.
The bisection loop handles this automatically: it finds the minimum feasible `n`
that still achieves target power from the constrained candidate set.

**Consequence 2 — poor numerical conditioning.** With very few feasible candidates
(relative to the number of model parameters `p`), the initial candidate set may
produce a near-singular model matrix.
The `allow_candidate_growth` option addresses this: if the first design found
has a condition number above `1e6`, the candidate set is grown by `growth_factor`
(default `2.0`) up to `cand_max` and the search is re-run at the same `n`.
This one-time growth step is tried at most once per bisection iteration.

```python
opts = DesignOptions(
    constraint_expr="A + B <= -0.3",   # tight constraint: only ~35% of box remains
    allow_candidate_growth=True,        # grow if conditioning is poor
    growth_factor=3.0,                  # triple the candidate set (capped at cand_max)
)
```

**How to tell if growth helped.** The `report["search_strategy"]` key includes
`"growth"` if the candidate set was expanded during the search.

**When `allow_candidate_growth` is not needed.** For most practical constraints
that remove one quadrant or less of the design space, the default `candidate_points=2000`
is more than sufficient and conditioning is not an issue.
`allow_candidate_growth` is primarily useful for very tight constraints that
leave fewer than `5 × p` feasible candidates in the initial set.

---

#### 17.5 Full worked example

**Scenario.** A chemical engineering team is optimising a batch process.
Two factors are varied in coded units:

- **A** (Temperature): coded `−1` = 60 °C, `+1` = 180 °C
- **B** (Pressure): coded `−1` = 1 bar, `+1` = 5 bar

The response is dimensionless product yield.
A reactor safety assessment has flagged that simultaneously running at
very high temperature (A > 0.5, above 150 °C) **and** very high pressure
(B > 0.5, above 4 bar) exceeds the vessel's rated limit.
Either extreme alone is acceptable; the prohibition is compound.

The team wants to detect a main-effect shift of 1 coded unit in Temperature
(equivalent to a half-range effect) with 80% power at α = 0.05,
assuming residual standard deviation σ = 2.

**Setting up the unconstrained design first.**

```python
from iopt_power_design.api import i_optimal_powered_design
from iopt_power_design.config import DesignOptions, PowerContrastConfig

formula = "A + B + A:B"
factors = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}

power_cfg = PowerContrastConfig(
    L=[[0, 1, 0, 0]],    # test A (Temperature) main effect
    delta=[1.0],           # minimum half-range effect
    sigma=2.0,
    alpha=0.05,
    power=0.80,
    max_n=100,
)

opts_unc = DesignOptions(starts=5, random_state=42)
res_unc = i_optimal_powered_design(formula, factors, power_cfg, opts_unc)
r = res_unc["report"]
print(f"n={r['n']}  power={r['achieved_power']:.4f}  "
      f"lambda={r['noncentrality_lambda']:.4f}  "
      f"df=({r['df_num']},{r['df_denom']})")
```

```
n=39  power=0.8070  lambda=8.4541  df=(1,35)
```

The unconstrained I-optimal design requires **39 runs**.
At this sample size, λ = 8.45 is sufficient to achieve 80.7% power with 35 error
degrees of freedom.
The design concentrates runs near the four corners of the coded factor space,
where the contrast between A = −1 and A = +1 is sharpest.

**Adding the safety constraint.**

```python
opts_con = DesignOptions(
    starts=5,
    random_state=42,
    constraint_expr="not (A > 0.5 and B > 0.5)",
)
res_con = i_optimal_powered_design(formula, factors, power_cfg, opts_con)
r2 = res_con["report"]
print(f"n={r2['n']}  power={r2['achieved_power']:.4f}  "
      f"lambda={r2['noncentrality_lambda']:.4f}  "
      f"df=({r2['df_num']},{r2['df_denom']})")
```

```
n=46  power=0.8038  lambda=8.3049  df=(1,42)
```

The constrained design requires **7 additional runs** (46 vs. 39) to achieve the same
power target.
The excluded corner (A > 0.5 **and** B > 0.5) covers only about 6% of the factor box,
but it contains one of the four extreme-corner support points that the unconstrained
I-optimal design uses.
The algorithm compensates by placing points just outside the boundary
(e.g. A = 0.5, B = 1 or A = 1, B = 0.5), which provide slightly less per-run information,
requiring a modest increase in sample size to recover the target power.

**What happens with a more restrictive constraint?** The run count increases
when the constraint removes a region that is genuinely informative for the
model you are fitting.
For comparison, excluding the full quadrant (A > 0 **and** B > 0) — four times
the area — creates a severe non-orthogonality between the A main effect and
the A×B interaction column.
The algorithm compensates with many additional runs and, depending on
`max_n`, may not fully recover the target power within a practical budget.
The key principle: **the cost of a constraint depends on how much of the
high-leverage region it removes, not just its area.**

**Verifying feasibility.** The design respects the constraint by construction — the
Fedorov exchange operates only on feasible candidates — but it is worth verifying
explicitly:

```python
d = res_con["design_df"]
violations = d[(d["A"] > 0.5) & (d["B"] > 0.5)]
print(f"Constraint violations: {len(violations)}")  # → 0
```

```
Constraint violations: 0
```

**Using `constraint_func` for the same constraint.** If the constraint logic is
written programmatically and YAML portability is not required, `constraint_func`
produces the same result:

```python
def safe_region(row):
    """Return True if the run is in the safe operating region."""
    return not (row["A"] > 0.5 and row["B"] > 0.5)

opts_func = DesignOptions(
    starts=5,
    random_state=42,
    constraint_func=safe_region,
)
res_func = i_optimal_powered_design(formula, factors, power_cfg, opts_func)
# → same n and power as constraint_expr version
```

**YAML equivalent.** The constrained design can be reproduced from the command line
without any Python code:

```yaml
formula: "A + B + A:B"

factors:
  A: [-1.0, 1.0]
  B: [-1.0, 1.0]

power:
  mode: contrast
  L: [[0, 1, 0, 0]]
  delta: [1.0]
  sigma: 2.0

alpha: 0.05
power: 0.80

design:
  constraint_expr: "not (A > 0.5 and B > 0.5)"
  starts: 5
  random_state: 42
```

**When does the constraint add more runs?**

A 7-run increase (≈18%) for excluding 6% of the factor space is a modest but
real cost.
Whether a constraint is "cheap" or "expensive" depends not on its area alone but
on whether it removes high-leverage support points.
The increase becomes significant when:

- The constraint removes a **majority of the high-information corners or extremes**
  (e.g., excluding the entire right half of the A range halves the useful range of A,
  roughly doubling the required n for a test of the A main effect)
- The excluded region **creates non-orthogonality** between model columns: for
  interaction models, excluding a region where one factor is positive forces a
  dependency between the main effect and the interaction column, inflating both
  their variances simultaneously
- The model has many terms (large `p`), making each feasible high-leverage point
  more critical
- The feasible region is a small fraction of the full factor box (less than ~20%),
  in which case `allow_candidate_growth=True` is also advisable

Conversely, the constraint adds **zero runs** when the excluded region does not
overlap with any of the design's preferred high-leverage positions.
For example, restricting a factor's upper end by a few percent rarely changes
the required sample size at all, because the optimal design would not concentrate
runs at the very extreme tip anyway.

---

### Chapter 18 — Augmenting an existing design

Augmentation answers the question: *you already have some data — what runs should you add?*
Rather than discarding existing observations and rebuilding from scratch, `augment_design`
fixes the rows you already have in place and greedily selects additional runs that do the
most to improve the chosen optimality criterion.

---

#### 18.1 When augmentation is appropriate

Three scenarios commonly arise in practice.

**Budget unlocks after the initial study.**
A pilot study was sized conservatively, and mid-project the team secures funding for more
runs. Discarding the pilot data is wasteful; augmentation lets those observations contribute
to the final analysis.

**The preliminary design falls short of the power target.**
After collecting the pilot data, re-evaluating power against the actual residual variance
reveals that 80% power requires more runs than originally planned. Augmentation directs
the new runs to the positions that most efficiently close that gap.

**An early result triggers a follow-up.**
A surprising interaction or unexpectedly large variance in the pilot suggests that the
original design did not adequately cover some part of the factor space. Augmentation can
target the informative region without redoing runs already completed.

**When augmentation is *not* the right tool:**
If the initial design is severely poorly placed — for example, all runs at a single factor
setting — or if the factor structure or model has changed substantially, rebuilding with
`i_optimal_powered_design` is preferable.
Greedy augmentation is constrained by the existing rows and cannot recover from designs
that have fundamental multicollinearity problems.

---

#### 18.2 `augment_design`: the API

```python
from iopt_power_design import augment_design
```

**Signature:**

```python
augment_design(
    design_df,    # existing design rows (a DataFrame)
    m,            # number of new runs to add (int, >= 1)
    formula,      # patsy formula (same as the original design)
    factors,      # factor specifications (same as the original design)
    design_opts,  # DesignOptions (optional; controls candidate set and criterion)
) -> (augmented_df, new_runs_df)
```

**Inputs:**

| Parameter | Type | Description |
|---|---|---|
| `design_df` | `pd.DataFrame` | Existing design — all columns for every factor |
| `m` | `int` | Number of new runs to add; must be ≥ 1 |
| `formula` | `str` | Patsy formula, identical to the original design call |
| `factors` | `dict` | Factor specifications, identical to the original design call |
| `design_opts` | `DesignOptions` | Controls candidate sizing, criterion, `random_state` |

**Outputs:**

| Key | Description |
|---|---|
| `augmented_df` | Full design: original rows followed by the `m` new rows |
| `new_runs_df` | Only the `m` newly added rows, indexed 0 to m−1 |

**How the new points are chosen.**
The algorithm fixes the existing rows in `design_df` and iterates `m` times.
At each step it evaluates every candidate point, computing what the chosen criterion
(I, D, or A) would be if that candidate were appended, then adds the single candidate
that gives the best score.
This is a greedy one-point-at-a-time exchange, not a full multi-start Fedorov search.
It is fast — O(m × N\_cand × p²) — but it does not guarantee a globally optimal result.

**Criterion.**
The criterion defaults to `"I"` (I-optimality, minimising prediction variance integrated
over the factor space) and is controlled by `design_opts.criterion`.
Use `"D"` or `"A"` if the original design was built with those criteria.

**Candidate set.**
A fresh candidate set is generated for each augmentation call using the same
`factors` and `design_opts` settings. This means new runs can appear anywhere in
the factor space, not just at the original pilot's candidate positions.

---

#### 18.3 Evaluating power after augmentation

`augment_design` returns a DataFrame, not a power report. To check whether the
augmented design meets the power target, evaluate the model matrix directly with
`build_model_matrix` and `contrast_power`:

```python
import numpy as np
from iopt_power_design import build_model_matrix
from iopt_power_design.power import contrast_power

formula = "A + B + A:B"
factors  = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
L = np.array([[0, 1, 0, 0]])

aug_df, new_df = augment_design(pilot_df, m=12, formula=formula, factors=factors)

X_aug, _ = build_model_matrix(formula, aug_df)
pr = contrast_power(X=X_aug, L=L, delta=np.array([1.0]), sigma=2.0, alpha=0.05)
print(f"Power after augmentation: {pr.power:.4f}  (lambda={pr.lam:.4f})")
```

`contrast_power` takes:

| Argument | Value |
|---|---|
| `X` | Model matrix from `build_model_matrix` (2D numpy array) |
| `L` | Contrast matrix (same as `PowerContrastConfig.L`, as a numpy array) |
| `delta` | Effect size array (same as `PowerContrastConfig.delta`, as a numpy array) |
| `sigma` | Residual standard deviation |
| `alpha` | Significance level |

It returns a named tuple with `.power` and `.lam` (the noncentrality parameter λ).

**Iterative augmentation.**
If the power after +m runs is still below target, simply call `augment_design` again
on the result, or use a loop:

```python
current_df = pilot_df.copy()
target_power = 0.80

while True:
    X, _ = build_model_matrix(formula, current_df)
    pr = contrast_power(X=X, L=L, delta=np.array([1.0]), sigma=2.0, alpha=0.05)
    if pr.power >= target_power:
        break
    current_df, _ = augment_design(current_df, m=2, formula=formula, factors=factors)

print(f"Reached power={pr.power:.4f} at n={len(current_df)}")
```

---

#### 18.4 Full worked example

**Scenario.** Continuing with the two-factor coded design from Chapter 17:
two process factors A and B, each in [−1, 1], model `A + B + A:B` (p = 4 parameters).
The team wants 80% power to detect an A main effect of size 1.0 with σ = 2.0 (α = 0.05).

A preliminary budget covers 30 runs.
After collecting the pilot data, a power check reveals 62% power — well short of target.
Additional budget is approved for 12 more runs.
The question: where should those runs go?

**Step 1 — Build the pilot design.**

```python
import numpy as np
from iopt_power_design.iopt_search import build_i_opt_design_with_idx
from iopt_power_design.candidate import build_candidate
from iopt_power_design import build_model_matrix
from iopt_power_design.power import contrast_power

formula = "A + B + A:B"
factors  = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
L        = np.array([[0, 1, 0, 0]])

cand = build_candidate(factors, candidate_points=500, seed=42)
pilot_df, _, _ = build_i_opt_design_with_idx(
    n=30, cand=cand, formula=formula, n_start=5, random_state=42
)

X_pilot, _ = build_model_matrix(formula, pilot_df)
pr_pilot = contrast_power(X=X_pilot, L=L, delta=np.array([1.0]), sigma=2.0, alpha=0.05)
print(f"Pilot:  n={len(pilot_df)}  power={pr_pilot.power:.4f}  lambda={pr_pilot.lam:.4f}")
```

```
Pilot:  n=30  power=0.6213  lambda=5.5530
```

Power is only 62% — the pilot study is underpowered for the A main-effect test.

**Step 2 — Augment by 12 runs.**

```python
from iopt_power_design import augment_design, DesignOptions

aug_df, new_df = augment_design(
    design_df=pilot_df,
    m=12,
    formula=formula,
    factors=factors,
    design_opts=DesignOptions(random_state=42),
)

X_aug, _ = build_model_matrix(formula, aug_df)
pr_aug = contrast_power(X=X_aug, L=L, delta=np.array([1.0]), sigma=2.0, alpha=0.05)
print(f"After augment:  n={len(aug_df)}  power={pr_aug.power:.4f}  lambda={pr_aug.lam:.4f}")
```

```
After augment:  n=42  power=0.8019  lambda=8.3062
```

The design now exceeds the 80% target.

**Step 3 — Inspect the new runs.**

```python
print(new_df.to_string())
```

```
           A         B
0  -0.923924  0.983562
1   0.998672  0.993178
2  -0.998686 -0.992841
3   0.909105 -0.983273
4  -0.923924  0.983562
5   0.998672  0.993178
6  -0.998686 -0.992841
7   0.909105 -0.983273
8  -0.923924  0.983562
9   0.998672  0.993178
10  0.909105 -0.983273
11 -0.998686 -0.992841
```

All 12 new points sit near the corners of the factor box (|A| ≈ 0.91–1.00,
|B| ≈ 0.98–1.00). This is the greedy algorithm expressing a clear preference:
corner runs provide the most information for a linear model, and the pilot design
did not have enough of them.

**Power comparison.**

| Design | n | Power | λ | Notes |
|---|---|---|---|---|
| Pilot (30 runs) | 30 | 0.621 | 5.55 | Under-powered |
| Augmented (+12 runs) | 42 | 0.802 | 8.31 | Meets target |
| Fresh I-optimal (Chapter 17) | 39 | 0.807 | 8.45 | Optimal from scratch |

The augmented design requires **3 more runs** than a fresh I-optimal design at
the same target.
This is the **greedy penalty**: the 12 added runs are chosen to be as good as
possible given the existing 30-run structure, but they cannot move or replace
those 30 runs.
A globally optimal design built from scratch at n = 39 outperforms the greedy
augmentation at n = 42 because it has the freedom to place all 39 points jointly.

**When the penalty is small vs. large.**
The 3-run penalty in this example is modest because the pilot design already
had reasonable structure — all runs were at the extremes of the factor range.
The penalty grows when the existing design has concentrated runs in a poorly
informative region (e.g., all runs near the centre), because the greedy algorithm
cannot undo the information lost to those runs.

**Practical rule.** If augmentation leaves a gap larger than ≈10–15% above the fresh
optimal n, it is worth considering whether a full rebuild would be more efficient.
Use augmentation when existing data must be preserved (e.g., it has already been
analysed and reported) and the gap is manageable.

---

## Part VI — Analysis, Visualisation, and Reports

### Chapter 19 — Power curves: visualising the design–power relationship

`i_optimal_powered_design` answers one question: *what is the minimum n that meets
my power target?* Power curves answer the follow-up questions: *how does power
change as n grows? as the effect gets smaller or larger? if my assumptions about
σ turn out to be wrong?*

This chapter covers the five curve-generating functions in the package and explains
when each is useful.

---

#### 19.1 Why power curves matter

A single number — "n = 39" — leaves several important questions unanswered:

- **Diminishing returns.** Is the jump from n = 35 to n = 39 critical, or does power
  change slowly in that region? If 36 runs give 79.5% power, the extra 3 runs may not
  be worth the cost.
- **Sensitivity to assumptions.** σ = 2.0 is an estimate. If the true σ is 2.5, does
  n = 39 still provide adequate power? A curve across σ values makes this risk visible.
- **Communicating to non-statisticians.** A reviewer who asks "what can this design
  actually detect?" is easier to answer with a curve than with a single number and
  a formula.

Power curves shift the conversation from a point answer to a surface, making it
possible to discuss the *robustness* of the design choice rather than just its
headline number.

---

#### 19.2 Power vs. sample size: `power_curve_by_n`

`power_curve_by_n` sweeps n over a range, builds an I-optimal design at each n,
and evaluates power.
It is the most informative single diagnostic because it shows both where the
80% threshold is crossed and how quickly power saturates at high n.

```python
from iopt_power_design import (
    power_curve_by_n, PowerContrastConfig, DesignOptions
)

formula  = "A + B + A:B"
factors  = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
power_cfg = PowerContrastConfig(
    L=[[0, 1, 0, 0]], delta=[1.0], sigma=2.0,
    alpha=0.05, power=0.80, max_n=60,
)
opts = DesignOptions(starts=3, random_state=42)

result = power_curve_by_n(
    formula, factors, power_cfg,
    n_range=(5, 55),      # sweep bounds
    n_points=10,          # number of n values to evaluate
    design_opts=opts,
    plot=False,
)
```

**Return value.** `power_curve_by_n` returns a dict:

| Key | Type | Contents |
|---|---|---|
| `"data"` | `pd.DataFrame` | One row per evaluated n; columns below |
| `"figure"` | Figure or `None` | Plot object if `plot=True`, else `None` |
| `"target_n"` | `int` or `None` | First n in the sweep that reaches the power target |

**DataFrame columns:**

| Column | Description |
|---|---|
| `n` | Sample size |
| `power` | Achieved power at the I-optimal design of size n |
| `lambda` | Noncentrality parameter λ |
| `d_efficiency` | D-efficiency of the design (0–1; higher is better) |
| `i_criterion` | I-criterion value (lower is better) |
| `condition_number` | Condition number of X'X (large values warn of near-multicollinearity) |

For the example above:

```python
print(result["target_n"])
# 42

print(result["data"].to_string())
```

```
    n     power     lambda  d_efficiency  i_criterion  condition_number
0   5  0.0747   1.0796      0.8978     0.3961          1.986
1   6  0.1090   1.3163      0.8739     0.3358          1.819
2   8  0.1818   1.8152      0.9232     0.2320          1.202
3  11  0.2764   2.4861      0.9016     0.1733          1.568
4  14  0.3609   3.1423      0.8964     0.1354          1.330
5  18  0.4607   3.9945      0.8883     0.1054          1.307
6  24  0.5905   5.2902      0.8757     0.0795          1.316
7  32  0.7227   6.9827      0.8594     0.0603          1.363
8  42  0.8297   8.9359      0.8432     0.0464          1.417
9  55  0.9137  11.4786      0.8284     0.0358          1.477
```

> **Note: `target_n` vs. the solver's n.** `power_curve_by_n` evaluates n at the
> geometric grid points you specify — here {5, 6, 8, 11, …, 55}. The first grid
> point that crosses 80% is n = 42, so `target_n = 42`. The bisection solver in
> `i_optimal_powered_design` searches more finely and finds n = 39.
> Use `i_optimal_powered_design` when you need the exact minimum n; use
> `power_curve_by_n` with a denser grid (larger `n_points`) or a tighter range
> when you want the curve detail.

**The two-panel plot.** With `plot=True`:

```python
result = power_curve_by_n(
    formula, factors, power_cfg,
    n_range=(5, 55), n_points=15,
    design_opts=opts,
    plot=True,
)
```

The matplotlib figure has two stacked panels:
- **Top panel:** Power vs. n, with a horizontal dashed line at the target power
  and a vertical marker at `target_n`.
- **Bottom panel:** I-criterion (left y-axis, green) and D-efficiency (right y-axis,
  orange) vs. n. These help you see whether additional design quality is being
  "purchased" as n grows or whether the design is already near-optimal.

**Plotly and Streamlit.** Pass `plot_backend="plotly"` to get an interactive figure:

```python
result = power_curve_by_n(
    formula, factors, power_cfg,
    n_range=(5, 55), n_points=15,
    design_opts=opts,
    plot=True,
    plot_backend="plotly",
)

fig = result["figure"]  # plotly.graph_objects.Figure

# In a Streamlit app:
import streamlit as st
st.plotly_chart(fig, use_container_width=True)
```

The Plotly figure supports hover tooltips showing exact (n, power, λ) values and
zoom/pan interactions, making it more useful than the static matplotlib version for
exploratory work.

**`analysis.power_curve_by_n` vs. `power_curves.power_curve_by_n`.**
The package exposes a thin wrapper in `iopt_power_design.analysis` that returns only
the DataFrame (discarding the figure and `target_n`):

```python
from iopt_power_design import power_curve_by_n   # returns DataFrame
```

Use the `analysis` version when you only need the data and don't want to manage the
extra dict keys. Use `power_curves.power_curve_by_n` directly when you need the figure
or the `target_n` key:

```python
from iopt_power_design.power_curves import power_curve_by_n as pcbn
result = pcbn(formula, factors, power_cfg, ...)   # returns full dict
```

---

#### 19.3 Power vs. effect size: `power_curve_by_effect`

`power_curve_by_effect` holds n fixed and sweeps the effect size.
It answers: *what is the minimum effect this design can detect at 80% power?*
— the **minimum detectable effect (MDE)**.

```python
from iopt_power_design import power_curve_by_effect

result = power_curve_by_effect(
    formula, factors,
    n=39,
    power_cfg=power_cfg,       # base delta=[1.0] is the reference scale
    effect_range=(0.3, 2.0),   # multipliers on the base delta
    effect_points=10,
    design_opts=opts,
    plot=False,
)
```

**Return value.**

| Key | Type | Contents |
|---|---|---|
| `"data"` | `pd.DataFrame` | Columns: `effect_size`, `power`, `lambda`, `actual_delta_norm` |
| `"figure"` | Figure or `None` | Plot if `plot=True` |
| `"min_detectable_effect"` | `float` or `None` | First grid point reaching 80% power |

**Reading the `effect_size` column.**
For contrast mode, `effect_size` is a *multiplier* on the base `delta` vector.
An `effect_size` of 1.0 means the design is evaluated at exactly the stated `delta`;
values < 1.0 represent smaller (harder to detect) effects; values > 1.0 represent
larger effects.

```
   effect_size     power     lambda  actual_delta_norm
0     0.300000  0.1357   0.7609           0.300
1     0.488889  0.2823   2.0206           0.489
2     0.677778  0.4827   3.8837           0.678
3     0.866667  0.6881   6.3500           0.867
4     1.055556  0.8471   9.4195           1.056
5     1.244444  0.9403  13.0923           1.244
...
```

```python
print(f"MDE = {result['min_detectable_effect']:.3f}x the stated delta")
# MDE = 1.056x the stated delta
```

At n = 39, the first grid point that crosses 80% power is at effect_size = 1.056 —
meaning a true effect of 1.056 × δ = 1.056 is just detectable at 80% power with
10 evenly spaced grid points. The exact crossing (from the full solver) is at
δ = 1.0 (that is what `i_optimal_powered_design` was designed for), so the grid
approximation is conservative by about 6%.
Increasing `effect_points` narrows this gap.

**Communicating to stakeholders.**
The effect-size curve is useful when reviewers ask "what if the true effect is
only half of your assumed minimum?" Read off the power at effect_size = 0.5 (i.e.,
50% of the stated δ) from the curve. In the example above, a 30% effect
(effect_size = 0.30) gives only 14% power — a clear illustration of why the minimum
detectable effect matters.

---

#### 19.4 Power vs. baseline (GLM): `power_curve_by_baseline`

For GLM designs (binomial or Poisson responses), power depends heavily on the
baseline event probability or rate because the Fisher information weight
`w = p₀(1 − p₀)` (binomial) or `w = μ₀` (Poisson) changes with the baseline.
`power_curve_by_baseline` holds the design fixed and sweeps the baseline.

```python
from iopt_power_design import power_curve_by_baseline, PowerGLMContrastConfig

glm_cfg = PowerGLMContrastConfig(
    L=[[0, 1, 0, 0]],
    delta=[0.5],
    baseline=0.3,       # assumed baseline event probability
    family="binomial",
    link="logit",
    alpha=0.05,
    power=0.80,
)

# Assume design_df is already built with i_optimal_powered_design
df_baseline = power_curve_by_baseline(
    formula=formula,
    factors=factors,
    design_df=design_df,
    cfg=glm_cfg,
    baseline_range=(0.05, 0.95),
    baseline_points=30,
)
# Columns: baseline, power, lam, family, link
```

**Key behaviour.**
For binomial models, power peaks near baseline = 0.5 (maximum Fisher information
weight) and drops toward 0 as baseline approaches 0 or 1.
If the true baseline probability is far from the assumed value, power can fall
significantly below the target — use this curve to understand that risk before
committing to a sample size.

For Poisson models, power increases monotonically with the baseline rate because
higher rates mean more Fisher information per observation.

> **Note:** `power_curve_by_baseline` takes an existing `design_df` — it does not
> rebuild the design as baseline varies. This is a pure sensitivity analysis on the
> fixed design matrix.

---

#### 19.5 The power surface: `power_surface_2d`

`power_surface_2d` sweeps two parameters simultaneously and returns a grid of power
values, producing a heatmap that shows the "safe" vs. "risky" regions of the
parameter space.

```python
from iopt_power_design import power_surface_2d

result = power_surface_2d(
    formula, factors, power_cfg,
    param1="n",        param1_range=(10, 60),
    param2="sigma",    param2_range=(1.0, 4.0),
    grid_points=15,    # 15 × 15 = 225 evaluations
    design_opts=opts,
    plot=True,
    plot_backend="plotly",
)

# result["data"]       — DataFrame with columns [param1, param2, power]
# result["power_grid"] — 2D numpy array of power values
# result["figure"]     — contour plot if plot=True
```

**Valid parameter choices** for `param1` and `param2`:

| Value | Meaning |
|---|---|
| `"n"` | Sample size (integer grid) |
| `"effect"` | Effect size multiplier on δ (contrast) or actual R² (R² mode) |
| `"sigma"` | Residual standard deviation (contrast mode only) |
| `"alpha"` | Significance level |

**Reading the surface.**
The contour line at power = 0.80 is the boundary between designs that meet the
target and designs that fall short.
- Everything *above* the 80% contour (in a (n, σ) surface) is a "safe" region: the
  design is adequately powered even if σ is larger than assumed.
- The gradient of the surface reveals where the design is most sensitive: a steep
  gradient in the σ direction means power changes rapidly with small changes in σ
  (a fragile design); a shallow gradient means the design is robust to that parameter.

> **Computational cost.** When `param1` or `param2` is `"n"`, the function builds a
> separate I-optimal design for each distinct n value. With `grid_points=15`, that is
> up to 15 separate design searches, each with `starts` multi-start runs. Use a smaller
> `grid_points` (8–10) or a tighter n range for interactive exploration.
>
> When neither parameter is `"n"` (e.g., a `"sigma"` × `"effect"` surface), the design
> is built once at a fixed n and power is recomputed analytically for each grid cell —
> this is very fast.

---

#### 19.6 Multi-response power curves: `power_curve_by_n_multiresponse`

When the design must satisfy power requirements for two or more responses
simultaneously, `power_curve_by_n_multiresponse` sweeps n and evaluates
per-response power as well as the combined power at each step.

```python
from iopt_power_design import (
    power_curve_by_n_multiresponse, MultiResponseOptions, ResponseSpec
)

multi_cfg = MultiResponseOptions(
    responses=[
        ResponseSpec(
            name="Yield",
            power_cfg=PowerContrastConfig(
                L=[[0, 1, 0, 0]], delta=[1.0], sigma=2.0,
                alpha=0.05, power=0.80,
            ),
        ),
        ResponseSpec(
            name="Purity",
            power_cfg=PowerContrastConfig(
                L=[[0, 0, 1, 0]], delta=[0.8], sigma=1.5,
                alpha=0.05, power=0.80,
            ),
        ),
    ],
    power_combination="all",   # combined power = minimum of per-response powers
)

df_multi = power_curve_by_n_multiresponse(
    formula, factors, multi_cfg,
    n_range=(10, 60),
    n_points=12,
    design_opts=opts,
)
# Columns: n, combined_power, Yield_power, Purity_power
```

**Interpreting the output.**
Each response gets its own power column (`<name>_power`).
The `combined_power` column reflects the combination rule set in
`MultiResponseOptions.power_combination`:

| Rule | `combined_power` definition |
|---|---|
| `"all"` | Minimum of per-response powers (every response must be powered) |
| `"any"` | Maximum of per-response powers |
| `"weighted"` | Weighted average using `ResponseSpec.weight` |

**Identifying the bottleneck response.**
The response whose power curve rises most slowly to the 80% threshold is the
bottleneck: it drives the required n. Reading the per-column curves makes this
immediately visible. If the bottleneck response has a much harder power target
than the others, consider loosening its δ or σ assumption, or redesigning for
that response specifically.

---

#### 19.7 Full worked example

This section brings all five curve types together for the running example
(A + B + A:B model, A, B ∈ [−1, 1], testing A main effect, δ = 1.0, σ = 2.0).

**Setup:**

```python
from iopt_power_design import (
    power_curve_by_n, power_curve_by_effect,
    power_surface_2d, PowerContrastConfig, DesignOptions,
    i_optimal_powered_design,
)
from iopt_power_design.power_curves import power_curve_by_n as pcbn_full

formula  = "A + B + A:B"
factors  = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
power_cfg = PowerContrastConfig(
    L=[[0, 1, 0, 0]], delta=[1.0], sigma=2.0,
    alpha=0.05, power=0.80, max_n=60,
)
opts = DesignOptions(starts=3, random_state=42)
```

**Curve 1 — Power vs. n:**

```python
result_n = pcbn_full(
    formula, factors, power_cfg,
    n_range=(5, 55), n_points=10,
    design_opts=opts,
    plot=True, plot_backend="plotly",
)
print(f"First n to reach 80% (in this grid): {result_n['target_n']}")
# 42
fig_n = result_n["figure"]
```

Key read: the curve is steep at small n and flattens above n ≈ 30.
Once power exceeds ≈ 85%, each additional run contributes less than 1 percentage
point. The I-criterion panel (bottom) shows that D-efficiency is already near its
maximum by n = 20, so further runs improve power mainly by increasing degrees of
freedom, not by placing points better.

**Curve 2 — Power vs. effect size:**

```python
from iopt_power_design.power_curves import power_curve_by_effect as pcbe_full

result_e = pcbe_full(
    formula, factors, n=39,
    power_cfg=power_cfg,
    effect_range=(0.3, 2.0),
    effect_points=10,
    design_opts=opts,
    plot=True, plot_backend="plotly",
)
print(f"MDE (grid approx): {result_e['min_detectable_effect']:.3f}x delta")
# 1.056x delta
fig_e = result_e["figure"]
```

Read from the curve: at n = 39, effects smaller than about 0.87 × δ achieve less
than 70% power. If the true effect turns out to be half the stated minimum (0.5×),
power drops to ≈ 18% — the design would rarely detect such a small signal.

**Curve 3 — Power surface (n × σ):**

```python
result_surf = power_surface_2d(
    formula, factors, power_cfg,
    param1="n",     param1_range=(15, 55),
    param2="sigma", param2_range=(1.0, 4.0),
    grid_points=10,
    design_opts=opts,
    plot=True, plot_backend="plotly",
)
fig_surf = result_surf["figure"]
```

The surface shows the 80% power contour as a line in (n, σ) space.
Runs along the contour (e.g., n = 39, σ = 2.0 → n = 70, σ = 2.7) are equivalent
from a power standpoint. If prior studies suggest σ may be as large as 3.0, reading
off the required n at σ = 3.0 shows how much the sample size needs to grow to
remain adequately powered.

**Curves 4 and 5 — Baseline and multi-response** are described in
§19.4 and §19.6 respectively.
For a binomial design, generate the baseline curve with `power_curve_by_baseline`
after building the GLM design; for a multi-response design, use
`power_curve_by_n_multiresponse` in place of `power_curve_by_n`.

---

### Chapter 20 — Sensitivity analysis and robustness

Every power calculation rests on assumptions that are, at best, educated guesses.
σ comes from a pilot study or a literature estimate; δ reflects expert judgement
about what constitutes a "meaningful" effect. When those assumptions are wrong,
the achieved power can fall far below the target even if the design was built
correctly.

Sensitivity analysis quantifies that risk. The functions in this chapter let you
answer: *how far off can my assumptions be before the design fails?*

---

#### 20.1 Why assumptions about σ, δ, and baseline are often wrong

**Residual standard deviation (σ).** The most common source of surprise. A pilot
study with n = 12 gives a σ estimate with large uncertainty: the 95% confidence
interval for σ spans roughly 0.7× to 1.6× the estimated value. A design built at
σ = 2.0 may encounter σ = 2.8 in the actual experiment — dropping power from
80% to around 55%.

**Minimum detectable effect (δ).** Practitioners often set δ by intuition: "we
want to detect a 10% improvement." But if the model has been parameterised in
coded units (−1 to 1) and the 10% improvement corresponds to a coefficient of 0.8
rather than 1.0, the design is sized for the wrong target.

**GLM baseline probability.** For a binomial endpoint, power depends on the
baseline event rate p₀ through the Fisher information weight w = p₀(1 − p₀).
A rate assumed to be 0.30 that turns out to be 0.15 cuts the weight in half,
roughly halving λ.

**The "just-passing" risk.** A design that achieves exactly 81% power at the
nominal assumptions has no headroom. If any assumption is slightly pessimistic,
power drops below target. `power_sensitivity` and `robustness_report` make this
risk visible before the experiment runs.

---

#### 20.2 `power_sensitivity`: sweeping σ or R²

`power_sensitivity` takes a **fixed** design and analytically evaluates power
across a range of σ (contrast mode) or R² (R² mode). No new designs are built.

```python
from iopt_power_design import power_sensitivity

sensitivity = power_sensitivity(
    formula,
    factors,
    power_cfg,          # PowerContrastConfig or PowerR2Config
    design_df,          # the fixed design to evaluate
    sigma_range=(1.0, 4.0),   # sweep σ from 1.0 to 4.0
    sigma_points=13,          # 13 evenly-spaced values
    plot=False,
)
```

**Return value (contrast mode):**

| Key | Type | Contents |
|---|---|---|
| `"data"` | `pd.DataFrame` | Columns: `sigma`, `power`, `noncentrality_lambda` |
| `"nominal_power"` | `float` | Power at `power_cfg.sigma` |
| `"sigma_nominal"` | `float` | The nominal σ from `power_cfg` |
| `"figure"` | Figure or `None` | Plot if `plot=True` |

**Return value (R² mode):**

| Key | Type | Contents |
|---|---|---|
| `"data"` | `pd.DataFrame` | Columns: `r2_target`, `power`, `noncentrality_lambda` |
| `"nominal_power"` | `float` | Power at `power_cfg.r2_target` |
| `"r2_nominal"` | `float` | The nominal R² from `power_cfg` |
| `"figure"` | Figure or `None` | Plot if `plot=True` |

For a σ sweep, the data looks like:

```
    sigma     power  noncentrality_lambda
0    1.00  0.9998             32.56
1    1.25  0.9933             20.84
2    1.50  0.9586             14.47
3    1.75  0.8864             10.63
4    2.00  0.7916              8.14
5    2.25  0.6929              6.43
6    2.50  0.6017              5.21
7    2.75  0.5225              4.31
8    3.00  0.4556              3.62
...
```

At the nominal σ = 2.0, power is 79.2% (the design achieved n = 38 in this run).
If σ rises to 2.5 — a 25% increase — power drops to 60%.
If σ doubles to 4.0, power falls to 28%.

**Plotting with reference lines.** Pass `plot=True` and `plot_backend="plotly"` to
get an interactive figure with vertical reference lines at the nominal σ and
horizontal lines at the power target:

```python
sensitivity = power_sensitivity(
    formula, factors, power_cfg, design_df,
    sigma_range=(1.0, 4.0), sigma_points=25,
    plot=True, plot_backend="plotly",
)
fig = sensitivity["figure"]
```

The Plotly figure shows hover-enabled labels at each σ value and highlights the
region where power falls below the target — making it easy to communicate the
risk to collaborators who are not statisticians.

---

#### 20.3 `robustness_report`: structured sensitivity summary

`power_sensitivity` sweeps a single axis. `robustness_report` sweeps three axes
simultaneously — effect size, σ, and α — and returns a structured summary with
threshold values and an overall pass-rate statistic.

```python
from iopt_power_design import robustness_report

rr = robustness_report(
    design_df,
    formula,
    factors,
    power_cfg,
    sigma_range=(1.0, 4.0),    sigma_points=11,
    effect_range=(0.4, 2.0),   effect_points=11,
    alpha_range=(0.01, 0.10),  alpha_points=9,
)
```

**Return value:**

| Key | Type | Contents |
|---|---|---|
| `"mode"` | `str` | `"contrast"` or `"r2"` |
| `"nominal_power"` | `float` | Power at nominal assumptions |
| `"effect_sweep"` | `pd.DataFrame` | Columns: `effect_scale` (or `r2_target`), `power`, `noncentrality_lambda` |
| `"sigma_sweep"` | `pd.DataFrame` or `None` | Columns: `sigma`, `power`, `noncentrality_lambda` (contrast only) |
| `"alpha_sweep"` | `pd.DataFrame` | Columns: `alpha`, `power`, `noncentrality_lambda` |
| `"summary"` | `dict` | `worst_power`, `median_power`, `best_power`, `power_target`, `pct_scenarios_passing` |
| `"thresholds"` | `dict` | `max_sigma_for_target`, `min_effect_for_target`, `min_alpha_for_target` |
| `"figure"` | Figure or `None` | Multi-panel plot if `plot=True` |

**Reading the thresholds.**
The threshold values are the boundary values where power transitions through the
target:

```python
print(rr["thresholds"])
# {
#   'max_sigma_for_target': 1.978,
#   'min_effect_for_target': 1.015,
#   'min_alpha_for_target': 0.053
# }
```

- `max_sigma_for_target = 1.978`: the design meets 80% power only if σ ≤ 1.978.
  With a nominal σ = 2.0, the design is marginally underpowered — a small σ
  over-estimate is enough to push power below target.
- `min_effect_for_target = 1.015`: a true effect of at least 1.015 × δ is required
  to achieve 80% power. Effects just below the stated δ are not reliably detected.
- `min_alpha_for_target = 0.053`: the design just barely meets the 80% target at
  α = 0.053; using the standard α = 0.05 leaves power slightly short.

**Reading the summary.**

```python
print(rr["summary"])
# {
#   'worst_power': 0.199,
#   'median_power': 0.804,
#   'best_power': 1.000,
#   'power_target': 0.80,
#   'pct_scenarios_passing': 0.516
# }
```

The `pct_scenarios_passing` value (51.6%) pools all scenarios from all three
sweeps and reports the fraction that achieve the power target.
A value above ~70% is a healthy sign; below 50% suggests the design has little
robustness and the assumptions should be revisited.

**When to use `robustness_report` vs. `power_sensitivity`.**
Use `power_sensitivity` when you want a focused, single-axis sweep (e.g., just
σ) with a plot. Use `robustness_report` when you need a compact executive summary
covering all three sensitivity axes simultaneously — for example, when preparing
a study protocol or answering reviewer questions about design adequacy.

---

#### 20.4 `multiresponse_sensitivity`: per-response σ sensitivity

For multi-response designs, `multiresponse_sensitivity` builds a single I-optimal
design at a fixed n and sweeps a common σ *scale factor* across all responses.
Each response's σ is multiplied by the scale factor at every point.

```python
from iopt_power_design import multiresponse_sensitivity, MultiResponseOptions, ResponseSpec

multi_cfg = MultiResponseOptions(
    responses=[
        ResponseSpec(
            name="Yield",
            power_cfg=PowerContrastConfig(
                L=[[0,1,0,0]], delta=[1.0], sigma=2.0, alpha=0.05, power=0.80,
            ),
        ),
        ResponseSpec(
            name="Purity",
            power_cfg=PowerContrastConfig(
                L=[[0,0,1,0]], delta=[0.8], sigma=1.5, alpha=0.05, power=0.80,
            ),
        ),
    ],
    power_combination="all",
)

df_ms = multiresponse_sensitivity(
    formula, factors, multi_cfg,
    fixed_n=42,
    sigma_range=(0.5, 3.0),   # scale factors: 0.5× to 3.0× each response's sigma
    sigma_points=20,
)
# Columns: sigma_scale, combined_power, Yield_power, Purity_power
```

**Key point: `sigma_range` is a scale factor, not an absolute σ value.**
A `sigma_scale` of 1.0 means each response is evaluated at its configured σ.
A `sigma_scale` of 1.3 multiplies every response's σ by 1.3 simultaneously —
a scenario where all responses are noisier than expected by the same factor.

**Identifying the bottleneck.**
The response whose power drops through the target at the smallest σ scale factor
is the bottleneck. In the example above, if Purity power reaches 80% at
σ_scale = 0.9 but Yield power reaches 80% only at σ_scale = 0.75, Yield is
the bottleneck — it drives the required n and is the more fragile assumption.

> **Restriction:** `multiresponse_sensitivity` only supports `PowerContrastConfig`
> responses. A `TypeError` is raised if any response uses `PowerR2Config`.

---

#### 20.5 Full worked example

**Scenario.** The team has built the A + B + A:B design from Chapter 17 (n = 38
at σ = 2.0, testing the A main effect with δ = 1.0, α = 0.05). Before running the
experiment, they want to answer three questions:

1. *If σ turns out to be 30% higher than assumed (σ = 2.6), how much power do we
   lose?*
2. *What is the maximum σ at which the design still meets 80% power?*
3. *Across all plausible combinations of assumptions, what fraction of scenarios
   still meet the target?*

**Setup:**

```python
from iopt_power_design import (
    i_optimal_powered_design, power_sensitivity, robustness_report,
    PowerContrastConfig, DesignOptions,
)

formula  = "A + B + A:B"
factors  = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
power_cfg = PowerContrastConfig(
    L=[[0, 1, 0, 0]], delta=[1.0], sigma=2.0,
    alpha=0.05, power=0.80, max_n=60,
)
opts = DesignOptions(starts=5, random_state=42)

result = i_optimal_powered_design(formula, factors, power_cfg, opts)
design_df = result["design_df"]
r = result["report"]
print(f"n={r['n']}  power={r['achieved_power']:.4f}")
```

```
n=38  power=0.7916
```

**Question 1 — Power at σ = 2.6:**

```python
sens = power_sensitivity(
    formula, factors, power_cfg, design_df,
    sigma_range=(1.0, 4.0),
    sigma_points=13,
)
df_s = sens["data"]
# Power at sigma=2.5:
row = df_s[df_s["sigma"] == 2.5].iloc[0]
print(f"sigma=2.5  power={row['power']:.4f}")
```

```
sigma=2.5  power=0.6017
```

A 25% increase in σ (from 2.0 to 2.5) reduces power from 79% to 60%.
For context, a σ estimate from a 12-run pilot has roughly a 90% confidence
interval of [0.74×, 1.57×] the true σ — meaning σ = 2.5 is well within the
plausible range.

**Question 2 — Maximum σ for 80% power:**

```python
rr = robustness_report(
    design_df, formula, factors, power_cfg,
    sigma_range=(1.0, 4.0),   sigma_points=11,
    effect_range=(0.4, 2.0),  effect_points=11,
    alpha_range=(0.01, 0.10), alpha_points=9,
)
print(f"max sigma for 80% power: {rr['thresholds']['max_sigma_for_target']:.3f}")
```

```
max sigma for 80% power: 1.978
```

The design achieves 80% power only if σ ≤ 1.978. With a nominal assumption of
σ = 2.0, there is essentially no headroom: if the true σ is even slightly above
the estimate, the design is underpowered.

**Question 3 — Overall pass rate:**

```python
print(rr["summary"])
```

```
{
  'worst_power': 0.199,
  'median_power': 0.804,
  'best_power': 1.000,
  'power_target': 0.80,
  'pct_scenarios_passing': 0.516
}
```

Only 52% of all plausible assumption scenarios (pooled from the three sweeps)
achieve 80% power. This confirms that the design is fragile: it was built right
at the edge of the target, with no buffer for assumption error.

**What to do with this result.**
Three options:

1. **Increase n.** Adding ≈10 runs raises power to ≈85% at σ = 2.0, giving more
   headroom against σ uncertainty. The power-vs-n curve from Chapter 19 shows
   exactly how many runs are needed.

2. **Widen σ assumption.** Re-estimate σ from historical data or accept a higher
   nominal σ and re-run the solver. This will increase the required n but produce
   a design that is accurately powered.

3. **Accept the risk.** If the experiment is cheap and a false-negative (missed
   effect) is recoverable, running at n = 38 with the understanding that power
   may be 60–80% is a deliberate, informed decision — not an oversight.

Sensitivity analysis does not mandate a particular choice; it ensures the choice
is made with clear information about the risks.

---

### Chapter 21 — Minimum detectable effect

`i_optimal_powered_design` answers a *forward* question: given a stated effect δ,
what n achieves 80% power? The **minimum detectable effect (MDE)** answers the
*inverse* question: given a fixed design with n runs, what is the smallest effect
that achieves the target power?

The MDE is useful in two situations:

- **Validating an inherited design.** A previous team built a 24-run design and
  left no power documentation. Before augmenting or discarding it, you want to know
  what it can actually detect.
- **Communicating capability to reviewers.** A reviewer asks "what is the smallest
  effect this experiment can detect?" The MDE gives a direct, interpretable answer
  in the same units as δ.

---

#### 21.1 The inverse question

Power analysis is normally a planning tool: fix δ and σ, find the required n.
The MDE inverts this: fix the design (which fixes n and the information matrix
X'X), and find the minimum δ at which the target power is achieved.

Mathematically, for contrast mode, the noncentrality parameter is

```
λ(scale) = (scale × δ)² / [σ² × L(X'X)⁻¹Lᵀ]
```

which increases monotonically in `scale`. The MDE is the `scale` at which the
F-test achieves the target power. A scale of 1.0 means the design is perfectly
calibrated: the stated δ is exactly detectable. A scale > 1.0 means only *larger*
effects are reliably detected — the design is underpowered for the stated δ.

---

#### 21.2 `min_detectable_effect`

```python
from iopt_power_design import min_detectable_effect

mde_result = min_detectable_effect(
    design_df,        # fixed design to evaluate
    formula,          # patsy formula used to build design_df
    factors,          # factor specifications
    power_cfg,        # PowerContrastConfig, PowerR2Config, or PowerGLMContrastConfig
    target_power=0.80,   # default 0.80
)
```

**Return value (contrast mode):**

| Key | Type | Contents |
|---|---|---|
| `"mde"` | `float` | Scale factor on δ at which `target_power` is achieved |
| `"achieved_power"` | `float` | Power at the MDE (≈ `target_power`, within tolerance) |
| `"n"` | `int` | Number of runs in the design |
| `"mode"` | `str` | `"contrast"` |

**Return value (R² mode):**

| Key | Type | Contents |
|---|---|---|
| `"mde"` | `float` | Minimum `r2_target` at which `target_power` is achieved |
| `"achieved_power"` | `float` | Power at the MDE |
| `"n"` | `int` | Number of runs |
| `"mode"` | `str` | `"r2"` |

**Return value (GLM mode):**
Same keys as contrast mode, plus `"min_delta_lp"` (the absolute LP-scale delta
corresponding to the MDE scale factor), `"family"`, and `"baseline"`.

**The `target_power` parameter.**
Changing `target_power` shifts the threshold at which the MDE is computed:

```python
mde_80 = min_detectable_effect(design_df, formula, factors, power_cfg, target_power=0.80)
mde_90 = min_detectable_effect(design_df, formula, factors, power_cfg, target_power=0.90)

print(f"MDE at 80% power: {mde_80['mde']:.3f}x delta")
print(f"MDE at 90% power: {mde_90['mde']:.3f}x delta")
```

A higher target power requires a larger effect to be reliably detected, so the
90% MDE is always larger than the 80% MDE. This gives you two numbers:
"effects above 1.38× δ are detectable at 80% confidence; above 1.59× δ at 90%."

**Interpreting the MDE in R² mode.**
For `PowerR2Config`, `mde["mde"]` is not a scale factor — it is the actual minimum
R² value at which the design achieves the target power. A value of 0.36 means the
design can detect model fits with R² ≥ 0.36 (a large effect by Cohen's conventions)
at 80% power; smaller R² effects are missed. Values below 0.10 indicate a sensitive
design; values above 0.25 indicate an underpowered one.

---

#### 21.3 Full worked example

**Scenario.** A process team has inherited a 24-run experiment from a previous
project. The original documentation is sparse — no power analysis was performed.
Before deciding whether to run the experiment as-is, augment it, or rebuild from
scratch, the team wants to know:

1. What effect size can the 24-run design detect at 80% power?
2. How does that compare to the 39-run design sized for the stated δ = 1.0?
3. What does the 24-run design detect at 70% power?

**Setup:**

```python
from iopt_power_design import min_detectable_effect, DesignOptions, PowerContrastConfig
from iopt_power_design.iopt_search import build_i_opt_design_with_idx
from iopt_power_design.candidate import build_candidate

formula = "A + B + A:B"
factors  = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
power_cfg = PowerContrastConfig(
    L=[[0, 1, 0, 0]], delta=[1.0], sigma=2.0,
    alpha=0.05, power=0.80,
)

# Build the inherited 24-run design
cand = build_candidate(factors, candidate_points=500, seed=42)
design_df_24, _, _ = build_i_opt_design_with_idx(
    n=24, cand=cand, formula=formula, n_start=5, random_state=42
)
```

**Question 1 — MDE at 80% power (24-run design):**

```python
mde_24 = min_detectable_effect(
    design_df_24, formula, factors, power_cfg, target_power=0.80
)
print(f"MDE (80%): {mde_24['mde']:.3f}x delta  "
      f"(achieved power: {mde_24['achieved_power']:.4f})")
```

```
MDE (80%): 1.376x delta  (achieved power: 0.8000)
```

The 24-run design can only reliably detect effects that are **at least 1.38× the
stated δ**. With δ = 1.0, this means effects below 1.38 are missed more than 20%
of the time. The design is substantially underpowered for the stated target.

**Question 2 — Comparison with the 39-run design:**

```python
design_df_39, _, _ = build_i_opt_design_with_idx(
    n=39, cand=cand, formula=formula, n_start=5, random_state=42
)
mde_39 = min_detectable_effect(
    design_df_39, formula, factors, power_cfg, target_power=0.80
)
print(f"MDE (80%) at n=39: {mde_39['mde']:.3f}x delta")
```

```
MDE (80%) at n=39: 1.075x delta
```

The properly sized 39-run design requires effects to be only **7.5% above the
stated δ** to be reliably detected. This gap reflects the precision of the
bisection tolerance — the solver found a design that essentially meets the stated
SESOI (δ = 1.0) at 80% power.

| Design | n | MDE | Interpretation |
|---|---|---|---|
| Inherited | 24 | 1.376 × δ | Misses effects up to 38% larger than stated δ |
| Properly sized | 39 | 1.075 × δ | Essentially meets the stated δ target |

**Question 3 — MDE at 70% power:**

```python
mde_70 = min_detectable_effect(
    design_df_24, formula, factors, power_cfg, target_power=0.70
)
print(f"MDE (70%): {mde_70['mde']:.3f}x delta")
```

```
MDE (70%): 1.220x delta
```

At the lower 70% power threshold, the 24-run design requires effects ≥ 1.22 × δ.
Presenting both numbers to a reviewer gives a useful range: "effects above 1.38×
are detected 80% of the time; effects above 1.22× at least 70% of the time."

**Decision: augment or rebuild?**
With MDE = 1.376 at n = 24, the team has two options:

- **Augment** using `augment_design` (Chapter 18): add 15 runs to reach n = 39,
  paying the greedy penalty of ~3 extra runs (see Chapter 18), arriving at n ≈ 42
  with MDE ≈ 1.1 × δ.
- **Rebuild** from scratch at n = 39 with `i_optimal_powered_design`: achieves
  MDE ≈ 1.075 × δ — marginally better than augmentation, and 3 fewer runs.

If the 24 runs have already been collected and analysed, augmentation preserves
their value. If the 24 runs are unrun (an inherited *plan*, not *data*), rebuilding
is usually preferable.

---

### Chapter 22 — Shareable reports

After a design is built, the results often need to be shared with collaborators,
documented in a study protocol, or reviewed by someone who does not have Python
installed. `generate_report` produces a self-contained HTML (or PDF) file that
captures everything about the design in a single document.

---

#### 22.1 What the report contains

A generated report has six sections:

**Configuration.** The formula, factor definitions, and power configuration
(contrast L, δ, σ, α, target power, or R² parameters) are displayed in a
human-readable table. This makes it possible to reproduce the design from the
document alone.

**Power metrics.** The key numerical results: required n, achieved power,
noncentrality parameter λ, and degrees of freedom (numerator and denominator).

**Design table.** The first `design_rows_shown` rows of the design DataFrame
(default: up to 30 rows). A note is shown when the full design has more rows.

**Buckets table.** The `buckets_df` summary from `i_optimal_powered_design`,
showing run-count groupings.

**Diagnostics.** Any metrics from the `report` dict that were recorded during
the search (timing, condition number, etc.).

**Power-curve figure.** When `include_power_curve=True`, a power-vs-n curve is
generated and embedded as a base64-encoded PNG — no external image files or
internet connection required.

---

#### 22.2 Generating a report: `generate_report`

**Installing the dependency.** HTML report generation requires `jinja2`:

```bash
pip install "iopt-power-design[report]"
```

PDF output additionally requires `weasyprint`:

```bash
pip install "iopt-power-design[report-pdf]"
```

**Signature:**

```python
from iopt_power_design import generate_report
from pathlib import Path

report_path = generate_report(
    result,          # dict returned by i_optimal_powered_design
    formula,         # Patsy formula string
    factors,         # factor specification dict
    power_cfg,       # PowerContrastConfig or PowerR2Config
    output_path,     # str or Path — file destination
    title="I-Optimal Design Report",   # optional heading
    include_power_curve=True,          # embed a power-vs-n figure
    design_rows_shown=30,              # max rows in the design table
)
# report_path is a pathlib.Path of the file that was written
```

**Output path rules:**

| `output_path` value | Result |
|---|---|
| `"report.html"` | Writes `report.html` |
| `"report.pdf"` | Writes `report.pdf` (requires WeasyPrint) |
| `"results/"` (directory) | Writes `results/iopt_report.html` |
| `"report"` (no extension) | Writes `report.html` (`.html` appended automatically) |

**Returns** the resolved `pathlib.Path` of the written file.

**`export_report_to` shortcut.**
Pass `export_report_to` directly to `i_optimal_powered_design` to generate the
report as part of the design call:

```python
result = i_optimal_powered_design(
    formula, factors, power_cfg, design_opts,
    export_report_to="my_design_report.html",
)
# → my_design_report.html is written automatically
# result["report"]["report_path"] holds the resolved path
```

When using this shortcut, `include_power_curve` is set to `False` to keep
the call fast. Use the explicit `generate_report` call with
`include_power_curve=True` if you want the embedded figure.

---

#### 22.3 The self-contained HTML format

The HTML report is designed to be **completely self-contained**:

- All CSS is inlined in the `<style>` block — no external stylesheet is fetched.
- The power-curve figure (when present) is base64-encoded and embedded directly
  in the HTML as a `<img src="data:image/png;base64,...">` tag.
- No JavaScript is required.

This means the file can be:
- Emailed as a single attachment without any auxiliary files.
- Opened in any modern web browser without Python or any other software installed.
- Archived alongside the raw data with no risk of broken image links.

---

#### 22.4 PDF output

Pass a `.pdf` extension to get PDF output:

```python
report_path = generate_report(
    result, formula, factors, power_cfg,
    output_path="design_report.pdf",
)
```

This requires the `weasyprint` package, which in turn has system-level
dependencies (Cairo, Pango) that vary by operating system.

**When to use PDF vs. HTML:**

| Situation | Recommendation |
|---|---|
| Sharing by email, archiving in a study folder | **HTML** — universally openable, no dependencies |
| Submitting to a regulatory authority or journal | **PDF** — fixed pagination, printable |
| Embedding in a Streamlit or Jupyter notebook | **HTML** — easier to iframe or link |
| Long-term archive with no Python environment | **HTML** — the file is self-describing |

---

#### 22.5 Full worked example

**Step 1 — Build the design and generate the report in one call.**

```python
from iopt_power_design import (
    i_optimal_powered_design, PowerContrastConfig, DesignOptions,
)

formula  = "A + B + A:B"
factors  = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
power_cfg = PowerContrastConfig(
    L=[[0, 1, 0, 0]], delta=[1.0], sigma=2.0,
    alpha=0.05, power=0.80, max_n=60,
)
opts = DesignOptions(starts=5, random_state=42)

result = i_optimal_powered_design(
    formula, factors, power_cfg, opts,
    export_report_to="design_report.html",
)
print(result["report"]["n"])           # → 39 (or similar)
print(result["report"].get("report_path"))  # → resolved path of written file
```

The file `design_report.html` is written to the current directory.
Open it in any browser to see the full design summary.

**Step 2 — Generate a richer report separately with the power curve.**

```python
from iopt_power_design import generate_report
from pathlib import Path

report_path = generate_report(
    result,
    formula,
    factors,
    power_cfg,
    output_path=Path("reports") / "design_with_curve.html",
    title="Polymer Process Optimisation — Design Report",
    include_power_curve=True,   # embed power-vs-n figure
    design_rows_shown=20,       # show first 20 rows of the design table
)
print(f"Report written to: {report_path}")
# → Report written to: /path/to/reports/design_with_curve.html
```

The `reports/` directory is created automatically if it does not exist.
The embedded power-curve figure shows power vs. n for the configured effect
and sigma assumptions, making the report self-documenting.

**Step 3 — Generate a PDF for submission.**

```python
# Requires: pip install "iopt-power-design[report-pdf]"
pdf_path = generate_report(
    result, formula, factors, power_cfg,
    output_path="design_report.pdf",
    title="Design Report — Protocol Submission",
    include_power_curve=False,   # skip figure for cleaner PDF layout
)
print(f"PDF written to: {pdf_path}")
```

**Embedding in a CLI pipeline.**
When running in a batch script, the return value of `generate_report` can be
logged or passed to the next step:

```python
import subprocess, sys

report_path = generate_report(result, formula, factors, power_cfg,
                               output_path="output/report.html")
# Open in the default browser (macOS / Linux)
subprocess.run(["open" if sys.platform == "darwin" else "xdg-open",
                str(report_path)])
```

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
