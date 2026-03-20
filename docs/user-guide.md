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

- 1.1 The problem: choosing where to run experiments
- 1.2 Optimality criteria — what they measure and why it matters
  - **I-optimality**: minimising average prediction variance across the design region; the right choice when you want reliable predictions anywhere in the factor space
  - **D-optimality**: maximising the determinant of the information matrix; the right choice when you care most about precise coefficient estimates
  - **A-optimality**: minimising the trace of the inverse information matrix; distributes estimation variance evenly across all coefficients
  - When the three criteria disagree, and how much it usually matters in practice
- 1.3 Power assurance — what it means to "guarantee" power
  - Statistical power defined: the probability of detecting an effect that is real
  - Why design and power are coupled: the same design determines both prediction quality and test sensitivity
  - How the package searches: the binary search over `n`, the inner Fedorov exchange, and the outer multi-start
- 1.4 The four power modes at a glance
  - Linear contrast (F-test on Lβ = δ)
  - Global R² (omnibus F-test)
  - GLM Wald χ² (binomial / Poisson)
  - Multi-response (per-response + combination rule)
- 1.5 The seven interfaces at a glance — a map for new users
  - Python API, CLI, Streamlit, Excel, Google Sheets, Jupyter Widgets, REST API
  - "Which interface should I use?" decision guide

---

### Chapter 2 — Installation and project layout

- 2.1 Python version requirements (≥ 3.9)
- 2.2 Installing the core package from source
- 2.3 Optional extras and when you need them
  - `[cli]` — YAML config support for the command-line tool
  - `[viz]` — Matplotlib + Plotly for power curve figures
  - `[app]` — Streamlit web UI
  - `[report]` — self-contained HTML report generation (Jinja2 + Pillow)
  - `[report-pdf]` — PDF export via WeasyPrint
  - `[extras]` — tqdm progress bars, openpyxl for Excel output
  - `[widgets]` — ipywidgets + Plotly for in-notebook interactive UI
  - `[all]` — everything at once
- 2.4 Verifying the install: a one-line smoke test
- 2.5 Project layout overview: `iopt_power_design/`, `app/`, `api_server/`, `docs/`

---

## Part II — Power Modes

*Each chapter in this part covers one power mode: the statistical concept, the configuration class, and a realistic end-to-end example. Examples build in complexity from chapter to chapter.*

---

### Chapter 3 — Linear contrasts: detecting a specific effect

**Running example:** A polymer chemistry lab is testing whether reaction temperature (continuous, 150–250 °C) and catalyst type (categorical: A / B / C) affect yield. The team wants 80% power to detect a 0.5 standard-deviation shift in yield attributable to temperature.

- 3.1 What a contrast is: L, δ, and the F-test for Lβ = δ
  - The model matrix X and its columns
  - The contrast matrix L: selecting one coefficient, combining several, or comparing treatment groups
  - δ: the minimum effect size worth detecting, in response units / σ
  - The noncentrality parameter λ = δᵀ [L(X'X)⁻¹Lᵀ]⁺ δ / σ² and where it comes from
- 3.2 Setting up `PowerContrastConfig`
  - Counting model-matrix columns from a Patsy formula (the most common source of confusion)
  - Constructing L manually for a single main-effect contrast
  - Setting δ, α, power, σ, and max_n
- 3.3 `contrast_from_scenarios`: building L and δ from two experimental scenarios
  - When to use scenarios vs. manual L
  - The `sesoi` parameter (smallest effect of interest) and its units
- 3.4 Running `i_optimal_powered_design` and reading the result
  - `design_df`: the run table
  - `buckets_df`: unique run allocations with counts
  - `report`: achieved power, n, noncentrality λ, degrees of freedom, timing
- 3.5 Multi-contrast tests: q > 1 rows in L
  - Joint F-test interpretation
  - Example: testing all treatment contrasts simultaneously
- 3.6 **Full worked example** (Python API, I-optimal, contrast mode, polymer chemistry)
  - Annotated code from formula definition through result interpretation
  - *(This example is also used as the baseline in Chapters 19–21)*

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
