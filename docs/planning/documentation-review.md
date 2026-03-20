# Documentation Review

Date: 2026-03-16

Reviewed against Git commit: `f65da0b`

## Overall assessment

The project is in better shape than many early open-source packages. The core README is fairly strong, the codebase is generally readable, and there is already meaningful documentation for the main usage paths.

That said, the package is not yet at a fully polished public-library standard across style, formatting, commenting, and user-facing documentation. The main issues are inconsistency, a small amount of stale guidance, and some uneven coverage across public features.

## Executive summary

- Code style is broadly reasonable, but not fully formatter-clean.
- Commenting is better than average, but still contains development-history comments that should be cleaned before treating the codebase as polished open source.
- Documentation is strong for the main Python / CLI / Streamlit flows, but still uneven for some exported public features.
- No rewrite is needed; a targeted cleanup pass would go a long way.

## 1. Style and formatting review

### What looks good

- The repo declares clear style standards in `pyproject.toml`:
  - `black` with `line-length = 100`
  - `mypy` with `disallow_untyped_defs = true`
- Tabs do not appear to be in use in Python files.
- The codebase is generally readable and follows consistent Python naming conventions.

### Issues found

1. The repository is not fully aligned with its declared Black formatting target.
   - Static scan found many Python lines over 100 characters.
   - Representative examples:
     - `iopt_power_design/__init__.py:40`
     - `app/pages/3_Run_Results.py:308`
     - `iopt_power_design/cli.py:85`

2. There is still trailing whitespace in multiple files.
   - Representative example:
     - `iopt_power_design/power_curves.py:58`

3. Formatting consistency is good enough for active development, but not yet at the level where a contributor can assume formatter-clean code across the repo.

### Recommendation

- Run a focused formatting cleanup pass before promoting the project as polished open source.
- Make `black --check` and a trailing-whitespace check part of CI if they are not already.

## 2. Commenting review

### What looks good

- Many core modules have strong module docstrings and function docstrings.
- The project often documents statistical assumptions and API intent well.
- Complex modules such as `iopt_power_design/api.py`, `iopt_power_design/power.py`, and `iopt_power_design/_request_builder.py` are significantly easier to follow because of their explanatory docstrings.

### Issues found

1. Some comments read like internal patch notes rather than durable documentation.
   - Examples include:
     - `# ADDED`
     - `# CHANGED`
     - inline ticket references such as `CR-17`
   - These are useful during active development but should not remain the dominant style in a public-facing codebase.

2. The code is better at sectioning than at explaining difficult control flow.
   - There are many divider comments and section headers.
   - In some large orchestration files, new contributors would still benefit from more explanation of why certain branches exist and how the control flow is intended to work.

3. Documentation coverage is not uniform.
   - Module and function docstring coverage is decent overall, but class-level documentation is less consistent.

### Recommendation

- Remove or rewrite development-history comments into durable explanatory comments.
- Prefer comments that explain:
  - why a branch exists
  - why a tradeoff was chosen
  - what assumption a block of code relies on
- Add docstrings to public-facing classes and exceptions where still missing.

## 3. User-facing documentation review

### What looks good

- `README.md` is fairly strong and does real work as the main onboarding document.
- The README covers:
  - installation
  - Python quick start
  - CLI
  - Streamlit
  - GLM mode
  - split-plot mode
  - reports
  - Google Sheets integration
- `docs/quickstart.md` is useful for getting started quickly.
- `docs/recipes.md` contains practical examples for common analytical tasks.

### Issues found

1. At least one onboarding doc is stale.
   - `docs/quickstart.md` still lists `"mean"` as a multi-response combination rule.
   - The implemented rules are `"min"`, `"product"`, and `"weighted_mean"`.

2. Coverage is uneven across exported public features.
   - The package root exports:
     - Excel helpers
     - Google Sheets helpers
     - widget helpers
   - But top-level documentation coverage is stronger for some of these than others.
   - Google Sheets is documented in the README; Excel and widgets are less visible in the main documentation set.

3. Recipes do not yet reflect the full feature breadth of the package.
   - There are useful recipes for criteria comparison, augmentation, sensitivity, reporting, Plotly, and split-plot.
   - There are not yet equivalent task-oriented recipes for:
     - GLM design workflows
     - multi-response design workflows
     - spreadsheet-driven workflows

4. The main docs are solid for the primary path, but not yet fully sufficient as the public documentation set for an MIT-licensed open-source library with this many optional surfaces.

### Recommendation

- Fix the stale quickstart entry immediately.
- Add top-level documentation for Excel and widgets if those are intended public features.
- Add recipes for:
  - GLM powered design
  - multi-response design
  - spreadsheet-based workflows
- Consider adding a short “Which interface should I use?” section in the README for Python API, CLI, Streamlit, Sheets, Excel, widgets, and API server usage.

## 4. Open-source readiness summary

### Already in good shape

- Main README quality
- Core docstring coverage
- Clear package purpose
- Reasonable naming and module organization

### Still worth cleaning up before broader public promotion

- Black/style consistency
- Trailing whitespace and long-line cleanup
- Development-history comments in code
- Uneven documentation across public interfaces
- A few stale or incomplete examples in onboarding docs

## 5. Suggested priority order

1. Fix stale documentation examples.
2. Do a formatting cleanup pass and enforce it in CI.
3. Rewrite `ADDED` / `CHANGED` / ticket-style comments into durable explanations.
4. Expand public docs for Excel, widgets, and multi-response / GLM recipes.

## 6. Action tracking

| # | Action | Priority | Status | Notes |
|---|--------|----------|--------|-------|
| DR-1 | Fix stale `docs/quickstart.md` — remove `"mean"` from `power_combination` rules | Now | **Done** | Removed `"mean"`; valid values are `"min"`, `"product"`, `"weighted_mean"` |
| DR-2 | Rewrite/remove 29 `# ADDED` / `# CHANGED` / `CR-xx` comments in `api.py`, `cli.py`, `config.py`, `blocked.py` | Soon | Pending | |
| DR-3 | Add `black --check` and trailing-whitespace check to CI | CI | Pending | |
| DR-4 | Add recipes for GLM powered design, multi-response design, spreadsheet-based workflows | Sprint | Pending | |
| DR-5 | Add top-level docs for Excel and widgets interfaces | Sprint | Pending | |
| DR-6 | Add "Which interface should I use?" orientation section to README | Later | Pending | |

## Final conclusion

The project is already credible as an open-source MIT-licensed package, especially for technically comfortable users. The remaining work is not foundational; it is a polish pass.

If the goal is for outside users to quickly trust the package, understand how it works, and contribute confidently, the next step should be a documentation and style cleanup sprint rather than more structural refactoring.

---

## Addendum — User Guide Review (2026-03-20)

Reviewed against Git commit: `b25e1f1`

### Overall assessment

The new `docs/user-guide.md` is a strong outline, but it is not yet a true user guide in the form you described.

Right now it behaves more like a curriculum map or table of contents for a future guide:

- it names the major capabilities,
- it suggests a sensible progression from simple to advanced topics,
- but it does not yet actually showcase each major capability with at least one worked example.

Because of that, it is promising as a structure, but not yet sufficient as an educational complement to the existing docs.

### User-guide-specific findings

#### UG-1 — High — The file is still mostly an outline, not a worked guide

The guide promises to explain the package from first principles, increase in complexity gradually, and cover every interface with realistic examples. In practice, almost all chapters are still bullet-point outlines and “Full worked example” placeholders rather than actual teaching content.

This is the single biggest gap relative to the intended purpose.

What is missing most:

- actual code examples in the chapters
- annotated outputs or screenshots where appropriate
- step-by-step transitions from simple use to advanced use
- explicit bridges back to README / quickstart / recipes for deeper reference

#### UG-2 — High — Several claims about current capabilities are inaccurate

Some sections describe capabilities the current codebase does not provide.

Examples:

1. The guide says “the six interfaces at a glance” but lists seven interfaces.
   - `docs/user-guide.md:32-34`

2. The widgets chapter says widget `power_mode` can be `"contrast"`, `"r2"`, or `"glm_binomial"` / `"glm_poisson"`.
   - The widget code documents and implements only `"r2"` and `"contrast"`.
   - `docs/user-guide.md:304`
   - `iopt_power_design/widgets.py:294-295`
   - `iopt_power_design/widgets.py:923-924`

3. The REST API chapter lists endpoint names that do not match the implemented routes.
   - Guide says:
     - `POST /multiresponse`
     - `POST /power_curve`
     - `POST /compare`
   - Code implements:
     - `POST /multiresponse_design`
     - `POST /power_curve/by_n`
     - `POST /power_curve/by_effect`
     - `POST /compare_criteria`
     - `POST /mde`
   - `docs/user-guide.md:323-329`
   - `api_server/routers/design.py:76`
   - `api_server/routers/power_curve.py:54`
   - `api_server/routers/power_curve.py:71`
   - `api_server/routers/compare.py:63`
   - `api_server/routers/sensitivity.py:93`

4. The split-plot chapter describes denominator-df methods that are not implemented.
   - Guide mentions Satterthwaite and Kenward-Roger options.
   - Code supports only `"auto"`, `"conservative"`, and `"sp_only"`.
   - `docs/user-guide.md:350`
   - `docs/user-guide.md:356`
   - `iopt_power_design/config.py:369`
   - `iopt_power_design/config.py:389`

#### UG-3 — Medium — The Streamlit chapter overstates analysis support

The Streamlit chapter says Page 4 supports power curves by baseline, multi-response analysis, and a broad GLM-friendly walkthrough.

Current code is narrower:

- `app/pages/4_Analysis.py` reconstructs only contrast and R² power configs.
- The results page explicitly says multi-response analytical power curves are not available in the UI and should be done from the Python API.

Relevant references:

- `docs/user-guide.md:243`
- `docs/user-guide.md:251`
- `app/pages/4_Analysis.py:74-107`
- `app/pages/3_Run_Results.py:560-567`

#### UG-4 — Medium — The Excel chapter describes the wrong workbook mental model

The guide describes the Excel template structure as “Config, Factors, Power, Responses, Output,” which implies multiple conceptual sheets/sections that do not match the actual sentinel-based Config sheet implementation.

The code uses a single `Config` sheet with sentinel sections:

- `[SETTINGS]`
- `[CONTRAST]`
- `[FACTORS]`
- `[RESPONSES]`

Relevant references:

- `docs/user-guide.md:262`
- `iopt_power_design/excel_template.py:10-24`

#### UG-5 — Medium — Appendix C’s interface matrix is not reliable yet

The comparison table appears to contain multiple incorrect feature assignments.

Examples:

1. It marks Widgets as supporting GLM mode, but widgets are contrast/R² only.
   - `docs/user-guide.md:580`
   - `iopt_power_design/widgets.py:294-295`

2. It marks Streamlit as not supporting blocking, but Streamlit exposes blocked design controls.
   - `docs/user-guide.md:583`
   - `app/state.py:43-46`
   - `app/pages/2_Power_Config.py:812-828`

Because this table is intended as a quick decision aid, inaccuracies here are especially risky.

#### UG-6 — Medium — The guide still needs true progression from simple to advanced

The chapter order is good, but the reader experience is not yet progressive in practice because the chapters do not actually walk through increasingly advanced examples.

For the guide to match the intended teaching arc, it should probably include at least:

1. one simple two-factor contrast example early,
2. one R² example,
3. one GLM example,
4. one multi-response example,
5. one advanced design-structure example (split-plot or blocking),
6. one interface-based example per major non-Python interface.

### Recommendation for the user guide

The best next step is not to add more chapter headings. It is to convert a subset of the existing outline into real instructional content.

Suggested implementation order:

1. Write Chapters 1–3 fully, including one real end-to-end contrast example.
2. Write one fully worked example each for:
   - R² mode
   - GLM mode
   - multi-response mode
3. Fix the inaccurate interface claims before expanding the rest.
4. Rebuild Appendix C from the actual code surface rather than from memory.
5. Treat the guide as a narrative layer above README/quickstart/recipes, not a repetition of them.

### Action tracking — User guide

| # | Action | Priority | Status | Notes |
|---|--------|----------|--------|-------|
| UG-1 | Convert `docs/user-guide.md` from outline to real instructional content with actual worked examples | Now | Pending | Highest-value improvement |
| UG-2 | Fix inaccurate capability claims in widgets, REST API, split-plot df-method, and interface count sections | Now | Pending | Several statements currently contradict the code |
| UG-3 | Rewrite Streamlit chapter to match current UI support, especially Page 4 analysis scope | Soon | Pending | Avoid promising UI features that require Python API instead |
| UG-4 | Correct the Excel chapter to describe the sentinel-based Config sheet structure | Soon | Pending | Current wording suggests the wrong workbook model |
| UG-5 | Rebuild Appendix C interface comparison table from implemented feature support | Soon | Pending | Current table appears partially inaccurate |
