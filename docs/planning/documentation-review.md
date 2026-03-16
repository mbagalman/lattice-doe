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

## Final conclusion

The project is already credible as an open-source MIT-licensed package, especially for technically comfortable users. The remaining work is not foundational; it is a polish pass.

If the goal is for outside users to quickly trust the package, understand how it works, and contribute confidently, the next step should be a documentation and style cleanup sprint rather than more structural refactoring.
