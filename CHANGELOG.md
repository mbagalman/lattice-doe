# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

Initial public release of `lattice-doe`.

### Added

- Power-assured optimal design search for linear models, R² tests, and GLMs (logistic, Poisson)
- I, D, and A optimality criteria via Fedorov exchange
- Multi-response designs with `min`, `product`, and `weighted_mean` power combination, plus joint Hotelling T² power
- Blocked designs and split-plot (hard-to-change factor) designs with η-controlled GLS information
- Declarative constraint expressions (`constraint_expr`), AST-validated against a restricted operator set
- Scenario-based contrast construction (`contrast_from_scenarios`)
- Top-level Python API: `find_optimal_design`, `find_multiresponse_design`, `compare_criteria`, `power_curve_*`, `min_detectable_effect`, `robustness_report`, `augment_design`, `generate_report`
- Command-line interface (`lattice`) with YAML/JSON config support and a `--template` scaffold generator
- Streamlit multi-page web UI, packaged with the `[app]` extra (`lattice-app`)
- FastAPI REST server (`lattice-api`)
- HTML/PDF report generation, diagnostic plots, and CSV/JSON diagnostic exports
- Google Sheets and Excel workbook integrations
- Jupyter `ipywidgets` UI for interactive design exploration

### Fixed

- The sample-size search no longer crashes when `max_n` exceeds the candidate
  set size (e.g. default `max_n=2000` with a 1000-point pool requested a
  1000+-run design on the first bisection step and raised `ValueError`). The
  search upper bound is now capped at what the candidate pool supports; if the
  capped search cannot reach the target power, a warning names both `max_n`
  and `candidate_points`, and a pool too small for the smallest estimable
  design raises an actionable error up front. (TICKET-039)
- Streamlit app: values entered on one page are no longer lost when
  navigating to another page and back.
- CLI output is forced to UTF-8 on Windows so `lattice --help` and design
  summaries render instead of raising `UnicodeEncodeError` on legacy consoles.

[0.1.0]: https://github.com/mbagalman/lattice-doe/releases/tag/v0.1.0
