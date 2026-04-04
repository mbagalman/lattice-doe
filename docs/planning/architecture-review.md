# Architecture Review

Date: 2026-03-16

Reviewed against Git commit: `97b7abd`

## Overall assessment

The project architecture is generally sound. The statistical core is separated from the interface layers well enough that the package does not need a redesign or rewrite.

The main architectural pressure is not in the core math modules. It is in the growing number of interface and adapter layers that each rebuild the same request objects in their own way. That duplication is now the clearest source of technical debt and the most likely cause of future feature drift.

## Summary judgment

- No full re-architecture is required.
- Some targeted refactoring is now warranted.
- The highest-value refactor is a shared request/config translation layer used by all interfaces.
- The second highest-value refactor is sharing more code between the Google Sheets and Excel connectors.

## What looks good

### 1. The core package still has a sensible dependency shape

The package mostly follows a healthy pattern:

- statistical and search logic in `lattice_doe`
- interface adapters in `app` and `api_server`
- connector-specific integrations in dedicated modules

The earlier `design.py` split appears to have helped. Candidate generation, model-matrix generation, and search logic are no longer collapsed into one monolith, and the compatibility wrapper is clearly marked as such.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/design.py:1`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/docs/planning/design-refactor.md:1`

### 2. The interface layers depend on the core rather than the reverse

This is an important architectural strength. The Streamlit app, FastAPI layer, widgets, Sheets, and Excel all call into the main package rather than embedding the statistical algorithms directly.

That means the project is still refactorable without major breakage.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/api_server/routers/design.py:1`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/app/pages/3_Run_Results.py:1`

## Refactoring that is now required

### 1. Centralize request/config construction across interfaces

This is the clearest refactor that should happen rather than wait indefinitely.

Right now, multiple interfaces independently construct:

- `PowerContrastConfig` / `PowerR2Config` / `PowerGLMContrastConfig`
- `MultiResponseOptions`
- `DesignOptions`
- split-plot / blocked / pre-allocation options

That logic currently lives in several places:

- Streamlit run page
- Streamlit analysis page
- widgets
- CLI
- API-server serialization layer
- Google Sheets parser
- Excel parser

This duplication is the architectural reason new features can land in the core but miss one or more interfaces.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/app/pages/3_Run_Results.py:79`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/app/pages/4_Analysis.py:74`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/widgets.py:100`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/cli.py:250`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/api_server/serialization.py:77`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/sheets.py:341`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/excel_template.py:293`

Recommended direction:

- Introduce a shared internal request-building module.
- Give every interface one job: collect input, call the shared builder, then call the API.
- Keep interface-specific validation only where the transport actually differs.

### 2. Reduce duplication between Sheets and Excel connectors

The Google Sheets and Excel integrations follow the same sentinel-based configuration model and appear to implement near-parallel parse/build/write flows in separate files.

That is workable, but expensive to maintain. Every new option or response mode tends to require synchronized edits in both connectors.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/sheets.py:1`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/excel_template.py:1`

Recommended direction:

- Extract a shared config schema parser independent of the storage backend.
- Let Sheets and Excel provide only the backend-specific read/write mechanics.
- Share response/result export helpers where practical.

## Refactoring that is not urgent, but is a good idea

### 3. Split large orchestration modules before they grow much further

The package has several large coordination-heavy modules:

- `api.py`
- `analysis.py`
- `iopt_search.py`

These are not broken, but they are large enough that continued feature growth will make them harder to reason about and test.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/api.py:128`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/api.py:964`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/analysis.py:53`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/analysis.py:1549`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/iopt_search.py:1`

Possible next split:

- `api_single.py`
- `api_multiresponse.py`
- `analysis_curves.py`
- `analysis_sensitivity.py`
- `analysis_multiresponse.py`

### 4. Narrow the top-level package export surface

`lattice_doe.__init__` currently re-exports a very broad mix of:

- core statistical APIs
- low-level helpers
- reports
- Sheets and Excel connectors
- widget helpers

This is convenient, but it also broadens coupling and makes the public API harder to reason about.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/lattice_doe/__init__.py:14`

This is not urgent, but over time it would be cleaner to keep the package root focused on the core analytical surface and let optional integrations live behind explicit submodule imports.

## Refactoring that does not appear necessary

### 5. A full rewrite or radical package reorganization

I do not see evidence that the project needs:

- a rewrite
- a service-oriented split
- a new framework
- a new package layout

The structure is still workable. The right move is to reduce duplication and keep interfaces thin, not to replace the overall architecture.

## Practical priority order

1. Build a shared request/config translation layer.
2. Share more parser/export logic between Sheets and Excel.
3. Split `api.py` and `analysis.py` into smaller orchestration modules.
4. Narrow the `__init__` export surface later if desired.

## Final assessment

The project architecture is good enough to support continued development. The core problem is not that the system is badly designed. It is that the number of interfaces has outgrown the amount of shared adapter code.

If the project makes only one architecture improvement soon, it should be the shared request/config builder. That refactor would likely remove the largest source of avoidable feature drift across the codebase.
