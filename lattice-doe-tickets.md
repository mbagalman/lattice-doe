# Lattice DOE — Consolidated Ticket Backlog

Synthesized from four AI reviews (ChatGPT, Grok, Venice, Gemini), two
meta-reviews of the synthesis itself (Grok meta-review, ChatGPT meta-review),
plus a direct look at the GitHub repo. This is the working ticket document;
hand it to a human or to an AI coding agent.

Repo state observed (Nov 2026): v0.1.0 in `pyproject.toml`, but no GitHub
release and not on PyPI. MIT, single contributor, 0 stars / 0 forks, Python
98.8% with `tests/`, `docs/`, `app/`, `api_server/`, Dockerfiles. The
package surface area is already large (linear/R²/GLM power, multi-response,
blocked, split-plot, CLI, Streamlit, API server, Sheets, Excel, reports).
The gap is not features — it's distribution, trust, and usability.

## Strategic framing

**The most valuable next phase is to make the core workflow boringly
trustworthy, not to add features.** ChatGPT's original review made the
case; Grok's meta-review reinforced it; ChatGPT's second meta-review then
pointed out that the first attempt to write that down still over-scoped
the first sprint.

This version reflects all of that. P0 is split into:
- **P0a** — first-impression blockers. Without these, the README example
  fails or looks unprofessional. This is the v0.1.0 release.
- **P0b** — trust infrastructure. CI matrix, statistical tests, docs
  site, profiling baseline, hygiene. This is v0.1.1.

PyPI name reservation moves into P0a; actual publication waits for the
end of P0a once the package install cleanly.

---

## Goals and non-goals for the first public release (v0.1.0)

### Goal
Ship a v0.1.0 to PyPI that a curious data scientist can `pip install`,
copy the README quick-start, and have it work the first time. The package
already has the math; this release just has to make the math reachable
without obstacles.

### Must have
- README imports work and match published package name
- Clean install from wheel
- Quick-start smoke test passes in CI
- Minimal CI green (install + pytest + wheel build)
- Basic troubleshooting docs
- `Typing :: Typed` classifier resolved (kept with `py.typed`, or removed)
- Patch-history comments removed
- Changelog / license / package metadata clean
- PyPI name reserved

### Should have for v0.1.0 (slip to v0.1.1 if needed)
- Better exceptions (TICKET-008)
- Diagnostic interpretations (TICKET-009)
- Candidate adequacy report (TICKET-017)
- Minimal docs site (TICKET-011A)

### Not in scope before first public release
- Simulation-based power validation
- Sequential / adaptive design workflow
- Guided wizard
- Numba / JAX accelerated backend
- Ray / Dask distributed multi-start
- New GLM families (Gamma, Negative Binomial)
- Bayesian / robust optimal design
- Mixture / constrained-mixture designs
- Response Surface Methodology helpers
- scikit-learn / statsmodels transformer
- R / JMP export
- Audience-specific report templates
- Baseline comparison helper
- 8–10 example notebook gallery (skeleton only for v0.1.0)

These are good ideas. None of them are first-release blockers, and any
one of them can absorb a sprint. They wait until P1 or later.

---

## Ticket dependency map

```
PyPI release (TICKET-022) requires:
  └─ TICKET-001 (naming)
  └─ TICKET-003A (minimal CI)
  └─ TICKET-004 (typing classifier)
  └─ TICKET-006A (test reorg)
  └─ TICKET-007 (PyPI name reserved)

Performance band (TICKET-035 onward) gated on:
  └─ TICKET-012 (profiling baseline)

Streamlit polish (TICKET-020) depends on:
  └─ TICKET-008 (typed exceptions for clean error display)

Audience reports (TICKET-019) depend on:
  └─ TICKET-009 (diagnostic interpretation text)
  └─ TICKET-017 (candidate adequacy report)

Wizard (TICKET-029) depends on:
  └─ TICKET-015 (model-matrix introspection)
  └─ TICKET-020 (Streamlit polish)

Sequential workflow (TICKET-024) depends on:
  └─ TICKET-023 (simulation power) for validating refits

API removal (TICKET-002C) depends on:
  └─ TICKET-002A (policy)
  └─ TICKET-002B (deprecation warnings shipped for ≥1 minor release)
```

---

## Ticket template

P0a and P0b tickets use the full template below because they are imminent
and likely to be handed to a coding agent. P1+ tickets use a lighter
format with acceptance criteria but skip "Files likely touched" and
"Risk level" — those become useful when a ticket is pulled into a sprint.

```
Ticket ID:
Title:
Priority:
Release target:
Depends on:
Why:
Tasks:
Acceptance criteria:
Out of scope (for this ticket):
Files likely touched:
Risk level:
```

---

## P0a — First-impression blockers (v0.1.0)

These must ship before the first PyPI release. None should take more than
a day or two each.

### TICKET-001: Fix naming/import inconsistencies (`iopt_power_design` → `lattice_doe`)
- **Status:** Done — 2026-05-06. The rename had already landed in prior commits (no `iopt_power_design` references remained anywhere in source, README, docs, or Dockerfiles). The README quick-start was failing in a fresh environment for an unrelated reason (search overshoots candidate set with default `max_n=2000`); fixed by hardening the README example with `sesoi=2.0` and `max_n=50`. The README import smoke-test sub-task is now covered by [tests/test_readme_smoke.py](tests/test_readme_smoke.py), wired into CI via TICKET-003A.
- **Priority:** P0a
- **Release target:** v0.1.0
- **Depends on:** none
- **Sources:** ChatGPT
- **Why:** The GitHub-rendered README still shows examples importing from
  `iopt_power_design` in places, while the package itself is `lattice_doe`.
  First-time users who hit a `ModuleNotFoundError` on the very first
  example never come back. This is the single highest-impact small fix.
- **Tasks:**
  - Grep the entire repo (including `docs/`, `app/`, `api_server/`,
    Dockerfiles, screenshots, notebooks) for `iopt_power_design`, old
    Docker image names, and old app references
  - Replace with `lattice_doe` everywhere, or explicitly mark old name
    as legacy with a deprecation note
  - Confirm one canonical import path appears in every example:
    `from lattice_doe import find_optimal_design, PowerContrastConfig, DesignOptions`
  - Add a smoke test that imports each example block from the README
- **Acceptance criteria:**
  - `grep -R "iopt_power_design" .` returns no active code or
    documentation references (excluding changelog / deprecation notes)
  - README quick-start runs from a clean virtual environment
  - `python -c "from lattice_doe import find_optimal_design"` succeeds
    after wheel install
  - CI includes a README smoke test that fails on regression
- **Out of scope:** API surface trim (TICKET-002), advanced typing checks
- **Files likely touched:** `README.md`, `docs/*.md`, `app/`,
  `api_server/`, `Dockerfile*`, all notebooks
- **Risk level:** Low

### TICKET-003A: Minimal CI workflow
- **Status:** Done — 2026-05-06. Added [.github/workflows/ci.yml](.github/workflows/ci.yml) with two parallel jobs (test + build/smoke-install) on Python 3.11, plus [tests/test_readme_smoke.py](tests/test_readme_smoke.py). The smoke test runs both during the regular test pass and again against the freshly-built wheel in a fresh venv. Will only go green on the remote once pushed; the failure mode it catches has already been verified locally (caught the README example bug noted in TICKET-001).
- **Priority:** P0a
- **Release target:** v0.1.0
- **Depends on:** none
- **Sources:** ChatGPT, Grok, Grok meta-review, ChatGPT meta-review
- **Why:** No GitHub Actions visible. Even a minimal "install, test, build"
  workflow catches regressions. The full CI matrix (lint, type, security,
  release) is split into TICKET-003B and TICKET-003C — don't block the
  first release on those.
- **Tasks:**
  - GitHub Actions workflow `ci.yml` running on push and PR
  - Steps: install package, run pytest, build wheel and sdist
  - Smoke install from the built wheel
  - README smoke test (from TICKET-001)
- **Acceptance criteria:**
  - GitHub Actions passes on Python 3.11 minimum (single version is fine
    for v0.1.0; full matrix waits for TICKET-003B)
  - Wheel installs in a fresh environment
  - `lattice --help` succeeds after install
  - pytest runs without requiring optional `[app]`/`[report]`/`[server]`
    extras
- **Out of scope:** Lint, mypy, coverage badges, dependency scanning,
  test matrix across Python versions, auto-publish
- **Files likely touched:** `.github/workflows/ci.yml`,
  possibly `pyproject.toml`
- **Risk level:** Low

### TICKET-004: Confirm or remove the `Typing :: Typed` claim
- **Status:** Done — 2026-05-06. Mypy reported 183 errors across 18 files on the current package, with many of them on the public API surface (e.g. `analysis.py` union-attr issues across `PowerContrastConfig | PowerR2Config`, missing return annotations in `cli.py`, undefined name imports in multi-response paths). That's well beyond the ticket's "quick" scope, so per the AC's second branch the `Typing :: Typed` classifier was removed from `pyproject.toml`. The `[tool.mypy]` config block is left in place as a north-star setting for future cleanup. Re-add the classifier when a real type-cleanup pass lands.
- **Priority:** P0a
- **Release target:** v0.1.0
- **Depends on:** none
- **Sources:** ChatGPT
- **Why:** The package metadata advertises typing support, but type
  coverage is uneven. Either commit to it or stop claiming it.
- **Tasks:**
  - Add `py.typed` marker file if missing
  - Run mypy on the public API surface; fix glaring issues
  - If the bar is too high to clear quickly for v0.1.0, drop the
    classifier and add it back when ready
- **Acceptance criteria:**
  - Either: `py.typed` is present and mypy passes on the public API
    (TICKET-002A's defined surface)
  - Or: `Typing :: Typed` classifier removed from `pyproject.toml`
- **Out of scope:** Pydantic migration (don't drag this in unless you
  have a concrete pain point — the existing dataclass validation may be
  perfectly adequate); type cleanup of internal modules
- **Files likely touched:** `pyproject.toml`, `lattice_doe/py.typed`,
  selected `lattice_doe/*.py` for type fixes
- **Risk level:** Low

### TICKET-005: Remove patch-history comments
- **Status:** Done — 2026-05-06. Swept all `# ADDED:`, `# CHANGED:`, `# FIXED:`, `# MODIFIED:`, `# CR-NN:`, and "this path should be unreachable" comments from `lattice_doe/` (9 source files). Deleted purely-redundant markers and rewrote the few that captured non-obvious WHY (rank-deficient contrast, R² sigma kept for symmetry, blocked-design Patsy column count). Verified with 293 targeted tests passing.
- **Priority:** P0a
- **Release target:** v0.1.0
- **Depends on:** none
- **Sources:** ChatGPT
- **Why:** ChatGPT flagged inline comments like `# ADDED:`, `# CHANGED:`,
  and "This path should be unreachable" in API validation and CLI parsing.
  These read like working notes, not production code, and quietly tell
  readers the codebase is mid-flight.
- **Tasks:**
  - Sweep the codebase for `# ADDED`, `# CHANGED`, `# FIXED`, `# TODO`,
    `# unreachable`
  - Either delete (if obvious from context) or rewrite as timeless
    explanations of intent
- **Acceptance criteria:**
  - `grep -RnE "# (ADDED|CHANGED|FIXED|REMOVED):" lattice_doe/` returns 0 lines
  - Surviving `# TODO` comments link to a tracked GitHub issue
  - Bad: `# ADDED: Early validation of formula, p, and max_n`
  - Better: `# Validate formula and parameter count before expensive candidate generation.`
- **Out of scope:** Broader refactoring; comment style guide enforcement
- **Files likely touched:** `lattice_doe/api.py`, `lattice_doe/cli.py`,
  `lattice_doe/iopt_search.py`, others as found
- **Risk level:** Very low

### TICKET-007: Reserve PyPI name
- **Priority:** P0a
- **Release target:** v0.1.0 (early)
- **Depends on:** none
- **Sources:** ChatGPT meta-review
- **Why:** If `lattice-doe` gets squatted, the entire naming cleanup
  becomes much more painful. Reserve early, publish later.
- **Tasks:**
  - Verify `lattice-doe` is available on PyPI
  - Either reserve via a placeholder upload (with a clear "name reserved,
    real release coming" notice) or confirm name availability and
    document the plan in the changelog
- **Acceptance criteria:**
  - `pip install lattice-doe` either fails with "no matching distribution"
    or installs a clearly-marked reservation package
  - The maintainer holds the name on PyPI before any P0b ticket lands
- **Out of scope:** Actual production release (that's TICKET-022)
- **Files likely touched:** None in repo; PyPI account configuration
- **Risk level:** Very low; high downside if skipped

### TICKET-008-mini: Basic troubleshooting docs
- **Priority:** P0a
- **Release target:** v0.1.0
- **Depends on:** none
- **Sources:** ChatGPT meta-review
- **Why:** Full typed-exception work (TICKET-008) is P0b. For v0.1.0,
  expand the README's existing Troubleshooting section so users have
  somewhere to land when the most common errors hit.
- **Tasks:**
  - Document the three most common failure modes:
    - "max_n must be greater than p" — what `p` means, how to fix
    - Contrast shape errors — how to inspect column count
    - Power did not converge — what `search_strategy` and warnings mean
  - Each entry should include: the exact error message, what it means,
    one or two concrete fixes, and a reference to the full Common Pitfalls
    doc once TICKET-010 ships
- **Acceptance criteria:**
  - README troubleshooting section covers the three failures above
  - Each entry has a working Python code example showing the fix
- **Out of scope:** Custom exception types (TICKET-008 in P0b)
- **Files likely touched:** `README.md` only
- **Risk level:** Very low

### TICKET-022-mini: Package metadata cleanup
- **Priority:** P0a
- **Release target:** v0.1.0
- **Depends on:** TICKET-001, TICKET-004
- **Sources:** ChatGPT meta-review
- **Why:** Before publishing to PyPI, all metadata should be clean.
- **Tasks:**
  - Verify `pyproject.toml` has correct: name, version, description,
    authors, license, classifiers, project URLs (homepage, docs, issues)
  - Verify LICENSE file present and correctly referenced
  - Add or update CHANGELOG.md with a v0.1.0 entry
  - Verify README renders correctly on TestPyPI
- **Acceptance criteria:**
  - `python -m build` produces wheel and sdist without warnings
  - TestPyPI upload renders README correctly
  - Project URLs all resolve
- **Out of scope:** Full release automation (TICKET-003C)
- **Files likely touched:** `pyproject.toml`, `LICENSE`, `CHANGELOG.md`,
  `README.md`
- **Risk level:** Low

### TICKET-022: Publish v0.1.0 to PyPI
- **Priority:** P0a (last ticket in P0a)
- **Release target:** v0.1.0
- **Depends on:** All other P0a tickets
- **Sources:** ChatGPT, Grok
- **Why:** With the cleanup landed, ship.
- **Tasks:**
  - Build wheels (`python -m build`)
  - Upload to TestPyPI; verify install works
  - Tag `v0.1.0`
  - Upload to PyPI
  - Announce in README install instructions
- **Acceptance criteria:**
  - `pip install lattice-doe` from a clean environment works
  - README quick-start runs end-to-end after that install
  - GitHub release created with changelog notes
- **Out of scope:** Auto-publish on tag (TICKET-003C, P0b)
- **Files likely touched:** Tag, GitHub release, PyPI; possibly
  `README.md` for install instructions
- **Risk level:** Medium (one-time public commitment)

---

## P0b — Trust infrastructure (v0.1.1)

These follow v0.1.0. They harden the package and set up the foundation
that future tickets depend on.

### TICKET-002A: Define and document API stability policy
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** none
- **Sources:** ChatGPT, Grok, ChatGPT meta-review
- **Why:** ChatGPT meta-review correctly flagged that silently shrinking
  `__init__.py` is risky even with no public users — good package hygiene
  starts now. Split TICKET-002 into policy → deprecation → removal.
- **Tasks:**
  - Decide and document the "happy path" public API: `find_optimal_design`,
    `DesignOptions`, the `Power*Config` classes, `contrast_from_scenarios`,
    `augment_design`, `generate_report`, `compare_criteria`
  - Document the planned submodule layout (avoid `experimental` as a
    junk drawer):
    - `lattice_doe.core` — main API
    - `lattice_doe.analysis` — power curves, sensitivity, MDE
    - `lattice_doe.contrasts` — contrast helpers
    - `lattice_doe.diagnostics` — design metrics
    - `lattice_doe.reports` — report generation
    - `lattice_doe.integrations` — Sheets, Excel, etc.
    - `lattice_doe.experimental` — used sparingly, with a clear policy
      on what graduates and how
  - Write `docs/api_stability.md` explaining the policy, semver
    commitments, and deprecation process
- **Acceptance criteria:**
  - Public API list documented and linked from README
  - `docs/api_stability.md` exists and explains the deprecation policy
  - `__all__` updated in `__init__.py` to match the documented surface
    (no removals yet — see TICKET-002B)
- **Out of scope:** Adding `DeprecationWarning` (TICKET-002B); actually
  removing exports (TICKET-002C)
- **Files likely touched:** `lattice_doe/__init__.py`, `docs/api_stability.md`
- **Risk level:** Low

### TICKET-002B: Add deprecation warnings for non-stable exports
- **Priority:** P0b
- **Release target:** v0.1.1 or v0.1.2
- **Depends on:** TICKET-002A
- **Sources:** ChatGPT meta-review
- **Why:** Silent removal is bad practice even with zero current users.
  A deprecation cycle is cheap and sets expectations.
- **Tasks:**
  - Add `DeprecationWarning` to top-level imports of objects scheduled
    for removal, pointing at the new submodule path
  - Document the timeline (e.g. "to be removed in v0.3.0")
- **Acceptance criteria:**
  - Importing a deprecated path emits a `DeprecationWarning` with the
    new path
  - Test that warnings fire from `pytest -W error::DeprecationWarning`
    (with intentional importing)
- **Out of scope:** Actual removal (TICKET-002C)
- **Files likely touched:** `lattice_doe/__init__.py`, possibly thin
  shim modules
- **Risk level:** Low

### TICKET-002C: Remove deprecated top-level exports (later)
- **Priority:** P1 or later
- **Release target:** v0.3.0 (no earlier than two minor releases after
  TICKET-002B)
- **Depends on:** TICKET-002B shipped for ≥1 minor release
- **Sources:** ChatGPT meta-review
- **Why:** Honor the deprecation cycle.
- **Tasks:**
  - Remove the top-level exports flagged in TICKET-002B
  - Update `__all__` to enforce the boundary
- **Acceptance criteria:**
  - Deprecated import paths raise `ImportError`
  - All in-repo code uses the new submodule paths
- **Out of scope:** Adding new deprecations
- **Files likely touched:** `lattice_doe/__init__.py`
- **Risk level:** Medium (intentional break)

### TICKET-003B: Full CI matrix — lint, type, coverage
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** TICKET-003A
- **Sources:** ChatGPT, Grok
- **Why:** Beyond minimal install/test, the package needs lint, type,
  and coverage signals on every PR.
- **Tasks:**
  - Test matrix: Python 3.9, 3.10, 3.11, 3.12
  - Add ruff + black --check
  - Add mypy on the public API (TICKET-002A's defined surface)
  - Coverage report; add badge to README; target ≥80% on public API
- **Acceptance criteria:**
  - All four Python versions green on a representative PR
  - ruff and black --check green
  - mypy clean on the public API surface
  - Coverage badge present and accurate
- **Out of scope:** Dependency scanning (TICKET-003C); release
  automation (TICKET-003D)
- **Files likely touched:** `.github/workflows/ci.yml`, `pyproject.toml`
- **Risk level:** Low

### TICKET-003C: Dependency security scanning
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** TICKET-003A
- **Sources:** Grok meta-review
- **Why:** Basic supply-chain hygiene for a numerics package.
- **Tasks:**
  - Add `pip-audit` (or `safety`) as a CI step
  - Enable Dependabot for `pip` and `github-actions` ecosystems
- **Acceptance criteria:**
  - `pip-audit` runs on every PR (failure is a warning initially, not
    a blocker, until the maintainer triages the existing tree)
  - Dependabot PRs land for outdated dependencies
- **Out of scope:** Aggressive dependency upgrades; SBOM generation
- **Files likely touched:** `.github/workflows/ci.yml`,
  `.github/dependabot.yml`
- **Risk level:** Very low

### TICKET-003D: Release automation
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** TICKET-003A, TICKET-022
- **Sources:** ChatGPT meta-review
- **Why:** Auto-publish on git tag once the manual v0.1.0 is in place.
- **Tasks:**
  - GitHub Actions workflow that on tag push:
    - builds wheel and sdist
    - uploads to PyPI via OIDC trusted publishing
- **Acceptance criteria:**
  - Tagging `v0.1.1` causes PyPI release without maintainer intervention
- **Out of scope:** Pre-release / release candidate flow
- **Files likely touched:** `.github/workflows/release.yml`
- **Risk level:** Low (gated by tag; reversible)

### TICKET-006A: Reorganize tests into unit and statistical
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** none
- **Sources:** ChatGPT, ChatGPT meta-review
- **Why:** Current tests are integration-style with tiny candidate sets.
  Reorg first; layer regression and property tests on top later.
- **Tasks:**
  - Move existing tests into `tests/unit/` and `tests/integration/`
  - Add `pytest -m slow` marker for slow tests
  - Update CI to run unit on every PR; integration nightly or on demand
- **Acceptance criteria:**
  - `pytest tests/unit` runs in under 30 seconds
  - `pytest -m slow` runs the full integration set
  - No tests dropped during reorg
- **Out of scope:** New test cases (TICKET-006B); property tests (006C);
  literature fixtures (006D)
- **Files likely touched:** `tests/`, `pyproject.toml`
- **Risk level:** Low

### TICKET-006B: Statistical regression tests
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** TICKET-006A
- **Sources:** ChatGPT
- **Why:** Plumbing tests are not enough. Specific math claims need
  specific tests.
- **Tasks (cases from ChatGPT):**
  - Known 2×2 factorial returns expected balanced allocation
  - D-optimal design for simple linear model chooses endpoints
  - I-optimal design differs from D-optimal in a known polynomial case
  - `contrast_from_scenarios` produces same L as manual Patsy matrix subtraction
  - GLM approximation matches hand-calculated scalar-weight Wald power
  - Constraints do not silently remove categorical cells
  - Repeated `random_state` gives identical design and report
- **Acceptance criteria:**
  - Each case above has at least one test
  - Tests live under `tests/statistical/` and run on `pytest -m slow`
- **Out of scope:** Property-based tests (006C); literature fixtures (006D)
- **Files likely touched:** `tests/statistical/`
- **Risk level:** Low

### TICKET-006C: Property-based tests with Hypothesis
- **Priority:** P1
- **Release target:** v0.2
- **Depends on:** TICKET-006A
- **Sources:** Grok
- **Why:** Property tests are powerful but fiddly in numerical code.
  Don't block CI on them; layer on after the basics work.
- **Tasks:**
  - Add `hypothesis` to dev dependencies
  - Properties to encode:
    - Returned design has full column rank (or warning is raised)
    - `achieved_power ≥ target_power − tol` when `search_strategy`
      doesn't include max_n_hit
    - `len(design_df) == report["n"]`
    - `buckets_df["count"].sum() == report["n"]`
    - For split-plot: `eta=0` design power equals OLS design power on
      the same design
- **Acceptance criteria:**
  - Each property above has a Hypothesis test
  - Tests run on `pytest -m slow` (Hypothesis is known to be slow)
- **Out of scope:** Shrinking strategies, custom strategies for
  contrast matrices
- **Files likely touched:** `tests/property/`, `pyproject.toml`

### TICKET-006D: Literature-standard known-good fixtures
- **Priority:** P1
- **Release target:** v0.2
- **Depends on:** TICKET-006A
- **Sources:** Grok meta-review
- **Why:** Cheap credibility insurance. Designs from Montgomery's *Design
  and Analysis of Experiments* and Goos & Jones's split-plot examples
  with known power and D-efficiency become regression anchors.
- **Tasks:**
  - Create `tests/fixtures/known_designs/`
  - Encode 5–10 textbook designs with their published power /
    D-efficiency / reference
  - Test that the package reproduces the reported metrics within tolerance
- **Acceptance criteria:**
  - At least 5 fixtures, each citing the source textbook and page
  - All tests pass on a representative sample
- **Out of scope:** New algorithms to match a fixture; reproducing
  proprietary commercial-tool designs

### TICKET-007-stress: Stress tests for hard regimes
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** TICKET-006A
- **Sources:** Grok
- **Why:** Coverage gaps in p≈n, high cardinality, extreme η, complex
  constraints, rank-deficient L.
- **Tasks:**
  - Test matrix:
    - p/n ratios from 0.1 to 0.95
    - 1, 5, 10, 20 categorical levels
    - η ∈ {0, 0.01, 1, 10, 100}
    - Constraints excluding 0%, 50%, 95% of candidates
    - Rank-deficient L (rows linearly dependent on intercept after
      blocking)
- **Acceptance criteria:**
  - Each axis has at least one test that exercises the boundary
  - Failures produce informative warnings (per TICKET-009)
- **Out of scope:** Performance benchmarks (TICKET-038)
- **Files likely touched:** `tests/integration/test_stress.py`
- **Risk level:** Low

### TICKET-008: Better exception types and validation messages
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** none
- **Sources:** ChatGPT, Venice
- **Why:** Patsy errors are cryptic; current code raises generic
  `ValueError` for several distinct failure modes. Typed exceptions help
  users and downstream tools (Streamlit app error display).
- **Tasks:**
  - Define `lattice_doe.exceptions` with: `ContrastShapeError`,
    `RankDeficientError`, `InfeasibleDesignError`, `PowerNotAchievedError`,
    `AliasingError`, `CandidateExhaustionError`
  - Each carries actionable remediation in the message
  - Update Troubleshooting docs to map exception → fix
- **Acceptance criteria:**
  - All distinct failure modes raise their specific exception, not
    generic `ValueError`
  - Tests verify exception types
  - Each exception type has at least one mention in the troubleshooting
    docs with a fix
- **Out of scope:** Streamlit app integration (TICKET-020)
- **Files likely touched:** `lattice_doe/exceptions.py`,
  `lattice_doe/api.py`, `lattice_doe/iopt_search.py`,
  `lattice_doe/power.py`, `docs/troubleshooting.md`
- **Risk level:** Low

### TICKET-009: Defensive warnings + decision-oriented diagnostic interpretation
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** none
- **Sources:** ChatGPT, Grok, Venice, Grok meta-review
- **Why:** Warnings without interpretation produce alarming numbers users
  can't act on; interpretation without warnings hides issues. Tightly
  coupled, ship together. ChatGPT's meta-review called this "one of the
  highest-value tickets" — the right product instinct.
- **Tasks:**
  - Audit numerical hot paths (`iopt_search`, `power.py`, `split_plot.py`)
    for "this could go wrong" branches
  - Add explicit warnings (not silent fallbacks) when:
    - Condition number > 1e8 even after candidate growth
    - η outside (1e-6, 1e6)
    - Categorical factor has fewer than 2 levels in candidate set
    - Pseudoinverse path is hit instead of inverse
  - Surface GLM approximation caveat directly in returned `report` dict
    and HTML output:

        ```
        GLM power approximation: constant Fisher weight at baseline.
        Recommended validation: simulation if expected probabilities vary
        substantially across the design region.
        ```

  - Add `interpret_diagnostics(metrics)` helper emitting plain English:

        ```
        Condition number: 418
        Interpretation: moderate collinearity; coefficient estimates may
        be unstable.
        ```

  - Include interpretations in HTML report by default
- **Acceptance criteria:**
  - Each numerical warning is reachable from a test
  - `interpret_diagnostics(metrics)` returns a dict mapping metric name
    to (raw value, interpretation string)
  - HTML report displays interpretations
  - GLM caveat appears in the `report["warnings"]` list when GLM mode is
    used
- **Out of scope:** Simulation validation (TICKET-023); aliasing
  (TICKET-027)
- **Files likely touched:** `lattice_doe/iopt_search.py`,
  `lattice_doe/power.py`, `lattice_doe/split_plot.py`,
  `lattice_doe/diagnostics.py`, `lattice_doe/report.py`
- **Risk level:** Low–medium (changes user-visible output)

### TICKET-039: n-search must not request n greater than the candidate set size
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** Coordinates with TICKET-008 (typed `CandidateExhaustionError`) and TICKET-009 (warning channel); can ship before either if the exception/warning surface is wired in compatibly.
- **Sources:** Discovered 2026-05-06 while wiring up the README smoke-test (TICKET-003A).
- **Why:** When `max_n` exceeds `n_cand` (default `max_n=2000` in `PowerContrastConfig` vs default `cand_max=1000`), the bisection in the n-search can pick a midpoint `n` larger than `n_cand` and the Fedorov path raises a generic `ValueError` whose message ("Requested design size n=1003 exceeds the candidate set size n_cand=1000") doesn't tell the user how to fix it. The README quick-start tripped this without anyone noticing because the example never ran end-to-end in CI before TICKET-003A. The current README workaround is to set `max_n=50` explicitly, which is fine pedagogically but papers over the real bug — any user with a moderate effect size and default `max_n` will hit this crash.
- **Tasks:**
  - Trace where `n` is chosen in the bisection / growth strategy in `iopt_search.py` and `api.py`; identify every call site that can request `n > n_cand`.
  - Pick one of three resolutions and apply consistently:
    1. Clamp `n` to `n_cand` and emit a warning that the candidate set bounded the search (cheapest; coordinates with TICKET-009).
    2. Auto-grow the candidate set (regenerate with larger `candidate_points`) and re-run the search at the higher n.
    3. Raise typed `CandidateExhaustionError` (from TICKET-008) with concrete remediation text: "increase `candidate_points` to at least N, or reduce `max_n`."
  - Add a regression test that reproduces the discovered crash: `find_optimal_design` with default `PowerContrastConfig` (no `max_n` override) and `auto_candidate=True` on a problem whose true optimum is small.
  - After the fix, remove the `max_n=50` workaround from the README quick-start and from `tests/test_readme_smoke.py`; the smoke test should still pass.
  - Document the `max_n` ↔ `candidate_points` relationship in the user guide and in the `PowerContrastConfig` / `DesignOptions` docstrings.
- **Acceptance criteria:**
  - The README quick-start example with no `max_n` set either runs end-to-end OR fails with an actionable error message that names both `max_n` and `candidate_points` and tells the user which to change.
  - No code path in `iopt_search.py` requests `n > n_cand` from `build_i_opt_design_with_idx`.
  - Regression test for the original crash exists under `tests/` and runs in the fast suite.
  - `tests/test_readme_smoke.py` passes with the `max_n=50` workaround removed (assuming resolution 1 or 2; if resolution 3 is chosen, smoke test asserts the typed error and remediation message instead).
- **Out of scope:** Reworking the search algorithm beyond what's needed to handle this case; auto-tuning bisection bounds in general; changing `max_n` defaults.
- **Files likely touched:** `lattice_doe/iopt_search.py`, `lattice_doe/api.py`, `lattice_doe/exceptions.py` (if TICKET-008 has landed), `tests/test_iopt_search.py` or new `tests/test_search_bounds.py`, `README.md`, `tests/test_readme_smoke.py`, `docs/user-guide.md`.
- **Risk level:** Low–medium. Touches core search code, but the failure mode is a hard crash with a cryptic error — any of the three resolutions is strictly better than current behavior. Test coverage on the n-search bisection should be checked before/after.

### TICKET-010: Common Pitfalls section in docs
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** none
- **Sources:** Grok
- **Why:** Patsy formula encoding (T.high vs. C(...)), contrast rank,
  candidate-set explosion in high D — the three things every new user
  gets wrong.
- **Tasks:**
  - Add `docs/common_pitfalls.md`
  - Worked example for each of: dummy coding traps, contrast rank
    mismatches, high-D candidate explosion
  - Cross-link from README troubleshooting (TICKET-008-mini)
- **Acceptance criteria:**
  - Three worked examples, each with broken code, error message, and fix
- **Out of scope:** Wider docs revamp (TICKET-011A/B)
- **Files likely touched:** `docs/common_pitfalls.md`, `README.md`
- **Risk level:** Very low

### TICKET-011A: Minimal docs site
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** TICKET-002A
- **Sources:** Grok, Venice, Gemini, Grok meta-review, ChatGPT meta-review
- **Why:** Docs site is part of trust. Build the skeleton in P0b; defer
  the example notebook gallery to P1.
- **Tasks:**
  - Sphinx + autodoc + myst-parser
  - Convert existing `docs/quickstart.md` and `docs/recipes.md` into the
    site
  - Auto-generated API reference from docstrings on the public surface
    (TICKET-002A)
  - ReadTheDocs hosting (free for OSS)
  - Link prominently from README
- **Acceptance criteria:**
  - `docs/` builds with `make html` without warnings
  - ReadTheDocs hosts the site at a stable URL
  - API reference covers all public exports from TICKET-002A
- **Out of scope:** Example notebook gallery (TICKET-011B);
  domain-specific tutorials
- **Files likely touched:** `docs/conf.py`, `docs/index.md`, RTD config
- **Risk level:** Low

### TICKET-012: Profile and document hot spots
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** TICKET-006A
- **Sources:** Grok meta-review
- **Why:** Prevents premature optimization. The P4 performance band is
  gated on this ticket producing evidence.
- **Tasks:**
  - Run cProfile + line_profiler (or py-spy) on a realistic 8-factor
    screening case
  - Document top 3 hot spots in `docs/internals/profiling.md`
  - Capture wall-clock baselines for: 5-factor CCD, 8-factor screening,
    6-factor split-plot
- **Acceptance criteria:**
  - `docs/internals/profiling.md` exists with reproducible commands and
    captured timings
  - Each baseline includes Python version, platform, and `cProfile`
    output
- **Out of scope:** Actual optimization work (TICKET-035)
- **Files likely touched:** `docs/internals/profiling.md`,
  possibly `benchmarks/profile_baseline.py`
- **Risk level:** Very low

### TICKET-013: Project hygiene — CHANGELOG, CONTRIBUTING
- **Priority:** P0b
- **Release target:** v0.1.1
- **Depends on:** none
- **Sources:** Grok meta-review, ChatGPT meta-review
- **Why:** Even for a solo maintainer, these set expectations and make
  release management possible. ChatGPT meta-review correctly flagged that
  enforcing commitlint in CI is overkill for a solo repo — keep it as a
  request in CONTRIBUTING.
- **Tasks:**
  - `CHANGELOG.md` following Keep a Changelog format (already created in
    TICKET-022-mini; this ticket extends it)
  - `CONTRIBUTING.md` covering: how to pick up a ticket, expected test/docs
    bar, code style, how to run tests locally, request to use Conventional
    Commit prefixes (no CI enforcement)
- **Acceptance criteria:**
  - Both files exist and are referenced from README
- **Out of scope:** commitlint CI enforcement (skipped per ChatGPT
  meta-review)
- **Files likely touched:** `CHANGELOG.md`, `CONTRIBUTING.md`,
  `README.md`
- **Risk level:** Very low

---

## P1 — Practical usability (v0.2)

Lighter ticket format from here down. Acceptance criteria are still
specified; "Files likely touched" and "Risk level" are added at sprint
pull time.

> **Sequencing note:** TICKET-014 is high enough value that ChatGPT's
> meta-review suggests it could be late P0b or top P1. Pull forward if
> v0.1.0 ships smoothly.

### TICKET-014: User-supplied candidate sets
- **Sources:** ChatGPT
- **Depends on:** TICKET-002A (public API stability)
- **Why:** ChatGPT calls this "the single most practical feature for
  business users." Real-world DOE users in marketing, media, operations,
  pricing, and product experimentation often already know the feasible
  run combinations: markets, stores, customers, recipes, media plans,
  product variants. The current system insists on generating candidates
  from factor specs.
- **Tasks:**
  - Accept `candidates_df=` argument to `find_optimal_design`
  - Validate that supplied DataFrame has all factor columns and produces
    a full-rank model matrix
  - Document workflow with a marketing/operations example
  - Ensure constraint and feasibility logic still applies on top of
    user-supplied candidates (or skip cleanly if user signals "already
    filtered")
- **Acceptance criteria:**
  - `find_optimal_design(..., candidates_df=df)` works without
    requiring `factors`
  - Errors clearly if formula references columns missing from `df`
  - Report marks candidate strategy as `user_supplied`
  - Existing factor-generated workflow remains backward-compatible
  - At least one notebook example uses the feature

### TICKET-015: `describe_model_matrix()` and `explain_contrast()` helpers
- **Sources:** ChatGPT
- **Depends on:** none
- **Why:** Patsy formulas are opaque to many data scientists. Make
  scenario-based contrasts the primary recommended path.
- **Tasks:**
  - `describe_model_matrix(formula, factors)` →

        ```
        Model columns:
        0. Intercept
        1. A[T.high]
        2. B
        3. A[T.high]:B
        ```

  - `explain_contrast(L, column_names)` → human-readable description
  - De-emphasize manual L specification in README; lead with
    `contrast_from_scenarios` everywhere
- **Acceptance criteria:**
  - Both helpers exist in the public API
  - README and quickstart use scenario-based contrasts as primary path
  - Docstrings include examples

### TICKET-016: Constraint-aware resampling
- **Sources:** ChatGPT
- **Depends on:** none
- **Why:** Constraint filtering happens after candidate generation. If
  80% of candidates get filtered out, the user is silently left with a
  weak set. Constraints are exactly where lattice-doe is positioned to win.
- **Tasks:**
  - New options on `DesignOptions`: `target_feasible_candidates`,
    `max_constraint_resample_rounds`
  - After filtering, regenerate and re-filter until target met or rounds
    exhausted
  - Warn (don't fail) if target unreachable
  - Surface retained-vs-requested count in report
- **Acceptance criteria:**
  - Resampling kicks in when retained < target
  - Warning emitted when target unreachable after `max_rounds`
  - Report shows `requested_candidates`, `retained_candidates`,
    `resample_rounds_used`

### TICKET-011B: Example notebook gallery
- **Sources:** ChatGPT, Grok, Gemini
- **Depends on:** TICKET-011A
- **Why:** Domain examples convert browsers into users. Skeleton ships
  with v0.1.0 (link in README); real gallery is a v0.2 deliverable.
- **Tasks:**
  - 8–10 rendered Jupyter notebooks under `docs/examples/`:
    - Marketing creative test (headline × image × offer × audience)
    - Pricing experiment (price × discount × bundle × channel)
    - Product growth experiment (onboarding × message timing × incentive)
    - Manufacturing process optimization (split-plot with HTC oven temp)
    - Media mix / message testing
    - 8-factor screening with budget of 32 runs
    - Pharma / clinical trial with binomial response
    - Mixed continuous + categorical with feasibility constraints
    - Augmenting an existing design with new runs
    - Side-by-side vs. fractional factorial / random sampling
  - Notebooks executable via `nbsphinx` or `myst-nb`
- **Acceptance criteria:**
  - All notebooks build cleanly in docs CI (slow CI job, not blocking)
  - Each notebook has a "what you'll learn" preface
  - Each domain example uses realistic factor names and values

### TICKET-017: Candidate adequacy report
- **Sources:** ChatGPT
- **Depends on:** TICKET-009
- **Why:** Users need to know if the candidate set itself is the
  bottleneck.
- **Tasks:**
  - New report section:

        ```
        Candidate set:
        - requested candidates: 1,000
        - retained after constraints: 642
        - categorical cells represented: 18 / 18
        - minimum candidates per categorical cell: 21
        - max pairwise column correlation in candidate matrix: 0.08
        - warning: continuous factor coverage may be sparse for 9 dimensions
        ```

  - Add `candidate_strategy="lhs" | "full_factorial" | "sobol" | "grid" | "user_supplied"`
- **Acceptance criteria:**
  - Report includes the section above
  - Sobol option produces measurably better discrepancy than LHS on a
    test case
  - Adequacy warnings link to interpretation text from TICKET-009

### TICKET-018: Design comparison against common baselines
- **Sources:** ChatGPT
- **Depends on:** none
- **Why:** Turns the package from "a method" into "evidence that this
  method is better for my problem."
- **Tasks:**
  - `compare_against_baselines(formula, factors, power_cfg, baselines=[...])`
  - Output comparison table with runs, power, I-score, D-efficiency
  - Use in docs to make value proposition concrete
- **Acceptance criteria:**
  - Function exists in public API
  - At least 4 baseline strategies supported
  - Used in at least one notebook example

### TICKET-019: Audience-specific report templates
- **Sources:** ChatGPT, Grok
- **Depends on:** TICKET-009, TICKET-017
- **Why:** Different audiences need different reports.
- **Tasks:**
  - Three templates: technical, executive, experiment handoff sheet
  - Add `template=` argument to `generate_report`
  - Each template includes recommended analysis script (Python and R)
- **Acceptance criteria:**
  - All three templates render without error from the same `result` dict
  - Executive template fits on 1 page when printed
  - Handoff sheet has explicit run order and randomization notes

### TICKET-020: Streamlit app polish
- **Sources:** Grok
- **Depends on:** TICKET-008
- **Why:** Big accessibility win, currently lacks advanced options and
  has rough error UX.
- **Tasks:**
  - Surface every `DesignOptions` parameter
  - Better error display (using typed exceptions from TICKET-008)
  - Loading spinners for long searches
  - Save/load app state to YAML
- **Acceptance criteria:**
  - All `DesignOptions` parameters surfaced
  - Typed exceptions render with clear messages instead of stack traces
  - Save/load round-trips correctly

### TICKET-021: Export to R and JMP
- **Sources:** Grok
- **Depends on:** none
- **Why:** Many DOE practitioners live in R or JMP.
- **Tasks:**
  - `result.to_r_script(path)` — emits R script using `lm()`/`glm()`
    with the same formula
  - `result.to_jmp_script(path)` — emits JSL
  - Document round-trip workflows in the gallery
- **Acceptance criteria:**
  - R script runs cleanly in a fresh R session with `tidyverse` only
  - JMP script imports cleanly into JMP 17+
  - Both scripts reproduce the same coefficient estimates as the Python
    analysis path

---

## P2 — Credibility leap (v0.3)

### TICKET-023: Simulation-based power validation
- **Sources:** ChatGPT, Grok, Venice, Grok meta-review
- **Depends on:** TICKET-006B
- **Why:** All reviews flag this. ChatGPT calls it "the biggest
  credibility booster." Currently no automated way to check the
  analytical approximations.
- **Tasks:**
  - `simulate_power(design_df, formula, outcome=..., true_params=..., n_sim=1000)`
  - Returns analytical vs. simulated power with Monte Carlo SE
  - Add `validation/` regression suite running representative grid
    nightly in CI
  - Focus on: GLM with wide covariate ranges, split-plot with unbalanced
    WP, near-singular contrasts
- **Acceptance criteria:**
  - Empirical rejection rate matches reported `achieved_power` within
    tolerance for a representative grid
  - Function exposed in public API
  - At least one notebook compares analytical vs. simulated for GLM

### TICKET-024: Sequential / adaptive design workflow
- **Sources:** ChatGPT, Grok, Venice, Gemini, Grok meta-review
- **Depends on:** TICKET-023
- **Why:** All four original reviews and both meta-reviews flag this.
  Single most-requested real-world workflow.
- **Tasks:**
  - `sequential_design()` API: takes existing design + observed `y`,
    refits, re-optimizes next batch of `m` runs given updated β
  - GLM: re-evaluate Fisher weights at fitted values, not null baseline
  - ChatGPT's user-facing framing:

        ```python
        plan = create_initial_design(...)
        updated = augment_after_results(
            existing_design_df,
            remaining_budget=12,
            objective="improve_power" | "improve_prediction" | "resolve_uncertainty"
        )
        ```

  - Optional Bayesian update mode (TICKET-026)
- **Acceptance criteria:**
  - Function exists in public API
  - Updated design preserves rows from original
  - GLM Fisher weights are evaluated at refit β values (verified by test)
  - At least one notebook walks through pilot → augment → analyze

### TICKET-025: Broader GLM coverage
- **Sources:** Grok
- **Depends on:** TICKET-023
- **Why:** Current GLM is Wald-only with single null-baseline Fisher
  weight. Wide covariate ranges or strong slopes invalidate it.
- **Tasks:**
  - Add Gamma and Negative Binomial families
  - Per-point Fisher weights (not just null-baseline scalar) when user
    provides working β estimate
  - Split-plot GLM (currently skipped in some paths per Grok)
- **Acceptance criteria:**
  - All four GLM families pass simulation validation (TICKET-023)

### TICKET-026: Bayesian / robust optimal design
- **Sources:** Grok, Venice
- **Depends on:** TICKET-024
- **Why:** Users with prior information shouldn't have to pick a single
  null baseline.
- **Tasks:**
  - Accept prior on β (Normal default) in power configs
  - Compute expected information by quadrature or Monte Carlo
  - Document trade-offs vs. locally-optimal designs
- **Acceptance criteria:**
  - Bayesian-optimal design recovers locally-optimal when prior variance
    is small
  - Robustness improvement vs. locally-optimal demonstrated on a test case

### TICKET-027: Automated alias structure mapping
- **Sources:** Venice, Grok
- **Depends on:** TICKET-009
- **Why:** Users running near-fractional designs need to know what's
  confounded with what.
- **Tasks:**
  - `design_aliasing(design_df, full_formula, requested_formula)`
  - Half-normal plot helper for unreplicated screening
  - Integrate into diagnostics report
- **Acceptance criteria:**
  - Output identifies aliased terms for at least the standard
    Plackett-Burman fixture
  - Half-normal plot renders to matplotlib and plotly

### TICKET-028: Documentation pass for derivations and theory
- **Sources:** Venice
- **Depends on:** TICKET-011A
- **Why:** Sophisticated users benefit from seeing the math they trust.
- **Tasks:**
  - `docs/theory.md` with full derivations
  - Cite primary sources (Fedorov 1972, Goos & Jones 2011, Wald χ²
    locally-optimal-design literature)
  - Cross-link from API docstrings
- **Acceptance criteria:**
  - Each `Power*Config` class has a docstring link to the relevant
    theory section

---

## P3 — Adoption layer (v0.4)

### TICKET-029: Guided design wizard (CLI + Streamlit)
- **Sources:** ChatGPT, Gemini, Grok meta-review
- **Depends on:** TICKET-015, TICKET-020
- **Why:** ChatGPT's "Priority 1," Grok's meta-review calls it "the
  single highest-UX win." Users who understand experiments but not DOE
  textbooks don't know which config class they need.
- **Tasks:**
  - `lattice wizard` CLI subcommand
  - Asks 6–8 questions about outcome type, goal, hard-to-change factors,
    blocking, smallest effect, noise level
  - Output: Python code, YAML config, design CSV, plain-English report
  - Streamlit version of the same wizard
- **Acceptance criteria:**
  - CLI wizard outputs valid YAML that the CLI can execute
  - Streamlit wizard reaches the same outputs
  - Wizard handles all five power modes (contrast, R², GLM binomial,
    GLM Poisson, multi-response)

### TICKET-030: scikit-learn / statsmodels integration
- **Sources:** Venice
- **Depends on:** none
- **Why:** Users with sklearn pipelines want the design matrix as a
  transformer.
- **Tasks:**
  - `LatticeDesignTransformer` exposing `.fit(X)` / `.transform(X)`
  - Helper returning a `statsmodels` model fitted to a hypothetical y
- **Acceptance criteria:**
  - Transformer works inside a `sklearn.pipeline.Pipeline`
  - Round-trips through `joblib.dump` / `load`

### TICKET-031: Mixture and constrained-mixture designs
- **Sources:** Grok
- **Depends on:** none
- **Why:** Common in formulation science. Broadens the user base.
- **Tasks:**
  - Mixture factor type (components sum to 1, individually bounded)
  - Simplex-lattice and simplex-centroid candidate generators
  - Scheffé canonical polynomials in the formula API
- **Acceptance criteria:**
  - Mixture design reproduces a known textbook simplex-lattice example

### TICKET-032: Response Surface Methodology helpers
- **Sources:** Venice
- **Depends on:** none
- **Why:** Natural extension once you have an optimal design.
- **Tasks:**
  - `lattice_doe.rsm` module
  - Stationary-point analysis (canonical analysis for second-order)
  - Local + global optimization in design region with constraints
- **Acceptance criteria:**
  - Recovers known stationary point on a textbook RSM example

### TICKET-033: Model-robustness compound criterion
- **Sources:** Grok
- **Depends on:** none
- **Why:** Designs optimal for a fitted model can be terrible if the
  model is wrong.
- **Tasks:**
  - Compound criterion taking primary formula + alternatives with weights
  - Optimize weighted combination of I/D-criteria
- **Acceptance criteria:**
  - Compound design measurably outperforms primary-only design when the
    true model includes the alternative terms

### TICKET-034: Consistent progress reporting + VS Code/Jupyter polish
- **Sources:** Grok
- **Depends on:** none
- **Why:** Lower-priority developer experience.
- **Tasks:**
  - Unify progress reporting through single helper; respect
    verbose/quiet uniformly
  - VS Code snippets for common config patterns
  - Polish Jupyter widgets

---

## P4 — Performance & scale (deferred, gated on TICKET-012)

### TICKET-035: Optional accelerated backend (numba or JAX)
- **Sources:** Grok, Venice
- **Depends on:** TICKET-012
- **Why:** Fedorov hot loop is `O(max_iter × n_cand × p²)` per start.
  Painful at p ≥ 10 with large candidate sets.
- **Tasks:**
  - Use TICKET-012's profile data to confirm actual bottleneck
  - Provide optional `pip install lattice-doe[accel]` extra
  - Numba JIT or JAX for inner exchange loop, NumPy fallback
- **Acceptance criteria:**
  - At least 5x speedup on the 8-factor screening baseline from
    TICKET-012
  - NumPy fallback produces identical designs

### TICKET-036: Coordinate-exchange and hybrid algorithms
- **Sources:** Grok
- **Depends on:** TICKET-012
- **Why:** `algo="coordinate"` is currently a no-op.
- **Tasks:**
  - Implement true coordinate exchange (continuous-only inner)
  - Optional Bayesian-optimization fallback for purely-continuous high-D
- **Acceptance criteria:**
  - Coordinate exchange beats Fedorov on at least one continuous-only
    benchmark in TICKET-038

### TICKET-037: Distributed multi-start (Ray / Dask)
- **Sources:** Grok
- **Depends on:** TICKET-012
- **Why:** Multi-machine multi-start for very large candidate sets.
- **Tasks:**
  - Optional `lattice-doe[distributed]` extra
  - Backend abstraction so `workers=` dispatches to local pool, Ray, or Dask
- **Acceptance criteria:**
  - Same designs produced on local pool vs. Ray for fixed `random_state`
  - Documented setup cost vs. wall-time trade-offs

### TICKET-038: Benchmark suite vs. dexpy / pyDOE2 / R AlgDesign
- **Sources:** Grok
- **Depends on:** TICKET-012
- **Why:** No public benchmarks against established tools.
- **Tasks:**
  - `benchmarks/` directory with repeatable scripts
  - Standard problems: 5-factor CCD, 8-factor screening, mixed
    categorical/continuous
  - Track D-efficiency, A-efficiency, I-criterion, wall time
  - Publish results in docs
- **Acceptance criteria:**
  - At least three external tools benchmarked on at least three problem
    types
  - Results reproduce in CI on a known fixture

---

## Recommendations not adopted (or where I'd diverge from the source review)

- **Venice's "agentic coordination" framing.** The repo has nothing of
  the kind — it's a deterministic Fedorov exchange. Venice appears to
  have been hedging without repo access.

- **Venice's "Variance Handling: use Student's t."** The package
  operates on noncentral F and χ² distributions for power, which already
  account for finite-sample df correctly. Substituting t-distributions
  would be wrong, not a fix.

- **Venice's "Matrix Validation: orthogonality checks for fractional
  designs."** The package does *optimal* design, not classical fractional
  factorial. Orthogonality is a constraint of the latter, not a goal of
  the former. The closer analog is alias-structure mapping (TICKET-027),
  which is kept.

- **Gemini's recommendations** are correct in spirit but generic. Each
  is subsumed by a more specific ticket.

- **Grok's "publish to PyPI immediately" framing.** Overruled by ChatGPT's
  case for cleaning up first; both meta-reviews agreed. Reserving the
  name immediately (TICKET-007) splits the difference.

- **Custom Fedorov vs. external library replacement.** Grok suggests
  benchmarking against dexpy, pyDOE2, R AlgDesign (TICKET-038, kept).
  No reviewer suggested *replacing* the custom implementation, and the
  integration with power search and split-plot GLS is the value
  proposition. Benchmark, don't rewrite.

- **commitlint enforcement in CI.** ChatGPT meta-review flagged this as
  overkill for a solo repo. CONTRIBUTING note requesting Conventional
  Commits is enough; CI enforcement deferred indefinitely.

- **Pydantic v2 migration for config classes.** ChatGPT meta-review
  flagged this as dragging in a public dependency without a concrete
  pain point. Removed from TICKET-004. Existing dataclass validation may
  be perfectly adequate; revisit only if validation complexity grows.

- **`lattice_doe.experimental` as a primary submodule.** ChatGPT
  meta-review correctly warned this can become a junk drawer. TICKET-002A
  uses it sparingly with a clear graduation policy.

- **Full ticket template (every field) for every ticket.** ChatGPT
  meta-review proposed a 12-field template. Applied in full to P0a/P0b
  where execution is imminent; lighter format for P1+ to avoid ceremony.
  When a P1+ ticket is pulled into a sprint, it should be expanded to the
  full template at that point.
