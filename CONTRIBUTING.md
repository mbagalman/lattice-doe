# Contributing to Lattice DOE

Thanks for your interest in improving Lattice DOE. This guide covers the
development setup, the quality gates every change must hold, and what a
good pull request looks like here.

## Development setup

```bash
git clone https://github.com/mbagalman/lattice-doe.git
cd lattice-doe
pip install -e ".[all]"            # or pick extras: [cli], [server], [report], [viz], ...
pip install pytest pytest-cov pytest-anyio
```

Python 3.9–3.12 are supported and tested in CI. The core package needs only
numpy/scipy/pandas/patsy; everything else is an optional extra guarded with
`try/except ImportError`, and the test suite must pass **without** any extras
installed (soft-dependency tests skip cleanly).

`weasyprint` (PDF export) is deliberately not part of `[all]`-style dev
setups here: it needs system libraries, and one test asserts the clean
ImportError guard when it is absent.

## Running the tests

```bash
pytest -m "not slow"       # fast suite (~1,750 tests, ~20 min)
pytest                     # full suite (adds 127 slow integration tests)
pytest tests/test_api.py   # one module
```

- Slow tests live only in `test_api.py`, `test_api_server.py`,
  `test_contrasts.py`, and `test_split_plot.py`, marked `@pytest.mark.slow`.
- Coverage is **opt-in**: `pytest -m "not slow" --cov=lattice_doe`.
  It is intentionally not in the default pytest options — with it there,
  any pytest invocation (even `--collect-only`) silently overwrites
  `.coverage`. The weekly `slow-suite` CI workflow owns the authoritative
  coverage number.
- Async API-server tests use `pytest-anyio` and run on both asyncio and
  trio backends. If you add an async test class, give it its own
  `@pytest.mark.anyio` decorator — and take care not to insert a new class
  between an existing decorator and its class.

## Quality gates (CI-enforced)

All of these run in CI on every push/PR and must hold:

| Gate | Command | Expectation |
|---|---|---|
| black | `black --check --line-length 100 lattice_doe/` | clean (the package is fully formatted) |
| ruff | `ruff check lattice_doe/` | zero findings |
| mypy | `mypy lattice_doe/` | no-grow baseline (see below) |
| tests | `pytest -m "not slow"` | green on Python 3.9–3.12 |

The mypy count is **measurement-environment-dependent** (installed typed
libraries change it), so CI pins the exact tool and dependency versions in
the `gates` job of `.github/workflows/ci.yml` and compares against
`MYPY_BASELINE` there. Two rules:

1. Never let the count grow.
2. If your change *lowers* a count, ratchet the corresponding
   `RUFF_BASELINE`/`MYPY_BASELINE` value in `ci.yml` down **in the same PR**.

## Code conventions

- Google-style docstrings; MIT license header at the top of every source file.
- `UPPER_CASE` constants, `PascalCase` classes, `snake_case` functions,
  `_private` for internals.
- Inputs are typed `@dataclass` configs, not raw dicts.
- Every stochastic function accepts `random_state` for reproducibility.
- Python 3.9 is the floor: no PEP 604 pipe unions (`X | Y`) in annotations —
  they break `typing.get_type_hints()` on 3.9 even under
  `from __future__ import annotations`, and a guard test scans for them.
  Names used in annotations must be importable at runtime (no
  `TYPE_CHECKING`-only annotation imports on public callables); soft-dep
  types get an `Any` fallback binding in the ImportError branch.
- Comments referencing ticket IDs (`SR-28`, `UX-57`, `TD-9`, …) are
  deliberate provenance pointing into `docs/planning/ENHANCEMENTS.md` —
  keep them; they have repeatedly been what made later bugs diagnosable.
- New call sites of the design builder pass `factors=` and never derive
  categorical columns locally (see the TD-9 note in `ENHANCEMENTS.md`).

## Tests for your change

- Prefer **exact-value regression tests** over smoke tests: pin computed
  numbers (tolerances of `1e-9` where float math allows), exact counts,
  exact error messages. Pin the *contract*, not whatever the current
  implementation happens to produce.
- Bug fixes need a test that fails on the pre-fix code.
- Optional-dependency code paths need both sides tested: the working path
  (skip-guarded with the module's `_HAS_*` flag or `pytest.importorskip`)
  and the degradation path (patch the flag off).
- No mocking of the core DOE computation — use real calculations at small
  problem sizes (`candidate_points=100`, `starts=1`, small `n`).

## Pull requests

- Keep commits focused; explain **why** in the message body, not just what.
- Run the fast suite and the three gates locally before opening the PR.
- If your change closes a ledgered finding, update its row in
  `docs/planning/ENHANCEMENTS.md` with the evidence.
- CI must be fully green — all four Python versions, gates, and the wheel
  smoke-install.

## Where things live

| Path | What |
|---|---|
| `lattice_doe/` | core package (api, power, search, analysis, connectors) |
| `lattice_doe/api_server/` | FastAPI REST server (`[server]` extra) |
| `lattice_doe/app/` | Streamlit UI (`[app]` extra; excluded from mypy by design) |
| `tests/` | the suite (~1,880 tests incl. slow) |
| `docs/` | quickstart, recipes, full user guide |
| `docs/planning/ENHANCEMENTS.md` | the ledger: roadmap, tickets, every review finding with its evidence trail |
