# diagnostics.py Refactor — Development Plan & Ticket Pack

Tracks all work for **TD-2** (split `diagnostics.py` into `diag_metrics.py`, `diag_plots.py`, `diag_export.py`).

**Rules for contributors:**
1. Before starting a ticket, set its `Status` to `Claimed` and fill in `Claimed by`.
2. When done, check the box in the Dashboard and set `Status` to `Done`.
3. Never start work on a ticket marked `Claimed` by someone else — pick a different `Open` ticket or coordinate first.
4. If you hit a usage limit mid-ticket, leave a `Progress note` in the ticket card so the next session can continue without re-reading the whole codebase.

---

## Dashboard

| ID | Title | Epic | Status | Claimed by |
|----|-------|------|--------|------------|
| [A1](#a1-create-three-new-module-stubs) | Create three new module stubs | Infrastructure | Done | Claude |
| [B1](#b1-migrate-pure-numpy-helpers-to-diag_metricspy) | Migrate pure-NumPy helpers to `diag_metrics.py` | Metrics Migration | Done | Claude |
| [B2](#b2-migrate-compute_leverages-and-compute_design_metrics) | Migrate `compute_leverages` and `compute_design_metrics` | Metrics Migration | Done | Claude |
| [C1](#c1-migrate-create_diagnostic_plots-to-diag_plotspy) | Migrate `create_diagnostic_plots` to `diag_plots.py` | Plots Migration | Done | Claude |
| [D1](#d1-migrate-export_diagnostics-to-diag_exportpy) | Migrate `export_diagnostics` to `diag_export.py` | Export Migration | Done | Claude |
| [E1](#e1-convert-diagnosticspy-to-thin-re-export-wrapper) | Convert `diagnostics.py` to thin re-export wrapper | Wiring | Done | Claude |
| [E2](#e2-update-callers-and-__init__py) | Update callers and `__init__.py` | Wiring | Done | Claude |
| [F1](#f1-verify-tests-pass-and-imports-clean) | Verify tests pass and imports clean | Verification | Done | Claude |

**Progress:** 8 / 8 tickets done.

---

## Design Decisions

### Why split now?

`diagnostics.py` has grown to **663 lines** and conflates three concerns:

1. **Pure-NumPy metrics** — linear algebra helpers and metric computation that carry no plotting dependency.
2. **Matplotlib figures** — `create_diagnostic_plots` hard-requires matplotlib and pollutes the import graph.
3. **File I/O and export** — `export_diagnostics` mixes CSV/PNG/HTML writing with HTML template rendering.

Currently, importing `api.py` (which imports `diagnostics.py`) pulls in matplotlib even in headless or pure-computation contexts. Splitting into three focused modules removes this coupling and makes each part independently testable.

### Module responsibilities after the split

| Module | Contains | matplotlib dep? |
|--------|----------|-----------------|
| `diag_metrics.py` | Private NumPy helpers (`_xtx`, `_pinv`, `_has_intercept`, `_compute_vif`); `compute_leverages`; `compute_design_metrics` | No |
| `diag_plots.py` | `create_diagnostic_plots` | Yes (hard) |
| `diag_export.py` | `export_diagnostics` (CSV, PNG, HTML writers; inline HTML template) | Yes (soft — used for PNG sub-step only) |
| `diagnostics.py` | Thin re-export wrapper for backward compatibility | Inherited only |

### Backward compatibility strategy

`diagnostics.py` is converted into a **thin re-export wrapper** rather than deleted. All existing public symbols remain importable from the old path:

```python
# diagnostics.py (after refactor)
from .diag_metrics import compute_leverages, compute_design_metrics
from .diag_plots import create_diagnostic_plots
from .diag_export import export_diagnostics

__all__ = [
    "compute_leverages",
    "compute_design_metrics",
    "create_diagnostic_plots",
    "export_diagnostics",
]
```

Code that does `from iopt_power_design.diagnostics import compute_design_metrics` continues to work without modification. This wrapper can be removed in a future major version bump once all internal callers have been updated to import from the new modules.

### Internal callers to update

The following files import directly from `diagnostics` and should be updated to import from the specific new module in ticket E2:

- `iopt_power_design/api.py` — imports `compute_leverages`, `compute_design_metrics`, `create_diagnostic_plots`, `export_diagnostics`
- `iopt_power_design/__init__.py` — re-exports the four public symbols
- `tests/test_diagnostics.py` (if it exists) — update imports; should pass unchanged if using the `diagnostics` wrapper

### matplotlib import guard

`diag_plots.py` and `diag_export.py` should each carry their own try/except matplotlib guard at module level, matching the pattern already used in the current `diagnostics.py`:

```python
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
```

`diag_metrics.py` must have **no** matplotlib import — this is the core correctness invariant of the split.

---

## Current file map (diagnostics.py, 663 lines)

| Lines | Content | Destination |
|-------|---------|-------------|
| 1–46 | Module docstring, imports, matplotlib try/except guard | Split into each new module's own imports |
| 48–129 | `_xtx`, `_pinv`, `_has_intercept`, `_compute_vif` (private helpers) | `diag_metrics.py` |
| 132–150 | `compute_leverages` | `diag_metrics.py` |
| 153–246 | `compute_design_metrics` | `diag_metrics.py` |
| 249–445 | `create_diagnostic_plots` | `diag_plots.py` |
| 448–663 | `export_diagnostics` (+ inline HTML template string) | `diag_export.py` |

---

## Epic A — Infrastructure

Create the three new module files as stubs so that all subsequent tickets have a valid import target. No logic is moved yet.

---

### A1 Create three new module stubs

**Status:** Done
**Claimed by:** Claude
**Est.:** 30 minutes
**Depends on:** nothing
**Progress note:** Complete. `diag_metrics.py`, `diag_plots.py`, `diag_export.py` created as 3-line stubs with module docstring and `__all__`. All three import cleanly. 229 existing tests still pass (10 pre-existing failures unrelated to this work).

**What to do:**

1. Create `iopt_power_design/diag_metrics.py` with:
   ```python
   """Pure-NumPy diagnostic metrics — no matplotlib dependency."""
   __all__ = ["compute_leverages", "compute_design_metrics"]
   ```
2. Create `iopt_power_design/diag_plots.py` with:
   ```python
   """Matplotlib diagnostic figures."""
   __all__ = ["create_diagnostic_plots"]
   ```
3. Create `iopt_power_design/diag_export.py` with:
   ```python
   """Diagnostic file export (CSV, PNG, HTML)."""
   __all__ = ["export_diagnostics"]
   ```
4. Confirm all three are importable: `python -c "import iopt_power_design.diag_metrics, iopt_power_design.diag_plots, iopt_power_design.diag_export"` exits with no error.

**Acceptance criteria:**
- [ ] Three new `.py` files exist in `iopt_power_design/`.
- [ ] All three are importable without error.
- [ ] Existing tests still pass (`pytest tests/` is green).

---

## Epic B — Metrics Migration

Move the pure-NumPy code from `diagnostics.py` into `diag_metrics.py`. Do B1 then B2; they are sequential because B2's functions depend on the helpers moved in B1.

---

### B1 Migrate pure-NumPy helpers to `diag_metrics.py`

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** A1
**Progress note:** Complete. Done together with B2 in a single pass — see B2 note.

**What to do:**

Move the four private helper functions from `diagnostics.py` lines 48–129 into `diag_metrics.py`:

- `_xtx(X)` — computes `X.T @ X`
- `_pinv(M)` — computes the pseudoinverse of M
- `_has_intercept(X)` — detects an intercept column
- `_compute_vif(X)` — computes variance inflation factors

Steps:
1. Copy the four functions verbatim into `diag_metrics.py`.
2. Add any imports they need at the top of `diag_metrics.py` (likely only `numpy as np`).
3. Do **not** yet remove them from `diagnostics.py` — leave duplicates in place until ticket E1 converts `diagnostics.py` to the wrapper.
4. Run `pytest tests/` — must be green.

**Acceptance criteria:**
- [ ] All four private helpers exist in `diag_metrics.py` and are callable.
- [ ] `diag_metrics.py` imports only `numpy` (no matplotlib, no pandas unless strictly required).
- [ ] Tests still pass.

---

### B2 Migrate `compute_leverages` and `compute_design_metrics`

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** B1
**Progress note:** Complete. All content from `diagnostics.py` lines 48–246 copied verbatim into `diag_metrics.py`. One bug fixed during migration: `np.diag(Rinv)` returns a read-only view — added `.copy()` before the in-place infinity replacement. No matplotlib imports. Both public functions smoke-tested. 229 tests still pass.

**What to do:**

Move the two public metric functions from `diagnostics.py` lines 132–246 into `diag_metrics.py`:

- `compute_leverages(X)` — lines 132–150
- `compute_design_metrics(X, *, criterion="I", ...)` — lines 153–246

Steps:
1. Copy both functions into `diag_metrics.py` (they call the private helpers already in place from B1).
2. Update `__all__` in `diag_metrics.py` if not already correct.
3. Do **not** yet remove them from `diagnostics.py`.
4. Add a quick smoke-test import check: `from iopt_power_design.diag_metrics import compute_leverages, compute_design_metrics`.
5. Run `pytest tests/` — must be green.

**Acceptance criteria:**
- [ ] `compute_leverages` and `compute_design_metrics` are importable from `diag_metrics`.
- [ ] `diag_metrics.py` has no matplotlib import (enforce by running `grep -n "matplotlib" iopt_power_design/diag_metrics.py` — should be empty).
- [ ] Tests still pass.

---

## Epic C — Plots Migration

Move `create_diagnostic_plots` into its own module.

---

### C1 Migrate `create_diagnostic_plots` to `diag_plots.py`

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** B2
**Progress note:** Complete. `diag_plots.py` now contains the full `create_diagnostic_plots` function with its own matplotlib try/except guard. Internal calls to `_compute_vif`, `_has_intercept`, and `compute_leverages` import from `diag_metrics`. Smoke-tested (graceful None return when matplotlib absent). 229 tests still pass.

**What to do:**

Move `create_diagnostic_plots` from `diagnostics.py` lines 249–445 into `diag_plots.py`.

Steps:
1. Add the matplotlib try/except import guard at the top of `diag_plots.py` (see the guard pattern in the Design Decisions section above).
2. Copy `create_diagnostic_plots` into `diag_plots.py`.
3. The function calls `compute_leverages` and `compute_design_metrics`. Update the import to pull these from `diag_metrics`:
   ```python
   from .diag_metrics import compute_leverages, compute_design_metrics
   ```
4. Do **not** yet remove from `diagnostics.py`.
5. Run `pytest tests/` — must be green.

**Acceptance criteria:**
- [ ] `create_diagnostic_plots` is importable from `diag_plots`.
- [ ] `diag_plots.py` uses `from .diag_metrics import ...` (not a copy of the metric code).
- [ ] Tests still pass.

---

## Epic D — Export Migration

Move `export_diagnostics` (and the inline HTML template string it carries) into `diag_export.py`.

---

### D1 Migrate `export_diagnostics` to `diag_export.py`

**Status:** Done
**Claimed by:** Claude
**Est.:** 1–2 hours
**Depends on:** C1
**Progress note:** Complete. `diag_export.py` contains the full `export_diagnostics` function with its own matplotlib try/except guard. Imports `compute_design_metrics` from `diag_metrics` and `create_diagnostic_plots` from `diag_plots`. Inline HTML template preserved verbatim. Smoke-tested CSV path (no matplotlib required). 229 tests still pass.

**What to do:**

Move `export_diagnostics` from `diagnostics.py` lines 448–663 into `diag_export.py`.

This function carries an inline HTML template string and calls several internal helpers. Steps:

1. Add the matplotlib try/except import guard at the top of `diag_export.py` (same pattern — matplotlib is used only for PNG sub-output, so it must remain a soft dependency guarded by `_HAS_MPL`).
2. Copy `export_diagnostics` into `diag_export.py`, including any module-level constants it references (e.g., the inline HTML template string at the bottom of `diagnostics.py`).
3. Update cross-module imports inside the function:
   ```python
   from .diag_metrics import compute_leverages, compute_design_metrics
   from .diag_plots import create_diagnostic_plots
   ```
4. Do **not** yet remove from `diagnostics.py`.
5. Run `pytest tests/` — must be green.

**Acceptance criteria:**
- [ ] `export_diagnostics` is importable from `diag_export`.
- [ ] The inline HTML template is present in `diag_export.py` (not a dangling reference).
- [ ] PNG export still works when matplotlib is available.
- [ ] Tests still pass.

---

## Epic E — Wiring

Replace `diagnostics.py` with a thin re-export wrapper and update all internal callers to import from the correct new module.

---

### E1 Convert `diagnostics.py` to thin re-export wrapper

**Status:** Done
**Claimed by:** Claude
**Est.:** 30 minutes
**Depends on:** D1
**Progress note:** Complete. `diagnostics.py` replaced with 19-line re-export wrapper. All four public symbols importable from the old path. Backward compat verified.

**What to do:**

Replace the entire body of `diagnostics.py` with:

```python
"""
Backward-compatibility re-export wrapper.

All implementations have moved to:
  - diag_metrics.py  (compute_leverages, compute_design_metrics)
  - diag_plots.py    (create_diagnostic_plots)
  - diag_export.py   (export_diagnostics)

This module will be removed in a future major version.
"""
from .diag_metrics import compute_leverages, compute_design_metrics
from .diag_plots import create_diagnostic_plots
from .diag_export import export_diagnostics

__all__ = [
    "compute_leverages",
    "compute_design_metrics",
    "create_diagnostic_plots",
    "export_diagnostics",
]
```

Steps:
1. Overwrite `diagnostics.py` with the wrapper above.
2. Run `pytest tests/` immediately — everything must still pass (the wrapper preserves all public symbols at the old import path).
3. Confirm `diagnostics.py` is now under 20 lines.

**Acceptance criteria:**
- [ ] `diagnostics.py` is ≤ 20 lines.
- [ ] `from iopt_power_design.diagnostics import compute_leverages` still works.
- [ ] `from iopt_power_design.diagnostics import export_diagnostics` still works.
- [ ] All tests pass.

---

### E2 Update callers and `__init__.py`

**Status:** Done
**Claimed by:** Claude
**Est.:** 1 hour
**Depends on:** E1
**Progress note:** Complete. Two callers updated: `api.py:43` (`from .diagnostics import ...` → `from .diag_metrics import compute_design_metrics` + `from .diag_export import export_diagnostics`); `power_curves.py:187` (lazy import updated to `from .diag_metrics import compute_design_metrics`). `__init__.py` had no direct diagnostics import. Verified `matplotlib` not in `sys.modules` after `import iopt_power_design.api`.

**What to do:**

Update internal callers to import from the specific new modules (not via the wrapper). This eliminates the hidden matplotlib dependency from `api.py` imports.

**`iopt_power_design/api.py`:**
- Find all `from .diagnostics import ...` or `from iopt_power_design.diagnostics import ...` lines.
- Replace with imports from the appropriate new module:
  ```python
  from .diag_metrics import compute_leverages, compute_design_metrics
  from .diag_plots import create_diagnostic_plots
  from .diag_export import export_diagnostics
  ```

**`iopt_power_design/__init__.py`:**
- Find the re-export lines for diagnostic symbols.
- Update them to import from `diag_metrics`, `diag_plots`, and `diag_export` directly (or leave them pointing to `diagnostics` — the wrapper is acceptable here since `__init__.py` does not have the matplotlib-at-import problem).

**`tests/test_diagnostics.py`** (if it exists):
- Leave test imports pointing at `diagnostics` (the wrapper keeps them working). No changes required.

Steps:
1. Search for all import references: `grep -rn "from.*diagnostics import\|import.*diagnostics" iopt_power_design/`.
2. Update each reference in `api.py` as described above.
3. Run `pytest tests/` — must be green.
4. Verify matplotlib is no longer imported at `api.py` load time (in a headless env): `python -c "import sys; import iopt_power_design.api; print('matplotlib' in sys.modules)"` should print `False`.

**Acceptance criteria:**
- [ ] `api.py` imports diagnostic functions from `diag_metrics` / `diag_plots` / `diag_export`, not from `diagnostics`.
- [ ] `python -c "import sys; import iopt_power_design.api; print('matplotlib' in sys.modules)"` prints `False`.
- [ ] All tests pass.

---

## Epic F — Verification

Confirm the refactor is complete, clean, and correct.

---

### F1 Verify tests pass and imports clean

**Status:** Done
**Claimed by:** Claude
**Est.:** 30 minutes
**Depends on:** E2
**Progress note:** Complete. All checklist items verified: 229 tests pass (10 pre-existing failures unchanged); `diag_metrics.py` = 204 lines, `diag_plots.py` = 209 lines, `diag_export.py` = 228 lines, `diagnostics.py` = 20 lines; zero matplotlib import lines in `diag_metrics.py`; backward compat confirmed; `matplotlib` not in `sys.modules` after `import iopt_power_design.api`; ENHANCEMENTS.md updated to mark TD-2 resolved.

**What to do:**

Run the full verification checklist:

1. **Full test suite:** `pytest tests/ -v` — all tests green, no new warnings.
2. **Line count sanity:**
   - `wc -l iopt_power_design/diagnostics.py` — should be ≤ 20 lines.
   - `wc -l iopt_power_design/diag_metrics.py` — should be ~130–150 lines (helpers + 2 public functions).
   - `wc -l iopt_power_design/diag_plots.py` — should be ~200–210 lines.
   - `wc -l iopt_power_design/diag_export.py` — should be ~220–230 lines.
3. **No matplotlib in metrics:** `grep -n "matplotlib" iopt_power_design/diag_metrics.py` — must return nothing.
4. **Backward compat:** `python -c "from iopt_power_design.diagnostics import compute_leverages, compute_design_metrics, create_diagnostic_plots, export_diagnostics; print('OK')"` prints `OK`.
5. **Clean api.py import:** `python -c "import sys; import iopt_power_design.api; print('matplotlib' in sys.modules)"` prints `False`.
6. **Update ENHANCEMENTS.md:** Move TD-2 from the Technical Debt backlog table to a "Completed" note, or mark it resolved with a date.

**Acceptance criteria:**
- [ ] Full test suite passes.
- [ ] `diag_metrics.py` contains zero matplotlib references.
- [ ] `diagnostics.py` is a thin wrapper (≤ 20 lines).
- [ ] `api.py` no longer triggers a matplotlib import on load.
- [ ] ENHANCEMENTS.md updated to reflect TD-2 resolved.

---

## Suggested Build Order

```
A1
 ↓
B1 → B2
      ↓
      C1
       ↓
       D1
        ↓
        E1 → E2
              ↓
              F1
```

All epics are sequential — each depends on the previous epic being complete. No parallelism is available, but each ticket is small enough (30 min – 2 hrs) to complete in a single focused session.

**Estimated total:** 1–2 days (8 tickets × 30 min – 2 hrs each).
