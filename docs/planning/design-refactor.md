# TD-1: Split `design.py` — Development Plan & Ticket Pack

Tracks all work for **TD-1** (splitting `design.py` into focused modules).

**Rules for contributors:**
1. Before starting a ticket, set its `Status` to `Claimed` and fill in `Claimed by`.
2. When done, check the box in the Dashboard and set `Status` to `Done`.
3. Never start work on a ticket marked `Claimed` by someone else — pick a different `Open` ticket or coordinate first.
4. If you hit a usage limit mid-ticket, leave a `Progress note` in the ticket card so the next session can continue without re-reading the whole codebase.

---

## Dashboard

| ID | Title | Epic | Status | Claimed by |
|----|-------|------|--------|------------|
| [A1](#a1-create-candidatepy) | Create `candidate.py` | Candidate | Done | Claude |
| [B1](#b1-create-model_matrixpy) | Create `model_matrix.py` | Model matrix | Done | Claude |
| [C1](#c1-create-iopt_searchpy--criterion-core) | Create `iopt_search.py` — criterion core | Search core | Done | Claude |
| [D1](#d1-iopt_searchpy--multi-start--public-api--augmentation) | `iopt_search.py` — multi-start + public API + augmentation | Search orchestration | Done | Claude |
| [E1](#e1-replace-designpy-with-backward-compat-wrapper) | Replace `design.py` with backward-compat wrapper | Wrapper | Done | Claude |
| [F1](#f1-update-callers-to-import-from-new-modules) | Update callers to import from new modules | Callers | Done | Claude |
| [G1](#g1-update-tests--final-verification) | Update tests + final verification | Tests | Done | Claude |
| [G2](#g2-documentation-updates) | Documentation updates | Docs | Done | Claude |

**Progress:** 8 / 8 tickets done. ✅

---

## Design Decisions

### Module boundaries

`design.py` (1,267 lines) mixes four distinct concerns that map cleanly to new modules:

| New file | Responsibility | Key functions | Est. lines |
|----------|---------------|---------------|-----------|
| `candidate.py` | Factor-type helpers, adaptive candidate sizing, LHS + categorical candidate generation | `_is_continuous_spec`, `estimate_candidate_size`, `build_candidate` | ~335 |
| `model_matrix.py` | Patsy formula → numpy model matrix | `build_model_matrix` | ~30 |
| `iopt_search.py` | Fedorov exchange core, criterion scorers, multi-start orchestration, public build + augment | `_fedorov_exchange_single`, `_optimal_indices_from_X`, `_i/_d/_a_criterion_for_indices`, `_criterion_score`, `_score_design`, `_one_start_worker`, `build_i_opt_design_with_idx`, `build_i_opt_design`, `augment_design` | ~830 |
| `design.py` (wrapper) | Backward-compat re-exports only — **do not add new code here** | all of the above, re-exported | ~35 |

### Why `augment_design` goes in `iopt_search.py`

`augment_design` calls `estimate_candidate_size`, `build_candidate`, `build_model_matrix`, and `_score_design`. It belongs with the search orchestration code, not with candidate generation alone. Placing it in `iopt_search.py` keeps the dependency chain clean:

```
candidate.py  ←── model_matrix.py  ←── iopt_search.py
     ↑                  ↑                    ↑
  no design deps     no search deps      imports both
```

### `design.py` becomes a thin wrapper (same pattern as TD-2)

After the split, `design.py` is identical in spirit to the TD-2 `diagnostics.py` wrapper: it re-exports everything from the new modules so that any code importing `from .design import ...` continues to work without change. The wrapper is **temporary** — TD-3 will clean up remaining stale imports.

### `_score_design` is in `__all__` — keep it exported

`_score_design` is prefixed with `_` but appears in `__all__` in the current `design.py`. Keep it in `iopt_search.py`'s `__all__` and in the wrapper re-exports so existing callers (including `test_design.py`) are unaffected.

### `test_design.py` imports internal functions — update in G1

`test_design.py` imports `_i_criterion_for_indices`, `_d_criterion_for_indices`, `_a_criterion_for_indices`, `_criterion_score`, `_score_design` from `lattice_doe.design`. After E1 these work via the wrapper. After G1 the test file is updated to import from the canonical new modules.

---

## Current `design.py` line map

| Lines | Content |
|-------|---------|
| 1–87 | Module docstring, all imports, `_is_continuous_spec` |
| 89–243 | `estimate_candidate_size` |
| 245–412 | `build_candidate` |
| 418–435 | `build_model_matrix` |
| 440–581 | `_fedorov_exchange_single` (142 lines, vectorised exchange) |
| 584–646 | `_optimal_indices_from_X` (serial multi-start loop) |
| 649–663 | `_i_criterion_for_indices` |
| 665–695 | `_d_criterion_for_indices` |
| 698–730 | `_a_criterion_for_indices` |
| 733–785 | `_criterion_score` (dispatcher) |
| 788–846 | `_score_design` (direct-matrix scorer used by augment) |
| 848–874 | `_one_start_worker` (parallel worker entry point) |
| 880–1073 | `build_i_opt_design_with_idx` (main public function) |
| 1075–1130 | `build_i_opt_design` (thin wrapper; drops idx) |
| 1133–1256 | `augment_design` |
| 1259–1267 | `__all__` |

---

## Callers that import from `design.py`

| File | Symbols imported |
|------|-----------------|
| `api.py` | `estimate_candidate_size`, `build_candidate`, `build_model_matrix`, `build_i_opt_design_with_idx` |
| `power_curves.py` | `build_candidate`, `build_model_matrix`, `build_i_opt_design`, `build_i_opt_design_with_idx`, `estimate_candidate_size` |
| `contrasts.py` | `build_candidate`, `build_model_matrix` |
| `__init__.py` | `build_candidate`, `build_model_matrix`, `augment_design` |
| `tests/test_design.py` | `build_candidate`, `build_model_matrix`, `estimate_candidate_size`, `build_i_opt_design`, `build_i_opt_design_with_idx`, `_i_criterion_for_indices`, `_d_criterion_for_indices`, `_a_criterion_for_indices`, `_criterion_score`, `_score_design`, `augment_design` |

After E1 all callers still work (via wrapper). After F1 the production callers are updated to import from the canonical new modules. After G1 `test_design.py` is updated too.

---

## Epic A — Candidate module

---

### A1 Create `candidate.py`

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** nothing

**What to do:**

Create `lattice_doe/candidate.py` containing:

1. Module docstring describing candidate generation responsibility.
2. Imports: `from __future__ import annotations`, `typing`, `math`, `warnings`, `itertools`, `random`, `numpy`, `pandas`, `scipy.stats.qmc.LatinHypercube`.
3. Copy verbatim from `design.py`:
   - `_is_continuous_spec` (lines 77–83)
   - `estimate_candidate_size` (lines 89–243)
   - `build_candidate` (lines 245–412)
4. Add `__all__ = ["estimate_candidate_size", "build_candidate"]`.

**Invariants:**
- Zero imports from `design.py`, `model_matrix.py`, or `iopt_search.py`.
- `_is_continuous_spec` is a module-private helper — not in `__all__`.
- Do **not** delete anything from `design.py` yet.

**Verification:**
```python
from lattice_doe.candidate import estimate_candidate_size, build_candidate
cand = build_candidate({"x": (-1, 1), "y": (-1, 1)}, candidate_points=50, seed=0)
assert len(cand) == 50
```
Run `pytest tests/ -q --tb=no` — all 248 tests pass (no changes to design.py yet).

**Acceptance criteria:**
- [ ] `lattice_doe/candidate.py` exists and is importable.
- [ ] `estimate_candidate_size` and `build_candidate` produce identical output to calling them via `design.py`.
- [ ] No imports from `design`, `model_matrix`, or `iopt_search`.
- [ ] 248 tests pass.

---

## Epic B — Model-matrix module

---

### B1 Create `model_matrix.py`

**Status:** Open
**Claimed by:**
**Est.:** 15 minutes
**Depends on:** nothing (independent of A1)

**What to do:**

Create `lattice_doe/model_matrix.py` containing:

1. Module docstring.
2. Imports: `from __future__ import annotations`, `typing`, `numpy`, `pandas`, `from patsy import dmatrix`.
3. Copy verbatim from `design.py`:
   - `build_model_matrix` (lines 418–434)
4. Add `__all__ = ["build_model_matrix"]`.

**Invariants:**
- Zero imports from `design.py`, `candidate.py`, or `iopt_search.py`.
- Do **not** delete anything from `design.py` yet.

**Verification:**
```python
import pandas as pd
from lattice_doe.model_matrix import build_model_matrix
df = pd.DataFrame({"x1": [-1, 0, 1], "x2": [-1, 0, 1]})
X, names = build_model_matrix("x1 + x2", df)
assert X.shape == (3, 3)   # intercept + 2 cols
```
Run `pytest tests/ -q --tb=no` — 248 tests pass.

**Acceptance criteria:**
- [ ] `lattice_doe/model_matrix.py` exists and is importable.
- [ ] `build_model_matrix` output matches the current `design.py` version identically.
- [ ] No imports from `design`, `candidate`, or `iopt_search`.
- [ ] 248 tests pass.

---

## Epic C — Search module: criterion core

---

### C1 Create `iopt_search.py` — criterion core

**Status:** Open
**Claimed by:**
**Est.:** 1.5 hours
**Depends on:** A1, B1

**What to do:**

Create `lattice_doe/iopt_search.py` containing only the exchange algorithm, scoring helpers, and multi-start orchestration that **do not** require building a candidate set. Leave the public `build_*` functions and `augment_design` for D1.

Contents (copy verbatim from `design.py`):

1. Module docstring.
2. Imports:
   ```python
   from __future__ import annotations
   from typing import Any, Dict, Optional, Tuple, List
   import warnings
   import numpy as np
   import pandas as pd
   from concurrent.futures import ProcessPoolExecutor, as_completed
   from .candidate import estimate_candidate_size, build_candidate, _is_continuous_spec
   from .model_matrix import build_model_matrix
   ```
3. Functions (copy verbatim):
   - `_fedorov_exchange_single` (lines 440–581)
   - `_optimal_indices_from_X` (lines 584–646)
   - `_i_criterion_for_indices` (lines 649–663)
   - `_d_criterion_for_indices` (lines 665–695)
   - `_a_criterion_for_indices` (lines 698–730)
   - `_criterion_score` (lines 733–785)
   - `_score_design` (lines 788–846)
   - `_one_start_worker` (lines 848–874)
4. **Do not** include `build_i_opt_design_with_idx`, `build_i_opt_design`, or `augment_design` yet (those come in D1).
5. Add `__all__ = ["_score_design"]` (only `_score_design` needs to be exported for backward compat).

**Invariants:**
- Import `_is_continuous_spec` from `candidate.py`, not re-defined.
- Do **not** delete anything from `design.py` yet.

**Verification:**
```python
import numpy as np
from lattice_doe.iopt_search import (
    _i_criterion_for_indices, _d_criterion_for_indices,
    _criterion_score, _score_design,
)
X_cand = np.eye(5)
idx = np.array([0, 1, 2])
assert np.isfinite(_i_criterion_for_indices(X_cand, idx))
assert np.isfinite(_criterion_score("D", X_cand, idx))
```
Run `pytest tests/ -q --tb=no` — 248 tests pass.

**Acceptance criteria:**
- [ ] All 8 functions listed are in `iopt_search.py` with identical bodies to `design.py`.
- [ ] `_is_continuous_spec` imported from `candidate`, not copied.
- [ ] 248 tests pass.

---

## Epic D — Search module: multi-start + public API + augmentation

---

### D1 `iopt_search.py` — multi-start + public API + augmentation

**Status:** Open
**Claimed by:**
**Est.:** 1.5 hours
**Depends on:** C1

**What to do:**

Append to `lattice_doe/iopt_search.py` (open for editing — do **not** recreate):

1. Copy verbatim from `design.py`:
   - `build_i_opt_design_with_idx` (lines 880–1073)
   - `build_i_opt_design` (lines 1075–1130)
   - `augment_design` (lines 1133–1256)

2. Replace the lazy `from .config import DesignOptions as _DesignOptions` inside `augment_design` with a module-level import at the top of `iopt_search.py`:
   ```python
   from .config import DesignOptions
   ```
   Update the body of `augment_design` to remove the lazy import and use `DesignOptions` directly.

3. Update `__all__` in `iopt_search.py` to:
   ```python
   __all__ = [
       "build_i_opt_design",
       "build_i_opt_design_with_idx",
       "augment_design",
       "_score_design",   # used by test_design.py
   ]
   ```

**Lazy-import note:** `augment_design` in `design.py` currently uses a lazy import of `DesignOptions` inside the function body to avoid a circular import at module level. Verify there is no `iopt_search.py → config.py → iopt_search.py` cycle before moving the import to module level. Expected import chain: `config.py` does **not** import from `iopt_search.py`, so a module-level import is safe.

**Verification:**
```python
import numpy as np
from lattice_doe.iopt_search import (
    build_i_opt_design, build_i_opt_design_with_idx, augment_design,
)
from lattice_doe.candidate import build_candidate
import pandas as pd

cand = build_candidate({"x1": (-1,1), "x2": (-1,1)}, candidate_points=80, seed=0)
df = build_i_opt_design(cand, "x1 + x2", n=8, random_state=0)
assert len(df) == 8
```
Run `pytest tests/ -q --tb=no` — 248 tests pass.

**Acceptance criteria:**
- [ ] `build_i_opt_design_with_idx`, `build_i_opt_design`, `augment_design` are in `iopt_search.py`.
- [ ] `augment_design` uses a module-level `DesignOptions` import (no lazy import inside function body).
- [ ] `__all__` updated.
- [ ] 248 tests pass.

---

## Epic E — Replace `design.py` with backward-compat wrapper

---

### E1 Replace `design.py` with backward-compat wrapper

**Status:** Open
**Claimed by:**
**Est.:** 30 minutes
**Depends on:** A1, B1, D1

**What to do:**

Replace the entire content of `lattice_doe/design.py` with the following thin wrapper:

```python
"""
Backward-compatibility re-export wrapper.

All implementations have moved to:
  candidate.py     — _is_continuous_spec, estimate_candidate_size, build_candidate
  model_matrix.py  — build_model_matrix
  iopt_search.py   — Fedorov exchange, criterion scorers, build_i_opt_design*, augment_design

This module will be removed in a future major version (see TD-3).
Import from the canonical modules directly where possible.
"""
from .candidate import estimate_candidate_size, build_candidate
from .model_matrix import build_model_matrix
from .iopt_search import (
    _fedorov_exchange_single,
    _optimal_indices_from_X,
    _i_criterion_for_indices,
    _d_criterion_for_indices,
    _a_criterion_for_indices,
    _criterion_score,
    _score_design,
    _one_start_worker,
    build_i_opt_design_with_idx,
    build_i_opt_design,
    augment_design,
)

__all__ = [
    "estimate_candidate_size",
    "build_candidate",
    "build_model_matrix",
    "build_i_opt_design",
    "build_i_opt_design_with_idx",
    "_score_design",
    "augment_design",
]
```

**Critical check:** Run `pytest tests/ -q --tb=no` immediately after. All 248 tests must pass without any changes to callers — proof that the wrapper provides complete backward compatibility.

**Acceptance criteria:**
- [ ] `design.py` is ≤ 35 lines and contains only imports and `__all__`.
- [ ] `from lattice_doe.design import <anything>` still works for every symbol listed in the original `__all__`.
- [ ] Internal symbols (`_fedorov_exchange_single`, `_i_criterion_for_indices`, etc.) accessible via the wrapper for `test_design.py`.
- [ ] 248 tests pass with **zero** changes to any caller.

---

## Epic F — Update callers to import from new modules

---

### F1 Update callers to import from new modules directly

**Status:** Open
**Claimed by:**
**Est.:** 45 minutes
**Depends on:** E1

**What to do:**

Update the import statements in four production files. Do **not** change `tests/test_design.py` yet (that is G1).

**`api.py`** (line 37):
```python
# Before
from .design import (
    estimate_candidate_size,
    build_candidate,
    build_model_matrix,
    build_i_opt_design_with_idx,
)

# After
from .candidate import estimate_candidate_size, build_candidate
from .model_matrix import build_model_matrix
from .iopt_search import build_i_opt_design_with_idx
```

**`power_curves.py`** (line 28):
```python
# Before
from .design import (
    build_candidate,
    build_model_matrix,
    build_i_opt_design,
    build_i_opt_design_with_idx,
    estimate_candidate_size,
)

# After
from .candidate import build_candidate, estimate_candidate_size
from .model_matrix import build_model_matrix
from .iopt_search import build_i_opt_design, build_i_opt_design_with_idx
```

**`contrasts.py`** (line 31):
```python
# Before
from .design import build_candidate, build_model_matrix

# After
from .candidate import build_candidate
from .model_matrix import build_model_matrix
```

**`__init__.py`** (line 28):
```python
# Before
from .design import build_candidate, build_model_matrix, augment_design  # noqa: F401

# After
from .candidate import build_candidate  # noqa: F401
from .model_matrix import build_model_matrix  # noqa: F401
from .iopt_search import augment_design  # noqa: F401
```

**Verification:** Run `pytest tests/ -q --tb=no` — 248 tests pass. Then verify the import chain is matplotlib-free:
```python
import sys
import lattice_doe.api
assert "matplotlib" not in sys.modules
```

**Acceptance criteria:**
- [ ] `api.py`, `power_curves.py`, `contrasts.py`, `__init__.py` import from `candidate`, `model_matrix`, `iopt_search` — not from `design`.
- [ ] `design.py` is no longer imported by any production code (only tests, until G1).
- [ ] matplotlib-free import check passes.
- [ ] 248 tests pass.

---

## Epic G — Tests & documentation

---

### G1 Update `test_design.py` + final verification

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** F1

**What to do:**

1. Update `tests/test_design.py` imports to point to the canonical new modules:
   ```python
   # Before
   from lattice_doe.design import (
       build_candidate, build_model_matrix, estimate_candidate_size,
       build_i_opt_design, build_i_opt_design_with_idx,
       _i_criterion_for_indices, _d_criterion_for_indices, _a_criterion_for_indices,
       _criterion_score, _score_design, augment_design,
   )

   # After
   from lattice_doe.candidate import (
       build_candidate, estimate_candidate_size,
   )
   from lattice_doe.model_matrix import build_model_matrix
   from lattice_doe.iopt_search import (
       build_i_opt_design, build_i_opt_design_with_idx,
       _i_criterion_for_indices, _d_criterion_for_indices, _a_criterion_for_indices,
       _criterion_score, _score_design, augment_design,
   )
   ```

2. Run the full suite: `pytest tests/ -q` — 248 tests pass.

3. Run a final import-chain check:
   ```python
   import sys
   import lattice_doe
   assert "matplotlib" not in sys.modules
   print("matplotlib-free import: OK")
   ```

4. Verify file sizes are in expected ranges:
   ```bash
   wc -l lattice_doe/candidate.py    # ~330
   wc -l lattice_doe/model_matrix.py # ~30
   wc -l lattice_doe/iopt_search.py  # ~830
   wc -l lattice_doe/design.py       # ~35
   ```

**Acceptance criteria:**
- [ ] `test_design.py` imports directly from `candidate`, `model_matrix`, `iopt_search`.
- [ ] `design.py` is no longer imported anywhere in production or test code.
- [ ] 248 tests pass (no regressions from any of A1–F1).
- [ ] matplotlib-free import confirmed.

---

### G2 Documentation updates

**Status:** Open
**Claimed by:**
**Est.:** 20 minutes
**Depends on:** G1

**What to do:**

1. **`ENHANCEMENTS.md`** — mark TD-1 as Done in the Technical Debt table:
   - Strike through the TD-1 row and replace with summary of new files and line counts.

2. **`README.md`** — no user-visible change needed (public API is unchanged); no edits required.

3. **`docs/planning/design-refactor.md`** (this file) — update all ticket statuses to Done and set `Progress: 8 / 8 tickets done. ✅`.

**Acceptance criteria:**
- [ ] ENHANCEMENTS.md TD-1 entry is marked Done with file list.
- [ ] Ticket pack progress updated to 8/8.

---

## Summary: before vs. after

| File | Before | After |
|------|--------|-------|
| `design.py` | 1,267 lines — all concerns mixed | ~35 lines — re-export wrapper |
| `candidate.py` | (does not exist) | ~335 lines — sizing + generation |
| `model_matrix.py` | (does not exist) | ~30 lines — Patsy wrapper |
| `iopt_search.py` | (does not exist) | ~830 lines — exchange + search + augment |

The split isolates the computationally expensive Fedorov exchange (`iopt_search.py`) from factor-space sampling (`candidate.py`), making it straightforward to:
- Unit-test candidate generation without triggering a full design search.
- Swap or benchmark the exchange algorithm without touching candidate code.
- Add new candidate strategies (e.g. Sobol sequences) without touching the optimizer.
