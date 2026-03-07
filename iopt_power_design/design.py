# design.py
# License: MIT
"""
Backward-compatibility wrapper for iopt_power_design.design
============================================================

All implementation has been split into focused modules:

  - ``candidate.py``    — candidate set generation (LHS, categorical, mixed)
  - ``model_matrix.py`` — Patsy formula → model matrix
  - ``iopt_search.py``  — Fedorov exchange, criterion scorers, multi-start,
                          public build + augment functions

This file re-exports every public symbol so that existing code using
``from .design import ...`` or ``from iopt_power_design.design import ...``
continues to work without modification.

Do not add new implementation here.  New code belongs in one of the
modules above.
"""
from __future__ import annotations

# Re-export candidate generation
from .candidate import (  # noqa: F401
    estimate_candidate_size,
    build_candidate,
)

# Re-export model matrix construction
from .model_matrix import build_model_matrix  # noqa: F401

# Re-export design search + augmentation
from .iopt_search import (  # noqa: F401
    build_i_opt_design,
    build_i_opt_design_with_idx,
    _i_criterion_for_indices,
    _d_criterion_for_indices,
    _a_criterion_for_indices,
    _criterion_score,
    _score_design,
    augment_design,
)

__all__ = [
    "estimate_candidate_size",
    "build_candidate",
    "build_model_matrix",
    "build_i_opt_design",
    "build_i_opt_design_with_idx",
    "_i_criterion_for_indices",
    "_d_criterion_for_indices",
    "_a_criterion_for_indices",
    "_criterion_score",
    "_score_design",
    "augment_design",
]
