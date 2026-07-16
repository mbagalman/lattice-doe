# model_matrix.py
# License: MIT
"""
Model matrix construction for optimal experimental designs
==========================================================

Thin wrapper around Patsy's ``dmatrix`` that returns a plain numpy array
alongside the column (parameter) names.  Called by ``iopt_search.py`` and
any other module that needs to evaluate a Patsy formula over a DataFrame.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:  # circular-import-safe: utils lazily imports this module
    from .utils import FactorSpec

import numpy as np
import pandas as pd
from patsy import dmatrix


# ---------------------------------------------------------------------
# Model matrix construction
# ---------------------------------------------------------------------
def build_model_matrix(formula: str, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Construct model matrix from a Patsy formula and dataframe.

    Parameters
    ----------
    formula : str
        Patsy formula string (e.g., "~ 1 + A + B + A:B").
    df : DataFrame
        Candidate or design set.

    Returns
    -------
    (np.ndarray, list[str])
        Model matrix (n x p) and list of p column (parameter) names.
    """
    X_df = dmatrix(formula, df, return_type="dataframe")
    return np.asarray(X_df), list(X_df.columns)


def resolve_design_matrix(
    formula: str,
    design_df: pd.DataFrame,
    factors: "FactorSpec",
    model_matrix: "pd.DataFrame | None" = None,
) -> Tuple[np.ndarray, List[str]]:
    """The model matrix for an EXISTING design, honoring the coding authority.

    Analysis functions that operate on a fixed design (sensitivity, MDE,
    robustness) must evaluate power on the exact basis the design run used.
    When *model_matrix* — ``result["model_matrix"]`` from that run — is given,
    it is validated against *design_df* and used as-is.

    Without it the matrix is rebuilt from *design_df*, which is only sound
    when the formula's coding is derivable from the factor spec. For
    data-dependent codings (learned spline knots, derived categorical levels)
    a rebuild re-learns those parameters from the n design rows and silently
    yields a numerically different basis than the one that was powered
    (UX-57), so this refuses with an actionable error instead.

    Returns
    -------
    (np.ndarray, list[str])
        Model matrix (n x p) and parameter names.
    """
    if model_matrix is not None:
        X = np.asarray(model_matrix, dtype=float)
        if X.ndim != 2 or X.shape[0] != len(design_df):
            raise ValueError(
                f"model_matrix has shape {X.shape}, which does not match the "
                f"{len(design_df)}-row design_df. Pass the design run's own "
                "result['model_matrix'] alongside its design_df."
            )
        names = (
            [str(c) for c in model_matrix.columns]
            if hasattr(model_matrix, "columns")
            else [f"x{j}" for j in range(X.shape[1])]
        )
        return X, names

    from .contrasts import coding_is_data_dependent

    reason = coding_is_data_dependent(formula, factors)
    if reason is not None:
        raise ValueError(
            "This analysis rebuilds the model matrix from design_df, but "
            + reason
            + " Pass model_matrix=result['model_matrix'] from the design run "
            "so the analysis evaluates the exact basis that was powered, or "
            "make the coding explicit in the formula (bs(x, knots=[...], "
            "lower_bound=..., upper_bound=...); C(..., levels=[...]))."
        )
    return build_model_matrix(formula, design_df)


def align_contrast_to_columns(
    L: np.ndarray,
    treat_names: List[str],
    target_names: List[str],
) -> np.ndarray:
    """Zero-expand a treatment-only contrast into a wider named basis.

    A blocked design powers the AUGMENTED model (treatment + block dummies),
    so a contrast written against the treatment columns must be re-addressed
    by NAME into the augmented matrix. Patsy orders categorical terms
    (including the block dummies) before numeric terms, so the block columns
    are not necessarily trailing and positional zero-padding would aim
    contrast rows at the wrong effects. Shared by ``find_optimal_design`` and
    the fixed-design analysis functions so the two can never disagree
    (UX-62).

    Returns *L* unchanged when *target_names* equals *treat_names*.
    """
    if list(target_names) == list(treat_names):
        return L
    pos = {name: j for j, name in enumerate(target_names)}
    missing = [nm for nm in treat_names if nm not in pos]
    if missing:
        raise ValueError(
            f"Treatment model columns {missing} were not found in the "
            f"augmented blocked model matrix (columns: {list(target_names)}); "
            "cannot align the contrast matrix L to the blocked design."
        )
    if L.shape[1] != len(treat_names):
        raise ValueError(
            f"Contrast L has {L.shape[1]} columns but the treatment model "
            f"has {len(treat_names)} parameters ({list(treat_names)})."
        )
    L_full = np.zeros((L.shape[0], len(target_names)))
    for j, nm in enumerate(treat_names):
        L_full[:, pos[nm]] = L[:, j]
    return L_full


def tested_column_indices(
    names: List[str],
    treat_names: List[str],
) -> "List[int] | None":
    """Positions of the TESTED (treatment-model) columns inside *names*.

    Returns None when the two name lists are identical — the matrix is not
    augmented and every column is a tested column. Raises when a treatment
    column cannot be found, since silently proceeding would either crash on
    shape (contrast mode) or count nuisance columns as tested predictors
    (R² mode, UX-62)."""
    if list(names) == list(treat_names):
        return None
    pos = {name: j for j, name in enumerate(names)}
    missing = [nm for nm in treat_names if nm not in pos]
    if missing:
        raise ValueError(
            f"Treatment model columns {missing} were not found in the model "
            f"matrix (columns: {list(names)}); cannot separate tested from "
            "nuisance columns."
        )
    return [pos[nm] for nm in treat_names]


__all__ = [
    "build_model_matrix",
    "resolve_design_matrix",
    "align_contrast_to_columns",
    "tested_column_indices",
]
