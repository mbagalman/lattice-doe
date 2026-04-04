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

from typing import List, Tuple

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


__all__ = ["build_model_matrix"]
