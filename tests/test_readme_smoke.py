"""Smoke test for the README quick-start example.

This must succeed in a fresh environment after `pip install lattice-doe`
with no optional extras installed. If it breaks, the very first thing a
new user copies from the README will fail. CI runs this test both against
the editable install and against the freshly-built wheel.
"""

import pandas as pd

from lattice_doe import find_optimal_design, PowerContrastConfig, DesignOptions
from lattice_doe.contrasts import contrast_from_scenarios


def test_readme_quickstart_runs_end_to_end():
    formula = "~ 1 + A + B + A:B"
    factors = {
        "A": (-1.0, 1.0),
        "B": (-1.0, 1.0),
    }

    L, delta = contrast_from_scenarios(
        formula=formula,
        factors=factors,
        scenario_a={"A": -1.0, "B": 0.0},
        scenario_b={"A": 1.0, "B": 0.0},
        sesoi=2.0,
    )

    result = find_optimal_design(
        formula=formula,
        factors=factors,
        power_cfg=PowerContrastConfig(L=L, delta=delta, power=0.80, sigma=1.0, max_n=50),
        design_opts=DesignOptions(criterion="I", auto_candidate=True),
    )

    assert isinstance(result["design_df"], pd.DataFrame)
    n = result["report"]["n"]
    assert n >= 1
    assert len(result["design_df"]) == n
    assert set(result["design_df"].columns) >= {"A", "B"}


def test_readme_top_level_imports():
    """The exact import line the README asks users to run must succeed."""
    from lattice_doe import find_optimal_design, PowerContrastConfig, DesignOptions  # noqa: F401
    from lattice_doe.contrasts import contrast_from_scenarios  # noqa: F401
