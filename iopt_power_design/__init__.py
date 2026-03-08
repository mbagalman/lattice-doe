# __init__.py
# License: MIT
"""
iopt_power_design
=================
I-optimal experimental designs with power assurance for linear models.
"""

from __future__ import annotations

# Public version string (keep in sync with pyproject.toml)
__version__ = "0.1.0"

# Re-export primary API and configuration types
from .api import i_optimal_powered_design  # noqa: F401
from .config import PowerContrastConfig, PowerR2Config, DesignOptions  # noqa: F401

# Power surface — canonical implementation in power_curves
from .power_curves import power_surface_2d  # noqa: F401

# Analysis utilities (includes DataFrame-returning wrappers for the two curves)
from .analysis import (  # noqa: F401
    power_curve_by_n,
    power_curve_by_effect,
    generate_power_curves,
    power_sensitivity,
    min_detectable_effect,
    compare_criteria,
)
from .candidate import build_candidate  # noqa: F401
from .model_matrix import build_model_matrix  # noqa: F401
from .iopt_search import augment_design  # noqa: F401
from .report import generate_report  # noqa: F401
from .sheets import SheetsError, sheets_run, create_sheet_template  # noqa: F401

__all__ = [
    "__version__",
    # Top-level API
    "i_optimal_powered_design",
    "power_curve_by_n",
    "power_curve_by_effect",
    "generate_power_curves",
    "power_sensitivity",
    "min_detectable_effect",
    "compare_criteria",
    # Design utilities
    "augment_design",
    # Config / options
    "PowerContrastConfig",
    "PowerR2Config",
    "DesignOptions",
    # Power surface
    "power_surface_2d",
    # Low-level utilities
    "build_candidate",
    "build_model_matrix",
    # Reports
    "generate_report",
    # Sheets integration
    "SheetsError",
    "sheets_run",
    "create_sheet_template",
]
