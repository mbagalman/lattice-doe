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
from .api import (  # noqa: F401
    i_optimal_powered_design,
    power_curve_by_n,
    power_curve_by_effect,
    generate_power_curves,
    power_sensitivity,
    min_detectable_effect,
    compare_criteria,
    PowerContrastConfig,
    PowerR2Config,
)
from .power_curves import power_surface_2d  # noqa: F401
from .config import DesignOptions  # noqa: F401
from .design import build_candidate, build_model_matrix, augment_design  # noqa: F401
from .report import generate_report  # noqa: F401

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
]
