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
from .api import i_optimal_powered_design, i_optimal_multiresponse_design  # noqa: F401
from .config import (  # noqa: F401
    PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig,
    glm_fisher_weight,
    DesignOptions, SplitPlotOptions,
    ResponseSpec, MultiResponseOptions,
)

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
    robustness_report,
    power_curve_by_wp,
    power_curve_by_n_multiresponse,
    multiresponse_sensitivity,
)
from .power import contrast_power_sp, global_r2_power_sp, eval_response_power, combine_powers, hotelling_t2_power, glm_contrast_power  # noqa: F401
from .allocation import i_optimal_allocation  # noqa: F401
from .candidate import build_candidate, build_split_plot_candidate  # noqa: F401
from .model_matrix import build_model_matrix  # noqa: F401
from .iopt_search import augment_design, build_split_plot_design  # noqa: F401
from .report import generate_report  # noqa: F401
from .sheets import SheetsError, sheets_run, create_sheet_template  # noqa: F401
from .excel_template import ExcelError, excel_run, create_excel_template  # noqa: F401
from .blocked import balanced_block_sizes, blocked_formula, build_blocked_design  # noqa: F401
from .widgets import WidgetsError, DesignWidget, design_widget  # noqa: F401
from .split_plot import (  # noqa: F401
    build_whole_plot_indicator,
    build_split_plot_covariance_inv,
    gls_information_matrix,
)

__all__ = [
    "__version__",
    # Top-level API
    "i_optimal_powered_design",
    "i_optimal_multiresponse_design",
    "power_curve_by_n",
    "power_curve_by_effect",
    "generate_power_curves",
    "power_sensitivity",
    "min_detectable_effect",
    "compare_criteria",
    "robustness_report",
    "power_curve_by_wp",
    # Multi-response analysis
    "power_curve_by_n_multiresponse",
    "multiresponse_sensitivity",
    # Design utilities
    "augment_design",
    "build_split_plot_design",
    # Config / options
    "PowerContrastConfig",
    "PowerR2Config",
    "PowerGLMContrastConfig",
    "glm_fisher_weight",
    "DesignOptions",
    "SplitPlotOptions",
    "ResponseSpec",
    "MultiResponseOptions",
    # Power surface
    "power_surface_2d",
    # Low-level utilities
    "i_optimal_allocation",
    "build_candidate",
    "build_split_plot_candidate",
    "build_model_matrix",
    # Reports
    "generate_report",
    # Sheets integration
    "SheetsError",
    "sheets_run",
    "create_sheet_template",
    # Excel integration
    "ExcelError",
    "excel_run",
    "create_excel_template",
    # Blocked design utilities
    "balanced_block_sizes",
    "blocked_formula",
    "build_blocked_design",
    # Jupyter widgets UI
    "WidgetsError",
    "DesignWidget",
    "design_widget",
    # Split-plot covariance utilities
    "build_whole_plot_indicator",
    "build_split_plot_covariance_inv",
    "gls_information_matrix",
    # Split-plot power functions
    "contrast_power_sp",
    "global_r2_power_sp",
    # GLM power
    "glm_contrast_power",
    # Multi-response power wrapper
    "eval_response_power",
    "combine_powers",
    # Hotelling T² joint power
    "hotelling_t2_power",
]
