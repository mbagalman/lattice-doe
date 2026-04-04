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
