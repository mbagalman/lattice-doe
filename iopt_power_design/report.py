# report.py
# License: MIT
"""
iopt_power_design.report
========================
Generate self-contained HTML (and optionally PDF) summary reports for
powered optimal designs.

Public API
----------
generate_report(result, formula, factors, power_cfg, output_path, ...)
    Render and write a shareable report.  Returns the Path written.
"""

from __future__ import annotations

import base64
import importlib.resources
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["generate_report"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_jinja_env():
    """Return a Jinja2 Environment that loads templates from this package."""
    try:
        from jinja2 import Environment, PackageLoader, select_autoescape
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "HTML report generation requires jinja2. "
            'Install it with: pip install "iopt-power-design[report]"'
        ) from exc

    return Environment(
        loader=PackageLoader("iopt_power_design", "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _fig_to_base64(fig) -> str | None:
    """Convert a Plotly or Matplotlib figure to a base64-encoded PNG string.

    Returns None if conversion fails or the figure type is unrecognised.
    """
    # --- Plotly ---
    try:
        import plotly.graph_objects as go  # noqa: F401

        if hasattr(fig, "to_image"):
            png_bytes = fig.to_image(format="png", width=800, height=350)
            return base64.b64encode(png_bytes).decode("ascii")
    except Exception:
        pass

    # --- Matplotlib ---
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        if hasattr(fig, "savefig"):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    result: dict,
    formula: str,
    factors: dict,
    power_cfg: Any,
    output_path: str | Path,
    title: str = "I-Optimal Design Report",
    include_power_curve: bool = True,
    design_rows_shown: int = 30,
) -> Path:
    """Render and write a shareable HTML (or PDF) report.

    Parameters
    ----------
    result:
        The dict returned by ``i_optimal_powered_design()``.
    formula:
        Patsy formula string used to generate the design.
    factors:
        Factor specification dict used to generate the design.
    power_cfg:
        ``PowerContrastConfig`` or ``PowerR2Config`` instance.
    output_path:
        Destination file path.  The suffix determines the format:
        ``.html`` (default) or ``.pdf`` (requires ``weasyprint``).
        If the path is a directory, ``iopt_report.html`` is written inside it.
    title:
        Report title shown in the HTML ``<title>`` tag and heading.
    include_power_curve:
        Whether to generate and embed a power-curve figure.  Set to
        ``False`` to skip (faster but less informative).
    design_rows_shown:
        Maximum number of design-table rows to include; a note is shown
        when the design is larger.

    Returns
    -------
    Path
        Resolved path of the file that was written.
    """
    raise NotImplementedError(
        "generate_report() is a stub. "
        "Ticket B1–C1 will fill in the implementation."
    )
