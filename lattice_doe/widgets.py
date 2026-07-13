# widgets.py
# License: MIT
"""
Jupyter ipywidgets UI for lattice-doe
============================================

Interactive in-notebook UI: dynamic factor-entry table, sliders for
alpha/power/sigma, a formula text field, a run button, and an inline
Plotly power curve. No server needed — runs inside any JupyterLab /
VS Code notebook.

Install::

    pip install "lattice-doe[widgets]"

Quick start::

    from lattice_doe.widgets import design_widget
    w = design_widget(
        formula="~ 1 + A + B + A:B",
        factors={"A": (-1.0, 1.0), "B": (-1.0, 1.0)},
        power_mode="r2",
        r2_target=0.20,
    )
    # After running the widget, retrieve the result:
    result = w.get_result()
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import f as scipy_f
from scipy.stats import ncf as scipy_ncf

from .api import find_optimal_design
from .config import DesignOptions, PowerContrastConfig, PowerR2Config

# ---------------------------------------------------------------------------
# Soft-dependency guards
# ---------------------------------------------------------------------------

try:
    import ipywidgets as _widgets
    from IPython.display import display as _ipy_display, clear_output as _clear_output
    _HAS_WIDGETS = True
except ImportError:  # pragma: no cover
    _HAS_WIDGETS = False

try:
    import plotly.graph_objects as _go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

_INSTALL_HINT = 'pip install "lattice-doe[widgets]"'


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class WidgetsError(RuntimeError):
    """Raised when ipywidgets/plotly are missing or widget state is invalid."""


def _require_widgets() -> None:
    """Raise WidgetsError if ipywidgets is not installed."""
    if not _HAS_WIDGETS:
        raise WidgetsError(
            f"ipywidgets is required for the Jupyter UI. Install with: {_INSTALL_HINT}"
        )


# ---------------------------------------------------------------------------
# Pure-Python helpers (no ipywidgets dependency — fully testable)
# ---------------------------------------------------------------------------

def _parse_matrix(text: str) -> np.ndarray:
    """Parse newline-separated rows of space/comma-delimited values into a 2-D array."""
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            rows.append([float(x) for x in line.replace(",", " ").split()])
    if not rows:
        raise ValueError("L matrix text is empty; provide at least one row.")
    return np.array(rows)


def _parse_vector(text: str) -> np.ndarray:
    """Parse space/comma-separated text into a 1-D float array."""
    text = text.strip()
    if not text:
        raise ValueError("delta vector text is empty.")
    return np.array([float(x) for x in text.replace(",", " ").split()])


def _build_power_cfg_from_state(
    state: dict,
) -> Union[PowerContrastConfig, PowerR2Config]:
    """Build a power config from a plain widget-state dict.

    Mirrors ``_build_power_cfg`` in ``app/pages/3_Run_Results.py``.
    """
    if state["power_mode"] == "contrast":
        L = _parse_matrix(state["L_text"])
        delta = _parse_vector(state["delta_text"])
        return PowerContrastConfig(
            L=L,
            delta=delta,
            alpha=float(state["alpha"]),
            power=float(state["power_target"]),
            sigma=float(state["sigma"]),
            max_n=int(state["max_n"]),
        )
    return PowerR2Config(
        r2_target=float(state["r2_target"]),
        alpha=float(state["alpha"]),
        power=float(state["power_target"]),
        max_n=int(state["max_n"]),
        lambda_mode=state["lambda_mode"],
    )


def _build_design_opts_from_state(
    state: dict,
    extra_kwargs: Optional[dict] = None,
) -> DesignOptions:
    """Build ``DesignOptions`` from a plain widget-state dict.

    Mirrors ``_build_design_opts`` in ``app/pages/3_Run_Results.py``.
    *extra_kwargs* carries non-exposed fields (e.g. ``n_blocks``) from
    a ``DesignOptions`` passed to the ``DesignWidget`` constructor.
    """
    kwargs: dict = {}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    # Widget-exposed fields take precedence over extra_kwargs
    kwargs.update(
        criterion=state["criterion"],
        starts=int(state["starts"]),
        random_state=int(state["random_state"]),
        auto_candidate=bool(state["auto_candidate"]),
    )
    if not state["auto_candidate"]:
        kwargs["candidate_points"] = int(state["candidate_points"])
    expr = state.get("constraint_expr", "").strip()
    if expr:
        kwargs["constraint_expr"] = expr
    return DesignOptions(**kwargs)


def _approx_power_curve(
    n_vals: List[int],
    report: dict,
    alpha: float,
    power_mode: str,
    lambda_mode: str,
) -> List[float]:
    """Scale the run's noncentrality to approximate power at each n.

    Mirrors ``_approx_power_curve`` in ``app/pages/3_Run_Results.py``.
    Fast — no additional design builds.
    """
    p = int(report["p"])
    df_num = int(report["df_num"])
    n_result = int(report["n"])
    lambda_result = float(report["noncentrality_lambda"])

    powers = []
    for n in n_vals:
        if n <= p:
            powers.append(0.0)
            continue
        df_denom = n - p
        if power_mode == "r2" and lambda_mode == "n_minus_p":
            denom_ref = max(n_result - p, 1)
            lambda_n = lambda_result * (df_denom / denom_ref)
        else:
            lambda_n = lambda_result * (n / n_result)
        f_crit = scipy_f.ppf(1.0 - alpha, df_num, df_denom)
        powers.append(float(scipy_ncf.sf(f_crit, df_num, df_denom, lambda_n)))
    return powers


# ---------------------------------------------------------------------------
# _FactorRow — one row in the dynamic factor table (requires ipywidgets)
# ---------------------------------------------------------------------------

class _FactorRow:
    """Widget bundle for a single factor entry in the factor table.

    Instantiated only when ipywidgets is present (guarded by DesignWidget).
    """

    def __init__(
        self,
        name: str = "",
        ftype: str = "Continuous",
        lo: float = -1.0,
        hi: float = 1.0,
        levels: str = "",
        on_remove=None,
        on_type_change=None,
    ) -> None:
        self.name_widget = _widgets.Text(
            value=name,
            placeholder="Factor name",
            layout=_widgets.Layout(width="120px"),
        )
        self.type_widget = _widgets.Dropdown(
            options=["Continuous", "Categorical"],
            value=ftype,
            layout=_widgets.Layout(width="130px"),
        )
        self.low_widget = _widgets.FloatText(
            value=lo,
            layout=_widgets.Layout(width="75px"),
        )
        self.high_widget = _widgets.FloatText(
            value=hi,
            layout=_widgets.Layout(width="75px"),
        )
        self.levels_widget = _widgets.Text(
            value=levels,
            placeholder="e.g. low,med,high",
            layout=_widgets.Layout(width="190px"),
        )
        self.remove_button = _widgets.Button(
            description="✕",
            button_style="danger",
            layout=_widgets.Layout(width="36px", height="30px"),
        )

        lo_lbl = _widgets.Label("Lo:", layout=_widgets.Layout(width="22px"))
        hi_lbl = _widgets.Label("Hi:", layout=_widgets.Layout(width="22px"))
        self._cont_box = _widgets.HBox([lo_lbl, self.low_widget, hi_lbl, self.high_widget])
        self._cat_box = _widgets.HBox([self.levels_widget])

        if ftype == "Categorical":
            self._cont_box.layout.display = "none"
        else:
            self._cat_box.layout.display = "none"

        self.row_box = _widgets.HBox([
            self.name_widget,
            self.type_widget,
            self._cont_box,
            self._cat_box,
            self.remove_button,
        ])

        if on_type_change is not None:
            self.type_widget.observe(
                lambda change, row=self: on_type_change(row, change), names="value"
            )
        if on_remove is not None:
            self.remove_button.on_click(lambda _btn, row=self: on_remove(row))

    def to_factor_spec(self) -> tuple:
        """Return ``(name, spec)`` for insertion into the API factors dict."""
        name = self.name_widget.value.strip()
        if not name:
            raise ValueError("Factor name cannot be empty.")
        if self.type_widget.value == "Continuous":
            lo = float(self.low_widget.value)
            hi = float(self.high_widget.value)
            if lo >= hi:
                raise ValueError(f"Factor '{name}': low ({lo}) must be less than high ({hi}).")
            return name, (lo, hi)
        levels = [x.strip() for x in self.levels_widget.value.split(",") if x.strip()]
        if not levels:
            raise ValueError(f"Factor '{name}': provide at least one level (comma-separated).")
        return name, levels


# ---------------------------------------------------------------------------
# DesignWidget — the main public class
# ---------------------------------------------------------------------------

class DesignWidget:
    """Interactive Jupyter widget for building I-optimal powered designs.

    Parameters
    ----------
    formula : str, optional
        Initial Patsy formula string. Defaults to ``"~ 1 + A + B"``.
    factors : dict, optional
        Initial factor spec in API format — continuous factors as ``(lo, hi)``
        tuples, categorical factors as lists of level strings.
        Pre-populates the factor table.
    power_mode : {"r2", "contrast"}, optional
        Initial power mode. Defaults to ``"r2"``.
    alpha : float, optional
        Initial significance level. Defaults to ``0.05``.
    power : float, optional
        Initial target power. Defaults to ``0.80``.
    sigma : float, optional
        Initial residual standard deviation (used in contrast mode).
        Defaults to ``1.0``.
    r2_target : float, optional
        Initial R² target (r2 mode only). Defaults to ``0.15``.
    max_n : int, optional
        Initial maximum sample size. Defaults to ``500``.
    design_opts : DesignOptions, optional
        Seed design options. Widget-exposed fields (criterion, starts,
        random_state, auto_candidate, constraint_expr) are pre-filled from
        this object. Non-exposed fields (e.g. ``n_blocks``) are preserved
        and merged at run time.
    show_advanced : bool, optional
        Whether to expand the Advanced options accordion on load. Defaults
        to ``False``.
    """

    def __init__(
        self,
        formula: str = "~ 1 + A + B",
        factors: Optional[Dict[str, Any]] = None,
        power_mode: str = "r2",
        alpha: float = 0.05,
        power: float = 0.80,
        sigma: float = 1.0,
        r2_target: float = 0.15,
        max_n: int = 500,
        design_opts: Optional[DesignOptions] = None,
        show_advanced: bool = False,
    ) -> None:
        _require_widgets()

        # Stash constructor defaults for reset()
        self._init_formula = formula
        self._init_factors = factors or {}
        self._init_power_mode = power_mode
        self._init_alpha = alpha
        self._init_power = power
        self._init_sigma = sigma
        self._init_r2_target = r2_target
        self._init_max_n = max_n
        self._init_show_advanced = show_advanced

        # Decompose design_opts — store non-exposed fields separately
        self._extra_do_kwargs: dict = {}
        _do_starts = 5
        _do_criterion = "I"
        _do_random_state = 123
        _do_auto_candidate = True
        _do_candidate_points = 2000
        _do_constraint_expr = ""
        if design_opts is not None:
            _do_starts = design_opts.starts
            _do_criterion = design_opts.criterion
            _do_random_state = design_opts.random_state
            _do_auto_candidate = design_opts.auto_candidate
            _do_candidate_points = design_opts.candidate_points
            _do_constraint_expr = design_opts.constraint_expr or ""
            # Collect non-exposed fields for passthrough
            _exposed = {
                "starts", "criterion", "random_state", "auto_candidate",
                "candidate_points", "constraint_expr",
            }
            for field in vars(design_opts):
                if field not in _exposed:
                    val = getattr(design_opts, field)
                    default_do = DesignOptions()
                    if val != getattr(default_do, field, None):
                        self._extra_do_kwargs[field] = val

        self._init_do_starts = _do_starts
        self._init_do_criterion = _do_criterion
        self._init_do_random_state = _do_random_state
        self._init_do_auto_candidate = _do_auto_candidate
        self._init_do_candidate_points = _do_candidate_points
        self._init_do_constraint_expr = _do_constraint_expr

        self._result: Optional[dict] = None
        self._factor_rows: List[_FactorRow] = []

        # --- Build all widget sections ---
        self._formula_widget = _widgets.Text(
            value=formula,
            placeholder="e.g. ~ 1 + A + B + A:B",
            description="Formula:",
            layout=_widgets.Layout(width="420px"),
            style={"description_width": "70px"},
        )

        # Power mode toggle
        self._mode_toggle = _widgets.ToggleButtons(
            options=[("R²  (global F-test)", "r2"), ("Contrast  (L·β = δ)", "contrast")],
            value=power_mode,
            description="",
            style={"button_width": "200px"},
        )

        # R² section
        self._r2_slider = _widgets.FloatSlider(
            value=r2_target,
            min=0.01, max=0.99, step=0.01,
            description="R² target:",
            readout_format=".2f",
            style={"description_width": "90px"},
            layout=_widgets.Layout(width="360px"),
        )
        self._lambda_mode_radio = _widgets.RadioButtons(
            options=[("n  (G*Power / statsmodels convention)", "n"),
                     ("n − p  (conservative)", "n_minus_p")],
            value="n",
            description="λ convention:",
            style={"description_width": "110px"},
        )
        self._r2_box = _widgets.VBox(
            [self._r2_slider, self._lambda_mode_radio],
            layout=_widgets.Layout(border="1px solid #ccc", padding="8px", margin="4px 0"),
        )

        # Contrast section
        self._L_text = _widgets.Textarea(
            value="",
            placeholder="Rows of space-separated values, one row per line",
            description="L matrix:",
            rows=3,
            layout=_widgets.Layout(width="420px"),
            style={"description_width": "80px"},
        )
        self._delta_text = _widgets.Textarea(
            value="",
            placeholder="Space-separated values",
            description="δ vector:",
            rows=2,
            layout=_widgets.Layout(width="420px"),
            style={"description_width": "80px"},
        )
        self._sigma_widget = _widgets.FloatText(
            value=sigma,
            description="σ (residual std):",
            layout=_widgets.Layout(width="250px"),
            style={"description_width": "130px"},
        )
        self._contrast_box = _widgets.VBox(
            [self._L_text, self._delta_text, self._sigma_widget],
            layout=_widgets.Layout(border="1px solid #ccc", padding="8px", margin="4px 0"),
        )

        # Shared power params (always visible)
        self._alpha_slider = _widgets.FloatSlider(
            value=alpha,
            min=0.001, max=0.20, step=0.001,
            description="α (sig. level):",
            readout_format=".3f",
            style={"description_width": "110px"},
            layout=_widgets.Layout(width="380px"),
        )
        self._power_slider = _widgets.FloatSlider(
            value=power,
            min=0.50, max=0.999, step=0.01,
            description="Target power:",
            readout_format=".2f",
            style={"description_width": "110px"},
            layout=_widgets.Layout(width="380px"),
        )
        self._max_n_widget = _widgets.IntText(
            value=max_n,
            description="Max n:",
            layout=_widgets.Layout(width="200px"),
            style={"description_width": "60px"},
        )

        # Advanced options
        self._criterion_dd = _widgets.Dropdown(
            options=["I", "D", "A"],
            value=_do_criterion,
            description="Criterion:",
            layout=_widgets.Layout(width="200px"),
            style={"description_width": "80px"},
        )
        self._starts_slider = _widgets.IntSlider(
            value=_do_starts,
            min=1, max=50, step=1,
            description="Starts:",
            style={"description_width": "80px"},
            layout=_widgets.Layout(width="300px"),
        )
        self._seed_widget = _widgets.IntText(
            value=_do_random_state,
            description="Random seed:",
            layout=_widgets.Layout(width="220px"),
            style={"description_width": "100px"},
        )
        self._auto_cand_checkbox = _widgets.Checkbox(
            value=_do_auto_candidate,
            description="Auto-size candidate set",
            indent=False,
        )
        self._cand_points_widget = _widgets.IntText(
            value=_do_candidate_points,
            description="Candidate pts:",
            layout=_widgets.Layout(width="220px"),
            style={"description_width": "100px"},
        )
        self._cand_points_widget.layout.display = "none" if _do_auto_candidate else ""
        self._constraint_expr_widget = _widgets.Text(
            value=_do_constraint_expr,
            placeholder='e.g. A + B <= 1',
            description="Constraint:",
            layout=_widgets.Layout(width="380px"),
            style={"description_width": "90px"},
        )
        _adv_content = _widgets.VBox([
            self._criterion_dd,
            self._starts_slider,
            self._seed_widget,
            self._auto_cand_checkbox,
            self._cand_points_widget,
            self._constraint_expr_widget,
        ])
        self._advanced_accordion = _widgets.Accordion(children=[_adv_content])
        self._advanced_accordion.set_title(0, "Advanced design options")
        self._advanced_accordion.selected_index = 0 if show_advanced else None

        # Factor table
        self._factor_table_box = _widgets.VBox([])
        self._add_factor_btn = _widgets.Button(
            description="+ Add factor",
            button_style="info",
            layout=_widgets.Layout(width="120px"),
        )
        _factor_hdr = _widgets.HTML(
            "<b>Name</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "<b>Type</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "<b>Range / Levels</b>"
        )
        self._factor_section = _widgets.VBox([
            _widgets.HTML("<h4 style='margin:0'>Factors</h4>"),
            _factor_hdr,
            self._factor_table_box,
            self._add_factor_btn,
        ])

        # Run button + output
        self._run_btn = _widgets.Button(
            description="Generate design",
            button_style="primary",
            icon="play",
            layout=_widgets.Layout(width="180px", height="38px"),
        )
        self._output = _widgets.Output()
        self._status_html = _widgets.HTML("")

        # --- Wire observers ---
        self._mode_toggle.observe(self._on_power_mode_change, names="value")
        self._auto_cand_checkbox.observe(self._on_auto_cand_change, names="value")
        self._add_factor_btn.on_click(lambda _: self._add_factor_row())
        self._run_btn.on_click(self._on_run_clicked)

        # --- Pre-populate factor rows from constructor factors dict ---
        for fname, spec in self._init_factors.items():
            if isinstance(spec, (list, tuple)) and len(spec) == 2 and not isinstance(spec[0], str):
                self._add_factor_row(name=fname, ftype="Continuous", lo=float(spec[0]), hi=float(spec[1]))
            else:
                levels_str = ",".join(str(lv) for lv in spec)
                self._add_factor_row(name=fname, ftype="Categorical", levels=levels_str)

        # If no factors provided, add two default rows
        if not self._init_factors:
            self._add_factor_row(name="A")
            self._add_factor_row(name="B")

        # Apply initial mode visibility
        self._apply_mode_visibility(power_mode)

        # --- Assemble full layout ---
        self._layout = _widgets.VBox([
            _widgets.HTML("<h3 style='margin:4px 0'>Lattice DOE</h3>"),
            self._factor_section,
            _widgets.HTML("<hr style='margin:6px 0'>"),
            self._formula_widget,
            _widgets.HTML("<hr style='margin:6px 0'>"),
            _widgets.HTML("<h4 style='margin:0'>Power Mode</h4>"),
            self._mode_toggle,
            self._r2_box,
            self._contrast_box,
            _widgets.HTML("<hr style='margin:6px 0'>"),
            _widgets.HTML("<h4 style='margin:0'>Power Parameters</h4>"),
            self._alpha_slider,
            self._power_slider,
            self._max_n_widget,
            _widgets.HTML("<hr style='margin:6px 0'>"),
            self._advanced_accordion,
            _widgets.HTML("<hr style='margin:6px 0'>"),
            self._run_btn,
            self._status_html,
            self._output,
        ])

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def display(self) -> None:
        """Render the widget in the current Jupyter notebook cell."""
        _ipy_display(self._layout)

    def get_result(self) -> Optional[dict]:
        """Return the last successful ``find_optimal_design`` result, or ``None``."""
        return self._result

    def get_design_df(self) -> Optional[pd.DataFrame]:
        """Return ``result["design_df"]`` from the last run, or ``None``."""
        if self._result is None:
            return None
        return self._result.get("design_df")

    def get_report(self) -> Optional[dict]:
        """Return ``result["report"]`` from the last run, or ``None``."""
        if self._result is None:
            return None
        return self._result.get("report")

    def reset(self) -> None:
        """Restore all widget values to constructor defaults and clear results."""
        self._formula_widget.value = self._init_formula
        self._mode_toggle.value = self._init_power_mode
        self._r2_slider.value = self._init_r2_target
        self._alpha_slider.value = self._init_alpha
        self._power_slider.value = self._init_power
        self._sigma_widget.value = self._init_sigma
        self._max_n_widget.value = self._init_max_n
        self._criterion_dd.value = self._init_do_criterion
        self._starts_slider.value = self._init_do_starts
        self._seed_widget.value = self._init_do_random_state
        self._auto_cand_checkbox.value = self._init_do_auto_candidate
        self._cand_points_widget.value = self._init_do_candidate_points
        self._constraint_expr_widget.value = self._init_do_constraint_expr
        self._result = None
        self._status_html.value = ""
        with self._output:
            _clear_output()

    # -----------------------------------------------------------------------
    # Private helpers — factor table
    # -----------------------------------------------------------------------

    def _add_factor_row(
        self,
        name: str = "",
        ftype: str = "Continuous",
        lo: float = -1.0,
        hi: float = 1.0,
        levels: str = "",
    ) -> None:
        row = _FactorRow(
            name=name,
            ftype=ftype,
            lo=lo,
            hi=hi,
            levels=levels,
            on_remove=self._remove_factor_row,
            on_type_change=self._on_factor_type_change,
        )
        self._factor_rows.append(row)
        self._factor_table_box.children = tuple(r.row_box for r in self._factor_rows)

    def _remove_factor_row(self, row: _FactorRow) -> None:
        self._factor_rows = [r for r in self._factor_rows if r is not row]
        self._factor_table_box.children = tuple(r.row_box for r in self._factor_rows)

    def _get_factor_spec(self) -> Dict[str, Any]:
        spec: Dict[str, Any] = {}
        for row in self._factor_rows:
            name, value = row.to_factor_spec()
            spec[name] = value
        return spec

    # -----------------------------------------------------------------------
    # Private helpers — observers
    # -----------------------------------------------------------------------

    def _on_factor_type_change(self, row: _FactorRow, change: dict) -> None:
        if change["new"] == "Categorical":
            row._cont_box.layout.display = "none"
            row._cat_box.layout.display = ""
        else:
            row._cont_box.layout.display = ""
            row._cat_box.layout.display = "none"

    def _on_power_mode_change(self, change: dict) -> None:
        self._apply_mode_visibility(change["new"])

    def _apply_mode_visibility(self, mode: str) -> None:
        if mode == "r2":
            self._r2_box.layout.display = ""
            self._contrast_box.layout.display = "none"
        else:
            self._r2_box.layout.display = "none"
            self._contrast_box.layout.display = ""

    def _on_auto_cand_change(self, change: dict) -> None:
        self._cand_points_widget.layout.display = "none" if change["new"] else ""

    # -----------------------------------------------------------------------
    # Private helpers — build state dict from widgets
    # -----------------------------------------------------------------------

    def _get_state(self) -> dict:
        """Read all widget values into a plain dict for config builders."""
        mode = self._mode_toggle.value
        state: dict = {
            "power_mode": mode,
            "alpha": self._alpha_slider.value,
            "power_target": self._power_slider.value,
            "sigma": self._sigma_widget.value,
            "max_n": self._max_n_widget.value,
            # R² fields
            "r2_target": self._r2_slider.value,
            "lambda_mode": self._lambda_mode_radio.value,
            # Design opts fields
            "criterion": self._criterion_dd.value,
            "starts": self._starts_slider.value,
            "random_state": self._seed_widget.value,
            "auto_candidate": self._auto_cand_checkbox.value,
            "candidate_points": self._cand_points_widget.value,
            "constraint_expr": self._constraint_expr_widget.value,
        }
        # Only include contrast fields when mode is contrast
        if mode == "contrast":
            state["L_text"] = self._L_text.value
            state["delta_text"] = self._delta_text.value
        return state

    # -----------------------------------------------------------------------
    # Private helpers — validate inputs
    # -----------------------------------------------------------------------

    def _validate_inputs(self) -> List[str]:
        """Return a list of human-readable error strings (empty = valid)."""
        errors: List[str] = []
        if not self._factor_rows:
            errors.append("Add at least one factor before running.")
        for row in self._factor_rows:
            try:
                row.to_factor_spec()
            except ValueError as exc:
                errors.append(str(exc))
        if not self._formula_widget.value.strip():
            errors.append("Formula cannot be empty.")
        return errors

    # -----------------------------------------------------------------------
    # Private helpers — run callback
    # -----------------------------------------------------------------------

    def _on_run_clicked(self, _btn) -> None:
        """Handle the Generate design button click."""
        # Validate
        errors = self._validate_inputs()
        if errors:
            self._status_html.value = (
                "<span style='color:red'>"
                + "<br>".join(f"• {e}" for e in errors)
                + "</span>"
            )
            return

        self._status_html.value = "<i style='color:gray'>Running… please wait.</i>"
        self._run_btn.disabled = True
        self._result = None

        try:
            state = self._get_state()
            factors = self._get_factor_spec()
            formula = self._formula_widget.value.strip()

            power_cfg = _build_power_cfg_from_state(state)
            design_opts = _build_design_opts_from_state(state, self._extra_do_kwargs)

            # Live progress in the status line (UX-3).
            from .progress import ProgressReporter

            def _on_ev(ev) -> None:
                _p = (f" — power {ev.current_power:.4f}"
                      if ev.current_power is not None else "")
                _n = f" n={ev.trial_n}" if ev.trial_n is not None else ""
                self._status_html.value = (
                    f"<i style='color:gray'>[{ev.elapsed_sec:.1f}s] "
                    f"{ev.phase}{_n}{_p}…</i>"
                )

            result = find_optimal_design(
                formula=formula,
                factors=factors,
                power_cfg=power_cfg,
                design_opts=design_opts,
                on_progress=ProgressReporter(_on_ev, min_interval=0.2),
            )
            self._result = result
            _report = result.get("report", {})
            # UX-7: a search that missed its target must not display an
            # unqualified success message.
            if _report.get("target_met") is False:
                _warn_lines = "".join(
                    f"<br>• {w}" for w in _report.get("warnings", [])
                )
                self._status_html.value = (
                    "<span style='color:#b45309'><b>⚠ Target power not "
                    "reached</b> (achieved "
                    f"{_report.get('achieved_power', float('nan')):.4f} of "
                    f"{_report.get('target_power', float('nan')):.4f}; "
                    f"reason: {_report.get('termination_reason', 'unknown')})."
                    f"{_warn_lines}</span>"
                )
            else:
                self._status_html.value = (
                    "<span style='color:green'>✓ Design generated successfully.</span>"
                )
            self._render_results(result, state)

        except Exception as exc:
            self._status_html.value = (
                f"<span style='color:red'><b>Error:</b> {exc}</span>"
            )
        finally:
            self._run_btn.disabled = False

    # -----------------------------------------------------------------------
    # Private helpers — results display
    # -----------------------------------------------------------------------

    def _render_results(self, result: dict, state: dict) -> None:
        """Render metrics, design table, and optional power curve in the Output widget."""
        report = result["report"]

        with self._output:
            _clear_output(wait=True)

            # --- Metrics summary ---
            from IPython.display import display as _d
            _d(_widgets.HTML("<h4>Run Summary</h4>"))
            metrics_html = (
                "<table style='border-collapse:collapse;font-size:13px'>"
                "<tr style='background:#f0f0f0'>"
                "<th style='padding:4px 12px;border:1px solid #ccc'>Metric</th>"
                "<th style='padding:4px 12px;border:1px solid #ccc'>Value</th>"
                "</tr>"
            )
            rows_data = [
                ("n (runs)", report.get("n", "—")),
                ("Achieved power", f"{report.get('achieved_power', 0):.3f}"),
                ("Target power", f"{report.get('target_power', 0):.3f}"),
                ("α", f"{report.get('alpha', 0):.3f}"),
                ("Noncentrality λ", f"{report.get('noncentrality_lambda', 0):.3f}"),
                ("Criterion", report.get("criterion", "—")),
                ("Elapsed (s)", f"{report.get('elapsed_sec', 0):.2f}"),
            ]
            for label, val in rows_data:
                metrics_html += (
                    f"<tr><td style='padding:3px 12px;border:1px solid #ddd'>{label}</td>"
                    f"<td style='padding:3px 12px;border:1px solid #ddd'>{val}</td></tr>"
                )
            metrics_html += "</table>"
            _d(_widgets.HTML(metrics_html))

            # --- Design table ---
            _d(_widgets.HTML("<h4>Design Matrix</h4>"))
            _d(result["design_df"])

            # --- Buckets table ---
            if result.get("buckets_df") is not None and len(result["buckets_df"]) > 0:
                _d(_widgets.HTML("<h4>Run Buckets</h4>"))
                _d(result["buckets_df"])

            # --- Power curve (Plotly) ---
            self._render_power_curve(result, state)

    def _render_power_curve(self, result: dict, state: dict) -> None:
        """Render an inline Plotly power vs n curve if plotly is available."""
        if not _HAS_PLOTLY:
            return

        from IPython.display import display as _d
        report = result["report"]
        n_opt = int(report["n"])
        n_max = max(int(state["max_n"]), n_opt + 10)
        n_min = max(1, n_opt - max(20, n_opt // 2))
        n_vals = list(range(n_min, n_max + 1))

        powers = _approx_power_curve(
            n_vals=n_vals,
            report=report,
            alpha=float(state["alpha"]),
            power_mode=state["power_mode"],
            lambda_mode=state.get("lambda_mode", "n"),
        )

        target_power = float(state["power_target"])
        fig = _go.Figure()
        fig.add_trace(_go.Scatter(
            x=n_vals, y=powers, mode="lines", name="Power",
            line=dict(color="#1f77b4", width=2),
        ))
        fig.add_hline(
            y=target_power, line_dash="dash", line_color="orange",
            annotation_text=f"Target {target_power:.2f}",
        )
        fig.add_vline(
            x=n_opt, line_dash="dot", line_color="green",
            annotation_text=f"n = {n_opt}",
        )
        fig.update_layout(
            title="Power vs Sample Size",
            xaxis_title="n (sample size)",
            yaxis_title="Power",
            yaxis=dict(range=[0, 1.05]),
            height=350,
            margin=dict(l=50, r=20, t=50, b=40),
            legend=dict(orientation="h", y=-0.15),
        )
        _d(fig)


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def design_widget(
    formula: str = "~ 1 + A + B",
    factors: Optional[Dict[str, Any]] = None,
    power_mode: str = "r2",
    alpha: float = 0.05,
    power: float = 0.80,
    sigma: float = 1.0,
    r2_target: float = 0.15,
    max_n: int = 500,
    design_opts: Optional[DesignOptions] = None,
    show_advanced: bool = False,
) -> "DesignWidget":
    """Create and immediately display an interactive design widget.

    Returns the :class:`DesignWidget` so the caller can later read
    :meth:`~DesignWidget.get_result`.

    Parameters
    ----------
    formula : str, optional
        Patsy formula. Defaults to ``"~ 1 + A + B"``.
    factors : dict, optional
        Factor spec — ``{name: (lo, hi)}`` for continuous factors,
        ``{name: ["lvl1", "lvl2", ...]}`` for categorical factors.
    power_mode : {"r2", "contrast"}, optional
        Power calculation mode. Defaults to ``"r2"``.
    alpha : float, optional
        Significance level. Defaults to ``0.05``.
    power : float, optional
        Target power. Defaults to ``0.80``.
    sigma : float, optional
        Residual standard deviation (contrast mode). Defaults to ``1.0``.
    r2_target : float, optional
        R² target (r2 mode). Defaults to ``0.15``.
    max_n : int, optional
        Maximum sample size. Defaults to ``500``.
    design_opts : DesignOptions, optional
        Seed design options. Widget-exposed fields are pre-filled; others
        are forwarded at run time.
    show_advanced : bool, optional
        Expand the Advanced options accordion on load. Defaults to ``False``.

    Returns
    -------
    DesignWidget
        The live widget instance.

    Raises
    ------
    WidgetsError
        If ipywidgets is not installed.

    Examples
    --------
    >>> from lattice_doe.widgets import design_widget
    >>> w = design_widget(
    ...     formula="~ 1 + A + B + A:B",
    ...     factors={"A": (-1.0, 1.0), "B": (-1.0, 1.0)},
    ...     power_mode="r2",
    ...     r2_target=0.20,
    ... )
    >>> # After clicking "Generate design" in the notebook:
    >>> result = w.get_result()
    """
    w = DesignWidget(
        formula=formula,
        factors=factors,
        power_mode=power_mode,
        alpha=alpha,
        power=power,
        sigma=sigma,
        r2_target=r2_target,
        max_n=max_n,
        design_opts=design_opts,
        show_advanced=show_advanced,
    )
    w.display()
    return w
