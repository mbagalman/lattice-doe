"""
Page 2 — Power configuration and design options.

Epics C (C1–C5) and D (D1–D3) are fully implemented here.

Layout
------
C1  Power mode radio (Contrast-based / Global R²)
C2  Contrast mode — L matrix + delta text areas with shape validation
C3  Contrast mode — scenario builder (factor-by-factor A vs B inputs)
C4  R² mode — r2_target slider + lambda_mode radio
C5  Shared power params (alpha, power_target, sigma, max_n)
D1  Criterion selectbox      } inside collapsible
D2  Search options           } "Advanced design options"
D3  Constraint expression    } expander
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from state import init_state, render_sidebar
from components.power_params import render_power_params

st.set_page_config(page_title="Power Config — I-Opt Power Design", layout="wide")
init_state()
render_sidebar()

try:
    from iopt_power_design.design import build_model_matrix
    from iopt_power_design.contrasts import contrast_from_scenarios
    _HAS_IOPT = True
except ImportError:
    _HAS_IOPT = False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _factors_to_spec(factors: list[dict]) -> dict:
    """Convert session-state factor list to the API factor-spec dict."""
    spec: dict = {}
    for f in factors:
        if f["type"] == "Continuous":
            spec[f["name"]] = (f["low"], f["high"])
        else:
            spec[f["name"]] = list(f["levels"])
    return spec


def _get_p(factors: list[dict], formula: str) -> int | None:
    """Return model parameter count p, or None if unavailable."""
    if not factors or not formula.strip() or not _HAS_IOPT:
        return None
    try:
        row: dict = {}
        for f in factors:
            if f["type"] == "Continuous":
                row[f["name"]] = [(f["low"] + f["high"]) / 2.0]
            else:
                row[f["name"]] = [f["levels"][0]] if f["levels"] else ["?"]
        _, col_names = build_model_matrix(formula, pd.DataFrame(row))
        return len(col_names)
    except Exception:
        return None


def _parse_matrix(text: str) -> np.ndarray:
    """Parse newline-separated rows of space/comma values into a 2D float array."""
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.replace(",", " ").split()])
    return np.array(rows)


def _parse_vector(text: str) -> np.ndarray:
    """Parse space/comma-separated text into a 1D float array."""
    return np.array([float(x) for x in text.replace(",", " ").split()])


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("Step 2 \u00b7 Power Configuration & Design Options")

factors = st.session_state.get("factors", [])
formula = st.session_state.get("formula", "")

# ===========================================================================
# C1 — Power mode toggle
# ===========================================================================
st.subheader("Power Mode")

_POWER_MODES = ["Contrast-based", "Global R\u00b2"]
_mode_idx = 0 if st.session_state["power_mode"] == "contrast" else 1
_selected_mode = st.radio(
    "Power mode",
    _POWER_MODES,
    index=_mode_idx,
    horizontal=True,
    label_visibility="collapsed",
    help=(
        "**Contrast-based**: test a specific linear combination of coefficients "
        "(e.g. 'does factor B shift the mean by at least 0.5\u03c3?').\n\n"
        "**Global R\u00b2**: test whether the full model explains at least R\u00b2 "
        "of total variance."
    ),
)
st.session_state["power_mode"] = "contrast" if _selected_mode == "Contrast-based" else "r2"

st.markdown("---")

# ===========================================================================
# C2 + C3  (contrast mode)  |  C4  (R² mode)
# ===========================================================================

if st.session_state["power_mode"] == "contrast":

    st.subheader("Contrast Specification")

    # Input method toggle
    _INPUT_METHODS = ["Matrix (L, \u03b4)", "Scenario builder"]
    _input_idx = 0 if st.session_state["contrast_input_mode"] == "matrix" else 1
    _selected_input = st.radio(
        "Input method",
        _INPUT_METHODS,
        index=_input_idx,
        horizontal=True,
    )
    st.session_state["contrast_input_mode"] = (
        "matrix" if _selected_input == _INPUT_METHODS[0] else "scenario"
    )

    st.markdown("")

    # -----------------------------------------------------------------------
    # C2 — L matrix + delta
    # -----------------------------------------------------------------------
    if st.session_state["contrast_input_mode"] == "matrix":

        p = _get_p(factors, formula)
        if p is not None:
            st.caption(
                f"Your formula has **p\u202f=\u202f{p}** model parameters. "
                f"Each row of L must have exactly {p} column(s). "
                "See Page 1 \u2192 'Model matrix columns' for parameter indices."
            )
        elif not factors:
            st.info("Define factors on Page 1 first \u2014 p is needed to validate L.")

        col_l, col_d = st.columns([3, 2])
        with col_l:
            st.markdown("**Contrast matrix L**")
            st.caption("One row per contrast; values space- or comma-separated.")
            st.text_area(
                "L",
                key="L_text",
                height=130,
                label_visibility="collapsed",
                placeholder="0  0  1  0\n0  0  0  1",
            )
        with col_d:
            st.markdown("**Effect size \u03b4**")
            st.caption("One value per row of L.")
            st.text_area(
                "delta",
                key="delta_text",
                height=130,
                label_visibility="collapsed",
                placeholder="0.5\n0.5",
            )

        # Live shape validation
        L_text = st.session_state.get("L_text", "").strip()
        delta_text = st.session_state.get("delta_text", "").strip()
        if L_text and delta_text:
            try:
                L = _parse_matrix(L_text)
                delta = _parse_vector(delta_text)
                if L.ndim != 2 or L.size == 0:
                    st.error("L must contain at least one row of values.")
                elif p is not None and L.shape[1] != p:
                    st.error(
                        f"L has {L.shape[1]} column(s) but the formula has p\u202f=\u202f{p}. "
                        "Each row of L needs one value per model parameter."
                    )
                elif L.shape[0] != len(delta):
                    st.error(
                        f"L has {L.shape[0]} row(s) but \u03b4 has {len(delta)} value(s) \u2014 "
                        "they must match."
                    )
                else:
                    st.success(
                        f"Valid \u2014 L is ({L.shape[0]}\u202f\u00d7\u202f{L.shape[1]}), "
                        f"\u03b4 has {len(delta)} value(s)."
                    )
            except ValueError as exc:
                st.error(f"Parse error: {exc}")
        elif L_text or delta_text:
            st.warning("Fill in both L and \u03b4 to validate.")

        with st.expander("What is a contrast matrix?"):
            st.markdown(
                """
A contrast matrix **L** (q\u202f\u00d7\u202fp) selects which linear combination of model
coefficients to test. Each row is one hypothesis: `H\u2080: L\u03b2\u202f=\u202f0`
vs `H\u2081: L\u03b2\u202f=\u202f\u03b4`.

**Example** \u2014 test that the B main effect (column index 2) equals 0.5:
```
L = [[0, 0, 1, 0]]
\u03b4 = [0.5]
```
**Tip**: go to Page 1, enter your formula, and open **"Model matrix columns"**
to find the index of each parameter.

**Alternative**: switch to **Scenario builder** to construct L and \u03b4 automatically.
"""
            )

    # -----------------------------------------------------------------------
    # C3 — Scenario builder
    # -----------------------------------------------------------------------
    else:
        if not factors:
            st.info("Define your factors on Page 1 before using the scenario builder.")
        else:
            st.markdown(
                "Specify two factor settings (Scenario A vs B). "
                "L and \u03b4 are computed automatically from the difference."
            )

            hcols = st.columns([2, 2.5, 2.5])
            hcols[0].markdown("**Factor**")
            hcols[1].markdown("**Scenario A**")
            hcols[2].markdown("**Scenario B**")

            for f in factors:
                fname = f["name"] or "(unnamed)"
                cols = st.columns([2, 2.5, 2.5])
                cols[0].markdown(f"`{fname}`")

                if f["type"] == "Continuous":
                    if f"scen_a_{fname}" not in st.session_state:
                        st.session_state[f"scen_a_{fname}"] = f["low"]
                    if f"scen_b_{fname}" not in st.session_state:
                        st.session_state[f"scen_b_{fname}"] = f["high"]
                    cols[1].number_input(
                        f"A:{fname}",
                        key=f"scen_a_{fname}",
                        label_visibility="collapsed",
                        format="%.4g",
                    )
                    cols[2].number_input(
                        f"B:{fname}",
                        key=f"scen_b_{fname}",
                        label_visibility="collapsed",
                        format="%.4g",
                    )
                else:
                    levels = f["levels"] if f["levels"] else ["(no levels)"]
                    # Reset any stale saved values that are no longer valid options
                    for prefix in ("scen_a", "scen_b"):
                        if st.session_state.get(f"{prefix}_{fname}") not in levels:
                            st.session_state[f"{prefix}_{fname}"] = levels[0]
                    # Default B to last level for a non-trivial contrast
                    if f"scen_b_{fname}" not in st.session_state:
                        st.session_state[f"scen_b_{fname}"] = levels[-1]

                    cols[1].selectbox(
                        f"A:{fname}",
                        options=levels,
                        key=f"scen_a_{fname}",
                        label_visibility="collapsed",
                    )
                    cols[2].selectbox(
                        f"B:{fname}",
                        options=levels,
                        key=f"scen_b_{fname}",
                        label_visibility="collapsed",
                    )

            st.markdown("")
            st.number_input(
                "SESOI (smallest effect of interest, in response units)",
                min_value=1e-6,
                step=0.1,
                format="%.4g",
                key="sesoi",
                help=(
                    "The minimum scientifically meaningful difference in the response "
                    "between Scenario A and B."
                ),
            )

            with st.expander("Preview L and \u03b4"):
                if not _HAS_IOPT:
                    st.warning(
                        "iopt_power_design not importable \u2014 "
                        "run `pip install -e '.[app]'` to enable preview."
                    )
                elif not formula.strip():
                    st.info("Set the formula on Page 1 to preview L and \u03b4.")
                else:
                    try:
                        factor_spec = _factors_to_spec(factors)
                        scenario_a = {
                            f["name"]: st.session_state.get(f"scen_a_{f['name']}")
                            for f in factors
                        }
                        scenario_b = {
                            f["name"]: st.session_state.get(f"scen_b_{f['name']}")
                            for f in factors
                        }
                        sesoi = float(st.session_state.get("sesoi", 1.0))
                        L, delta = contrast_from_scenarios(
                            formula=formula,
                            factors=factor_spec,
                            scenario_a=scenario_a,
                            scenario_b=scenario_b,
                            sesoi=sesoi,
                        )
                        st.markdown(f"**L** (1\u202f\u00d7\u202f{L.shape[1]}):")
                        st.code(np.array2string(L, precision=4, suppress_small=True))
                        st.markdown(
                            f"**\u03b4** = `{np.array2string(delta, precision=4, suppress_small=True)}`"
                        )
                    except Exception as exc:
                        st.error(f"Could not compute contrast: {exc}")

# ===========================================================================
# C4 — R² mode
# ===========================================================================
else:
    st.subheader("R\u00b2 Effect Size")
    st.markdown(
        "Detect whether the full model explains at least this fraction of total variance."
    )

    st.slider(
        "R\u00b2 target",
        min_value=0.01,
        max_value=0.99,
        step=0.01,
        key="r2_target",
        help=(
            "Minimum R\u00b2 to detect at the specified power. "
            "Cohen\u2019s benchmarks: small\u202f=\u202f0.02, "
            "medium\u202f=\u202f0.13, large\u202f=\u202f0.26."
        ),
    )

    _LAMBDA_KEYS = ["n", "n_minus_p"]
    _LAMBDA_LABELS = [
        "n  (matches G\u2217Power / statsmodels)",
        "n \u2212 p  (more conservative)",
    ]
    _lambda_idx = _LAMBDA_KEYS.index(st.session_state.get("lambda_mode", "n"))
    _selected_lambda = st.radio(
        "Noncentrality convention (\u03bb\u202f=\u202ff\u00b2\u202f\u00d7\u202f\u22ef)",
        _LAMBDA_LABELS,
        index=_lambda_idx,
        help=(
            "**n**: \u03bb = f\u00b2 \u00b7 n \u2014 standard; matches G\u2217Power.\n\n"
            "**n \u2212 p**: \u03bb = f\u00b2 \u00b7 (n \u2212 p) \u2014 uses residual df; "
            "more conservative for small samples."
        ),
    )
    st.session_state["lambda_mode"] = _LAMBDA_KEYS[_LAMBDA_LABELS.index(_selected_lambda)]

st.markdown("---")

# ===========================================================================
# C5 — Shared power parameters
# ===========================================================================
render_power_params()

st.markdown("---")

# ===========================================================================
# D1 + D2 + D3 — Advanced design options
# ===========================================================================
with st.expander("Advanced design options"):

    # D1 — Criterion
    st.selectbox(
        "Optimality criterion",
        options=["I", "D", "A"],
        key="criterion",
        help=(
            "**I**: minimise average prediction variance over the factor space "
            "(recommended when prediction accuracy matters).\n\n"
            "**D**: maximise det(X\u2019X) \u2014 most precise coefficient estimates.\n\n"
            "**A**: minimise trace((X\u2019X)\u207b\u00b9) \u2014 "
            "equalises coefficient estimate variances."
        ),
    )

    st.markdown("---")

    # D2 — Search options
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Random starts",
            min_value=1,
            max_value=50,
            key="starts",
            help="More starts reduce the chance of a suboptimal design; increase runtime.",
        )
        st.number_input(
            "Random state (seed)",
            min_value=0,
            step=1,
            key="random_state",
            help="Integer seed for fully reproducible designs.",
        )
    with col2:
        st.checkbox(
            "Auto-size candidate set (recommended)",
            key="auto_candidate",
            help=(
                "Automatically sizes the candidate set based on factor count, "
                "type, and categorical complexity."
            ),
        )
        if not st.session_state.get("auto_candidate", True):
            st.number_input(
                "Candidate points",
                min_value=100,
                max_value=50000,
                step=100,
                key="candidate_points",
                help="Fixed candidate set size when auto-size is off.",
            )

    st.markdown("---")

    # D3 — Constraint expression
    st.text_input(
        "Feasibility constraint expression",
        key="constraint_expr",
        placeholder="not (Temperature > 70 and Time < 2)",
        help=(
            "Optional. Excludes infeasible candidate points using a Python expression. "
            "Factor names are available as variables.\n\n"
            "Allowed functions: abs, min, max, round, sqrt, log, log2, log10, exp, "
            "floor, ceil, pi."
        ),
    )
    _expr = st.session_state.get("constraint_expr", "").strip()
    if _expr:
        try:
            compile(_expr, "<constraint_expr>", "eval")
            st.success("Valid expression syntax.")
        except SyntaxError as exc:
            st.error(f"Syntax error: {exc.msg} (at position {exc.offset})")
