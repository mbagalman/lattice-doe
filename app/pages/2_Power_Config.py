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
import streamlit as st

from state import init_state, render_sidebar
from components.power_params import render_power_params

st.set_page_config(page_title="Power Config — Lattice DOE", layout="wide")
init_state()
render_sidebar()

try:
    from lattice_doe.design import build_model_matrix
    from lattice_doe.contrasts import contrast_from_scenarios
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


def _get_p(factors: list[dict], formula: str) -> tuple[int, bool] | None:
    """Return (model parameter count p, exact), or None if unavailable.

    Uses a representative frame over the categorical levels (UX-1): a
    single-row example with only the first level made Patsy drop the
    remaining dummy columns, so p was undercounted for categorical models.
    ``exact`` is False when the count is provisional — continuous factors are
    previewed at midpoints (and huge categorical spaces with a compact level
    cover), so data-derived terms may add columns at run time.
    """
    if not factors or not formula.strip() or not _HAS_IOPT:
        return None
    try:
        from lattice_doe.utils import model_matrix_preview

        p, _, exact = model_matrix_preview(
            formula, _factors_to_spec(factors), return_exact=True
        )
        return p, exact
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

_POWER_MODES = ["Contrast-based", "Global R\u00b2", "GLM (logistic/Poisson)"]
_MODE_KEYS = ["contrast", "r2", "glm"]
_mode_idx = _MODE_KEYS.index(st.session_state.get("power_mode", "contrast"))
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
        "of total variance.\n\n"
        "**GLM (logistic/Poisson)**: Wald \u03c7\u00b2 power for binary or count "
        "outcomes. Effect size \u03b4 is on the linear-predictor scale "
        "(log-odds for logistic; log-rate for Poisson)."
    ),
)
st.session_state["power_mode"] = _MODE_KEYS[_POWER_MODES.index(_selected_mode)]

st.markdown("---")

# ===========================================================================
# C2 + C3  (contrast mode)  |  C4  (R² mode)
# ===========================================================================

if st.session_state["power_mode"] == "glm":

    st.subheader("GLM Specification")

    # Family + baseline
    _GLM_FAMILY_OPTS = ["binomial", "poisson"]
    _GLM_FAMILY_LABELS = ["Binomial (logistic)", "Poisson (log)"]
    _glm_fam_idx = _GLM_FAMILY_OPTS.index(st.session_state.get("glm_family", "binomial"))
    _glm_fam_sel = st.selectbox(
        "Family",
        _GLM_FAMILY_LABELS,
        index=_glm_fam_idx,
        help=(
            "**Binomial**: binary / proportion outcomes (logit link by default).\n\n"
            "**Poisson**: count outcomes (log link by default)."
        ),
    )
    st.session_state["glm_family"] = _GLM_FAMILY_OPTS[_GLM_FAMILY_LABELS.index(_glm_fam_sel)]

    _glm_family = st.session_state["glm_family"]
    if _glm_family == "binomial":
        _baseline_label = "Baseline event probability (0 \u2013 1)"
        _baseline_help = (
            "Event probability under the null / reference scenario. "
            "E.g. 0.20 means a 20\u202f% baseline conversion rate. "
            "Must be strictly between 0 and 1."
        )
        _baseline_min = 0.001
        _baseline_max = 0.999
    else:
        _baseline_label = "Baseline event rate (\u03bc\u2080 > 0)"
        _baseline_help = (
            "Expected count under the null scenario. "
            "E.g. 2.5 means 2.5 events per unit. Must be positive."
        )
        _baseline_min = 0.001
        _baseline_max = 1000.0

    st.number_input(
        _baseline_label,
        min_value=_baseline_min,
        max_value=_baseline_max,
        step=0.01 if _glm_family == "binomial" else 0.1,
        format="%.4g",
        key="glm_baseline",
        help=_baseline_help,
    )

    st.markdown("")
    st.caption(
        "\u03b4 (effect size) is on the **linear-predictor scale**: "
        "log-odds difference for logistic (e.g.\u202f0.5\u202f\u2248\u202f0.5\u202fnat increase "
        "in log-odds, odds ratio\u202f\u2248\u202f1.65); "
        "log-rate difference for Poisson (e.g.\u202f0.3\u202f\u2248\u202f30\u202f% increase in rate)."
    )

    # L matrix + delta (same widgets as contrast mode; delta is on LP scale)
    _p_info = _get_p(factors, formula)
    if _p_info is not None:
        p, _p_exact = _p_info
        st.caption(
            f"Your formula has **p\u202f=\u202f{p}** model parameters"
            + ("" if _p_exact else " (provisional \u2014 data-derived terms may add columns at run time)")
            + f". Each row of L must have exactly {p} column(s)."
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
        st.markdown("**Effect size \u03b4 (LP scale)**")
        st.caption("One value per row of L (log-odds or log-rate units).")
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

    if _glm_family == "binomial" and float(st.session_state.get("glm_baseline", 0.20)) in (0.0, 1.0):
        st.error("Baseline must be strictly between 0 and 1 for the binomial family.")

elif st.session_state["power_mode"] == "contrast":

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

        _p_info = _get_p(factors, formula)
        if _p_info is not None:
            p, _p_exact = _p_info
            st.caption(
                f"Your formula has **p\u202f=\u202f{p}** model parameters"
                + ("" if _p_exact else " (provisional \u2014 data-derived terms may add columns at run time)")
                + f". Each row of L must have exactly {p} column(s). "
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
                        "lattice_doe not importable \u2014 "
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
elif st.session_state["power_mode"] == "r2":
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
# MR — Multi-response (optional)
# ===========================================================================
with st.expander("Multi-response (optional)"):
    st.markdown(
        "Optimise a **single** design that simultaneously achieves adequate power "
        "for multiple response variables.  Each response may have its own sigma, "
        "contrast or R² criterion, and relative weight."
    )
    st.checkbox(
        "Enable multi-response mode",
        key="mr_enabled",
        help=(
            "When enabled, the Run page will call the compound multi-response "
            "optimiser instead of the single-response path."
        ),
    )

    if st.session_state.get("mr_enabled", False):
        ss = st.session_state

        # Combination rule
        _MR_COMB_KEYS = ["min", "product", "weighted_mean"]
        _MR_COMB_LABELS = [
            "min — all responses must reach target (recommended)",
            "product — joint probability (assumes independence)",
            "weighted_mean — weighted average power",
        ]
        _comb_idx = _MR_COMB_KEYS.index(ss.get("mr_combination", "min"))
        _sel_comb = st.radio(
            "Power combination rule",
            _MR_COMB_LABELS,
            index=_comb_idx,
            help="How per-response powers are aggregated into a single scalar for the n-search.",
        )
        ss["mr_combination"] = _MR_COMB_KEYS[_MR_COMB_LABELS.index(_sel_comb)]

        st.markdown("---")
        st.markdown("**Joint response covariance (optional)**")
        st.caption(
            "For Hotelling T² joint power (MR-3): enter a k×k symmetric positive-definite "
            "covariance matrix σ_joint where k = number of responses. "
            "Available only for shared-formula contrast-mode responses. Leave blank to use "
            "independent per-response power (recommended in most cases)."
        )
        st.text_area(
            "σ_joint matrix (space/comma-separated rows, one per line)",
            key="mr_sigma_joint",
            height=80,
            placeholder="1.0  0.3\n0.3  1.0",
            help=(
                "k×k covariance matrix of the response residuals. "
                "Enables Hotelling T² joint power calculation instead of independent combination."
            ),
        )
        # Load the response list BEFORE the covariance validation below uses
        # its length (reading it after caused a NameError as soon as any
        # σ_joint text was entered).
        mr_responses: list = ss.get("mr_responses", [])

        _sj_text = ss.get("mr_sigma_joint", "").strip()
        if _sj_text:
            try:
                _sj_rows = []
                for _line in _sj_text.splitlines():
                    _line = _line.strip()
                    if _line:
                        _sj_rows.append([float(x) for x in _line.replace(",", " ").split()])
                _sj_arr = np.array(_sj_rows)
                k = len(mr_responses)
                if _sj_arr.shape != (k, k):
                    st.warning(
                        f"σ_joint is {_sj_arr.shape[0]}×{_sj_arr.shape[1]} "
                        f"but you have {k} response(s) — must be {k}×{k}."
                    )
                else:
                    st.success(f"Valid {k}×{k} σ_joint matrix.")
            except ValueError as _sj_err:
                st.error(f"σ_joint parse error: {_sj_err}")

        st.markdown("---")
        st.markdown("**Response list**")

        # Add response button
        if st.button("+ Add response"):
            _cur_mode = ss.get("power_mode", "contrast")
            mr_responses.append({
                "name": f"Response{len(mr_responses) + 1}",
                "power_mode": _cur_mode if _cur_mode in ("contrast", "r2", "glm") else "contrast",
                "sigma": float(ss.get("sigma", 1.0)),
                "alpha": float(ss.get("alpha", 0.05)),
                "power": float(ss.get("power_target", 0.80)),
                "weight": 1.0,
                "L_text": ss.get("L_text", ""),
                "delta_text": ss.get("delta_text", ""),
                "r2_target": float(ss.get("r2_target", 0.15)),
                "formula": "",
                "glm_family": ss.get("glm_family", "binomial"),
                "glm_baseline": float(ss.get("glm_baseline", 0.20)),
                "glm_link": ss.get("glm_link", ""),
            })
            ss["mr_responses"] = mr_responses
            st.rerun()

        if not mr_responses:
            st.info("Click **+ Add response** to add at least 2 response variables.")
        else:
            for i, r in enumerate(mr_responses):
                with st.expander(f"Response {i + 1}: {r.get('name', '')}"):
                    r["name"] = st.text_input(
                        "Name", value=r.get("name", ""), key=f"mr_name_{i}"
                    )
                    r["formula"] = st.text_input(
                        "Per-response formula (blank = use global formula)",
                        value=r.get("formula", ""),
                        key=f"mr_formula_{i}",
                        placeholder="~ 1 + A + B  (leave blank for global formula)",
                        help=(
                            "Override the global model formula for this response. "
                            "When any response has a different formula, the compound-criterion "
                            "path is activated (MR-5). Leave blank to share the global formula."
                        ),
                    )
                    r_mode_opts = ["contrast", "r2", "glm"]
                    r_mode_labels = ["Contrast", "Global R\u00b2", "GLM"]
                    _r_mode_val = r.get("power_mode", "contrast")
                    if _r_mode_val not in r_mode_opts:
                        _r_mode_val = "contrast"
                    r_mode_idx = r_mode_opts.index(_r_mode_val)
                    _r_mode_sel = st.radio(
                        "Power mode", r_mode_labels, index=r_mode_idx,
                        horizontal=True, key=f"mr_mode_{i}",
                    )
                    r["power_mode"] = r_mode_opts[r_mode_labels.index(_r_mode_sel)]

                    _r_is_glm = r["power_mode"] == "glm"
                    c_sigma, c_alpha, c_power, c_weight = st.columns(4)
                    if not _r_is_glm:
                        r["sigma"] = c_sigma.number_input(
                            "\u03c3", value=r.get("sigma", 1.0), min_value=1e-6,
                            format="%.4g", key=f"mr_sigma_{i}",
                        )
                    else:
                        c_sigma.markdown("**\u03c3**")
                        c_sigma.caption("N/A (GLM)")
                    r["alpha"] = c_alpha.number_input(
                        "\u03b1", value=r.get("alpha", 0.05), min_value=1e-4,
                        max_value=0.5, format="%.3f", key=f"mr_alpha_{i}",
                    )
                    r["power"] = c_power.number_input(
                        "Target power", value=r.get("power", 0.80),
                        min_value=0.01, max_value=0.9999, format="%.2f",
                        key=f"mr_power_{i}",
                    )
                    r["weight"] = c_weight.number_input(
                        "Weight", value=r.get("weight", 1.0), min_value=0.01,
                        format="%.3g", key=f"mr_weight_{i}",
                    )
                    if r["power_mode"] in ("contrast", "glm"):
                        if _r_is_glm:
                            _glm_r_fam_opts = ["binomial", "poisson"]
                            _glm_r_fam_labels = ["Binomial", "Poisson"]
                            _glm_r_fam_idx = _glm_r_fam_opts.index(r.get("glm_family", "binomial"))
                            _glm_r_fam_sel = st.selectbox(
                                "Family", _glm_r_fam_labels, index=_glm_r_fam_idx,
                                key=f"mr_glm_family_{i}",
                            )
                            r["glm_family"] = _glm_r_fam_opts[_glm_r_fam_labels.index(_glm_r_fam_sel)]
                            r["glm_baseline"] = st.number_input(
                                "Baseline",
                                value=float(r.get("glm_baseline", 0.20)),
                                min_value=0.001,
                                max_value=0.999 if r["glm_family"] == "binomial" else 1000.0,
                                format="%.4g",
                                key=f"mr_glm_baseline_{i}",
                                help="Event probability (binomial) or rate (Poisson).",
                            )
                        _delta_label = "\u03b4 (LP scale)" if _r_is_glm else "\u03b4 (effect size)"
                        r["L_text"] = st.text_area(
                            "Contrast matrix L (one row; space/comma-separated)",
                            value=r.get("L_text", ""),
                            height=80, key=f"mr_L_{i}",
                        )
                        r["delta_text"] = st.text_input(
                            _delta_label, value=r.get("delta_text", ""),
                            key=f"mr_delta_{i}",
                        )
                    else:
                        r["r2_target"] = st.number_input(
                            "R\u00b2 target", value=r.get("r2_target", 0.15),
                            min_value=0.01, max_value=0.99, format="%.3f",
                            key=f"mr_r2_{i}",
                        )
                    if st.button(f"Remove response {i + 1}", key=f"mr_remove_{i}"):
                        mr_responses.pop(i)
                        ss["mr_responses"] = mr_responses
                        st.rerun()

            ss["mr_responses"] = mr_responses

        if len(mr_responses) >= 2:
            st.success(f"{len(mr_responses)} responses configured.")
        elif len(mr_responses) == 1:
            st.warning("Add at least one more response (minimum 2 required).")

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

    st.markdown("---")

    # D4 — Blocked design
    st.markdown("**Blocked Design**")
    st.caption(
        "Groups runs into blocks to absorb nuisance variation (e.g. day-to-day "
        "or operator effects). Set *Number of blocks* ≥ 2 to enable; "
        "leave at 0 for an unblocked design."
    )
    col_blk1, col_blk2 = st.columns(2)
    with col_blk1:
        st.number_input(
            "Number of blocks (0 = unblocked)",
            min_value=0,
            max_value=100,
            step=1,
            key="n_blocks",
            help=(
                "0 disables blocking. "
                "≥ 2 runs independent I-optimal searches within each block and "
                "adds block dummy columns to the power calculation."
            ),
        )
    with col_blk2:
        st.text_input(
            "Block factor name",
            key="block_factor_name",
            help=(
                "Column name for the block assignment in the design table. "
                "Must not clash with any treatment factor name."
            ),
        )
    if int(st.session_state.get("n_blocks", 0)) == 1:
        st.warning("Number of blocks must be 0 (unblocked) or ≥ 2.")

    st.markdown("---")

    # D6 — Split-Plot / Hard-to-Change Factors
    st.markdown("**Split-Plot / Hard-to-Change Factors**")
    st.caption(
        "Use when some factors are expensive or time-consuming to reset between runs. "
        "Groups runs into whole plots (WPs); HTC factors are constant within each WP. "
        "Enable the toggle below to activate split-plot mode."
    )
    st.checkbox(
        "Enable split-plot design",
        key="split_plot_enabled",
        help=(
            "Activates the split-plot exchange algorithm and GLS power calculation. "
            "Mutually exclusive with blocked designs."
        ),
    )
    if st.session_state.get("split_plot_enabled", False):
        factor_names = [f["name"] for f in factors if f.get("name")]
        if not factor_names:
            st.info("Define factors on Page 1 first.")
        else:
            col_sp1, col_sp2 = st.columns(2)
            with col_sp1:
                st.multiselect(
                    "Hard-to-change (WP) factors",
                    options=factor_names,
                    key="sp_htc_factors",
                    help=(
                        "Factors whose settings are fixed for all runs within a whole plot. "
                        "Easy-to-change factors vary freely within each WP."
                    ),
                )
                st.number_input(
                    "Number of whole plots (≥ 2)",
                    min_value=2,
                    max_value=200,
                    step=1,
                    key="sp_n_whole_plots",
                    help="How many whole plots (outer randomisation units) to generate.",
                )
            with col_sp2:
                st.slider(
                    "Variance ratio η = σ²_wp / σ²_sp",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    key="sp_eta",
                    help=(
                        "Ratio of whole-plot to sub-plot variance. "
                        "η = 0 reduces to OLS (no WP random effect). "
                        "Typical values: 0.5–5. Higher η → lower WP-factor power."
                    ),
                )
                st.number_input(
                    "Sub-plots per WP (0 = auto)",
                    min_value=0,
                    max_value=100,
                    step=1,
                    key="sp_subplots_per_wp",
                    help=(
                        "Number of runs within each whole plot. "
                        "0 = auto-compute from model size and number of WPs."
                    ),
                )
            st.selectbox(
                "Denominator df method",
                options=["auto", "conservative", "sp_only"],
                key="sp_df_method",
                help=(
                    "**auto**: classifies each contrast as WP or SP and uses the appropriate df.\n\n"
                    "**conservative**: always uses WP df (safest; never anti-conservative).\n\n"
                    "**sp_only**: always uses SP df (may be anti-conservative for WP effects)."
                ),
            )
        _n_blocks = int(st.session_state.get("n_blocks", 0))
        if _n_blocks >= 2:
            st.warning(
                "Split-plot and blocked design are both enabled. "
                "They cannot be used together — the run will fail. "
                "Set **Number of blocks** to 0 or disable split-plot."
            )

    st.markdown("---")

    # D5 — Categorical pre-allocation
    st.markdown("**Categorical Pre-Allocation**")
    st.caption(
        "When your design includes categorical factors, the Wynn multiplicative "
        "algorithm can pre-allocate runs across factor-level cells before the "
        "Fedorov point-exchange search, improving balance."
    )
    st.checkbox(
        "Enable categorical pre-allocation (Wynn algorithm)",
        key="preallocate_categorical",
        help=(
            "Applies the Wynn multiplicative I-optimal allocation to distribute "
            "runs across categorical cells before the exchange search. "
            "Ignored for purely continuous designs."
        ),
    )
    if st.session_state.get("preallocate_categorical", False):
        col_alloc1, col_alloc2 = st.columns(2)
        with col_alloc1:
            st.number_input(
                "Min runs per cell",
                min_value=1,
                max_value=100,
                step=1,
                key="alloc_min_per_cell",
                help="Minimum number of runs guaranteed in each factor-level cell.",
            )
        with col_alloc2:
            st.number_input(
                "Max runs per cell (0 = no limit)",
                min_value=0,
                max_value=10000,
                step=1,
                key="alloc_max_per_cell",
                help="Upper bound on runs per cell. 0 means no upper limit.",
            )
