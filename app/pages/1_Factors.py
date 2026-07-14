"""
Page 1 — Factors & Formula builder (Epic B, tickets B1–B4).

Manual persistence test checklist (B4):
  [ ] Add 2 continuous factors and 1 categorical factor.
  [ ] Navigate to Page 2 via the sidebar.
  [ ] Navigate back to Page 1 — all three factors should still be present
      with the same names, types, and values.
  [ ] Edit the formula and navigate away; return to confirm formula persists.
  [ ] Click "Clear all factors" — factors and formula reset to defaults;
      power config (alpha, power, sigma, etc.) and any result are unchanged.
"""

from __future__ import annotations

import streamlit as st

from state import init_state, render_sidebar
from components.factor_table import clear_all_factors, render_factor_table

# set_page_config must be the first Streamlit call.
st.set_page_config(page_title="Factors — Lattice DOE", layout="wide")
init_state()
render_sidebar()

# lattice_doe is installed via `pip install -e ".[app]"` so importable directly.
try:
    from lattice_doe.design import build_model_matrix

    _HAS_IOPT = True
except ImportError:
    _HAS_IOPT = False

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Step 1 · Factors & Formula")

# ---------------------------------------------------------------------------
# Factor table (B1 + B2)
# ---------------------------------------------------------------------------
st.subheader("Factors")
st.markdown(
    "Add the independent variables in your experiment. "
    "**Continuous** factors need a numeric range; "
    "**Categorical** factors need a comma-separated list of levels."
)

render_factor_table()

# Clear-all button — placed before the formula input so that when it triggers
# st.rerun(), the formula text_input re-renders with the reset value.
if st.session_state.get("factor_ids"):
    st.markdown("")
    if st.button("Clear all factors", type="secondary"):
        clear_all_factors()
        # Reset formula to default before rerun so the text_input picks it up.
        st.session_state["formula"] = "~ 1 + A + B"
        st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Formula input + Patsy validation (B3)
# ---------------------------------------------------------------------------
st.subheader("Model Formula")
st.markdown(
    "Specify which effects to estimate using "
    "[Patsy](https://patsy.readthedocs.io/) notation. "
    "The formula determines the model matrix columns and the number of "
    "parameters **p** — you will need p when specifying the contrast matrix L on Page 2."
)

# key="formula" links directly to st.session_state["formula"] (set by init_state).
st.text_input(
    "Formula",
    key="formula",
    placeholder="~ 1 + A + B + A:B",
    help=(
        "Examples:\n"
        "  ~ 1 + A + B           main effects only\n"
        "  ~ 1 + A + B + A:B     main effects + two-way interaction\n"
        "  ~ 1 + A + B + C + A:B + A:C   multiple interactions\n\n"
        "Factor names must match exactly what you typed above."
    ),
)

# --- Live validation ---
factors = st.session_state.get("factors", [])
formula = st.session_state.get("formula", "")

if not factors:
    st.info("Add at least one factor above to validate the formula.")
elif not formula.strip():
    st.warning("Enter a formula above.")
elif not _HAS_IOPT:
    st.warning(
        "Could not import `lattice_doe` — run `pip install -e '.[app]'` "
        "from the project root to enable formula validation."
    )
else:
    try:
        # Representative preview over the full cross of categorical levels
        # (UX-1): a one-row frame with only the first level made Patsy drop
        # the remaining dummy columns, so p was undercounted for
        # categorical models and interactions.
        from lattice_doe.utils import model_matrix_preview

        spec: dict = {}
        for f in factors:
            if not f["name"].strip():
                raise ValueError("One or more factors have an empty name.")
            if f["type"] == "Continuous":
                spec[f["name"]] = (f["low"], f["high"])
            else:
                if not f["levels"]:
                    raise ValueError(
                        f"Factor '{f['name']}' has no levels — "
                        "enter at least one level (e.g. 'low, high')."
                    )
                spec[f["name"]] = list(f["levels"])

        p, col_names = model_matrix_preview(formula, spec)

        st.success(f"Valid formula — **p = {p}** model parameter{'s' if p != 1 else ''}.")
        with st.expander(f"Model matrix columns ({p} total)"):
            for i, name in enumerate(col_names):
                st.text(f"  [{i}]  {name}")

    except Exception as exc:
        st.error(f"Formula error: {exc}")
