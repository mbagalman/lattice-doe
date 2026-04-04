"""
Lattice DOE — Streamlit home page.

Run with:
    streamlit run app/app.py

Streamlit adds the app/ directory to sys.path, so all pages can import
from state.py and components/ without any path manipulation.
"""

import streamlit as st
from state import init_state, render_sidebar

st.set_page_config(
    page_title="Lattice DOE",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
render_sidebar()

st.title("Lattice DOE")
st.markdown(
    "**I-optimal experimental designs with guaranteed statistical power** — "
    "a point-and-click interface for the `lattice-doe` package."
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("#### 1 · Factors")
    st.markdown("Define your experimental factors (continuous or categorical) and the Patsy model formula.")
with col2:
    st.markdown("#### 2 · Power Config")
    st.markdown("Choose contrast-based or global R² power mode, set effect assumptions, and configure the design search.")
with col3:
    st.markdown("#### 3 · Run & Results")
    st.markdown("Generate the optimal design, inspect the report, and download design files.")
with col4:
    st.markdown("#### 4 · Analysis")
    st.markdown("Run sensitivity analysis, find the minimum detectable effect, and compare optimality criteria.")

st.markdown("---")
st.markdown(
    "Use the **sidebar** (left) to navigate between pages. "
    "Start at **1 · Factors** and work through the pages in order."
)

with st.expander("Quick reference — what do I fill in?"):
    st.markdown("""
**Factors** — The independent variables in your experiment.
- *Continuous*: specify a numeric low/high range (e.g. Temperature 20–80 °C).
- *Categorical*: specify discrete levels (e.g. Catalyst: A, B, C).

**Formula** — A Patsy model formula describing which effects to estimate.
Examples: `~ 1 + A + B`, `~ 1 + A + B + A:B`, `~ 1 + A + B + C + A:B + A:C`.

**Power mode** — How you specify the effect you want to detect:
- *Contrast-based*: test a specific linear combination of model coefficients (e.g. "does factor B shift the mean by at least 0.5σ?").
- *Global R²*: test that the full model explains at least R² of the total variance.

**Criterion** — How the design is optimised:
- *I-optimal*: minimises average prediction variance over the factor space (recommended for prediction).
- *D-optimal*: maximises `det(X'X)`, giving the most precise coefficient estimates.
- *A-optimal*: minimises `trace((X'X)⁻¹)`, equalising coefficient estimate variances.
""")
