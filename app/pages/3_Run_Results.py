"""
Page 3 — Run the design and display results.

Tickets: E1 (run + error handling), E2 (report card),
         E3 (design table), E4 (buckets table),
         E5 (power curve chart), E6 (Excel download).
"""

import streamlit as st
from state import init_state, render_sidebar

st.set_page_config(page_title="Run & Results — I-Opt Power Design", layout="wide")
init_state()
render_sidebar()

st.title("Step 3 · Run & Results")
st.markdown(
    "Complete **Step 1 (Factors)** and **Step 2 (Power Config)** first, "
    "then come here to generate the design. *(Epic E — coming soon)*"
)

result = st.session_state.get("result")
if result:
    st.success(
        f"A result is already available: n={result['report']['n']}, "
        f"power={result['report']['achieved_power']:.3f}. "
        "Full display coming in Epic E."
    )
else:
    st.info("No design generated yet.")
