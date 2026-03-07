"""
Page 4 — Advanced analysis.

Tickets: F1 (sensitivity), F2 (MDE), F3 (compare criteria).
Also: G1 (YAML export), G2 (JSON report download).
"""

import streamlit as st
from state import init_state, render_sidebar

st.set_page_config(page_title="Analysis — I-Opt Power Design", layout="wide")
init_state()
render_sidebar()

st.title("Step 4 · Analysis")
st.markdown(
    "Run a design on **Step 3** first, then return here for sensitivity analysis, "
    "minimum detectable effect, and criteria comparison. *(Epics F & G — coming soon)*"
)

if st.session_state.get("result") is None:
    st.warning("No result available. Generate a design on page 3 first.")
