"""
Page 1 — Factors & Formula builder.

Tickets: B1 (continuous rows), B2 (categorical rows),
         B3 (formula + Patsy validation), B4 (persistence).
"""

import streamlit as st
from state import init_state, render_sidebar
from components.factor_table import render_factor_table

st.set_page_config(page_title="Factors — I-Opt Power Design", layout="wide")
init_state()
render_sidebar()

st.title("Step 1 · Factors & Formula")
st.markdown("Define the factors in your experiment and the model formula. *(Epic B — coming soon)*")

render_factor_table()
