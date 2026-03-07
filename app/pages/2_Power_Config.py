"""
Page 2 — Power configuration and design options.

Tickets: C1–C5 (power mode, contrast, R², shared params),
         D1–D3 (criterion, search options, constraint expr).
"""

import streamlit as st
from state import init_state, render_sidebar
from components.power_params import render_power_params

st.set_page_config(page_title="Power Config — I-Opt Power Design", layout="wide")
init_state()
render_sidebar()

st.title("Step 2 · Power Configuration & Design Options")
st.markdown("Set your power mode, effect assumptions, and design search options. *(Epics C & D — coming soon)*")

render_power_params()
