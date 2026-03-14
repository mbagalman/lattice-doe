"""
Shared session state schema and sidebar renderer.

Every page should call init_state() before any other st commands, and
render_sidebar() to display the shared config summary in the left panel.
"""

from __future__ import annotations

import streamlit as st


def init_state() -> None:
    """Populate st.session_state with defaults for any missing keys. Idempotent."""
    defaults: dict = {
        # --- Factors & formula ---
        "factors": [],          # list of dicts: {name, type, low, high} or {name, type, levels}
        "formula": "~ 1 + A + B",

        # --- Power config ---
        "power_mode": "contrast",          # "contrast" | "r2"
        "contrast_input_mode": "matrix",   # "matrix" | "scenario"
        "L_text": "",
        "delta_text": "",
        "scenario_a": {},
        "scenario_b": {},
        "sesoi": 1.0,
        "r2_target": 0.15,
        "lambda_mode": "n",
        "alpha": 0.05,
        "power_target": 0.80,
        "sigma": 1.0,
        "max_n": 500,

        # --- Design options ---
        "criterion": "I",
        "starts": 8,
        "auto_candidate": True,
        "candidate_points": 2000,
        "random_state": 42,
        "constraint_expr": "",

        # --- Blocked design options (Enhancement 20) ---
        # 0 means unblocked; ≥ 2 enables blocking.
        "n_blocks": 0,
        "block_factor_name": "Block",

        # --- Categorical pre-allocation options (Enhancement 26) ---
        "preallocate_categorical": False,
        "alloc_min_per_cell": 1,
        # 0 is the sentinel for "no upper limit" (maps to alloc_max_per_cell=None).
        "alloc_max_per_cell": 0,

        # --- Split-plot / hard-to-change factor options (Enhancement 22) ---
        "split_plot_enabled": False,
        "sp_htc_factors": [],      # list of factor names that are hard-to-change
        "sp_n_whole_plots": 4,     # number of whole plots (≥ 2)
        "sp_eta": 1.0,             # variance ratio σ²_wp / σ²_sp
        "sp_subplots_per_wp": 0,   # 0 = auto-compute
        "sp_df_method": "auto",    # "auto" | "conservative" | "sp_only"

        # --- Multi-response options (MR-8) ---
        "mr_enabled": False,          # enable multi-response mode
        "mr_combination": "min",      # "min" | "product" | "weighted_mean"
        "mr_responses": [],           # list of response dicts (see Page 2)

        # --- Results (populated after a successful run) ---
        "result": None,       # full dict from i_optimal_powered_design
        "run_error": None,    # error string if the last run failed
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    """Render the shared config summary sidebar. Call from every page."""
    with st.sidebar:
        st.markdown("## I-Opt Power Design")
        st.markdown("---")

        factors = st.session_state["factors"]
        formula = st.session_state["formula"]
        power_mode = st.session_state["power_mode"]
        alpha = st.session_state["alpha"]
        power_target = st.session_state["power_target"]
        sigma = st.session_state["sigma"]
        result = st.session_state["result"]

        st.markdown("**Config summary**")
        n_factors = len(factors)
        factor_label = f"{n_factors} defined" if n_factors else "none yet"
        st.markdown(f"- Factors: **{factor_label}**")
        if formula:
            st.markdown(f"- Formula: `{formula}`")
        mode_label = "Contrast-based" if power_mode == "contrast" else "Global R\u00b2"
        st.markdown(f"- Mode: **{mode_label}**")
        if power_mode == "contrast":
            st.markdown(f"- \u03b1={alpha} · power={power_target} · \u03c3={sigma}")
        else:
            r2 = st.session_state["r2_target"]
            st.markdown(f"- \u03b1={alpha} · power={power_target} · R\u00b2={r2}")

        st.markdown("---")
        if result is not None:
            n = result["report"]["n"]
            pwr = result["report"]["achieved_power"]
            st.success(f"Result ready: n={n}, power={pwr:.3f}")
        else:
            st.info("No result yet — run a design on page 3.")

        st.markdown("---")
        if st.button("Reset all", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
