"""
Shared power parameter widgets — alpha, power target, sigma, max_n.

Ticket: C5.

render_power_params() reads power_mode from session state so it can hide
sigma automatically when the user is in R² mode.
"""

import streamlit as st


def render_power_params() -> None:
    """Render the four shared power parameters as a single row of inputs."""
    power_mode = st.session_state.get("power_mode", "contrast")

    st.subheader("Power Parameters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.number_input(
            "Significance level (\u03b1)",
            min_value=0.001,
            max_value=0.20,
            step=0.005,
            format="%.3f",
            key="alpha",
            help="Type I error rate. Typical value: 0.05.",
        )

    with col2:
        st.number_input(
            "Target power (1\u2212\u03b2)",
            min_value=0.50,
            max_value=0.99,
            step=0.01,
            format="%.2f",
            key="power_target",
            help="Desired probability of detecting the effect. Typical value: 0.80.",
        )

    with col3:
        if power_mode == "contrast":
            st.number_input(
                "Residual \u03c3",
                min_value=1e-6,
                step=0.1,
                format="%.4g",
                key="sigma",
                help=(
                    "Estimated residual standard deviation of the response. "
                    "Effect size \u03b4 is in the same units."
                ),
            )
        else:
            st.markdown("**Residual \u03c3**")
            st.caption("Not used in Global R\u00b2 mode.")

    with col4:
        st.number_input(
            "Max sample size",
            min_value=10,
            max_value=5000,
            step=10,
            key="max_n",
            help="Hard upper bound on the sample size search.",
        )
