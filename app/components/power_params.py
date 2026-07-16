"""
Shared power parameter widgets — alpha, power target, sigma, max_n.

Ticket: C5.

render_power_params() reads power_mode from session state so it can hide
sigma automatically when the user is in R² mode.

Also hosts scenario_contrast(), the app-wide wrapper around
contrast_from_scenarios() that rewrites coding errors into advice the app can
actually act on (UX-46).
"""

from typing import Any, Tuple

import streamlit as st

#: Remedy when even the run's own candidate set cannot establish the coding —
#: in practice the combined-term level cross exceeding the cap.
UI_CODING_REMEDY = (
    "The Scenario builder cannot construct L for this formula. Reduce the "
    "number of levels in the combined term, or switch **Contrast input** to "
    "**Matrix (L, \u03b4)** and enter them directly. See the user guide "
    "\u00a73.3, \"Formulas whose coding is learned from the data\"."
)

#: Remedy for the preview only: the exact coding comes from the candidate set
#: the run builds, which the preview has no design options for.
UI_PREVIEW_REMEDY = (
    "This formula's coding is established from the candidate set the search "
    "generates, so L cannot be previewed here \u2014 it is built correctly "
    "when you run the design on the **Run / Results** page."
)

#: Remedy when the coding is data-dependent AND split-plot options are set.
#: The split-plot search learns its coding from separately built whole-plot/
#: sub-plot pools, not from the ordinary candidate set, so no candidate built
#: here can be the authority (UX-50).
UI_SPLIT_PLOT_REMEDY = (
    "This run uses split-plot options, and a split-plot search learns its "
    "model coding from separately built whole-plot/sub-plot pools \u2014 not "
    "from the ordinary candidate set \u2014 so no scenario contrast built up "
    "front can be guaranteed to match it. Switch **Contrast input** to "
    "**Matrix (L, \u03b4)** and enter them directly, or use a formula whose "
    "coding does not depend on the data."
)

#: Remedy when the coding is data-dependent AND candidate growth is enabled.
#: Growth rebuilds the candidate set mid-search and re-derives the coding
#: while L stays fixed, so the contrast would silently stop matching it.
UI_GROWTH_REMEDY = (
    "Because the coding is learned from the candidate set, L is built from "
    "the candidate set this run will use \u2014 but **Allow candidate growth** "
    "is enabled, which lets the search rebuild that candidate set mid-run and "
    "re-derive the coding, leaving L behind. Turn candidate growth off (it is "
    "off by default), or switch **Contrast input** to **Matrix (L, "
    "\u03b4)**."
)


def scenario_contrast(design_opts: Any = None, **kwargs: Any) -> Tuple[Any, Any]:
    """Build L and \u03b4 for a scenario pair, coded against the run's own data.

    When the formula's coding is learned from realized data (splines, derived
    categoricals), the coding authority must be the candidate set the search
    will actually select from. Passing *design_opts* builds exactly that
    candidate set; reconstructing it by hand from a guessed seed or size gives
    a same-width but numerically different L, with no error to signal it
    (UX-48).

    Parameters
    ----------
    design_opts : DesignOptions, optional
        The options the design run will use. Omit only where no run is being
        configured (the Page 2 preview); a data-dependent coding then reports
        UI_PREVIEW_REMEDY rather than guessing an authority.
    **kwargs
        Forwarded to ``contrast_from_scenarios``.

    lattice_doe is imported lazily: the pages import this module
    unconditionally and must still render their "not installed" notice when
    the package is absent.
    """
    from lattice_doe.candidate import build_search_candidate
    from lattice_doe.contrasts import (
        ContrastCodingError,
        coding_is_data_dependent,
        contrast_from_scenarios,
    )

    reason = coding_is_data_dependent(kwargs["formula"], kwargs["factors"])
    if reason is not None:
        if design_opts is None:
            raise ContrastCodingError(reason, UI_PREVIEW_REMEDY)
        if design_opts.split_plot is not None:
            raise ContrastCodingError(reason, UI_SPLIT_PLOT_REMEDY)
        if design_opts.allow_candidate_growth:
            raise ContrastCodingError(reason, UI_GROWTH_REMEDY)
        kwargs["coding_data"], _ = build_search_candidate(
            kwargs["formula"], kwargs["factors"], design_opts,
        )
    try:
        return contrast_from_scenarios(**kwargs)
    except ContrastCodingError as exc:
        raise ContrastCodingError(exc.reason, UI_CODING_REMEDY) from exc


# The one authoring contract for target power (UX-72). Every widget that
# accepts or re-displays a target — the global widget below, the
# per-response widgets on Page 2, the MDE widget on Page 4 — must share
# these bounds: a target authored inside one widget's range crashes any
# differently-bounded widget that later re-displays it
# (StreamlitValueBelowMinError). The config layer accepts any value in
# the open interval (0, 1); these clip only the unusable extremes.
# The display format is part of the contract (UX-77): the bounds permit
# four-decimal targets, and a two-decimal display would render the valid
# 0.9999 as the impossible-looking "1.00".
POWER_TARGET_MIN = 0.01
POWER_TARGET_MAX = 0.9999
POWER_TARGET_FORMAT = "%.4f"


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
            min_value=POWER_TARGET_MIN,
            max_value=POWER_TARGET_MAX,
            step=0.01,
            format=POWER_TARGET_FORMAT,
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
        elif power_mode == "glm":
            st.markdown("**Residual \u03c3**")
            st.caption("Not used in GLM mode \u2014 set Baseline on Page 2.")
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
