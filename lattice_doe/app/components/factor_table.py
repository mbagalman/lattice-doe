"""
Factor entry widget — Epic B (tickets B1, B2, B4).

Renders an editable table of experimental factors. Factors are stored as a
list of dicts in st.session_state["factors"]. Internally each factor also has
a stable UUID ("id") used as widget key prefixes so that add/delete operations
do not confuse Streamlit's widget reconciliation.

Factor dict schema
------------------
Continuous:  {"id": str, "name": str, "type": "Continuous", "low": float, "high": float}
Categorical: {"id": str, "name": str, "type": "Categorical", "levels": list[str]}
"""

from __future__ import annotations

import uuid

import streamlit as st


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_factor_ids() -> None:
    if "factor_ids" not in st.session_state:
        st.session_state["factor_ids"] = []


def _add_factor(ftype: str) -> None:
    """Append a new factor with a stable short UUID and sensible defaults."""
    fid = str(uuid.uuid4())[:8]
    n = len(st.session_state["factor_ids"]) + 1
    st.session_state["factor_ids"].append(fid)
    # Pre-populate widget keys so they show defaults on the next render.
    st.session_state[f"fname_{fid}"] = f"X{n}"
    st.session_state[f"ftype_{fid}"] = ftype
    if ftype == "Continuous":
        st.session_state[f"flow_{fid}"] = 0.0
        st.session_state[f"fhigh_{fid}"] = 1.0
    else:
        st.session_state[f"flevels_{fid}"] = "low, high"


def _delete_factor(fid: str) -> None:
    """Remove a factor and clean up all its widget keys."""
    st.session_state["factor_ids"] = [
        x for x in st.session_state["factor_ids"] if x != fid
    ]
    for prefix in ("fname", "ftype", "flow", "fhigh", "flevels", "fdel"):
        st.session_state.pop(f"{prefix}_{fid}", None)


def _sync_factors() -> None:
    """
    Rebuild st.session_state["factors"] from the current widget key values.

    Called after all factor widgets are rendered so the list reflects whatever
    the user has typed or selected this frame.
    """
    factors: list[dict] = []
    for fid in st.session_state.get("factor_ids", []):
        ftype = st.session_state.get(f"ftype_{fid}", "Continuous")
        fname = st.session_state.get(f"fname_{fid}", "")
        if ftype == "Continuous":
            factors.append(
                {
                    "id": fid,
                    "name": fname,
                    "type": "Continuous",
                    "low": float(st.session_state.get(f"flow_{fid}", 0.0)),
                    "high": float(st.session_state.get(f"fhigh_{fid}", 1.0)),
                }
            )
        else:
            raw = st.session_state.get(f"flevels_{fid}", "")
            levels = [lv.strip() for lv in raw.split(",") if lv.strip()]
            factors.append(
                {
                    "id": fid,
                    "name": fname,
                    "type": "Categorical",
                    "levels": levels,
                }
            )
    st.session_state["factors"] = factors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clear_all_factors() -> None:
    """Remove every factor and reset factor_ids. Called by the Clear button."""
    for fid in list(st.session_state.get("factor_ids", [])):
        _delete_factor(fid)
    st.session_state["factor_ids"] = []
    st.session_state["factors"] = []


def render_factor_table() -> None:
    """Render the interactive factor entry table (B1 + B2)."""
    _ensure_factor_ids()
    factor_ids = st.session_state["factor_ids"]

    if not factor_ids:
        st.info("No factors defined yet. Use the buttons below to add your first factor.")
    else:
        # Header row — column widths match the data rows below.
        hcols = st.columns([2.5, 1.5, 2.2, 2.2, 0.6])
        hcols[0].markdown("**Name**")
        hcols[1].markdown("**Type**")
        hcols[2].markdown("**Low / Levels**")
        hcols[3].markdown("**High**")
        hcols[4].markdown("Del")

        to_delete: str | None = None

        for fid in factor_ids:
            cols = st.columns([2.5, 1.5, 2.2, 2.2, 0.6])

            cols[0].text_input(
                "Name",
                key=f"fname_{fid}",
                label_visibility="collapsed",
            )
            cols[1].selectbox(
                "Type",
                options=["Continuous", "Categorical"],
                key=f"ftype_{fid}",
                label_visibility="collapsed",
            )

            ftype = st.session_state.get(f"ftype_{fid}", "Continuous")
            if ftype == "Continuous":
                cols[2].number_input(
                    "Low",
                    key=f"flow_{fid}",
                    label_visibility="collapsed",
                    step=1.0,
                    format="%.4g",
                )
                cols[3].number_input(
                    "High",
                    key=f"fhigh_{fid}",
                    label_visibility="collapsed",
                    step=1.0,
                    format="%.4g",
                )
            else:
                cols[2].text_input(
                    "Levels (comma-separated)",
                    key=f"flevels_{fid}",
                    label_visibility="collapsed",
                    placeholder="low, med, high",
                )
                cols[3].markdown("&nbsp;", unsafe_allow_html=True)  # spacer

            if cols[4].button("✕", key=f"fdel_{fid}", help="Remove this factor"):
                to_delete = fid

        # Sync factors list from widget values (reads current frame state).
        _sync_factors()

        if to_delete is not None:
            _delete_factor(to_delete)
            st.rerun()

    # Always render after the table (even when empty, so sync covers 0-factor case).
    if factor_ids:
        _sync_factors()

    # Add-factor buttons
    st.markdown("")
    btn_cols = st.columns([1.5, 1.8, 4])
    if btn_cols[0].button("+ Continuous", use_container_width=True):
        _add_factor("Continuous")
        st.rerun()
    if btn_cols[1].button("+ Categorical", use_container_width=True):
        _add_factor("Categorical")
        st.rerun()
