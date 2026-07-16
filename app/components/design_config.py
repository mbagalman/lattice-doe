"""
Shared design-options builder (UX-70).

Reconstructs :class:`~lattice_doe.config.DesignOptions` from session state.
Used by the Run page (to launch the search) and by the Analysis page (as a
fallback when the run's own preserved options are unavailable) — one builder,
so the pages can never drift. The Run/Analysis pages previously carried two
diverging copies, and the Analysis copy silently dropped the split-plot
settings: η sensitivity then fell back to the ``sp_only`` df method with no
HTC factors, overstating power for whole-plot effects.

The Run page also preserves the exact options object of the completed run in
``session_state["result_design_opts"]``; Analysis prefers that over any
reconstruction, so later widget edits cannot silently change what a fixed
design is analyzed with.

lattice_doe is imported lazily: the pages import this module unconditionally
and must still render their "not installed" notice when the package is
absent.
"""

from typing import Any


def build_design_opts_from_state(ss: Any) -> "Any":
    """Build DesignOptions from session state (all sections, incl. split-plot)."""
    from lattice_doe._request_builder import build_design_opts

    _do_d: dict = dict(
        criterion=ss["criterion"],
        starts=int(ss["starts"]),
        random_state=int(ss["random_state"]),
        auto_candidate=bool(ss["auto_candidate"]),
        n_blocks=int(ss.get("n_blocks", 0)),
        block_factor_name=ss.get("block_factor_name", "Block"),
        preallocate_categorical=bool(ss.get("preallocate_categorical", False)),
        alloc_min_per_cell=int(ss.get("alloc_min_per_cell", 1)),
        alloc_max_per_cell=int(ss.get("alloc_max_per_cell", 0)),
        constraint_expr=ss.get("constraint_expr", "").strip() or None,
    )
    if not ss["auto_candidate"]:
        _do_d["candidate_points"] = int(ss["candidate_points"])
    # Split-plot options (UX-70: dropping these silently changes the df
    # method of every downstream split-plot power computation).
    if ss.get("split_plot_enabled", False):
        htc_names = ss.get("sp_htc_factors") or []
        if htc_names:
            _do_d["split_plot"] = dict(
                htc_factors=list(htc_names),
                n_whole_plots=int(ss.get("sp_n_whole_plots", 4)),
                eta=float(ss.get("sp_eta", 1.0)),
                subplots_per_wp=int(ss.get("sp_subplots_per_wp", 0)),
                df_method=str(ss.get("sp_df_method", "auto")),
            )
    return build_design_opts(_do_d)
