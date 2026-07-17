# License: MIT
"""Tests for the Streamlit app's shared session-state management (``app/state.py``).

Regression coverage for the multipage state-persistence bug: values entered on
one page were lost when the user navigated to another page and back, because
Streamlit garbage-collects a widget's stored value once the widget is no longer
rendered. ``init_state()`` now re-anchors widget-backed keys on every page load
so they survive navigation.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# The Streamlit app ships inside the package (``lattice_doe/app``) and its
# modules import each other by bare name (``from state import ...``), relying
# on Streamlit putting the app dir on sys.path at runtime. Resolve the dir
# through the installed package and replicate that for the test process.
import lattice_doe as _lattice_doe

_APP_DIR = Path(_lattice_doe.__file__).resolve().parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


class TestPersistWidgetState:
    """Unit tests for the key-classification logic of ``_persist_widget_state``."""

    def _run_persist(self, seed: dict) -> set[str]:
        """Run ``_persist_widget_state`` against a recording session_state.

        Returns the set of keys that were written back (re-anchored). A fake
        ``streamlit`` module is installed so ``state`` can be imported without a
        running Streamlit server.
        """

        class RecordingState(dict):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.reassigned: list[str] = []

            def __setitem__(self, key, value):
                self.reassigned.append(key)
                super().__setitem__(key, value)

        fake_st = types.ModuleType("streamlit")
        fake_st.session_state = RecordingState(seed)
        saved = sys.modules.get("streamlit")
        sys.modules["streamlit"] = fake_st
        # Ensure ``state`` binds to the fake streamlit module.
        sys.modules.pop("state", None)
        try:
            import state  # noqa: PLC0415

            static_keys = {k for k in seed if not k.startswith(
                ("fname_", "ftype_", "flow_", "fhigh_", "flevels_",
                 "scen_a_", "scen_b_", "fdel_", "mr_name_", "mr_sigma_",
                 "mr_remove_"))}
            fake_st.session_state.reassigned.clear()
            state._persist_widget_state(static_keys)
            return set(fake_st.session_state.reassigned)
        finally:
            if saved is not None:
                sys.modules["streamlit"] = saved
            else:
                sys.modules.pop("streamlit", None)
            sys.modules.pop("state", None)

    def test_static_and_dynamic_input_keys_are_reanchored(self):
        touched = self._run_persist({
            "alpha": 0.10,                 # static config
            "criterion": "D",
            "fname_ab12": "Temperature",   # factor-row input widgets
            "ftype_ab12": "Continuous",
            "flow_ab12": 20.0,
            "scen_a_Temperature": 20.0,    # scenario-builder input widgets
            "scen_b_Temperature": 80.0,
        })
        for key in ("alpha", "criterion", "fname_ab12", "ftype_ab12",
                    "flow_ab12", "scen_a_Temperature", "scen_b_Temperature"):
            assert key in touched, f"{key} must be re-anchored to survive navigation"

    def test_button_keys_are_not_reanchored(self):
        # Streamlit forbids setting a button's value via session_state, so the
        # per-row delete/remove button keys must be left untouched.
        touched = self._run_persist({
            "alpha": 0.05,
            "fdel_ab12": False,
            "mr_remove_0": False,
        })
        assert "fdel_ab12" not in touched
        assert "mr_remove_0" not in touched

    def test_per_response_value_managed_keys_are_not_reanchored(self):
        # Per-response multi-response widgets persist via the ``mr_responses``
        # list using each widget's ``value=`` argument; re-anchoring them would
        # clash with that default. Only the per-row keys are excluded here.
        touched = self._run_persist({
            "mr_enabled": True,     # static key beginning "mr_" -> MUST persist
            "mr_name_0": "Yield",   # per-response value=-managed -> MUST skip
            "mr_sigma_0": 1.0,
        })
        assert "mr_enabled" in touched
        assert "mr_name_0" not in touched
        assert "mr_sigma_0" not in touched


# AppTest needs a real Streamlit install ([app] extra). The unit tests above
# use a fake streamlit module, so only this end-to-end class is skipped on a
# core install — a module-level importorskip would needlessly skip those too.
_HAS_STREAMLIT_TESTING = (
    importlib.util.find_spec("streamlit") is not None
    and importlib.util.find_spec("streamlit.testing.v1") is not None
)


@pytest.mark.skipif(
    not _HAS_STREAMLIT_TESTING, reason="app extra (streamlit) not installed"
)
class TestPageNavigationPersistence:
    """End-to-end: a value entered on one page survives navigating away and back."""

    def _fresh_app(self):
        from streamlit.testing.v1 import AppTest

        # Import through the same app-dir path the pages use.
        app_main = str(_APP_DIR / "app.py")
        return AppTest.from_file(app_main, default_timeout=60)

    def test_alpha_survives_round_trip_navigation(self):
        at = self._fresh_app()
        at.run()

        # Go to Power Config and change alpha away from its 0.05 default.
        at.switch_page("pages/2_Power_Config.py")
        at.run()
        alpha = next(w for w in at.number_input if w.key == "alpha")
        assert alpha.value == pytest.approx(0.05)
        alpha.set_value(0.10).run()
        assert at.session_state["alpha"] == pytest.approx(0.10)

        # Navigate to the Factors page (which does not render the alpha widget)
        # and back to Power Config.
        at.switch_page("pages/1_Factors.py")
        at.run()
        at.switch_page("pages/2_Power_Config.py")
        at.run()

        assert not at.exception
        # The entered value must be remembered, not reset to the default.
        assert at.session_state["alpha"] == pytest.approx(0.10)
        alpha_after = next(w for w in at.number_input if w.key == "alpha")
        assert alpha_after.value == pytest.approx(0.10)

    def test_sigma_joint_text_does_not_crash(self):
        """UX-30 regression: entering σ_joint text executed
        ``len(mr_responses)`` before the local variable was assigned further
        down the page, raising an uncaught NameError."""
        at = self._fresh_app()
        at.run()
        at.switch_page("pages/2_Power_Config.py")
        at.session_state["mr_enabled"] = True
        at.run()
        assert not at.exception

        sj = next(w for w in at.text_area if w.key == "mr_sigma_joint")
        sj.set_value("1.0 0.3\n0.3 1.0").run()
        assert not at.exception  # NameError previously surfaced here

        # And with responses present, the k×k shape validation still runs.
        at.session_state["mr_responses"] = [
            {"name": "y1"}, {"name": "y2"},
        ]
        at.run()
        assert not at.exception

    def test_stateful_formula_preview_points_at_the_run(self):
        """UX-46/UX-48: the Page 2 preview has no design options, so it cannot
        know the candidate set that will establish a stateful formula's coding.
        It must say so and point at the Run page — which builds L correctly —
        rather than crashing or naming a Python-only argument the app user
        cannot pass."""
        at = self._fresh_app()
        at.run()

        at.session_state["formula"] = "~ 1 + bs(x, df=3)"
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.switch_page("pages/2_Power_Config.py")
        at.session_state["power_mode"] = "contrast"
        at.session_state["contrast_input_mode"] = "scenario"
        at.session_state["scen_a_x"] = 0.2
        at.session_state["scen_b_x"] = 0.8
        at.session_state["sesoi"] = 0.5
        at.run()

        assert not at.exception, "the page must not crash on a stateful formula"

        errors = " ".join(e.value for e in at.error)
        assert "Run" in errors, (
            f"expected the preview to defer to the Run page; got: {errors!r}"
        )
        # It must not tell an app user to pass a Python-only argument, nor
        # hand them a snippet whose seed/size may not match the real run.
        assert "Pass coding_data=" not in errors
        assert "seed=42" not in errors

    def test_spline_scenario_defaults_are_interior(self):
        """UX-54: sampled candidates never reach the declared bounds, and a
        spline cannot extrapolate past its outermost knots — so defaulting
        Scenario A/B to the bounds made the supported bs() workflow fail out
        of the box. Data-dependent codings must default strictly inside."""
        at = self._fresh_app()
        at.run()
        at.session_state["formula"] = "~ 1 + bs(x, df=3)"
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.switch_page("pages/2_Power_Config.py")
        at.session_state["power_mode"] = "contrast"
        at.session_state["contrast_input_mode"] = "scenario"
        at.run()
        assert not at.exception
        a = at.session_state["scen_a_x"]
        b = at.session_state["scen_b_x"]
        assert 0.0 < a < b < 1.0, f"defaults must be interior; got A={a}, B={b}"

    def test_plain_formula_scenario_defaults_stay_at_bounds(self):
        """The interior inset is scoped to data-dependent codings — ordinary
        formulas keep the full-range defaults users are used to."""
        at = self._fresh_app()
        at.run()
        at.session_state["formula"] = "~ 1 + x"
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.switch_page("pages/2_Power_Config.py")
        at.session_state["power_mode"] = "contrast"
        at.session_state["contrast_input_mode"] = "scenario"
        at.run()
        assert not at.exception
        assert at.session_state["scen_a_x"] == 0.0
        assert at.session_state["scen_b_x"] == 1.0

    def test_scenario_defaults_migrate_when_formula_becomes_stateful(self):
        """UX-61: the interior inset must also apply to an EXISTING session —
        a user who starts with a linear formula (defaults at the bounds) and
        then switches to bs() previously kept the boundary defaults and hit
        the outermost-knots failure anyway. Automatic values migrate; values
        the user typed do not."""
        at = self._fresh_app()
        at.run()
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.session_state["formula"] = "~ 1 + x"
        at.switch_page("pages/2_Power_Config.py")
        at.session_state["power_mode"] = "contrast"
        at.session_state["contrast_input_mode"] = "scenario"
        at.run()
        assert at.session_state["scen_a_x"] == 0.0  # bound defaults first

        # Switch the same session to a learned spline: autos must migrate.
        at.session_state["formula"] = "~ 1 + bs(x, df=3)"
        at.run()
        assert not at.exception
        a, b = at.session_state["scen_a_x"], at.session_state["scen_b_x"]
        assert 0.0 < a < b < 1.0, f"stale bound defaults kept: A={a}, B={b}"

        # A user-entered value must survive the next migration untouched.
        at.session_state["scen_a_x"] = 0.42
        at.session_state["formula"] = "~ 1 + x"
        at.run()
        assert at.session_state["scen_a_x"] == 0.42
        assert at.session_state["scen_b_x"] == 1.0  # auto value migrated back

    def test_analysis_page_binds_selected_compound_response(self):
        """UX-65: for a compound multi-response run the Analysis page must
        bind sensitivity to the SELECTED response's formula, power config and
        per-response matrix — the global reconstruction silently analyzes a
        different model. Exact check: nominal power for y2 equals y2's power
        in the run report (same L, sigma and basis)."""
        import numpy as np

        from lattice_doe import DesignOptions, find_multiresponse_design
        from lattice_doe.config import (
            MultiResponseOptions, PowerContrastConfig, ResponseSpec,
        )

        opts = DesignOptions(candidate_points=100, random_state=4, starts=1)
        cfg1 = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=18,
        )
        cfg2 = PowerContrastConfig(
            L=np.array([[0.0, 1.0, 0.0, 0.0, -1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=18,
        )
        multi = MultiResponseOptions(responses=[
            ResponseSpec(name="y1", power_cfg=cfg1),
            ResponseSpec(name="y2", power_cfg=cfg2,
                         formula="~ 1 + bs(x, df=4)"),
        ])
        import warnings as W
        with W.catch_warnings():
            W.simplefilter("ignore")
            result = find_multiresponse_design(
                "~ 1 + x", {"x": (0.0, 1.0)}, multi, opts,
            )
        y2_reported = next(
            r["power"] for r in result["report"]["responses"]
            if r["name"] == "y2"
        )

        at = self._fresh_app()
        at.run()
        at.session_state["formula"] = "~ 1 + x"
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.session_state["random_state"] = 4
        at.session_state["candidate_points"] = 100
        at.session_state["auto_candidate"] = False
        at.session_state["starts"] = 1
        at.session_state["result"] = result
        at.session_state["mr_responses"] = [
            {"name": "y1", "power_mode": "contrast",
             "L_text": "0 1", "delta_text": "0.5", "sigma": 1.0},
            {"name": "y2", "power_mode": "contrast",
             "L_text": "0 1 0 0 -1", "delta_text": "0.5", "sigma": 1.0,
             "formula": "~ 1 + bs(x, df=4)"},
        ]
        at.switch_page("pages/4_Analysis.py")
        at.run()
        assert not at.exception

        sel = next(w for w in at.selectbox if w.key == "analysis_response")
        sel.set_value("y2").run()
        assert not at.exception

        btn = next(b for b in at.button if b.key == "btn_sensitivity")
        btn.click().run()
        assert not at.exception

        sens = at.session_state["_sensitivity_result"]
        assert np.isclose(sens["nominal_power"], y2_reported, atol=1e-9), (
            f"sensitivity analyzed a different model: nominal="
            f"{sens['nominal_power']} vs y2 reported={y2_reported}"
        )

    def test_switching_response_clears_stale_analysis_results(self):
        """UX-68: cached sensitivity/MDE/comparison/eta results belong to one
        (run, response, mode) context. Switching the selected response must
        clear them — otherwise the page shows results computed for the
        PREVIOUS response (and mode mixes can crash the renderers)."""
        import numpy as np

        from lattice_doe import DesignOptions, find_multiresponse_design
        from lattice_doe.config import (
            MultiResponseOptions, PowerContrastConfig, ResponseSpec,
        )

        opts = DesignOptions(candidate_points=100, random_state=4, starts=1)
        cfg1 = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=18,
        )
        cfg2 = PowerContrastConfig(
            L=np.array([[0.0, 1.0, 0.0, 0.0, -1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=2.0, max_n=18,
        )
        multi = MultiResponseOptions(responses=[
            ResponseSpec(name="y1", power_cfg=cfg1),
            ResponseSpec(name="y2", power_cfg=cfg2,
                         formula="~ 1 + bs(x, df=4)"),
        ])
        import warnings as W
        with W.catch_warnings():
            W.simplefilter("ignore")
            result = find_multiresponse_design(
                "~ 1 + x", {"x": (0.0, 1.0)}, multi, opts,
            )

        at = self._fresh_app()
        at.run()
        at.session_state["formula"] = "~ 1 + x"
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.session_state["result"] = result
        at.session_state["mr_responses"] = [
            {"name": "y1", "power_mode": "contrast",
             "L_text": "0 1", "delta_text": "0.5", "sigma": 1.0},
            {"name": "y2", "power_mode": "contrast",
             "L_text": "0 1 0 0 -1", "delta_text": "0.5", "sigma": 2.0,
             "formula": "~ 1 + bs(x, df=4)"},
        ]
        at.switch_page("pages/4_Analysis.py")
        at.run()
        assert not at.exception

        # Run sensitivity for y1 and confirm it cached.
        btn = next(b for b in at.button if b.key == "btn_sensitivity")
        btn.click().run()
        assert not at.exception
        assert "_sensitivity_result" in at.session_state
        y1_sigma_min = at.session_state["sens_sigma_min"]

        # Switch to y2: the y1 results and response-derived widget defaults
        # must be gone, and nothing stale may render.
        sel = next(w for w in at.selectbox if w.key == "analysis_response")
        sel.set_value("y2").run()
        assert not at.exception
        assert "_sensitivity_result" not in at.session_state
        assert "_mde_result" not in at.session_state
        # sweep range re-defaults from y2's sigma (2.0), not y1's (1.0)
        assert at.session_state["sens_sigma_min"] != y1_sigma_min

        # A NEW run must also invalidate, even with the same response chosen.
        btn = next(b for b in at.button if b.key == "btn_sensitivity")
        btn.click().run()
        assert "_sensitivity_result" in at.session_state
        result2 = dict(result)
        result2["report"] = dict(result["report"])
        result2["report"]["elapsed_sec"] = (
            float(result["report"].get("elapsed_sec", 0.0)) + 999.0
        )
        at.session_state["result"] = result2
        at.run()
        assert not at.exception
        assert "_sensitivity_result" not in at.session_state

    def test_eta_sweep_uses_the_runs_split_plot_options(self):
        """UX-70: the Analysis page's options reconstruction dropped the
        split-plot settings, so eta sensitivity silently fell back to the
        sp_only df method with no HTC factors and overstated power. The page
        must analyze with the EXACT options the run used (preserved by the
        Run page), matching a direct computation to 1e-9."""
        import warnings as W

        import numpy as np

        from lattice_doe import DesignOptions, find_optimal_design
        from lattice_doe.analysis import power_sensitivity
        from lattice_doe.config import PowerContrastConfig, SplitPlotOptions

        factors = {"x": (0.0, 1.0), "w": ["a", "b"]}
        formula = "~ 1 + C(w) + x"
        run_opts = DesignOptions(
            candidate_points=60, random_state=2, starts=1,
            split_plot=SplitPlotOptions(
                htc_factors=["w"], n_whole_plots=4,
                df_method="conservative",
            ),
        )
        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0, 0.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=24,
        )
        with W.catch_warnings():
            W.simplefilter("ignore")
            result = find_optimal_design(formula, factors, cfg, run_opts)

        correct = power_sensitivity(
            formula=formula, factors=factors, power_cfg=cfg,
            design_df=result["design_df"], design_opts=run_opts,
            eta_range=(0.5, 2.0), eta_points=20,
            model_matrix=result["model_matrix"],
        )["eta_sweep"]

        at = self._fresh_app()
        at.run()
        at.session_state["formula"] = formula
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
            {"id": "f2", "name": "w", "type": "Categorical",
             "levels": ["a", "b"]},
        ]
        at.session_state["power_mode"] = "contrast"
        at.session_state["contrast_input_mode"] = "matrix"
        at.session_state["L_text"] = "0 1 0"
        at.session_state["delta_text"] = "0.5"
        at.session_state["result"] = result
        # what the Run page preserves at run time (UX-70)
        at.session_state["result_design_opts"] = run_opts
        at.switch_page("pages/4_Analysis.py")
        at.run()
        assert not at.exception

        # Match the direct computation's sweep grid via the widgets. The
        # points slider keeps its default of 20 (AppTest slider set_value
        # does not propagate reliably), so the reference above uses 20 too.
        next(w for w in at.number_input if w.key == "sp_eta_min").set_value(0.5)
        next(w for w in at.number_input if w.key == "sp_eta_max").set_value(2.0)
        at.run()
        assert not at.exception

        btn = next(b for b in at.button if b.key == "btn_eta_sweep")
        btn.click().run()
        assert not at.exception
        page_sweep = at.session_state["_sp_eta_result"]["eta_sweep"]
        assert np.allclose(
            np.asarray(page_sweep["power"]),
            np.asarray(correct["power"]), atol=1e-9,
        ), (
            f"page eta sweep {list(page_sweep['power'])} != correct "
            f"{list(correct['power'])} — split-plot options were dropped"
        )

    def test_design_opts_fallback_builder_includes_split_plot(self):
        """UX-70 (fallback path): reconstructing options from session state —
        for sessions predating the preserved-options contract — must include
        the split-plot settings the old Analysis-page copy dropped."""
        import sys

        sys.path.insert(0, str(_APP_DIR)) if str(_APP_DIR) not in sys.path else None
        pytest.importorskip("streamlit")
        from components.design_config import build_design_opts_from_state

        ss = {
            "criterion": "I", "starts": 2, "random_state": 1,
            "auto_candidate": False, "candidate_points": 80,
            "split_plot_enabled": True,
            "sp_htc_factors": ["w"], "sp_n_whole_plots": 4,
            "sp_eta": 1.5, "sp_subplots_per_wp": 0,
            "sp_df_method": "conservative",
        }
        opts = build_design_opts_from_state(ss)
        assert opts.split_plot is not None
        assert opts.split_plot.htc_factors == ["w"]
        assert opts.split_plot.df_method == "conservative"
        assert opts.split_plot.eta == 1.5

    def test_target_power_display_contract(self):
        """UX-77: the shared bounds permit four-decimal targets, so the
        shared display format must render them faithfully — a two-decimal
        format shows the valid 0.9999 as an impossible-looking '1.00'.
        Every target widget must render through the shared constant."""
        import sys

        sys.path.insert(0, str(_APP_DIR)) if str(_APP_DIR) not in sys.path else None
        pytest.importorskip("streamlit")
        from components.power_params import (
            POWER_TARGET_FORMAT, POWER_TARGET_MAX, POWER_TARGET_MIN,
        )

        assert POWER_TARGET_FORMAT % POWER_TARGET_MAX == "0.9999"
        assert float(POWER_TARGET_FORMAT % POWER_TARGET_MAX) < 1.0
        assert POWER_TARGET_FORMAT % POWER_TARGET_MIN == "0.0100"
        # every target widget renders through the shared format (a site
        # that stops using it would also leave an unused import for lint)
        for rel in ("components/power_params.py",
                    "pages/2_Power_Config.py",
                    "pages/4_Analysis.py"):
            src = (_APP_DIR / rel).read_text(encoding="utf-8")
            assert "format=POWER_TARGET_FORMAT" in src, (
                f"{rel} does not render the target through the shared format"
            )

    def test_analysis_target_power_follows_selected_response(self):
        """UX-71: each response can carry its own target power; MDE (and the
        target reference lines) must use the bound response's power_cfg.power,
        not the global widget — and switching responses must re-default the
        MDE widget to the newly selected response's target."""
        import numpy as np

        from lattice_doe import DesignOptions, find_multiresponse_design
        from lattice_doe.config import (
            MultiResponseOptions, PowerContrastConfig, ResponseSpec,
        )

        opts = DesignOptions(candidate_points=100, random_state=4, starts=1)
        cfg1 = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.80, sigma=1.0, max_n=18,
        )
        cfg2 = PowerContrastConfig(
            L=np.array([[0.0, 1.0, 0.0, 0.0, -1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.90, sigma=1.0, max_n=18,
        )
        multi = MultiResponseOptions(responses=[
            ResponseSpec(name="y1", power_cfg=cfg1),
            ResponseSpec(name="y2", power_cfg=cfg2,
                         formula="~ 1 + bs(x, df=4)"),
        ])
        import warnings as W
        with W.catch_warnings():
            W.simplefilter("ignore")
            result = find_multiresponse_design(
                "~ 1 + x", {"x": (0.0, 1.0)}, multi, opts,
            )

        at = self._fresh_app()
        at.run()
        at.session_state["formula"] = "~ 1 + x"
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.session_state["result"] = result
        at.session_state["power_target"] = 0.70   # global differs from BOTH
        at.session_state["mr_responses"] = [
            {"name": "y1", "power_mode": "contrast", "power": 0.80,
             "L_text": "0 1", "delta_text": "0.5", "sigma": 1.0},
            {"name": "y2", "power_mode": "contrast", "power": 0.90,
             "L_text": "0 1 0 0 -1", "delta_text": "0.5", "sigma": 1.0,
             "formula": "~ 1 + bs(x, df=4)"},
        ]
        at.switch_page("pages/4_Analysis.py")
        at.run()
        assert not at.exception

        mde_w = next(w for w in at.number_input if w.key == "mde_power_target")
        assert mde_w.value == pytest.approx(0.80), "y1's own target, not 0.70"

        sel = next(w for w in at.selectbox if w.key == "analysis_response")
        sel.set_value("y2").run()
        assert not at.exception
        mde_w = next(w for w in at.number_input if w.key == "mde_power_target")
        assert mde_w.value == pytest.approx(0.90), (
            "switching responses must re-default MDE to y2's target"
        )

    def test_full_target_range_is_usable_for_mde(self):
        """UX-72: per-response configuration accepts targets in
        [0.01, 0.9999], and the selected response's target seeds the MDE
        widget — so the MDE widget must share those bounds. With the old
        0.50–0.999 bounds a valid target of 0.40 raised
        StreamlitValueBelowMinError and took down the whole Analysis page."""
        import numpy as np

        from lattice_doe import DesignOptions, find_multiresponse_design
        from lattice_doe.config import (
            MultiResponseOptions, PowerContrastConfig, ResponseSpec,
        )

        opts = DesignOptions(candidate_points=100, random_state=4, starts=1)
        cfg_hi = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.9999, sigma=1.0, max_n=30,
        )
        cfg_low = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.40, sigma=1.0, max_n=30,
        )
        multi = MultiResponseOptions(responses=[
            ResponseSpec(name="y1", power_cfg=cfg_hi),
            ResponseSpec(name="y2", power_cfg=cfg_low),
        ])
        import warnings as W
        with W.catch_warnings():
            W.simplefilter("ignore")
            result = find_multiresponse_design(
                "~ 1 + x", {"x": (0.0, 1.0)}, multi, opts,
            )

        at = self._fresh_app()
        at.run()
        at.session_state["formula"] = "~ 1 + x"
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.session_state["result"] = result
        at.session_state["mr_responses"] = [
            {"name": "y1", "power_mode": "contrast", "power": 0.9999,
             "L_text": "0 1", "delta_text": "0.5", "sigma": 1.0},
            {"name": "y2", "power_mode": "contrast", "power": 0.40,
             "L_text": "0 1", "delta_text": "0.5", "sigma": 1.0},
        ]
        at.switch_page("pages/4_Analysis.py")
        at.run()
        assert not at.exception, "target 0.9999 must render"
        mde_w = next(w for w in at.number_input if w.key == "mde_power_target")
        assert mde_w.value == pytest.approx(0.9999, abs=1e-12)

        sel = next(w for w in at.selectbox if w.key == "analysis_response")
        sel.set_value("y2").run()
        assert not at.exception, "target 0.40 must render"
        mde_w = next(w for w in at.number_input if w.key == "mde_power_target")
        assert mde_w.value == pytest.approx(0.40, abs=1e-12)

    def test_replacement_run_with_identical_report_invalidates_caches(self):
        """UX-74: (n, achieved_power, elapsed_sec) are report VALUES, not a
        run identity — a re-run of the same configuration repeats them, so
        cached analyses survived a replacement run. The guard must key on
        the run token the Run page mints per completed run."""
        import copy

        import numpy as np

        from lattice_doe import DesignOptions, find_multiresponse_design
        from lattice_doe.config import (
            MultiResponseOptions, PowerContrastConfig, ResponseSpec,
        )

        opts = DesignOptions(candidate_points=100, random_state=4, starts=1)
        cfg = PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.5]),
            alpha=0.05, power=0.8, sigma=1.0, max_n=18,
        )
        multi = MultiResponseOptions(responses=[
            ResponseSpec(name="y1", power_cfg=cfg),
            ResponseSpec(name="y2", power_cfg=cfg),
        ])
        import warnings as W
        with W.catch_warnings():
            W.simplefilter("ignore")
            result = find_multiresponse_design(
                "~ 1 + x", {"x": (0.0, 1.0)}, multi, opts,
            )

        at = self._fresh_app()
        at.run()
        at.session_state["formula"] = "~ 1 + x"
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.session_state["result"] = result
        at.session_state["result_run_id"] = 1     # minted by the Run page
        at.session_state["mr_responses"] = [
            {"name": "y1", "power_mode": "contrast",
             "L_text": "0 1", "delta_text": "0.5", "sigma": 1.0},
            {"name": "y2", "power_mode": "contrast",
             "L_text": "0 1", "delta_text": "0.5", "sigma": 1.0},
        ]
        at.switch_page("pages/4_Analysis.py")
        at.run()
        assert not at.exception

        btn = next(b for b in at.button if b.key == "btn_sensitivity")
        btn.click().run()
        assert not at.exception
        assert "_sensitivity_result" in at.session_state

        # A replacement run whose report values are ALL equal (deep copy):
        # only the freshly minted token distinguishes it.
        at.session_state["result"] = copy.deepcopy(result)
        at.session_state["result_run_id"] = 2
        at.run()
        assert not at.exception
        assert "_sensitivity_result" not in at.session_state, (
            "equal report values masked the replacement run"
        )

    def test_generate_design_mints_run_token(self):
        """UX-74 (minting site): each completed run on the Run page must
        carry a fresh token — including a re-run of the SAME configuration,
        whose report values repeat — and clearing the result must retire
        it. Exercises the real Generate/Clear buttons end to end."""
        at = self._fresh_app()
        at.run()
        at.session_state["formula"] = "~ 1 + x"
        at.session_state["factors"] = [
            {"id": "f1", "name": "x", "type": "Continuous",
             "low": 0.0, "high": 1.0},
        ]
        at.session_state["power_mode"] = "contrast"
        at.session_state["contrast_input_mode"] = "matrix"
        at.session_state["L_text"] = "0 1"
        at.session_state["delta_text"] = "0.5"
        at.session_state["sigma"] = 1.0
        at.session_state["power_target"] = 0.80
        at.session_state["max_n"] = 18
        at.session_state["auto_candidate"] = False
        at.session_state["candidate_points"] = 100
        at.session_state["starts"] = 1
        at.session_state["random_state"] = 4
        at.switch_page("pages/3_Run_Results.py")
        at.run()
        assert not at.exception
        assert "result_run_id" not in at.session_state

        gen = next(b for b in at.button if "Generate design" in (b.label or ""))
        gen.click().run()
        assert not at.exception
        assert not at.session_state["run_error"], at.session_state["run_error"]
        assert at.session_state["result"] is not None
        assert at.session_state["result_run_id"] == 1

        # Same configuration again: report values repeat, the token must not.
        gen = next(b for b in at.button if "Generate design" in (b.label or ""))
        gen.click().run()
        assert not at.exception
        assert at.session_state["result_run_id"] == 2

        clear = next(b for b in at.button if b.label == "Clear result")
        clear.click().run()
        assert not at.exception
        assert at.session_state["result"] is None
        assert at.session_state["result_run_id"] is None
