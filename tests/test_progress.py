# tests/test_progress.py
"""Unit tests for the unified progress reporter (UX-3)."""
from __future__ import annotations

import numpy as np
import pytest

from lattice_doe.progress import (
    Phase,
    ProgressEvent,
    ProgressReporter,
    SearchCancelled,
    _coerce_reporter,
)


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestProgressReporter:
    def test_event_fields_and_sequence(self):
        seen = []
        r = ProgressReporter(seen.append, min_interval=0.0)
        r.emit(Phase.VALIDATING, message="hi", target_power=0.8)
        r.emit(Phase.OPTIMIZING, trial_n=12, current_power=0.5, iteration=3)
        assert [e.phase for e in seen] == ["validating", "optimizing"]
        assert seen[0].seq == 1 and seen[1].seq == 2
        assert seen[1].trial_n == 12 and seen[1].current_power == 0.5
        assert isinstance(seen[0], ProgressEvent)

    def test_throttle_drops_unforced_but_keeps_forced(self):
        clk = _FakeClock()
        seen = []
        r = ProgressReporter(seen.append, min_interval=1.0, clock=clk)
        r.emit(Phase.OPTIMIZING)                 # t=0, first delivered
        clk.t = 0.4
        r.emit(Phase.OPTIMIZING)                 # dropped (within interval)
        clk.t = 0.5
        r.emit(Phase.DONE, force=True)           # forced, delivered
        clk.t = 0.6
        r.emit(Phase.OPTIMIZING)                 # dropped
        clk.t = 2.0
        r.emit(Phase.OPTIMIZING)                 # delivered (interval elapsed)
        phases = [e.phase for e in seen]
        assert phases == ["optimizing", "done", "optimizing"]

    def test_last_event_always_current_even_when_throttled(self):
        clk = _FakeClock()
        seen = []
        r = ProgressReporter(seen.append, min_interval=10.0, clock=clk)
        r.emit(Phase.OPTIMIZING, trial_n=5)      # delivered
        clk.t = 0.1
        r.emit(Phase.OPTIMIZING, trial_n=6)      # throttled from callback
        assert len(seen) == 1
        assert r.last_event.trial_n == 6         # but state is current

    def test_cancellation_raises_regardless_of_throttle(self):
        state = {"cancel": False}
        r = ProgressReporter(min_interval=1e9, cancelled=lambda: state["cancel"])
        r.emit(Phase.OPTIMIZING)                 # fine
        state["cancel"] = True
        with pytest.raises(SearchCancelled):
            r.emit(Phase.OPTIMIZING)

    def test_faulty_callback_is_swallowed(self):
        def boom(_ev):
            raise RuntimeError("nope")
        r = ProgressReporter(boom, min_interval=0.0)
        # Must not raise
        ev = r.emit(Phase.OPTIMIZING)
        assert ev.phase == "optimizing"

    def test_coerce_reporter(self):
        assert _coerce_reporter(None) is None
        rep = ProgressReporter()
        assert _coerce_reporter(rep) is rep
        wrapped = _coerce_reporter(lambda e: None)
        assert isinstance(wrapped, ProgressReporter)


class TestProgressIntegration:
    _FACTORS = {"x1": (-1.0, 1.0), "x2": (-1.0, 1.0)}

    def _cfg(self, delta=1.2, max_n=60):
        from lattice_doe.config import PowerContrastConfig
        return PowerContrastConfig(
            L=np.array([[0.0, 1.0, 0.0]]), delta=np.array([delta]),
            sigma=1.0, alpha=0.05, power=0.8, max_n=max_n,
        )

    def _opts(self):
        from lattice_doe.config import DesignOptions
        return DesignOptions(random_state=0, starts=1, candidate_points=100)

    def test_single_response_phase_sequence(self):
        from lattice_doe.api import find_optimal_design
        events = []
        find_optimal_design(
            "~ 1 + x1 + x2", self._FACTORS, self._cfg(), self._opts(),
            on_progress=ProgressReporter(events.append, min_interval=0.0),
        )
        phases = [e.phase for e in events]
        assert phases[0] == "validating"
        assert "generating_candidates" in phases
        assert "optimizing" in phases
        assert phases[-1] == "done"
        opt = [e for e in events if e.phase == "optimizing" and e.current_power is not None]
        assert opt and opt[0].target_power == 0.8

    def test_bare_callable_wrapped(self):
        from lattice_doe.api import find_optimal_design
        seen = []
        find_optimal_design(
            "~ 1 + x1 + x2", self._FACTORS, self._cfg(), self._opts(),
            on_progress=lambda e: seen.append(e.phase),
        )
        assert "done" in seen

    def test_cancellation_aborts_search(self):
        from lattice_doe.api import find_optimal_design
        calls = {"n": 0}

        def cancel_after_3():
            calls["n"] += 1
            return calls["n"] > 3

        with pytest.raises(SearchCancelled):
            find_optimal_design(
                "~ 1 + x1 + x2", self._FACTORS, self._cfg(), self._opts(),
                on_progress=ProgressReporter(min_interval=0.0,
                                             cancelled=cancel_after_3),
            )

    def test_multiresponse_phase_sequence(self):
        from lattice_doe.api import find_multiresponse_design
        from lattice_doe.config import (PowerContrastConfig, ResponseSpec,
                                        MultiResponseOptions)
        events = []
        resp = [ResponseSpec(name=nm, power_cfg=PowerContrastConfig(
                    L=np.array([[0.0, 1.0, 0.0]]), delta=np.array([1.0]),
                    sigma=1.0, alpha=0.05, power=0.8, max_n=60))
                for nm in ("y1", "y2")]
        find_multiresponse_design(
            "~ 1 + x1 + x2", self._FACTORS,
            MultiResponseOptions(responses=resp, power_combination="min"),
            design_opts=self._opts(),
            on_progress=ProgressReporter(events.append, min_interval=0.0),
        )
        phases = [e.phase for e in events]
        assert phases[0] == "validating" and phases[-1] == "done"
        assert "optimizing" in phases

    def test_faulty_callback_does_not_break_search(self):
        from lattice_doe.api import find_optimal_design

        def boom(_ev):
            raise RuntimeError("nope")

        result = find_optimal_design(
            "~ 1 + x1 + x2", self._FACTORS, self._cfg(), self._opts(),
            on_progress=boom,
        )
        assert result["report"]["n"] > 0

    def test_pre_build_event_forced_through_throttle(self):
        """The pre-build 'Building design at n=…' event must reach the callback
        even under aggressive throttling — it is forced so a long build never
        hides the current trial n (P2 #5)."""
        from lattice_doe.api import find_optimal_design
        events = []
        find_optimal_design(
            "~ 1 + x1 + x2", self._FACTORS, self._cfg(), self._opts(),
            # Huge min_interval: only forced events get delivered.
            on_progress=ProgressReporter(events.append, min_interval=1e9),
        )
        build_events = [e for e in events
                        if e.phase == "optimizing" and e.trial_n is not None
                        and "Building design" in e.message]
        assert build_events, "forced pre-build events were throttled away"

    def test_verification_scan_emits_per_check_events(self):
        """P1 regression: the phase-2 verification scan performed expensive
        builds with no reporter events, so cancellation and busy signaling
        were dead throughout it. Per-check events are forced, so they must
        arrive even under aggressive throttling."""
        from lattice_doe.api import find_optimal_design
        events = []
        res = find_optimal_design(
            "~ 1 + x1 + x2", self._FACTORS, self._cfg(), self._opts(),
            # Huge min_interval: only forced events get delivered.
            on_progress=ProgressReporter(events.append, min_interval=1e9),
        )
        assert "verification" in res["report"]["search_strategy"], (
            "test config no longer triggers the phase-2 scan; pick one that does"
        )
        checks = [e for e in events
                  if e.phase == "verifying" and e.iteration > 0]
        assert checks, "no per-check verifying events reached the callback"

    # --- Multi-response compound path (MR-5) --------------------------------
    def _compound_multi(self):
        from lattice_doe.config import (PowerContrastConfig, ResponseSpec,
                                        MultiResponseOptions)
        r1 = ResponseSpec(name="y1", power_cfg=PowerContrastConfig(
            L=np.array([[0.0, 1.0, 0.0]]), delta=np.array([0.9]),
            sigma=1.0, alpha=0.05, power=0.8, max_n=60))
        # Distinct per-response formula → compound criterion path.
        r2 = ResponseSpec(name="y2", formula="~ 1 + x1",
                          power_cfg=PowerContrastConfig(
            L=np.array([[0.0, 1.0]]), delta=np.array([0.9]),
            sigma=1.0, alpha=0.05, power=0.8, max_n=60))
        return MultiResponseOptions(responses=[r1, r2],
                                    power_combination="min")

    def test_multiresponse_compound_progress(self):
        from lattice_doe.api import find_multiresponse_design
        events = []
        find_multiresponse_design(
            "~ 1 + x1 + x2", self._FACTORS, self._compound_multi(),
            design_opts=self._opts(),
            on_progress=ProgressReporter(events.append, min_interval=0.0),
        )
        phases = [e.phase for e in events]
        assert phases[0] == "validating" and phases[-1] == "done"
        assert "optimizing" in phases
        # A build event exposes the trial n before the expensive build.
        assert any(e.phase == "optimizing" and e.trial_n is not None
                   for e in events)

    def test_multiresponse_compound_cancellation(self):
        """Cancellation must be honored at the per-iteration checkpoints, not
        just the early phase events: arm the predicate only after the first
        completed evaluation is observed, then require the next emit to raise.
        """
        from lattice_doe.api import find_multiresponse_design
        armed = {"flag": False}

        def observe(ev):
            # A completed evaluation carries the achieved combined power.
            if ev.phase == "optimizing" and ev.current_power is not None:
                armed["flag"] = True

        with pytest.raises(SearchCancelled):
            find_multiresponse_design(
                "~ 1 + x1 + x2", self._FACTORS, self._compound_multi(),
                design_opts=self._opts(),
                on_progress=ProgressReporter(
                    observe, min_interval=0.0,
                    cancelled=lambda: armed["flag"]),
            )
        assert armed["flag"], (
            "search finished before any evaluation completed — the test "
            "never exercised the per-iteration cancellation checkpoint"
        )

    # --- Multi-response split-plot path -------------------------------------
    def _sp_opts(self):
        from lattice_doe.config import DesignOptions, SplitPlotOptions
        return DesignOptions(
            candidate_points=100, starts=1, max_iter=10, random_state=0,
            split_plot=SplitPlotOptions(
                htc_factors=["x1"], n_whole_plots=2, subplots_per_wp=3, eta=1.0),
        )

    def _sp_multi(self):
        from lattice_doe.config import (PowerContrastConfig, ResponseSpec,
                                        MultiResponseOptions)
        rs = [ResponseSpec(name=nm, power_cfg=PowerContrastConfig(
                  L=np.array([[0.0, 1.0, 0.0]]), delta=np.array([0.5]),
                  sigma=1.0, alpha=0.05, power=0.8, max_n=40, max_iter=10))
              for nm in ("y1", "y2")]
        return MultiResponseOptions(responses=rs, power_combination="min")

    def test_multiresponse_split_plot_progress(self):
        from lattice_doe.api import find_multiresponse_design
        events = []
        find_multiresponse_design(
            "~ 1 + x1 + x2", self._FACTORS, self._sp_multi(),
            design_opts=self._sp_opts(),
            on_progress=ProgressReporter(events.append, min_interval=0.0),
        )
        phases = [e.phase for e in events]
        assert phases[0] == "validating" and phases[-1] == "done"
        assert "optimizing" in phases
        assert any(e.phase == "optimizing" and e.trial_n is not None
                   for e in events)

    def test_multiresponse_split_plot_cancellation(self):
        """Same arm-after-first-evaluation scheme as the compound test: proves
        the split-plot loop's own checkpoints honor cancellation."""
        from lattice_doe.api import find_multiresponse_design
        armed = {"flag": False}

        def observe(ev):
            if ev.phase == "optimizing" and ev.current_power is not None:
                armed["flag"] = True

        with pytest.raises(SearchCancelled):
            find_multiresponse_design(
                "~ 1 + x1 + x2", self._FACTORS, self._sp_multi(),
                design_opts=self._sp_opts(),
                on_progress=ProgressReporter(
                    observe, min_interval=0.0,
                    cancelled=lambda: armed["flag"]),
            )
        assert armed["flag"], (
            "search finished before any evaluation completed — the test "
            "never exercised the per-iteration cancellation checkpoint"
        )
