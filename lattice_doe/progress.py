# progress.py
# License: MIT
"""Unified search-progress reporting (UX-3).

A single :class:`ProgressEvent` describes where a design search is, and
:class:`ProgressReporter` delivers those events to any interface — CLI stderr,
the notebook widget, a Streamlit ``st.status`` box, or a REST job resource —
through one throttled, cancellation-aware callback.

Design notes
------------
* **Throttling.** ``emit`` builds an event every time it is called (so the
  most recent state is always available via ``last_event``) but invokes the
  user callback at most once per ``min_interval`` seconds. Phase transitions
  and completion pass ``force=True`` so they are never dropped.
* **Cancellation.** A ``cancelled`` predicate is checked on *every* ``emit``
  regardless of throttling; when it returns True the emit raises
  :class:`SearchCancelled`, which the search lets propagate so callers (e.g.
  the REST job manager) can transition to a cancelled state. Because emits
  occur at least once per bisection iteration, cancellation latency is bounded
  by one design-build cycle — sub-iteration cancellation is future work.
* **Granularity.** Searches emit a forced event at each phase boundary and,
  within the optimizing phase, one event *before* each expensive design build
  (``trial_n`` known, power not yet) and one *after* (achieved power). This
  brackets the long Fedorov build so a poller never sees a long silent gap at
  iteration granularity. A heartbeat *within* a single very long optimizer
  start is not yet emitted (it would require instrumenting the parallel-capable
  Fedorov engine).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class Phase(str, Enum):
    """Ordered phases of a design search."""

    VALIDATING = "validating"
    GENERATING_CANDIDATES = "generating_candidates"
    OPTIMIZING = "optimizing"
    VERIFYING = "verifying"
    WRITING_OUTPUT = "writing_output"
    DONE = "done"


class SearchCancelled(Exception):
    """Raised inside a search when its cancellation predicate returns True."""


@dataclass
class ProgressEvent:
    """A single progress observation.

    Attributes
    ----------
    phase : str
        One of the :class:`Phase` values.
    message : str
        Human-readable one-line description.
    iteration : int
        Bisection iteration index (0 outside the optimizing phase).
    trial_n : int or None
        Sample size being evaluated (None outside the optimizing phase).
    current_power : float or None
        Achieved power at ``trial_n`` (None before the build completes).
    target_power : float or None
        Target power for the search.
    elapsed_sec : float
        Seconds since the reporter was created.
    seq : int
        Monotonically increasing sequence number (useful for detecting a
        stalled search: a running search whose ``seq`` stops advancing for a
        long time may warrant cancellation).
    """

    phase: str
    message: str = ""
    iteration: int = 0
    trial_n: Optional[int] = None
    current_power: Optional[float] = None
    target_power: Optional[float] = None
    elapsed_sec: float = 0.0
    seq: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProgressReporter:
    """Throttled, cancellation-aware progress dispatcher.

    Parameters
    ----------
    callback : callable, optional
        Invoked with a :class:`ProgressEvent`. Exceptions raised by the
        callback are swallowed (a faulty progress sink must never break a
        search). ``None`` disables delivery but still tracks ``last_event``.
    min_interval : float, default 0.5
        Minimum seconds between (non-forced) callback invocations.
    cancelled : callable, optional
        Zero-argument predicate; when it returns True, the next ``emit``
        raises :class:`SearchCancelled`.
    clock : callable, optional
        Monotonic clock, overridable for testing.
    """

    def __init__(
        self,
        callback: Optional[Callable[[ProgressEvent], None]] = None,
        *,
        min_interval: float = 0.5,
        cancelled: Optional[Callable[[], bool]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._cb = callback
        self._min_interval = max(0.0, float(min_interval))
        self._cancelled = cancelled
        self._clock = clock
        self._t0 = clock()
        self._last_emit = float("-inf")
        self._seq = 0
        self.last_event: Optional[ProgressEvent] = None

    def emit(
        self,
        phase: "Phase | str",
        *,
        message: str = "",
        iteration: int = 0,
        trial_n: Optional[int] = None,
        current_power: Optional[float] = None,
        target_power: Optional[float] = None,
        force: bool = False,
    ) -> ProgressEvent:
        """Record an event; deliver it if throttling allows or ``force``.

        Always checks the cancellation predicate first (raising
        :class:`SearchCancelled` if set), so cancellation is honored even when
        the callback itself is throttled.
        """
        if self._cancelled is not None and self._cancelled():
            raise SearchCancelled("Search cancelled by request.")

        now = self._clock()
        self._seq += 1
        phase_str = phase.value if isinstance(phase, Phase) else str(phase)
        ev = ProgressEvent(
            phase=phase_str,
            message=message,
            iteration=int(iteration),
            trial_n=trial_n,
            current_power=current_power,
            target_power=target_power,
            elapsed_sec=now - self._t0,
            seq=self._seq,
        )
        self.last_event = ev

        if self._cb is not None and (force or (now - self._last_emit) >= self._min_interval):
            self._last_emit = now
            try:
                self._cb(ev)
            except Exception:
                # A faulty progress sink must never break the search.
                pass
        return ev


def _coerce_reporter(
    on_progress: "ProgressReporter | Callable[[ProgressEvent], None] | None",
) -> Optional[ProgressReporter]:
    """Return a ProgressReporter for a reporter, a bare callback, or None."""
    if on_progress is None:
        return None
    if isinstance(on_progress, ProgressReporter):
        return on_progress
    return ProgressReporter(on_progress)


__all__ = ["Phase", "ProgressEvent", "ProgressReporter", "SearchCancelled"]
