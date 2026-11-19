"""Where a translation's time goes, in a form short enough for one log line.

Each part logs the seconds of its stages and the latency of its translator
requests; the document logs the sum. Stages run one after another within a
part, so a stage's seconds are wall time; translator requests overlap.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# A request slower than this is logged on its own: one slow request holds up
# the whole translation stage of its part.
SLOW_REQUEST_SECONDS = 60.0


def _percentile(values: list[float], q: float) -> float:
    return values[min(len(values) - 1, int(q * len(values)))]


class RequestLatency:
    """Latency of translator requests, safe to record from the pool's threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._seconds: list[float] = []
        self._errors = 0
        self._in_flight = 0
        self._max_in_flight = 0

    @contextmanager
    def measure(self, method: str):
        with self._lock:
            self._in_flight += 1
            self._max_in_flight = max(self._max_in_flight, self._in_flight)
        start = time.monotonic()
        ok = False
        try:
            yield
            ok = True
        finally:
            seconds = time.monotonic() - start
            with self._lock:
                self._in_flight -= 1
                self._seconds.append(seconds)
                self._errors += not ok
            if seconds >= SLOW_REQUEST_SECONDS:
                logger.warning(
                    "slow translator request: %s took %.1fs (%s)",
                    method,
                    seconds,
                    "ok" if ok else "failed",
                )

    def merge(self, other: RequestLatency) -> None:
        with other._lock:
            seconds, errors, peak = other._seconds[:], other._errors, other._max_in_flight
        with self._lock:
            self._seconds += seconds
            self._errors += errors
            self._max_in_flight = max(self._max_in_flight, peak)

    def summary(self) -> str:
        with self._lock:
            values = sorted(self._seconds)
            errors, peak = self._errors, self._max_in_flight
        if not values:
            return "requests 0"
        return (
            f"requests {len(values)} (failed {errors}, at once max {peak}), "
            f"p50 {_percentile(values, 0.5):.1f}s p90 {_percentile(values, 0.9):.1f}s "
            f"p99 {_percentile(values, 0.99):.1f}s max {values[-1]:.1f}s, "
            f"sum {sum(values):.0f}s"
        )


def stage_summary(stage_seconds: dict[str, float]) -> str:
    """Stages in the order they ran, with whole seconds."""
    return ", ".join(f"{name} {seconds:.1f}s" for name, seconds in stage_seconds.items())
