"""Send a slow translator request a second time and take whichever answers first.

A part's translation stage lasts as long as its slowest request, so one
request that hangs on the provider side holds up the whole part. With
BABELDOC_HEDGE_AFTER=<seconds>, a request still running after that long is
sent again; the first answer wins and the other is left to finish on its own
(an HTTP call cannot be withdrawn), so a hedged request is paid for twice.
Unset, every request runs once, as before.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from typing import TypeVar

logger = logging.getLogger(__name__)

HEDGE_AFTER_ENV = "BABELDOC_HEDGE_AFTER"

T = TypeVar("T")


def hedge_after() -> float | None:
    """Seconds after which a request is sent again, or None when hedging is off."""
    raw = os.environ.get(HEDGE_AFTER_ENV, "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        seconds = 0.0
    if seconds > 0:
        return seconds
    logger.warning("Ignoring %s=%r: expected a positive number of seconds", HEDGE_AFTER_ENV, raw)
    return None


class Hedger:
    """Runs calls on its own threads so a slow one can be raced by a copy."""

    def __init__(self, after: float):
        self.after = after
        self._pool = ThreadPoolExecutor(thread_name_prefix="babeldoc-hedge", max_workers=512)
        self._lock = threading.Lock()
        self.hedged = 0
        self.won_by_copy = 0

    def call(self, fn: Callable[[], T]) -> T:
        first = self._pool.submit(fn)
        done, _ = wait([first], timeout=self.after)
        if done:
            return first.result()
        second = self._pool.submit(fn)
        with self._lock:
            self.hedged += 1
        pending: set[Future] = {first, second}
        error: BaseException | None = None
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                if future.exception() is None:
                    if future is second:
                        with self._lock:
                            self.won_by_copy += 1
                    return future.result()
                error = error or future.exception()
        raise error

    def summary(self) -> str:
        with self._lock:
            return f"hedged {self.hedged} after {self.after:g}s, copy won {self.won_by_copy}"

    def close(self) -> None:
        # Losing copies may still be waiting on the provider; let them end alone.
        self._pool.shutdown(wait=False)
