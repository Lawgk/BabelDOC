import itertools
import logging
import threading
import time

import pymupdf
import pytest

from babeldoc import hedging


def _calls(*behaviours):
    """A call whose n-th invocation follows behaviours[n]: (seconds, result or exception)."""
    counter = itertools.count()
    lock = threading.Lock()

    def call():
        with lock:
            n = next(counter)
        seconds, outcome = behaviours[n]
        time.sleep(seconds)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return call


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("30", 30.0), ("2.5", 2.5), ("0", None), ("soon", None)],
)
def test_hedge_after_reads_the_environment(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(hedging.HEDGE_AFTER_ENV, raising=False)
    else:
        monkeypatch.setenv(hedging.HEDGE_AFTER_ENV, raw)
    assert hedging.hedge_after() == expected


def test_fast_request_is_not_hedged():
    hedger = hedging.Hedger(0.5)
    assert hedger.call(_calls((0.0, "a"))) == "a"
    assert (hedger.hedged, hedger.won_by_copy) == (0, 0)
    hedger.close()


def test_slow_request_is_raced_by_a_copy():
    hedger = hedging.Hedger(0.1)
    start = time.monotonic()
    assert hedger.call(_calls((2.0, "slow"), (0.0, "copy"))) == "copy"
    assert time.monotonic() - start < 1.0
    assert (hedger.hedged, hedger.won_by_copy) == (1, 1)
    hedger.close()


def test_the_original_can_still_win():
    hedger = hedging.Hedger(0.1)
    assert hedger.call(_calls((0.3, "original"), (2.0, "copy"))) == "original"
    assert (hedger.hedged, hedger.won_by_copy) == (1, 0)
    hedger.close()


def test_a_failed_copy_does_not_hide_a_late_answer():
    hedger = hedging.Hedger(0.1)
    assert hedger.call(_calls((0.4, "original"), (0.0, RuntimeError("copy")))) == "original"
    hedger.close()


def test_failure_before_the_deadline_is_raised_without_a_copy():
    hedger = hedging.Hedger(1.0)
    with pytest.raises(RuntimeError, match="first"):
        hedger.call(_calls((0.0, RuntimeError("first"))))
    assert hedger.hedged == 0
    hedger.close()


def test_both_failing_raises():
    hedger = hedging.Hedger(0.1)
    with pytest.raises(RuntimeError):
        hedger.call(_calls((0.3, RuntimeError("first")), (0.0, RuntimeError("copy"))))
    hedger.close()


def test_part_timing_reports_hedging(monkeypatch, translate_pdf, caplog):
    monkeypatch.setenv(hedging.HEDGE_AFTER_ENV, "30")
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    for line in range(25):
        page.insert_text((40, 60 + 13 * line), f"Line {line} carries body text.", fontsize=9)
    with caplog.at_level(logging.INFO, logger="babeldoc.format.pdf.high_level"):
        translate_pdf(doc.tobytes())
    line = next(r.getMessage() for r in caplog.records if "part 1/1 timing" in r.getMessage())
    assert "hedged 0 after 30s, copy won 0" in line
