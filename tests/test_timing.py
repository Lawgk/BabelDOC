import logging
import re

import pymupdf

from babeldoc import timing


def _book(pages: int) -> bytes:
    doc = pymupdf.open()
    for number in range(1, pages + 1):
        page = doc.new_page(width=300, height=400)
        for line in range(25):
            page.insert_text(
                (40, 60 + 13 * line), f"Line {line} of page {number} carries body text.", fontsize=9
            )
    return doc.tobytes()


def _split(pages_per_part: int):
    from babeldoc.format.pdf.translation_config import TranslationConfig

    return TranslationConfig.create_max_pages_per_part_split_strategy(pages_per_part)


def test_latency_summary_counts_failures_and_overlap():
    latency = timing.RequestLatency()
    with latency.measure("translate"):
        with latency.measure("translate"):
            pass
    try:
        with latency.measure("translate"):
            raise ValueError
    except ValueError:
        pass
    summary = latency.summary()
    assert summary.startswith("requests 3 (failed 1, at once max 2)")
    assert re.search(r"p50 [\d.]+s p90 [\d.]+s p99 [\d.]+s max [\d.]+s", summary)


def test_empty_latency_summary():
    assert timing.RequestLatency().summary() == "requests 0"


def test_slow_request_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(timing, "SLOW_REQUEST_SECONDS", 0.0)
    with caplog.at_level(logging.WARNING, logger="babeldoc.timing"):
        with timing.RequestLatency().measure("llm_translate"):
            pass
    assert "slow translator request: llm_translate took" in caplog.text


def test_part_timing_lists_stages_and_requests(translate_pdf, caplog):
    with caplog.at_level(logging.INFO, logger="babeldoc.format.pdf.high_level"):
        translate_pdf(_book(1))
    line = next(r.getMessage() for r in caplog.records if "part 1/1 timing" in r.getMessage())
    assert "Translate Paragraphs" in line
    assert re.search(r"translator: requests [1-9]", line)
    assert "document timing" not in caplog.text


def test_document_timing_sums_the_parts(translate_pdf, caplog):
    with caplog.at_level(logging.INFO, logger="babeldoc.format.pdf.high_level"):
        translate_pdf(_book(4), split_strategy=_split(2))
    parts = [r.getMessage() for r in caplog.records if re.match(r"part \d/2 timing", r.getMessage())]
    assert len(parts) == 2
    document = next(r.getMessage() for r in caplog.records if "document timing" in r.getMessage())
    assert "over 2 parts" in document
    requests = [int(re.search(r"translator: requests (\d+)", p).group(1)) for p in parts]
    assert f"translator: requests {sum(requests)} " in document
