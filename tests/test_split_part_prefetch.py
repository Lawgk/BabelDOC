import logging
import threading
import time

import pymupdf
import pytest

PAGES = 6


def _book() -> bytes:
    doc = pymupdf.open()
    for number in range(1, PAGES + 1):
        page = doc.new_page(width=300, height=400)
        page.insert_text(
            (40, 60), f"Page {number} holds a sentence worth translating.", fontsize=10
        )
        # Enough text that the scanned-file check, run by part 0, passes.
        for line in range(1, 25):
            page.insert_text(
                (40, 60 + 13 * line), f"Line {line} of page {number} carries body text.", fontsize=9
            )
    return doc.tobytes()


def _split(pages_per_part: int):
    from babeldoc.format.pdf.translation_config import TranslationConfig

    return TranslationConfig.create_max_pages_per_part_split_strategy(pages_per_part)


def _prefetch_threads():
    return [t for t in threading.enumerate() if t.name.startswith("babeldoc-part-prefetch")]


def test_each_part_after_the_first_is_prepared_ahead(translate_pdf, caplog):
    with caplog.at_level(logging.INFO, logger="babeldoc.format.pdf.high_level"):
        mono, _ = translate_pdf(_book(), split_strategy=_split(2))
    assert "part 1 was prepared ahead" in caplog.text
    assert "part 2 was prepared ahead" in caplog.text
    sent = " | ".join(translate_pdf.inputs)
    for number in range(1, PAGES + 1):
        assert f"Page {number} holds" in sent
    with pymupdf.open(mono) as doc:
        assert doc.page_count == PAGES
        for number, page in enumerate(doc, 1):
            assert f"Page {number} holds" in page.get_text()
    assert not _prefetch_threads()


def test_a_failing_part_releases_the_part_prepared_ahead(
    translate_pdf, monkeypatch
):
    from babeldoc.format.pdf import high_level

    closed = []
    close = high_level._PreparedPart.close

    def tracking_close(self):
        closed.append(self.config._capture_part_index)
        close(self)

    def failing_finish(prepared, run_context):
        prepared.close()
        raise RuntimeError("part failed")

    monkeypatch.setattr(high_level._PreparedPart, "close", tracking_close)
    monkeypatch.setattr(high_level, "_finish_single", failing_finish)
    with pytest.raises(RuntimeError, match="part failed"):
        translate_pdf(_book(), split_strategy=_split(2))
    # Part 0 failed; part 1, prepared ahead of it, was closed too.
    assert set(closed) == {0, 1}
    assert not _prefetch_threads()


PROGRESS_EVENTS = ("progress_start", "progress_update", "progress_end")


@pytest.mark.parametrize("late", [False, True], ids=["prepared-early", "prepared-late"])
def test_overall_progress_never_goes_back(translate_pdf, monkeypatch, late):
    # pdf2zh reports overall_progress to the user as the task's percentage.
    from babeldoc.format.pdf import high_level

    if late:
        written = [threading.Event() for _ in range(PAGES)]
        prepare, finish = high_level._prepare_part, high_level._finish_single

        def late_prepare(pm, config, doc, split_points, i, run_context):
            # Hold part i back until part i-1 is written: all of its preparation
            # then reports while it is already the current part.
            if i > 0:
                assert written[i - 1].wait(60)
            return prepare(pm, config, doc, split_points, i, run_context)

        def marking_finish(prepared, run_context):
            try:
                return finish(prepared, run_context)
            finally:
                written[prepared.config._capture_part_index].set()

        monkeypatch.setattr(high_level, "_prepare_part", late_prepare)
        monkeypatch.setattr(high_level, "_finish_single", marking_finish)

    translate_pdf(_book(), split_strategy=_split(2))
    progress = [
        e["overall_progress"] for e in translate_pdf.events if e["type"] in PROGRESS_EVENTS
    ]
    assert {e.get("part_index") for e in translate_pdf.events} >= {1, 2, 3}
    assert progress == sorted(progress)
    assert progress[-1] <= 100


def test_cancelling_releases_the_part_prepared_ahead(translate_pdf, monkeypatch):
    from babeldoc.format.pdf import high_level

    prepared = []
    prepare, finish = high_level._prepare_single, high_level._finish_single

    def tracking_prepare(*args, **kwargs):
        part = prepare(*args, **kwargs)
        prepared.append(part)
        return part

    def finish_once_next_is_ready(part, run_context):
        # Translate part 0 only once part 1 is prepared, so the cancellation
        # below finds a whole part waiting in the background.
        deadline = time.monotonic() + 60
        while len(prepared) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        return finish(part, run_context)

    def cancel_when_translating(event, config):
        # pdf2zh cancels this way, from its own thread, when the user cancels.
        if event["type"] == "progress_start" and event["stage"] == "Translate Paragraphs":
            config.cancel_translation()

    monkeypatch.setattr(high_level, "_prepare_single", tracking_prepare)
    monkeypatch.setattr(high_level, "_finish_single", finish_once_next_is_ready)
    started = time.monotonic()
    with pytest.raises(Exception, match="Cancel"):
        translate_pdf(_book(), split_strategy=_split(2), on_event=cancel_when_translating)
    assert time.monotonic() - started < 60
    assert not any(e["type"] == "finish" for e in translate_pdf.events)
    assert len(prepared) == 2
    for part in prepared:
        assert part.run is None
        assert part.doc_pdf2zh.is_closed
    assert not _prefetch_threads()
