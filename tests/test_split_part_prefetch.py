import logging
import threading

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
