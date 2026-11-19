import threading

import pymupdf
import pytest

from babeldoc.format.pdf import high_level

PAGES = 6


def _book() -> bytes:
    doc = pymupdf.open()
    for number in range(1, PAGES + 1):
        page = doc.new_page(width=300, height=400)
        for line in range(25):
            page.insert_text(
                (40, 60 + 13 * line), f"Line {line} of page {number} carries body text.", fontsize=9
            )
    return doc.tobytes()


def _split(pages_per_part: int):
    from babeldoc.format.pdf.translation_config import TranslationConfig

    return TranslationConfig.create_max_pages_per_part_split_strategy(pages_per_part)


def _parts() -> dict:
    """Three parts of two pages. Automatic term extraction, on by default,
    would keep them sequential."""
    return {"split_strategy": _split(2), "auto_extract_glossary": False}


def _text(path) -> list[str]:
    return [page.get_text() for page in pymupdf.open(path)]


class _Config:
    def __init__(self, auto_extract_glossary=False):
        self.auto_extract_glossary = auto_extract_glossary


@pytest.mark.parametrize(
    ("raw", "auto_terms", "expected"),
    [(None, False, 1), ("2", False, 2), ("3", False, 3), ("0", False, 1), ("two", False, 1), ("2", True, 1)],
)
def test_parts_in_flight_reads_the_environment(monkeypatch, raw, auto_terms, expected):
    if raw is None:
        monkeypatch.delenv(high_level.PARTS_IN_FLIGHT_ENV, raising=False)
    else:
        monkeypatch.setenv(high_level.PARTS_IN_FLIGHT_ENV, raw)
    assert high_level.parts_in_flight(_Config(auto_terms)) == expected


def _track_finishes(monkeypatch, linger: float):
    """Count parts finishing at once.

    Preparing a part takes longer than finishing it with the stub translator,
    so each finish waits until another part joins it, or `linger` seconds.
    """
    original = high_level._finish_single
    joined = threading.Condition()
    state = {"active": 0, "most": 0}

    def tracked(prepared, run_context):
        with joined:
            state["active"] += 1
            state["most"] = max(state["most"], state["active"])
            joined.notify_all()
            joined.wait_for(lambda: state["most"] >= 2, timeout=linger)
        try:
            return original(prepared, run_context)
        finally:
            with joined:
                state["active"] -= 1

    monkeypatch.setattr(high_level, "_finish_single", tracked)
    return state


def test_parts_finish_one_at_a_time_by_default(monkeypatch, translate_pdf):
    monkeypatch.delenv(high_level.PARTS_IN_FLIGHT_ENV, raising=False)
    state = _track_finishes(monkeypatch, linger=1.5)
    translate_pdf(_book(), **_parts())
    assert state["most"] == 1


def test_two_parts_in_flight_give_the_same_output(monkeypatch, translate_pdf):
    monkeypatch.delenv(high_level.PARTS_IN_FLIGHT_ENV, raising=False)
    sequential, _ = translate_pdf(_book(), **_parts())
    sequential_text = _text(sequential)
    monkeypatch.setenv(high_level.PARTS_IN_FLIGHT_ENV, "2")
    state = _track_finishes(monkeypatch, linger=30)
    progress = []

    def on_event(event, config):
        if "overall_progress" in event:
            progress.append(event["overall_progress"])

    parallel, _ = translate_pdf(_book(), **_parts(), on_event=on_event)
    assert state["most"] == 2
    assert _text(parallel) == sequential_text
    assert progress == sorted(progress)


def test_a_failing_part_stops_the_others(monkeypatch, translate_pdf):
    monkeypatch.setenv(high_level.PARTS_IN_FLIGHT_ENV, "2")
    original = high_level._finish_single

    def fail_second(prepared, run_context):
        if prepared.pm.part_index == 1:
            raise RuntimeError("part 2 broke")
        return original(prepared, run_context)

    monkeypatch.setattr(high_level, "_finish_single", fail_second)
    with pytest.raises(RuntimeError):
        translate_pdf(_book(), **_parts())
    assert not [t for t in threading.enumerate() if t.name.startswith("babeldoc-part-")]
