from babeldoc.format.pdf.new_parser.interpreter import BeginTextObjectEvent
from babeldoc.format.pdf.new_parser.interpreter import EndTextObjectEvent
from babeldoc.format.pdf.new_parser.interpreter import TextRunEvent
from babeldoc.format.pdf.new_parser.xobject_content_execution import (
    interpret_content_stream,
)


def _text_runs(events):
    return [e for e in events if isinstance(e, TextRunEvent)]


def test_text_shown_after_et_is_kept_with_its_matrix():
    # Shape of an IOPscience cover page: the journal line and URL are emitted
    # after the closing ET, positioned by a bare Tm.
    stream = (
        b"BT /F1 10 Tf 1 0 0 1 50 700 Tm (in object) Tj ET\n"
        b"1 0 0 1 50 620 Tm (2013 Nonlinearity 26 35) Tj\n"
        b"1 0 0 1 50 600 Tm (http://iopscience.iop.org) Tj\n"
    )
    events = interpret_content_stream(stream)
    runs = _text_runs(events)
    assert [r.segments[0] for r in runs] == [
        b"in object",
        b"2013 Nonlinearity 26 35",
        b"http://iopscience.iop.org",
    ]
    assert runs[1].text_matrix == (1, 0, 0, 1, 50, 620)
    assert runs[2].text_matrix == (1, 0, 0, 1, 50, 600)
    assert runs[1].font_name == runs[0].font_name
    begins = sum(isinstance(e, BeginTextObjectEvent) for e in events)
    ends = sum(isinstance(e, EndTextObjectEvent) for e in events)
    assert begins == ends == 3


def test_implicit_text_object_does_not_reset_the_matrices():
    # A real ET resets the text matrix; the implicit one must not, or the
    # second stray run would jump back to the origin.
    stream = b"/F1 10 Tf 1 0 0 1 10 100 Tm (ab) Tj (cd) Tj\n"
    runs = _text_runs(interpret_content_stream(stream))
    assert [r.text_matrix for r in runs] == [(1, 0, 0, 1, 10, 100)] * 2


def test_real_bt_after_stray_text_resets_as_usual():
    stream = b"1 0 0 1 10 10 Tm (x) Tj BT /F1 10 Tf (y) Tj ET\n"
    runs = _text_runs(interpret_content_stream(stream))
    assert runs[1].text_matrix == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
