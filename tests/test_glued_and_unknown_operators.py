import logging

from babeldoc.format.pdf.new_parser.interpreter import DEGRADATION_LOG_ATTR
from babeldoc.format.pdf.new_parser.interpreter import RestoreGraphicsStateEvent
from babeldoc.format.pdf.new_parser.interpreter import SaveGraphicsStateEvent
from babeldoc.format.pdf.new_parser.interpreter import TextRunEvent
from babeldoc.format.pdf.new_parser.tokenizer import tokenize_operations
from babeldoc.format.pdf.new_parser.xobject_content_execution import (
    interpret_content_stream,
)


def _operators(data: bytes) -> list[str]:
    return [op.operator for op in tokenize_operations(data)]


def test_glued_save_restore_is_split():
    # PDFium output: every page ends with "Qq 1 0 0 1 0 0 cm /FFT0 Do Q".
    assert _operators(b"Q\n\nQq 1 0 0 1 0 0 cm /X Do Q\n") == [
        "Q",
        "Q",
        "q",
        "cm",
        "Do",
        "Q",
    ]
    assert _operators(b"qq\nq\n0 g\n") == ["q", "q", "q", "g"]


def test_operands_stay_with_the_operator_after_the_split():
    ops = tokenize_operations(b"Qq 1 0 0 1 5 6 cm")
    assert ops[-1].operator == "cm"
    assert ops[-1].operands == [1, 0, 0, 1, 5, 6]


def test_glued_keywords_keep_the_graphics_stack_balanced():
    events = interpret_content_stream(b"qq 2 0 0 2 0 0 cm Q Qq Q\n")
    saves = sum(isinstance(e, SaveGraphicsStateEvent) for e in events)
    restores = sum(isinstance(e, RestoreGraphicsStateEvent) for e in events)
    assert saves == restores == 3


def test_unknown_operator_is_skipped_with_its_operands(caplog):
    stream = b"1 2 3 zz BT /F1 10 Tf 1 0 0 1 10 20 Tm (kept) Tj ET\n"
    with caplog.at_level(logging.WARNING):
        events = interpret_content_stream(stream)
    runs = [e for e in events if isinstance(e, TextRunEvent)]
    assert [r.segments[0] for r in runs] == [b"kept"]
    assert runs[0].text_matrix == (1, 0, 0, 1, 10, 20)
    assert "'zz'" in caplog.text


def test_unknown_operator_warns_once_per_operator(caplog):
    with caplog.at_level(logging.WARNING):
        interpret_content_stream(b"zz zz zz yy\n")
    assert caplog.text.count("'zz'") == 1
    assert caplog.text.count("'yy'") == 1


def test_skipped_operator_is_marked_as_a_degradation(caplog):
    with caplog.at_level(logging.WARNING):
        interpret_content_stream(b"zz\n")
    marked = [
        getattr(r, DEGRADATION_LOG_ATTR)
        for r in caplog.records
        if hasattr(r, DEGRADATION_LOG_ATTR)
    ]
    assert marked == [{"kind": "unknown_operator", "detail": "zz"}]
