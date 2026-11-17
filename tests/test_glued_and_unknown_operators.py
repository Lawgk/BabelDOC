import logging

DEGRADATION_LOG_ATTR = "babeldoc_degradation"
TEXT = b"BT /F1 12 Tf 72 700 Td (Kept) Tj ET\n"


def _degradations(caplog):
    return [
        getattr(r, DEGRADATION_LOG_ATTR)
        for r in caplog.records
        if hasattr(r, DEGRADATION_LOG_ATTR)
    ]


def test_glued_restore_save_is_split(page_words):
    # PDFium output: every page ends with "Qq 1 0 0 1 0 0 cm /FFT0 Do Q".
    # If "Qq" were dropped, the shift would leak into the text below.
    words = page_words(b"q 1 0 0 1 100 0 cm Qq 1 0 0 1 0 0 cm " + TEXT + b"Q\n", ["Kept"])
    assert abs(words["Kept"][0] - 72) < 1


def test_glued_saves_keep_the_graphics_stack_balanced(page_words):
    # Two saves glued as "qq", two restores: the shift must not survive.
    words = page_words(b"qq 1 0 0 1 100 0 cm Q Q " + TEXT, ["Kept"])
    assert abs(words["Kept"][0] - 72) < 1


def test_glued_save_restore_ends_an_inline_image(page_words):
    # No /L, so the image ends at the first "EI" followed by an operator.
    inline = b"q BI /W 1 /H 1 /BPC 8 /CS /G ID \x80 EI Qq "
    words = page_words(inline + TEXT + b"Q\n", ["Kept"])
    assert "Kept" in words


def test_unknown_operator_is_skipped_with_its_operands(page_words, caplog):
    with caplog.at_level(logging.WARNING):
        words = page_words(b"1 2 3 zz " + TEXT, ["Kept"])
    assert abs(words["Kept"][0] - 72) < 1
    assert "'zz'" in caplog.text


def test_unknown_operator_inside_text_object_is_skipped(page_words):
    words = page_words(b"BT /F1 12 Tf 72 700 Td 5 zz (Kept) Tj ET\n", ["Kept"])
    assert abs(words["Kept"][0] - 72) < 1


def test_unknown_operator_warns_once_per_operator(page_words, caplog):
    with caplog.at_level(logging.WARNING):
        page_words(b"zz zz zz yy " + TEXT, ["Kept"])
    assert caplog.text.count("'zz'") == 1
    assert caplog.text.count("'yy'") == 1


def test_skipped_operator_is_marked_as_a_degradation(page_words, caplog):
    with caplog.at_level(logging.WARNING):
        page_words(b"zz " + TEXT, ["Kept"])
    assert _degradations(caplog) == [{"kind": "unknown_operator", "detail": "zz"}]


def test_text_outside_text_object_is_marked_as_a_degradation(page_words, caplog):
    with caplog.at_level(logging.WARNING):
        page_words(b"1 0 0 1 72 650 Tm /F1 12 Tf (Kept) Tj\n", ["Kept"])
    assert _degradations(caplog) == [
        {"kind": "text_outside_text_object", "detail": "Tj"}
    ]
