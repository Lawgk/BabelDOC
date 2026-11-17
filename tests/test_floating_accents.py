import pymupdf
from conftest import build_pdf

BOLD = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"

# Helvetica at 12pt: the spacing accents (WinAnsi 0x88 circumflex, 0xA8
# dieresis) are 3.996pt wide. TJ kerning strikes each accent over its letter,
# 0.9pt clear of the neighbouring letter so no gap reads as a word space.
PAGE = (
    # Advent 3B2 draws the letter, then moves back and strikes the accent
    # over it; TeX's \accent draws the accent first.
    b"BT /F1 12 Tf 72 700 Td [(Visitors from T) -75 (\\210) 408"
    b" (ACHE met in Mittelha) 408 (\\250) -75 (usern last week.)] TJ ET\n"
    # A second style in the paragraph sends it down the rich-text path.
    b"BT /F1 12 Tf 72 600 Td [(The office in Zu) 408 (\\250) -75 (rich is )] TJ"
    b" /F2 12 Tf (open) Tj /F1 12 Tf ( every day for visitors.) Tj ET\n"
)


def test_floating_accents_reach_the_translator_as_accented_letters(translate_pdf):
    mono, _ = translate_pdf(build_pdf(PAGE, extra_fonts={b"F2": BOLD}))
    sent = " | ".join(translate_pdf.inputs)
    for word in ("TÂCHE", "Mittelhäusern", "Zürich"):
        assert word in sent
    for leftover in ("¨", "ˆ", "{v"):
        assert leftover not in sent
    with pymupdf.open(mono) as doc:
        text = doc[0].get_text()
    for word in ("TÂCHE", "Mittelhäusern", "Zürich"):
        assert word in text


def test_free_standing_accents_are_left_alone(translate_pdf):
    # A dieresis after a gap is not over any letter.
    translate_pdf(
        build_pdf(b"BT /F1 12 Tf 72 700 Td (The mark a \\250 stays apart from it.) Tj ET\n")
    )
    sent = " | ".join(translate_pdf.inputs)
    assert "ä" not in sent
