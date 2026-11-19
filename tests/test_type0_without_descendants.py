import pymupdf
import pytest
from conftest import build_pdf

# PD4ML's stand-in for a glyph it could not embed: a Type0 font with no
# descendant, used only to show empty strings.
NO_DESCENDANT = (
    b"<< /Type /Font /Subtype /Type0 /BaseFont /Times_Subset /Encoding /Identity-H >>"
)
EMPTY_DESCENDANTS = (
    b"<< /Type /Font /Subtype /Type0 /BaseFont /Times_Subset /Encoding /Identity-H"
    b" /DescendantFonts [] >>"
)
VALID = (
    b"<< /Type /Font /Subtype /Type0 /BaseFont /Times_Subset /Encoding /Identity-H"
    b" /DescendantFonts [<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Times_Subset"
    b" /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> >>] >>"
)
SENTENCE = b"Keep the bedroom cool and quiet before going to bed."


def _pdf(font: bytes) -> bytes:
    content = (
        b"BT /F1 12 Tf 72 700 Td (" + SENTENCE + b") Tj ET\n"
        b"BT /F2 12 Tf 400 700 Td <> Tj ET"
    )
    return build_pdf(content, extra_fonts={b"F2": font})


@pytest.mark.parametrize(
    "font", [NO_DESCENDANT, EMPTY_DESCENDANTS], ids=["missing", "empty"]
)
def test_document_with_a_font_without_descendants_translates(translate_pdf, font):
    mono, _ = translate_pdf(_pdf(font))

    assert any("bedroom" in text for text in translate_pdf.inputs)
    with pymupdf.open(mono) as doc:
        assert "bedroom" in doc[0].get_text()


def test_only_fonts_without_descendants_are_repaired(tmp_path):
    from babeldoc.format.pdf.high_level import fix_type0_without_descendants

    path = tmp_path / "in.pdf"
    path.write_bytes(
        build_pdf(
            b"",
            extra_fonts={b"F2": NO_DESCENDANT, b"F3": EMPTY_DESCENDANTS, b"F4": VALID},
        )
    )
    with pymupdf.open(path) as doc:
        valid_before = doc.xref_object(8, compressed=True)

        assert fix_type0_without_descendants(doc) == [6, 7]

        for xref in (6, 7):
            kind, value = doc.xref_get_key(xref, "DescendantFonts")
            assert kind == "array"
            descendant = int(value.strip("[]").split()[0])
            assert doc.xref_get_key(descendant, "Subtype") == ("name", "/CIDFontType2")
            assert doc.xref_get_key(descendant, "BaseFont") == ("name", "/Times_Subset")
        assert doc.xref_object(8, compressed=True) == valid_before
        assert fix_type0_without_descendants(doc) == []
