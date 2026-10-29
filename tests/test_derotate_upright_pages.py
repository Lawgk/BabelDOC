import numpy as np
import pymupdf
import pytest
from babeldoc.format.pdf.high_level import derotate_upright_pages
from babeldoc.format.pdf.new_parser.native_parse import (
    parse_with_new_parser_to_legacy_ir,
)

ROTATIONS = [90, 180, 270]
TEXT = "Landscape page text"


def _write_pdf(path, rotate, *, upright):
    # upright: the text is drawn turned against /Rotate, so the page reads normally
    # on screen. That is what printing a landscape page through PostScript produces.
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((150, 200), TEXT, fontsize=14, rotate=rotate if upright else 0)
    page.set_rotation(rotate)
    doc.save(path)


def _pixels(page):
    pix = page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY)
    return (pix.width, pix.height), np.frombuffer(pix.samples, dtype=np.uint8)


@pytest.mark.parametrize("rotate", ROTATIONS)
def test_upright_text_is_parsed_horizontally_after_derotation(tmp_path, rotate):
    src = tmp_path / "in.pdf"
    _write_pdf(src, rotate, upright=True)
    before = parse_with_new_parser_to_legacy_ir(src, working_dir=tmp_path).page[0]
    # Every glyph is vertical (90/270) or upside down and dropped (180).
    assert not [c for c in before.pdf_character if not c.vertical]

    doc = pymupdf.open(src)
    size, pixels = _pixels(doc[0])
    assert derotate_upright_pages(doc) == [0]
    assert doc[0].rotation == 0
    derotated_size, derotated_pixels = _pixels(doc[0])
    assert derotated_size == size
    assert np.abs(derotated_pixels.astype(int) - pixels).mean() < 1

    out = tmp_path / "derotated.pdf"
    doc.save(out)
    chars = parse_with_new_parser_to_legacy_ir(out, working_dir=tmp_path).page[0]
    assert "".join(c.char_unicode for c in chars.pdf_character) == TEXT
    assert not [c for c in chars.pdf_character if c.vertical]


@pytest.mark.parametrize("rotate", ROTATIONS)
def test_sideways_text_keeps_its_rotation(tmp_path, rotate):
    # The text already runs horizontally in user space and only looks turned on
    # screen; derotating would turn it away from what the IL handles.
    src = tmp_path / "in.pdf"
    _write_pdf(src, rotate, upright=False)
    doc = pymupdf.open(src)
    assert derotate_upright_pages(doc) == []
    assert doc[0].rotation == rotate


def test_page_without_text_keeps_its_rotation():
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.draw_rect(pymupdf.Rect(40, 60, 140, 120), color=(0, 0, 0), fill=(0, 0, 0))
    page.set_rotation(90)
    assert derotate_upright_pages(doc) == []
    assert doc[0].rotation == 90
