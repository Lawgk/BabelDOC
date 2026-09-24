import numpy as np
import pymupdf
import pytest
from babeldoc.format.pdf.document_il.backend.pdf_creater import PDFCreater
from babeldoc.format.pdf.high_level import fix_media_box
from babeldoc.format.pdf.new_parser.native_parse import (
    parse_with_new_parser_to_legacy_ir,
)
from babeldoc.format.pdf.parse_shared import build_parse_only_config

ROTATIONS = [0, 90, 180, 270]


def _write_pdf(path, rotate, *, text):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.draw_rect(pymupdf.Rect(40, 60, 140, 120), color=(0, 0, 0), fill=(0, 0, 0))
    if text:
        page.insert_text((40, 200), "Rotated page text", fontsize=14)
    page.set_rotation(rotate)
    doc.save(path)


def _pixels(page):
    pix = page.get_pixmap(dpi=36)
    return np.frombuffer(pix.samples, dtype=np.uint8).copy()


@pytest.mark.parametrize("rotate", ROTATIONS)
def test_crop_box_is_not_swapped_on_rotated_pages(tmp_path, rotate):
    src = tmp_path / "in.pdf"
    _write_pdf(src, rotate, text=True)

    page = parse_with_new_parser_to_legacy_ir(src, working_dir=tmp_path).page[0]

    box = page.cropbox.box
    assert (box.x, box.y, box.x2, box.y2) == (0, 0, 300, 400)
    for char in page.pdf_character:
        assert box.x <= char.box.x and char.box.x2 <= box.x2
        assert box.y <= char.box.y and char.box.y2 <= box.y2


@pytest.mark.parametrize("rotate", ROTATIONS)
def test_rebuilt_content_stays_on_rotated_pages(tmp_path, rotate):
    # Rebuilding the content stream is what both the scanned-file check and the
    # final output do; a wrong page offset moves the drawing off the page.
    src = tmp_path / "in.pdf"
    _write_pdf(src, rotate, text=False)
    docs = parse_with_new_parser_to_legacy_ir(src, working_dir=tmp_path)
    pdf = pymupdf.open(src)
    before = _pixels(pdf[0])

    config = build_parse_only_config(src, working_dir=tmp_path)
    creater = PDFCreater(str(src), docs, config, fix_media_box(pdf))
    creater.update_page_content_stream(False, docs.page[0], pdf, config, True)

    assert np.array_equal(_pixels(pdf[0]), before)
