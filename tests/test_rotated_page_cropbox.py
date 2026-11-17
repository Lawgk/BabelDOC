import numpy as np
import pymupdf
import pytest

ROTATIONS = [0, 90, 180, 270]


def _write_pdf(path, rotate):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.draw_rect(pymupdf.Rect(40, 60, 140, 120), color=(0, 0, 0), fill=(0, 0, 0))
    page.insert_text((40, 200), "Rotated page text", fontsize=14)
    page.set_rotation(rotate)
    doc.save(path)


def _dark_mask(page):
    pix = page.get_pixmap(dpi=36, colorspace=pymupdf.csGRAY)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return pixels < 64


@pytest.mark.parametrize("rotate", ROTATIONS)
def test_crop_box_is_not_swapped_on_rotated_pages(tmp_path, rotate):
    from babeldoc.rust_parser_service import parse_pdf_to_il

    src = tmp_path / "in.pdf"
    _write_pdf(src, rotate)
    page = parse_pdf_to_il(src).page[0]

    box = page.cropbox.box
    assert (box.x, box.y, box.x2, box.y2) == (0, 0, 300, 400)
    for char in page.pdf_character:
        assert box.x <= char.box.x and char.box.x2 <= box.x2
        assert box.y <= char.box.y and char.box.y2 <= box.y2


@pytest.mark.parametrize("rotate", ROTATIONS)
def test_rotated_page_keeps_its_drawing_in_the_output(tmp_path, translate_pdf, rotate):
    src = tmp_path / "src.pdf"
    _write_pdf(src, rotate)
    mono, _ = translate_pdf(src.read_bytes())
    with pymupdf.open(src) as before, pymupdf.open(mono) as after:
        assert after[0].rotation == rotate
        want = _dark_mask(before[0])
        got = _dark_mask(after[0])
    # The filled rectangle is the only large dark area; it must be where it was.
    assert want.sum() > 100
    assert (want & got).sum() >= 0.95 * want.sum()
