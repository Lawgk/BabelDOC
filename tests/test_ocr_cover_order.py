import numpy as np
import pymupdf


def _dark_pixels(page, clip):
    pix = page.get_pixmap(dpi=144, clip=clip, colorspace=pymupdf.csGRAY)
    return int((np.frombuffer(pix.samples, dtype=np.uint8) < 128).sum())


def test_untranslated_text_stays_above_its_ocr_cover(tmp_path, translate_pdf):
    # In OCR mode every paragraph gets a white cover over the scan. A table
    # cell or page number too short to translate keeps its original chars,
    # which must still be drawn on top of that cover.
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((40, 80), "A paragraph long enough to be translated.", fontsize=10)
    page.insert_text((40, 200), "1685", fontsize=14)
    page.insert_text((140, 380), "12", fontsize=10)
    source = doc.tobytes()

    mono, _ = translate_pdf(source, ocr_workaround=True)
    with pymupdf.open(stream=source) as before:
        areas = [before[0].search_for(word)[0] for word in ("1685", "12")]
    with pymupdf.open(mono) as after:
        for area in areas:
            assert _dark_pixels(after[0], area) > 0
