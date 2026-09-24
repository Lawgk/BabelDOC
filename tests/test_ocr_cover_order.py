import numpy as np
import pymupdf
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.backend.pdf_creater import PDFCreater
from babeldoc.format.pdf.document_il.midend.detect_scanned_file import DetectScannedFile
from babeldoc.format.pdf.document_il.utils.style_helper import WHITE
from babeldoc.format.pdf.high_level import fix_media_box
from babeldoc.format.pdf.new_parser.native_parse import (
    parse_with_new_parser_to_legacy_ir,
)
from babeldoc.format.pdf.parse_shared import build_parse_only_config


def _dark_pixels(page, clip):
    pix = page.get_pixmap(dpi=144, clip=clip, colorspace=pymupdf.csGRAY)
    return int((np.frombuffer(pix.samples, dtype=np.uint8) < 128).sum())


def test_untranslated_text_is_drawn_above_its_ocr_cover(tmp_path):
    # A table cell too short to translate keeps its original chars; in OCR mode
    # it still gets a white cover, and it must stay visible on top of it.
    src = tmp_path / "in.pdf"
    doc = pymupdf.open()
    doc.new_page(width=300, height=400).insert_text((40, 200), "1685", fontsize=14)
    doc.save(src)

    docs = parse_with_new_parser_to_legacy_ir(src, working_dir=tmp_path)
    config = build_parse_only_config(src, working_dir=tmp_path)
    config.ocr_workaround = True
    DetectScannedFile(config).clean_render_order_for_chars(docs)
    page = docs.page[0]
    chars = page.pdf_character
    page.pdf_rectangle.append(
        il_version_1.PdfRectangle(
            box=il_version_1.Box(
                min(c.box.x for c in chars) - 2,
                min(c.box.y for c in chars) - 2,
                max(c.box.x2 for c in chars) + 2,
                max(c.box.y2 for c in chars) + 2,
            ),
            fill_background=True,
            graphic_state=WHITE,
            debug_info=False,
            xobj_id=chars[0].xobj_id,
        )
    )

    pdf = pymupdf.open(src)
    text_area = pdf[0].search_for("1685")[0]
    creater = PDFCreater(str(src), docs, config, fix_media_box(pdf))
    creater.update_page_content_stream(False, page, pdf, config)

    assert _dark_pixels(pdf[0], text_area) > 0
