"""PP-DocLayoutV3 (PaddlePaddle, Apache-2.0) as BabelDOC's layout model.

The model's 25 classes are mapped onto the DocStructBench names that
DocLayout-YOLO produces and the later stages are tuned for. PP-DocLayoutV3
does not tell table captions from figure ones; on OmniDocBench these rules
recovered them (AP@0.5, DocLayout-YOLO in brackets):

- figure_title takes the type of the nearest table or figure;
- vision_footnote is a table footnote next to a table and a figure caption
  otherwise: most photo captions in newspapers and magazines come out as
  vision_footnote (figure captions 0.38 -> 0.62 [0.56]);
- a paragraph_title right above a table is its caption: report-style
  "Table 3 ..." headings (table captions 0.43 -> 0.60 [0.52]).

Select it with BABELDOC_LAYOUT_MODEL=pp-doclayoutv3.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator

import cv2
import numpy as np
import onnxruntime
import pymupdf

from babeldoc.assets.assets import get_pp_doclayoutv3_onnx_model_path
from babeldoc.docvision.base_doclayout import DocLayoutModel
from babeldoc.docvision.base_doclayout import YoloResult
from babeldoc.docvision.inference import layout_threads
from babeldoc.docvision.inference import openvino_session
from babeldoc.docvision.inference import ort_session_options
from babeldoc.format.pdf.document_il.il_version_1 import Page
from babeldoc.format.pdf.document_il.utils.raster_geometry import RasterGeometry
from babeldoc.format.pdf.document_il.utils.raster_geometry import with_target_long_edge

logger = logging.getLogger(__name__)

LABELS = (
    "abstract", "algorithm", "aside_text", "chart", "content", "display_formula",
    "doc_title", "figure_title", "footer", "footer_image", "footnote",
    "formula_number", "header", "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal", "table", "text",
    "vertical_text", "vision_footnote",
)  # fmt: skip
NAMES = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}
_CLASS = {name: index for index, name in NAMES.items()}
DIRECT = {
    "text": "plain text",
    "abstract": "plain text",
    "content": "plain text",
    "reference": "plain text",
    "reference_content": "plain text",
    "algorithm": "plain text",
    "vertical_text": "plain text",
    "doc_title": "title",
    "paragraph_title": "title",
    "header": "abandon",
    "footer": "abandon",
    "number": "abandon",
    "footnote": "abandon",
    "header_image": "abandon",
    "footer_image": "abandon",
    "aside_text": "abandon",
    "image": "figure",
    "chart": "figure",
    "seal": "figure",
    "table": "table",
    "display_formula": "isolate_formula",
    "formula_number": "formula_caption",
}
# inline_formula has no DocStructBench class: the formula stays in its paragraph.
INPUT_SIZE = 800
SCORE_CUT = 0.5


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms(dets, same=0.6, other=0.98):
    """Greedy NMS: strict within a label, only near-duplicates across labels."""
    keep = []
    for det in sorted(dets, key=lambda d: -d[1]):
        if all(_iou(det[0], k[0]) <= (same if det[2] == k[2] else other) for k in keep):
            keep.append(det)
    return keep


def _owner(box, anchors) -> str | None:
    """Label of the table or figure a caption belongs to: the nearest one."""
    cx = (box[0] + box[2]) / 2
    best, owner = None, None
    for anchor, label in anchors:
        dx = max(anchor[0] - cx, 0.0, cx - anchor[2])
        dy = max(anchor[1] - box[3], 0.0, box[1] - anchor[3])
        if best is None or dx + dy < best:
            best, owner = dx + dy, label
    return owner


def _heads_table(box, tables, height) -> bool:
    """Whether a box sits right above a table, over its width."""
    for table in tables:
        gap = table[1] - box[3]
        overlap = min(box[2], table[2]) - max(box[0], table[0])
        if -0.005 * height <= gap <= 0.02 * height and overlap >= 0.5 * (box[2] - box[0]):
            return True
    return False


def postprocess(rows: np.ndarray, width: int, height: int, cut: float = SCORE_CUT):
    """Model rows (label, score, x0, y0, x1, y1, ...) in image pixels to
    YOLO-style rows (x0, y0, x1, y1, score, class) over NAMES."""
    dets = []
    for row in rows:
        label, score = int(row[0]), float(row[1])
        if score < cut or not 0 <= label < len(LABELS):
            continue
        x0, x1 = np.clip([row[2], row[4]], 0, width)
        y0, y1 = np.clip([row[3], row[5]], 0, height)
        if x1 <= x0 or y1 <= y0:
            continue
        dets.append(((float(x0), float(y0), float(x1), float(y1)), score, LABELS[label]))
    dets = _nms(dets)
    anchors = [
        (box, DIRECT[label])
        for box, _, label in dets
        if DIRECT.get(label) in ("figure", "table")
    ]
    tables = [box for box, label in anchors if label == "table"]
    out = []
    for box, score, label in dets:
        name = DIRECT.get(label)
        if label == "figure_title":
            table = _owner(box, anchors) == "table"
            name = "table_caption" if table else "figure_caption"
        elif label == "vision_footnote":
            table = _owner(box, anchors) == "table"
            name = "table_footnote" if table else "figure_caption"
        elif label == "paragraph_title" and _heads_table(box, tables, height):
            name = "table_caption"
        if name is not None:
            out.append((*box, score, _CLASS[name]))
    return np.array(out, dtype=np.float32).reshape(-1, 6)


class PPDocLayoutV3Model(DocLayoutModel):
    def __init__(self, model_path):
        threads = layout_threads()
        logger.info(f"PP-DocLayoutV3 intra-op threads: {threads or 'runtime default'}")
        self.model = openvino_session(model_path, threads)
        if self.model is None:
            self.model = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=ort_session_options(threads),
                providers=["CPUExecutionProvider"],
            )
        self.lock = threading.Lock()

    @staticmethod
    def from_pretrained():
        return PPDocLayoutV3Model(get_pp_doclayoutv3_onnx_model_path())

    @property
    def stride(self):
        return 32

    def predict(self, image, geometry: RasterGeometry | None = None):
        """Layout of one BGR page image; boxes in points when geometry is given."""
        height, width = image.shape[:2]
        # As the model's inference.yml: RGB, stretched to 800x800, scaled to [0, 1].
        pix = cv2.resize(
            image[:, :, ::-1], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC
        )
        pix = np.ascontiguousarray(pix.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        feeds = {
            "image": pix,
            "im_shape": np.array([[INPUT_SIZE, INPUT_SIZE]], dtype=np.float32),
            # The model divides its boxes by this, giving page-image pixels.
            "scale_factor": np.array(
                [[INPUT_SIZE / height, INPUT_SIZE / width]], dtype=np.float32
            ),
        }
        preds = postprocess(self.model.run(None, feeds)[0], width, height)
        if geometry is not None and len(preds):
            preds[:, [0, 2]] = geometry.px_len_to_pt(preds[:, [0, 2]], "x")
            preds[:, [1, 3]] = geometry.px_len_to_pt(preds[:, [1, 3]], "y")
        return YoloResult(boxes_data=preds, names=NAMES)

    def handle_document(
        self,
        pages: list[Page],
        mupdf_doc: pymupdf.Document,
        translate_config,
        save_debug_image,
    ) -> Generator[tuple[Page, YoloResult], None, None]:
        for page in pages:
            translate_config.raise_if_cancelled()
            with self.lock:
                # 72 dpi, as for DocLayout-YOLO: an A4 page is 842 px high,
                # close to the model's 800 px input.
                geometry = with_target_long_edge(
                    mupdf_doc[page.page_number], 72, 1024, normalize_rotation=True
                )
            image = geometry.image[:, :, ::-1]
            result = self.predict(image, geometry=geometry)
            save_debug_image(image, result, page.page_number + 1)
            yield page, result
