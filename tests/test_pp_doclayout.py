"""PP-DocLayoutV3 output mapped onto the DocStructBench classes BabelDOC uses."""

import numpy as np
import pytest

from babeldoc.assets import assets
from babeldoc.docvision import inference
from babeldoc.docvision import pp_doclayout
from babeldoc.docvision.base_doclayout import DocLayoutModel
from babeldoc.docvision.pp_doclayout import LABELS
from babeldoc.docvision.pp_doclayout import NAMES
from babeldoc.docvision.pp_doclayout import PPDocLayoutV3Model
from babeldoc.docvision.pp_doclayout import postprocess


def row(label, box, score=0.9):
    return [LABELS.index(label), score, *box, 0.0]


def classes(rows, width=600, height=800, **kwargs):
    preds = postprocess(np.array(rows, dtype=np.float32), width, height, **kwargs)
    return sorted((NAMES[int(p[5])], tuple(int(round(v)) for v in p[:4])) for p in preds)


def test_labels_map_onto_docstructbench():
    rows = [
        row("doc_title", (50, 20, 550, 60)),
        row("paragraph_title", (50, 80, 300, 100)),
        row("text", (50, 110, 550, 200)),
        row("header", (50, 0, 550, 15)),
        row("number", (290, 780, 310, 795)),
        row("display_formula", (100, 210, 500, 250)),
        row("formula_number", (520, 220, 550, 240)),
        row("chart", (50, 300, 550, 500)),
        row("inline_formula", (60, 120, 90, 130)),  # stays inside its paragraph
    ]
    assert classes(rows) == [
        ("abandon", (50, 0, 550, 15)),
        ("abandon", (290, 780, 310, 795)),
        ("figure", (50, 300, 550, 500)),
        ("formula_caption", (520, 220, 550, 240)),
        ("isolate_formula", (100, 210, 500, 250)),
        ("plain text", (50, 110, 550, 200)),
        ("title", (50, 20, 550, 60)),
        ("title", (50, 80, 300, 100)),
    ]


def test_score_cut_and_unknown_labels():
    rows = [
        row("text", (50, 110, 550, 200), score=0.49),
        row("text", (50, 210, 550, 300), score=0.5),
        [len(LABELS), 0.9, 0, 0, 10, 10, 0],
    ]
    assert classes(rows) == [("plain text", (50, 210, 550, 300))]
    assert len(classes(rows, cut=0.3)) == 2


def test_boxes_are_clipped_and_empty_ones_dropped():
    rows = [row("text", (-5, -3, 620, 400)), row("text", (610, 10, 650, 20))]
    assert classes(rows) == [("plain text", (0, 0, 600, 400))]


def test_nms_within_a_label_but_not_across():
    rows = [
        row("text", (50, 100, 550, 200), 0.9),
        row("text", (52, 102, 548, 198), 0.8),  # duplicate, dropped
        row("table", (52, 102, 548, 198), 0.7),  # other label, kept
    ]
    assert classes(rows) == [
        ("plain text", (50, 100, 550, 200)),
        ("table", (52, 102, 548, 198)),
    ]


def test_captions_and_footnotes_follow_their_table_or_figure():
    rows = [
        row("figure_title", (100, 70, 500, 90)),  # above the table
        row("table", (50, 100, 550, 300)),
        row("vision_footnote", (60, 305, 400, 320)),  # under the table
        row("image", (50, 400, 550, 600)),
        row("figure_title", (100, 610, 500, 630)),  # under the figure
        row("vision_footnote", (60, 640, 400, 655)),  # under a photo: its caption
    ]
    assert classes(rows) == [
        ("figure", (50, 400, 550, 600)),
        ("figure_caption", (60, 640, 400, 655)),
        ("figure_caption", (100, 610, 500, 630)),
        ("table", (50, 100, 550, 300)),
        ("table_caption", (100, 70, 500, 90)),
        ("table_footnote", (60, 305, 400, 320)),
    ]


def test_heading_right_above_a_table_is_its_caption():
    rows = [
        row("paragraph_title", (60, 85, 300, 98)),  # 2 px above the table
        row("table", (50, 100, 550, 300)),
        row("paragraph_title", (60, 20, 300, 35)),  # far above: a heading
        row("paragraph_title", (520, 85, 600, 98)),  # beside it, mostly off the table
        row("paragraph_title", (60, 302, 300, 315)),  # below it
    ]
    assert classes(rows) == [
        ("table", (50, 100, 550, 300)),
        ("table_caption", (60, 85, 300, 98)),
        ("title", (60, 20, 300, 35)),
        ("title", (60, 302, 300, 315)),
        ("title", (520, 85, 600, 98)),
    ]


def test_caption_without_table_or_figure_is_a_figure_caption():
    assert classes([row("figure_title", (100, 70, 500, 90))]) == [
        ("figure_caption", (100, 70, 500, 90))
    ]


class FakeSession:
    def __init__(self, rows):
        self.rows = np.array(rows, dtype=np.float32)
        self.feeds = None

    def run(self, _outputs, feeds):
        self.feeds = feeds
        return [self.rows, np.array([len(self.rows)], dtype=np.int32)]


class Geometry:
    def px_len_to_pt(self, length, axis):
        return length / (2.0 if axis == "x" else 4.0)


def test_predict_feeds_and_result():
    model = object.__new__(PPDocLayoutV3Model)
    model.model = FakeSession([row("text", (60, 80, 300, 400))])
    image = np.zeros((842, 595, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    result = model.predict(image, geometry=Geometry())
    feeds = model.model.feeds
    assert feeds["image"].shape == (1, 3, 800, 800)
    assert feeds["image"][0, 2].max() == 1.0  # RGB order: blue last
    assert feeds["image"][0, 0].max() == 0.0
    assert np.allclose(feeds["scale_factor"], [[800 / 842, 800 / 595]])
    (box,) = result.boxes
    assert np.allclose(box.xyxy, [30, 20, 150, 100])
    assert result.names[box.cls] == "plain text"
    assert isinstance(box.conf.item(), float)


def test_empty_page():
    model = object.__new__(PPDocLayoutV3Model)
    model.model = FakeSession(np.zeros((0, 7)))
    assert model.predict(np.zeros((100, 80, 3), dtype=np.uint8)).boxes == []


@pytest.mark.parametrize(
    ("value", "pp"), [(None, False), ("pp-doclayoutv3", True), ("PP-DocLayoutV3", True), ("yolo11", False)]
)
def test_selection(monkeypatch, value, pp):
    if value is None:
        monkeypatch.delenv(inference.LAYOUT_MODEL_ENV, raising=False)
    else:
        monkeypatch.setenv(inference.LAYOUT_MODEL_ENV, value)
    monkeypatch.setattr(PPDocLayoutV3Model, "from_pretrained", staticmethod(lambda: "pp"))
    from babeldoc.docvision import doclayout

    monkeypatch.setattr(doclayout.OnnxModel, "from_pretrained", staticmethod(lambda: "yolo"))
    assert DocLayoutModel.load_onnx() == ("pp" if pp else "yolo")
    names = [m["name"] for m in assets.generate_all_assets_file_list()["models"]]
    assert (assets.PP_DOCLAYOUTV3_ONNX_FILE_NAME in names) == pp
