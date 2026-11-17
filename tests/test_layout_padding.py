"""DocLayout-YOLO input: padded to the stride on CPU, square only for CoreML."""

import numpy as np
import pytest

from babeldoc.docvision.doclayout import OnnxModel


class RecordingSession:
    """Answers with one box covering the page's top-left quarter, in input pixels."""

    def __init__(self):
        self.shapes = []

    def run(self, _outputs, feeds):
        batch = feeds["images"]
        self.shapes.append(batch.shape)
        h, w = batch.shape[2:]
        # Content sits centred: find it from the padding colour (114 / 255).
        content = np.argwhere(np.abs(batch[0, 0] - 114 / 255) > 1e-6)
        (y0, x0), (y1, x1) = content.min(0), content.max(0) + 1
        box = [x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, 0.9, 1]
        return [np.array([[box]], dtype=np.float32)]


def model(square):
    m = object.__new__(OnnxModel)
    m._stride = 32
    m._names = {0: "title", 1: "plain text"}
    m._square_input = square
    m.model = RecordingSession()
    return m


@pytest.mark.parametrize(
    ("square", "shape"), [(False, (1, 3, 1024, 736)), (True, (1, 3, 1024, 1024))]
)
def test_input_shape_and_boxes_map_back(square, shape):
    m = model(square)
    page = np.zeros((842, 595, 3), dtype=np.uint8)  # A4 at 72 dpi, never 114
    result = m.predict(page)[0]
    assert m.model.shapes == [shape]
    (box,) = result.boxes
    assert np.allclose(box.xyxy, [0, 0, 595 / 2, 842 / 2], atol=1.5)
    assert result.names[box.cls] == "plain text"


def test_landscape_page_pads_height():
    m = model(False)
    m.predict(np.zeros((612, 792, 3), dtype=np.uint8))
    assert m.model.shapes == [(1, 3, 800, 1024)]
