"""BABELDOC_LAYOUT_BACKEND=openvino runs the layout model on OpenVINO, else ORT."""

import sys

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import TensorProto
from onnx import helper

from babeldoc.docvision import inference


@pytest.fixture
def conv_model(tmp_path):
    """A 3->2 channel convolution with a dynamic spatial size."""
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((2, 3, 3, 3)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["images", "w"], ["out"], pads=[1, 1, 1, 1])],
        "conv",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, "h", "w"])],
        [helper.make_tensor("w", TensorProto.FLOAT, weight.shape, weight.ravel())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "conv.onnx"
    onnx.save(model, path)
    return path


@pytest.mark.parametrize("value", [None, "", "onnxruntime", "tensorrt"])
def test_onnxruntime_unless_asked(monkeypatch, conv_model, value):
    if value is None:
        monkeypatch.delenv(inference.LAYOUT_BACKEND_ENV, raising=False)
    else:
        monkeypatch.setenv(inference.LAYOUT_BACKEND_ENV, value)
    assert inference.openvino_session(conv_model, 2) is None


def test_falls_back_without_openvino(monkeypatch, conv_model):
    monkeypatch.setenv(inference.LAYOUT_BACKEND_ENV, "openvino")
    monkeypatch.setitem(sys.modules, "openvino", None)  # import raises ImportError
    assert inference.openvino_session(conv_model, 2) is None


def test_falls_back_when_the_model_does_not_load(monkeypatch, tmp_path):
    pytest.importorskip("openvino")
    monkeypatch.setenv(inference.LAYOUT_BACKEND_ENV, "OpenVINO")
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"not a model")
    assert inference.openvino_session(broken, 2) is None


def test_openvino_matches_onnxruntime(monkeypatch, conv_model):
    pytest.importorskip("openvino")
    monkeypatch.setenv(inference.LAYOUT_BACKEND_ENV, "openvino")
    session = inference.openvino_session(conv_model, 2)
    assert isinstance(session, inference.OpenVINOSession)
    reference = onnxruntime.InferenceSession(
        str(conv_model), providers=["CPUExecutionProvider"]
    )
    for shape in [(1, 3, 64, 32), (1, 3, 32, 96)]:  # shape changes between pages
        x = np.random.default_rng(1).random(shape, dtype=np.float32)
        (got,) = session.run(None, {"images": x})
        (want,) = reference.run(None, {"images": x})
        assert got.shape == want.shape
        # OpenVINO may compute in bf16 (x86 with AVX512_BF16/AMX) or f16 (ARM).
        assert np.allclose(got, want, rtol=2e-2, atol=2e-2)
