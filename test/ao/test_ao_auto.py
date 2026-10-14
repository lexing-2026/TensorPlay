"""Tests for the one-call auto-quantization API."""

import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.ao import quantization as q
from tensorplay.ao.quantization.auto import auto_convert, auto_prepare, auto_quantize


def _conv_linear_model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 4, 3)
            self.relu = nn.ReLU()
            self.linear = nn.Linear(4 * 4 * 4, 8)

        def forward(self, x):
            return self.linear(self.relu(self.conv(x)).flatten(1))

    return M().eval()


def _calibrate(model):
    model(tp.rand(1, 2, 6, 6))


def test_auto_prepare_attaches_observers():
    model = _conv_linear_model()
    prepared = auto_prepare(model)
    assert hasattr(prepared.conv, "activation_post_process")
    assert hasattr(prepared.linear, "activation_post_process")
    # original untouched when not inplace
    assert not hasattr(model.conv, "activation_post_process")


def test_auto_prepare_with_calibration():
    model = _conv_linear_model()
    prepared = auto_prepare(model, calibration_fn=_calibrate)
    # observer has seen data
    scale, zp = prepared.conv.activation_post_process.calculate_qparams()
    assert scale.numel() == 1


def test_auto_quantize_end_to_end():
    model = _conv_linear_model()
    qmodel = auto_quantize(model, calibration_fn=_calibrate)
    assert isinstance(qmodel.linear, q.QuantizedLinear)
    assert "Conv" in type(qmodel.conv).__name__
    xq = tp.quantize_per_tensor(tp.rand(1, 2, 6, 6), 0.01, 0, tp.qint8)
    out = qmodel(xq)
    assert out.dtype == tp.qint8
    assert out.shape == (1, 8)


def test_auto_convert_prepared_model():
    model = _conv_linear_model()
    prepared = auto_prepare(model, calibration_fn=_calibrate)
    converted = auto_convert(prepared)
    assert isinstance(converted.linear, q.QuantizedLinear)


def test_public_export():
    assert q.auto_quantize is auto_quantize
    assert q.auto_prepare is auto_prepare
    assert q.auto_convert is auto_convert