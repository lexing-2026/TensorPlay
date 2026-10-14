"""Tests for the beyond-upstream benchmark and FX stub workflow utilities."""

import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.ao import quantization as q
from tensorplay.ao.quantization.auto import auto_quantize
from tensorplay.ao.quantization.benchmark import (
    benchmark_latency,
    benchmark_quantized,
    compare_outputs,
    model_size_bytes,
    quantize_input_for_model,
)


def _linear_model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    return M().eval()


def _calibrate(model):
    model(tp.rand(8, 64))


def test_model_size_bytes():
    model = _linear_model()
    assert model_size_bytes(model) > 0
    qmodel = auto_quantize(model, calibration_fn=_calibrate)
    assert model_size_bytes(qmodel) < model_size_bytes(model)


def test_benchmark_latency():
    model = _linear_model()
    latency = benchmark_latency(model, tp.rand(8, 64), num_iterations=5, warmup=2)
    assert latency > 0
    assert latency < 1000  # ms


def test_compare_outputs():
    model = _linear_model()
    qmodel = auto_quantize(model, calibration_fn=_calibrate)
    metric = compare_outputs(model, qmodel, tp.rand(8, 64))
    assert metric >= 0


def test_benchmark_quantized():
    model = _linear_model()
    qmodel = auto_quantize(model, calibration_fn=_calibrate)
    res = benchmark_quantized(model, qmodel, tp.rand(8, 64), num_iterations=5)
    assert set(res) == {
        "float_latency_ms",
        "quantized_latency_ms",
        "speedup",
        "float_size_bytes",
        "quantized_size_bytes",
        "size_ratio",
        "output_metric",
    }
    assert res["float_size_bytes"] > res["quantized_size_bytes"]
    assert res["size_ratio"] < 1.0


def test_quantize_input_for_model():
    x = tp.rand(4, 4)
    xq = quantize_input_for_model(None, x)
    assert xq.dtype == tp.qint8
    # already-quantized tensors pass through
    assert quantize_input_for_model(None, xq) is xq


# ---------------------------------------------------------------------------
# FX stub workflow + public exports
# ---------------------------------------------------------------------------


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = q.QuantStub()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.dequant = q.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.relu(self.linear(self.quant(x))))


def test_prepare_fx_public_export():
    assert q.prepare_fx is not None
    assert q.convert_fx is not None
    assert q.fuse_fx is not None
    assert all(name in q.__all__ for name in ("prepare_fx", "convert_fx", "fuse_fx"))


def test_fx_stub_workflow():
    model = _StubModel().eval()
    qconfig = q.QConfig(q.MinMaxObserver.with_args(dtype=tp.qint8), q.default_per_channel_weight_observer)
    mapping = q.QConfigMapping().set_global(qconfig).set_object_type(nn.Linear, qconfig)
    gm = q.prepare_fx(model, mapping, (tp.rand(2, 4),))
    gm(tp.rand(2, 4))
    cgm = q.convert_fx(gm)
    module_types = {type(m).__name__ for m in cgm.modules()}
    assert "Quantize" in module_types
    assert "DeQuantize" in module_types
    assert "QuantizedLinear" in module_types
    out = cgm(tp.rand(2, 4))
    assert out.dtype == tp.float32  # dequantized output


def test_fx_stub_workflow_quantized_output():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = q.QuantStub()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(self.quant(x))

    model = M().eval()
    qconfig = q.QConfig(q.MinMaxObserver.with_args(dtype=tp.qint8), q.default_per_channel_weight_observer)
    mapping = q.QConfigMapping().set_global(qconfig)
    gm = q.prepare_fx(model, mapping, (tp.rand(2, 4),))
    gm(tp.rand(2, 4))
    cgm = q.convert_fx(gm)
    out = cgm(tp.rand(2, 4))  # float input, Quantize stub quantizes internally
    assert out.dtype == tp.qint8