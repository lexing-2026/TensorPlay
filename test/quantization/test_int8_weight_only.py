"""INT8 weight-only quantized training.

Covers the :class:`Int8QuantizedTrainingLinearWeight` parameter subclass that
quantizes weights to int8 row-wise codes on the fly with a straight-through
estimator, the ``quantize_(model, int8_weight_only_quantized_training())``
entry point, and FSDP/DDP helpers. The parameter stores the full-precision
weight so any optimizer (including fused kernels) works unchanged.
"""
import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

from tensorplay.ao.quantization import (
    Int8QuantizedTrainingLinearWeight,
    int8_weight_only_quantized_training,
    quantize_,
)

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])


def manual_int8wo_linear(x, w, b=None):
    int_data = w.int_data
    scale = w.scale
    out = (x @ int_data.T.to(x.dtype)) * scale
    if b is not None:
        out = out + b
    return out


@pytest.mark.parametrize('device', DEVICES)
def test_from_float_and_forward(device):
    tp.manual_seed(0)
    x = tp.randn(3, 8, device=device, requires_grad=True)
    w = tp.randn(5, 8, device=device, requires_grad=True)
    b = tp.randn(5, device=device)

    quant_weight = Int8QuantizedTrainingLinearWeight.from_float(w)
    assert quant_weight.int_data.dtype == tp.int8
    assert quant_weight.scale.shape == (5,)
    tp.testing.assert_close(
        quant_weight.dequantize(),
        quant_weight.int_data.to(tp.float32) * quant_weight.scale.view(-1, 1),
        atol=1e-5, rtol=1e-5,
    )

    out = F.linear(x, quant_weight, b)
    ref = manual_int8wo_linear(x, quant_weight, b)
    tp.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    out.sum().backward()
    int_data = quant_weight.int_data
    scale = quant_weight.scale
    grad_input_expected = (tp.ones_like(out) * scale) @ int_data.to(tp.float32)
    tp.testing.assert_close(x.grad, grad_input_expected, atol=1e-5, rtol=1e-5)
    grad_weight_expected = tp.ones_like(out).reshape(-1, 5).T @ x.reshape(-1, 8)
    tp.testing.assert_close(quant_weight.grad, grad_weight_expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_quantize_converts_linear_model(device):
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)
    quantize_(model, int8_weight_only_quantized_training())
    assert isinstance(model[0].weight, Int8QuantizedTrainingLinearWeight)
    assert isinstance(model[2].weight, Int8QuantizedTrainingLinearWeight)
    assert model[0].weight.int_data.dtype == tp.int8

    x = tp.randn(3, 8, device=device)
    out = model(x)
    ref0 = manual_int8wo_linear(x, model[0].weight, model[0].bias)
    ref1 = manual_int8wo_linear(
        nn.functional.relu(ref0), model[2].weight, model[2].bias
    )
    tp.testing.assert_close(out, ref1, atol=1e-5, rtol=1e-5)

    loss = out.square().mean()
    loss.backward()
    assert model[0].weight.grad is not None

    # The optimizer updates the float storage; the int8 codes follow on the
    # next quantization (fused kernels bypass Python dispatch entirely).
    opt = tp.optim.SGD(model.parameters(), lr=0.5)
    before_float = model[0].weight._data.clone()
    before_codes = model[0].weight.int_data.clone()
    opt.step()
    assert not tp.allclose(before_float, model[0].weight._data)
    assert model[0].weight.int_data.dtype == tp.int8
    assert not tp.allclose(before_codes, model[0].weight.int_data)
    tp.testing.assert_close(
        model[0].weight._data, model[0].weight.detach(), atol=0, rtol=0
    )


@pytest.mark.parametrize('device', DEVICES)
def test_copy_from_float(device):
    w = tp.randn(4, 6, device=device)
    quant_weight = Int8QuantizedTrainingLinearWeight.from_float(w)
    new_float = tp.randn(4, 6, device=device)

    quant_weight.copy_(new_float)
    tp.testing.assert_close(quant_weight._data, new_float, atol=0, rtol=0)
    assert quant_weight.int_data.dtype == tp.int8
    tp.testing.assert_close(
        quant_weight.dequantize(),
        quant_weight.int_data.to(tp.float32) * quant_weight.scale.view(-1, 1),
        atol=1e-5, rtol=1e-5,
    )


@pytest.mark.parametrize('device', DEVICES)
def test_fsdp_hooks_roundtrip(device):
    w = tp.randn(8, 16, device=device)
    quant_weight = Int8QuantizedTrainingLinearWeight.from_float(w)
    data, metadata = quant_weight.fsdp_pre_all_gather(None)
    int_data, scale = data
    assert int_data.shape == (8, 16)
    assert scale.shape == (8,)
    assert metadata is None

    rebuilt, inner = quant_weight.fsdp_post_all_gather(data, metadata, tp.float32)
    assert isinstance(rebuilt, Int8QuantizedTrainingLinearWeight)
    assert len(inner) == 2
    tp.testing.assert_close(rebuilt.dequantize(), quant_weight.dequantize(),
                            atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('device', DEVICES)
def test_data_property_sync(device):
    w = tp.randn(4, 6, device=device)
    quant_weight = Int8QuantizedTrainingLinearWeight.from_float(w)
    assert quant_weight.data is quant_weight._data

    new_data = tp.randn(4, 6, device=device)
    quant_weight.data = new_data
    assert quant_weight._data is new_data
    tp.testing.assert_close(quant_weight.detach(), new_data, atol=0, rtol=0)


def test_quantize_then_module_apply_roundtrip():
    model = nn.Linear(4, 6)
    quantize_(model, int8_weight_only_quantized_training())
    model = model.to(tp.float64)
    assert model.weight.int_data.dtype == tp.int8
    x = tp.randn(2, 4, dtype=tp.float64)
    out = model(x)
    assert tuple(out.shape) == (2, 6)
    model = model.to(tp.float32)
    tp.testing.assert_close(model.weight._data, model.weight.detach(), atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_int8_weight_only_end_to_end(device):
    model = nn.Linear(8, 8).to(device)
    quantize_(model, int8_weight_only_quantized_training())
    opt = tp.optim.SGD(model.parameters(), lr=0.05)
    x = tp.randn(16, 8, device=device)
    target = tp.randn(16, 8, device=device)
    initial = float(model(x).sub(target).square().mean())
    for _ in range(20):
        opt.zero_grad()
        loss = model(x).sub(target).square().mean()
        loss.backward()
        opt.step()
    final = float(model(x).sub(target).square().mean())
    assert final < initial


def test_zeros_like_returns_plain_tensor():
    w = tp.randn(4, 6)
    quant_weight = Int8QuantizedTrainingLinearWeight.from_float(w)
    z = tp.zeros_like(quant_weight)
    assert not isinstance(z, Int8QuantizedTrainingLinearWeight)
    assert tuple(z.shape) == (4, 6)
    tp.testing.assert_close(z, tp.zeros((4, 6)), atol=0, rtol=0)


def test_sub_mul_dequantize():
    w = tp.randn(4, 6)
    quant_weight = Int8QuantizedTrainingLinearWeight.from_float(w)
    other = tp.randn(4, 6)
    # Arithmetic on the parameter operates on its float storage (native path).
    sub = quant_weight - other
    mul = quant_weight * 2.0
    assert not isinstance(sub, Int8QuantizedTrainingLinearWeight)
    assert not isinstance(mul, Int8QuantizedTrainingLinearWeight)
    tp.testing.assert_close(sub, quant_weight._data - other, atol=1e-5, rtol=1e-5)
    tp.testing.assert_close(mul, quant_weight._data * 2.0, atol=1e-5, rtol=1e-5)


def test_split_cat_dim0():
    w = tp.randn(6, 8)
    quant_weight = Int8QuantizedTrainingLinearWeight.from_float(w)
    halves = quant_weight.split(3, dim=0)
    assert len(halves) == 2
    assert all(isinstance(h, Int8QuantizedTrainingLinearWeight) for h in halves)
    tp.testing.assert_close(
        tp.cat(halves, dim=0).dequantize(),
        quant_weight.dequantize(),
        atol=1e-5, rtol=1e-5,
    )


def test_quantize_custom_filter():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    quantize_(
        model,
        int8_weight_only_quantized_training(),
        filter_fn=lambda m, name: name.endswith("1"),
    )
    assert not isinstance(model[0].weight, Int8QuantizedTrainingLinearWeight)
    assert isinstance(model[1].weight, Int8QuantizedTrainingLinearWeight)


def test_tensor_flatten_unflatten_roundtrip():
    w = tp.randn(4, 8)
    quant_weight = Int8QuantizedTrainingLinearWeight.from_float(w)
    attrs, ctx = quant_weight.__tensor_flatten__()
    inner = {name: getattr(quant_weight, name) for name in attrs}
    rebuilt = Int8QuantizedTrainingLinearWeight.__tensor_unflatten__(inner, ctx)
    tp.testing.assert_close(rebuilt.dequantize(), quant_weight.dequantize(),
                            atol=1e-5, rtol=1e-5)