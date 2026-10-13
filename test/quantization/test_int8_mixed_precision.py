"""INT8 mixed-precision training built on native quantized ops.

Covers the :class:`Int8MixedPrecisionTrainingLinearWeight` tensor subclass that
routes ``linear`` through dynamic int8 quantization of activations and weights,
the config-driven ``quantize_(model, int8_mixed_precision_training())`` entry
point, module-swap mode, and the FSDP all-gather hooks.
"""
import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

from tensorplay.ao.quantization import (
    Int8MixedPrecisionTrainingConfig,
    Int8MixedPrecisionTrainingLinear,
    Int8MixedPrecisionTrainingLinearWeight,
    int8_mixed_precision_training,
    quantize_,
)
from tensorplay.ao.quantization.bitnet import quantize_int8_rowwise

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])


def manual_int8mp_linear(x, w, b=None):
    """Float reference for the dynamically quantized int8 linear forward."""
    A_i8, A_scale = quantize_int8_rowwise(x.reshape(-1, x.shape[-1]))
    B_t_i8, B_scale = quantize_int8_rowwise(w)
    out = (A_i8.to(tp.float32) @ B_t_i8.to(tp.float32).T) * A_scale * B_scale.view(1, -1)
    out = out.view(*x.shape[:-1], w.shape[0])
    if b is not None:
        out = out + b
    return out


def manual_grad_input(x, w, grad_output):
    g = grad_output.reshape(-1, grad_output.shape[-1])
    g_i8, g_scale = quantize_int8_rowwise(g)
    w_t_i8, w_scale = quantize_int8_rowwise(w.T)
    grad = (g_i8.to(tp.float32) @ w_t_i8.to(tp.float32).T) * g_scale * w_scale.view(1, -1)
    return grad.view(*x.shape[:-1], w.shape[1])


def manual_grad_weight(x, w, grad_output):
    x2 = x.reshape(-1, x.shape[-1])
    go = grad_output.reshape(-1, grad_output.shape[-1])
    x_t_i8, x_scale = quantize_int8_rowwise(x2.T)
    go_t_i8, go_scale = quantize_int8_rowwise(go.T)
    grad = ((x_t_i8.to(tp.float32) @ go_t_i8.to(tp.float32).T) * x_scale * go_scale.view(1, -1)).T
    return grad


@pytest.mark.parametrize('device', DEVICES)
def test_int8mp_weight_forward_matches_reference(device):
    tp.manual_seed(0)
    x = tp.randn(3, 8, device=device, requires_grad=True)
    w = tp.randn(5, 8, device=device, requires_grad=True)
    b = tp.randn(5, device=device)

    int8mp_weight = Int8MixedPrecisionTrainingLinearWeight(w.detach())
    out = F.linear(x, int8mp_weight, b)
    ref = manual_int8mp_linear(x, w, b)
    tp.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    out.sum().backward()
    grad_input_expected = manual_grad_input(x, w, tp.ones_like(out))
    tp.testing.assert_close(x.grad, grad_input_expected, atol=1e-3, rtol=1e-3)
    grad_weight_expected = manual_grad_weight(x, w, tp.ones_like(out))
    tp.testing.assert_close(int8mp_weight.grad, grad_weight_expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('device', DEVICES)
def test_quantize_converts_linear_model(device):
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)
    quantize_(model, int8_mixed_precision_training())
    assert isinstance(model[0].weight, Int8MixedPrecisionTrainingLinearWeight)
    assert isinstance(model[2].weight, Int8MixedPrecisionTrainingLinearWeight)

    x = tp.randn(3, 8, device=device)
    out = model(x)
    ref0 = manual_int8mp_linear(x, model[0].weight._data, model[0].bias)
    ref1 = manual_int8mp_linear(
        nn.functional.relu(ref0), model[2].weight._data, model[2].bias
    )
    tp.testing.assert_close(out, ref1, atol=1e-3, rtol=1e-3)

    loss = out.square().mean()
    loss.backward()
    assert model[0].weight.grad is not None
    assert model[2].weight.grad is not None
    assert model[0].bias.grad is not None

    opt = tp.optim.SGD(model.parameters(), lr=0.01)
    before = model[0].weight._data.clone()
    opt.step()
    assert not tp.allclose(before, model[0].weight._data)
    tp.testing.assert_close(model[0].weight._data, model[0].weight.detach(), atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_config_flags(device):
    x = tp.randn(2, 6, device=device, requires_grad=True)
    w = tp.randn(4, 6, device=device, requires_grad=True)
    b = tp.randn(4, device=device)

    # output=False: the forward is an exact float linear.
    cfg = Int8MixedPrecisionTrainingConfig(output=False)
    weight = Int8MixedPrecisionTrainingLinearWeight(w.detach(), cfg)
    out = F.linear(x, weight, b)
    tp.testing.assert_close(out, x @ w.T + b, atol=1e-6, rtol=1e-6)

    # grad_input=False and grad_weight=False: float backward matmuls.
    cfg2 = Int8MixedPrecisionTrainingConfig(grad_input=False, grad_weight=False)
    weight2 = Int8MixedPrecisionTrainingLinearWeight(w.detach(), cfg2)
    out2 = F.linear(x, weight2, b)
    out2.sum().backward()
    grad_input_expected = tp.ones_like(out2) @ w
    tp.testing.assert_close(x.grad, grad_input_expected, atol=1e-5, rtol=1e-5)
    grad_weight_expected = tp.ones_like(out2).reshape(-1, 4).T @ x.reshape(-1, 6)
    tp.testing.assert_close(weight2.grad, grad_weight_expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_module_swap(device):
    model = nn.Linear(4, 6).to(device)
    quantize_(model, Int8MixedPrecisionTrainingConfig(module_swap=True))
    assert isinstance(model, Int8MixedPrecisionTrainingLinear)

    x = tp.randn(2, 4, device=device)
    out = model(x)
    ref = manual_int8mp_linear(x, model.weight, model.bias)
    tp.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('device', DEVICES)
def test_fsdp_hooks_roundtrip(device):
    w = tp.randn(8, 16, device=device)
    weight = Int8MixedPrecisionTrainingLinearWeight(w.detach())
    data, metadata = weight.fsdp_pre_all_gather(None)
    assert data[0].shape == (8, 16)
    assert metadata[0] == weight.config

    rebuilt, inner = weight.fsdp_post_all_gather(data, metadata, tp.float32)
    assert isinstance(rebuilt, Int8MixedPrecisionTrainingLinearWeight)
    assert len(inner) == 1
    tp.testing.assert_close(rebuilt._data, w, atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_data_property_sync(device):
    w = tp.randn(4, 6, device=device)
    weight = Int8MixedPrecisionTrainingLinearWeight(w.detach())
    assert weight.data is weight._data

    new_data = tp.randn(4, 6, device=device)
    weight.data = new_data
    assert weight._data is new_data
    tp.testing.assert_close(weight.detach(), new_data, atol=0, rtol=0)


def test_quantize_then_module_apply_roundtrip():
    model = nn.Linear(4, 6)
    quantize_(model, int8_mixed_precision_training())
    model = model.to(tp.float64)
    assert model.weight.dtype == tp.float64
    assert model.weight._data.dtype == tp.float64
    x = tp.randn(2, 4, dtype=tp.float64)
    out = model(x)
    assert tuple(out.shape) == (2, 6)
    model = model.to(tp.float32)
    assert model.weight.dtype == tp.float32
    tp.testing.assert_close(model.weight._data, model.weight.detach(), atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_int8mp_end_to_end(device):
    model = nn.Linear(8, 8).to(device)
    quantize_(model, int8_mixed_precision_training())
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


def test_quantize_requires_config_type():
    with pytest.raises(TypeError):
        quantize_(nn.Linear(4, 4), object())


def test_quantize_custom_filter():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    quantize_(
        model,
        int8_mixed_precision_training(),
        filter_fn=lambda m, name: name.endswith("1"),
    )
    assert not isinstance(model[0].weight, Int8MixedPrecisionTrainingLinearWeight)
    assert isinstance(model[1].weight, Int8MixedPrecisionTrainingLinearWeight)


def test_tensor_flatten_unflatten_roundtrip():
    w = tp.randn(4, 8)
    weight = Int8MixedPrecisionTrainingLinearWeight(w.detach())
    attrs, ctx = weight.__tensor_flatten__()
    inner = {name: getattr(weight, name) for name in attrs}
    rebuilt = Int8MixedPrecisionTrainingLinearWeight.__tensor_unflatten__(inner, ctx)
    tp.testing.assert_close(rebuilt._data, w, atol=0, rtol=0)
    assert rebuilt.config == weight.config


@pytest.mark.parametrize('device', DEVICES)
def test_scaled_int8_mm(device):
    from tensorplay.ao.quantization import scaled_int8_mm

    A = tp.randint(-127, 128, (4, 8), dtype=tp.int8, device=device)
    B = tp.randint(-127, 128, (8, 6), dtype=tp.int8, device=device)
    row_scale = tp.rand(4, device=device)
    col_scale = tp.rand(6, device=device)
    out = scaled_int8_mm(A, B, row_scale, col_scale)
    ref = (A.to(tp.float32) @ B.to(tp.float32)) * row_scale.view(-1, 1) * col_scale.view(1, -1)
    tp.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)