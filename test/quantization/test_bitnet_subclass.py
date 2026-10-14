"""BitNet b1.58 training tensor subclass and model-level quantization API.

Covers the native ``__tensorplay_dispatch__`` subclass path:

* :class:`BitNetTrainingLinearWeight` routes ``linear`` through the straight-
  through estimator while in-place optimizer steps stay visible through the
  shared storage.
* :func:`quantize_` converts ``nn.Linear`` weights to BitNet in-place.
* :class:`BitNetPacked2bitLinearWeight` provides packed-weight training and
  the FSDP all-gather hooks.
"""
import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

from tensorplay.ao.quantization.bitnet import (
    BitNetPacked2bitLinearWeight,
    BitNetTrainingConfig,
    BitNetTrainingLinearWeight,
    bitnet_linear,
    bitnet_linear_packed,
    bitnet_training,
    pack_ternary,
    precompute_bitnet_scale_for_fsdp,
    quantize_,
    unpack_ternary,
)

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_bitnet_training_weight_linear_matches_reference(device):
    tp.manual_seed(0)
    x = tp.randn(4, 8, device=device)
    w = tp.randn(5, 8, device=device, requires_grad=True)
    b = tp.randn(5, device=device)

    bitnet_weight = BitNetTrainingLinearWeight(w.detach())
    out = F.linear(x, bitnet_weight, b)
    ref = bitnet_linear(x, w, b)
    tp.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    # Gradients flow through the subclass parameter.
    out.sum().backward()
    assert bitnet_weight.grad is not None
    ref_grad_w = tp.autograd.grad(ref.sum(), w)[0]
    tp.testing.assert_close(bitnet_weight.grad, ref_grad_w, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_quantize_converts_linear_model(device):
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    model = model.to(device)

    quantize_(model, bitnet_training())
    assert isinstance(model[0].weight, BitNetTrainingLinearWeight)
    assert isinstance(model[2].weight, BitNetTrainingLinearWeight)

    x = tp.randn(3, 8, device=device)
    out = model(x)
    ref0 = bitnet_linear(x, model[0].weight._data, model[0].bias)
    ref1 = bitnet_linear(nn.functional.relu(ref0), model[2].weight._data, model[2].bias)
    tp.testing.assert_close(out, ref1, atol=1e-5, rtol=1e-5)

    loss = out.square().mean()
    loss.backward()
    assert model[0].weight.grad is not None
    assert model[2].weight.grad is not None
    assert model[0].bias.grad is not None

    # An optimizer step mutates the shared storage in place.
    opt = tp.optim.SGD(model.parameters(), lr=0.01)
    before = model[0].weight._data.clone()
    opt.step()
    assert not tp.allclose(before, model[0].weight._data)
    tp.testing.assert_close(model[0].weight._data, model[0].weight.detach(), atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_optimizer_inplace_stays_in_sync(device):
    w = tp.randn(6, 10, device=device)
    bitnet_weight = BitNetTrainingLinearWeight(w.detach())
    opt = tp.optim.SGD([bitnet_weight], lr=0.1)

    x = tp.randn(2, 10, device=device)
    out = F.linear(x, bitnet_weight)
    out.square().mean().backward()
    opt.step()

    # In-place ``add_`` on the parameter updates the shared storage, so the
    # float view used for quantization matches the parameter's own values.
    tp.testing.assert_close(bitnet_weight._data, bitnet_weight.detach(), atol=0, rtol=0)


def test_quantize_then_module_apply_roundtrip():
    model = nn.Linear(4, 6)
    quantize_(model, bitnet_training())

    # dtype transfer through Module._apply must keep _data in sync.
    model = model.to(tp.float64)
    assert model.weight.dtype == tp.float64
    assert model.weight._data.dtype == tp.float64
    x = tp.randn(2, 4, dtype=tp.float64)
    out = model(x)
    assert tuple(out.shape) == (2, 6)
    assert out.dtype == tp.float64

    model = model.to(tp.float32)
    assert model.weight.dtype == tp.float32
    assert model.weight._data.dtype == tp.float32
    tp.testing.assert_close(model.weight._data, model.weight.detach(), atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_bitnet_training_end_to_end(device):
    model = nn.Linear(8, 8).to(device)
    quantize_(model, bitnet_training())
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


@pytest.mark.parametrize('device', DEVICES)
def test_packed_weight_linear_matches_reference(device):
    tp.manual_seed(1)
    in_features, out_features = 12, 7
    w = tp.randn(out_features, in_features, device=device)
    scale = w.abs().mean().clamp(min=1e-5)
    packed = pack_ternary((w / scale).round().clamp(-1, 1).to(tp.int8))

    packed_weight = BitNetPacked2bitLinearWeight(packed, scale)
    assert tuple(packed_weight.shape) == (out_features, in_features)
    assert packed_weight.dtype == scale.dtype

    x = tp.randn(3, in_features, device=device, requires_grad=True)
    b = tp.randn(out_features, device=device)
    out = F.linear(x, packed_weight, b)
    ref = bitnet_linear_packed(x, packed, scale, in_features, b)
    tp.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    # Activation gradient flows through the packed path (STE formula).
    out.sum().backward()
    codes = unpack_ternary(packed, in_features)
    grad_expected = (tp.ones_like(out) @ codes.to(tp.float32)) * scale
    tp.testing.assert_close(x.grad, grad_expected, atol=1e-5, rtol=1e-5)

    # dequantize matches the unpacked ternary weight.
    dq = packed_weight.dequantize()
    tp.testing.assert_close(dq, codes.to(scale.dtype) * scale, atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_fsdp_hooks_roundtrip(device):
    w = tp.randn(8, 16, device=device)
    bitnet_weight = BitNetTrainingLinearWeight(w.detach())

    packed, metadata = bitnet_weight.fsdp_pre_all_gather(None)
    (packed_tensor,) = packed
    (scale,) = metadata
    assert packed_tensor.shape[-1] == 16 // 4
    assert scale.dtype == tp.float32

    param_dtype = tp.float32
    result = bitnet_weight.fsdp_post_all_gather(packed, metadata, param_dtype)
    rebuilt, inner = result
    assert isinstance(rebuilt, BitNetPacked2bitLinearWeight)
    assert len(inner) == 1

    # The rebuilt packed weight dequantizes to the ternary-quantized weight.
    codes = unpack_ternary(packed_tensor, 16)
    tp.testing.assert_close(rebuilt.dequantize(), codes.to(tp.float32) * scale,
                            atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_data_property_sync(device):
    w = tp.randn(4, 6, device=device)
    bitnet_weight = BitNetTrainingLinearWeight(w.detach())
    assert bitnet_weight.data is bitnet_weight._data

    new_data = tp.randn(4, 6, device=device)
    bitnet_weight.data = new_data
    assert bitnet_weight._data is new_data
    tp.testing.assert_close(bitnet_weight.detach(), new_data, atol=0, rtol=0)


def test_tensor_flatten_unflatten_roundtrip():
    w = tp.randn(4, 8)
    scale = w.abs().mean().clamp(min=1e-5)
    bitnet_weight = BitNetTrainingLinearWeight(w.detach())
    attrs, ctx = bitnet_weight.__tensor_flatten__()
    inner = {name: getattr(bitnet_weight, name) for name in attrs}
    rebuilt = BitNetTrainingLinearWeight.__tensor_unflatten__(inner, ctx)
    tp.testing.assert_close(rebuilt._data, w, atol=0, rtol=0)

    packed = pack_ternary((w / scale).round().clamp(-1, 1).to(tp.int8))
    packed_weight = BitNetPacked2bitLinearWeight(packed, scale)
    attrs, ctx = packed_weight.__tensor_flatten__()
    inner = {name: getattr(packed_weight, name) for name in attrs}
    rebuilt_packed = BitNetPacked2bitLinearWeight.__tensor_unflatten__(inner, ctx)
    tp.testing.assert_close(rebuilt_packed.dequantize(), packed_weight.dequantize(),
                            atol=0, rtol=0)


def test_precompute_scale_without_dist():
    model = nn.Linear(4, 4)
    quantize_(model, bitnet_training())
    precompute_bitnet_scale_for_fsdp(model)
    param = model.weight
    assert param._precomputed_scale is not None
    tp.testing.assert_close(param._precomputed_scale, param._data.abs().mean(),
                            atol=1e-6, rtol=1e-6)


def test_quantize_requires_config():
    model = nn.Linear(4, 4)
    with pytest.raises(TypeError):
        quantize_(model, object())


def test_quantize_custom_filter():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    quantize_(model, BitNetTrainingConfig(), filter_fn=lambda m, name: name.endswith("1"))
    assert not isinstance(model[0].weight, BitNetTrainingLinearWeight)
    assert isinstance(model[1].weight, BitNetTrainingLinearWeight)