"""INT8-dynamic-activation / INT2 and / 1-bit weight modules.

Covers the new low-bit tensor-core path: activations are quantized to int8
per row and weights to int2 (codes {-2,-1,0,1}) or 1-bit (codes {-1,+1}),
then multiplied through the native signed integer matmul path.
"""

import pytest

import tensorplay as tp
from tensorplay import nn

from tensorplay.ao.quantization import (
    Int8DynActBinaryWeightLinear,
    Int8DynActInt2WeightLinear,
    int8_dynamic_activation_binary_weight,
    int8_dynamic_activation_int2_weight,
    quantize_,
)

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])


def _model():
    return nn.Sequential(nn.Linear(64, 64, bias=False), nn.ReLU(), nn.Linear(64, 16, bias=False))


@pytest.mark.parametrize('device', DEVICES)
def test_8da2w_forward_close_to_float(device):
    tp.manual_seed(0)
    x = tp.randn(4, 64, device=device)
    m = _model().to(device)
    ref = m(x)
    qm = quantize_(m, int8_dynamic_activation_int2_weight())
    assert isinstance(qm[0], Int8DynActInt2WeightLinear)
    out = qm(x)
    assert out.shape == (4, 16)
    # int2 weight error is bounded; output must stay in the right ballpark.
    rel = ((out - ref).abs().mean() / ref.abs().mean()).item()
    assert rel < 1.0


@pytest.mark.parametrize('device', DEVICES)
def test_8da1w_forward_close_to_float(device):
    tp.manual_seed(0)
    x = tp.randn(4, 64, device=device)
    m = _model().to(device)
    ref = m(x)
    qm = quantize_(m, int8_dynamic_activation_binary_weight())
    assert isinstance(qm[0], Int8DynActBinaryWeightLinear)
    out = qm(x)
    assert out.shape == (4, 16)
    # 1-bit weights carry only the sign, so the output error is naturally
    # larger than int2; keep the assertion on the correlation with float.
    cos = ((out.reshape(-1) @ ref.reshape(-1)).abs() / (out.norm() * ref.norm())).item()
    assert cos > 0.5


@pytest.mark.parametrize('device', DEVICES)
def test_8da2w_training_ste_gradients(device):
    tp.manual_seed(1)
    x = tp.randn(3, 8, device=device, requires_grad=True)
    mod = Int8DynActInt2WeightLinear(8, 5).to(device)
    out = mod(x)
    assert out.shape == (3, 5)
    out.sum().backward()
    assert mod.weight.grad is not None
    assert mod.weight.grad.shape == (5, 8)
    assert x.grad is not None
    expected = tp.ones_like(out) @ mod.weight.detach()
    tp.testing.assert_close(x.grad, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('device', DEVICES)
def test_8da1w_training_ste_gradients(device):
    tp.manual_seed(2)
    x = tp.randn(3, 8, device=device, requires_grad=True)
    mod = Int8DynActBinaryWeightLinear(8, 5).to(device)
    out = mod(x)
    assert out.shape == (3, 5)
    out.sum().backward()
    assert mod.weight.grad is not None
    assert mod.weight.grad.shape == (5, 8)
    assert x.grad is not None
    expected = tp.ones_like(out) @ mod.weight.detach()
    tp.testing.assert_close(x.grad, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('device', DEVICES)
def test_lowbit_quantize_kernels_match_reference(device):
    tp.manual_seed(3)
    w = tp.randn(6, 16, device=device)
    # int2 kernel
    codes, scales = tp._C._quantize_int2_rowwise(w, 1e-8)
    scale_ref = w.abs().amax(dim=-1, keepdim=True) / 2.0
    codes_ref = (w / scale_ref).round().clamp(-2, 1)
    assert bool((codes == codes_ref).all())
    assert bool((scales.unsqueeze(-1) - scale_ref).abs().max() < 1e-6)
    # binary kernel
    codes_b, scales_b = tp._C._quantize_binary_rowwise(w, 1e-8)
    assert bool((codes_b == tp.where(w >= 0, 1, -1).to(tp.int8)).all())
    assert bool((scales_b.unsqueeze(-1) - w.abs().amax(dim=-1, keepdim=True)).abs().max() < 1e-6)