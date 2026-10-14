"""FP8 weight-only and mixed-precision training modules.

Covers the ``float8_weight_only()`` and
``float8_dynamic_activation_float8_weight()`` entry points built on native
``quantize_affine_float8`` / ``choose_scale_float8`` primitives: forward
matching a manual reference, straight-through training (loss decreases),
dtype options and CUDA/CPU consistency.
"""

import numpy as np
import pytest
import tensorplay as tp
from tensorplay import nn

from tensorplay.ao.quantization import (
    Float8MixedPrecisionTrainingLinear,
    Float8WeightOnlyQuantizedTrainingLinear,
    float8_dynamic_activation_float8_weight,
    float8_weight_only,
    quantize_,
)

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])


def manual_fp8_wo_linear(x, w, b=None, dtype='float8_e4m3fn'):
    fp8 = getattr(tp, dtype)
    scale = tp._C.choose_scale_float8(w, [1, w.shape[1]], fp8).clamp(min=1e-12)
    q = tp._C.quantize_affine_float8(w, scale, fp8)
    dq = tp._C.dequantize_affine_float8(q, scale, x.dtype)
    out = x @ dq.T
    if b is not None:
        out = out + b
    return out


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', ['float8_e4m3fn', 'float8_e5m2'])
def test_weight_only_forward_matches_manual(device, dtype):
    tp.manual_seed(0)
    x = tp.randn(4, 16, device=device)
    w = tp.randn(8, 16, device=device)
    b = tp.randn(8, device=device)
    lin = Float8WeightOnlyQuantizedTrainingLinear(16, 8, bias=True, dtype=dtype).to(device)
    lin.weight = nn.Parameter(w)
    lin.bias = nn.Parameter(b)
    out = lin(x)
    ref = manual_fp8_wo_linear(x, w, b, dtype)
    np.testing.assert_allclose(
        out.detach().cpu().numpy(), ref.detach().cpu().numpy(), rtol=0.2, atol=0.2)


@pytest.mark.parametrize('device', DEVICES)
def test_quantize_weight_only_converts_model(device):
    tp.manual_seed(1)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)
    x = tp.randn(3, 16, device=device)
    ref = model(x)
    quantize_(model, float8_weight_only())
    assert isinstance(model[0], Float8WeightOnlyQuantizedTrainingLinear)
    out = model(x)
    # Weight-only quantization should stay close to the float reference.
    err = (out - ref).abs().max().item()
    assert err < 0.5, err


@pytest.mark.parametrize('device', DEVICES)
def test_weight_only_training_loss_decreases(device):
    tp.manual_seed(2)
    lin = Float8WeightOnlyQuantizedTrainingLinear(8, 4, dtype='float8_e4m3fn').to(device)
    opt = tp.optim.SGD(lin.parameters(), lr=0.1)
    x = tp.randn(16, 8, device=device)
    target = tp.randn(16, 4, device=device)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        out = lin(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8, losses


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', ['float8_e4m3fn', 'float8_e5m2'])
def test_mixed_precision_forward(device, dtype):
    tp.manual_seed(3)
    x = tp.randn(4, 16, device=device)
    lin = Float8MixedPrecisionTrainingLinear(16, 8, bias=True, dtype=dtype).to(device)
    out = lin(x)
    assert out.shape == (4, 8)
    assert out.dtype == x.dtype
    # Activations and weights are both FP8-quantized; output should be a
    # reasonable approximation of the float linear.
    ref = x @ lin.weight.detach().T + lin.bias
    err = (out - ref).abs().max().item()
    assert err < 1.0, err


@pytest.mark.parametrize('device', DEVICES)
def test_mixed_precision_training_loss_decreases(device):
    tp.manual_seed(4)
    lin = Float8MixedPrecisionTrainingLinear(8, 4).to(device)
    opt = tp.optim.SGD(lin.parameters(), lr=0.1)
    x = tp.randn(16, 8, device=device)
    target = tp.randn(16, 4, device=device)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        out = lin(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8, losses


@pytest.mark.parametrize('device', DEVICES)
def test_quantize_mixed_precision_converts_model(device):
    tp.manual_seed(5)
    model = nn.Linear(12, 6, bias=True).to(device)
    quantize_(model, float8_dynamic_activation_float8_weight())
    assert isinstance(model, Float8MixedPrecisionTrainingLinear)
    out = model(tp.randn(3, 12, device=device))
    assert out.shape == (3, 6)


@pytest.mark.parametrize('device', DEVICES)
def test_gradients_flow_to_full_precision_weight(device):
    tp.manual_seed(6)
    lin = Float8WeightOnlyQuantizedTrainingLinear(6, 3).to(device)
    x = tp.randn(5, 6, device=device)
    out = lin(x)
    out.sum().backward()
    assert lin.weight.grad is not None
    assert lin.weight.grad.abs().sum().item() > 0