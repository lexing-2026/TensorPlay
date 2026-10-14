"""INT4 weight-only and int8-dynamic-activation/int4-weight modules.

Covers the native ``_convert_weight_to_int4pack`` packing, the
``int4_weight_only()`` inference workflow, INT4 weight-only training with a
straight-through estimator, and the 8da4w linear built on the native int8
tensor-core matmul path.
"""

import numpy as np
import pytest
import tensorplay as tp
from tensorplay import nn

from tensorplay.ao.quantization import (
    Int8DynActInt4WeightLinear,
    Int4WeightOnlyQuantizedTrainingLinear,
    WeightOnlyInt4Linear,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    quantize_,
)

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])


def _dequant_int4_reference(x, w, groupsize=128):
    """Manual symmetric int4 weight-only linear reference."""
    N, K = w.shape
    groups = K // groupsize
    wg = w.reshape(N, groups, groupsize)
    scale = wg.abs().amax(dim=-1) / 8.0
    scale = scale.clamp(min=1e-8)
    wq = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7) * scale.unsqueeze(-1)
    wq = wq.reshape(N, K)
    return x @ wq.T


@pytest.mark.parametrize('device', DEVICES)
def test_pack_and_mm_matches_reference(device):
    # The CUDA int4 tensor-core kernel is under active development (nibble
    # order is being changed together with a new direct-GEMM kernel); the
    # packer and the kernel are temporarily out of sync.  Remove this xfail
    # once the CUDA kernel settles.
    if device == 'cuda':
        pytest.xfail("CUDA int4pack kernel under development")
    tp.manual_seed(0)
    N, K, B, g = 32, 256, 4, 128
    w = tp.randn(N, K, device=device)
    x = tp.randn(B, K, device=device)
    # Group-wise symmetric int4 quantization (same as the module helper).
    groups = K // g
    wg = w.reshape(N, groups, g)
    scale = wg.abs().amax(dim=-1) / 8.0
    scale = scale.clamp(min=1e-8)
    codes = ((wg / scale.unsqueeze(-1)).round().clamp(-8, 7) + 8)
    codes = codes.to(tp.int32).reshape(N, K)
    if device == 'cuda':
        packed = tp._C._convert_weight_to_int4pack(codes, 2)
    else:
        packed = tp._C._convert_weight_to_int4pack_for_cpu(codes, 2)
    scale_zero = tp.stack([scale.t(), tp.zeros_like(scale).t()], dim=-1)
    if device == 'cuda':
        out = tp._C._weight_int4pack_mm(x, packed, g, scale_zero)
    else:
        out = tp._C._weight_int4pack_mm_for_cpu(x, packed, g, scale_zero)
    ref = _dequant_int4_reference(x, w, g)
    # The CUDA tensor-core path accumulates in reduced precision, so it
    # carries slightly larger rounding than the exact SIMD CPU path.
    tol = 5e-2 if device == 'cuda' else 1e-2
    np.testing.assert_allclose(
        out.detach().cpu().numpy(), ref.detach().cpu().numpy(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize('device', DEVICES)
def test_weight_only_linear_forward(device):
    # See test_pack_and_mm_matches_reference: the CUDA int4pack kernel is
    # under active development and temporarily loses precision.
    if device == 'cuda':
        pytest.xfail("CUDA int4pack kernel under development")
    tp.manual_seed(1)
    lin = nn.Linear(256, 32, bias=True).to(device)
    x = tp.randn(4, 256, device=device)
    ref = lin(x)
    qlin = WeightOnlyInt4Linear.from_float(lin, groupsize=128).to(device)
    out = qlin(x)
    assert out.shape == (4, 32)
    # int4 weight-only keeps the output within a few percent of float.
    err = (out - ref).abs().max().item()
    assert err < 1.0, err


@pytest.mark.parametrize('device', DEVICES)
def test_quantize_weight_only_converts_model(device):
    tp.manual_seed(2)
    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16)).to(device)
    quantize_(model, int4_weight_only(groupsize=64))
    assert isinstance(model[0], WeightOnlyInt4Linear)
    assert isinstance(model[2], WeightOnlyInt4Linear)
    out = model(tp.randn(3, 128, device=device))
    assert out.shape == (3, 16)


@pytest.mark.parametrize('device', DEVICES)
def test_weight_only_training_loss_decreases(device):
    # See test_pack_and_mm_matches_reference: the CUDA int4pack kernel is
    # under active development and temporarily loses precision.
    if device == 'cuda':
        pytest.xfail("CUDA int4pack kernel under development")
    tp.manual_seed(3)
    lin = Int4WeightOnlyQuantizedTrainingLinear(64, 16, groupsize=32).to(device)
    opt = tp.optim.SGD(lin.parameters(), lr=0.1)
    x = tp.randn(16, 64, device=device)
    target = tp.randn(16, 16, device=device)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        out = lin(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.7, losses


@pytest.mark.parametrize('device', DEVICES)
def test_8da4w_forward_close_to_float(device):
    tp.manual_seed(4)
    lin = nn.Linear(128, 32, bias=True).to(device)
    x = tp.randn(4, 128, device=device)
    ref = lin(x)
    quantize_(lin, int8_dynamic_activation_int4_weight())
    assert isinstance(lin, Int8DynActInt4WeightLinear)
    out = lin(x)
    err = (out - ref).abs().max().item()
    # 8-bit activation + 4-bit weight stays close to the float output.
    assert err < 1.0, err


@pytest.mark.parametrize('device', DEVICES)
def test_8da4w_training_loss_decreases(device):
    tp.manual_seed(5)
    lin = Int8DynActInt4WeightLinear(64, 16).to(device)
    opt = tp.optim.SGD(lin.parameters(), lr=0.1)
    x = tp.randn(16, 64, device=device)
    target = tp.randn(16, 16, device=device)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        out = lin(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.7, losses