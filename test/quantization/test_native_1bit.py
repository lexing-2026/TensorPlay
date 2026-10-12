"""Native 1-bit packed affine quantization and BitNet b1.58 ternary quantization.

The 1-bit dtype ``quint1x8`` packs eight logical elements per storage byte.
Reference semantics follow torchao's ``quantize_affine``/``dequantize_affine``
with ``torch.uint1`` (values {0, 1} scaled by an affine grid). The BitNet
section checks ternary codes {-1, 0, 1}, two-bit packing and the native linear
path against a straightforward float reference.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import tensorplay as tp

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])

REFERENCE_Q = pytest.importorskip('torchao.quantization.quant_primitives')


def torchao_quantize_affine_uint1(x, scale, zp, block_size=None):
    """Reference: torchao affine quantization to uint1 (unpacked {0, 1})."""
    import torch
    if block_size is None:
        block_size = tuple(x.shape)
    q = REFERENCE_Q.quantize_affine(
        torch.tensor(x, dtype=torch.float32), block_size,
        torch.tensor(scale), torch.tensor(zp), torch.uint1)
    return q


def torchao_dequantize_affine_uint1(q, scale, zp, block_size=None):
    import torch
    if block_size is None:
        block_size = tuple(q.shape)
    return REFERENCE_Q.dequantize_affine(
        torch.tensor(q, dtype=torch.uint8), block_size,
        torch.tensor(scale), torch.tensor(zp), torch.uint1)


def expected_packed_bytes(codes_01):
    """Pack a flat {0, 1} code array into bytes, least significant code first."""
    codes = np.asarray(codes_01, dtype=np.uint8).reshape(-1)
    bytes_count = (codes.size + 7) // 8
    packed = np.zeros(bytes_count, dtype=np.uint8)
    for i, code in enumerate(codes):
        packed[i // 8] |= (code & 1) << (i % 8)
    return packed


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', [(1,), (3,), (8,), (9,), (2, 3, 5), (2, 3, 2, 5)])
@pytest.mark.parametrize('zp', [0, 1])
def test_one_bit_per_tensor_affine_rounding_tail_and_grid(device, shape, zp):
    rng = np.random.default_rng(3101)
    count = int(np.prod(shape))
    values = rng.uniform(-2, 5, size=count).astype('float32')
    values[:min(count, 8)] = np.array(
        [-.5, .5, 1.5, 2.5, 3.5, 4.5, 7.5, 15.5], dtype='float32')[:min(count, 8)] * .25
    values = values.reshape(shape)
    actual = tp.quantize_per_tensor(tp.tensor(values, device=device), .25, zp, tp.quint1x8)
    assert actual.dtype == tp.quint1x8 and tuple(actual.shape) == shape
    assert actual.q_scale() == .25 and actual.q_zero_point() == zp
    reference_codes = torchao_quantize_affine_uint1(values, .25, zp)
    expected_real = torchao_dequantize_affine_uint1(
        reference_codes.numpy(), .25, zp).numpy()
    np.testing.assert_array_equal(actual.dequantize().cpu().numpy(), expected_real)
    np.testing.assert_array_equal(
        actual.int_repr().cpu().numpy(),
        expected_packed_bytes(reference_codes.numpy().reshape(-1)).reshape(actual.int_repr().shape))
    for result in (actual.clone(), actual.cpu()):
        assert result.dtype == tp.quint1x8
        np.testing.assert_array_equal(result.dequantize().cpu().numpy(), expected_real)
    reshaped = actual.reshape((count,))
    assert reshaped.dtype == tp.quint1x8
    np.testing.assert_array_equal(reshaped.dequantize().cpu().numpy(), expected_real.reshape(-1))
    destination = tp._C._empty_affine_quantized(shape, dtype=tp.quint1x8, device=device,
                                                scale=.5, zero_point=0)
    destination.copy_(actual)
    assert destination.q_scale() == .25 and destination.q_zero_point() == zp


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', [(3, 5, 7), (3, 0, 7), (0, 5, 7)])
def test_one_bit_float_channel_grid_and_checkpoint(device, shape):
    rng = np.random.default_rng(723)
    values = rng.normal(size=shape).astype('float32')
    axis = 1
    scales = np.linspace(.07, .43, values.shape[axis], dtype='float32')
    zeros = np.linspace(-1.25, 4.75, values.shape[axis], dtype='float32')
    actual = tp.quantize_per_channel(tp.tensor(values, device=device),
                                    tp.tensor(scales, device=device),
                                    tp.tensor(zeros, device=device), axis, tp.quint1x8)
    assert actual.qscheme() == tp.per_channel_affine_float_qparams
    dequantized = actual.dequantize()
    reference = np.empty_like(dequantized.cpu().numpy())
    if values.shape[axis] and values.shape[0]:
        # Reference: per-channel affine with float zero points. The native
        # sub-byte path rounds (x/scale + zero_point) as a single value.
        scale_b = scales.reshape([values.shape[axis] if d == axis else 1 for d in range(values.ndim)])
        zero_b = zeros.reshape([values.shape[axis] if d == axis else 1 for d in range(values.ndim)])
        codes = np.clip(np.round(values / scale_b + zero_b), 0, 1)
        reference = (codes - zero_b) * scale_b
    np.testing.assert_array_equal(dequantized.cpu().numpy(), reference)
    stream = io.BytesIO()
    tp.save({'weight': actual}, stream)
    stream.seek(0)
    loaded = tp.load(stream)['weight']
    assert loaded.dtype == tp.quint1x8 and tuple(loaded.shape) == tuple(actual.shape)
    assert loaded.q_per_channel_axis() == axis
    np.testing.assert_array_equal(loaded.dequantize().cpu().numpy(), dequantized.cpu().numpy())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape,block_size', [
    ((2, 8), (2, 8)),
    ((2, 8), (1, 4)),
    ((2, 8), (2, 2)),
    ((3, 5), (3, 5)),
    ((3, 5), (1, 5)),
    ((3, 5), (3, 1)),
    ((4, 4), (4, 4)),
    ((4, 4), (1, 4)),
    ((4, 4), (2, 2)),
    ((2, 3, 4), (2, 3, 4)),
    ((2, 3, 4), (1, 1, 4)),
    ((2, 3, 4), (2, 3, 2)),
])
def test_one_bit_quantize_affine_matches_reference(device, shape, block_size):
    rng = np.random.default_rng(4102)
    values = rng.uniform(-3, 3, size=shape).astype('float32')
    blocks = tuple((s + b - 1) // b for s, b in zip(shape, block_size))
    scale = .37
    zp = 0.0
    if blocks == tuple(shape):
        scales = tp.tensor(scale, device=device)
        zeros = tp.tensor(zp, device=device)
        reference_scales = torch.tensor(scale)
        reference_zeros = torch.tensor(zp)
    else:
        scales = tp.full(blocks, scale, dtype=tp.float32, device=device)
        zeros = tp.zeros(blocks, dtype=tp.float32, device=device)
        reference_scales = torch.full(blocks, scale)
        reference_zeros = torch.zeros(blocks)
    actual = tp.quantize_affine(tp.tensor(values, device=device), block_size,
                               scales, zeros, tp.quint1x8, 0, 1)
    reference_codes = REFERENCE_Q.quantize_affine(
        torch.tensor(values), block_size, reference_scales, reference_zeros, torch.uint1)
    expected_real = REFERENCE_Q.dequantize_affine(
        reference_codes, block_size, reference_scales, reference_zeros, torch.uint1).numpy()
    # quantize_affine returns raw packed codes; dequantize_affine decodes them.
    dequantized = tp.dequantize_affine(actual, block_size, scales, zeros)
    np.testing.assert_array_equal(dequantized.cpu().numpy(), expected_real)
    # Compare the packed storage bytes against the reference codes.
    raw = tp.empty((actual.untyped_storage().nbytes(),), dtype=tp.uint8, device=device)
    raw.set_(actual.untyped_storage(), 0, (actual.untyped_storage().nbytes(),), (1,))
    expected_bytes = expected_packed_bytes(reference_codes.numpy().reshape(-1))
    np.testing.assert_array_equal(raw.cpu().numpy()[:expected_bytes.size], expected_bytes)
    # Round trip through tp.dequantize_affine as well
    round_trip = tp.dequantize_affine(actual, block_size, scales, zeros)
    np.testing.assert_array_equal(round_trip.cpu().numpy(), expected_real)


@pytest.mark.parametrize('device', DEVICES)
def test_one_bit_serialization_roundtrip(device):
    values = np.random.default_rng(502).normal(size=(3, 5)).astype('float32')
    tensor = tp.quantize_per_tensor(tp.tensor(values, device=device), .3, 1, tp.quint1x8)
    stream = io.BytesIO()
    tp.save(tensor, stream)
    stream.seek(0)
    loaded = tp.load(stream)
    assert loaded.dtype == tp.quint1x8
    np.testing.assert_array_equal(loaded.int_repr().cpu().numpy(), tensor.int_repr().cpu().numpy())
    np.testing.assert_array_equal(loaded.dequantize().cpu().numpy(), tensor.dequantize().cpu().numpy())


@pytest.mark.parametrize('device', DEVICES)
def test_one_bit_channels_last_layout_and_checkpoint(device):
    shape = (2, 3, 2, 5)
    values = np.random.default_rng(510).uniform(-1, 4, size=shape).astype('float32')
    native = tp.tensor(values, device=device).contiguous(memory_format=tp.channels_last)
    actual = tp.quantize_per_tensor(native, .25, 1, tp.quint1x8)
    expected_real = torchao_dequantize_affine_uint1(
        torchao_quantize_affine_uint1(values, .25, 1).numpy(), .25, 1).numpy()
    stream = io.BytesIO()
    tp.save(actual, stream)
    stream.seek(0)
    for result in (actual, actual.clone(), actual.cpu(), tp.load(stream)):
        assert tuple(result.stride()) == tuple(actual.stride())
        np.testing.assert_array_equal(result.dequantize().cpu().numpy(), expected_real)


# ---------------------------------------------------------------------------
# BitNet b1.58 ternary quantization
# ---------------------------------------------------------------------------

def manual_bitnet_linear(x, weight, scale=None):
    """Straightforward float reference for the BitNet b1.58 linear."""
    weight = np.asarray(weight, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if scale is None:
        scale = np.abs(weight).mean()
    codes_w = np.clip(np.round(weight / scale), -1, 1).astype(np.int32)
    absmax = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-5)
    row_scale = absmax / 127.0
    codes_x = np.clip(np.round(x / row_scale), -127, 127).astype(np.int32)
    return (codes_x @ codes_w.T) * row_scale * scale


def test_bitnet_scale_and_ternary_codes():
    from tensorplay.ao.quantization.bitnet import get_bitnet_scale, quantize_bitnet_weight
    weight = tp.tensor([[.4, -.1, .2, -.3],
                        [.0, .6, -.8, .1]], dtype=tp.float32)
    scale = get_bitnet_scale(weight)
    np.testing.assert_allclose(scale.item(), np.abs(weight.numpy()).mean(), rtol=1e-6)
    codes, scale2 = quantize_bitnet_weight(weight, scale)
    assert codes.dtype == tp.int8
    assert set(np.unique(codes.numpy()).tolist()) <= {-1, 0, 1}
    reference = np.clip(np.round(np.asarray(weight, dtype=np.float32) / scale.item()), -1, 1)
    np.testing.assert_array_equal(codes.numpy(), reference)


def test_bitnet_pack_unpack_roundtrip():
    from tensorplay.ao.quantization.bitnet import pack_ternary, unpack_ternary
    codes = tp.tensor([[1, -1, 0, 1, -1, 0, 0, 1, -1],
                       [0, 1, -1, -1, 1, 0, 1, -1, 0]], dtype=tp.int8)
    packed = pack_ternary(codes)
    assert packed.dtype == tp.uint8
    assert packed.shape[-1] == (codes.shape[-1] + 3) // 4
    restored = unpack_ternary(packed, codes.shape[-1])
    np.testing.assert_array_equal(restored.numpy(), codes.numpy())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', [(4, 8), (3, 5), (2, 3, 4)])
def test_bitnet_linear_matches_float_reference(device, shape):
    from tensorplay.ao.quantization.bitnet import bitnet_linear
    rng = np.random.default_rng(611)
    x = rng.normal(size=shape).astype('float32')
    weight = rng.normal(size=(6, shape[-1])).astype('float32')
    actual = bitnet_linear(tp.tensor(x, device=device), tp.tensor(weight, device=device))
    expected = manual_bitnet_linear(x, weight)
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_bitnet_linear_with_bias_and_scale(device):
    from tensorplay.ao.quantization.bitnet import bitnet_linear, get_bitnet_scale
    rng = np.random.default_rng(612)
    x = rng.normal(size=(4, 8)).astype('float32')
    weight = rng.normal(size=(6, 8)).astype('float32')
    bias = rng.normal(size=(6,)).astype('float32')
    scale = get_bitnet_scale(tp.tensor(weight, device=device))
    actual = bitnet_linear(tp.tensor(x, device=device), tp.tensor(weight, device=device),
                          bias=tp.tensor(bias, device=device), scale=scale)
    expected = manual_bitnet_linear(x, weight, scale.item()) + bias
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


def test_bitnet_linear_module():
    from tensorplay.ao.quantization.bitnet import BitNetLinear
    module = BitNetLinear(8, 6)
    rng = np.random.default_rng(613)
    x = tp.tensor(rng.normal(size=(2, 3, 8)).astype('float32'))
    y = module(x)
    assert tuple(y.shape) == (2, 3, 6)