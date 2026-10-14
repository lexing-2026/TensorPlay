"""Native FP8 affine quantization primitives.

Covers ``quantize_affine_float8``, ``dequantize_affine_float8`` and
``choose_scale_float8`` on CPU and CUDA: all supported float8 dtypes, the
scale broadcast rules (per-tensor, per-channel, per-group, per-element),
block-scale selection and round-trip error bounds.

Float8 formats have a fixed small mantissa, so round-trip error is bounded
*relative* to the value magnitude: e4m3fn/e4m3fnuz carry 3 mantissa bits
(relative error ~6.25%), e5m2/e5m2fnuz carry 2 mantissa bits (relative error
~25%).  All assertions below use a relative bound with a tiny absolute floor.
"""

import numpy as np
import pytest
import tensorplay as tp

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])

# (public dtype attr, max value, relative round-trip tolerance)
FLOAT8_DTYPES = [
    ('float8_e4m3fn', 448.0, 0.1),
    ('float8_e4m3fnuz', 240.0, 0.1),
    ('float8_e5m2', 57344.0, 0.3),
    ('float8_e5m2fnuz', 57344.0, 0.3),
]


def _make_input(shape, device, seed=1234):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 2.0, size=shape).astype('float32')
    return tp.tensor(x, device=device)


def _assert_roundtrip(dq, x, rel_tol, abs_floor=1e-4):
    diff = (dq - x).abs()
    bound = rel_tol * x.abs() + abs_floor
    assert bool((diff <= bound).all()), (
        f"round-trip violated rel_tol={rel_tol}: max diff {diff.max().item():.4f}, "
        f"max bound {bound.max().item():.4f}"
    )


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype_name,maxv,rel_tol', FLOAT8_DTYPES)
def test_float8_roundtrip_per_tensor(device, dtype_name, maxv, rel_tol):
    dtype = getattr(tp, dtype_name)
    x = _make_input((64, 32), device)
    scale = tp.tensor(0.05, device=device)
    q = tp._C.quantize_affine_float8(x, scale, dtype)
    assert q.dtype == dtype
    dq = tp._C.dequantize_affine_float8(q, scale, tp.float32)
    _assert_roundtrip(dq, x, rel_tol)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype_name,maxv,rel_tol', FLOAT8_DTYPES)
def test_float8_clamp_at_max(device, dtype_name, maxv, rel_tol):
    dtype = getattr(tp, dtype_name)
    x = tp.tensor([[1e6, -1e6, 1.0]], device=device)
    scale = tp.tensor(1.0, device=device)
    q = tp._C.quantize_affine_float8(x, scale, dtype)
    dq = tp._C.dequantize_affine_float8(q, scale, tp.float32)
    vals = dq.reshape(-1).tolist()
    # Values beyond the format max must clamp onto the max representable
    # value (the finite max itself is exactly representable).
    assert vals[0] == pytest.approx(maxv, rel=1e-3)
    assert vals[1] == pytest.approx(-maxv, rel=1e-3)
    assert abs(vals[2] - 1.0) < rel_tol


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype_name,maxv,rel_tol', FLOAT8_DTYPES)
def test_float8_per_channel_scale(device, dtype_name, maxv, rel_tol):
    dtype = getattr(tp, dtype_name)
    x = _make_input((8, 32), device)
    # Per-row scales, broadcast as (8, 1).  The smallest scale keeps
    # x/scale below the float8 max for every supported dtype.
    row_scale = tp.tensor(np.linspace(0.05, 0.2, 8).reshape(8, 1), device=device)
    q = tp._C.quantize_affine_float8(x, row_scale, dtype)
    assert q.shape == x.shape
    dq = tp._C.dequantize_affine_float8(q, row_scale, tp.float32)
    _assert_roundtrip(dq, x, rel_tol)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype_name,maxv,rel_tol', FLOAT8_DTYPES)
def test_float8_per_group_scale(device, dtype_name, maxv, rel_tol):
    dtype = getattr(tp, dtype_name)
    x = _make_input((4, 64), device, seed=0)
    # Group size 16 along the last dim -> scale shape (4, 4).
    scale = tp.tensor(
        np.abs(np.random.default_rng(0).normal(0.05, 0.01, (4, 4))),
        device=device)
    q = tp._C.quantize_affine_float8(x, scale, dtype)
    dq = tp._C.dequantize_affine_float8(q, scale, tp.float32)
    _assert_roundtrip(dq, x, rel_tol)


@pytest.mark.parametrize('device', DEVICES)
def test_float8_per_element_scale(device):
    x = _make_input((16, 16), device, seed=1)
    scale = tp.tensor(
        np.abs(np.random.default_rng(1).normal(0.05, 0.01, (16, 16))),
        device=device)
    q = tp._C.quantize_affine_float8(x, scale, tp.float8_e4m3fn)
    dq = tp._C.dequantize_affine_float8(q, scale, tp.float32)
    _assert_roundtrip(dq, x, 0.1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype_name,maxv,rel_tol', FLOAT8_DTYPES)
def test_choose_scale_float8_matches_manual(device, dtype_name, maxv, rel_tol):
    x = _make_input((8, 64), device, seed=2)
    block_size = [1, 16]
    scale = tp._C.choose_scale_float8(x, block_size, getattr(tp, dtype_name))
    assert scale.shape == (8, 4)
    # Manual reference: per-block max-abs divided by the float8 max, so that
    # x/scale saturates exactly at the format maximum.
    xa = np.abs(x.cpu().numpy() if device == 'cuda' else x.numpy())
    ref = np.zeros((8, 4), dtype='float32')
    for i in range(8):
        for j in range(4):
            ref[i, j] = xa[i, j * 16:(j + 1) * 16].max() / maxv
    np.testing.assert_allclose(
        scale.cpu().numpy() if device == 'cuda' else scale.numpy(),
        ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('device', DEVICES)
def test_choose_scale_float8_block_rank_must_match(device):
    x = _make_input((8, 64), device)
    with pytest.raises(ValueError):
        tp._C.choose_scale_float8(x, [16], tp.float8_e4m3fn)


@pytest.mark.parametrize('device', DEVICES)
def test_float8_scale_must_be_on_input_device(device):
    x = _make_input((8, 8), device)
    other = 'cpu' if device != 'cpu' else 'cuda'
    scale = tp.tensor(0.05, device=other)
    with pytest.raises(RuntimeError):
        tp._C.quantize_affine_float8(x, scale, tp.float8_e4m3fn)


@pytest.mark.parametrize('device', DEVICES)
def test_float8_rejects_non_float8_dtype(device):
    x = _make_input((8, 8), device)
    with pytest.raises(TypeError):
        tp._C.quantize_affine_float8(x, tp.tensor(0.05, device=device), tp.int8)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype_name,maxv,rel_tol', FLOAT8_DTYPES)
def test_float8_dequantize_output_dtypes(device, dtype_name, maxv, rel_tol):
    x = _make_input((16, 16), device, seed=3)
    scale = tp.tensor(0.05, device=device)
    q = tp._C.quantize_affine_float8(x, scale, getattr(tp, dtype_name))
    for out_dtype in (tp.float32, tp.float16, tp.bfloat16):
        dq = tp._C.dequantize_affine_float8(q, scale, out_dtype)
        assert dq.dtype == out_dtype
        _assert_roundtrip(dq.float(), x, rel_tol, abs_floor=0.02)


@pytest.mark.parametrize('device', DEVICES)
def test_float8_empty_input(device):
    x = tp.empty((0, 8), device=device)
    scale = tp.tensor(0.05, device=device)
    q = tp._C.quantize_affine_float8(x, scale, tp.float8_e4m3fn)
    assert q.shape == (0, 8)
    dq = tp._C.dequantize_affine_float8(q, scale, tp.float32)
    assert dq.shape == (0, 8)


@pytest.mark.parametrize('device', DEVICES)
def test_float8_noncontiguous_input(device):
    x = _make_input((16, 32), device, seed=4)
    x = x.transpose(0, 1)  # (32, 16) non-contiguous view
    scale = tp.tensor(0.05, device=device)
    q = tp._C.quantize_affine_float8(x, scale, tp.float8_e4m3fn)
    assert q.shape == (32, 16)
    dq = tp._C.dequantize_affine_float8(q, scale, tp.float32)
    _assert_roundtrip(dq, x, 0.1)