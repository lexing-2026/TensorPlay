"""Numerical comparison tests for the numpy compatibility layer.

Every functional result is compared against the real numpy ground truth.
"""

import math

import numpy as real_np
import pytest

import tensorplay as tp
import tensorplay._numpy as np


def _close(got, want, rtol=1e-5, atol=1e-7):
    got = real_np.asarray(got.tolist() if hasattr(got, "tolist") else got)
    want = real_np.asarray(want.tolist() if hasattr(want, "tolist") else want)
    return real_np.allclose(got, want, rtol=rtol, atol=atol, equal_nan=True)


def test_array_creation_matches_numpy():
    for spec, dtype, real_dtype in [
        ([[1.0, 2.0], [3.0, 4.0]], None, None),
        ([[1, -2, 3]], np.int32, real_np.int32),
        ([0.5, 1.5], np.float32, real_np.float32),
    ]:
        got = np.array(spec) if dtype is None else np.array(spec, dtype=dtype)
        want = (real_np.array(spec) if real_dtype is None
                else real_np.array(spec, dtype=real_dtype))
        assert _close(got, want)


def test_arange_linspace():
    assert _close(np.arange(10), real_np.arange(10))
    assert _close(np.arange(2.0, 10.0, 0.5), real_np.arange(2.0, 10.0, 0.5))
    assert _close(np.linspace(0, 1, 7), real_np.linspace(0, 1, 7))


def test_ufunc_math():
    a = np.array([0.1, 0.7, 1.4, -2.0])
    b = real_np.array([0.1, 0.7, 1.4, -2.0])
    for name in ("abs", "sqrt", "exp", "log", "sin", "cos", "tan",
                 "floor", "ceil", "expm1", "log1p", "tanh", "reciprocal"):
        got = getattr(np, name)(a)
        want = getattr(real_np, name)(b)
        assert _close(got, want), name


def test_binary_ufuncs():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = real_np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _close(a + a, b + b)
    assert _close(a - 0.5, b - 0.5)
    assert _close(a * a, b * b)
    assert _close(a / 2.0, b / 2.0)
    assert _close(a ** 3, b ** 3)
    assert _close(np.maximum(a, 2.0), real_np.maximum(b, 2.0))
    assert _close(np.minimum(a, 2.5), real_np.minimum(b, 2.5))


def test_reductions():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = real_np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for fn in (np.sum, np.prod, np.min, np.max, np.mean, np.var, np.std):
        assert _close(fn(a), fn(b)), fn.__name__
        assert _close(fn(a, axis=0), fn(b, axis=0)), f"{fn.__name__} axis=0"
        assert _close(fn(a, axis=1), fn(b, axis=1)), f"{fn.__name__} axis=1"
    assert np.argmax(a, axis=1).tolist() == real_np.argmax(b, axis=1).tolist()
    assert np.argmin(a, axis=0).tolist() == real_np.argmin(b, axis=0).tolist()


def test_sum_axis_alias():
    # numpy-compatibility keyword spelling resolved by the binding layer
    a = np.arange(12).reshape(3, 4)
    b = real_np.arange(12).reshape(3, 4)
    assert _close(a.sum(axis=1), b.sum(axis=1))
    assert _close(np.sum(a, axis=0), real_np.sum(b, axis=0))


def test_matmul_and_linalg():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = real_np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _close(a @ a.T, b @ b.T)
    assert _close(np.matmul(a, a), real_np.matmul(b, b))

    m = np.array([[2.0, 1.0], [1.0, 3.0]])
    rhs = np.array([3.0, 5.0])
    m_np = real_np.array([[2.0, 1.0], [1.0, 3.0]])
    rhs_np = real_np.array([3.0, 5.0])
    assert _close(np.linalg.solve(m, rhs), real_np.linalg.solve(m_np, rhs_np))
    assert _close(np.linalg.inv(m), real_np.linalg.inv(m_np))
    assert _close(np.linalg.norm(rhs), real_np.linalg.norm(rhs_np))


def test_fft_matches_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    x_np = real_np.array([1.0, 2.0, 3.0, 4.0])
    assert _close(np.fft.fft(x), real_np.fft.fft(x_np))
    assert _close(np.fft.ifft(x), real_np.fft.ifft(x_np))
    assert _close(np.fft.rfft(x), real_np.fft.rfft(x_np))


def test_random_produces_valid_values():
    values = np.random.rand(3, 4)
    flat = real_np.asarray(values.tolist())
    assert flat.shape == (3, 4)
    assert ((flat >= 0.0) & (flat < 1.0)).all()


def test_dtype_promotion():
    # int32 and float32 cannot represent each other exactly; both layers
    # promote to float64
    assert np.result_type(np.int32, np.float32) == np.float64
    assert real_np.result_type(real_np.int32, real_np.float32) == real_np.float64
    assert bool(np.can_cast(np.int32, np.float64)) == real_np.can_cast(
        real_np.int32, real_np.float64)
    assert bool(np.can_cast(np.float64, np.int32, casting="safe")) == (
        real_np.can_cast(real_np.float64, real_np.int32, casting="safe"))


def test_ndarray_methods_and_indexing():
    a = np.arange(24).reshape(2, 3, 4)
    b = real_np.arange(24).reshape(2, 3, 4)
    assert _close(a.T, b.T)
    assert _close(a.transpose(1, 0, 2), b.transpose(1, 0, 2))
    assert _close(a[1, :, ::2], b[1, :, ::2])
    assert _close(a.reshape(4, 6), b.reshape(4, 6))
    assert a.tolist() == b.tolist()


def test_concatenate_and_stack():
    a = np.ones((2, 2))
    b = real_np.ones((2, 2))
    assert _close(np.concatenate([a, a], axis=1), real_np.concatenate([b, b], axis=1))
    assert _close(np.stack([a, a]), real_np.stack([b, b]))
    assert _close(np.concatenate([a, 2 * a]), real_np.concatenate([b, 2 * b]))


def test_nep50_scalar_promotion():
    a = np.array([1, 2, 3], dtype=np.int64)
    b = real_np.array([1, 2, 3], dtype=real_np.int64)
    assert _close(a + 0.5, b + 0.5)
    assert (a + 0.5).dtype == np.float64
    assert real_np.result_type(b + 0.5) == real_np.float64


def test_getlimits():
    info = np.finfo(np.float32)
    assert info.max == real_np.finfo(real_np.float32).max
    iinfo = np.iinfo(np.int16)
    assert iinfo.min == real_np.iinfo(real_np.int16).min
    assert iinfo.max == real_np.iinfo(real_np.int16).max


def test_where_and_masks():
    a = np.array([-1.0, 2.0, -3.0, 4.0])
    b = real_np.array([-1.0, 2.0, -3.0, 4.0])
    assert _close(np.where(a > 0, a, 0.0), real_np.where(b > 0, b, 0.0))
    assert _close(np.clip(a, -2.0, 2.0), real_np.clip(b, -2.0, 2.0))
