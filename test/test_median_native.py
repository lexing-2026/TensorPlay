import numpy as np
import pytest

import tensorplay as tp


def _as_tensor(array, device=None):
    value = tp.from_numpy(np.ascontiguousarray(array))
    return value if device is None else value.to(device)


def _expected_median(array):
    flat = np.sort(np.asarray(array).reshape(-1))
    return flat[(flat.size - 1) // 2]


def _expected_nanmedian(array):
    flat = np.asarray(array).reshape(-1)
    if flat.dtype.kind == "f":
        flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return np.nan
    return np.sort(flat)[(flat.size - 1) // 2]


def _value(tensor):
    return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()


@pytest.mark.parametrize("size", [1, 2, 3, 16, 4097, 16384, 16385, 100003])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_median_flat_matches_lower_middle(size, dtype):
    rng = np.random.default_rng(size + 17)
    if np.issubdtype(dtype, np.integer):
        array = rng.integers(-50, 50, size=size).astype(dtype)
    else:
        array = rng.standard_normal(size).astype(dtype)
    got = _value(tp.median(_as_tensor(array)))
    np.testing.assert_allclose(got, _expected_median(array), rtol=0, atol=0)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [1, 2, 3, 16384, 16385, 100003, 1 << 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_median_flat_cuda_matches_lower_middle(size, dtype):
    rng = np.random.default_rng(size + 5)
    if np.issubdtype(dtype, np.integer):
        array = rng.integers(-1000, 1000, size=size).astype(dtype)
    else:
        array = rng.standard_normal(size).astype(dtype)
    got = _value(tp.median(_as_tensor(array, "cuda")))
    np.testing.assert_allclose(got, _expected_median(array), rtol=0, atol=0)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [16384, 16385, 100003, 1 << 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_median_flat_cuda_nan_orders_last(size, dtype):
    rng = np.random.default_rng(size + 9)
    array = rng.standard_normal(size).astype(dtype)
    # A negative sign on a NaN must not change where it orders.
    array[rng.integers(0, size)] = -np.nan
    assert np.isnan(_value(tp.median(_as_tensor(array, "cuda"))))


@pytest.mark.parametrize("size", [3, 16384, 16385, 100003, 1 << 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_nanmedian_flat_ignores_nan(size, dtype):
    rng = np.random.default_rng(size + 21)
    if np.issubdtype(dtype, np.integer):
        array = rng.integers(-100, 100, size=size).astype(dtype)
    else:
        array = rng.standard_normal(size).astype(dtype)
        array[rng.random(size) < 0.2] = np.nan
    got = _value(tp.nanmedian(_as_tensor(array)))
    np.testing.assert_allclose(got, _expected_nanmedian(array), rtol=0, atol=0)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [3, 16384, 16385, 100003, 1 << 20, 4 * (1 << 20)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_nanmedian_flat_cuda_ignores_nan(size, dtype):
    rng = np.random.default_rng(size + 33)
    if np.issubdtype(dtype, np.integer):
        array = rng.integers(-100, 100, size=size).astype(dtype)
    else:
        array = rng.standard_normal(size).astype(dtype)
        array[rng.random(size) < 0.2] = np.nan
    got = _value(tp.nanmedian(_as_tensor(array, "cuda")))
    np.testing.assert_allclose(got, _expected_nanmedian(array), rtol=0, atol=0)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [16385, 100003])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nanmedian_flat_cuda_signed_nan_majority(size, dtype):
    rng = np.random.default_rng(size + 41)
    array = rng.standard_normal(size).astype(dtype)
    array[rng.random(size) < 0.4] = np.nan
    array[: size // 2 + 7] = -np.nan
    got = _value(tp.nanmedian(_as_tensor(array, "cuda")))
    np.testing.assert_allclose(got, _expected_nanmedian(array), rtol=0, atol=0)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_nanmedian_cuda_all_nan(dtype):
    array = np.full(100003, -np.nan, dtype=dtype)
    assert np.isnan(_value(tp.nanmedian(_as_tensor(array, "cuda"))))
