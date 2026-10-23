import numpy as np
import pytest

import tensorplay as tp


def _as_tensor(array, device=None):
    value = tp.from_numpy(np.ascontiguousarray(array))
    return value if device is None else value.to(device)


def _value(tensor):
    return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()


def _expected_kth(array, k, dim):
    moved = np.moveaxis(np.asarray(array), dim, -1)
    rows = moved.reshape(-1, moved.shape[-1])
    out_values = np.empty(rows.shape[0], dtype=rows.dtype)
    out_indices = np.empty(rows.shape[0], dtype=np.int64)
    for r, row in enumerate(rows):
        value = np.sort(row)[k - 1]
        out_values[r] = value
        if row.dtype.kind == "f" and np.isnan(value):
            out_indices[r] = int(np.flatnonzero(np.isnan(row))[0])
        else:
            out_indices[r] = int(np.flatnonzero(row == value)[0])
    return (
        out_values.reshape(moved.shape[:-1]),
        out_indices.reshape(moved.shape[:-1]),
    )


@pytest.mark.parametrize("size", [1, 2, 3, 2048, 16384, 16385, 100003, 1 << 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_kthvalue_flat_matches_lower_selection(size, dtype):
    rng = np.random.default_rng(size + 7)
    if np.issubdtype(dtype, np.integer):
        array = rng.integers(-200, 200, size=size).astype(dtype)
    else:
        array = rng.standard_normal(size).astype(dtype)
    for k in sorted({1, (size + 1) // 2, size}):
        expected_value, _ = _expected_kth(array, k, 0)
        got = tp.kthvalue(_as_tensor(array), k, 0)
        values = np.atleast_1d(_value(got.values))
        indices = np.atleast_1d(_value(got.indices))
        np.testing.assert_allclose(
            values, np.atleast_1d(expected_value), rtol=0, atol=0
        )
        # The host path may report any occurrence of the k-th value, so its
        # index is checked for validity rather than an exact position.
        for r in range(values.size):
            assert np.asarray(array).reshape(-1)[indices[r]] == values[r]


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [1, 2, 3, 2048, 16384, 16385, 100003, 1 << 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_kthvalue_flat_cuda_matches_lower_selection(size, dtype):
    rng = np.random.default_rng(size + 13)
    if np.issubdtype(dtype, np.integer):
        array = rng.integers(-200, 200, size=size).astype(dtype)
    else:
        array = rng.standard_normal(size).astype(dtype)
    for k in sorted({1, (size + 1) // 2, size}):
        expected_value, expected_index = _expected_kth(array, k, 0)
        got = tp.kthvalue(_as_tensor(array, "cuda"), k, 0)
        np.testing.assert_allclose(_value(got.values), expected_value, rtol=0, atol=0)
        np.testing.assert_array_equal(_value(got.indices), expected_index)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [100, 16385, 100003])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_kthvalue_cuda_ties_report_first_original_index(size, dtype):
    rng = np.random.default_rng(size + 19)
    if np.issubdtype(dtype, np.integer):
        array = rng.integers(0, 3, size=size).astype(dtype)
    else:
        array = rng.integers(0, 3, size=size).astype(dtype)
    for k in sorted({1, (size + 1) // 2, size}):
        expected_value, expected_index = _expected_kth(array, k, 0)
        got = tp.kthvalue(_as_tensor(array, "cuda"), k, 0)
        np.testing.assert_allclose(_value(got.values), expected_value, rtol=0, atol=0)
        np.testing.assert_array_equal(_value(got.indices), expected_index)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [16385, 100003])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nan_sign", [np.nan, -np.nan])
def test_kthvalue_cuda_nan_ranks_last(size, dtype, nan_sign):
    rng = np.random.default_rng(size + 23)
    array = rng.standard_normal(size).astype(dtype)
    holes = rng.random(size) < 0.05
    array[holes] = nan_sign
    valid = int((~holes).sum())
    for k in sorted({1, valid, valid + 1, size}):
        expected_value, expected_index = _expected_kth(array, k, 0)
        got = tp.kthvalue(_as_tensor(array, "cuda"), k, 0)
        np.testing.assert_allclose(_value(got.values), expected_value, rtol=0, atol=0)
        np.testing.assert_array_equal(_value(got.indices), expected_index)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [16385, 100003])
def test_kthvalue_cuda_all_nan(size):
    array = np.full(size, np.nan, dtype=np.float32)
    got = tp.kthvalue(_as_tensor(array, "cuda"), size // 2, 0)
    assert np.isnan(_value(got.values))
    np.testing.assert_array_equal(_value(got.indices), 0)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [3, 2048, 16385, 100003])
def test_kthvalue_cuda_half(size):
    rng = np.random.default_rng(size + 29)
    array = rng.standard_normal(size).astype(np.float16)
    for k in sorted({1, (size + 1) // 2, size}):
        expected_value, expected_index = _expected_kth(array, k, 0)
        got = tp.kthvalue(_as_tensor(array, "cuda"), k, 0)
        np.testing.assert_allclose(_value(got.values), expected_value, rtol=0, atol=0)
        np.testing.assert_array_equal(_value(got.indices), expected_index)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("shape,dim", [((5, 20003), 0), ((5, 20003), 1), ((3, 16385, 4), 1)])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_kthvalue_cuda_along_dim(shape, dim, dtype):
    rng = np.random.default_rng(sum(shape) + dim)
    if np.issubdtype(dtype, np.integer):
        array = rng.integers(-50, 50, size=shape).astype(dtype)
    else:
        array = rng.standard_normal(shape).astype(dtype)
    d_size = shape[dim]
    for k in sorted({1, (d_size + 1) // 2, d_size}):
        expected_value, expected_index = _expected_kth(array, k, dim)
        got = tp.kthvalue(_as_tensor(array, "cuda"), k, dim)
        np.testing.assert_allclose(_value(got.values), expected_value, rtol=0, atol=0)
        np.testing.assert_array_equal(_value(got.indices), expected_index)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_kthvalue_cuda_strided_input():
    rng = np.random.default_rng(31)
    array = rng.integers(-40, 40, size=(4, 5, 16385)).astype(np.int64)
    tensor = _as_tensor(array, "cuda").transpose(0, 2)
    for k in (1, 2, 8193, 16385):
        expected_value, expected_index = _expected_kth(array, k, 2)
        got = tp.kthvalue(tensor, k, 0)
        np.testing.assert_allclose(
            _value(got.values), expected_value.T, rtol=0, atol=0
        )
        np.testing.assert_array_equal(_value(got.indices), expected_index.T)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_kthvalue_cuda_keepdim():
    rng = np.random.default_rng(37)
    array = rng.standard_normal((3, 16385)).astype(np.float32)
    got = tp.kthvalue(_as_tensor(array, "cuda"), 100, 1, keepdim=True)
    assert tuple(got.values.shape) == (3, 1)
    assert tuple(got.indices.shape) == (3, 1)
    expected_value, expected_index = _expected_kth(array, 100, 1)
    np.testing.assert_allclose(_value(got.values).reshape(3), expected_value, rtol=0, atol=0)
    np.testing.assert_array_equal(_value(got.indices).reshape(3), expected_index)
