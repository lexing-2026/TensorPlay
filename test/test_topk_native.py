import numpy as np
import pytest

import tensorplay as tp


def _as_tensor(array, device=None):
    value = tp.from_numpy(np.ascontiguousarray(array))
    return value if device is None else value.to(device)


def _as_numpy(value):
    return value.cpu().numpy() if value.is_cuda else value.numpy()


def _expected_indices(array, k, dim, largest):
    moved = np.moveaxis(array, dim, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    result = np.empty((flat.shape[0], k), dtype=np.int64)
    floating = np.issubdtype(array.dtype, np.floating)
    for row_index, row in enumerate(flat):
        if floating:
            nan_indices = np.flatnonzero(np.isnan(row)).tolist()
            finite_indices = np.flatnonzero(~np.isnan(row)).tolist()
        else:
            nan_indices = []
            finite_indices = list(range(row.shape[0]))
        finite_indices.sort(key=lambda index: row[index], reverse=largest)
        order = nan_indices + finite_indices if largest else finite_indices + nan_indices
        result[row_index] = order[:k]
    return result.reshape(moved.shape[:-1] + (k,))


def _assert_sorted_result(array, k, dim, largest, device, impl):
    value, index = tp.topk(_as_tensor(array, device), k, dim, largest, True, impl)
    got_index = np.moveaxis(_as_numpy(index), dim, -1)
    got_value = np.moveaxis(_as_numpy(value), dim, -1)
    expected_index = _expected_indices(array, k, dim, largest)
    expected_value = np.take_along_axis(np.moveaxis(array, dim, -1), expected_index, -1)
    np.testing.assert_array_equal(got_index, expected_index)
    np.testing.assert_allclose(got_value, expected_value, rtol=2e-3, atol=2e-3, equal_nan=True)


@pytest.mark.parametrize(
    "array, k, dim, largest",
    [
        (np.array([[9, 2, 7], [4, 8, 1]], dtype=np.int64), 1, 0, True),
        (np.array([[9, 2, 7], [4, 8, 1]], dtype=np.int32), 2, 1, False),
        (np.array([[1, 8, 3, 6], [7, 2, 9, 4]], dtype=np.uint8), 3, 1, True),
        (np.array([[1.5, -2.0, 4.0], [7.0, 0.5, -1.0]], dtype=np.float64), 2, 1, True),
        (np.array([[1.5, -2.0, 4.0], [7.0, 0.5, -1.0]], dtype=np.float16), 2, 0, False),
    ],
)
def test_topk_native_cpu(array, k, dim, largest):
    _assert_sorted_result(array, k, dim, largest, None, 0)


@pytest.mark.parametrize("largest", [True, False])
def test_topk_native_cpu_nan_order(largest):
    array = np.array([[np.nan, 4.0, 2.0, np.nan], [3.0, np.nan, -1.0, 5.0]], dtype=np.float64)
    _assert_sorted_result(array, 3, 1, largest, None, 0)


def test_topk_native_cpu_unsorted_membership():
    array = np.array([[8.0, 1.0, 7.0, 3.0, 6.0], [2.0, 9.0, 4.0, 5.0, 0.0]], dtype=np.float32)
    value, index = tp.topk(_as_tensor(array), 3, 1, True, False, 0)
    got_index = _as_numpy(index)
    got_value = _as_numpy(value)
    expected_index = _expected_indices(array, 3, 1, True)
    for row in range(array.shape[0]):
        assert set(got_index[row].tolist()) == set(expected_index[row].tolist())
        np.testing.assert_array_equal(got_value[row], array[row, got_index[row]])


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("impl", [0, 1])
@pytest.mark.parametrize(
    "array, k, dim, largest",
    [
        (np.array([[[9, 2, 7], [4, 8, 1]], [[3, 6, 0], [5, 2, 9]]], dtype=np.int64), 2, 1, True),
        (np.array([[[9, 2, 7], [4, 8, 1]], [[3, 6, 0], [5, 2, 9]]], dtype=np.int32), 1, 0, False),
        (np.array([[[1.5, -2.0, 4.0], [7.0, 0.5, -1.0]]], dtype=np.float64), 2, 1, True),
        (np.array([[[1.5, -2.0, 4.0], [7.0, 0.5, -1.0]]], dtype=np.float16), 1, -1, False),
    ],
)
def test_topk_native_cuda(array, k, dim, largest, impl):
    _assert_sorted_result(array, k, dim, largest, "cuda", impl)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("largest", [True, False])
def test_topk_native_cuda_nan_order(largest):
    array = np.array([[np.nan, 4.0, 2.0, np.nan], [3.0, np.nan, -1.0, 5.0]], dtype=np.float32)
    _assert_sorted_result(array, 3, 1, largest, "cuda", 0)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_topk_native_cuda_large_dimension_fallback():
    array = np.linspace(-1.0, 1.0, 4097, dtype=np.float32).reshape(1, -1)
    _assert_sorted_result(array, 3, 1, True, "cuda", 0)
