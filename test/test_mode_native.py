import numpy as np
import pytest

import tensorplay as tp


def _as_tensor(array, device=None):
    value = tp.from_numpy(np.ascontiguousarray(array))
    return value if device is None else value.to(device)


def _value(tensor):
    return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()


def _expected_mode(array, dim):
    """Mode of every line along ``dim`` for the accelerator path.

    The reported position is the last occurrence of the mode when the line is
    at most 2048 long and the first one above that, so the expectation carries
    both rules.  Count ties settle on the smallest value.  The host path may
    report any occurrence, so its index is only checked for validity.
    """
    moved = np.moveaxis(np.asarray(array), dim, -1)
    rows = moved.reshape(-1, moved.shape[-1])
    size = rows.shape[-1]
    out_values = np.empty(rows.shape[0], dtype=rows.dtype)
    out_indices = np.empty(rows.shape[0], dtype=np.int64)
    for r, row in enumerate(rows):
        if row.dtype == np.bool_:
            mode = int(row.sum()) * 2 > size
            value = np.bool_(mode)
        else:
            distinct, counts = np.unique(row, return_counts=True)
            value = distinct[int(np.argmax(counts))]
        out_values[r] = value
        positions = np.flatnonzero(row == value)
        out_indices[r] = positions[-1] if size <= 2048 else positions[0]
    return (
        out_values.reshape(moved.shape[:-1]),
        out_indices.reshape(moved.shape[:-1]),
    )


def _dominant_rows(rng, shape, dominant):
    rows = rng.integers(1, 5, size=shape)
    for r in range(shape[0]):
        chosen = rng.permutation(shape[1])[:dominant]
        rows[r, chosen] = 0
    return rows


@pytest.mark.parametrize("size", [1, 2, 3, 1000, 2048, 2049, 16385, 100003])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_mode_along_last_dim_matches_rule(size, dtype):
    rng = np.random.default_rng(size + 43)
    array = _dominant_rows(rng, (4, size), max(2, size // 2 + 1)).astype(dtype)
    expected_value, _ = _expected_mode(array, 1)
    got = tp.mode(_as_tensor(array), 1)
    np.testing.assert_array_equal(_value(got.values), expected_value)
    values = _value(got.values)
    indices = _value(got.indices)
    for r in range(array.shape[0]):
        assert array[r, indices[r]] == values[r]


@pytest.mark.parametrize("size", [1, 3, 1000, 2048])
def test_mode_cpu_reports_a_position_of_the_mode(size):
    rng = np.random.default_rng(size + 47)
    array = _dominant_rows(rng, (3, size), max(2, size // 2 + 1)).astype(np.int64)
    got = tp.mode(_as_tensor(array), 1)
    values = _value(got.values)
    indices = _value(got.indices)
    expected_value, _ = _expected_mode(array, 1)
    np.testing.assert_array_equal(values, expected_value)
    for r in range(array.shape[0]):
        assert array[r, indices[r]] == values[r]


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [2, 3, 1000, 2048, 2049, 16385, 100003, 1 << 20])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_mode_cuda_along_last_dim_matches_rule(size, dtype):
    rng = np.random.default_rng(size + 53)
    array = _dominant_rows(rng, (3, size), max(2, size // 2 + 1)).astype(dtype)
    expected_value, expected_index = _expected_mode(array, 1)
    got = tp.mode(_as_tensor(array, "cuda"), 1)
    np.testing.assert_array_equal(_value(got.values), expected_value)
    np.testing.assert_array_equal(_value(got.indices), expected_index)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [2049, 16384, 100000, 1 << 20])
def test_mode_cuda_count_ties_keep_smallest_value(size):
    rng = np.random.default_rng(size + 59)
    array = np.tile(np.arange(size) % 4, (2, 1)).astype(np.int64)
    array[1] = rng.permutation(array[1])
    expected_value, expected_index = _expected_mode(array, 1)
    assert np.all(expected_value == 0)
    got = tp.mode(_as_tensor(array, "cuda"), 1)
    np.testing.assert_array_equal(_value(got.values), expected_value)
    np.testing.assert_array_equal(_value(got.indices), expected_index)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [2, 1000, 2048, 2049, 3000, 100000])
def test_mode_cuda_bool(size):
    rng = np.random.default_rng(size + 61)
    array = rng.random((3, size)) < 0.45
    expected_value, expected_index = _expected_mode(array, 1)
    got = tp.mode(_as_tensor(array, "cuda"), 1)
    np.testing.assert_array_equal(_value(got.values), expected_value)
    np.testing.assert_array_equal(_value(got.indices), expected_index)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("shape,dim", [((5, 1000), 0), ((5, 20003), 0), ((3, 16385, 4), 1)])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_mode_cuda_along_dim(shape, dim, dtype):
    rng = np.random.default_rng(sum(shape) + dim)
    array = rng.integers(0, 6, size=shape).astype(dtype)
    moved = np.moveaxis(array, dim, -1)
    dominant = max(2, moved.shape[-1] // 2 + 1)
    for line in moved.reshape(-1, moved.shape[-1]):
        line[rng.permutation(line.size)[:dominant]] = 0
    expected_value, expected_index = _expected_mode(array, dim)
    got = tp.mode(_as_tensor(array, "cuda"), dim)
    np.testing.assert_array_equal(_value(got.values), expected_value)
    np.testing.assert_array_equal(_value(got.indices), expected_index)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_mode_cuda_strided_input():
    rng = np.random.default_rng(67)
    array = rng.integers(0, 8, size=(4, 5, 20003)).astype(np.int64)
    for r in range(array.shape[0]):
        for c in range(array.shape[1]):
            array[r, c, rng.permutation(array.shape[2])[:12000]] = 7
    tensor = _as_tensor(array, "cuda").transpose(0, 2)
    expected_value, expected_index = _expected_mode(array, 2)
    got = tp.mode(tensor, 0)
    np.testing.assert_array_equal(_value(got.values), expected_value.T)
    np.testing.assert_array_equal(_value(got.indices), expected_index.T)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_mode_cuda_keepdim_and_scalar():
    rng = np.random.default_rng(71)
    array = rng.integers(0, 6, size=(3, 5000)).astype(np.int64)
    array[:, :3000] = 4
    got = tp.mode(_as_tensor(array, "cuda"), 1, keepdim=True)
    assert tuple(got.values.shape) == (3, 1)
    assert tuple(got.indices.shape) == (3, 1)
    expected_value, expected_index = _expected_mode(array, 1)
    np.testing.assert_array_equal(_value(got.values).reshape(3), expected_value)
    np.testing.assert_array_equal(_value(got.indices).reshape(3), expected_index)

    scalar = tp.from_numpy(np.array(5, dtype=np.int64)).to("cuda")
    scalar_mode = tp.mode(scalar)
    assert int(_value(scalar_mode.values)) == 5
    assert int(_value(scalar_mode.indices)) == 0


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_mode_cuda_single_element_lines():
    array = np.arange(24, dtype=np.int64).reshape(4, 6)
    got = tp.mode(_as_tensor(array, "cuda"), 0)
    np.testing.assert_array_equal(_value(got.values), np.arange(6))
    np.testing.assert_array_equal(_value(got.indices), np.zeros(6, dtype=np.int64))
