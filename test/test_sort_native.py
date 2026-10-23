import numpy as np
import pytest

import tensorplay as tp


def _as_tensor(array, device=None):
    value = tp.from_numpy(np.ascontiguousarray(array))
    return value if device is None else value.to(device)


def _as_numpy(value):
    return value.cpu().numpy() if value.is_cuda else value.numpy()


def _assert_sorted(array, dim, descending, device):
    value, index = tp.sort(_as_tensor(array, device), dim, descending)
    got_value = _as_numpy(value)
    got_index = _as_numpy(index)
    moved = np.moveaxis(array, dim, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    # A sort only fixes the order of values that compare equal up to the
    # position it gives them, so the order within a run of equals is checked by
    # reading the input at the returned positions rather than by comparing the
    # positions themselves.
    ordered = np.sort(flat, axis=-1)
    if descending:
        ordered = ordered[:, ::-1]
    got_rows = np.moveaxis(got_value, dim, -1).reshape(flat.shape)
    index_rows = np.moveaxis(got_index, dim, -1).reshape(flat.shape)
    np.testing.assert_array_equal(got_rows, ordered)
    gathered = np.take_along_axis(flat, index_rows, axis=-1)
    np.testing.assert_array_equal(gathered, got_rows)
    assert got_value.shape == array.shape


CASES = [
    # A handful of long runs, which is the few-run walk.
    ((2, 5000), 1),
    ((8, 8192), 1),
    ((127, 4100), 1),
    # One run past the few-run walk and the short-run walk.
    ((128, 4100), 1),
    # Runs that are not the last axis.
    ((4, 5001, 3), 1),
    ((3, 3, 9000), 2),
    ((5000, 3, 2), 0),
    # A single run, and a short one.
    ((30000,), 0),
    ((16, 2048), 1),
]


@pytest.mark.parametrize("shape, dim", CASES)
@pytest.mark.parametrize("descending", [False, True])
def test_sort_native_values_and_positions(shape, dim, descending):
    device = tp.device("cuda") if tp.cuda.is_available() else None
    rng = np.random.RandomState(11)
    array = rng.randint(-40, 40, size=shape).astype(np.int64)
    _assert_sorted(array, dim, descending, device)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_native_floats_keep_nan_at_the_deep_end(dtype, descending):
    device = tp.device("cuda") if tp.cuda.is_available() else None
    rng = np.random.RandomState(12)
    array = rng.randn(8, 6000).astype(dtype)
    array[0, :3] = dtype(np.nan)
    array[1, :2] = dtype(-0.0)
    array[1, 2:4] = dtype(0.0)
    array[2, :50] = dtype(1.5)
    _assert_sorted(array, 1, descending, device)
