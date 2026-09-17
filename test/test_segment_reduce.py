"""Forward coverage for the internal _segment_reduce op (CPU kernel).

Each expectation is computed by hand from the op's contract: split the
reduction axis into consecutive segments described by lengths (segment
sizes) or offsets (cumulative boundaries), fold each segment with the
requested reduction seeded by ``initial`` (identity per reduction when
absent), and for mean divide the accumulated sum by the segment length.
An empty segment folds to the seed value; mean without an explicit seed
over an empty segment is undefined and yields NaN.

Boundary tensors are 1-D only when the reduced axis is 0; in general the
last dimension of lengths/offsets must be the reduced axis, so per-row
boundaries are 2-D for axis=1.
"""

import math

import pytest

import tensorplay as tp


def _flatten(x):
    if isinstance(x, tp.Tensor):
        return tp.reshape(x, [-1]).tolist()
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    return [x]


def _allclose(a, b, tol=1e-5):
    flat_a = _flatten(a)
    flat_b = _flatten(b)
    assert len(flat_a) == len(flat_b), (flat_a, flat_b)
    for x, y in zip(flat_a, flat_b):
        if math.isinf(y) or math.isinf(x):
            assert x == y or (math.isnan(x - y) and x * y < 0), (flat_a, flat_b)
        elif math.isnan(y):
            assert math.isnan(x), (flat_a, flat_b)
        else:
            assert abs(x - y) <= tol * max(1.0, abs(y)), (flat_a, flat_b)


def _seg(data, reduce, lengths=None, offsets=None, axis=0, initial=None):
    return tp._C._segment_reduce(
        data, reduce, lengths=lengths, offsets=offsets, axis=axis,
        initial=initial)


# ----------------------------------------------------------------- lengths


def test_lengths_mode_reductions_1d():
    data = tp.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lengths = tp.tensor([2, 0, 3, 1])
    # segments: [1,2] / [] / [3,4,5] / [6]
    _allclose(_seg(data, 'sum', lengths=lengths), [3.0, 0.0, 12.0, 6.0])
    _allclose(_seg(data, 'prod', lengths=lengths), [2.0, 1.0, 60.0, 6.0])
    _allclose(_seg(data, 'max', lengths=lengths), [2.0, float('-inf'), 5.0, 6.0])
    _allclose(_seg(data, 'min', lengths=lengths), [1.0, float('inf'), 3.0, 6.0])
    out = _seg(data, 'mean', lengths=lengths)
    values = out.tolist()
    assert math.isnan(values[1])
    _allclose([values[0], values[2], values[3]], [1.5, 4.0, 6.0])


def test_offsets_mode_reductions_1d():
    data = tp.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    offsets = tp.tensor([0, 2, 2, 5])
    # segments: [1,2] / [] / [3,4,5]
    _allclose(_seg(data, 'sum', offsets=offsets), [3.0, 0.0, 12.0])
    _allclose(_seg(data, 'prod', offsets=offsets), [2.0, 1.0, 60.0])
    _allclose(_seg(data, 'max', offsets=offsets), [2.0, float('-inf'), 5.0])
    _allclose(_seg(data, 'min', offsets=offsets), [1.0, float('inf'), 3.0])
    out = _seg(data, 'mean', offsets=offsets)
    values = out.tolist()
    assert math.isnan(values[1])
    _allclose([values[0], values[2]], [1.5, 4.0])


def test_initial_seeds_every_reduction():
    data = tp.tensor([1.0, 2.0, 3.0])
    lengths = tp.tensor([2, 1])
    # sum: 10 + fold; prod: 2 * fold
    _allclose(_seg(data, 'sum', lengths=lengths, initial=10.0), [13.0, 13.0])
    _allclose(_seg(data, 'prod', lengths=lengths, initial=2.0), [4.0, 6.0])
    # extrema are seeded by initial as well: 5 dominates every element
    _allclose(_seg(data, 'max', lengths=lengths, initial=5.0), [5.0, 5.0])
    _allclose(_seg(data, 'min', lengths=lengths, initial=-5.0), [-5.0, -5.0])
    # mean folds with the seed, then divides by the segment length
    _allclose(_seg(data, 'mean', lengths=lengths, initial=1.0), [2.0, 4.0])
    # offsets path takes the same seeding
    offsets = tp.tensor([0, 2, 3])
    _allclose(_seg(data, 'sum', offsets=offsets, initial=1.0), [4.0, 4.0])


def test_empty_segments_with_and_without_initial():
    lengths = tp.tensor([0, 2])
    data = tp.tensor([1.0, 2.0])
    _allclose(_seg(data, 'sum', lengths=lengths), [0.0, 3.0])
    _allclose(_seg(data, 'sum', lengths=lengths, initial=7.0), [7.0, 10.0])
    _allclose(_seg(data, 'prod', lengths=lengths), [1.0, 2.0])
    _allclose(_seg(data, 'prod', lengths=lengths, initial=3.0), [3.0, 6.0])
    got = _seg(data, 'mean', lengths=lengths)
    assert math.isnan(got.tolist()[0])
    _allclose([got.tolist()[1]], [1.5])
    # an explicit seed survives an empty mean segment undivided
    got = _seg(tp.tensor([5.0]), 'mean', lengths=tp.tensor([0, 1]), initial=9.0)
    _allclose(got, [9.0, 14.0])
    # same rules on the offsets path
    offsets = tp.tensor([0, 0, 2])
    _allclose(_seg(data, 'sum', offsets=offsets, initial=7.0), [7.0, 10.0])
    got = _seg(data, 'mean', offsets=offsets)
    assert math.isnan(got.tolist()[0])
    _allclose([got.tolist()[1]], [1.5])


# ---------------------------------------------------------------- 2-D data


def test_2d_data_axis0_lengths():
    data = tp.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lengths = tp.tensor([1, 2])
    # rows [0] and [1, 2] reduce elementwise down axis 0
    _allclose(_seg(data, 'sum', lengths=lengths), [[1.0, 2.0], [8.0, 10.0]])
    _allclose(_seg(data, 'mean', lengths=lengths), [[1.0, 2.0], [4.0, 5.0]])
    _allclose(_seg(data, 'max', lengths=lengths), [[1.0, 2.0], [5.0, 6.0]])
    _allclose(_seg(data, 'min', lengths=lengths), [[1.0, 2.0], [3.0, 4.0]])
    _allclose(_seg(data, 'prod', lengths=lengths), [[1.0, 2.0], [15.0, 24.0]])
    assert _seg(data, 'sum', lengths=lengths).shape == (2, 2)


def test_2d_data_axis0_offsets():
    data = tp.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    offsets = tp.tensor([0, 1, 3])
    _allclose(_seg(data, 'sum', offsets=offsets), [[1.0, 2.0], [8.0, 10.0]])
    _allclose(_seg(data, 'min', offsets=offsets), [[1.0, 2.0], [3.0, 4.0]])


def test_2d_data_axis1_per_row_lengths():
    data = tp.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    # axis 1 requires 2-D boundaries: one row of lengths per data row
    lengths = tp.tensor([[1, 3], [2, 2]])
    _allclose(_seg(data, 'sum', lengths=lengths, axis=1),
              [[1.0, 9.0], [11.0, 15.0]])
    _allclose(_seg(data, 'mean', lengths=lengths, axis=1),
              [[1.0, 3.0], [5.5, 7.5]])
    _allclose(_seg(data, 'max', lengths=lengths, axis=1),
              [[1.0, 4.0], [6.0, 8.0]])
    _allclose(_seg(data, 'min', lengths=lengths, axis=1),
              [[1.0, 2.0], [5.0, 7.0]])
    _allclose(_seg(data, 'prod', lengths=lengths, axis=1),
              [[1.0, 24.0], [30.0, 56.0]])


def test_2d_data_axis1_per_row_offsets():
    data = tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    offsets = tp.tensor([[0, 2, 3], [0, 1, 3]])
    _allclose(_seg(data, 'sum', offsets=offsets, axis=1), [[3.0, 3.0], [4.0, 11.0]])
    _allclose(_seg(data, 'mean', offsets=offsets, axis=1),
              [[1.5, 3.0], [4.0, 5.5]])


def test_non_contiguous_data_is_normalized():
    base = tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data = base.t()  # shape (3, 2), values [[1,4],[2,5],[3,6]]
    assert not data.is_contiguous()
    lengths = tp.tensor([1, 2])
    _allclose(_seg(data, 'sum', lengths=lengths), [[1.0, 4.0], [5.0, 11.0]])


# ------------------------------------------------------ dtypes and devices


def test_output_dtype_and_device_follow_data():
    data64 = tp.tensor([1.0, 2.0, 3.0], dtype=tp.float64)
    out = _seg(data64, 'sum', lengths=tp.tensor([1, 2]))
    assert out.dtype == tp.float64
    _allclose(out, [1.0, 5.0])

    data32 = tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32)
    out = _seg(data32, 'sum', lengths=tp.tensor([1, 2]))
    assert out.dtype == tp.float32


def test_int32_boundary_tensors():
    data = tp.tensor([1.0, 2.0, 3.0, 4.0])
    _allclose(_seg(data, 'sum', lengths=tp.tensor([2, 2], dtype=tp.int32)),
              [3.0, 7.0])
    _allclose(_seg(data, 'sum', offsets=tp.tensor([0, 2, 4], dtype=tp.int32)),
              [3.0, 7.0])


# ------------------------------------------------------------- error cases


def test_rejects_both_or_neither_boundaries():
    data = tp.tensor([1.0, 2.0, 3.0])
    with pytest.raises(Exception):
        _seg(data, 'sum')
    with pytest.raises(Exception):
        _seg(data, 'sum', lengths=tp.tensor([1, 2]),
             offsets=tp.tensor([0, 1, 3]))


def test_rejects_inconsistent_lengths():
    data = tp.tensor([1.0, 2.0, 3.0])
    with pytest.raises(Exception):
        # lengths must cover the reduced axis exactly
        _seg(data, 'sum', lengths=tp.tensor([1, 1]))
    with pytest.raises(Exception):
        _seg(data, 'sum', lengths=tp.tensor([-1, 4]))


def test_rejects_axis_boundary_mismatch():
    data = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(Exception):
        # axis 0 cannot be the last dimension of a 2-D boundary tensor
        _seg(data, 'sum', lengths=tp.tensor([[1, 1], [2, 0]]), axis=0)
    with pytest.raises(Exception):
        _seg(data, 'sum', lengths=tp.tensor([1, 1]), axis=5)


def test_rejects_bad_reduce_and_boundary_dtype():
    data = tp.tensor([1.0, 2.0, 3.0])
    with pytest.raises(Exception):
        _seg(data, 'median', lengths=tp.tensor([1, 2]))
    with pytest.raises(Exception):
        _seg(data, 'sum', lengths=tp.tensor([1.0, 2.0]))
