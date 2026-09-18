"""Tests for nested (ragged) tensors.

Covers construction from lists and dense batches, the per-constituent
metadata accessors, padding back to a dense tensor, and the documented
error behavior.  The strided layout carries rank-R constituents and
reports dim() == R + 1.
"""

import pytest

import tensorplay as tp
from tensorplay.nested import as_nested_tensor, nested_tensor, to_padded_tensor


def test_is_nested_and_accessors():
    parts = [tp.tensor([[1.0, 2.0], [3.0, 4.0]]), tp.tensor([[5.0], [6.0], [7.0]])]
    nt = nested_tensor(parts)
    assert nt.is_nested()
    assert tp.is_nested(nt)
    assert not tp.tensor([1.0]).is_nested()

    sizes = nt._nested_tensor_size()
    strides = nt._nested_tensor_strides()
    offsets = nt._nested_tensor_storage_offsets()
    assert tp.equal(sizes, tp.tensor([[2, 2], [3, 1]], dtype=tp.int64))
    # Row-major strides: the last axis moves by one element, the axis
    # before it by the extent of the last axis.
    assert tp.equal(strides, tp.tensor([[2, 1], [1, 1]], dtype=tp.int64))
    # Packed rows: the second constituent starts right after the first.
    assert tp.equal(offsets, tp.tensor([0, 4], dtype=tp.int64))


def test_metadata_view_of_geometry():
    nt = nested_tensor([tp.ones(2), tp.ones(3)])
    assert nt.dim() == 2
    assert nt.size(0) == 2
    # Deeper dimensions report the largest extent across constituents.
    assert nt.size(1) == 3
    assert nt.size(-1) == 3
    assert nt.numel() == 5
    assert nt.dtype == tp.float32
    with pytest.raises(RuntimeError, match="dense size"):
        nt.shape
    with pytest.raises(RuntimeError, match="dense stride"):
        nt.strides
    with pytest.raises(IndexError):
        nt.size(2)


def test_constituent_values_packed_in_order():
    parts = [tp.tensor([1.0, 2.0, 3.0]), tp.tensor([4.0])]
    nt = nested_tensor(parts)
    sizes = nt._nested_tensor_size().tolist()
    offsets = nt._nested_tensor_storage_offsets().tolist()
    # Row i occupies the product of its size row starting at its offset.
    assert [(off, row) for row, off in zip(sizes, offsets)] == [
        (0, [3]),
        (3, [1]),
    ]
    padded = to_padded_tensor(nt, 0.0)
    assert tp.allclose(padded, tp.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]]))


def test_as_nested_tensor_from_dense_shares_storage():
    dense = tp.arange(0, 12, dtype=tp.float32).reshape(3, 4)
    nt = as_nested_tensor(dense)
    assert nt.is_nested()
    # The rows are rank-1 constituents, so the view reports dim() == 2.
    assert nt.dim() == 2
    assert nt.size(0) == 3
    assert nt.numel() == 12
    sizes = nt._nested_tensor_size()
    assert tp.equal(sizes, tp.tensor([[4], [4], [4]], dtype=tp.int64))

    # A dtype conversion must copy: the result is a nested tensor of the
    # requested dtype while the source stays untouched.
    nt64 = as_nested_tensor(dense, dtype=tp.float64)
    assert nt64.dtype == tp.float64
    assert dense.dtype == tp.float32


def test_as_nested_tensor_list_copies():
    a = tp.tensor([1.0, 2.0])
    b = tp.tensor([3.0])
    nt = as_nested_tensor([a, b])
    assert nt.numel() == 3
    assert tp.equal(
        nt._nested_tensor_size(), tp.tensor([[2], [1]], dtype=tp.int64)
    )
    # Rank mismatch is rejected.
    with pytest.raises(RuntimeError, match="same rank"):
        as_nested_tensor([a, tp.ones(2, 2)])


def test_to_padded_tensor_manual_reference():
    parts = [
        tp.tensor([[1.0, 2.0], [3.0, 4.0]]),
        tp.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
    ]
    nt = nested_tensor(parts)
    padded = to_padded_tensor(nt, 0.0)
    want = tp.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    assert padded.dim() == 3
    assert tuple(padded.shape) == (2, 3, 2)
    assert tp.allclose(padded, want)

    padded_ones = to_padded_tensor(nt, -1.0)
    assert float(padded_ones[0][2][0]) == -1.0
    assert float(padded_ones[1][2][1]) == 10.0

    # An explicit output size may pad further but never truncate.
    large = to_padded_tensor(nt, 9.0, (2, 4, 3))
    assert tuple(large.shape) == (2, 4, 3)
    assert float(large[0][3][2]) == 9.0
    with pytest.raises(RuntimeError, match="truncate"):
        to_padded_tensor(nt, 0.0, (2, 2, 2))


def test_to_padded_tensor_method_and_1d():
    nt = nested_tensor([tp.tensor([1.0, 2.0]), tp.tensor([3.0])])
    padded = nt.to_padded_tensor(0.0)
    assert tp.allclose(padded, tp.tensor([[1.0, 2.0], [3.0, 0.0]]))


def test_empty_constituent():
    nt = nested_tensor([tp.zeros(0), tp.ones(2)])
    assert nt.numel() == 2
    padded = to_padded_tensor(nt, 5.0)
    assert tuple(padded.shape) == (2, 2)
    assert tp.allclose(padded, tp.tensor([[5.0, 5.0], [1.0, 1.0]]))


def test_dtype_and_device_options():
    parts = [tp.tensor([1, 2]), tp.tensor([3])]
    nt = nested_tensor(parts, dtype=tp.float64)
    assert nt.dtype == tp.float64
    assert nt._nested_tensor_size().dtype == tp.int64


def test_repr_shows_constituents():
    nt = nested_tensor([tp.tensor([1.0, 2.0]), tp.tensor([3.0])])
    text = repr(nt)
    assert text.startswith("nested_tensor([")
    assert "tensor([1., 2.]" in text or "tensor([1.,  2.]" in text


def test_accessors_reject_dense_input():
    t = tp.ones(3)
    for fn in (tp._nested_tensor_size, tp._nested_tensor_strides,
               tp._nested_tensor_storage_offsets):
        with pytest.raises(RuntimeError, match="nested"):
            fn(t)
    with pytest.raises(RuntimeError, match="nested"):
        to_padded_tensor(t, 0.0)


def test_requires_grad_option():
    nt = nested_tensor([tp.ones(2), tp.ones(1)], requires_grad=True)
    assert nt.requires_grad


def test_non_contiguous_and_offset_rows():
    base = tp.arange(0, 20, dtype=tp.float32).reshape(4, 5)
    # Take ragged row lengths from a 2-D buffer by hand: rows of length 5
    # and 3 packed into one buffer, with the offsets tensor routing each
    # constituent to its slice.
    buffer = tp.cat([base[0], base[1, :3]], 0)
    sizes = tp.tensor([[5], [3]], dtype=tp.int64)
    offsets = tp.tensor([0, 5], dtype=tp.int64)
    strides = tp.tensor([[1], [1]], dtype=tp.int64)
    nt = tp._nested_view_from_buffer(buffer, sizes, strides, offsets)
    assert nt.numel() == 8
    assert nt.size(0) == 2
    assert nt.size(1) == 5
    padded = to_padded_tensor(nt, 0.0)
    assert tp.allclose(padded[0], base[0])
    assert tp.allclose(padded[1, :3], base[1, :3])
    # Blanks in the buffer stay blanks: padding fills them, not neighbors.
    assert float(padded[1][3]) == 0.0


def test_dense_view_on_cuda_keeps_metadata_on_device():
    if not tp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dense = tp.arange(0, 12, dtype=tp.float32).reshape(3, 4).to("cuda")
    nt = as_nested_tensor(dense)
    assert nt.is_nested()
    assert nt.dim() == 2 and nt.size(0) == 3 and nt.numel() == 12
    assert nt._nested_tensor_size().device.type == "cuda"
    padded = to_padded_tensor(nt, 0.0)
    assert tp.equal(padded.cpu(), dense.cpu())


def test_nested_accessor_family():
    nt = nested_tensor([tp.tensor([[1.0, 2.0], [3.0, 4.0]]), tp.tensor([[5.0]])])
    vals = tp._C._nested_get_values(nt)
    assert vals.numel() == 5
    assert vals.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    copy = tp._C._nested_get_values_copy(nt)
    assert copy.tolist() == vals.tolist()
    assert tp._C._nested_get_offsets(nt).tolist() == [0, 4]
    # The ragged axis is the first size column: lengths [2, 1].
    assert tp._C._nested_get_lengths(nt).tolist() == [2, 1]
    assert tp._C._nested_get_ragged_idx(nt) == 0
    assert tp._C._nested_get_min_seqlen(nt).tolist() == [1]
    assert tp._C._nested_get_max_seqlen(nt).tolist() == [2]
    dummy = tp._C._nested_get_jagged_dummy(nt)
    assert dummy.numel() == 0 and dummy.dtype == nt.dtype


def test_nested_compute_contiguous_strides_offsets():
    sizes = tp.tensor([[2, 3], [1, 3]], dtype=tp.int64)
    strides, offsets = tp._C._nested_compute_contiguous_strides_offsets(sizes)
    assert strides.tolist() == [[3, 1], [3, 1]]
    assert offsets.tolist() == [0, 6]


def test_nested_view_from_jagged():
    values = tp.arange(0, 8, dtype=tp.float32)
    offsets = tp.tensor([0, 5, 8], dtype=tp.int64)
    dummy = tp._C._nested_get_jagged_dummy(values)
    nt = tp._C._nested_view_from_jagged(values, offsets, dummy, None, 1,
                                        None, None)
    assert nt.is_nested() and nt.dim() == 2 and nt.numel() == 8
    assert nt._nested_tensor_size().tolist() == [[5], [3]]
    assert nt._nested_tensor_storage_offsets().tolist() == [0, 5]
    padded = to_padded_tensor(nt, -1.0)
    assert padded.tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, -1.0, -1.0]]

    # A ragged first axis with trailing extents: offsets index rows of the
    # values batch and scale by the trailing volume inside the buffer.
    values2 = tp.arange(0, 12, dtype=tp.float32).reshape(6, 2)
    offsets2 = tp.tensor([0, 4, 6], dtype=tp.int64)
    nt2 = tp._C._nested_view_from_jagged(values2, offsets2, dummy, None, 1,
                                         None, None)
    assert nt2._nested_tensor_size().tolist() == [[4, 2], [2, 2]]
    p2 = to_padded_tensor(nt2, 0.0)
    assert tuple(p2.shape) == (2, 4, 2)
    assert p2[1].tolist() == [[8.0, 9.0], [10.0, 11.0], [0.0, 0.0], [0.0, 0.0]]

    with pytest.raises(RuntimeError, match="ragged_idx"):
        tp._C._nested_view_from_jagged(values, offsets, dummy, None, 2,
                                       None, None)
    with pytest.raises(RuntimeError, match="past the values"):
        tp._C._nested_view_from_jagged(
            values, tp.tensor([0, 5, 99], dtype=tp.int64), dummy, None, 1,
            None, None)


def test_nested_view_from_jagged_copy_does_not_alias():
    values = tp.arange(0, 4, dtype=tp.float32)
    dummy = tp._C._nested_get_jagged_dummy(values)
    nt = tp._C._nested_view_from_jagged_copy(
        values, tp.tensor([0, 2, 4], dtype=tp.int64), dummy, None, 1, None,
        None)
    tp._C._nested_get_values(nt).fill_(0.0)
    assert values.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_nested_from_padded_and_mask():
    padded = tp.arange(0, 24, dtype=tp.float32).reshape(2, 4, 3)
    sizes = tp.tensor([[3, 3], [2, 3]], dtype=tp.int64)
    nt = tp._C._nested_from_padded(padded, sizes, False)
    assert nt.numel() == 15 and nt.dim() == 3 and nt.size(0) == 2
    assert nt._nested_tensor_size().tolist() == [[3, 3], [2, 3]]
    assert nt._nested_tensor_storage_offsets().tolist() == [0, 12]

    mask = tp.tensor([[True, True, True, False], [True, True, False, False]])
    nt2 = tp._C._nested_tensor_from_mask(padded, mask, True)
    assert tp.equal(nt2._nested_tensor_size(), sizes)
    assert tp.equal(tp._C._nested_get_values(nt2), tp._C._nested_get_values(nt))
    assert tp._C._nested_tensor_from_mask_left_aligned(padded, mask)

    gapped = tp.tensor([[True, False, True, False], [True, True, True, True]])
    assert not tp._C._nested_tensor_from_mask_left_aligned(padded, gapped)
    with pytest.raises(RuntimeError, match="left-aligned"):
        tp._C._nested_tensor_from_mask(padded, gapped, True)


def test_nested_sum_backward_repeats_along_the_ragged_axis():
    nt = nested_tensor([tp.tensor([[1.0, 2.0], [3.0, 4.0]]),
                        tp.tensor([[5.0, 6.0]])])
    grad = nested_tensor([tp.tensor([10.0, 20.0]), tp.tensor([30.0])])
    b = tp._C._nested_sum_backward(grad, nt, None, False)
    assert tp._C._nested_get_values(b).tolist() == [
        10.0, 10.0, 20.0, 20.0, 30.0, 30.0]
    assert tp.equal(b._nested_tensor_size(), nt._nested_tensor_size())


def test_nested_select_backward_places_grad_at_the_selection():
    nt = nested_tensor([tp.tensor([[1.0, 2.0], [3.0, 4.0]]),
                        tp.tensor([[5.0, 6.0]])])
    # dim 0: only the selected constituent receives its gradient.
    b0 = tp._C._nested_select_backward(
        tp.tensor([[1.0, 2.0], [3.0, 4.0]]), nt, 0, 0)
    assert tp._C._nested_get_values(b0).tolist() == [1.0, 2.0, 3.0, 4.0, 0.0, 0.0]
    # dim 1, index 0: the first row of each constituent.
    b1 = tp._C._nested_select_backward(
        tp.tensor([[1.0, 2.0], [5.0, 6.0]]), nt, 1, 0)
    assert tp._C._nested_get_values(b1).tolist() == [1.0, 2.0, 0.0, 0.0, 5.0, 6.0]
    with pytest.raises(IndexError):
        tp._C._nested_select_backward(
            tp.tensor([[1.0, 2.0], [5.0, 6.0]]), nt, 1, 1)


def test_nested_softmax_with_shape_normalizes_each_ragged_row():
    nt = nested_tensor([tp.tensor([[1.0, 2.0], [3.0, 4.0]]),
                        tp.tensor([[5.0]])])
    s = tp._C._nested_tensor_softmax_with_shape(nt, tp.zeros(2, 1))
    assert tp.equal(s._nested_tensor_size(), nt._nested_tensor_size())
    # softmax([1, 2]) = [1/(1+e), e/(1+e)]
    q = 1.0 / (1.0 + 2.718281828)
    assert tp.allclose(
        tp._C._nested_get_values(s),
        tp.tensor([q, 1.0 - q, q, 1.0 - q, 1.0]), atol=1e-5)
