import pytest

import tensorplay as tp
from tensorplay._C import DType


def f32(t):
    return t.to(DType.float32)


def arange_f(*shape):
    n = 1
    for s in shape:
        n *= s
    return tp.arange(n).to(DType.float32).reshape(list(shape))


def scalar0(v=3.0):
    return tp.tensor([float(v)]).reshape([])


class TestBroadcastShapes:
    def test_right_aligned(self):
        assert tp.broadcast_shapes((2, 1, 3), (4, 3), (5, 1, 1, 3)) == (5, 2, 4, 3)

    def test_single(self):
        assert tp.broadcast_shapes((3, 4)) == (3, 4)

    def test_zero_dim(self):
        assert tp.broadcast_shapes((), (2, 3)) == (2, 3)
        assert tp.broadcast_shapes((1,), ()) == (1,)

    def test_mismatch_raises(self):
        with pytest.raises(RuntimeError):
            tp.broadcast_shapes((2, 3), (2, 4))

    def test_negative_raises(self):
        with pytest.raises(RuntimeError):
            tp.broadcast_shapes((-1,))


class TestBroadcastTensors:
    def test_shapes_and_values(self):
        a = arange_f(1, 3)
        b = arange_f(2, 1).to(DType.float32) * 10
        outs = tp.broadcast_tensors(a, b.reshape([2, 1]))
        assert all(tuple(o.size()) == (2, 3) for o in outs)
        assert outs[0].tolist()[0] == [0.0, 1.0, 2.0]
        col = outs[1].tolist()
        assert [c[0] for c in col] == [0.0, 10.0]

    def test_empty(self):
        assert tp.broadcast_tensors() == []

    def test_non_tensor_raises(self):
        with pytest.raises(TypeError):
            tp.broadcast_tensors(tp.arange(3), 1.0)

    def test_grad_flows_like_expand_backward(self):
        x = tp.arange(3).to(DType.float32)
        x.requires_grad = True
        out = tp.broadcast_tensors(x, tp.ones([2, 3]))[0]
        out.sum().backward()
        assert x.grad.tolist() == [2.0, 2.0, 2.0]

    def test_grad_sums_size_one_dims(self):
        x = arange_f(3, 1)
        x.requires_grad = True
        y = tp.ones([1, 4])
        out = tp.broadcast_tensors(x, tp.ones([3, 4]))[0]
        _ = y
        (out * 1.0).sum().backward()
        assert x.grad.tolist() == [[4.0], [4.0], [4.0]]


class TestAtLeast:
    def test_atleast_1d(self):
        assert tp.atleast_1d(scalar0()).size(0) == 1
        v = arange_f(3)
        assert tp.atleast_1d(v) is v

    def test_atleast_2d(self):
        assert tuple(tp.atleast_2d(scalar0()).size()) == (1, 1)
        assert tuple(tp.atleast_2d(arange_f(3)).size()) == (1, 3)
        m = arange_f(2, 3)
        assert tp.atleast_2d(m) is m

    def test_atleast_3d(self):
        assert tuple(tp.atleast_3d(scalar0()).size()) == (1, 1, 1)
        assert tuple(tp.atleast_3d(arange_f(3)).size()) == (1, 3, 1)
        assert tuple(tp.atleast_3d(arange_f(2, 3)).size()) == (2, 3, 1)
        c = arange_f(2, 3, 4)
        assert tp.atleast_3d(c) is c

    def test_multi_input_returns_list(self):
        outs = tp.atleast_2d(scalar0(), arange_f(3))
        assert isinstance(outs, list) and len(outs) == 2
        assert tuple(outs[0].size()) == (1, 1)
        assert tuple(outs[1].size()) == (1, 3)


class TestStack:
    def test_hstack_1d(self):
        a, b = arange_f(3), arange_f(3) + 10
        out = tp.hstack([a, b])
        assert tuple(out.size()) == (6,)
        assert out.tolist() == [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]

    def test_hstack_2d(self):
        out = tp.hstack([arange_f(2, 3), arange_f(2, 4) + 100])
        assert tuple(out.size()) == (2, 7)

    def test_hstack_zero_dim(self):
        out = tp.hstack([scalar0(1), scalar0(2)])
        assert tuple(out.size()) == (2,)
        assert out.tolist() == [1.0, 2.0]

    def test_hstack_mixed_ndim_raises(self):
        with pytest.raises(RuntimeError):
            tp.hstack([arange_f(3), arange_f(1, 3)])

    def test_vstack(self):
        out = tp.vstack([arange_f(3), arange_f(3) + 10])
        assert tuple(out.size()) == (2, 3)
        assert out.tolist()[1] == [10.0, 11.0, 12.0]

    def test_vstack_zero_dim(self):
        out = tp.vstack([scalar0(5), scalar0(7)])
        assert tuple(out.size()) == (2, 1)
        assert [r[0] for r in out.tolist()] == [5.0, 7.0]

    def test_row_stack_alias(self):
        a = [arange_f(3), arange_f(3)]
        r1, r2 = tp.row_stack(a), tp.vstack(a)
        assert r1.tolist() == r2.tolist()

    def test_dstack_1d(self):
        out = tp.dstack([arange_f(3), arange_f(3) + 10])
        assert tuple(out.size()) == (1, 3, 2)
        assert out.tolist()[0][0] == [0.0, 10.0]

    def test_dstack_2d(self):
        out = tp.dstack([arange_f(2, 3), arange_f(2, 3) + 100])
        assert tuple(out.size()) == (2, 3, 2)

    def test_column_stack(self):
        out = tp.column_stack([arange_f(3), arange_f(3) + 10])
        assert tuple(out.size()) == (3, 2)
        assert [r[0] for r in out.tolist()] == [0.0, 1.0, 2.0]
        assert [r[1] for r in out.tolist()] == [10.0, 11.0, 12.0]

    def test_column_stack_2d(self):
        out = tp.column_stack([arange_f(2, 3), arange_f(2, 2) + 50])
        assert tuple(out.size()) == (2, 5)

    def test_stack_grad(self):
        a = arange_f(3)
        a.requires_grad = True
        tp.hstack([a, a]).sum().backward()
        assert a.grad.tolist() == [2.0, 2.0, 2.0]


class TestTensorSplit:
    def test_int_sections(self):
        pieces = tp.tensor_split(arange_f(7, 3), 3)
        assert [tuple(p.size()) for p in pieces] == [(3, 3), (2, 3), (2, 3)]

    def test_list_points(self):
        pieces = tp.tensor_split(arange_f(7, 3), [2, 5])
        assert [p.size(0) for p in pieces] == [2, 3, 2]
        assert pieces[0].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

    def test_unsorted_points_clamp(self):
        pieces = tp.tensor_split(arange_f(6), [4, 2])
        assert [p.numel() for p in pieces] == [4, 0, 4]

    def test_over_split_empty_pieces(self):
        pieces = tp.tensor_split(arange_f(2), 4)
        assert [p.numel() for p in pieces] == [1, 1, 0, 0]

    def test_views_alias_input(self):
        base = arange_f(4)
        pieces = tp.tensor_split(base, 2)
        pieces[1][0] = 99.0
        assert base.tolist()[2] == 99.0

    def test_hsplit_vsplit_dsplit(self):
        v = tp.vsplit(arange_f(6, 4), 2)
        assert [tuple(p.size()) for p in v] == [(3, 4), (3, 4)]
        h = tp.hsplit(arange_f(6), 3)
        assert [p.numel() for p in h] == [2, 2, 2]
        d = tp.dsplit(arange_f(2, 3, 4), 2)
        assert [tuple(p.size()) for p in d] == [(2, 3, 2), (2, 3, 2)]

    def test_min_dim_errors(self):
        with pytest.raises(RuntimeError):
            tp.vsplit(arange_f(6), 2)
        with pytest.raises(RuntimeError):
            tp.dsplit(arange_f(2, 3), 2)
        with pytest.raises(RuntimeError):
            tp.hsplit(scalar0(), 1)


class TestTensordot:
    def test_dims_two(self):
        a, b = arange_f(2, 3, 4), arange_f(3, 4, 5)
        out = tp.tensordot(a, b, 2)
        assert tuple(out.size()) == (2, 5)

    def test_dims_pair_matches_reference(self):
        a, b = arange_f(2, 3, 4), arange_f(4, 3, 5)
        out = tp.tensordot(a, b, ([1, 2], [1, 0]))
        ref = tp.einsum("ijk,kjl->il", a, b)
        assert out.tolist() == ref.tolist()

    def test_dims_zero_outer(self):
        out = tp.tensordot(arange_f(2, 3), arange_f(4, 3), 0)
        assert tuple(out.size()) == (2, 3, 4, 3)

    def test_single_dim_lists(self):
        out = tp.tensordot(arange_f(2, 3), arange_f(3, 4), ([1], [0]))
        assert tuple(out.size()) == (2, 4)

    def test_contracted_mismatch_raises(self):
        with pytest.raises(RuntimeError):
            tp.tensordot(arange_f(2, 3, 4), arange_f(4, 3, 5), ([1, 2], [0, 1]))

    def test_negative_int_rejected(self):
        with pytest.raises(RuntimeError):
            tp.tensordot(arange_f(2, 3), arange_f(3, 2), -1)

    def test_grad(self):
        a = arange_f(2, 3)
        a.requires_grad = True
        b = arange_f(3, 4)
        out = tp.tensordot(a, b, 1)
        out.sum().backward()
        assert a.grad.size(0) == 2 and a.grad.size(1) == 3
        # d/da of sum(a @ b) is dOut @ b^T: each row of a.grad is the
        # row-sums of b ([6.0, 22.0, 38.0]).
        row_sums = [sum(r) for r in b.tolist()]
        assert a.grad.tolist() == [list(row_sums) for _ in range(2)]


class TestBlockDiag:
    def test_mixed_blocks(self):
        m = arange_f(2, 2) + 0
        v = tp.arange(2).to(DType.float32) + 1
        s = scalar0(5)
        out = tp.block_diag(m, v, s)
        # diagonal row block, so mixed shapes yield a rectangular result.
        assert tuple(out.size()) == (4, 5)
        expect = [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0],
        ]
        assert out.tolist() == expect

    def test_empty_call(self):
        assert tuple(tp.block_diag().size()) == (1, 0)

    def test_three_dim_raises(self):
        with pytest.raises(RuntimeError):
            tp.block_diag(arange_f(2, 2, 2))

    def test_dtype_promotion(self):
        mi = tp.arange(4).reshape([2, 2])
        mf = arange_f(1, 1)
        out = tp.block_diag(mi, mf)
        assert out.dtype == DType.float32
        assert float(out.tolist()[2][2]) == 0.0

    def test_grad(self):
        v = tp.arange(2).to(DType.float32) + 1
        v.requires_grad = True
        tp.block_diag(v).sum().backward()
        assert v.grad.tolist() == [1.0, 1.0]


class TestUnravelIndex:
    def test_basic(self):
        coords = tp.unravel_index(tp.tensor([0, 5, 11]), (2, 3, 2))
        assert len(coords) == 3
        assert all(c.dtype == DType.int64 for c in coords)
        got = [c.tolist() for c in coords]
        assert got == [[0, 0, 1], [0, 2, 2], [0, 1, 1]]

    def test_roundtrip_with_ravel(self):
        shape = [3, 5, 2]
        flat = tp.arange(30)
        coords = tp.unravel_index(flat, shape)
        back = sum(
            coords[d] * int(__import__("math").prod(shape[d + 1:]))
            for d in range(len(shape))
        )
        assert back.tolist() == flat.tolist()

    def test_negative_wraps(self):
        coords = tp.unravel_index(tp.tensor([-1]), (2, 3))
        assert coords[0].tolist() == [1]
        assert coords[1].tolist() == [2]

    def test_scalar_index_keeps_shape(self):
        coords = tp.unravel_index(tp.tensor([5]).reshape([]), (2, 3))
        assert coords[0].dim() == 0 and coords[1].dim() == 0

    def test_non_tensor_raises(self):
        with pytest.raises(TypeError):
            tp.unravel_index([1, 2], (2, 3))

    def test_empty_shape_raises(self):
        with pytest.raises(RuntimeError):
            tp.unravel_index(tp.tensor([1]), (0,))
