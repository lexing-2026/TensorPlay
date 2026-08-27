"""Spec tests: native scatter_reduce / index_reduce vs local torch 2.13.

The fused reduce-scatter family was ported down to real CPU kernels
(p10/src/backend/cpu/IndexingKernels.cpp) with CUDA ports pending remote
compilation. Forward mirrors ATen scatter_impl +
scatter_reduce_exclude_self_helper + the mean count epilogue; backward
mirrors FunctionsManual.cpp scatter_reduce_backward / index_reduce_backward.
Covers all five reduce ops x include_self on/off, multi-inner layouts,
negative dims/indices, autograd through both self and src/source, and the
Tensor method bindings.
"""

import pytest
import torch

import tensorplay as tp


def _tp(t):
    return tp.tensor(t.tolist())


REDUCES = ["sum", "prod", "mean", "amin", "amax"]


def _cases():
    # (self shape/values builder, index, src, dim)
    yield (
        torch.arange(8.0).reshape(2, 4),
        torch.tensor([[0, 2], [3, 1]]),
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        1,
    )
    # multi-inner along dim
    yield (
        torch.arange(24.0).reshape(2, 3, 4),
        torch.tensor([[[0, 1], [2, 0]]]),
        torch.arange(4.0).reshape(1, 2, 2) * 0.25 + 0.5,
        1,
    )
    # negative dim + duplicate indices across rows
    yield (
        torch.arange(6.0).reshape(3, 2),
        torch.tensor([[1, 0], [2, 2]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        -2,
    )
    # collisions: many sources into one slot
    yield (
        torch.ones(4) * 0.75,
        torch.tensor([1, 1, 1, 2]),
        torch.tensor([1.25, 0.5, 2.0, 3.0]),
        0,
    )


@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce_forward(reduce, include_self):
    for self_t, idx_t, src_t, dim in _cases():
        ref = self_t.clone()
        got = ref.scatter_reduce(
            dim, idx_t, src_t, reduce=reduce, include_self=include_self
        )
        ours = tp.scatter_reduce(
            _tp(self_t), dim, _tp(idx_t), _tp(src_t), reduce,
            include_self=include_self)
        assert close(got, ours)


def _ir_cases():
    # torch index_reduce contract: index is 1-D; source has self's rank with
    # source.size(dim) == index.numel() and equal sizes elsewhere.
    yield (
        torch.arange(8.0).reshape(2, 4),
        1,
        torch.tensor([0, 2]),
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
    )
    yield (
        torch.arange(24.0).reshape(2, 3, 4),
        1,
        torch.tensor([0, 0, 2]),
        torch.arange(24.0).reshape(2, 3, 4) * 0.25 + 0.5,
    )
    yield (
        torch.arange(6.0).reshape(3, 2),
        0,
        torch.tensor([1, 1]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    )


IR_REDUCES = ["prod", "mean", "amin", "amax"]


@pytest.mark.parametrize("reduce", IR_REDUCES)
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce_forward(reduce, include_self):
    for self_t, dim, idx_t, src_t in _ir_cases():
        ref = self_t.clone()
        got = ref.index_reduce(
            dim, idx_t, src_t, reduce=reduce, include_self=include_self
        )
        ours = tp.index_reduce(
            _tp(self_t), dim, _tp(idx_t), _tp(src_t), reduce,
            include_self=include_self)
        assert close(got, ours)


def close(ref_tp_tensor, ours, tol=1e-4):
    if isinstance(ref_tp_tensor, torch.Tensor):
        ref_tp_tensor = ref_tp_tensor.tolist()
    return _nested_close(ref_tp_tensor, ours.tolist(), tol)


def _nested_close(a, b, tol):
    if isinstance(a, list):
        return len(a) == len(b) and all(
            _nested_close(x, y, tol) for x, y in zip(a, b)
        )
    if a != a and b != b:  # NaN == NaN
        return True
    return abs(a - b) <= tol * max(1.0, abs(a))


@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce_gradients(reduce, include_self):
    for self_t, idx_t, src_t, dim in _cases():
        r_self = self_t.clone().requires_grad_()
        r_src = src_t.clone().requires_grad_()
        out = r_self.scatter_reduce(
            dim, idx_t, r_src, reduce=reduce, include_self=include_self
        )
        loss = (out * out).sum()
        loss.backward()

        o_self = _tp(self_t).requires_grad_(True)
        o_src = _tp(src_t).requires_grad_(True)
        o_out = tp.scatter_reduce(
            o_self, dim, _tp(idx_t), o_src, reduce,
            include_self=include_self)
        (o_out * o_out).sum().backward()

        assert close(r_self.grad, o_self.grad), f"self grad {reduce}"
        assert close(r_src.grad, o_src.grad), f"src grad {reduce}"


@pytest.mark.parametrize("reduce", IR_REDUCES)
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce_gradients(reduce, include_self):
    for self_t, dim, idx_t, src_t in _ir_cases():
        r_self = self_t.clone().requires_grad_()
        r_src = src_t.clone().requires_grad_()
        out = r_self.index_reduce(
            dim, idx_t, r_src, reduce=reduce, include_self=include_self
        )
        (out * out).sum().backward()

        o_self = _tp(self_t).requires_grad_(True)
        o_src = _tp(src_t).requires_grad_(True)
        o_out = tp.index_reduce(
            o_self, dim, _tp(idx_t), o_src, reduce,
            include_self=include_self)
        (o_out * o_out).sum().backward()

        assert close(r_self.grad, o_self.grad)
        assert close(r_src.grad, o_src.grad)


def test_index_reduce_rejects_sum_and_ndim_index_like_torch():
    s234 = tp.tensor(torch.arange(24.0).reshape(2, 3, 4).tolist())
    v234 = tp.tensor((torch.arange(24.0).reshape(2, 3, 4) * 0.25 + 0.5).tolist())
    with pytest.raises(Exception):
        tp.index_reduce(s234, 1, tp.tensor([0, 2]), v234, "sum")
    with pytest.raises(Exception):
        tp.index_reduce(s234, 1,
                        _tp(torch.tensor([[0, 1], [2, 0]])), v234, "mean")


def test_prod_gradient_zero_handling():
    # zeros scattered to the same slot: single-zero special case must route
    # gradient around the zeroed factor exactly like FunctionsManual.
    self_t = torch.tensor([2.0, 3.0, 4.0])
    idx = torch.tensor([0, 0, 1])
    src_t = torch.tensor([0.0, 5.0, 2.0])
    r_src = src_t.clone().requires_grad_()
    out = self_t.scatter_reduce(0, idx, r_src, reduce="prod",
                                include_self=True)
    out.sum().backward()

    o_src = _tp(src_t)
    o_src.requires_grad = True
    o_out = tp.scatter_reduce(
        _tp(self_t), 0, _tp(idx), o_src, "prod", include_self=True)
    o_out.sum().backward()
    assert close(r_src.grad, o_src.grad)


def test_include_self_false_keeps_unwritten_positions():
    # ATen resets only the indexed slices; untouched positions keep self.
    self_t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    idx = torch.tensor([0])
    src_t = torch.tensor([7.0])
    for reduce, expect_identity in [
        ("prod", None), ("amin", None), ("amax", None)
    ]:
        ref = self_t.scatter_reduce(0, idx, src_t, reduce=reduce,
                                    include_self=False)
        ours = tp.scatter_reduce(
            _tp(self_t), 0, _tp(idx), _tp(src_t), reduce,
            include_self=False)
        # positions 1..3 must remain 2.,3.,4. regardless of identity value
        assert ref[1].item() == 2.0 and ref[3].item() == 4.0
        assert close(ref, ours)


def test_mean_int_dtype_floor_division():
    # torch: integral mean divides with floor rounding
    self_t = torch.tensor([[7, 5], [3, 9]], dtype=torch.int64)
    idx = torch.tensor([[0, 1], [0, 1]])
    src_t = torch.tensor([[2, 4], [3, 1]], dtype=torch.int64)
    for include_self in (True, False):
        ref = self_t.clone().scatter_reduce(
            1, idx, src_t, reduce="mean", include_self=include_self)
        ours = tp.scatter_reduce(
            tp.tensor(self_t.tolist()), 1, tp.tensor(idx.tolist()),
            tp.tensor(src_t.tolist()), "mean", include_self=include_self)
        assert ref.tolist() == ours.tolist()


def test_method_binding_and_function_alias():
    x = tp.tensor([1.0, 2.0, 3.0])
    got = x.scatter_reduce(0, tp.tensor([1]), tp.tensor([10.0]), "sum")
    assert isinstance(got, tp.Tensor)
    assert tp.scatter_reduce is not None and tp.index_reduce is not None


def test_bad_reduce_raises():
    with pytest.raises(Exception):
        tp.scatter_reduce(
            tp.tensor([1.0]), 0, tp.tensor([0]), tp.tensor([1.0]), "avg")


def test_out_of_range_index_raises():
    with pytest.raises(Exception):
        tp.scatter_reduce(
            tp.tensor([1.0, 2.0]), 0, tp.tensor([5]), tp.tensor([1.0]), "sum")


def test_negative_index_rejected_like_torch():
    # torch rejects negative indices in the scatter family
    with pytest.raises(Exception):
        tp.scatter_reduce(
            tp.tensor([1.0, 2.0, 3.0]), 0, tp.tensor([-1]),
            tp.tensor([5.0]), "sum")
    with pytest.raises(Exception):
        tp.index_reduce(
            tp.tensor([1.0, 2.0, 3.0]), 0, tp.tensor([-1]),
            tp.tensor([5.0]), "sum")
