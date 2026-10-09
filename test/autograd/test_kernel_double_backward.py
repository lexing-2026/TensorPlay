"""Backward kernels that a second pass differentiates or refuses.

Under create_graph the first backward pass runs kernels such as
clamp_backward or the dropout backwards on tensors that belong to a graph.
Kernels that are linear in the incoming gradient differentiate to
themselves (or to their adjoint); kernels with no derivative raise when a
second pass needs one instead of handing back a silently missing term.
"""
import numpy as np
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay import _C
from tensorplay.autograd import gradcheck, gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def rand(*shape, device="cpu", seed=0):
    tp.manual_seed(4321 + seed)
    return tp.randn(*shape, dtype=tp.float64).to(device).requires_grad_(True)


def check(fn, tensors, nondet_tol=0.0):
    assert gradgradcheck(fn, tuple(tensors), atol=1e-6, rtol=1e-4, nondet_tol=nondet_tol)


@pytest.mark.parametrize("device", DEVICES)
def test_clamp_differentiates_twice(device):
    x = rand(4, 5, device=device)
    check(lambda t: tp.clamp(t, -0.4, 0.6) * t, [x])
    check(lambda t: tp.clamp(t, min=-0.3) ** 2, [x])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("std", [False, True])
@pytest.mark.parametrize("dim", [None, -1])
def test_variance_differentiates_twice(device, correction, std, dim):
    x = rand(2, 3, device=device, seed=7)

    def fn(value):
        method = value.std if std else value.var
        if dim is None:
            return method(correction=correction)
        return method(dim, correction=correction)

    check(fn, [x])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shapes", [((3, 4), (5, 4)), ((2, 3, 4), (4,)), ((), (3, 2)), ((4,), ())])
def test_inner_differentiates_twice(device, shapes):
    a = rand(*shapes[0], device=device, seed=1)
    b = rand(*shapes[1], device=device, seed=2)
    check(lambda s, t: tp.inner(s, t), [a, b])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shapes", [((2, 3, 4), (4,)), ((), (3, 2)), ((4,), ()), ((2, 0), (3, 0))])
def test_inner_backward_kernels_return_each_operand_shape(device, shapes):
    a = rand(*shapes[0], device=device, seed=5).detach()
    b = rand(*shapes[1], device=device, seed=6).detach()
    grad = tp.ones(*tp.inner(a, b).shape, dtype=tp.float64, device=device)
    da = tp.inner_backward_self(grad, a, b)
    db = tp.inner_backward_other(grad, a, b)
    assert tuple(da.shape) == tuple(a.shape)
    assert tuple(db.shape) == tuple(b.shape)
    if a.dim() == 0:
        np.testing.assert_allclose(da.cpu().numpy(), b.sum().cpu().numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_complex_abs_differentiates_twice(device):
    # |z| curves: its gradient sgn(z) is itself differentiable.
    tp.manual_seed(17)
    z = tp.randn(5, dtype=tp.complex128).to(device).requires_grad_(True)
    check(lambda t: t.abs(), [z])
    check(lambda t: t.abs() ** 2, [z])


@pytest.mark.parametrize("device", DEVICES)
def test_dropout_backwards_differentiate_twice(device):
    x = rand(3, 4, 5, device=device, seed=3)

    def seeded(fn):
        def run(t):
            tp.manual_seed(7)
            return fn(t) * t
        return run

    check(seeded(lambda t: F.dropout(t, 0.3, training=True)), [x])
    check(seeded(lambda t: F.alpha_dropout(t, 0.3, training=True)), [x])
    check(seeded(lambda t: F.feature_dropout(t, 0.4, training=True)), [x])


@pytest.mark.parametrize("device", DEVICES)
def test_rrelu_with_noise_differentiates_twice(device):
    x = rand(4, 6, device=device, seed=4)
    noise = tp.zeros(4, 6, dtype=tp.float64, device=device)

    def fn(t):
        tp.manual_seed(11)
        return tp.rrelu_with_noise(t, noise.clone(), 0.1, 0.3, training=True) * t

    check(fn, [x])


@pytest.mark.parametrize("device", DEVICES)
def test_rrelu_with_noise_backward_keeps_double_precision(device):
    grad = rand(3, 5, device=device, seed=7).detach()
    x = rand(3, 5, device=device, seed=8).detach()
    noise = (tp.rand(3, 5, dtype=tp.float64) * 0.2 + 0.1).to(device)
    got = tp.rrelu_with_noise_backward(grad, x, noise, 0.1, 0.3, True, False)
    np.testing.assert_array_equal(got.cpu().numpy(), (grad * noise).cpu().numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_semi_structured_gather_is_adjoint_of_to_dense(device):
    if device == "cpu":
        rows, cols = 4, 8
    else:
        rows, cols = 32, 64
    tp.manual_seed(5)
    dense = tp.randn(rows, cols, device=device)
    packed, meta = _C._to_sparse_semi_structured(dense)
    p = packed.clone().requires_grad_(True)
    out = _C._sparse_semi_structured_to_dense(p, meta)
    g = tp.randn(rows, cols, device=device).requires_grad_(True)
    (gp,) = tp.autograd.grad(out, p, g, create_graph=True)
    v = tp.randn(*gp.shape, device=device)
    (gg,) = tp.autograd.grad((gp * v).sum(), g)
    expected = _C._sparse_semi_structured_to_dense(v, meta)
    np.testing.assert_allclose(gg.cpu().numpy(), expected.cpu().numpy(), rtol=1e-6, atol=1e-6)


def second_pass(loss, inputs):
    grads = tp.autograd.grad(loss, inputs, create_graph=True)
    total = sum((g * g).sum() for g in grads if g is not None)
    return tp.autograd.grad(total, inputs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("weighted", [False, True])
def test_multi_margin_loss_differentiates_twice(device, p, reduction, weighted):
    x = rand(4, 5, device=device, seed=6)
    target = tp.tensor([0, 2, 4, 2]).to(device)
    w = (tp.rand(5, dtype=tp.float64) + 0.5).to(device) if weighted else None
    check(lambda t: F.multi_margin_loss(t, target, p=p, margin=0.7, weight=w,
                                        reduction=reduction), [x])


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_multi_margin_loss_backward_slots(p, reduction):
    x = rand(3, 4, seed=7)
    target = tp.tensor([1, 0, 3])
    w = rand(4, seed=8).abs().detach().requires_grad_(True)
    go = rand(*([3] if reduction == "none" else []), seed=9)
    red = {"none": 0, "mean": 1, "sum": 2}[reduction]

    def fn(g, t, wt):
        return _C.multi_margin_loss_backward(g, t, target, p, 0.6, wt, red)

    assert gradcheck(fn, (go, x, w), atol=1e-6, rtol=1e-4)
    assert gradgradcheck(fn, (go, x, w), atol=1e-6, rtol=1e-4)


def test_multi_margin_loss_double_backward_shapes():
    for x, target in ((rand(5, seed=10), tp.tensor([3])),
                      (rand(5, seed=10), tp.tensor(3)),
                      (rand(seed=11), tp.tensor(0))):
        check(lambda t: F.multi_margin_loss(t, target, p=2, reduction="none"), [x])
    empty = tp.zeros(0, 3, dtype=tp.float64).requires_grad_(True)
    loss = F.multi_margin_loss(empty, tp.zeros(0, dtype=tp.int64), p=2, reduction="sum")
    (g,) = tp.autograd.grad(loss, [empty], create_graph=True)
    (gg,) = tp.autograd.grad((g * g).sum(), [empty], allow_unused=True)
    assert gg is None or gg.shape == empty.shape


def test_multilabel_margin_loss_differentiates_twice():
    x = rand(2, 5, seed=14)
    target = tp.tensor([[0, 2, -1, -1, -1], [1, 4, 3, -1, -1]])
    loss = F.multilabel_margin_loss(x, target)
    (g,) = tp.autograd.grad(loss, [x], create_graph=True)
    (gg,) = tp.autograd.grad((g * g).sum(), [x])
    assert gg.shape == x.shape


def roi_boxes(device, empty=False):
    if empty:
        return tp.zeros(0, 5, dtype=tp.float64).to(device)
    rows = [[0, 0.3, 0.6, 4.2, 5.1],
            [1, 1.7, 0.2, 7.9, 6.6],
            [0, -1.5, 3.0, 9.5, 8.4],
            [1, 2.0, 2.0, 2.4, 2.6],
            [5, 0.0, 0.0, 3.0, 3.0]]
    return tp.tensor(rows, dtype=tp.float64).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampling_ratio", [0, 2])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_roi_align_differentiates_twice(device, sampling_ratio, aligned, empty):
    x = rand(2, 3, 7, 8, device=device, seed=15)
    rois = roi_boxes(device, empty)

    def fn(t):
        out = _C.roi_align(t, rois, 0.9, 2, 3, sampling_ratio, aligned)
        return out * out

    check(fn, [x], nondet_tol=1e-6)
    go = rand(rois.shape[0], 3, 2, 3, device=device, seed=16)
    back = lambda g: _C.roi_align_backward(g, rois, 0.9, 2, 3, sampling_ratio, aligned,
                                           [2, 3, 7, 8])
    assert gradcheck(back, (go,), atol=1e-6, rtol=1e-4)
    assert gradgradcheck(back, (go,), atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampling_ratio", [0, 2])
@pytest.mark.parametrize("empty", [False, True])
def test_ps_roi_align_differentiates_twice(device, sampling_ratio, empty):
    x = rand(2, 12, 7, 8, device=device, seed=17)
    rois = roi_boxes(device, empty)

    def fn(t):
        out = _C.ps_roi_align(t, rois, 0.9, 2, 3, sampling_ratio)
        return out * out

    check(fn, [x], nondet_tol=1e-6)
    go = rand(rois.shape[0], 2, 2, 3, device=device, seed=18)
    back = lambda g: _C.ps_roi_align_backward(g, rois, 0.9, 2, 3, sampling_ratio,
                                              [2, 12, 7, 8])
    assert gradcheck(back, (go,), atol=1e-6, rtol=1e-4)
    assert gradgradcheck(back, (go,), atol=1e-6, rtol=1e-4)


def pool_boxes(device, empty=False):
    if empty:
        return tp.zeros(0, 5, dtype=tp.float64).to(device)
    rows = [[0, 0.3, 0.6, 4.2, 5.1],
            [1, 1.7, 0.2, 7.9, 6.6],
            [0, -1.5, 3.0, 9.5, 8.4],
            [1, 2.0, 2.0, 2.4, 2.6],
            [0, 0.0, 0.0, 7.0, 7.0],
            [1, 6.0, 6.0, 8.0, 8.0]]
    return tp.tensor(rows, dtype=tp.float64).to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_roi_pool_second_pass_raises(device):
    # The backward scatters each output gradient onto the input's argmax cell,
    # so its own derivative in grad_output would be a gather at those same
    # cells.  Every kernel re-derives argmax positions from the tensor it
    # pools, so no composition computes that gather: the second pass fails
    # loudly instead of silently gathering at the tangent's own argmax.
    x = rand(2, 3, 7, 8, device=device, seed=19)
    rois = pool_boxes(device)

    def pool(t):
        return _C.roi_pool(t, rois, 0.9, 2, 3)

    (g,) = tp.autograd.grad((pool(x) ** 2).sum(), [x], create_graph=True)
    with pytest.raises(NotImplementedError, match="roi_pool_backward"):
        tp.autograd.grad((g * g).sum(), [x])

    go = rand(6, 3, 2, 3, device=device, seed=20)
    back = lambda t: _C.roi_pool_backward(t, x, rois, 0.9, 2, 3)
    with pytest.raises(NotImplementedError, match="roi_pool_backward"):
        tp.autograd.grad(back(go).sum(), [go])


@pytest.mark.parametrize("device", DEVICES)
def test_roi_pool_backward_input_slot_is_a_step_function(device):
    # A maximum makes the pooled value constant inside every window, so the
    # backward sends nothing to the input.
    x = rand(2, 3, 7, 8, device=device, seed=19)
    rois = pool_boxes(device)
    # The incoming gradient is a constant here: differentiating the backward
    # call in grad_output has no derivative and would fail loudly instead.
    go = rand(6, 3, 2, 3, device=device, seed=31).detach()
    back = lambda g: _C.roi_pool_backward(g, x, rois, 0.9, 2, 3)
    (gx,) = tp.autograd.grad(back(go).sum(), [x])
    zeros = np.zeros(tuple(gx.shape))
    np.testing.assert_allclose(gx.detach().cpu().numpy(), zeros)


@pytest.mark.parametrize("device", DEVICES)
def test_ps_roi_pool_second_pass_matches_two_first_order_passes(device):
    # Averaging is linear, so for a pooled tangent w the second pass of
    # (y * y).sum() is 2 * Bᵀ(B w) as well.
    x = rand(2, 12, 7, 8, device=device, seed=21)
    rois = pool_boxes(device)

    def pool(t):
        return _C.ps_roi_pool(t, rois, 0.9, 2, 3)

    w = rand(*x.shape, device=device, seed=31)
    # One ordinary pass gives Bᵀ(B w); the graph pass has to be twice it.
    (expected,) = tp.autograd.grad((pool(x) * pool(w)).sum(), [x])
    (g,) = tp.autograd.grad((pool(x) ** 2).sum(), [x], create_graph=True)
    (gg,) = tp.autograd.grad((g * w).sum(), [x])
    np.testing.assert_allclose(gg.detach().cpu().numpy(),
                               (2 * expected).detach().cpu().numpy(), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_ps_roi_pool_backward_is_the_forward_adjoint(device):
    # Averaging a bin and scattering over the same bin are transposes, so the
    # inner product of an output tangent with the pooled input equals the inner
    # product of the input with the pooled tangent.
    x = rand(2, 12, 7, 8, device=device, seed=23)
    rois = pool_boxes(device)
    y = _C.ps_roi_pool(x, rois, 0.9, 2, 3)
    u = rand(y.shape, device=device, seed=24)
    np.testing.assert_allclose(
        float((y * u).sum()),
        float((x * _C.ps_roi_pool_backward(u, rois, 0.9, 2, 3, [2, 12, 7, 8])).sum()),
        rtol=1e-9, atol=1e-9)


def segment_boundaries(boundary):
    if boundary == "lengths":
        return {"lengths": tp.tensor([[2, 3], [1, 4]], dtype=tp.int64)}
    return {"offsets": tp.tensor([[0, 2, 5], [0, 1, 5]], dtype=tp.int64)}


@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("boundary", ["lengths", "offsets"])
def test_segment_reduce_differentiates_twice(reduce, boundary):
    x = rand(2, 5, 3, seed=25)
    check(lambda t: (tp.segment_reduce(t, reduce, axis=1, **segment_boundaries(boundary)) ** 2).sum(),
          [x])


def test_segment_reduce_empty_segment_second_pass():
    x = rand(2, seed=29)
    lengths = tp.tensor([2, 0], dtype=tp.int64)
    out = tp.segment_reduce(x, "sum", axis=0, lengths=lengths)
    (g,) = tp.autograd.grad(out.sum(), [x], create_graph=True)
    # The full segment scatters the unit gradient; the empty one scatters
    # nothing.  That gradient does not depend on the input, so the pass that
    # differentiates it is zero.
    np.testing.assert_allclose(g.detach().numpy(), [1.0, 1.0])
    (gg,) = tp.autograd.grad(g.sum(), [x])
    np.testing.assert_allclose(gg.detach().numpy(), [0.0, 0.0])


@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("boundary", ["lengths", "offsets"])
def test_segment_reduce_backward_slot_is_the_transpose(reduce, boundary):
    # A linear scatter is its own transpose exactly when the inner product of
    # its output with a tangent equals the inner product of its input with the
    # gradient that pass produces, which is what the formula has to deliver.
    x = rand(2, 5, 3, seed=26)
    kwargs = segment_boundaries(boundary)
    y = tp.segment_reduce(x, reduce, axis=1, **kwargs)

    def back(g):
        return _C._segment_reduce_backward(g, y, x, reduce, lengths=kwargs.get("lengths"),
                                          offsets=kwargs.get("offsets"), axis=1)

    go = rand(y.shape, seed=27)
    v = rand(x.shape, seed=30)
    lhs = float((back(go) * v).sum())
    (transposed,) = tp.autograd.grad((back(go) * v).sum(), [go])
    np.testing.assert_allclose(float((transposed * go).sum()), lhs, rtol=1e-9, atol=1e-9)


def test_segment_reduce_prod_data_slot_raises():
    x = rand(4, seed=28)
    lengths = tp.tensor([2, 2], dtype=tp.int64)
    y = tp.segment_reduce(x, "prod", axis=0, lengths=lengths)
    (g,) = tp.autograd.grad(y.sum(), [x], create_graph=True)
    with pytest.raises(NotImplementedError, match="product reduction"):
        tp.autograd.grad(g.sum(), [x])


def sparse_grid_case(scale=1.0):
    """A (3, 4) COO tensor whose rows hold 2, 2 and 1 stored entries."""
    indices = tp.tensor([[0, 1, 1, 2, 0], [0, 1, 3, 0, 2]])
    values = tp.tensor([1.0, 2.0, -1.5, 4.0, 0.5], dtype=tp.float64) * scale
    return tp.sparse_coo_tensor(indices, values, (3, 4))


def test_sparse_sum_backward_transposes_to_the_sum():
    # Summing row 0 away keeps the columns, so the reduced gradient has one
    # entry per column of the (3, 4) grid.
    x = sparse_grid_case().detach().clone().requires_grad_(True)
    go = tp.sparse_coo_tensor(tp.tensor([[0, 1, 2, 3]]),
                              tp.tensor([10.0, 20.0, 30.0, 40.0], dtype=tp.float64),
                              (4,)).detach().requires_grad_(True)
    v = sparse_grid_case(3.0).detach()
    g = _C._sparse_sum_backward(go, x, [0])
    total = (v.to_dense() * g.to_dense()).sum()
    (gg,) = tp.autograd.grad(total, [go])
    # Each reduced cell collects the incoming gradient of every stored cell
    # that folded into it, which is exactly what summing v over its row gives.
    # Column 0 holds two stored cells (3.0 and 12.0), the rest hold one each.
    expected = _C._sparse_sum(v, [0])
    np.testing.assert_allclose(gg.to_dense().numpy(), [15.0, 6.0, 1.5, -4.5])
    np.testing.assert_allclose(gg.to_dense().numpy(), expected.to_dense().numpy())


def test_sparse_sum_backward_dense_result_scales_with_the_cell_count():
    x = sparse_grid_case().detach().clone().requires_grad_(True)
    go = tp.tensor(2.0, dtype=tp.float64, requires_grad=True)
    g = _C._sparse_sum_backward(go, x, [0, 1])
    (gg,) = tp.autograd.grad(g.to_dense().sum(), [go])
    assert float(gg) == 5.0  # one broadcast cell per stored entry


@pytest.mark.parametrize("dim", [[0], [1], [0, 1]])
def test_sparse_sum_dim_second_pass_runs(dim):
    x = sparse_grid_case().detach().clone().requires_grad_(True)
    out = _C._sparse_sum(x, dim)
    total = out.to_dense().sum() if out.is_sparse else out.sum()
    (g,) = tp.autograd.grad(total, [x], create_graph=True)
    # Both slots of the broadcast are the identity on a zero contraction.
    (gx,) = tp.autograd.grad(g.to_dense().sum(), [x], allow_unused=True)
    assert gx is None or float(gx.to_dense().sum()) == 0.0
    (gu,) = tp.autograd.grad(g.to_dense().sum(), [out], allow_unused=True)
    assert gu is None or float(gu.to_dense().sum() if gu.is_sparse else gu.sum()) == 0.0


def test_ctc_loss_second_pass_runs():
    tp.manual_seed(8)
    log_probs = tp.randn(6, 2, 4, dtype=tp.float64).log_softmax(2).detach().requires_grad_(True)
    targets = tp.tensor([[1, 2], [3, 1]])
    loss = F.ctc_loss(log_probs, targets, tp.tensor([6, 6]), tp.tensor([2, 2]))
    (second,) = second_pass(loss, [log_probs])
    assert second.shape == log_probs.shape
    assert np.isfinite(second.numpy()).all()


def test_multi_output_kernel_second_pass_raises():
    x = rand(1, 1, 4, 4, seed=9)
    w = rand(1, 1, 3, 3, seed=10)
    offset = rand(1, 18, 4, 4, seed=11)
    mask = rand(1, 9, 4, 4, seed=12)
    out = _C.deform_conv2d(x, w, offset, mask, None, [1, 1], [1, 1], [1, 1], 1, 1, True)
    with pytest.raises(NotImplementedError, match="deform_conv2d_backward"):
        second_pass(out.sum(), [x, w, offset, mask])


def test_first_pass_through_undifferentiated_kernel_still_works():
    # Only a pass that needs the missing derivative raises: a gradient taken
    # with create_graph and then used as data is fine.
    tp.manual_seed(12)
    x = tp.randn(6, 2, 4, dtype=tp.float64).log_softmax(2).detach().requires_grad_(True)
    targets = tp.tensor([[1, 2], [3, 1]])

    def loss():
        return F.ctc_loss(x, targets, tp.tensor([6, 6]), tp.tensor([2, 2]))

    (g,) = tp.autograd.grad(loss(), x, create_graph=True)
    assert g.requires_grad
    expected = tp.autograd.grad(loss(), x)[0]
    np.testing.assert_allclose(g.detach().numpy(), expected.numpy())


# ---------------------------------------------------------------------------
# Integration, reductions into slots, covariance
# ---------------------------------------------------------------------------

def sample_points(*shape, device="cpu", seed=0):
    tp.manual_seed(77 + seed)
    steps = tp.rand(*shape, dtype=tp.float64) + 0.2
    return steps.cumsum(-1).to(device).requires_grad_(True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["trapezoid", "cumulative_trapezoid"])
def test_integration_differentiates_in_samples_and_points(device, op):
    fn = getattr(tp, op)
    y = rand(3, 6, device=device, seed=13)
    x_full = sample_points(3, 6, device=device, seed=1)
    x_line = sample_points(6, device=device, seed=2)
    for x in (x_full, x_line):
        assert gradcheck(lambda s, t: fn(s, t, dim=-1), (y, x), atol=1e-6, rtol=1e-4)
        check(lambda s, t: fn(s, t, dim=-1), [y, x])
    check(lambda s: fn(s, dx=0.4, dim=1), [y])


SLOT_REDUCTIONS = ["sum", "prod", "mean", "amax", "amin"]


def distinct(*shape, device="cpu", seed=0):
    # Well-separated values away from zero, so amax/amin have no ties and the
    # products have no zeros; the per-seed offset keeps two tensors that meet
    # in one slot apart as well.
    tp.manual_seed(900 + seed)
    perm = tp.randperm(int(np.prod(shape))).reshape(*shape).to(tp.float64)
    return (perm * 0.37 + 0.5 + 0.013 * seed).to(device).requires_grad_(True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduce", SLOT_REDUCTIONS)
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce_differentiates_twice(device, reduce, include_self):
    self_ = distinct(3, 4, device=device, seed=1)
    src = distinct(5, 4, device=device, seed=2)
    index = tp.tensor([[0, 1, 2, 0], [1, 0, 2, 1], [2, 2, 0, 1], [0, 1, 1, 2], [1, 0, 0, 0]],
                      device=device)
    check(lambda s, t: s.scatter_reduce(0, index, t, reduce, include_self=include_self),
          [self_, src])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduce", ["prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce_differentiates_twice(device, reduce, include_self):
    self_ = distinct(4, 3, device=device, seed=3)
    source = distinct(5, 3, device=device, seed=4)
    index = tp.tensor([0, 2, 1, 0, 3], device=device)
    check(lambda s, t: s.index_reduce(0, index, t, reduce, include_self=include_self),
          [self_, source])


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_prod_with_one_zero_per_slot_differentiates_twice(device):
    self_ = distinct(2, 3, device=device, seed=5)
    src = distinct(3, 3, device=device, seed=6).detach()
    src[1, 2] = 0.0
    src.requires_grad_(True)
    index = tp.tensor([[0, 1, 0], [1, 0, 0], [0, 1, 1]], device=device)
    check(lambda s, t: s.scatter_reduce(0, index, t, "prod"), [self_, src])


def test_scatter_prod_with_zeros_sharing_a_slot_raises_on_second_pass():
    self_ = distinct(2, 3, seed=7)
    src = tp.tensor([[0.0, 2.0, 3.0], [0.0, 5.0, 6.0]], dtype=tp.float64, requires_grad=True)
    index = tp.tensor([[0, 1, 0], [0, 0, 1]])
    out = self_.scatter_reduce(0, index, src, "prod")
    (g,) = tp.autograd.grad(out.sum(), src, create_graph=True)
    with pytest.raises(NotImplementedError, match="more than one zero"):
        tp.autograd.grad((g * g).sum(), src)


@pytest.mark.parametrize("device", DEVICES)
def test_cov_and_corrcoef_differentiate_twice(device):
    x = rand(3, 7, device=device, seed=14)
    fweights = tp.tensor([1, 2, 1, 3, 1, 2, 1], device=device)
    aweights = (tp.rand(7, dtype=tp.float64) + 0.5).to(device).requires_grad_(True)
    check(lambda t: tp.cov(t), [x])
    check(lambda t: tp.cov(t, correction=0, fweights=fweights), [x])
    for correction in (0, 1, 2):
        fn = lambda t, a: tp.cov(t, correction=correction, fweights=fweights, aweights=a)  # noqa: E731
        assert gradcheck(fn, (x, aweights), atol=1e-6, rtol=1e-4)
        check(fn, [x, aweights])
    check(lambda t: tp.corrcoef(t), [x])
    check(lambda t: tp.cov(t), [rand(9, device=device, seed=15)])
