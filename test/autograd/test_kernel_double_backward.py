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


def check(fn, tensors):
    assert gradgradcheck(fn, tuple(tensors), atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_clamp_differentiates_twice(device):
    x = rand(4, 5, device=device)
    check(lambda t: tp.clamp(t, -0.4, 0.6) * t, [x])
    check(lambda t: tp.clamp(t, min=-0.3) ** 2, [x])


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

    check(fn, [x])
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

    check(fn, [x])
    go = rand(rois.shape[0], 2, 2, 3, device=device, seed=18)
    back = lambda g: _C.ps_roi_align_backward(g, rois, 0.9, 2, 3, sampling_ratio,
                                              [2, 12, 7, 8])
    assert gradcheck(back, (go,), atol=1e-6, rtol=1e-4)
    assert gradgradcheck(back, (go,), atol=1e-6, rtol=1e-4)


def test_ctc_loss_second_pass_raises():
    tp.manual_seed(8)
    log_probs = tp.randn(6, 2, 4, dtype=tp.float64).log_softmax(2).detach().requires_grad_(True)
    targets = tp.tensor([[1, 2], [3, 1]])
    loss = F.ctc_loss(log_probs, targets, tp.tensor([6, 6]), tp.tensor([2, 2]))
    with pytest.raises(NotImplementedError, match="_ctc_loss_backward"):
        second_pass(loss, [log_probs])


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
