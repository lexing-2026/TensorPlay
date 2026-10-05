"""Losses differentiate in every input, and twice.

Two things are checked against finite differences: the gradient of the
target (it used to be dropped for several losses), and the second derivative
through the loss backward kernels (under create_graph the first backward
runs those kernels on graph tensors).  Inputs stay clear of the kinks of the
piecewise losses.  Losses whose target only picks a branch (hinge embedding,
cosine embedding) are checked for a zero target gradient instead, since a
finite-difference step in the target would flip the branch.
"""
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.autograd import gradcheck, gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
REDUCTIONS = ["none", "mean", "sum"]
F64 = tp.float64


def _rand(shape, device, low=-1.0, high=1.0):
    return (tp.rand(*shape, dtype=F64) * (high - low) + low).to(device)


def _leaf(t):
    return t.detach().clone().requires_grad_(True)


def _pair_away_from_kink(device, near_range=(0.2, 0.7), far_range=(1.5, 2.5)):
    """Prediction and target whose gap falls in one of two ranges, one on each
    side of the loss's transition point."""
    t = _rand((3, 4), device, -1.0, 1.0)
    near = _rand((3, 4), device, *near_range)
    far = _rand((3, 4), device, *far_range)
    mag = tp.where(_rand((3, 4), device, 0.0, 1.0) > 0.5, near, far)
    sign = tp.where(_rand((3, 4), device, 0.0, 1.0) > 0.5, tp.ones_like(mag), -tp.ones_like(mag))
    return _leaf(t + sign * mag), _leaf(t)


def _check(fn, inputs, second=True):
    assert gradcheck(fn, inputs)
    if second:
        assert gradgradcheck(fn, inputs)


# ---------------------------------------------------------------------------
# elementwise losses of (prediction, target)
# ---------------------------------------------------------------------------

DIFF_LOSSES = {
    "mse": lambda x, t, r: F.mse_loss(x, t, reduction=r),
    "l1": lambda x, t, r: F.l1_loss(x, t, reduction=r),
    "smooth_l1": lambda x, t, r: F.smooth_l1_loss(x, t, reduction=r, beta=1.0),
    "smooth_l1_beta": lambda x, t, r: F.smooth_l1_loss(x, t, reduction=r, beta=0.5),
    "huber": lambda x, t, r: F.huber_loss(x, t, reduction=r, delta=1.0),
    "huber_delta": lambda x, t, r: F.huber_loss(x, t, reduction=r, delta=0.5),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("name", sorted(DIFF_LOSSES))
def test_difference_losses(name, reduction, device):
    tp.manual_seed(3)
    if name.endswith("_beta") or name.endswith("_delta"):
        x, t = _pair_away_from_kink(device, (0.1, 0.35), (0.8, 1.5))
    else:
        x, t = _pair_away_from_kink(device)
    fn = lambda a, b: DIFF_LOSSES[name](a, b, reduction)
    _check(fn, (x, t))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("weighted", [False, True])
def test_binary_cross_entropy(reduction, weighted, device):
    tp.manual_seed(4)
    x = _leaf(_rand((3, 4), device, 0.05, 0.95))
    t = _leaf(_rand((3, 4), device, 0.05, 0.95))
    w = _rand((3, 4), device, 0.5, 2.0) if weighted else None
    fn = lambda a, b: F.binary_cross_entropy(a, b, weight=w, reduction=reduction)
    _check(fn, (x, t))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum", "batchmean"])
@pytest.mark.parametrize("log_target", [False, True])
def test_kl_div(reduction, log_target, device):
    tp.manual_seed(5)
    x = _leaf(_rand((3, 4), device, -2.0, 0.0))
    t = _leaf(_rand((3, 4), device, -2.0, 0.0) if log_target
              else _rand((3, 4), device, 0.1, 1.0))
    fn = lambda a, b: F.kl_div(a, b, reduction=reduction, log_target=log_target)
    _check(fn, (x, t))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_soft_margin_loss(reduction, device):
    tp.manual_seed(6)
    x = _leaf(_rand((3, 4), device, -2.0, 2.0))
    t = _leaf(_rand((3, 4), device, -2.0, 2.0))
    fn = lambda a, b: F.soft_margin_loss(a, b, reduction=reduction)
    _check(fn, (x, t))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("log_input,full", [(True, False), (True, True), (False, False), (False, True)])
def test_poisson_nll_loss(log_input, full, reduction, device):
    tp.manual_seed(7)
    if log_input:
        x = _leaf(_rand((3, 4), device, -1.0, 1.0))
    else:
        x = _leaf(_rand((3, 4), device, 0.5, 2.0))
    # targets on both sides of the Stirling cut-off at 1, none near it
    low = _rand((3, 4), device, 0.2, 0.6)
    high = _rand((3, 4), device, 1.8, 4.0)
    t = _leaf(tp.where(_rand((3, 4), device, 0.0, 1.0) > 0.5, high, low))
    fn = lambda a, b: F.poisson_nll_loss(a, b, log_input=log_input, full=full, reduction=reduction)
    _check(fn, (x, t))


# ---------------------------------------------------------------------------
# losses with a margin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_margin_ranking_loss(reduction, device):
    tp.manual_seed(8)
    for _ in range(100):
        x1 = _rand((3, 4), device, -1.0, 1.0)
        x2 = _rand((3, 4), device, -1.0, 1.0)
        t = _rand((3, 4), device, -1.5, 1.5)
        pre = -(x1 - x2) * t + 0.3
        if float(pre.abs().min()) > 0.08:
            break
    fn = lambda a, b, c: F.margin_ranking_loss(a, b, c, margin=0.3, reduction=reduction)
    _check(fn, (_leaf(x1), _leaf(x2), _leaf(t)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_hinge_embedding_loss(reduction, device):
    tp.manual_seed(9)
    x = _rand((3, 4), device, -2.0, 2.0)
    x = tp.where((x - 1.0).abs() < 0.2, x - 0.5, x)
    t = tp.where(_rand((3, 4), device, 0.0, 1.0) > 0.5, tp.ones_like(x), -tp.ones_like(x))
    fn = lambda a: F.hinge_embedding_loss(a, t, margin=1.0, reduction=reduction)
    _check(fn, (_leaf(x),))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_cosine_embedding_loss(reduction, device):
    tp.manual_seed(10)
    for _ in range(100):
        x1 = _rand((4, 3), device, -1.0, 1.0)
        x2 = _rand((4, 3), device, -1.0, 1.0)
        cos = (x1 * x2).sum(1) / ((x1 * x1).sum(1) * (x2 * x2).sum(1)).sqrt()
        if float((cos - 0.2).abs().min()) > 0.08 and float((x1 * x1).sum(1).min()) > 0.2 \
                and float((x2 * x2).sum(1).min()) > 0.2:
            break
    t = tp.tensor([1.0, -1.0, 1.0, -1.0], dtype=F64).to(device)
    fn = lambda a, b: F.cosine_embedding_loss(a, b, t, margin=0.2, reduction=reduction)
    _check(fn, (_leaf(x1), _leaf(x2)))


@pytest.mark.parametrize("device", DEVICES)
def test_branch_selecting_targets_get_zero_gradient(device):
    x = _leaf(_rand((4, 3), device, -1.0, 1.0))
    y = _leaf(_rand((4, 3), device, -1.0, 1.0))
    t = _leaf(tp.tensor([1.0, -1.0, 1.0, -1.0], dtype=F64).to(device))
    F.cosine_embedding_loss(x, y, t, margin=0.1, reduction="sum").backward()
    assert t.grad is not None and float(t.grad.abs().max()) == 0.0

    p = _leaf(_rand((4, 3), device, -2.0, 2.0))
    ht = _leaf(tp.ones(4, 3, dtype=F64).to(device))
    F.hinge_embedding_loss(p, ht, reduction="sum").backward()
    assert ht.grad is not None and float(ht.grad.abs().max()) == 0.0


# ---------------------------------------------------------------------------
# nll_loss: class-index targets are not differentiable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("ignore", [False, True])
def test_nll_loss(reduction, weighted, ignore, device):
    tp.manual_seed(11)
    logp = _leaf(F.log_softmax(_rand((5, 4), device, -2.0, 2.0), dim=1))
    target = tp.tensor([0, 3, 1, 2, 3], dtype=tp.int64).to(device)
    w = _rand((4,), device, 0.5, 2.0) if weighted else None
    ignore_index = 3 if ignore else -100
    fn = lambda a: F.nll_loss(a, target, weight=w, ignore_index=ignore_index, reduction=reduction)
    _check(fn, (logp,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("weighted", [False, True])
def test_nll_loss_spatial(reduction, weighted, device):
    tp.manual_seed(12)
    logp = _leaf(F.log_softmax(_rand((2, 3, 2, 2), device, -2.0, 2.0), dim=1))
    target = tp.tensor([[[0, 2], [1, 1]], [[2, 0], [1, 2]]], dtype=tp.int64).to(device)
    w = _rand((3,), device, 0.5, 2.0) if weighted else None
    fn = lambda a: F.nll_loss(a, target, weight=w, ignore_index=2, reduction=reduction)
    _check(fn, (logp,))


@pytest.mark.parametrize("device", DEVICES)
def test_cross_entropy_double_backward(device):
    tp.manual_seed(13)
    x = _leaf(_rand((5, 4), device, -2.0, 2.0))
    target = tp.tensor([0, 3, 1, 2, 3], dtype=tp.int64).to(device)
    _check(lambda a: F.cross_entropy(a, target), (x,))


# ---------------------------------------------------------------------------
# explicit values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_mse_sum_target_gradient(device):
    x = tp.tensor([1.0, 2.0, -0.5], dtype=F64).to(device).requires_grad_(True)
    t = tp.tensor([0.5, 3.0, 0.5], dtype=F64).to(device).requires_grad_(True)
    F.mse_loss(x, t, reduction="sum").backward()
    expected = -2.0 * (x.detach() - t.detach())
    assert tp.allclose(t.grad, expected)
    assert tp.allclose(x.grad, -expected)


@pytest.mark.parametrize("device", DEVICES)
def test_bce_target_gradient_value(device):
    x = tp.tensor([0.25, 0.8], dtype=F64).to(device).requires_grad_(True)
    t = tp.tensor([0.5, 0.1], dtype=F64).to(device).requires_grad_(True)
    F.binary_cross_entropy(x, t, reduction="sum").backward()
    expected = tp.log(1 - x.detach()) - tp.log(x.detach())
    assert tp.allclose(t.grad, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_soft_margin_gradient_value(device):
    # at x = 0 the slope is -t / 2 whatever the target is
    x = tp.zeros(3, dtype=F64).to(device).requires_grad_(True)
    t = tp.tensor([1.0, -1.0, 2.0], dtype=F64).to(device)
    F.soft_margin_loss(x, t, reduction="sum").backward()
    assert tp.allclose(x.grad, -t / 2)


@pytest.mark.parametrize("device", DEVICES)
def test_cosine_embedding_other_targets_have_no_gradient(device):
    x = _leaf(_rand((3, 4), device, -1.0, 1.0))
    y = _leaf(_rand((3, 4), device, -1.0, 1.0))
    t = tp.zeros(3, dtype=F64).to(device)
    F.cosine_embedding_loss(x, y, t, margin=0.0, reduction="sum").backward()
    assert float(x.grad.abs().max()) == 0.0 and float(y.grad.abs().max()) == 0.0


# ---------------------------------------------------------------------------
# a conversion inside a double-backward formula stays on the graph
# ---------------------------------------------------------------------------

def test_softmax_third_derivative_through_dtype_change():
    tp.manual_seed(14)
    weights = tp.randn(3, 5)

    def chain(x, dtype):
        y = F.softmax(x, -1, dtype=dtype) if dtype is not None else F.softmax(x, -1)
        g1 = tp.autograd.grad(y[..., 0].sum(), x, create_graph=True)[0]
        g2 = tp.autograd.grad((g1 * g1).sum(), x, create_graph=True)[0]
        return tp.autograd.grad((g2 * weights.to(g2.dtype)).sum(), x)[0]

    x = tp.randn(3, 5).to(tp.bfloat16).requires_grad_(True)
    low = chain(x, tp.float32)
    ref = chain(x.detach().float().requires_grad_(True), None)
    assert tp.allclose(low.float(), ref, atol=2e-3, rtol=5e-2)
