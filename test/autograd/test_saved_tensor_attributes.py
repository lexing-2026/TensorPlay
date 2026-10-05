"""Inputs a backward formula only measures are not kept.

When a derivative reads nothing of an input but its shape, type or element
count, the node holds those attributes instead of the tensor.  The input may
then be updated in place before the backward pass, as long as the update
leaves the measured attributes alone, and its memory is not pinned by the
graph.
"""
import pytest

import tensorplay as tp

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

MEASURING = {
    "sum": lambda a: a.sum(),
    "sum_dim": lambda a: a.sum(0) * tp.arange(2, dtype=a.dtype, device=a.device),
    "mean_dim": lambda a: a.mean(1, keepdim=True) * 4,
    "view": lambda a: a.view(4).cumsum(0),
    "reshape": lambda a: a.reshape(4, 1).sum(),
    "expand": lambda a: a.expand(3, 2, 2).sum(),
    "repeat": lambda a: a.repeat(2, 1).sum(),
}
EXPECTED = {
    "sum": [[2.0, 2.0], [2.0, 2.0]],
    "sum_dim": [[0.0, 2.0], [0.0, 2.0]],
    "mean_dim": [[4.0, 4.0], [4.0, 4.0]],
    "view": [[8.0, 6.0], [4.0, 2.0]],
    "reshape": [[2.0, 2.0], [2.0, 2.0]],
    "expand": [[6.0, 6.0], [6.0, 6.0]],
    "repeat": [[4.0, 4.0], [4.0, 4.0]],
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(MEASURING))
def test_an_input_that_is_only_measured_may_change_in_place(device, name):
    x = tp.ones(2, 2, dtype=tp.float64, device=device, requires_grad=True)
    a = x * 2
    out = MEASURING[name](a)
    a.add_(1)
    out.sum().backward()
    assert x.grad.tolist() == EXPECTED[name]


@pytest.mark.parametrize("device", DEVICES)
def test_softmax_keeps_its_output_and_only_the_inputs_type(device):
    x = tp.tensor([[0.5, -1.0, 2.0]], dtype=tp.float64, device=device, requires_grad=True)
    scores = x * 1
    probs = tp.softmax(scores, -1)
    scores.zero_()
    (g,) = tp.autograd.grad((probs * tp.tensor([[1.0, 2.0, 3.0]], dtype=tp.float64,
                                                device=device)).sum(), x)
    p = tp.softmax(x.detach(), -1)
    w = tp.tensor([[1.0, 2.0, 3.0]], dtype=tp.float64, device=device)
    assert tp.allclose(g, p * (w - (p * w).sum(-1, keepdim=True)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["mul", "matmul", "div", "mul_"])
def test_a_factor_only_the_constant_side_reads_is_not_kept(device, op):
    # The gradient of `a` reads only the constant factor, so `a` itself is not
    # kept and may change in place before the backward pass.
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float64, device=device,
                  requires_grad=True)
    c = tp.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=tp.float64, device=device)
    a = x * 1
    if op == "mul":
        out, want = a * c, c
    elif op == "matmul":
        out, want = a.mm(c), tp.ones(2, 2, dtype=tp.float64, device=device).mm(c.t())
    elif op == "div":
        out, want = a / c, 1 / c
    else:
        out = a.clone()
        out.mul_(c)
        want = c
    a.add_(10.0)
    out.sum().backward()
    assert tp.allclose(x.grad, want)


@pytest.mark.parametrize("device", DEVICES)
def test_both_factors_are_kept_when_both_want_gradients(device):
    x = tp.tensor([1.0, 2.0], dtype=tp.float64, device=device, requires_grad=True)
    w = tp.tensor([3.0, 5.0], dtype=tp.float64, device=device, requires_grad=True)
    a = x * 1
    out = a * w
    a.add_(1.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()
    out = (x * 1) * w
    out.sum().backward()
    assert x.grad.tolist() == [3.0, 5.0] and w.grad.tolist() == [1.0, 2.0]


@pytest.mark.skipif(not tp.cuda.is_available(), reason="measures device memory")
def test_a_dropped_output_does_not_keep_its_graph_alive():
    # exp, sigmoid and softmax keep their own output for the backward pass;
    # dropping the output without a backward must free it.
    x = tp.randn(512, 512, device="cuda", requires_grad=True)
    tp.cuda.synchronize()
    before = tp.cuda.memory_allocated()
    for _ in range(8):
        for op in (tp.exp, tp.sigmoid, lambda t: tp.softmax(t, -1)):
            y = op(x)
            del y
    tp.cuda.synchronize()
    assert tp.cuda.memory_allocated() == before


@pytest.mark.parametrize("device", DEVICES)
def test_a_saved_output_still_carries_the_second_derivative(device):
    x = tp.tensor([0.5, -1.0], dtype=tp.float64, device=device, requires_grad=True)
    (g,) = tp.autograd.grad(tp.exp(x).sum(), x, create_graph=True)
    (gg,) = tp.autograd.grad(g.sum(), x)
    assert tp.allclose(gg, tp.exp(x.detach()))


@pytest.mark.parametrize("device", DEVICES)
def test_reductions_over_dimensions_differentiate_twice(device):
    x = tp.tensor([[0.5, -1.0], [2.0, 1.5]], dtype=tp.float64, device=device,
                  requires_grad=True)
    for reduce, scale in ((lambda t: t.sum(0), 1.0), (lambda t: t.mean(1), 0.5)):
        (g,) = tp.autograd.grad(reduce(x ** 3).sum(), x, create_graph=True)
        assert tp.allclose(g, 3 * scale * x.detach() ** 2)
        (gg,) = tp.autograd.grad(g.sum(), x)
        assert tp.allclose(gg, 6 * scale * x.detach())
