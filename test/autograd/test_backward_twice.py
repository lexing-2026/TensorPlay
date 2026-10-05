"""A second backward pass over a graph the first one released.

Releasing a graph frees what its nodes saved, not the links between them.  A
graph that saved nothing can be walked again and accumulates again; one that
saved a tensor fails on the second walk with an error that says why, unless
the first pass kept the graph.
"""
import pytest

import tensorplay as tp
from tensorplay import autograd

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
TWICE = "Trying to backward through the graph a second time"


@pytest.mark.parametrize("device", DEVICES)
def test_a_graph_that_saved_nothing_backwards_twice_and_accumulates(device):
    x = tp.ones(3, device=device, requires_grad=True)
    y = x * 2
    s = y.sum()
    s.backward()
    s.backward()
    assert x.grad.tolist() == [4.0, 4.0, 4.0]


@pytest.mark.parametrize("device", DEVICES)
def test_a_graph_that_saved_nothing_differentiates_twice_with_grad(device):
    x = tp.ones(3, device=device, requires_grad=True)
    s = (x * 2).sum()
    (g1,) = autograd.grad(s, x)
    (g2,) = autograd.grad(s, x)
    assert g1.tolist() == g2.tolist() == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("device", DEVICES)
def test_a_graph_with_saved_tensors_raises_on_the_second_backward(device):
    x = tp.ones(3, device=device, requires_grad=True)
    s = (x * x).sum()
    s.backward()
    with pytest.raises(RuntimeError, match=TWICE) as info:
        s.backward()
    assert "Specify retain_graph=True" in str(info.value)
    assert x.grad.tolist() == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("device", DEVICES)
def test_autograd_grad_raises_on_the_second_pass(device):
    x = tp.ones(3, device=device, requires_grad=True)
    s = (x * x).sum()
    autograd.grad(s, x)
    with pytest.raises(RuntimeError, match=TWICE):
        autograd.grad(s, x)


@pytest.mark.parametrize("device", DEVICES)
def test_retain_graph_allows_both_passes(device):
    x = tp.ones(3, device=device, requires_grad=True)
    s = (x * x).sum()
    s.backward(retain_graph=True)
    s.backward()
    assert x.grad.tolist() == [4.0, 4.0, 4.0]
    t = (x * x).sum()
    autograd.grad(t, x, retain_graph=True)
    (g,) = autograd.grad(t, x)
    assert g.tolist() == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("device", DEVICES)
def test_a_saved_output_raises_on_the_second_pass(device):
    x = tp.ones(3, device=device, requires_grad=True)
    s = tp.exp(x).sum()
    s.backward()
    with pytest.raises(RuntimeError, match=TWICE):
        s.backward()


@pytest.mark.parametrize("device", DEVICES)
def test_a_shared_branch_is_not_released_early(device):
    x = tp.ones(3, device=device, requires_grad=True)
    y = x * x
    (y.sum() + (y * 3).sum()).backward()
    assert x.grad.tolist() == [8.0, 8.0, 8.0]


CONTEXTS = []


class Square(autograd.Function):
    @staticmethod
    def forward(ctx, a):
        CONTEXTS.append(ctx)
        ctx.save_for_backward(a)
        return a * a

    @staticmethod
    def backward(ctx, g):
        (a,) = ctx.saved_tensors
        return 2 * a * g


@pytest.mark.parametrize("device", DEVICES)
def test_custom_function_saved_tensors_raise_after_release(device):
    x = tp.ones(3, device=device, requires_grad=True)
    y = Square.apply(x)
    ctx = CONTEXTS[-1]
    y.sum().backward()
    assert x.grad.tolist() == [2.0, 2.0, 2.0]
    with pytest.raises(RuntimeError, match=TWICE):
        ctx.saved_tensors
    with pytest.raises(RuntimeError, match=TWICE):
        y.sum().backward()


@pytest.mark.parametrize("device", DEVICES)
def test_custom_function_with_retain_graph_keeps_saved_tensors(device):
    x = tp.ones(3, device=device, requires_grad=True)
    y = Square.apply(x)
    s = y.sum()
    s.backward(retain_graph=True)
    s.backward(retain_graph=True)
    assert x.grad.tolist() == [4.0, 4.0, 4.0]
    assert CONTEXTS[-1].saved_tensors[0].tolist() == [1.0, 1.0, 1.0]


@pytest.mark.skipif("cuda" not in DEVICES, reason="needs CUDA")
def test_a_released_graph_is_freed_once_its_outputs_are_dropped():
    x = tp.randn(512, 512, device="cuda", requires_grad=True)

    def run():
        for op in (tp.exp, tp.sigmoid, lambda t: tp.softmax(t, -1)):
            y = op(x)
            y.sum().backward()
            x.grad = None
            del y

    run()
    tp.cuda.synchronize()
    before = tp.cuda.memory_allocated()
    for _ in range(8):
        run()
    tp.cuda.synchronize()
    assert tp.cuda.memory_allocated() == before
