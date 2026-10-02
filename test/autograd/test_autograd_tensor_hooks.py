"""Gradient hooks on leaf tensors, module backward hooks and node names."""

import tensorplay as tp
from tensorplay.autograd import Function


def _backward_twice(p):
    for _ in range(2):
        p.grad = None
        (p * tp.ones(3)).sum().backward()


def test_leaf_hook_registered_before_any_graph_fires_every_pass():
    p = tp.nn.Parameter(tp.zeros(3))
    calls = []
    p.register_hook(lambda g: calls.append(g.shape))
    _backward_twice(p)
    assert len(calls) == 2


def test_leaf_hook_replaces_gradient_and_can_be_removed():
    p = tp.nn.Parameter(tp.zeros(3))
    handle = p.register_hook(lambda g: g * 3)
    (p * tp.ones(3)).sum().backward()
    assert p.grad.tolist() == [3.0, 3.0, 3.0]

    handle.remove()
    p.grad = None
    (p * tp.ones(3)).sum().backward()
    assert p.grad.tolist() == [1.0, 1.0, 1.0]


def test_post_accumulate_grad_hook_sees_updated_grad_every_pass():
    p = tp.nn.Parameter(tp.zeros(3))
    seen = []
    p.register_post_accumulate_grad_hook(lambda t: seen.append(t.grad.sum().item()))
    _backward_twice(p)
    assert seen == [3.0, 3.0]


def test_module_full_backward_hook_reports_gradients():
    lin = tp.nn.Linear(3, 2)
    seen = []
    lin.register_full_backward_hook(
        lambda mod, grad_in, grad_out: seen.append(
            (tuple(grad_in[0].shape), tuple(grad_out[0].shape))
        )
    )
    x = tp.randn(4, 3, requires_grad=True)
    lin(x).sum().backward()
    assert seen == [((4, 3), (4, 2))]


class _Double(Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, grad):
        return grad * 2


def test_node_names():
    x = tp.randn(3, requires_grad=True)
    assert _Double.apply(x).grad_fn.name() == "_DoubleBackward"
    assert (x * 2).grad_fn.name() == "MulScalarBackward"
