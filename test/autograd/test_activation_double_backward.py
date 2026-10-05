"""Activations differentiate twice.

The backward of an activation runs a dedicated kernel (sigmoid_backward,
gelu_backward, ...).  Under create_graph that kernel is itself recorded, so a
second pass reaches both the incoming gradient and the activation's input
through it.  Each case is checked against finite differences of the first
derivative; inputs stay clear of the kinks of the piecewise functions.
"""
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.autograd import gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

POINTS = [[-3.7, -2.2, -0.8, -0.3], [0.2, 0.7, 1.4, 4.1]]

ACTIVATIONS = {
    "sigmoid": tp.sigmoid,
    "tanh": tp.tanh,
    "softmax": lambda x: F.softmax(x, dim=1),
    "log_softmax": lambda x: F.log_softmax(x, dim=1),
    "softmax_dim0": lambda x: F.softmax(x, dim=0),
    "relu": F.relu,
    "threshold": lambda x: F.threshold(x, 0.5, -1.0),
    "elu": F.elu,
    "celu": lambda x: F.celu(x, alpha=0.7),
    "selu": F.selu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "mish": F.mish,
    "hardtanh": F.hardtanh,
    "relu6": F.relu6,
    "leaky_relu": lambda x: F.leaky_relu(x, 0.2),
    "softplus": lambda x: F.softplus(x, beta=2.0, threshold=5.0),
    "hardswish": F.hardswish,
    "hardsigmoid": F.hardsigmoid,
    "logsigmoid": F.logsigmoid,
    "hardshrink": F.hardshrink,
    "softshrink": F.softshrink,
    "glu": lambda x: F.glu(x, dim=1),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_an_activation_differentiates_twice(device, name):
    x = tp.tensor(POINTS, dtype=tp.float64, device=device, requires_grad=True)
    assert gradgradcheck(ACTIVATIONS[name], (x,))


@pytest.mark.parametrize("device", DEVICES)
def test_logit_differentiates_twice_inside_its_band(device):
    x = tp.tensor([0.15, 0.4, 0.65, 0.85], dtype=tp.float64, device=device,
                  requires_grad=True)
    assert gradgradcheck(lambda t: tp.logit(t), (x,))
    assert gradgradcheck(lambda t: tp.logit(t, eps=0.1), (x,))


@pytest.mark.parametrize("device", DEVICES)
def test_the_second_pass_reaches_a_weight_through_the_activation(device):
    # d/dw of d/dx sum(w * sigmoid(x)) is s (1 - s); the path from the
    # first gradient to w runs through sigmoid_backward.
    x = tp.tensor([0.3, -1.2], dtype=tp.float64, device=device, requires_grad=True)
    w = tp.tensor([2.0, 5.0], dtype=tp.float64, device=device, requires_grad=True)
    (gx,) = tp.autograd.grad((w * tp.sigmoid(x)).sum(), x, create_graph=True)
    assert gx.grad_fn is not None
    gw, gxx = tp.autograd.grad(gx.sum(), (w, x))
    s = tp.sigmoid(x.detach())
    assert tp.allclose(gw, s * (1 - s))
    assert tp.allclose(gxx, w.detach() * s * (1 - s) * (1 - 2 * s))


@pytest.mark.parametrize("device", DEVICES)
def test_sigmoid_differentiates_three_times(device):
    x = tp.tensor([0.3, -1.2], dtype=tp.float64, device=device, requires_grad=True)
    (g1,) = tp.autograd.grad(tp.sigmoid(x).sum(), x, create_graph=True)
    (g2,) = tp.autograd.grad(g1.sum(), x, create_graph=True)
    (g3,) = tp.autograd.grad(g2.sum(), x)
    s = tp.sigmoid(x.detach())
    assert tp.allclose(g3, s * (1 - s) * (1 - 6 * s + 6 * s * s))


@pytest.mark.parametrize("device", DEVICES)
def test_a_pass_without_create_graph_records_nothing(device):
    x = tp.tensor([0.3, -1.2], dtype=tp.float64, device=device, requires_grad=True)
    for fn in (F.silu, F.mish, tp.logit):
        (g,) = tp.autograd.grad(fn(x.sigmoid()).sum(), x)
        assert g.grad_fn is None and not g.requires_grad
