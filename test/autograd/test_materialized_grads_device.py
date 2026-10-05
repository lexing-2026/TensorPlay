"""Gradients the engine fills in for outputs nobody differentiated.

A function with several outputs is handed a zero gradient for each output the
loss never read.  The zeros are made from what was recorded about that output
-- its shape, type and device -- so they live where the output lived.
"""
import pytest

import tensorplay as tp
from tensorplay.autograd import Function

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


class TwoOutputs(Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2, x * 3

    @staticmethod
    def backward(ctx, first, second):
        assert first.device == second.device
        return first * 2 + second * 3


@pytest.mark.parametrize("device", DEVICES)
def test_unread_outputs_get_zeros_on_their_device(device):
    x = tp.tensor([1.0, -2.0, 0.5], device=device, requires_grad=True)
    first, _second = TwoOutputs.apply(x)
    first.sum().backward()
    assert x.grad.device == x.device
    assert x.grad.tolist() == [2.0, 2.0, 2.0]


class DropsGradient(Function):
    """Passes its input through and hands back no gradient for it."""

    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return None


@pytest.mark.parametrize("op", [
    lambda x: x.t(),
    lambda x: x.sum(0),
    lambda x: x.reshape(6),
    lambda x: x @ x.t(),
    lambda x: tp.linalg.pinv(x),
], ids=["t", "sum", "reshape", "mm", "pinv"])
def test_a_missing_gradient_never_takes_the_input_shape(op):
    # The node below a function that returned no gradient must not be handed
    # zeros shaped like its own input: for a shape-changing op that is the
    # wrong shape, and it used to reach the leaf or fail inside the formula.
    x = tp.tensor([[1.5, 0.3], [0.4, 2.0], [-0.3, 0.2]], dtype=tp.float64,
                  requires_grad=True)
    out = DropsGradient.apply(op(x))
    (grad,) = tp.autograd.grad(out, x, tp.ones_like(out), allow_unused=True)
    assert grad is None or (tuple(grad.shape) == (3, 2) and not bool(grad.any()))


@pytest.mark.parametrize("kind", ["lstm", "gru", "rnn_tanh"])
def test_a_recurrent_layer_read_only_through_its_final_state(kind):
    # Only the final hidden state is read, so the sequence output's gradient
    # is missing; the layer differentiates through the hidden state alone.
    tp.manual_seed(0)
    steps, batch, features, hidden = 4, 2, 3, 5
    gates = {"lstm": 4, "gru": 3, "rnn_tanh": 1}[kind]
    x = tp.randn(steps, batch, features, dtype=tp.float64, requires_grad=True)
    params = [
        tp.randn(gates * hidden, features, dtype=tp.float64, requires_grad=True),
        tp.randn(gates * hidden, hidden, dtype=tp.float64, requires_grad=True),
        tp.randn(gates * hidden, dtype=tp.float64, requires_grad=True),
        tp.randn(gates * hidden, dtype=tp.float64, requires_grad=True),
    ]
    h0 = tp.zeros(1, batch, hidden, dtype=tp.float64)

    def run():
        if kind == "lstm":
            out, hn, _ = tp._C.lstm(x, [h0, h0], params, True, 1, 0.0, True, False, False)
        else:
            out, hn = getattr(tp._C, kind)(x, h0, params, True, 1, 0.0, True, False, False)
        return out, hn

    _, hn = run()
    only_state = tp.autograd.grad(hn.sum(), [x] + params)
    out, hn = run()
    explicit = tp.autograd.grad([out, hn], [x] + params,
                                [tp.zeros_like(out), tp.ones_like(hn)])
    for got, want in zip(only_state, explicit):
        assert tuple(got.shape) == tuple(want.shape)
        assert bool(tp.allclose(got, want))
    assert bool(only_state[0].any())
