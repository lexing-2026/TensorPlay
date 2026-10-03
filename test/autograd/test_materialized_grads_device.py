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
