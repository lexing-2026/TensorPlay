#!/usr/bin/env python3
"""Holding a destination-bearing call to the destination it was handed.

A schema may name its destination something other than "out" -- a backward
map spells it "grad_input", a forward map spells it "output" -- while the
public wrapper still calls that slot "out".  The binding takes the
destination under the name the schema gives it, so the forward has to carry
that name; when it carried "out" instead, the destination could not be
given at all.

A schema may write more than one tensor, and the return says which of them
it hands back through the write alias it carries.  The dispatcher took the
first writable argument instead, so a map that draws beside its answer
handed the factors back.

Every assertion here reads the tensor the caller handed in, since that is
what the op contractually fills; the plain call has no destination to read,
so its answer comes from the tensor it returns.
"""
import numpy as np
import pytest
import tensorplay as tp
from tensorplay._ops import NATIVE_NAMESPACE

ops = getattr(tp.ops, NATIVE_NAMESPACE)

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def read(t):
    """The logical elements: a width narrower than single precision is only
    faithfully enumerable as Python numbers."""
    return np.array(t.tolist(), dtype=np.float64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", [
    "sigmoid_backward",
    "tanh_backward",
    "logit_backward",
    "gelu_backward",
])
def test_a_backward_map_fills_the_destination_it_was_handed(device, name):
    # The backward map writes its answer into the destination and hands
    # that same tensor back.
    gradient = tp.tensor([[0.5, -1.0, 2.0]], dtype=tp.float32, device=device)
    answer = tp.tensor([[0.25, 0.75, 0.5]], dtype=tp.float32, device=device)
    destination = tp.zeros_like(gradient)

    returned = getattr(tp, name)(gradient, answer, out=destination)

    assert returned.data_ptr() == destination.data_ptr()
    assert np.allclose(read(returned), read(destination), equal_nan=True)
    # What the plain call answers is what the destination received.
    plain = getattr(tp, name)(gradient, answer)
    assert np.allclose(read(destination), read(plain), equal_nan=True)


@pytest.mark.parametrize("device", DEVICES)
def test_a_forward_map_fills_the_destination_it_was_handed(device):
    # A forward map that spells its destination "output" takes it under
    # that name and still answers the caller.
    source = tp.ones([1, 1, 4, 4, 4], dtype=tp.float32, device=device)
    weight = tp.ones([1, 1, 2, 2, 2], dtype=tp.float32, device=device)
    destination = tp.zeros([1, 1, 3, 3, 3], dtype=tp.float32, device=device)
    address = destination.data_ptr()

    returned = tp.slow_conv3d_forward(source, weight, [2, 2, 2], None,
                                      [1, 1, 1], [0, 0, 0], out=destination)

    assert returned.data_ptr() == address
    plain = tp.slow_conv3d_forward(source, weight, [2, 2, 2], None,
                                   [1, 1, 1], [0, 0, 0])
    assert np.allclose(read(destination), read(plain))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["relu", "celu"])
def test_the_activation_destinations_fill_what_they_were_handed(device, name):
    values = tp.tensor([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=tp.float32,
                        device=device)
    destination = tp.zeros_like(values)
    address = destination.data_ptr()

    returned = getattr(ops, name).out(values, out=destination)

    assert returned.data_ptr() == address
    plain = getattr(tp, name)(values)
    assert np.allclose(read(destination), read(plain))


@pytest.mark.parametrize("device", DEVICES)
def test_the_backward_destination_fills_what_it_was_handed(device):
    gradient = tp.tensor([0.5, -1.0, 2.0], dtype=tp.float32, device=device)
    answer = tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32, device=device)
    destination = tp.zeros_like(gradient)
    address = destination.data_ptr()

    returned = ops.hardswish_backward.out(gradient, answer, out=destination)

    assert returned.data_ptr() == address
    plain = tp.hardswish_backward(gradient, answer)
    assert np.allclose(read(destination), read(plain))
