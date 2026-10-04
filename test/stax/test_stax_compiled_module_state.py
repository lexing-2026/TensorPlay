"""A compiled module reads its parameters and buffers as they are at each call.

The region takes the module's tensors as inputs, fetched on every call: a
parameter replaced after compiling, or values loaded into the existing ones,
are what the next call computes with -- in training, with the gradient
reaching the parameter that is there now.
"""

import pytest

import tensorplay as tp
import tensorplay.nn as nn

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        self.register_buffer("shift", tp.zeros(4))

    def forward(self, x):
        return self.body(x) + self.shift


@pytest.mark.parametrize("device", DEVICES)
def test_state_changed_after_compiling_is_read(device):
    tp.manual_seed(0)
    net = _Net().to(device)
    compiled = tp.compile(net)
    x = tp.randn(5, 8, device=device)
    assert tp.allclose(compiled(x), net(x), atol=1e-6)

    # A parameter replaced outright, and a buffer filled in place.
    net.body[2].weight = nn.Parameter(tp.randn(4, 16, device=device))
    net.shift.fill_(3.0)
    assert tp.allclose(compiled(x), net(x), atol=1e-5)

    # Values loaded into the existing tensors.
    other = _Net().to(device)
    net.load_state_dict(other.state_dict())
    assert tp.allclose(compiled(x), other(x), atol=1e-5)

    # The gradient reaches the parameter that is there now.
    compiled(x).sum().backward()
    assert net.body[2].weight.grad is not None
    expected = tp.autograd.grad(other(x).sum(), other.body[2].weight)[0]
    assert tp.allclose(net.body[2].weight.grad, expected, atol=1e-5)
