"""Pooling kernels read strided operands by value, not by storage order.

A channels-last or otherwise non-row-major operand holds the same values as
its row-major copy, so every pooling call must answer the same for both.
"""
import numpy as np
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F


def _layouts(a):
    t = tp.tensor(a)
    yield "row-major", t
    yield "channels-last", t.contiguous(memory_format=tp.channels_last)
    # A view with a step: neither order.
    wide = tp.tensor(np.repeat(a, 2, axis=3))
    yield "strided", wide[:, :, :, ::2]


def _np(t):
    return t.detach().numpy()


FORWARD = [
    ("avg_pool2d", lambda x: F.avg_pool2d(x, 2)),
    ("avg_pool2d_pad", lambda x: F.avg_pool2d(x, 3, stride=2, padding=1)),
    ("max_pool2d", lambda x: F.max_pool2d(x, 2)),
    ("adaptive_avg_pool2d", lambda x: F.adaptive_avg_pool2d(x, (3, 2))),
    ("adaptive_max_pool2d", lambda x: F.adaptive_max_pool2d(x, (3, 2))),
]


@pytest.mark.parametrize("name,fn", FORWARD)
def test_forward_and_gradient_do_not_depend_on_operand_layout(name, fn):
    rng = np.random.RandomState(0)
    a = rng.randn(4, 6, 8, 8).astype(np.float32)
    want = None
    want_grad = None
    for label, x in _layouts(a):
        x = x.detach().requires_grad_(True)
        out = fn(x)
        weight = tp.tensor(
            np.cos(np.arange(out.numel(), dtype=np.float32)).reshape(tuple(out.shape)))
        (grad,) = tp.autograd.grad((out * weight).sum(), [x])
        if want is None:
            want, want_grad = _np(out), _np(grad)
            continue
        np.testing.assert_allclose(_np(out), want, rtol=0, atol=1e-6, err_msg=label)
        np.testing.assert_allclose(_np(grad), want_grad, rtol=0, atol=1e-6, err_msg=label)


def test_avg_pool2d_backward_with_channels_last_gradient():
    rng = np.random.RandomState(1)
    x = rng.randn(4, 6, 8, 8).astype(np.float32)
    go = rng.randn(4, 6, 4, 4).astype(np.float32)
    op = tp.ops.tp.avg_pool2d_backward.default
    want = _np(op(tp.tensor(go), tp.tensor(x), [2, 2], [2, 2], [0, 0], False, True, None))
    # Each input element belongs to one window of four: its gradient is a
    # quarter of that window's.
    np.testing.assert_allclose(want, np.repeat(np.repeat(go, 2, axis=2), 2, axis=3) / 4.0, atol=1e-7)
    cl = lambda a: tp.tensor(a).contiguous(memory_format=tp.channels_last)
    for g, i in ((cl(go), tp.tensor(x)), (tp.tensor(go), cl(x)), (cl(go), cl(x))):
        got = op(g, i, [2, 2], [2, 2], [0, 0], False, True, None)
        np.testing.assert_allclose(_np(got), want, rtol=0, atol=1e-7)
