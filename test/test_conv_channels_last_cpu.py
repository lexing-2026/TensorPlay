"""Channels-last cpu convolutions agree with a direct NumPy evaluation.

Channels-last activations are handed to the convolution engine in the order
they are stored.  The forward result and both gradients must equal the
row-major results, land in channels-last buffers, and match a direct
evaluation of the definition.
"""
import numpy as np
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay import _C


def _conv_reference(x, w, b, stride, pad):
    n, c_in, h, wd = x.shape
    c_out, _, kh, kw = w.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    h_out = (h + 2 * pad - kh) // stride + 1
    w_out = (wd + 2 * pad - kw) // stride + 1
    out = np.zeros((n, c_out, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            patch = xp[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[:, :, i, j] = np.tensordot(patch, w.astype(np.float64), axes=([1, 2, 3], [1, 2, 3]))
    return out + b.astype(np.float64)[None, :, None, None]


def _grad_reference(x, w, grad_out, stride, pad):
    n, c_in, h, wd = x.shape
    c_out, _, kh, kw = w.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    g_xp = np.zeros_like(xp)
    g_w = np.zeros(w.shape)
    go = grad_out.astype(np.float64)
    for i in range(go.shape[2]):
        for j in range(go.shape[3]):
            sl = (slice(None), slice(None), slice(i * stride, i * stride + kh),
                  slice(j * stride, j * stride + kw))
            g_w += np.tensordot(go[:, :, i, j], xp[sl], axes=([0], [0]))
            g_xp[sl] += np.tensordot(go[:, :, i, j], w.astype(np.float64), axes=([1], [0]))
    g_x = g_xp[:, :, pad:pad + h, pad:pad + wd]
    return g_x, g_w


CASES = [
    # n, c_in, h, w, c_out, k, stride, pad
    (16, 16, 9, 9, 16, 3, 1, 1),
    (16, 24, 8, 8, 8, 3, 1, 1),
    (16, 8, 8, 8, 24, 1, 1, 0),
    (16, 16, 9, 9, 16, 3, 2, 1),
    (16, 1, 10, 10, 8, 3, 1, 1),
    (16, 3, 7, 7, 5, 3, 1, 1),
]


@pytest.mark.parametrize("case", CASES)
def test_channels_last_matches_reference(case):
    n, c_in, h, wd, c_out, k, stride, pad = case
    rng = np.random.RandomState(0)
    x = rng.randn(n, c_in, h, wd).astype(np.float32)
    w = rng.randn(c_out, c_in, k, k).astype(np.float32)
    b = rng.randn(c_out).astype(np.float32)
    want = _conv_reference(x, w, b, stride, pad)
    grad_out = rng.randn(*want.shape).astype(np.float32)
    want_gx, want_gw = _grad_reference(x, w, grad_out, stride, pad)

    for channels_last in (False, True):
        def lay(a):
            t = tp.tensor(a)
            return t.contiguous(memory_format=tp.channels_last) if channels_last else t

        tx, tw, tgo = lay(x), lay(w), lay(grad_out)
        out = F.conv2d(tx, tw, tp.tensor(b), stride=stride, padding=pad)
        gx = _C.conv2d_grad_input(tgo, tx, tw, [stride, stride], [pad, pad], [1, 1], 1)
        gw = _C.conv2d_grad_weight(tgo, tx, tw, [stride, stride], [pad, pad], [1, 1], 1)
        if channels_last:
            assert out.is_contiguous(memory_format=tp.channels_last)
            assert gx.is_contiguous(memory_format=tp.channels_last)
        for got, ref in ((out, want), (gx, want_gx), (gw, want_gw)):
            assert tuple(got.shape) == ref.shape
            err = np.abs(got.numpy() - ref).max() / max(1.0, np.abs(ref).max())
            assert err < 2e-5, (channels_last, err)


def test_mixed_layout_gradient_operands():
    # A row-major gradient against a channels-last input: the gradient is
    # brought to the input's layout and the results are unchanged.
    rng = np.random.RandomState(1)
    x = rng.randn(16, 16, 6, 6).astype(np.float32)
    w = rng.randn(16, 16, 3, 3).astype(np.float32)
    go = rng.randn(16, 16, 6, 6).astype(np.float32)
    want_gx, want_gw = _grad_reference(x, w, go, 1, 1)
    tx = tp.tensor(x).contiguous(memory_format=tp.channels_last)
    gx = _C.conv2d_grad_input(tp.tensor(go), tx, tp.tensor(w), [1, 1], [1, 1], [1, 1], 1)
    gw = _C.conv2d_grad_weight(tp.tensor(go), tx, tp.tensor(w), [1, 1], [1, 1], [1, 1], 1)
    assert np.abs(gx.numpy() - want_gx).max() < 2e-4
    assert np.abs(gw.numpy() - want_gw).max() < 2e-3
