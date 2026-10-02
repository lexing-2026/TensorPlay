"""Group norm over channels-last storage agrees with a NumPy evaluation.

A channels-last activation is normalized where it lies: the result and the
input gradient come back channels-last, and every value matches a direct
float64 evaluation of the definition.
"""
import numpy as np
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F


def _reference(x, w, b, groups, eps, grad_out):
    n, c, h, wd = x.shape
    x64 = x.astype(np.float64).reshape(n, groups, -1)
    mean = x64.mean(axis=2, keepdims=True)
    var = x64.var(axis=2, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = ((x64 - mean) * inv_std).reshape(n, c, h, wd)
    w64 = w.astype(np.float64).reshape(1, c, 1, 1)
    out = x_hat * w64 + b.astype(np.float64).reshape(1, c, 1, 1)

    go = grad_out.astype(np.float64)
    d_w = (go * x_hat).sum(axis=(0, 2, 3))
    d_b = go.sum(axis=(0, 2, 3))
    d_xhat = (go * w64).reshape(n, groups, -1)
    xh = x_hat.reshape(n, groups, -1)
    m = xh.shape[2]
    d_x = (inv_std / m) * (
        m * d_xhat - d_xhat.sum(axis=2, keepdims=True)
        - xh * (d_xhat * xh).sum(axis=2, keepdims=True))
    return out, d_x.reshape(n, c, h, wd), d_w, d_b


CASES = [
    # n, c, h, w, groups
    (16, 16, 6, 5, 4),
    (16, 24, 4, 4, 8),
    (20, 8, 3, 7, 1),
    (16, 12, 2, 2, 12),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("grad_channels_last", [True, False])
def test_channels_last_group_norm_matches_reference(case, grad_channels_last):
    n, c, h, wd, groups = case
    rng = np.random.RandomState(0)
    x = (rng.randn(n, c, h, wd) * 2.0 + 0.5).astype(np.float32)
    w = rng.randn(c).astype(np.float32)
    b = rng.randn(c).astype(np.float32)
    grad_out = rng.randn(n, c, h, wd).astype(np.float32)
    want_out, want_dx, want_dw, want_db = _reference(x, w, b, groups, 1e-5, grad_out)

    tx = tp.tensor(x).contiguous(memory_format=tp.channels_last)
    tw, tb = tp.tensor(w), tp.tensor(b)
    out = F.group_norm(tx, groups, tw, tb)
    assert out.is_contiguous(memory_format=tp.channels_last)
    assert np.abs(out.numpy() - want_out).max() < 2e-5

    tgo = tp.tensor(grad_out)
    if grad_channels_last:
        tgo = tgo.contiguous(memory_format=tp.channels_last)
    dx, dw, db = tp.ops.tp.group_norm_backward.default(tgo, tx, groups, tw, tb, 1e-5)
    assert dx.is_contiguous(memory_format=tp.channels_last)
    assert np.abs(dx.numpy() - want_dx).max() < 5e-5
    assert np.abs(dw.numpy() - want_dw).max() < 5e-4
    assert np.abs(db.numpy() - want_db).max() < 5e-4


def test_channels_last_matches_row_major_through_autograd():
    rng = np.random.RandomState(1)
    x = rng.randn(16, 16, 5, 5).astype(np.float32)
    gn = tp.nn.GroupNorm(4, 16)
    grads = []
    for channels_last in (False, True):
        t = tp.tensor(x)
        if channels_last:
            t = t.contiguous(memory_format=tp.channels_last)
        t.requires_grad_(True)
        gn.zero_grad()
        (gn(t) ** 2).sum().backward()
        grads.append((t.grad.numpy().copy(), gn.weight.grad.numpy().copy()))
    assert np.abs(grads[0][0] - grads[1][0]).max() < 1e-4
    assert np.abs(grads[0][1] - grads[1][1]).max() < 1e-3
