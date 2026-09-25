"""CUDA group_norm forward/backward against a closed-form float64 reference.

Rows wider than one warp's worth of work reduce across several warps; the
shapes below cover the single-warp, vectorized and scalar kernel paths.
"""

import numpy as np
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")


def _reference(x, weight, bias, grad_out, groups, eps=1e-5):
    n, c = x.shape[:2]
    rows = x.reshape(n, groups, -1)
    mean = rows.mean(2, keepdims=True)
    rstd = 1.0 / np.sqrt(rows.var(2, keepdims=True) + eps)
    xhat = ((rows - mean) * rstd).reshape(x.shape)
    channel = (1, c) + (1,) * (x.ndim - 2)
    y = xhat * weight.reshape(channel) + bias.reshape(channel)
    reduce_dims = (0,) + tuple(range(2, x.ndim))
    dbias = grad_out.sum(reduce_dims)
    dweight = (grad_out * xhat).sum(reduce_dims)
    g = (grad_out * weight.reshape(channel)).reshape(n, groups, -1)
    xh = xhat.reshape(n, groups, -1)
    dx = rstd * (g - g.mean(2, keepdims=True) - xh * (g * xh).mean(2, keepdims=True))
    return y, dx.reshape(x.shape), dweight, dbias


def _rel(actual, expected):
    actual = actual.detach().float().cpu().numpy().astype(np.float64)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


@pytest.mark.parametrize(
    "shape,groups",
    [
        ((4, 16, 5, 5), 4),       # single warp per row, scalar path
        ((8, 64, 28, 28), 8),     # multi-warp rows, vectorized path
        ((2, 384, 7, 7), 8),      # multi-warp rows, odd spatial extent
        ((1, 1, 32, 32), 1),      # one wide row
        ((1, 1, 4097, 1), 1),     # wide row, scalar path
    ],
)
def test_group_norm_matches_closed_form(shape, groups):
    tp.manual_seed(0)
    x = tp.randn(*shape, device="cuda") * 3 + 1
    c = shape[1]
    weight = tp.randn(c, device="cuda")
    bias = tp.randn(c, device="cuda")
    grad_out = tp.randn(*shape, device="cuda")
    expected = _reference(
        *(t.cpu().numpy().astype(np.float64) for t in (x, weight, bias, grad_out)),
        groups,
    )

    xs = x.clone().requires_grad_(True)
    ws = weight.clone().requires_grad_(True)
    bs = bias.clone().requires_grad_(True)
    y = F.group_norm(xs, groups, ws, bs)
    y.backward(grad_out)

    for actual, ref in zip((y, xs.grad, ws.grad, bs.grad), expected):
        assert _rel(actual, ref) < 1e-5


def test_group_norm_mixed_precision_matches_closed_form():
    tp.manual_seed(0)
    shape, groups = (8, 64, 28, 28), 8
    x = (tp.randn(*shape, device="cuda") * 3 + 1).half()
    weight = tp.randn(64, device="cuda")
    bias = tp.randn(64, device="cuda")
    grad_out = tp.randn(*shape, device="cuda")
    expected = _reference(
        *(t.float().cpu().numpy().astype(np.float64) for t in (x, weight, bias, grad_out)),
        groups,
    )

    xs = x.clone().requires_grad_(True)
    ws = weight.clone().requires_grad_(True)
    bs = bias.clone().requires_grad_(True)
    with tp.amp.autocast("cuda"):
        y = F.group_norm(xs, groups, ws, bs)
    y.float().backward(grad_out)

    assert _rel(y, expected[0]) < 1e-4
    assert _rel(xs.grad, expected[1]) < 1e-3
    assert _rel(ws.grad, expected[2]) < 1e-4
    assert _rel(bs.grad, expected[3]) < 1e-4
