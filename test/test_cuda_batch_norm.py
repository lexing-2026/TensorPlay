"""CUDA batch_norm forward/backward against a closed-form float64 reference.

Training normalizes with the batch statistics and updates the running ones;
evaluation is an affine map per channel whose gradients are a scale of the
incoming gradient.  Half-precision inputs keep float32 parameters, the way
autocast hands them over.
"""

import numpy as np
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")

EPS = 1e-5
MOMENTUM = 0.1


def _reference(x, weight, bias, grad_out, running_mean, running_var, training):
    axes = (0,) + tuple(range(2, x.ndim))
    channel = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    count = x.size // x.shape[1]
    if training:
        mean = x.mean(axes)
        var = x.var(axes)
    else:
        mean, var = running_mean, running_var
    rstd = 1.0 / np.sqrt(var + EPS)
    xhat = (x - mean.reshape(channel)) * rstd.reshape(channel)
    y = xhat * weight.reshape(channel) + bias.reshape(channel)
    dbias = grad_out.sum(axes)
    dweight = (grad_out * xhat).sum(axes)
    g = grad_out * weight.reshape(channel)
    if training:
        dx = rstd.reshape(channel) / count * (
            count * g - g.sum(axes).reshape(channel) - xhat * (g * xhat).sum(axes).reshape(channel)
        )
        new_mean = (1 - MOMENTUM) * running_mean + MOMENTUM * mean
        new_var = (1 - MOMENTUM) * running_var + MOMENTUM * x.var(axes, ddof=1)
    else:
        dx = g * rstd.reshape(channel)
        new_mean, new_var = running_mean, running_var
    return y, dx, dweight, dbias, new_mean, new_var


@pytest.mark.parametrize("shape", [(4, 3, 5, 6), (2, 8, 33), (16, 4, 1, 1)])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
def test_batch_norm_matches_closed_form(shape, training, dtype):
    rng = np.random.default_rng(0)
    c = shape[1]
    x_np = rng.standard_normal(shape)
    w_np = rng.uniform(0.5, 1.5, c)
    b_np = rng.standard_normal(c)
    rm_np = rng.standard_normal(c) * 0.1
    rv_np = rng.uniform(0.5, 2.0, c)
    g_np = rng.standard_normal(shape)
    if dtype == tp.float16:
        # The reference reads the values the kernel reads.
        x_np = x_np.astype(np.float16).astype(np.float64)
        g_np = g_np.astype(np.float16).astype(np.float64)

    x = tp.tensor(x_np, dtype=dtype, device="cuda", requires_grad=True)
    w = tp.tensor(w_np, dtype=tp.float32, device="cuda", requires_grad=True)
    b = tp.tensor(b_np, dtype=tp.float32, device="cuda", requires_grad=True)
    rm = tp.tensor(rm_np, dtype=tp.float32, device="cuda")
    rv = tp.tensor(rv_np, dtype=tp.float32, device="cuda")
    out = F.batch_norm(x, rm, rv, w, b, training=training, momentum=MOMENTUM, eps=EPS)
    out.backward(tp.tensor(g_np, dtype=dtype, device="cuda"))

    y, dx, dw, db, new_mean, new_var = _reference(x_np, w_np, b_np, g_np, rm_np, rv_np, training)
    tol = 1e-4 if dtype == tp.float32 else 2e-2
    assert out.dtype == dtype
    np.testing.assert_allclose(out.detach().float().cpu().numpy(), y, rtol=tol, atol=tol)
    np.testing.assert_allclose(x.grad.float().cpu().numpy(), dx, rtol=tol, atol=tol)
    np.testing.assert_allclose(w.grad.cpu().numpy(), dw, rtol=tol, atol=tol * 10)
    np.testing.assert_allclose(b.grad.cpu().numpy(), db, rtol=tol, atol=tol * 10)
    np.testing.assert_allclose(rm.cpu().numpy(), new_mean, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rv.cpu().numpy(), new_var, rtol=1e-3, atol=1e-3)


def test_batch_norm_trains_under_autocast():
    tp.manual_seed(0)
    model = tp.nn.Sequential(tp.nn.Conv2d(3, 8, 3, padding=1), tp.nn.BatchNorm2d(8), tp.nn.ReLU()).cuda()
    ref = tp.nn.Sequential(tp.nn.Conv2d(3, 8, 3, padding=1), tp.nn.BatchNorm2d(8), tp.nn.ReLU()).cuda()
    ref.load_state_dict(model.state_dict())
    x = tp.randn(4, 3, 10, 10, device="cuda")
    with tp.autocast("cuda", dtype=tp.float16):
        out = model(x)
    out.float().pow(2).mean().backward()
    ref(x).pow(2).mean().backward()
    for p, q in zip(model.parameters(), ref.parameters()):
        assert p.grad is not None
        assert tp.allclose(p.grad, q.grad, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
def test_channels_last_input_made_by_strides(dtype):
    # A channels-last buffer made by a strided allocation -- the way compiled
    # code makes every buffer -- is normalized like one repacked into that
    # order.  The native kernels lay the result out like the input; the
    # library path single precision takes repacks it first.
    tp.manual_seed(0)
    dense = tp.randn(4, 8, 5, 6, device="cuda").to(dtype)
    strided = tp.empty_strided(dense.shape, (240, 1, 48, 8), dtype=dtype, device="cuda")
    strided.copy_(dense)
    w = tp.randn(8, device="cuda")
    b = tp.randn(8, device="cuda")
    outs = []
    for x in (dense, strided):
        rm = tp.zeros(8, device="cuda")
        rv = tp.ones(8, device="cuda")
        outs.append(F.batch_norm(x, rm, rv, w, b, training=True, momentum=MOMENTUM, eps=EPS))
    if dtype == tp.float16:
        assert outs[1].stride() == strided.stride()
    tol = 1e-5 if dtype == tp.float32 else 2e-3
    assert tp.allclose(outs[0].float(), outs[1].float(), atol=tol, rtol=tol)


def test_like_allocations_follow_the_strides():
    t = tp.empty_strided((2, 3, 4, 5), (60, 1, 15, 3), device="cuda")
    assert tp.empty_like(t).stride() == (60, 1, 15, 3)
    t3 = tp.empty_strided((2, 3, 2, 4, 5), (120, 1, 60, 15, 3), device="cuda")
    assert tp.empty_like(t3).stride() == (120, 1, 60, 15, 3)
    target = tp.empty(1, device="cuda")
    target.resize_as_(t, memory_format=tp.preserve_format)
    assert target.stride() == (60, 1, 15, 3)
