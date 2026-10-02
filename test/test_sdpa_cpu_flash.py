"""Blocked cpu flash attention against a float64 NumPy evaluation.

The kernel tiles the query axis (32 rows below 192 tokens, 64 below 768,
256 above) and the key axis (512 columns) and merges key tiles with a
running max and sum, so the shapes below cross every tile boundary: several
query tiles, several key tiles, and causal rows whose last visible key falls
in the middle of a key tile.
"""
import numpy as np
import pytest

import tensorplay as tp
from tensorplay import _C


def _reference(q, k, v, grad_out, causal, mask, scale):
    """Output and dQ/dK/dV in float64; rows with no visible key give zeros."""
    q, k, v, grad_out = (a.astype(np.float64) for a in (q, k, v, grad_out))
    heads, kv_heads = q.shape[1], k.shape[1]
    if heads != kv_heads:
        k = np.repeat(k, heads // kv_heads, axis=1)
        v = np.repeat(v, heads // kv_heads, axis=1)
    tq, skv, dim = q.shape[2], k.shape[2], q.shape[3]
    scale = 1.0 / np.sqrt(dim) if scale is None else scale
    scores = q @ k.transpose(0, 1, 3, 2) * scale
    if causal:
        closed = np.triu(np.ones((tq, skv), dtype=bool), k=1)
        scores = np.where(closed, -np.inf, scores)
    if mask is not None:
        scores = scores + mask.astype(np.float64)
    top = scores.max(axis=-1, keepdims=True)
    dead = np.isneginf(top)
    weights = np.exp(scores - np.where(dead, 0.0, top))
    total = weights.sum(axis=-1, keepdims=True)
    probs = np.where(dead, 0.0, weights / np.where(dead, 1.0, total))
    out = probs @ v
    d_probs = grad_out @ v.transpose(0, 1, 3, 2)
    d_scores = probs * (d_probs - (d_probs * probs).sum(axis=-1, keepdims=True))
    d_q = d_scores @ k * scale
    d_k = d_scores.transpose(0, 1, 3, 2) @ q * scale
    d_v = probs.transpose(0, 1, 3, 2) @ grad_out
    if heads != kv_heads:
        rep = heads // kv_heads
        shape = (k.shape[0], kv_heads, rep, skv, dim)
        d_k = d_k.reshape(shape).sum(axis=2)
        d_v = d_v.reshape(shape).sum(axis=2)
    return out, d_q, d_k, d_v


def _run(q, k, v, grad_out, causal, mask, scale, dtype):
    tq = tp.tensor(q).to(dtype).requires_grad_(True)
    tk = tp.tensor(k).to(dtype).requires_grad_(True)
    tv = tp.tensor(v).to(dtype).requires_grad_(True)
    tm = None if mask is None else tp.tensor(mask)
    out, lse = _C._scaled_dot_product_flash_attention_for_cpu(
        tq, tk, tv, 0.0, causal, attn_mask=tm, scale=scale)
    assert tuple(lse.shape) == q.shape[:3]
    d_q, d_k, d_v = tp.autograd.grad(
        out, [tq, tk, tv], grad_outputs=[tp.tensor(grad_out).to(dtype)])
    return [t.detach().to(tp.float64).numpy() for t in (out, d_q, d_k, d_v)]


def _inputs(seed, batch, heads, kv_heads, tq, skv, dim):
    rng = np.random.RandomState(seed)
    q = rng.randn(batch, heads, tq, dim).astype(np.float32)
    k = rng.randn(batch, kv_heads, skv, dim).astype(np.float32)
    v = rng.randn(batch, kv_heads, skv, dim).astype(np.float32)
    grad_out = rng.randn(batch, heads, tq, dim).astype(np.float32)
    return rng, q, k, v, grad_out


SHAPES = [
    # batch, heads, kv_heads, tq, skv, dim
    (2, 3, 3, 8, 8, 16),
    (2, 4, 4, 49, 49, 32),     # two query tiles
    (1, 2, 2, 70, 33, 8),      # more queries than keys
    (1, 2, 2, 33, 70, 8),      # more keys than queries
    (2, 4, 2, 40, 40, 8),      # grouped key heads
    (1, 1, 1, 200, 600, 4),    # two key tiles, 64-row query tiles
    (1, 1, 1, 800, 530, 4),    # 256-row query tiles
    (1, 1, 1, 1, 1, 1),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mask_kind", ["none", "2d", "4d", "broadcast", "closed_rows"])
def test_matches_float64_evaluation(shape, causal, mask_kind):
    batch, heads, kv_heads, tq, skv, dim = shape
    rng, q, k, v, grad_out = _inputs(0, *shape)
    mask = None
    if mask_kind == "2d":
        mask = (rng.randn(tq, skv) * 2).astype(np.float32)
    elif mask_kind == "4d":
        mask = (rng.randn(batch, heads, tq, skv) * 2).astype(np.float32)
    elif mask_kind == "broadcast":
        mask = (rng.randn(batch, 1, 1, skv) * 2).astype(np.float32)
    elif mask_kind == "closed_rows":
        mask = np.zeros((tq, skv), np.float32)
        mask[rng.rand(tq, skv) < 0.4] = -np.inf
        mask[0, :] = -np.inf
    for scale in (None, 0.37):
        want = _reference(q, k, v, grad_out, causal, mask, scale)
        got = _run(q, k, v, grad_out, causal, mask, scale, tp.float32)
        for name, g, w in zip(("out", "dq", "dk", "dv"), got, want):
            assert g.shape == w.shape, name
            assert not np.isnan(g).any(), name
            err = np.abs(g - w).max() / max(1.0, np.abs(w).max())
            assert err < 2e-5, (name, err)


def test_float64_keeps_double_precision():
    _, q, k, v, grad_out = _inputs(1, 2, 2, 2, 37, 41, 8)
    want = _reference(q, k, v, grad_out, True, None, None)
    got = _run(q, k, v, grad_out, True, None, None, tp.float64)
    for g, w in zip(got, want):
        assert np.abs(g - w).max() < 1e-12


@pytest.mark.parametrize("dtype,tol", [(tp.float16, 2e-2), (tp.bfloat16, 8e-2)])
def test_reduced_precision_inputs(dtype, tol):
    _, q, k, v, grad_out = _inputs(2, 2, 2, 2, 20, 24, 8)
    q16, k16, v16, g16 = (
        tp.tensor(a).to(dtype).to(tp.float32).numpy() for a in (q, k, v, grad_out))
    want = _reference(q16, k16, v16, g16, True, None, None)
    tq = tp.tensor(q).to(dtype).requires_grad_(True)
    out, lse = _C._scaled_dot_product_flash_attention_for_cpu(
        tq, tp.tensor(k).to(dtype), tp.tensor(v).to(dtype), 0.0, True,
        attn_mask=None, scale=None)
    assert out.dtype == dtype
    assert lse.dtype == tp.float32
    (d_q,) = tp.autograd.grad(out, [tq], grad_outputs=[tp.tensor(grad_out).to(dtype)])
    assert d_q.dtype == dtype
    assert np.abs(out.detach().to(tp.float64).numpy() - want[0]).max() < tol
    assert np.abs(d_q.to(tp.float64).numpy() - want[1]).max() < tol


def test_fully_closed_row_gives_zero_output_and_zero_gradient():
    _, q, k, v, grad_out = _inputs(3, 1, 1, 1, 4, 5, 3)
    mask = np.zeros((4, 5), np.float32)
    mask[2, :] = -np.inf
    out, d_q, d_k, d_v = _run(q, k, v, grad_out, False, mask, None, tp.float32)
    np.testing.assert_array_equal(out[0, 0, 2], np.zeros(3))
    np.testing.assert_array_equal(d_q[0, 0, 2], np.zeros(3))


def test_empty_sequence_returns_zeros():
    q = tp.zeros(2, 2, 0, 4)
    kv = tp.zeros(2, 2, 3, 4)
    out, lse = _C._scaled_dot_product_flash_attention_for_cpu(
        q, kv, kv, 0.0, False, attn_mask=None, scale=None)
    assert tuple(out.shape) == (2, 2, 0, 4)
    assert tuple(lse.shape) == (2, 2, 0)


def test_rejects_dropout_and_mismatched_heads():
    q = tp.zeros(1, 3, 4, 4)
    kv = tp.zeros(1, 2, 4, 4)
    with pytest.raises(Exception):
        _C._scaled_dot_product_flash_attention_for_cpu(
            q, q, q, 0.1, False, attn_mask=None, scale=None)
    with pytest.raises(Exception):
        _C._scaled_dot_product_flash_attention_for_cpu(
            q, kv, kv, 0.0, False, attn_mask=None, scale=None)
