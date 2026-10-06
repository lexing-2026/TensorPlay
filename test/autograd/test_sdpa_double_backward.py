"""scaled_dot_product_attention differentiates twice at dropout_p == 0.

Both backward spellings -- the math one and the log-sum-exp one -- are
differentiated with respect to the incoming gradient, the query, the key,
the value, the mask and, for the log-sum-exp spelling, the saved output
and the constant.  End to end the math spelling is reached through the
math backend, the log-sum-exp one by a plain CUDA float32 call, and a
non-default implementation id falls back to the composed math.  A call
that dropped probabilities still refuses a second derivative.
"""
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.autograd import gradgradcheck
from tensorplay.functional import (
    _scaled_dot_product_attention_backward_with_lse,
    _scaled_dot_product_attention_with_lse,
    scaled_dot_product_attention_backward,
)

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
CUDA = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA only")


def leaf(*shape, device, seed):
    tp.manual_seed(2024 + seed)
    t = tp.rand(*shape, dtype=tp.float64) * 2 - 1
    return t.to(device).requires_grad_(True)


def leaf_f32(*shape, device, seed):
    tp.manual_seed(2024 + seed)
    t = tp.rand(*shape, dtype=tp.float32) * 2 - 1
    return t.to(device).requires_grad_(True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_math_backward_differentiates_twice(device, is_causal):
    # The backward op itself with every tensor input a leaf: the incoming
    # gradient, the three operands and an additive mask, over a context
    # longer than the query.
    grad = leaf(2, 3, 5, 4, device=device, seed=1)
    q = leaf(2, 3, 5, 4, device=device, seed=2)
    k = leaf(2, 3, 6, 4, device=device, seed=3)
    v = leaf(2, 3, 6, 4, device=device, seed=4)
    mask = leaf(5, 6, device=device, seed=5)

    def fn(g, query, key, value, attn_mask):
        return scaled_dot_product_attention_backward(
            g, query, key, value, attn_mask=attn_mask, is_causal=is_causal)

    assert gradgradcheck(fn, (grad, q, k, v, mask), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_math_backward_differentiates_twice_with_scale(device):
    grad = leaf(1, 2, 4, 3, device=device, seed=6)
    q = leaf(1, 2, 4, 3, device=device, seed=7)
    k = leaf(1, 2, 4, 3, device=device, seed=8)
    v = leaf(1, 2, 4, 3, device=device, seed=9)

    def fn(g, query, key, value):
        return scaled_dot_product_attention_backward(
            g, query, key, value, is_causal=True, scale=0.37)

    assert gradgradcheck(fn, (grad, q, k, v), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_math_backward_differentiates_twice_grouped(device):
    # Two kv heads serve four query heads; the key and value gradients sum
    # over the group axis in both passes.
    grad = leaf(2, 4, 5, 3, device=device, seed=10)
    q = leaf(2, 4, 5, 3, device=device, seed=11)
    k = leaf(2, 2, 5, 3, device=device, seed=12)
    v = leaf(2, 2, 5, 3, device=device, seed=13)

    def fn(g, query, key, value):
        return scaled_dot_product_attention_backward(
            g, query, key, value, enable_gqa=True)

    assert gradgradcheck(fn, (grad, q, k, v), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_math_backward_differentiates_twice_with_bool_mask(device):
    # A bool mask only selects positions; it takes no gradient of its own,
    # and a fully masked row contributes none anywhere.
    grad = leaf(1, 1, 4, 4, device=device, seed=14)
    q = leaf(1, 1, 4, 4, device=device, seed=15)
    k = leaf(1, 1, 4, 4, device=device, seed=16)
    v = leaf(1, 1, 4, 4, device=device, seed=17)
    tp.manual_seed(2024 + 18)
    mask = (tp.rand(4, 4) > 0.3).to(device)

    def fn(g, query, key, value):
        return scaled_dot_product_attention_backward(
            g, query, key, value, attn_mask=mask)

    assert gradgradcheck(fn, (grad, q, k, v), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_attention_differentiates_twice_end_to_end(device, is_causal):
    # The math backend keeps the call on the composed entry whose derivative
    # is the math backward.
    q = leaf(2, 2, 5, 4, device=device, seed=19)
    k = leaf(2, 2, 6, 4, device=device, seed=20)
    v = leaf(2, 2, 6, 4, device=device, seed=21)

    def fn(query, key, value):
        return F.scaled_dot_product_attention(query, key, value,
                                              is_causal=is_causal,
                                              backend="math")

    assert gradgradcheck(fn, (q, k, v), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_attention_differentiates_twice_end_to_end_grouped(device):
    q = leaf(1, 4, 4, 3, device=device, seed=22)
    k = leaf(1, 2, 4, 3, device=device, seed=23)
    v = leaf(1, 2, 4, 3, device=device, seed=24)

    def fn(query, key, value):
        return F.scaled_dot_product_attention(query, key, value,
                                              enable_gqa=True)

    assert gradgradcheck(fn, (q, k, v), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("is_causal", [False, True])
def test_fused_cpu_attention_differentiates_twice_end_to_end(is_causal):
    # A plain CPU call takes the fused entry; its backward differentiates
    # through the composed math, the saved output and constant taking none.
    q = leaf(2, 2, 5, 4, device="cpu", seed=38)
    k = leaf(2, 2, 6, 4, device="cpu", seed=39)
    v = leaf(2, 2, 6, 4, device="cpu", seed=40)
    mask = leaf(5, 6, device="cpu", seed=41)

    def fn(query, key, value, attn_mask):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=None if is_causal else attn_mask,
            is_causal=is_causal)

    assert gradgradcheck(fn, (q, k, v, mask), atol=1e-5, rtol=1e-4)


@CUDA
@pytest.mark.parametrize("is_causal", [False, True])
def test_lse_backward_differentiates_twice(is_causal):
    # float32 self-attention is what the log-sum-exp kernel serves: the
    # probabilities come back from the constant and the row statistic from
    # the saved output, and both take gradients of their own.
    device = "cuda"
    q = leaf_f32(2, 3, 5, 5, device=device, seed=25)
    k = leaf_f32(2, 3, 5, 5, device=device, seed=26)
    v = leaf_f32(2, 3, 5, 5, device=device, seed=27)
    grad = leaf_f32(2, 3, 5, 5, device=device, seed=28)
    with tp.no_grad():
        out, lse = _scaled_dot_product_attention_with_lse(q, k, v,
                                                          is_causal=is_causal)
    out = out.detach().requires_grad_(True)
    lse = lse.detach().requires_grad_(True)

    def fn(g, query, key, value, output, logsumexp):
        return _scaled_dot_product_attention_backward_with_lse(
            g, query, key, value, output, logsumexp, is_causal=is_causal)

    assert gradgradcheck(fn, (grad, q, k, v, out, lse),
                         eps=1e-4, atol=1e-3, rtol=1e-2)


@CUDA
def test_lse_backward_math_fallback_differentiates_twice():
    # A non-default implementation id misses the log-sum-exp kernel and runs
    # the composed math: the node answers with the math adjoints and the
    # saved output and constant take no gradient, exactly as the composed
    # backward read neither.
    device = "cuda"
    q = leaf_f32(2, 2, 4, 4, device=device, seed=29)
    k = leaf_f32(2, 2, 4, 4, device=device, seed=30)
    v = leaf_f32(2, 2, 4, 4, device=device, seed=31)
    grad = leaf_f32(2, 2, 4, 4, device=device, seed=32)
    with tp.no_grad():
        out, lse = _scaled_dot_product_attention_with_lse(q, k, v, impl=1)
    out = out.detach().requires_grad_(True)
    lse = lse.detach().requires_grad_(True)

    def fn(g, query, key, value, output, logsumexp):
        return _scaled_dot_product_attention_backward_with_lse(
            g, query, key, value, output, logsumexp, impl=1)

    assert gradgradcheck(fn, (grad, q, k, v, out, lse),
                         eps=1e-4, atol=1e-3, rtol=1e-2)


@CUDA
@pytest.mark.parametrize("is_causal", [False, True])
def test_lse_attention_differentiates_twice_end_to_end(is_causal):
    # The plain CUDA float32 call runs the log-sum-exp forward and backward;
    # differentiating the backward again walks the recorded node.
    device = "cuda"
    q = leaf_f32(2, 2, 4, 4, device=device, seed=38)
    k = leaf_f32(2, 2, 4, 4, device=device, seed=39)
    v = leaf_f32(2, 2, 4, 4, device=device, seed=40)

    def fn(query, key, value):
        return F.scaled_dot_product_attention(query, key, value,
                                              is_causal=is_causal)

    assert gradgradcheck(fn, (q, k, v), eps=1e-4, atol=1e-3, rtol=1e-2)


def test_dropout_backward_does_not_differentiate():
    grad = leaf(1, 1, 3, 2, device="cpu", seed=33)
    q = leaf(1, 1, 3, 2, device="cpu", seed=34)
    k = leaf(1, 1, 3, 2, device="cpu", seed=35)
    v = leaf(1, 1, 3, 2, device="cpu", seed=36)
    gq, gk, gv = scaled_dot_product_attention_backward(
        grad, q, k, v, dropout_p=0.5)
    w = leaf(1, 1, 3, 2, device="cpu", seed=37).detach()
    with pytest.raises(NotImplementedError):
        tp.autograd.grad((gq * w).sum(), grad)
