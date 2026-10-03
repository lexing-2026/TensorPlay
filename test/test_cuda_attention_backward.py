"""Gradients of scaled dot-product attention on CUDA, against float64.

The backward is computed as batched matrix products around one row softmax:
the scores and their softmax are formed again from q and k, and the three
gradients are products of the probabilities, the incoming gradient and the
saved operands.  Each case compares q, k and v gradients with the same
attention written out in float64, for contiguous and interleaved layouts and
with and without the causal mask.
"""
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="needs CUDA")

SHAPES = [(2, 3, 17, 8), (1, 2, 64, 64), (2, 4, 33, 40)]
TOLERANCE = {tp.float32: 2e-4, tp.float16: 2e-2, tp.bfloat16: 5e-2}


def _attention_by_hand(q, k, v, causal):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if causal:
        length = q.shape[-2]
        allowed = tp.ones(length, length, device=q.device).tril().bool()
        scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.softmax(-1) @ v


def _operands(shape, dtype, layout):
    batch, heads, length, width = shape
    if layout == "contiguous":
        return [tp.randn(*shape, device="cuda").to(dtype) for _ in range(3)]
    # q, k and v interleaved in one projection, heads read across it.
    packed = tp.randn(batch, length, 3, heads, width, device="cuda").to(dtype)
    return [packed[:, :, i].transpose(1, 2) for i in range(3)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [tp.float32, tp.float16, tp.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "interleaved"])
def test_gradients_match_float64(shape, causal, dtype, layout):
    tp.manual_seed(0)
    q, k, v = (t.detach().requires_grad_(True) for t in _operands(shape, dtype, layout))
    grad = tp.randn(*shape, device="cuda").to(dtype)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    got = tp.autograd.grad(out, (q, k, v), grad)

    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    ref = _attention_by_hand(q64, k64, v64, causal)
    want = tp.autograd.grad(ref, (q64, k64, v64), grad.double())

    for name, a, b in zip("qkv", got, want):
        assert a.dtype == dtype and a.shape == b.shape
        err = ((a.double() - b).abs().max() / (b.abs().max() + 1e-6)).item()
        assert err < TOLERANCE[dtype], (name, err)


def test_fully_masked_free_rows_and_empty_batches():
    # A single query attends to one key under the causal mask: its
    # probabilities are exactly one, so dq is zero and dv is the gradient.
    q, k, v = (tp.randn(1, 1, 1, 4, device="cuda", requires_grad=True) for _ in range(3))
    grad = tp.randn(1, 1, 1, 4, device="cuda")
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    dq, dk, dv = tp.autograd.grad(out, (q, k, v), grad)
    assert dq.abs().max().item() == 0.0 and dk.abs().max().item() == 0.0
    assert tp.allclose(dv, grad)

    empty = [tp.randn(0, 2, 5, 4, device="cuda", requires_grad=True) for _ in range(3)]
    out = F.scaled_dot_product_attention(*empty)
    grads = tp.autograd.grad(out, empty, tp.ones_like(out))
    assert all(g.shape == (0, 2, 5, 4) for g in grads)


@pytest.mark.parametrize("shape", [(2, 4, 33, 64), (1, 2, 64, 128), (2, 2, 40, 96)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [tp.float32, tp.float16, tp.bfloat16])
def test_the_pair_a_compiled_region_calls(shape, causal, dtype):
    # The forward hands back each row's log-sum-exp where its schedule writes
    # one, and the backward reads the probabilities off it and the row
    # statistic off the output.  The heads are read out of one interleaved
    # projection, as a compiled region passes them.
    if dtype == tp.float32 and shape[-1] == 96:
        pytest.skip("no float32 schedule for this head width")
    tp.manual_seed(0)
    q, k, v = _operands(shape, dtype, "interleaved")
    ops = tp.ops.tp
    out, lse = ops._scaled_dot_product_attention_with_lse(q, k, v, causal, 0)
    assert lse.shape == shape[:3] and lse.dtype == tp.float32
    grad = tp.randn(*shape, device="cuda").to(dtype)
    got = ops._scaled_dot_product_attention_backward_with_lse(grad, q, k, v, out, lse, causal, 0)

    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    ref = _attention_by_hand(q64, k64, v64, causal)
    want = tp.autograd.grad(ref, (q64, k64, v64), grad.double())
    assert ((out.double() - ref).abs().max() / ref.abs().max()).item() < TOLERANCE[dtype]
    for name, a, b in zip("qkv", got, want):
        assert a.dtype == dtype and a.shape == b.shape
        err = ((a.double() - b).abs().max() / (b.abs().max() + 1e-6)).item()
        assert err < TOLERANCE[dtype], (name, err)


def test_autocast_inputs_of_mixed_types_take_the_fused_pair():
    # Under autocast a query and key left in single precision (a rotary
    # embedding) and a half-precision value all reach the kernel in half
    # precision, so the call takes the fused entry and its backward.
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_python_dispatch import RecordingMode

    tp.manual_seed(0)
    q = tp.randn(2, 4, 64, 64, device="cuda", requires_grad=True)
    k = tp.randn(2, 4, 64, 64, device="cuda", requires_grad=True)
    v = tp.randn(2, 4, 64, 64, device="cuda").half().requires_grad_(True)
    with tp.autocast("cuda", dtype=tp.float16), RecordingMode() as mode:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert any(n.startswith("_scaled_dot_product_attention_with_lse") for n in mode.names()), mode.names()
    grad = tp.randn(2, 4, 64, 64, device="cuda").half()
    got = tp.autograd.grad(out, (q, k, v), grad)

    q64, k64, v64 = (t.detach().half().double().requires_grad_(True) for t in (q, k, v))
    ref = _attention_by_hand(q64, k64, v64, True)
    want = tp.autograd.grad(ref, (q64, k64, v64), grad.double())
    for name, a, b in zip("qkv", got, want):
        err = ((a.double() - b).abs().max() / (b.abs().max() + 1e-6)).item()
        assert err < 2e-2, (name, err)
