"""Attention with a mask written as a function, compiled.

The mask says which key positions a query position may read.  A causal one,
``query >= key``, makes the call the same computation as causal scaled
dot-product attention, which is what each case is checked against -- and
against the unmasked product too, so that a mask quietly dropped on the way
shows up as the wrong answer rather than as agreement.
"""
import importlib

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

omni = importlib.import_module("tensorplay.nn.attention.omni_attention")

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def causal(batch, head, query, key):
    return query >= key


def test_lowering_a_region_loads_the_attention_lowerings():
    from tensorplay.compiler.backends.stax.op_lowerings import LOWERINGS

    compiled = tp.compile(lambda v: v * 2.0, backend="stax")
    assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 4.0]
    assert "omni_attention" in LOWERINGS
    assert "omni_attention_backward" in LOWERINGS


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length, width", [(128, 16), (256, 16), (128, 4)])
def test_a_causal_mask_gives_causal_attention(device, length, width):
    tp.manual_seed(0)
    mask = omni.create_block_mask(causal, 1, 1, length, length, device=device)
    q = tp.randn((1, 2, length, width), device=device)
    k = tp.randn((1, 2, length, width), device=device)
    v = tp.randn((1, 2, length, width), device=device)

    out = omni.omni_attention(q, k, v, block_mask=mask)

    masked = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    unmasked = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == masked.shape
    assert float((out - masked).abs().max()) < 1e-4
    # The last query position reads every key either way; the first reads one
    # key under the mask and all of them without it.
    assert float((out[..., 0, :] - v[..., 0, :]).abs().max()) < 1e-4
    assert float((out - unmasked).abs().max()) > 1e-2


def window(batch, head, query, key):
    return (query - key).abs() <= 40


def _processor_attention(mask, *, score_mod=None, enable_gqa=False):
    return tp.compile(
        lambda q, k, v: omni.omni_attention(
            q, k, v, score_mod=score_mod, block_mask=mask, enable_gqa=enable_gqa
        ),
        backend="stax",
        fullgraph=True,
    )


def _lowered(compiled):
    lowering = next(iter(compiled._tensorplay_cache.values()))
    return getattr(lowering, "_tensorplay_codegen", None) is not None


def _attention_by_hand(q, k, v, keep, bias=None):
    if k.size(1) != q.size(1):
        repeat = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeat, 1)
        v = v.repeat_interleave(repeat, 1)
    scores = (q.float() @ k.float().transpose(-2, -1)) / q.size(-1) ** 0.5
    if bias is not None:
        scores = scores + bias
    scores = scores.masked_fill(~keep, float("-inf"))
    return scores.softmax(-1) @ v.float()


def _keep(mask_fn, length_q, length_k):
    rows = tp.arange(length_q).view(length_q, 1)
    cols = tp.arange(length_k).view(1, length_k)
    return mask_fn(None, None, rows, cols)


# On a processor a compiled call is one generated kernel: the mask is read as
# blocks, the scores of a block are formed and softened in registers, and the
# whole of it is checked against the product written out by hand.  The lengths
# are chosen so that the last block is a partial one.
@pytest.mark.parametrize(
    "mask_fn, length_q, length_k, width",
    [(causal, 200, 200, 32), (window, 300, 300, 64), (causal, 128, 384, 64)],
)
def test_a_processor_call_is_one_generated_kernel(mask_fn, length_q, length_k, width):
    tp.manual_seed(0)
    mask = omni.create_block_mask(mask_fn, 1, 1, length_q, length_k, device="cpu")
    q = tp.randn((2, 4, length_q, width))
    k = tp.randn((2, 4, length_k, width))
    v = tp.randn((2, 4, length_k, width))

    compiled = _processor_attention(mask)
    out = compiled(q, k, v)

    assert _lowered(compiled)
    expected = _attention_by_hand(q, k, v, _keep(mask_fn, length_q, length_k))
    assert float((out - expected).abs().max()) < 1e-5


def test_a_score_written_as_a_function_is_applied_in_the_kernel():
    tp.manual_seed(0)
    length = 256
    mask = omni.create_block_mask(causal, 1, 1, length, length, device="cpu")
    q, k, v = (tp.randn((1, 2, length, 64)) for _ in range(3))

    def relative(score, batch, head, query, key):
        return score + (query - key) * 0.05

    compiled = _processor_attention(mask, score_mod=relative)
    out = compiled(q, k, v)

    assert _lowered(compiled)
    rows = tp.arange(length).view(length, 1)
    cols = tp.arange(length).view(1, length)
    bias = (rows - cols).float() * 0.05
    expected = _attention_by_hand(q, k, v, _keep(causal, length, length), bias)
    assert float((out - expected).abs().max()) < 1e-5


def test_grouped_heads_read_their_shared_keys_in_the_kernel():
    tp.manual_seed(0)
    length = 256
    mask = omni.create_block_mask(causal, 1, 1, length, length, device="cpu")
    q = tp.randn((1, 4, length, 64))
    k = tp.randn((1, 2, length, 64))
    v = tp.randn((1, 2, length, 64))

    compiled = _processor_attention(mask, enable_gqa=True)
    out = compiled(q, k, v)

    assert _lowered(compiled)
    expected = _attention_by_hand(q, k, v, _keep(causal, length, length))
    assert float((out - expected).abs().max()) < 1e-5


@pytest.mark.parametrize("dtype, tolerance", [(tp.bfloat16, 3e-2), (tp.float16, 5e-3)])
def test_narrow_types_are_computed_in_float_in_the_kernel(dtype, tolerance):
    tp.manual_seed(0)
    length = 256
    mask = omni.create_block_mask(causal, 1, 1, length, length, device="cpu")
    q, k, v = (tp.randn((2, 4, length, 64)).to(dtype) for _ in range(3))

    compiled = _processor_attention(mask)
    out = compiled(q, k, v)

    assert _lowered(compiled)
    assert out.dtype == dtype
    expected = _attention_by_hand(q, k, v, _keep(causal, length, length))
    assert float((out.float() - expected).abs().max()) < tolerance


@pytest.mark.skipif(not tp.cuda.is_available(), reason="needs CUDA")
def test_compiled_training_reaches_every_input():
    # The attention is recorded whole in a traced region, and its backward
    # reads the forward's result through a conversion to the type it already
    # has; the region still returns the forward's own result and every input
    # gets its gradient.
    tp.manual_seed(0)
    length = 128
    mask = omni.create_block_mask(causal, 1, 1, length, length, device="cuda")
    q, k, v = (tp.randn((1, 2, length, 32), device="cuda", requires_grad=True) for _ in range(3))

    def loss(q, k, v):
        return omni.omni_attention(q, k, v, block_mask=mask).sum()

    got = tp.compile(loss, backend="stax")(q, k, v)
    got_grads = tp.autograd.grad(got, (q, k, v))
    want = loss(q, k, v)
    want_grads = tp.autograd.grad(want, (q, k, v))
    assert abs(float(got - want)) < 1e-3
    for g, w in zip(got_grads, want_grads):
        assert float((g - w).abs().max()) < 1e-4
