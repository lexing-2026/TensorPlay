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
