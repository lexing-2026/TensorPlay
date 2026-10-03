"""Batch-first multi-head attention keeps a tensor passed twice as one tensor.

The input projection recognizes self-attention by identity and then computes
query, key and value in one product.  Converting batch-first inputs to the
sequence-first layout one argument at a time would hand it three distinct
tensors and three products.
"""

import pytest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F


def _pair(embed, heads):
    first = nn.MultiheadAttention(embed, heads, batch_first=True)
    second = nn.MultiheadAttention(embed, heads, batch_first=False)
    second.load_state_dict(first.state_dict())
    return first, second


@pytest.mark.parametrize("case", ["self", "shared_kv", "distinct"])
def test_batch_first_matches_sequence_first(case, monkeypatch):
    tp.manual_seed(0)
    first, second = _pair(16, 4)
    x = tp.randn(3, 5, 16)
    y = tp.randn(3, 7, 16)
    z = tp.randn(3, 7, 16)
    q, k, v = {"self": (x, x, x), "shared_kv": (x, y, y), "distinct": (x, y, z)}[case]

    products = []
    linear = F.linear

    def counting(*args, **kwargs):
        products.append(args[1].shape)
        return linear(*args, **kwargs)

    monkeypatch.setattr(F, "linear", counting)
    out, weights = first(q, k, v)
    monkeypatch.setattr(F, "linear", linear)

    # Input projections plus the output projection.
    expected = {"self": 2, "shared_kv": 3, "distinct": 4}[case]
    assert len(products) == expected, products
    ref, ref_weights = second(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
    assert tp.allclose(out, ref.transpose(0, 1), atol=1e-5)
    assert tp.allclose(weights, ref_weights, atol=1e-5)


def test_constructor_takes_layout_and_factory_options():
    m = nn.MultiheadAttention(8, 2, batch_first=True, dtype=tp.float64)
    assert m.batch_first and m.in_proj_weight.dtype == tp.float64
    assert m.out_proj.weight.dtype == tp.float64
    assert not nn.MultiheadAttention(8, 2).batch_first
    x = tp.randn(2, 3, 8, dtype=tp.float64)
    assert m(x, x, x)[0].shape == (2, 3, 8)
