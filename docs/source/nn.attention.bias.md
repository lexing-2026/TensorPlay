# tensorplay.nn.attention.bias

```{eval-rst}
.. automodule:: tensorplay.nn.attention.bias
```

A causal mask is the one attention bias common enough to deserve its own type.
{class}`tensorplay.nn.attention.bias.CausalBias` is that type: a non-materialized
boolean mask that `scaled_dot_product_attention` (and its fused kernels, when eligible)
interpret directly, without building the `(L, S)` mask tensor in memory.

## Variants

{class}`tensorplay.nn.attention.bias.CausalVariant` selects the alignment:

- `UPPER_LEFT` — standard causal attention. Position `i` may attend to positions
  `j <= i`. The equivalent materialized mask is `tensorplay.tril(ones(L, S, dtype=bool))`.
- `LOWER_RIGHT` — the allowed region is anchored to the lower-right corner instead, so
  the *last* query positions see the full history. With equal query and key lengths the
  two variants coincide; they differ when packing sequences of different lengths.

## Constructing a bias

{func}`tensorplay.nn.attention.bias.causal_upper_left` and
{func}`tensorplay.nn.attention.bias.causal_lower_right` take the query and key sequence
lengths and return the matching bias:

```python
import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.nn.attention.bias import (
    causal_upper_left,
    CausalBias,
    CausalVariant,
)

q = tp.randn(1, 1, 8, 16)
k = tp.randn(1, 1, 8, 16)
v = tp.randn(1, 1, 8, 16)

bias = causal_upper_left(8, 8)
out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
print(out.shape)    # (1, 1, 8, 16)
```

The same object comes from the constructor form
`CausalBias(CausalVariant.UPPER_LEFT, seq_len_q, seq_len_kv)`; `CausalVariant.LOWER_RIGHT`
gives the lower-right alignment. Because the bias is not a materialized tensor, it costs
no `L × S` memory and is the mask form the flash-attention kernels prefer.

## Flash-attention probes

The module re-exports the eligibility probes the dispatcher uses, so mask-related
routing questions can be answered in one place:

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    CausalBias
    CausalVariant
    causal_upper_left
    causal_lower_right
```

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.backends.cuda.is_flash_attention_available
    tensorplay.backends.cuda.can_use_flash_attention
    tensorplay.backends.cuda.can_use_efficient_attention
```
