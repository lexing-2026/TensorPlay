# tensorplay.nn.attention.varlen

```{eval-rst}
.. automodule:: tensorplay.nn.attention.varlen
```

Batched attention APIs take a rectangular `(B, H, S, D)` block, so sequences of
different lengths must be padded to a common length — and the padding positions then
cost compute and pollute the softmax unless masked. The variable-length path avoids the
rectangle entirely: sequences are *packed* along the sequence dimension, and one
cumulative-lengths tensor describes where each sequence starts and ends.

## Packed layout

Inputs are shaped `(total_tokens, H, D)`, where `total_tokens` is the sum of the
sequence lengths in the batch. `cu_seq_q` and `cu_seq_k` are int32 tensors of cumulative
query and key/value lengths, starting at 0:

```python
import tensorplay as tp

cu_seq_q = tp.tensor([0, 5, 16], dtype=tp.int32)
# two sequences: query lengths 5 and 11, packed into 16 total rows
```

`max_q` and `max_k` are the largest single sequence lengths — the kernels use them to
size their tiles, so over-stating them wastes memory while under-stating them is an
error.

{func}`tensorplay.nn.attention.varlen.varlen_attn` computes attention over the packed
batch; `return_aux` requests the auxiliary outputs (the softmax log-sum-exp and seed
state that the backward pass reuses). {func}`tensorplay.nn.attention.varlen.varlen_attn_out`
is the `out=` form that writes into a caller-provided output tensor.

Both run through the flash-attention kernels and require a CUDA device; calling them on
CPU raises `NotImplementedError` naming the missing kernel backend.

The keyword arguments cover the flash-attention feature set: `scale` overrides the
`1/sqrt(head_dim)` default, `window_size` restricts each query to a sliding window of
keys (`(-1, -1)` disables the window), `enable_gqa` allows fewer key/value heads than
query heads, `seqused_k` bounds the usable key length per batch entry, and `block_table`
points at paged key/value blocks so the cache does not have to be contiguous.

## API

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.attention.varlen.varlen_attn
    tensorplay.nn.attention.varlen.varlen_attn_out
```
