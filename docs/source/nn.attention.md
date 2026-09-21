# tensorplay.nn.attention

```{eval-rst}
.. automodule:: tensorplay.nn.attention
```

`tensorplay.nn.attention` collects the pieces around
{func}`tensorplay.nn.functional.scaled_dot_product_attention`: the backend-selection
context, the flash-attention implementation registry, helpers for the causal and
variable-length variants, and `omni_attention` — the programmable form of attention
that takes user-written score-modification and block-mask functions.

`scaled_dot_product_attention` routes each call to one of a small set of kernels — the
math fallback, flash attention, or memory-efficient attention. Which one is eligible
depends on the inputs (dtype, head dim, mask type, whether gradients are needed), and
{func}`tensorplay.nn.attention.sdpa_kernel` lets you restrict that choice from the
outside, either to force a specific kernel or to study how the candidates behave.

```python
import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.nn.attention import SDPBackend, sdpa_kernel

q = k = v = tp.randn(1, 2, 8, 16)

# restrict routing to the math implementation while experimenting
with sdpa_kernel(SDPBackend.MATH):
    out = F.scaled_dot_product_attention(q, k, v)

# a list means "any of these, in this preference order"
with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
    out = F.scaled_dot_product_attention(q, k, v)
```

The backend names are {class}`tensorplay.nn.attention.SDPBackend` members:
`MATH` (the reference implementation), `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, and
`CUDNN_ATTENTION`.

The registry functions manage which flash-attention implementation backs the
`FLASH_ATTENTION` route. {func}`tensorplay.nn.attention.list_flash_attention_impls`
reports the implementations this build knows about (`"FA3"` and `"FA4"`).
{func}`tensorplay.nn.attention.activate_flash_attention_impl` switches to one of them
and returns a handle; {func}`tensorplay.nn.attention.restore_flash_attention_impl`
returns to the previous state. Activating an implementation requires its kernel package
to be importable — the loader raises `ModuleNotFoundError` naming the missing module
when it is not. {func}`tensorplay.nn.attention.register_flash_attention_impl` registers
a custom implementation under a new name, which is how additional flash-attention
backends plug in. By default no implementation is active:
{func}`tensorplay.nn.attention.current_flash_attention_impl` reports `None`.

## Utils

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.attention.sdpa_kernel
    tensorplay.nn.attention.SDPBackend
    tensorplay.nn.attention.omni_attention
    tensorplay.nn.attention.BlockMask
    tensorplay.nn.attention.create_block_mask
    tensorplay.nn.attention.register_flash_attention_impl
    tensorplay.nn.attention.activate_flash_attention_impl
    tensorplay.nn.attention.list_flash_attention_impls
    tensorplay.nn.attention.current_flash_attention_impl
    tensorplay.nn.attention.restore_flash_attention_impl
    tensorplay.nn.attention.can_use_flash_attention
    tensorplay.nn.attention.can_use_efficient_attention
    tensorplay.nn.attention.create_mask
    tensorplay.nn.attention.and_masks
    tensorplay.nn.attention.or_masks
    tensorplay.nn.attention.noop_mask
```

{func}`tensorplay.nn.attention.can_use_flash_attention` and
{func}`tensorplay.nn.attention.can_use_efficient_attention` answer the routing question
for one concrete call: they take an `SDPAParams` record (query/key/value tensors, dropout
probability, and mask) and report whether the corresponding kernel would accept it. They
are the same probes the dispatcher consults.

## Mask helpers

Four functions turn mask *predicates* — plain functions of
`(b, h, q_idx, kv_idx)` returning a boolean tensor — into mask material you can pass
around. {func}`tensorplay.nn.attention.create_mask` compiles a predicate into a dense
boolean mask, and {func}`tensorplay.nn.attention.and_masks` /
{func}`tensorplay.nn.attention.or_masks` compose predicates before compiling;
{func}`tensorplay.nn.attention.noop_mask` is the allow-everything predicate:

```python
import tensorplay as tp
from tensorplay.nn.attention import and_masks, create_mask

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def window(b, h, q_idx, kv_idx):
    return q_idx - kv_idx < 4

mask = create_mask(and_masks(causal, window), B=1, H=1, Q_LEN=6, KV_LEN=6)
print(mask.shape, mask.dtype)   # (1, 1, 6, 6) bool
```

## Submodules

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.nn.attention.omni_attention
    tensorplay.nn.attention.bias
    tensorplay.nn.attention.varlen
```

```{eval-rst}
.. toctree::
    :hidden:

    nn.attention.omni_attention
    nn.attention.bias
    nn.attention.varlen
```
