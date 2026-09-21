# tensorplay.nn.attention.omni_attention

```{eval-rst}
.. automodule:: tensorplay.nn.attention.omni_attention
```

`omni_attention` is TensorPlay's programmable attention: one entry point that takes
ordinary Python functions describing *how the attention scores should be modified* and
*which query/key blocks can be skipped*, and runs the result. Instead of choosing among
fixed kernel variants and their fixed feature lists, you write the modification you
want and the implementation handles the rest.

{func}`tensorplay.nn.attention.omni_attention` keeps the scaled-dot-product calling
shape — query, key, value — and adds two programming hooks on top:

```python
import tensorplay as tp
from tensorplay.nn.attention import omni_attention

q = k = v = tp.randn(1, 2, 16, 8)

# a distance-decay bias applied to every score: score - (q_idx - kv_idx)
def alibi(score, b, h, q_idx, kv_idx):
    return score - (q_idx - kv_idx)

out = omni_attention(q, k, v, score_mod=alibi)
```

A `score_mod` receives the current score and its coordinates — batch `b`, head `h`,
query position `q_idx`, key position `kv_idx` — and returns the modified score. Any
expression over those five values is allowed; positional biases, distance penalties,
per-head scales, and soft caps are all one-line `score_mod`s. The result matches the
hand-written masked-softmax computation of the same modification.

## Block masks

A {class}`tensorplay.nn.attention.BlockMask` is the compiled form of a block-sparsity
pattern. You describe the pattern as a `mask_mod` predicate — a function of
`(b, h, q_idx, kv_idx)` returning a boolean — and
{func}`tensorplay.nn.attention.create_block_mask` compiles it:

```python
from tensorplay.nn.attention import and_masks, create_block_mask, omni_attention

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def window(b, h, q_idx, kv_idx):
    return q_idx - kv_idx < 4      # each query sees at most 4 past keys

mask = create_block_mask(
    and_masks(causal, window), B=1, H=1, Q_LEN=16, KV_LEN=16, device="cpu"
)
out = omni_attention(q, k, v, block_mask=mask)
```

`and_masks` / `or_masks` compose predicates before compilation and
`noop_mask` is the allow-everything predicate;
{func}`tensorplay.nn.attention.create_mask` materializes a predicate into a dense
boolean mask when you want to look at it or feed a kernel that takes dense masks.

`create_block_mask` builds the mask on a CUDA device by default; pass `device="cpu"`
when the tensors are on CPU — the call checks that query, key, value, and the block
mask agree and raises otherwise.

## Fusion

Called directly, `omni_attention` runs an **unfused** implementation that materializes
the full scores matrix — correct, but with the memory and speed of a manual softmax,
and it says so: the runtime warns that it "will use an unfused implementation that
materializes the full scores matrix instead of generating a fused kernel" and points
at the fix, wrapping the call in `tensorplay.compile` (`tp.compile(omni_attention)`).
Compilation is where the `score_mod` and `mask_mod` functions are traced into the
kernel's score loop and the block-sparse schedule is lowered.

When debugging a `score_mod` you can drop a breakpoint in it by disabling that
internal compilation
(`tensorplay.nn.attention.omni_attention._OMNI_ATTENTION_DISABLE_COMPILE_DEBUG = True`)
— a debugging affordance only: the unfused path does not backpropagate through the
modification.

## Outputs

By default the function returns the output tensor. `return_lse=True` additionally
returns the log-sum-exp of the attention weights (what decoders with incremental
caching recombine), and the `return_aux` keyword requests the auxiliary outputs the
backward pass reuses, as an `AuxOutput`.

## API

```{eval-rst}
.. autosummary::
    :nosignatures:

    omni_attention
    tensorplay.nn.attention.BlockMask
    tensorplay.nn.attention.create_block_mask
    tensorplay.nn.attention.create_mask
    tensorplay.nn.attention.and_masks
    tensorplay.nn.attention.or_masks
    tensorplay.nn.attention.noop_mask
    tensorplay.nn.attention.AuxOutput
    tensorplay.nn.attention.AuxRequest
```
