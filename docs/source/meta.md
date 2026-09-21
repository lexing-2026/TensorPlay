# Meta device

The "meta" device is an abstract device whose tensors record only metadata — shape,
dtype, strides — and no data. Meta tensors answer "what would the result look like"
without spending compute or memory on values, which makes them the tool for abstract
analysis: tracing a model's shapes end to end, checking where dtypes change, or sizing
activations before allocating anything.

## What works on meta

All the factory functions accept `device="meta"`, and so does the
{func}`tensorplay.device` context manager, which redirects construction calls that do
not name a device:

```python
import tensorplay as tp

m = tp.zeros(3, 4, device="meta")
print(m.shape, m.numel(), m.is_meta)   # (3, 4) 12 True

with tp.device("meta"):
    t = tp.randn(30, 30)               # factories without a device= land on meta
print(t.device)                        # meta
```

Most shape-only operations run on meta tensors and produce new meta tensors carrying
the resulting metadata:

```python
with tp.device("meta"):
    x = tp.randn(8, 4)
    w = tp.randn(2, 4)

print((x + 1).shape)          # (8, 4)
print((x @ w.T).shape)        # (8, 2)
print(x.sum(dim=0).shape)     # (4,)
```

Coverage is not total: some operations whose results are shape-only in principle still
lack a meta kernel in this build — `softmax`, `stack`, `unsqueeze`, `chunk` among them —
and raise the same `NotImplementedError` as the data-dependent ones below. When a trace
hits such a gap, rewrite the step in terms of the primitives that do work (slicing and
`reshape` cover most reshaping needs).

`tensorplay.zeros_like` / `empty_like` on a meta tensor stay on meta, so a shape-tracing
pass can build its intermediates the way real code would.

## What does not

A meta tensor has no data, so anything that must read a value fails with
`NotImplementedError` ("Kernel not found for op: ... on backend: Meta"):

- data-dependent shapes: `nonzero`, `item`, `masked_select`-style operations;
- `.to("cpu")` on a meta tensor — copying out would require data to copy
  (the copy kernel reports it only supports CPU/CUDA-style sources). Use
  `empty_like(t, device="cpu")` and fill it yourself instead;
- `tp.load(..., map_location="meta")` — the deserializer only supports the `cpu` and
  CUDA map targets and refuses `meta` outright.

Module construction under a meta context hits the same wall in this build: layers
initialize their parameters by drawing random numbers (`uniform_` and friends), and
those kernels have no meta implementation. Build the module on a real device and reason
about shapes with meta *tensors* rather than moving whole modules to meta.

## Idioms

Shape-checking a sequence of operations before running them for real:

```python
def infer_shapes(seq_len, d_model, n_heads):
    with tp.device("meta"):
        x = tp.randn(seq_len, d_model)
        qkv = x @ tp.randn(d_model, 3 * d_model)
        head_dim = d_model // n_heads
        q = qkv[:, :d_model].reshape(seq_len, n_heads, head_dim)
        k = qkv[:, d_model:2 * d_model].reshape(seq_len, n_heads, head_dim)
        return q.transpose(0, 1).shape, k.transpose(0, 1).shape

print(infer_shapes(128, 512, 8))
# (tensorplay.Size([8, 128, 64]), tensorplay.Size([8, 128, 64]))
```

The [compiler](compiler.md) and the fake-tensor machinery behind shape specialization
use exactly this idea — metadata-only execution — to reason about programs without
running them.
