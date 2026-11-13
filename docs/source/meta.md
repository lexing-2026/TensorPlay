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

Coverage is broad but not total. Arithmetic and comparisons (including
`isclose`), the pointwise unary family (functional and in-place, including the
modulo, division and two-operand transcendental pairs plus the
`isnan`/`isinf`/`signbit`/`isreal` predicates), reductions (`sum`, `mean`,
`prod`, `amax`/`amin`, `argmax`/`argmin`, `logsumexp`,
`all`/`any`, `mode`/`kthvalue`, `median`/`nanmedian`, `cummax`/`cummin`), matrix
products (`matmul` and friends, `dot`/`mv`/`inner`/`ger`, `cross`,
`baddbmm`/`addbmm`, `conv1d`/`conv2d`/`conv3d`/`conv_transpose2d`), views
(`reshape`, `transpose`, `slice`, `squeeze`, `unsqueeze`, `chunk`, `unbind`,
`split_with_sizes`, `cat`, `diff`), indexing (`gather`, `scatter`, `index_select`,
`take`, `searchsorted`, `bucketize`, `masked_fill`, `masked_scatter`,
`_index_put_impl_`), shape builders (`tril`/`triu`, `diag_embed`/`diagflat`,
`rot90`, `cartesian_prod`, `kron`, `one_hot`, `trace`,
`upsample_bilinear2d`, `pixel_shuffle`), the activations and normalization
layers (`softmax`, `threshold`, `layer_norm`, `group_norm`, `instance_norm`,
`batch_norm`, `rms_norm`), the
norm and distance family (`dist`, `renorm`), and the losses (`nll_loss`,
`cross_entropy`, `mse_loss`, Huber, soft-margin, hinge-embedding,
margin-ranking, cosine-embedding and the multi-class/multi-label margin
losses) all run on meta tensors. The margin losses still enforce their target
layout (a 1-D index vector for the multi-class form, a rank-2 label table for
the multi-label one); only the per-entry index range is a value check that
cannot run.

Linear algebra reports the full result geometry: `cholesky`/`cholesky_ex`,
`inv`/`inv_ex`, `det`/`slogdet`, the LU family (`lu`, `lu_factor_ex`,
`lu_solve`, `lu_unpack`), `ldl_factor`/`ldl_factor_ex`/`ldl_solve`,
`solve`/`solve_ex`/`solve_triangular`, `eigh`/`eigvalsh`/`eig`/`eigvals`,
`svd`/`svdvals`, `qr`, `householder_product`, `lstsq`, `pinv` and `diagonal`,
including the integer `info`, pivot and rank planes and the dtype changes
(a real input's `eig` answers complex tensors). Argument contracts are
enforced as on a materialized tensor: square and matrix requirements, the
floating-point dtype gate, and the `driver`/`mode` names. Composed helpers
(`matrix_power`, `matrix_rank`, `cond`, `norm`, `polar`) build on those.

The random fills (`uniform_`, `normal_`, `cauchy_`, `exponential_`,
`geometric_`, `log_normal_`, `bernoulli_` and the `random_` family) accept a
meta tensor and leave its metadata unchanged — there is no data to sample into,
so only the argument checks run. That is enough for default initialization:
building a module inside a meta context produces meta parameters, and a forward
pass on meta inputs answers shapes without computing.

```python
with tp.device("meta"):
    model = tp.nn.Linear(4, 3)
print(model.weight.shape)              # (3, 4)
print(model(tp.randn(2, 4)).shape)     # (2, 3)
```

What remains missing is genuinely value-dependent — `nonzero`, `masked_select`,
`unique`, `bincount`, `count_nonzero`, `item` — and raises the same
`NotImplementedError` as the cases below. When a trace hits such a gap, rewrite the
step in terms of the primitives that do work (slicing and `reshape` cover most
reshaping needs).

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

Value-dependent checks inside the random fills are skipped on meta: bounds and
probability scalars are validated, but a probability or bound *tensor* is never
read, so a draw that would fail on stored values passes here.

Factorization failures that depend on the stored numbers cannot fire either.
`cholesky` returns the factor shape instead of rejecting a matrix that is not
positive definite, and a singular `inv` answers its shape rather than raising;
the `_ex` forms answer an `info` plane without evaluating any entry, so a later
consumer that inspects `info` has nothing meaningful to read. `matrix_sqrth`
and `matrix_exp` fail outright because their scaling decisions read stored
values (an `item` call and a copy out of the tensor respectively).

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
