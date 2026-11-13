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
`isnan`/`isinf`/`signbit`/`isreal` predicates, the arc-prefixed and word
spellings of those names, and the `logaddexp`/`logaddexp2` pair), reductions
(`sum`, `mean`, `prod`, `amax`/`amin`, `argmax`/`argmin`, `logsumexp`,
`nansum`/`nanmean`, `logcumsumexp`,
`all`/`any`, `mode`/`kthvalue`, `median`/`nanmedian`, `cummax`/`cummin`,
`trapezoid`/`cumulative_trapezoid`, `cov`/`corrcoef`, `count_nonzero`, and the
bin-count `histogram` contract), matrix
products (`matmul` and friends, `dot`/`mv`/`inner`/`ger`, `cross`,
`baddbmm`/`addbmm`, `conv1d`/`conv2d`/`conv3d`/`conv_transpose1d`/`conv_transpose2d`/`conv_transpose3d`), views
(`reshape`, `transpose`, `slice`, `squeeze`, `unsqueeze`, `chunk`, `unbind`,
`split_with_sizes`, `cat`, `diff`), the complex views (`real`, `imag`, `conj`,
`adjoint`, `view_as_real`, `view_as_complex` — the last two re-stride the
source's storage, so writes through the view and the source stay in sync),
indexing (`gather`, `scatter`, `index_select`,
`take`, `searchsorted`, `bucketize`, `masked_fill`, `masked_scatter`,
`_index_put_impl_`), shape builders (`tril`/`triu`, `diag_embed`/`diagflat`,
`rot90`, `cartesian_prod`, `kron`, `one_hot`, `trace`, `block_diag`,
`broadcast_tensors`, `pixel_shuffle`/`pixel_unshuffle`, `channel_shuffle`,
the scatter forms (`select_scatter`/`slice_scatter`/`diagonal_scatter`) and
`vander`), resampling (`upsample_nearest1d/2d/3d`, `upsample_linear1d`,
`upsample_bilinear2d`, `upsample_bicubic2d`, `upsample_trilinear3d`, the
nearest-exact variants, the antialiasing pair with their scale-factor forms,
and the matching backwards), the activations and normalization
layers (`softmax`, `threshold`, `layer_norm`, `group_norm`, `instance_norm`,
`batch_norm`, `rms_norm`) together with the activation
backwards (`gelu_backward`, `silu_backward`, `tanh_backward`, `glu_backward`,
the shrink pair, `rrelu_with_noise_backward` and the rest of the family), the
norm and distance family (`dist`, `renorm`, `pdist`), and the losses
(`nll_loss`, `cross_entropy`, `mse_loss`, L1, KL, Poisson NLL, Huber,
binary cross entropy with and without logits, soft-margin,
multi-label soft-margin, hinge-embedding, margin-ranking, cosine-embedding,
triplet-margin and the multi-class/multi-label margin
losses) all run on meta tensors. The resampling entries police the geometry
they are handed: a recorded sweep that would reach outside the gradient or
the answer it names, and recorded lists shorter than the spatial rank they
describe, are refused as errors instead of left undefined. The activation
backwards police the saved plane their sweep reads: a plane of a different
width than the gradient, or one smaller than the gradient's element count,
is refused as an error, and `log_sigmoid_backward` sweeps in floating widths
only. Concatenation keeps a zero-valued plane's geometry: its length along
the joined dimension counts, and only size-[0] 1-d inputs are exempt from
the wrap and shape checks. The plain loss names average to a scalar at
the double type when their leading operand has one, single precision
otherwise. The margin losses still enforce their target
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

The complex factories answer their broadcast geometry and promote to the
matching complex type: `complex` pairs a real and an imaginary plane, `polar`
combines magnitudes and angles; both require floating inputs of one dtype on
one device. The non-constant padding family (`reflection_pad1d/2d/3d/nd`,
`replication_pad1d/2d/3d/nd`, `circular_pad_nd` and their `_backward` forms)
adds the pad amounts to the padded dimensions and enforces the same
constraints as a materialized backend: even-length pad lists, pad amounts
that fit the dimensionality, mode-specific bounds (a reflection may not reach
past one traversal of the dimension, a circular wrap may not go around more
than once), and the floating dtype gate — the forward accepts a few narrow
integers, the backward is float-only.

The assorted entries behave as their shapes dictate: `native_alpha_dropout`
samples its keep mask in single precision and rescales at the promoted type,
`histc` answers the bin count (when the range endpoints coincide the span is
normally derived from the stored extrema, which a metadata-only device cannot
read, so that validation is left to a materializing backend), and `frexp`
returns the mantissa in the input dtype beside a 32-bit exponent plane.

The special-function families answer their shape arithmetic directly: the
Bessel and Airy tables (`bessel_j0`/`j1`, `bessel_y0`/`y1`, `airy_ai`,
`spherical_bessel_j0`, the `modified_bessel_i/k` pair and the scaled K pair),
the error-function tail and normal-distribution maps (`erfcx`, `ndtr`,
`ndtri`, `log_ndtr`), the Bessel error companions (`i0e`, `i1`, `i1e`), the
orthogonal polynomials (the four Chebyshev degrees, their shifted variants,
Hermite, Laguerre and Legendre), the zeta and incomplete-gamma pairs (`zeta`,
`igamma`, `igammac`, `gammainc`, `gammaincc`) and the `xlog1py` product form,
together with the `special_` spellings of every one of these names. Every
sweep in these families computes at floating precision and reads no values, so
a floating operand keeps its width and every other operand — integers,
booleans, complex planes — is answered in single precision. The members with
no floating-width pass over those planes (`erf`, `erfc`, `lgamma`, the
rounding family, the softmax pair) refuse them outright, and `polygamma`
refuses orders the recurrence cannot be run for before any geometry is
answered. The `out=` forms of `digamma`, `i0`, `polygamma`, `igamma` and
`igammac` write through the same rules: a destination naming another geometry
is resized by the write, and an integral destination that cannot be cast to
safely is refused. `resize_` itself rewrites the sizes and grows the recorded
storage length in place — a call that keeps the current sizes is a no-op for
the strides, and negative or overflowing sizes are refused.

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
