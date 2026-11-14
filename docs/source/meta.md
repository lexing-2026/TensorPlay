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
otherwise. The squared difference and the switched absolute blend run in
the pair's promoted type: a boolean pair has no subtraction to feed them,
and a whole-number pair answers at single precision (for the mean
reduction of the squared difference, and at every reduction for the
switched blend). The scored loss runs over the four wide element types
only, and the logarithmic loss blends a logarithm of the input, the target
shifted by a fractional scalar and an optional weight multiply, so
whole-number planes answer at single precision and the weight joins the
promotion. The margin losses still enforce their target
layout (a 1-D index vector for the multi-class form, a rank-2 label table for
the multi-label one); only the per-entry index range is a value check that
cannot run. The gathered reads and writes share one dimension wrap: a
negative axis counts from the end, and an axis past the end is refused
after the wrap, which reports the axis the caller passed. The flat read
walks real and complex planes alike. The normalization family's
lower-level forward refuses the half-to-float conversion request before
anything else is read, and its backwards answer at the incoming gradient's
own type — narrowed back to half width when a half plane received a
single-precision gradient.

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
The factorization unpacks its permutation into a matrix written in the
operand's own type, so complex operands are refused as type errors; a
negative power inverts its operand first, and that inversion runs over the
linalg element types only, so every other operand is refused before any
geometry is answered. The Cholesky solve answers in the right-hand side's
own geometry and type; the factor is only read for its element type, so the
floating gate runs over both operands before anything else.

The attention entries compose their answer from recordable primitives: a
transposed score product, an optional additive mask, and a weighted value
product. The four wide element types are accepted and every other operand
is refused; the head extent read from the query must be non-zero and match
the key's; the key and value must describe the same weights, with a
whole-group head count when grouped queries are asked for and matching
leading extents when the key carries the query's rank. Masks of any
broadcastable rank join the scores lifted to the accumulation type, so a
wider mask is narrowed and the answer keeps the query's type;
reduced-width operands accumulate in single precision and return to their
own width. The variant that also reports the softmax normalizing constant
skips the geometry checks the plain form makes — the products enforce
their own — and always reports that plane in single precision.

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

The window factories (`bartlett_window`, `blackman_window`, `hann_window`,
`hamming_window`) answer the requested length in the default floating width
and refuse a negative count; the `bernoulli` fill keeps its operand's
metadata. `linear` folds the weight transpose and the optional bias into a
matrix product: the input's leading extents join the weight's rows, a rank-1
input keeps its vector geometry, and a weight whose columns miss the input's
last extent is refused. The loss backwards for the squared distance and the
switched absolute blend answer the input's own geometry after the shape
agreement check, in the gradient's dtype — a half-width gradient keeps its
width — for every reduction mode. The slice backward spreads the gradient
into the full shape along the wrapped dimension with the slice's own step.

The convolution dispatcher answers every rank the spatial dimensions
describe: the plain and transposed forms report the stride-extended geometry
from the input, weight and optional bias planes, and the backward reports
the input, weight and bias gradients the requested-slot mask asks for —
masked slots stay undefined, the bias gradient spans the gradient's channel
extent (which the transposed form maps to its output channels), and the
transposed form skips the forward geometry cross-check because the weight
describes the inverse mapping. The pooling backwards answer the input's own
geometry: `avg_pool3d` re-derives the padded extent with or without ceiling
mode, refuses a zero divisor override and a zero stride, and folds an
unbatched gradient into a batch of one; the adaptive forms accept the
batched ranks of their rank (`_adaptive_avg_pool3d_backward` and the max
pair, plus the unbatched 3-D form of the 2-D max backward that the forward
and the wider contract both support); and the with-indices backwards require
the saved index plane, refusing a missing one or one whose geometry misses
the gradient's, with the unbatched 2-D form likewise folded into a batch of
one.

The embedding backwards answer the table geometry: the dense lookup backward
repeats the gradient's trailing extents behind the row count, refuses index
planes outside the two stored widths, and enforces the gradient's own dtype
gates (8-bit floats have no accumulate form, booleans no subtraction); the
bagged backward answers the same table for every reduction mode,
cross-checking the recorded winner buffer against the gradient plane in max
mode and requiring one offset2bag entry per lookup otherwise; per-sample
weights ride only with the sum mode and answer one scalar per lookup. The
segment-reduce backward spreads the gradient over the data's geometry after
wrapping the axis and requiring either a lengths or an offsets plane, with
the reduction name checked against the family's spellings; the 3-D
grid-sampler backward always answers both gradients — the input's geometry
and the grid's sampling extents carrying the three coordinate components —
refusing the bicubic sweep on 5-D inputs and non-floating operands, while
the requested-slot mask only steers buffer fill on a materializing backend.

The sparse COO factory reuses the shared constructor: the index plane and
the values' dense extents validate against the requested size unchanged,
but inferring the size from the stored index maxima reads element data, so
an explicit size is required. The layout reduction (`_sparse_sum`) collapses
every extent into a scalar in the values' dtype and registers under the
layout key, so a dense operand reaches no kernel at all — the same outcome
a dense tensor gets on a computing backend. The layout product
(`_sparse_addmm`) dispatches from its dense operands and re-runs the dense
side's gates: the sparse operand must be a rank-2 coordinate plane (blocked
layouts aside, no hybrid), all operands share one device and dtype, and the
inner extents must meet before the answer geometry is written. The list
sweeps walk their operands pairwise: `_foreach_copy` clones each source, so
the answer carries the source's metadata, `_foreach_mm` products each pair
through the plain matrix product, and both refuse unequal list lengths.

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

The logarithm, exponential, root and trigonometric sweeps (`exp`, `log` and
its bases, `sqrt`, `rsqrt`, the six hyperbolic and three inverse-trigonometric
pairs, `tanh`, `sigmoid`, `tanhshrink`) compute over real numbers or a complex
lane of their own: a whole-number operand is answered in single precision, a
boolean or 8-bit float operand has no form to run in and is refused, and a
floating or complex operand keeps its width. The single-form sweeps and
activations (`erf`, `erfc`, `lgamma`, `silu`, `gelu`, `mish`, `softplus`, the
`elu`/`selu`/`celu` trio, `leaky_relu`, `hardsigmoid`, `hardswish`, `relu6`,
`log_sigmoid`, `hardtanh`, the clamp family) refuse complex and 8-bit float
operands outright. The same split holds in place: a sweep with a complex lane
accepts a complex target, a single-form sweep refuses it, and the copy-back
activations accept a whole-number target and keep its dtype. The members
without a whole-number form answer what they can — `frac` refuses integral
planes, the angle conversions (`deg2rad`, `rad2deg`) answer complex operands
in single precision — and `sign`, `sgn`, the boolean forms of `pow` and the
step family keep their own promotion tables.

The selection and reduction families police their dtypes before answering
geometry. Every form of `where` reads its condition element by element and
refuses a non-boolean condition. The extrema split by shape: `max`, `min`,
`argmax`, `argmin` and the flat `median` have no complex form and say so,
while `amax`, `amin` and `aminmax` refuse complex planes as type errors only
after the named dimensions pass the range checks. `nanmedian` answers its
flat form in a comparable type — a floating plane keeps its width, every
other plane answers as a whole-number scalar — while its dim form computes
over floating planes only and refuses both whole-number and half-width
operands. The vector norm answers a floating or complex operand directly
(the complex planes in their real component type); a whole-number operand
runs the norm as a chain of powers, so order 1 keeps the operand type, any
other finite order answers in single precision, an infinite order
short-circuits to the largest element and keeps the operand type, a negative
whole-number order is refused, and a boolean operand has no power form and
runs only the infinite orders. The pairwise distances accumulate at single
precision and lift a half-width operand accordingly, and the spread measures
(`var`, `std`, with their paired means) answer a complex operand's spread in
its real component type while the mean keeps the complex plane. An ordering
is undefined over complex planes: the tensor forms of `lt`, `le`, `gt` and
`ge` refuse a complex operand outright, while equality keeps one, and every
scalar comparison form keeps a boolean answer. The ordering selections
split by whether they would actually reorder elements: a sort walks
whole-number and floating planes of any width and refuses complex ones,
while `topk` answers an empty selection, or one over an empty source, for
every plane — nothing is written, so the answer planes exist for all types
— and refuses the ordering walk for boolean and complex planes.

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
