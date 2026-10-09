# tensorplay.stax

```{eval-rst}
.. currentmodule:: tensorplay.stax
```

Static-graph optimization and acceleration. `stax` traces eager execution into
a static graph, applies compiler passes (constant folding, dead code
elimination, operator decomposition), and lowers the result to the available
backends.

This module is TensorPlay-specific.

## Dynamic shapes

Use `tensorplay.compile(fn, backend="stax", dynamic=True)` to keep input
extents symbolic. Shape reads through `shape`, `size()` and `numel()` remain
expressions; generated CPU and CUDA kernels bind their extents and inferred
strides from the tensors supplied on each call. Pointwise operations,
reductions, reshapes and basic indexing can reuse an artifact across compatible sizes.
Slices preserve symbolic lengths, strides and offsets, including positive steps,
negative bounds and bounds computed from input sizes. Integer indexing and
`select` guard index validity; returned views retain shared storage with their inputs.

```python
import tensorplay as tp

def flatten_rows(x):
    return x.reshape(x.size(0), -1) * 2

compiled = tp.compile(flatten_rows, dynamic=True, strict_native=True)
compiled(tp.randn(3, 7))
compiled(tp.randn(5, 11))
```

Reuse checks the shape decisions and layout relations made during capture and
lowering. A different branch, broadcast pattern or stride arrangement selects
another guarded artifact. Sizes zero and one have separate specializations.
`strict_native=True` raises when a region cannot be generated instead of
executing its graph as a fallback.

Training preserves saved dimensions as symbolic scalar values for pointwise
operations, sum/mean reductions, broadcasting, reshape/expand, squeeze,
slice/select, unflatten, diagonal and repeat derivatives. Variance and standard
deviation derivatives, including combined mean outputs, compute reduction counts
from runtime dimensions. Zero standard deviation masks the incoming variance
gradient, and zero repeats restore a zero gradient of the original input shape.
Compatible sizes reuse one forward artifact and one backward artifact; each
forward invocation saves its own dimensions for the later backward call.
Gradient formulas that still save concrete shape metadata, scalar operands or
shape-derived operator arguments select guarded specializations when that metadata
changes. Variance corrections greater than one also select specializations to
preserve decisions about nonpositive degrees of freedom. Backward input dimensions
are independent even when their sample values happen to coincide.
Symbolic graph capture is available through
`make_graph(..., tracing_mode="symbolic")`; storage-free fake tracing and
data-dependent output extents remain unsupported by that capture entry point.
Tensor inputs nested in containers and call-outs with concrete output metadata
use guarded specializations rather than sharing an artifact across sizes.

## Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    stax
    is_available
```
