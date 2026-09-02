"""

Complements the codegen-generated ``tensorplay/functional.py``. These
composites are expressed over existing primitives (reshape/permute/
cat/stack/mm/narrow/expand), so autograd flows through them wherever the
underlying ops are differentiable. ``broadcast_tensors`` carries an
explicit backward (ExpandBackward equivalent) because raw ``expand``
views are not yet recorded by the autograd codegen.
"""

import tensorplay
import tensorplay._C as _C
from tensorplay._C import DType
from tensorplay.graph import capture_call as _capture_call
from tensorplay.autograd.function import Function

__all__ = [
    "broadcast_shapes",
    "broadcast_tensors",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "hstack",
    "vstack",
    "row_stack",
    "dstack",
    "column_stack",
    "tensor_split",
    "hsplit",
    "vsplit",
    "dsplit",
    "tensordot",
    "block_diag",
    "unravel_index",
]


class _BroadcastTo(Function):
    """``expand`` with an ExpandBackward-style backward."""

    @staticmethod
    def forward(ctx, input, shape):
        ctx.input_shape = tuple(input.size())
        return input.expand(list(shape))

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.input_shape
        extra = grad_output.dim() - len(shape)
        dims = list(range(max(0, extra)))
        for i, s in enumerate(shape):
            if s == 1:
                dims.append(extra + i)
        if dims:
            grad_output = grad_output.sum(dims, keepdim=True)
        grad_output = grad_output.reshape(list(shape))
        return (grad_output,) + (None,) * (len(ctx.needs_input_grad) - 1)


def broadcast_shapes(*shapes):
    """Returns the broadcast shape of the given shapes (right-aligned)."""
    _captured = _capture_call(broadcast_shapes, shapes, {})
    if _captured is not None:
        return _captured
    result = []
    for shape in shapes:
        sh = tuple(int(d) for d in shape)
        for d in sh:
            if d < 0:
                raise RuntimeError(
                    f"Dimension sizes must be non-negative, got {d}."
                )
        if len(sh) > len(result):
            result = [1] * (len(sh) - len(result)) + result
        offset = len(result) - len(sh)
        for j, d in enumerate(sh):
            idx = offset + j
            r = result[idx]
            if d == 1:
                continue
            if r == 1:
                result[idx] = d
            elif d != r:
                raise RuntimeError(
                    "Shape mismatch: objects cannot be broadcast to a single shape."
                )
    return tuple(result)


def broadcast_tensors(*tensors):
    """Broadcasts the given tensors to a common shape."""
    _captured = _capture_call(broadcast_tensors, tensors, {})
    if _captured is not None:
        return _captured
    if not tensors:
        return []
    for t in tensors:
        if not isinstance(t, tensorplay.Tensor):
            raise TypeError(
                f"broadcast_tensors(): expected Tensor arguments, got {type(t).__name__}"
            )
    shape = broadcast_shapes(*(tuple(t.size()) for t in tensors))
    return [_BroadcastTo.apply(t, shape) for t in tensors]


def atleast_1d(*tensors):
    """Returns each tensor with at least 1 dimension."""
    _captured = _capture_call(atleast_1d, tensors, {})
    if _captured is not None:
        return _captured
    out = []
    for t in tensors:
        if t.dim() == 0:
            out.append(t.reshape([1]))
        else:
            out.append(t)
    return out[0] if len(out) == 1 else out


def atleast_2d(*tensors):
    """Returns each tensor with at least 2 dimensions."""
    _captured = _capture_call(atleast_2d, tensors, {})
    if _captured is not None:
        return _captured
    out = []
    for t in tensors:
        ndim = t.dim()
        if ndim == 0:
            out.append(t.reshape([1, 1]))
        elif ndim == 1:
            out.append(t.reshape([1, t.size(0)]))
        else:
            out.append(t)
    return out[0] if len(out) == 1 else out


def atleast_3d(*tensors):
    """Returns each tensor with at least 3 dimensions."""
    _captured = _capture_call(atleast_3d, tensors, {})
    if _captured is not None:
        return _captured
    out = []
    for t in tensors:
        ndim = t.dim()
        if ndim == 0:
            out.append(t.reshape([1, 1, 1]))
        elif ndim == 1:
            out.append(t.reshape([1, t.size(0), 1]))
        elif ndim == 2:
            out.append(t.reshape([t.size(0), t.size(1), 1]))
        else:
            out.append(t)
    return out[0] if len(out) == 1 else out


def hstack(tensors):
    """Stacks tensors in sequence horizontally (column wise).

    All-1-D inputs concatenate along dim 0; otherwise all inputs must have
    at least 2 dimensions and concatenate along dim 1.
    """
    _captured = _capture_call(hstack, (tensors,), {})
    if _captured is not None:
        return _captured
    ts = list(tensors)
    promoted = [t.reshape([1]) if t.dim() == 0 else t for t in ts]
    ndims = sorted({t.dim() for t in promoted})
    if len(ndims) > 1:
        raise RuntimeError(
            f"Tensors must have same number of dimensions: got {ndims[0]} and {ndims[-1]}"
        )
    if ndims[0] == 1:
        return _C.cat(tensors=promoted, dim=0)
    return _C.cat(tensors=promoted, dim=1)


def vstack(tensors):
    """Stacks tensors in sequence vertically (row wise)."""
    _captured = _capture_call(vstack, (tensors,), {})
    if _captured is not None:
        return _captured
    ts = list(tensors)
    promoted = [t.reshape([1]) if t.dim() == 0 else t for t in ts]
    aligned = [t.reshape([1, t.numel()]) if t.dim() == 1 else t for t in promoted]
    return _C.cat(tensors=aligned, dim=0)


def row_stack(tensors):
    """Alias of :func:`vstack`."""
    return vstack(tensors)


def dstack(tensors):
    """Stacks tensors in sequence depth wise (along third dimension)."""
    _captured = _capture_call(dstack, (tensors,), {})
    if _captured is not None:
        return _captured
    aligned = []
    for t in tensors:
        ndim = t.dim()
        if ndim == 0:
            aligned.append(t.reshape([1, 1, 1]))
        elif ndim == 1:
            aligned.append(t.reshape([1, t.numel(), 1]))
        elif ndim == 2:
            aligned.append(t.reshape([t.size(0), t.size(1), 1]))
        else:
            aligned.append(t)
    return _C.cat(tensors=aligned, dim=2)


def column_stack(tensors):
    """Stacks 1-D tensors as columns and concatenates along dim 1."""
    _captured = _capture_call(column_stack, (tensors,), {})
    if _captured is not None:
        return _captured
    aligned = []
    for t in tensors:
        ndim = t.dim()
        if ndim == 0:
            aligned.append(t.reshape([1, 1]))
        elif ndim == 1:
            aligned.append(t.reshape([t.numel(), 1]))
        else:
            aligned.append(t)
    return _C.cat(tensors=aligned, dim=1)


def _normalize_dim(dim, ndim):
    d = int(dim)
    if d < 0:
        d += ndim
    if d < 0 or d >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )
    return d


def _split_sections(dim_size, sections):
    if sections < 1:
        raise RuntimeError(
            f"Number of sections must be greater than 0, but got {sections}"
        )
    base, rem = divmod(dim_size, sections)
    sizes = [base + 1] * rem + [base] * (sections - rem)
    points = []
    acc = 0
    for s in sizes[:-1]:
        acc += s
        points.append(acc)
    return points


def tensor_split(input, indices_or_sections, dim=0):
    """Splits ``input`` into multiple views along ``dim``.

    ``indices_or_sections`` is either an int (near-equal sections) or a
    """
    _captured = _capture_call(tensor_split, (input, indices_or_sections, dim), {})
    if _captured is not None:
        return _captured
    ndim = input.dim()
    d = _normalize_dim(dim, ndim)
    dim_size = input.size(d)
    if isinstance(indices_or_sections, int) and not isinstance(indices_or_sections, bool):
        points = _split_sections(dim_size, indices_or_sections)
    else:
        points = []
        for p in indices_or_sections:
            p = int(p)
            if p < 0 or p > dim_size:
                raise RuntimeError(
                    f"tensor_split: split point {p} is out of range for dimension size {dim_size}"
                )
            points.append(p)
    bounds = [0] + points + [dim_size]
    pieces = []
    prev = bounds[0]
    for b in bounds[1:]:
        length = max(0, b - prev)
        pieces.append(input.narrow(d, prev, length))
        prev = b if b > prev else prev
    return tuple(pieces)


def hsplit(input, indices_or_sections):
    """Splits along dimension 1 (dimension 0 for 1-D tensors)."""
    _captured = _capture_call(hsplit, (input, indices_or_sections), {})
    if _captured is not None:
        return _captured
    if input.dim() < 1:
        raise RuntimeError(
        )
    dim = 0 if input.dim() == 1 else 1
    return tensor_split(input, indices_or_sections, dim=dim)


def vsplit(input, indices_or_sections):
    """Splits along dimension 0 (requires at least 2 dimensions)."""
    _captured = _capture_call(vsplit, (input, indices_or_sections), {})
    if _captured is not None:
        return _captured
    if input.dim() < 2:
        raise RuntimeError(
        )
    return tensor_split(input, indices_or_sections, dim=0)


def dsplit(input, indices_or_sections):
    """Splits along dimension 2 (requires at least 3 dimensions)."""
    _captured = _capture_call(dsplit, (input, indices_or_sections), {})
    if _captured is not None:
        return _captured
    if input.dim() < 3:
        raise RuntimeError(
        )
    return tensor_split(input, indices_or_sections, dim=2)


def tensordot(input, other, dims=2):
    """Contracts ``input`` and ``other`` over the given dimensions.

    ``dims`` may be a non-negative int (contract the last ``dims`` dims of
    ``input`` with the first ``dims`` dims of ``other``) or a pair
    ``(dims_a, dims_b)`` of dimension lists paired positionally.
    """
    _captured = _capture_call(tensordot, (input, other, dims), {})
    if _captured is not None:
        return _captured
    nd_a = input.dim()
    nd_b = other.dim()

    if isinstance(dims, int) and not isinstance(dims, bool):
        if dims < 0:
            raise RuntimeError(f"tensordot expects dims >= 0, but got dims={dims}")
        if dims > min(nd_a, nd_b):
            raise RuntimeError(
                f"tensordot: dims ({dims}) must be <= the minimum number of dimensions of the inputs"
            )
        dims_a = list(range(nd_a - dims, nd_a))
        dims_b = list(range(dims))
    else:
        da, db = dims
        dims_a = [int(da)] if isinstance(da, int) else [int(x) for x in da]
        dims_b = [int(db)] if isinstance(db, int) else [int(x) for x in db]
        if len(dims_a) != len(dims_b):
            raise RuntimeError("tensordot: both dimension lists must have the same length")
        norm_a = []
        for x in dims_a:
            n = x + nd_a if x < 0 else x
            if n < 0 or n >= nd_a:
                raise RuntimeError(f"tensordot: dimension {x} out of range")
            norm_a.append(n)
        norm_b = []
        for x in dims_b:
            n = x + nd_b if x < 0 else x
            if n < 0 or n >= nd_b:
                raise RuntimeError(f"tensordot: dimension {x} out of range")
            norm_b.append(n)
        dims_a, dims_b = norm_a, norm_b

    for x, y in zip(dims_a, dims_b):
        sa = input.size(x)
        sb = other.size(y)
        if sa != sb:
            raise RuntimeError(
                f"contracted dimensions need to match, but first has size {sa} in dim {x} "
                f"and second has size {sb} in dim {y}"
            )

    free_a = [i for i in range(nd_a) if i not in set(dims_a)]
    free_b = [i for i in range(nd_b) if i not in set(dims_b)]

    # of `input` aligns with the t-th of `other`.  The joint order is fixed by
    # `other`'s axis indices (sorted), and `input`'s pairing follows it --
    pair_order = sorted(range(len(dims_a)), key=lambda t: dims_b[t])
    da_ordered = [dims_a[t] for t in pair_order]
    db_sorted = sorted(dims_b)

    perm_a = free_a + da_ordered
    perm_b = db_sorted + free_b

    contract_size = 1
    for x in dims_a:
        contract_size *= input.size(x)
    fa_shape = [input.size(i) for i in free_a]
    fb_shape = [other.size(i) for i in free_b]

    if not dims_a:
        a2 = input.reshape([-1, 1])
        b2 = other.reshape([1, -1])
    else:
        a2 = input.permute(perm_a).reshape([-1, contract_size])
        b2 = other.permute(perm_b).reshape([contract_size, -1])
    out = a2.mm(b2)
    return out.reshape(fa_shape + fb_shape)


_DTYPE_ORDER = {
    DType.bool: 0,
    DType.uint8: 1,
    DType.int8: 1,
    DType.int16: 2,
    DType.uint16: 2,
    DType.int32: 3,
    DType.uint32: 4,
    DType.int64: 5,
    DType.uint64: 6,
    DType.bfloat16: 7,
    DType.float16: 8,
    DType.float32: 9,
    DType.float64: 10,
}


def _promote_dtypes(dtypes):
    best = dtypes[0]
    for dt in dtypes[1:]:
        if _DTYPE_ORDER.get(dt, -1) > _DTYPE_ORDER.get(best, -1):
            best = dt
    return best


def _device_key(device):
    try:
        return (str(device.type), device.index)
    except AttributeError:
        return str(device)


def block_diag(*tensors):
    """Builds a block diagonal matrix from the given blocks.

    0-D blocks become 1x1 matrices and 1-D blocks become diagonal
    """
    _captured = _capture_call(block_diag, tensors, {})
    if _captured is not None:
        return _captured
    if len(tensors) == 0:
        return _C.zeros(size=[1, 0], dtype=DType.float32)

    blocks = []
    devices = []
    for t in tensors:
        if not isinstance(t, tensorplay.Tensor):
            raise TypeError(
                f"block_diag(): expected Tensor arguments, got {type(t).__name__}"
            )
        ndim = t.dim()
        if ndim == 0:
            blocks.append(t.reshape([1, 1]))
        elif ndim == 1:
            blocks.append(t.reshape([1, t.numel()]))
        elif ndim == 2:
            blocks.append(t)
        else:
            raise RuntimeError(
                f"Expected tensors to have 0, 1, or 2 dimensions, but got {ndim}-D"
            )
        devices.append(_device_key(t.device))

    for key in devices[1:]:
        if key != devices[0]:
            raise RuntimeError(
                "block_diag(): all tensors are expected to be on the same device"
            )

    dtype = _promote_dtypes([b.dtype for b in blocks])
    row_sizes = [b.size(0) for b in blocks]
    col_sizes = [b.size(1) for b in blocks]

    rows = []
    for i, blk in enumerate(blocks):
        pieces = []
        for j in range(len(blocks)):
            if j == i:
                pieces.append(blk.to(dtype) if blk.dtype != dtype else blk)
            else:
                pieces.append(_C.zeros(size=[row_sizes[i], col_sizes[j]], dtype=dtype))
        rows.append(_C.cat(tensors=pieces, dim=1))
    return _C.cat(tensors=rows, dim=0)


def unravel_index(indices, shape):
    """Converts flat indices into coordinate tuples (one LongTensor per dim).

    """
    _captured = _capture_call(unravel_index, (indices, shape), {})
    if _captured is not None:
        return _captured
    if not isinstance(indices, tensorplay.Tensor):
        raise TypeError(
            f"unravel_index(): expected Tensor indices, got {type(indices).__name__}"
        )
    shp = [int(s) for s in shape]
    total = 1
    for s in shp:
        if s < 0:
            raise RuntimeError(f"unravel_index(): shapes must be non-negative, got {s}")
        total *= s
    if total == 0:
        raise RuntimeError("unravel_index(): shape must have at least one non-zero element")

    idx = indices.to(DType.int64)
    remaining = idx - _int_floordiv(idx, total) * total

    coords = []
    for s in reversed(shp):
        c = remaining - _int_floordiv(remaining, s) * s
        coords.append(c)
        remaining = _int_floordiv(remaining - c, s)
    coords.reverse()
    return tuple(coords)


def _int_floordiv(x, m):
    q = (x.to(DType.float64) / m).floor()
    return q.to(DType.int64)
