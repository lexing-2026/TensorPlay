"""Operator lowerings from a dispatch-level graph to the loop IR.

Each lowering receives the graph node and its already-lowered arguments
(``TensorBox`` for tensors, plain Python values otherwise) and returns the
lowered result.  Element types and shapes always come from the traced value
recorded on the node, so every lowering stores exactly the type the captured
program produced; arithmetic itself runs at float32 (float64 when involved)
and only an explicit conversion rounds in between.

Normalization layers decompose into a welford reduction plus pointwise
work, and their gradients into per-channel sums plus pointwise work, so the
surrounding elementwise code fuses into the same kernels.  Anything without
a lowering runs as a library call on realized inputs.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Callable

from .loops import (
    Buffer,
    Const,
    Pointwise,
    Reduction,
    TensorBox,
    V,
    View,
    as_index,
    contiguous_strides,
    dtype_name,
    floordiv,
    modular_indexing,
    ops,
    prod,
)

LOWERINGS: dict[str, Callable[..., Any]] = {}


def register(*names: str):
    def wrap(fn):
        for name in names:
            LOWERINGS[name] = fn
        return fn

    return wrap


def target_name(target) -> str:
    if target is operator.getitem:
        return "getitem"
    return str(getattr(target, "__name__", target))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def node_val(node, index: int | None = None):
    val = node.meta.get("val")
    if index is not None and isinstance(val, (tuple, list)):
        val = val[index]
    return val


def val_info(val):
    return (
        tuple(int(s) for s in val.shape),
        val.dtype,
        val.device,
    )


def is_tensor_box(x) -> bool:
    return isinstance(x, TensorBox)


def broadcast_loader(x, out_size):
    """Loader of ``x`` indexed in the broadcast output space."""

    if not is_tensor_box(x):
        value = x

        def const(index):
            if isinstance(value, bool):
                return ops.constant(value, "bool")
            if isinstance(value, int):
                return ops.constant(value, "int64")
            return ops.constant(float(value), "float32")

        return const
    size = x.get_size()
    loader = x.make_loader()
    offset = len(out_size) - len(size)

    def load(index):
        return loader(
            [Const(0) if int(size[k]) == 1 else index[k + offset] for k in range(len(size))]
        )

    return load


def pointwise(node, fn, *inputs, val=None):
    val = node_val(node) if val is None else val
    size, dtype, device = val_info(val)
    loaders = [broadcast_loader(x, size) for x in inputs]

    def inner(index):
        return fn(*[load(index) for load in loaders])

    return TensorBox(Pointwise(device, dtype, inner, size))


def cast_to(value, dtype):
    return ops.to_dtype(value, dtype)


def normalize_dim(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


# ---------------------------------------------------------------------------
# pointwise
# ---------------------------------------------------------------------------


def _alpha(args, kwargs, position):
    if len(args) > position:
        return args[position]
    return kwargs.get("alpha", 1)


@register("add.Tensor", "add.Scalar")
def lower_add(node, a, b, *rest, **kwargs):
    alpha = _alpha((a, b, *rest), kwargs, 2)
    if alpha == 1:
        return pointwise(node, ops.add, a, b)
    return pointwise(node, lambda x, y: ops.add(x, ops.mul(y, ops.constant(float(alpha), "float32"))), a, b)


@register("sub.Tensor", "sub.Scalar")
def lower_sub(node, a, b, *rest, **kwargs):
    alpha = _alpha((a, b, *rest), kwargs, 2)
    if alpha == 1:
        return pointwise(node, ops.sub, a, b)
    return pointwise(node, lambda x, y: ops.sub(x, ops.mul(y, ops.constant(float(alpha), "float32"))), a, b)


@register("rsub.Scalar", "rsub.Tensor")
def lower_rsub(node, a, b, *rest, **kwargs):
    return pointwise(node, lambda x, y: ops.sub(y, x), a, b)


@register("mul.Tensor", "mul.Scalar")
def lower_mul(node, a, b):
    return pointwise(node, ops.mul, a, b)


@register("div.Tensor", "div.Scalar")
def lower_div(node, a, b):
    return pointwise(node, ops.truediv, a, b)


def _unary(op_name):
    def lower(node, x):
        return pointwise(node, getattr(ops, op_name), x)

    return lower


for _name, _op in {
    "neg.default": "neg", "exp.default": "exp", "log.default": "log",
    "sigmoid.default": "sigmoid", "rsqrt.default": "rsqrt", "sqrt.default": "sqrt",
    "reciprocal.default": "reciprocal", "abs.default": "abs", "sin.default": "sin",
    "cos.default": "cos", "tanh.default": "tanh", "relu.default": "relu",
}.items():
    LOWERINGS[_name] = _unary(_op)


@register("silu.default")
def lower_silu(node, x):
    # silu(x) = x * sigmoid(x)
    return pointwise(node, lambda v: ops.mul(v, ops.sigmoid(v)), x)


@register("silu_backward.default")
def lower_silu_backward(node, grad, x):
    # grad * s * (1 + x * (1 - s)),  s = sigmoid(x)
    def fn(g, v):
        s = ops.sigmoid(v)
        one = ops.constant(1.0, "float32")
        return ops.mul(ops.mul(g, s), ops.add(one, ops.mul(v, ops.sub(one, s))))

    return pointwise(node, fn, grad, x)


@register("to.dtype", "to.device", "to.dtype_layout", "_to_copy.default")
def lower_to(node, x, *args, **kwargs):
    size, dtype, _ = val_info(node_val(node))
    if is_tensor_box(x) and dtype_name(x.get_dtype()) == dtype_name(dtype) and not kwargs.get("copy", False):
        return x
    return pointwise(node, lambda v: cast_to(v, dtype), x)


@register("clone.default", "contiguous.default")
def lower_clone(node, x, *args, **kwargs):
    return pointwise(node, lambda v: v, x)


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def make_view(x: TensorBox, size, reindex) -> TensorBox:
    return TensorBox(View(x, size, reindex))


def _flat_index(index, size):
    expr = Const(0)
    for i, extent in zip(index, size):
        expr = expr * int(extent) + as_index(i)
    return expr


def _unflatten_index(flat, size):
    out = []
    stride = prod(size)
    for extent in size:
        stride //= max(int(extent), 1)
        if int(extent) == 1:
            out.append(Const(0))
        elif stride == 1:
            out.append(modular_indexing(flat, 1, int(extent)) if out else flat)
        else:
            out.append(modular_indexing(flat, stride, int(extent)) if out else floordiv(flat, Const(stride)))
    return out


def reshape(x: TensorBox, new_size) -> TensorBox:
    old_size = tuple(int(s) for s in x.get_size())
    new_size = tuple(int(s) for s in new_size)
    if old_size == new_size:
        return x
    node = x.node
    if isinstance(node, Buffer) and node.layout.is_contiguous():
        # A contiguous buffer is addressed by the flat element index.
        name = node.name
        offset = node.layout.offset

        def loader_reindex(index):
            return _flat_index(index, new_size)

        class _FlatView(View):
            def make_loader(self):
                def loader(index):
                    return ops.load(name, offset + loader_reindex(index))

                return loader

        return TensorBox(_FlatView(x, new_size, loader_reindex))

    def reindex(index):
        return _unflatten_index(_flat_index(index, new_size), old_size)

    return make_view(x, new_size, reindex)


def _resolve_size(size, numel):
    size = [int(s) for s in size]
    if -1 in size:
        known = prod(s for s in size if s != -1)
        size[size.index(-1)] = numel // max(known, 1)
    return size


@register("view.default", "reshape.default", "_unsafe_view.default", "view.dtype_unused")
def lower_view(node, x, size):
    return reshape(x, _resolve_size(size, x.numel()))


@register("permute.default")
def lower_permute(node, x, dims):
    rank = len(x.get_size())
    dims = [normalize_dim(d, rank) for d in dims]
    size = [x.get_size()[d] for d in dims]
    inverse = [0] * rank
    for position, d in enumerate(dims):
        inverse[d] = position

    def reindex(index):
        return [index[inverse[d]] for d in range(rank)]

    return make_view(x, size, reindex)


@register("permute_backward.default")
def lower_permute_backward(node, grad, _input, dims):
    rank = len(grad.get_size())
    dims = [normalize_dim(d, rank) for d in dims]
    inverse = [0] * rank
    for position, d in enumerate(dims):
        inverse[d] = position
    return lower_permute(node, grad, inverse)


@register("transpose.default", "transpose.int")
def lower_transpose(node, x, d0, d1):
    rank = len(x.get_size())
    dims = list(range(rank))
    a, b = normalize_dim(d0, rank), normalize_dim(d1, rank)
    dims[a], dims[b] = dims[b], dims[a]
    return lower_permute(node, x, dims)


@register("t.default")
def lower_t(node, x):
    return lower_permute(node, x, list(reversed(range(len(x.get_size())))))


@register("unsqueeze.default")
def lower_unsqueeze(node, x, dim):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size) + 1)
    new_size = size[:dim] + [1] + size[dim:]

    def reindex(index):
        return list(index[:dim]) + list(index[dim + 1 :])

    return make_view(x, new_size, reindex)


@register("squeeze.dim", "squeeze.dims", "squeeze.default")
def lower_squeeze(node, x, dim=None):
    size = list(x.get_size())
    rank = len(size)
    if dim is None:
        dims = [d for d in range(rank) if size[d] == 1]
    elif isinstance(dim, (list, tuple)):
        dims = [normalize_dim(d, rank) for d in dim if size[normalize_dim(d, rank)] == 1]
    else:
        d = normalize_dim(dim, rank)
        dims = [d] if size[d] == 1 else []
    if not dims:
        return x
    new_size = [s for d, s in enumerate(size) if d not in dims]

    def reindex(index):
        it = iter(index)
        return [Const(0) if d in dims else next(it) for d in range(rank)]

    return make_view(x, new_size, reindex)


@register("expand.default")
def lower_expand(node, x, size, *args, **kwargs):
    size = [int(s) for s in size]
    old = list(x.get_size())
    offset = len(size) - len(old)
    size = [old[d - offset] if s == -1 else s for d, s in enumerate(size)]

    def reindex(index):
        return [Const(0) if old[k] == 1 else index[k + offset] for k in range(len(old))]

    return make_view(x, size, reindex)


def _slice(x, dim, start, end, step):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    extent = size[dim]
    start = 0 if start is None else int(start)
    end = extent if end is None else int(end)
    if start < 0:
        start += extent
    if end < 0:
        end += extent
    start = max(0, min(start, extent))
    end = max(start, min(end, extent))
    step = int(step or 1)
    new_size = list(size)
    new_size[dim] = (end - start + step - 1) // step
    if start == 0 and step == 1 and end == extent:
        return x

    def reindex(index):
        out = list(index)
        out[dim] = index[dim] * step + start
        return out

    return make_view(x, new_size, reindex)


@register("slice.Tensor")
def lower_slice(node, x, dim=0, start=None, end=None, step=1):
    return _slice(x, dim, start, end, step)


@register("chunk.default")
def lower_chunk(node, x, chunks, dim=0):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    piece = (size[dim] + chunks - 1) // chunks
    out = []
    start = 0
    while start < size[dim]:
        out.append(_slice(x, dim, start, min(start + piece, size[dim]), 1))
        start += piece
    return tuple(out)


@register("split.Tensor")
def lower_split(node, x, split_size, dim=0):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    out = []
    start = 0
    while start < size[dim]:
        out.append(_slice(x, dim, start, min(start + int(split_size), size[dim]), 1))
        start += int(split_size)
    return tuple(out)


@register("split_with_sizes.default")
def lower_split_with_sizes(node, x, sizes, dim=0):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    out = []
    start = 0
    for s in sizes:
        out.append(_slice(x, dim, start, start + int(s), 1))
        start += int(s)
    return tuple(out)


@register("cat.default")
def lower_cat(node, tensors, dim=0):
    """Concatenation as one pointwise loop selecting its source per index."""

    size, dtype, device = val_info(node_val(node))
    dim = normalize_dim(dim, len(size))
    inputs = [t for t in tensors if is_tensor_box(t) and t.get_size()[dim] > 0]
    if not inputs:
        # Every operand was a constant or empty, so there is no source to read
        # per index.  Say so instead of emitting a body that yields nothing.
        raise NotImplementedError(
            "cat with no tensor operand to read per index"
            f" (operands={len(tensors)}, dim={dim})"
        )
    starts = []
    start = 0
    for t in inputs:
        starts.append(start)
        start += int(t.get_size()[dim])
    loaders = [t.make_loader() for t in inputs]

    def inner(index):
        position = ops.index_expr(index[dim], "int64")
        value = None
        for k in range(len(inputs)):
            lo = starts[k]
            hi = lo + int(inputs[k].get_size()[dim])
            shifted = list(index)
            shifted[dim] = index[dim] - lo
            if k == 0:
                cond = ops.lt(position, ops.constant(hi, "int64"))
            elif k == len(inputs) - 1:
                cond = ops.ge(position, ops.constant(lo, "int64"))
            else:
                cond = ops.and_(
                    ops.ge(position, ops.constant(lo, "int64")),
                    ops.lt(position, ops.constant(hi, "int64")),
                )
            loaded = ops.masked(cond, lambda k=k, shifted=shifted: loaders[k](shifted), 0.0)
            value = loaded if value is None else ops.where(cond, loaded, value)
        return value

    return TensorBox(Pointwise(device, dtype, inner, size))


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------


def make_reduction(x: TensorBox, dims, keepdim, dtype, device, rtype="sum", prologue=None) -> TensorBox:
    size = list(x.get_size())
    rank = len(size)
    dims = sorted({normalize_dim(d, rank) for d in dims})
    out_ranges = [size[d] for d in range(rank) if d not in dims]
    red_ranges = [size[d] for d in dims]
    loader = x.make_loader()

    def inner(index, rindex):
        it = iter(index)
        rt = iter(rindex)
        full = [next(rt) if d in dims else next(it) for d in range(rank)]
        value = loader(full)
        if prologue is not None:
            value = prologue(value, full)
        return value

    box = TensorBox(Reduction(device, dtype, inner, out_ranges, red_ranges, rtype))
    box.realize()
    if keepdim:
        kept = [1 if d in dims else size[d] for d in range(rank)]

        def reindex(index):
            return [index[d] for d in range(rank) if d not in dims]

        return make_view(box, kept, reindex)
    return box


@register("sum.dim_IntList", "sum.default")
def lower_sum(node, x, dims=None, keepdim=False, **kwargs):
    size, dtype, device = val_info(node_val(node))
    if not dims:
        dims = list(range(len(x.get_size())))
    return make_reduction(x, dims, keepdim, dtype, device, "sum")


@register("mean.dim")
def lower_mean(node, x, dims, keepdim=False, **kwargs):
    size, dtype, device = val_info(node_val(node))
    count = prod(x.get_size()[normalize_dim(d, len(x.get_size()))] for d in dims)
    total = make_reduction(x, dims, keepdim, dtype, device, "sum")
    return pointwise(node, lambda v: ops.truediv(v, ops.constant(float(count), "float32")), total)


@register("amax.default")
def lower_amax(node, x, dims=None, keepdim=False, **kwargs):
    size, dtype, device = val_info(node_val(node))
    if not dims:
        dims = list(range(len(x.get_size())))
    return make_reduction(x, dims, keepdim, dtype, device, "max")


@register("amin.default")
def lower_amin(node, x, dims=None, keepdim=False, **kwargs):
    size, dtype, device = val_info(node_val(node))
    if not dims:
        dims = list(range(len(x.get_size())))
    return make_reduction(x, dims, keepdim, dtype, device, "min")


@register("conv2d_grad_bias.default", "conv_grad_bias.default")
def lower_conv_grad_bias(node, grad_out, *args):
    """The bias gradient is the output gradient summed over all but channels."""

    size, dtype, device = val_info(node_val(node))
    rank = len(grad_out.get_size())
    return make_reduction(grad_out, [0] + list(range(2, rank)), False, dtype, device, "sum")


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def _group_view(x: TensorBox, n, groups, row):
    """``x`` addressed as (N, G, C/G * spatial) rows."""

    return reshape(x, (n, groups, row))


@register("native_group_norm.default")
def lower_native_group_norm(node, x, weight, bias, n, c, hxw, groups, eps):
    out_val, mean_val, rstd_val = node_val(node)
    out_size, out_dtype, device = val_info(out_val)
    stat_dtype = mean_val.dtype
    cpg = c // groups
    row = cpg * hxw
    rows = _group_view(x, n, groups, row)
    rows_loader = rows.make_loader()

    def welford_inner(index, rindex):
        return ops.to_dtype(rows_loader([index[0], index[1], rindex[0]]), "float32")

    stats = TensorBox(
        Reduction(device, stat_dtype, welford_inner, (n, groups), (row,), "welford")
    )
    mean_buf, m2_buf = V.graph.register_welford(stats.node)
    mean_box = TensorBox(mean_buf)
    m2_loader = TensorBox(m2_buf).make_loader()
    mean_loader = mean_box.make_loader()

    def rstd_at(ng):
        var = ops.truediv(m2_loader(ng), ops.constant(float(row), "float32"))
        return ops.rsqrt(ops.add(var, ops.constant(float(eps), "float32")))

    rstd_box = TensorBox(
        Pointwise(device, stat_dtype, lambda idx: rstd_at(idx), (n, groups))
    )
    x_loader = x.make_loader()
    w_loader = weight.make_loader() if is_tensor_box(weight) else None
    b_loader = bias.make_loader() if is_tensor_box(bias) else None
    spatial = list(out_size[2:])

    def out_inner(index):
        channel = index[1]
        group = floordiv(as_index(channel), Const(cpg))
        ng = [index[0], group]
        value = ops.to_dtype(x_loader(index), "float32")
        value = ops.mul(ops.sub(value, mean_loader(ng)), rstd_at(ng))
        if w_loader is not None:
            value = ops.mul(value, w_loader([channel]))
        if b_loader is not None:
            value = ops.add(value, b_loader([channel]))
        return value

    out = TensorBox(Pointwise(device, out_dtype, out_inner, out_size))
    return (out, mean_box, rstd_box)


@register("native_group_norm_backward.default")
def lower_native_group_norm_backward(node, grad_out, x, mean, rstd, gamma, n, c, hxw, groups, output_mask):
    vals = node_val(node)
    cpg = c // groups
    device = grad_out.get_device()
    rank = len(x.get_size())
    spatial = list(x.get_size()[2:])
    dy_loader = grad_out.make_loader()
    x_loader = x.make_loader()
    mean_loader = mean.make_loader()
    rstd_loader = rstd.make_loader()
    gamma_loader = gamma.make_loader() if is_tensor_box(gamma) else None
    f32 = "float32"

    # Per (n, c) sums over the spatial extent: ds = sum(dy * x), db = sum(dy).
    def full_index(index, rindex):
        s = rindex[0]
        return [index[0], index[1]] + _unflatten_index(s, spatial)

    def ds_inner(index, rindex):
        full = full_index(index, rindex)
        return ops.mul(ops.to_dtype(dy_loader(full), f32), ops.to_dtype(x_loader(full), f32))

    def db_inner(index, rindex):
        return ops.to_dtype(dy_loader(full_index(index, rindex)), f32)

    ds = TensorBox(Reduction(device, "float32", ds_inner, (n, c), (hxw,), "sum"))
    db = TensorBox(Reduction(device, "float32", db_inner, (n, c), (hxw,), "sum"))
    ds.realize()
    db.realize()
    ds_loader = ds.make_loader()
    db_loader = db.make_loader()

    def gamma_at(ch):
        return ops.to_dtype(gamma_loader([ch]), f32) if gamma_loader is not None else ops.constant(1.0, f32)

    results = [None, None, None]
    s = 1.0 / (hxw * cpg)
    if output_mask[0]:
        def dsv_inner(index, rindex):
            ch = index[1] * cpg + rindex[0]
            return ops.mul(ds_loader([index[0], ch]), gamma_at(ch))

        def dbv_inner(index, rindex):
            ch = index[1] * cpg + rindex[0]
            return ops.mul(db_loader([index[0], ch]), gamma_at(ch))

        ds_val = TensorBox(Reduction(device, "float32", dsv_inner, (n, groups), (cpg,), "sum"))
        db_val = TensorBox(Reduction(device, "float32", dbv_inner, (n, groups), (cpg,), "sum"))
        ds_val.realize()
        db_val.realize()
        dsv = ds_val.make_loader()
        dbv = db_val.make_loader()

        def c2_at(ng):
            r = ops.to_dtype(rstd_loader(ng), f32)
            m = ops.to_dtype(mean_loader(ng), f32)
            num = ops.sub(ops.mul(dbv(ng), m), dsv(ng))
            return ops.mul(ops.mul(ops.mul(ops.mul(num, r), r), r), ops.constant(s, f32))

        def c3_at(ng):
            r = ops.to_dtype(rstd_loader(ng), f32)
            m = ops.to_dtype(mean_loader(ng), f32)
            left = ops.mul(ops.neg(c2_at(ng)), m)
            right = ops.mul(ops.mul(dbv(ng), r), ops.constant(s, f32))
            return ops.sub(left, right)

        c2 = TensorBox(Pointwise(device, "float32", c2_at, (n, groups)))
        c3 = TensorBox(Pointwise(device, "float32", c3_at, (n, groups)))
        c2.realize()
        c3.realize()
        c2_loader = c2.make_loader()
        c3_loader = c3.make_loader()
        dx_size, dx_dtype, _ = val_info(vals[0])

        def dx_inner(index):
            ch = index[1]
            ng = [index[0], floordiv(as_index(ch), Const(cpg))]
            c1 = ops.mul(ops.to_dtype(rstd_loader(ng), f32), gamma_at(ch))
            dy = ops.to_dtype(dy_loader(index), f32)
            xv = ops.to_dtype(x_loader(index), f32)
            return ops.add(ops.add(ops.mul(dy, c1), ops.mul(xv, c2_loader(ng))), c3_loader(ng))

        results[0] = TensorBox(Pointwise(device, dx_dtype, dx_inner, dx_size))
    if output_mask[1]:
        dg_size, dg_dtype, _ = val_info(vals[1])

        def dgamma_inner(index, rindex):
            ch = index[0]
            ng = [rindex[0], floordiv(as_index(ch), Const(cpg))]
            m = ops.to_dtype(mean_loader(ng), f32)
            r = ops.to_dtype(rstd_loader(ng), f32)
            nc = [rindex[0], ch]
            return ops.mul(ops.sub(ds_loader(nc), ops.mul(db_loader(nc), m)), r)

        results[1] = TensorBox(Reduction(device, dg_dtype, dgamma_inner, (c,), (n,), "sum"))
        results[1].realize()
    if output_mask[2]:
        db_size, db_dtype, _ = val_info(vals[2])

        def dbeta_inner(index, rindex):
            return db_loader([rindex[0], index[0]])

        results[2] = TensorBox(Reduction(device, db_dtype, dbeta_inner, (c,), (n,), "sum"))
        results[2].realize()
    return tuple(results)


__all__ = ["LOWERINGS", "broadcast_loader", "make_reduction", "pointwise", "register", "reshape", "target_name"]


# ---------------------------------------------------------------------------
# sampling operators
# ---------------------------------------------------------------------------


def _pair(value, count: int) -> list:
    """A kernel/stride/padding argument as one entry per spatial axis."""

    if isinstance(value, int):
        return [int(value)] * count
    items = [int(v) for v in value]
    if len(items) == 1:
        return items * count
    return items


def _pool_output_size(extent: int, kernel: int, stride: int, padding: int,
                      ceil_mode: bool) -> int:
    """Output extent of one pooling axis."""

    if ceil_mode:
        return -((-(extent + 2 * padding - kernel)) // stride) + 1
    return (extent + 2 * padding - kernel) // stride + 1


@register("upsample_nearest2d.default", "_upsample_nearest_exact2d.default",
          "upsample_nearest3d.default", "_upsample_nearest_exact3d.default")
def lower_upsample_nearestnd(node, x, output_size, scales_h=None, scales_w=None,
                             **kwargs):
    """Nearest upsampling as an index remap of the source.

    Each output element reads the input element the scale maps it to, so the
    operator is a view with a remapped address rather than a call: it fuses
    with whatever consumes it instead of standing on its own.
    """

    size, dtype, device = val_info(node_val(node))
    in_size = list(x.get_size())
    ndim = 3 if "3d" in target_name(node.target) else 2
    out_spatial = [int(s) for s in output_size][-ndim:]
    in_spatial = in_size[-ndim:]
    prefix = in_size[:-ndim]

    def reindex(index):
        # Nearest maps output position i to floor(i / scale) of the input.
        return [
            *index[: len(prefix)],
            *[
                floordiv(
                    as_index(index[len(prefix) + axis]) * i, Const(o)
                )
                for axis, (i, o) in enumerate(zip(in_spatial, out_spatial))
            ],
        ]

    return make_view(x, size, reindex)


@register("avg_pool2d.default", "avg_pool3d.default")
def lower_avg_poolnd(node, x, kernel_size, stride=(), padding=0, ceil_mode=False,
                     count_include_pad=True, divisor_override=None, **kwargs):
    """Average pooling as a window sum followed by the window's divisor.

    A window that is both large and overlapping is left to the operator: the
    decomposition reads the input once per window, which stops paying once
    the windows overlap heavily.
    """

    size, dtype, device = val_info(node_val(node))
    in_size = list(x.get_size())
    ndim = 3 if "3d" in target_name(node.target) else 2
    kernel = _pair(kernel_size, ndim)
    stride = _pair(stride, ndim) if stride else list(kernel)
    padding = _pair(padding, ndim) if padding else [0] * ndim
    window = 1
    for extent in kernel:
        window *= extent
    if window > 25 and any(k != s for k, s in zip(kernel, stride)):
        raise NotImplementedError(
            f"average pooling with an overlapping {window}-element window"
        )
    spatial_in = in_size[-ndim:]
    spatial_out = [
        _pool_output_size(extent, k, s, p, bool(ceil_mode))
        for extent, k, s, p in zip(spatial_in, kernel, stride, padding)
    ]
    prefix = in_size[: len(in_size) - ndim]
    loader = x.make_loader()
    boundary = any(padding)
    f32 = "float32"

    def inner(index, rindex):
        full = list(index[: len(prefix)])
        for axis in range(ndim):
            base = index[len(prefix) + axis]
            full.append(base * stride[axis] - padding[axis] + rindex[axis])
        if not boundary:
            return loader(full)
        # A padded window reads zero outside the source, so the sum skips it.
        outside = None
        for axis in range(ndim):
            # The address is index arithmetic; a bound test needs it as a value.
            position = ops.index_expr(full[len(prefix) + axis], "int64")
            low = ops.ge(position, ops.constant(0, "int64"))
            high = ops.lt(
                position, ops.constant(spatial_in[axis], "int64")
            )
            inside = ops.and_(low, high)
            outside = inside if outside is None else ops.and_(outside, inside)
        return ops.masked(outside, lambda: loader(full), ops.constant(0.0, f32))

    total = TensorBox(
        Reduction(
            device, f32, inner, (*prefix, *spatial_out), tuple(kernel), "sum"
        )
    )
    total.realize()
    if divisor_override is not None:
        divisor = float(divisor_override)
    elif count_include_pad or not any(padding):
        divisor = float(window)
    else:
        # Only the positions inside the source contribute to the divisor.
        divisor = None
    if divisor is None:
        return _pool_with_masked_divisor(
            node, total, prefix, spatial_out, kernel, stride, padding,
            spatial_in, f32, device,
        )
    return pointwise(
        node,
        lambda value: ops.truediv(value, ops.constant(divisor, f32)),
        total,
    )


def _pool_with_masked_divisor(node, total, prefix, spatial_out, kernel, stride,
                              padding, spatial_in, f32, device):
    """Average pooling whose divisor counts only the positions inside."""

    def inner(index, rindex):
        full = list(index[: len(prefix)])
        for axis in range(len(kernel)):
            full.append(
                index[len(prefix) + axis] * stride[axis] - padding[axis] + rindex[axis]
            )
        inside = None
        for axis in range(len(kernel)):
            position = ops.index_expr(full[len(prefix) + axis], "int64")
            term = ops.and_(
                ops.ge(position, ops.constant(0, "int64")),
                ops.lt(position, ops.constant(spatial_in[axis], "int64")),
            )
            inside = term if inside is None else ops.and_(inside, term)
        return ops.masked(inside, lambda: ops.constant(1.0, f32),
                          ops.constant(0.0, f32))

    counted = TensorBox(
        Reduction(
            device, f32, inner, (*prefix, *spatial_out), tuple(kernel), "sum",
        )
    )
    counted.realize()
    return pointwise(
        node,
        lambda value, count: ops.truediv(value, count),
        total,
        counted,
    )
