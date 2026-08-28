# Ported from torch/distributed/_functional_collectives.py.
#
# This module provides tensor-collectives that participate in autograd
# (``*_autograd`` variants use ``tensorplay.autograd.Function``), matching
# the public surface torch's async-TP and python-reducer DDP paths rely on.
#
# The *_coalesced family is implemented natively with NCCL group semantics:
# every per-tensor enqueue is issued inside a single ``groupStart/groupEnd``
# window so the communicator batches the launch, mirroring torch's C++
# coalescing manager. Ops enqueue on the current stream and return real
# tensors; ``wait_tensor`` is kept for torch API parity (it synchronizes
# torch's AsyncCollectiveTensor, here it is a passthrough).

import warnings

import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.autograd.function import Function
from tensorplay.distributed.distributed_c10d import (
    _get_default_group as _c10d_default_group,
    _resolve_group as _c10d_resolve_group,
)


__all__ = [
    "wait_tensor",
    "broadcast",
    "all_reduce",
    "all_gather_tensor",
    "all_gather_tensor_autograd",
    "reduce_scatter_tensor",
    "reduce_scatter_tensor_autograd",
    "all_reduce_coalesced",
    "all_gather_single_coalesced",
    "reduce_scatter_single_coalesced",
    "all_gather_into_tensor_coalesced",
    "reduce_scatter_tensor_coalesced",
]


RANK_TYPES = "int | ProcessGroup | str"

_REDUCE_OPS = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "product": dist.ReduceOp.PRODUCT,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}


def wait_tensor(tensor, timeout=None):
    """Wait until the collective that produced ``tensor`` completes."""
    if isinstance(tensor, tp.Tensor) and hasattr(tensor, "_wait_tensor"):
        tensor._wait_tensor(timeout)
    return tensor


def _resolve_group(group):
    if group is None:
        return _c10d_default_group()
    if isinstance(group, dist.ProcessGroup):
        return group
    if isinstance(group, str):
        return _c10d_resolve_group(group)
    if isinstance(group, int):  # global rank prefix of a sub-group
        return _c10d_default_group()
    raise ValueError(f"Unsupported group type: {type(group)}")


class _CollectiveFunctionBase(Function):
    @staticmethod
    def _group_size(group):
        return max(1, group.size())


class AllReduceWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, reduce_op, tensor_input, op):
        ctx.group_name = group_name
        ctx.op = op
        output = tensor_input.clone()
        pg = _resolve_group(group_name)
        dist.all_reduce(output, op=op, group=pg)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.clone()
        pg = _resolve_group(ctx.group_name)
        dist.all_reduce(grad_out, op=ctx.op, group=pg)
        grad_out.div_(pg.size())
        return None, None, grad_out, None


def all_reduce(tensor_input, reduce_op="sum", group=None, *, op=None):
    """All-reduce the input across the entire group with autograd support."""
    if op is not None:  # torch positional-compat alias
        reduce_op = op
    op_int = {
        "sum": dist.ReduceOp.SUM,
        "avg": dist.ReduceOp.AVG,
        "product": dist.ReduceOp.PRODUCT,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
    }.get(reduce_op, dist.ReduceOp.SUM)
    pg = _resolve_group(group)
    return AllReduceWithAutograd.apply(pg.group_name, reduce_op, tensor_input,
                                       op_int)


class BroadcastWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, src_rank, group_name, tensor_input):
        ctx.src = src_rank
        pg = _resolve_group(group_name)
        output = tensor_input.detach().clone()
        src_global = dist.get_global_rank(pg, src_rank)
        dist.broadcast(output, src=src_global, group=pg)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Broadcast is treated as non-differentiable (torch parity).
        return None, None, None


def broadcast(tensor_input, src, group=None):
    """Broadcast the tensor from a rank to the whole group."""
    pg = _resolve_group(group)
    return BroadcastWithAutograd.apply(src, pg.group_name, tensor_input)


class AllGatherTensorWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, group_size, rank, tensor_input):
        ctx.group_name = group_name
        ctx.rank = rank
        ctx.group_size = group_size
        pg = _resolve_group(group_name)
        n = tensor_input.numel()
        out = tp.zeros(group_size * n, dtype=tensor_input.dtype,
                       device=tensor_input.device)
        send_t = tensor_input.contiguous() if hasattr(tensor_input, "contiguous") \
            else tensor_input
        from tensorplay._C import _distributed as _C

        _C.all_gather(out, send_t, pg.comm)
        out = out.view([group_size] + list(tensor_input.shape))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        pg = _resolve_group(ctx.group_name)
        g = grad_output[ctx.rank]
        # Sum contributions from other ranks' shards (they depend on this
        # rank's data through subsequent ops).
        total = g.contiguous().reshape(-1)
        for r in range(ctx.group_size):
            if r == ctx.rank:
                continue
            total = total + grad_output[r].contiguous().reshape(-1)
        grad_in = total.reshape(g.shape)
        return None, None, None, grad_in


def all_gather_tensor(tensor_input, gather_dim=0, group=None):
    """All-gather a tensor along ``gather_dim`` with autograd support."""
    pg = _resolve_group(group)
    rank = pg.rank()
    size = pg.size()
    out = AllGatherTensorWithAutograd.apply(pg.group_name, size, rank,
                                            tensor_input)
    return out.movedim(0, gather_dim).reshape(
        [d * size if i == gather_dim else d
         for i, d in enumerate(tensor_input.shape)])


all_gather_tensor_autograd = all_gather_tensor


class ReduceScatterTensorWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, group_size, rank, reduce_op, tensor_input):
        ctx.group_name = group_name
        ctx.rank = rank
        pg = _resolve_group(group_name)
        from tensorplay._C import _distributed as _C

        shard = tensor_input.numel() // group_size
        out = tp.zeros(shard, dtype=tensor_input.dtype,
                       device=tensor_input.device)
        send_t = tensor_input.contiguous() if hasattr(tensor_input, "contiguous") \
            else tensor_input
        _C.reduce_scatter(out, send_t, int(reduce_op), pg.comm)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        pg = _resolve_group(ctx.group_name)
        from tensorplay._C import _distributed as _C

        n = grad_output.numel()
        inp = tp.zeros(ctx.group_size * n, dtype=grad_output.dtype,
                       device=grad_output.device)
        _C.all_gather(inp, grad_output.contiguous(), pg.comm)
        inp.div_(ctx.group_size)
        return None, None, None, None, inp


def reduce_scatter_tensor(tensor_input, reduce_op="sum", group=None,
                          scatter_dim=0):
    """Reduce-scatter a tensor along ``scatter_dim`` with autograd support."""
    pg = _resolve_group(group)
    op_int = {
        "sum": dist.ReduceOp.SUM,
        "avg": dist.ReduceOp.AVG,
        "product": dist.ReduceOp.PRODUCT,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
    }.get(reduce_op, dist.ReduceOp.SUM)
    size = pg.size()
    shape = list(tensor_input.shape)
    assert shape[scatter_dim] % size == 0, (
        f"collective reduce_scatter_tensor requires the dim {scatter_dim} "
        f"to be equally split across the given group ({size}), but the "
        f"dim size {shape[scatter_dim]} is not divisible by {size}.")
    flat = tensor_input.movedim(scatter_dim, 0).reshape(-1)
    out = ReduceScatterTensorWithAutograd.apply(pg.group_name, size,
                                                pg.rank(), op_int, flat)
    shard_shape = [d // size if i == 0 else d for i, d in enumerate(
        [shape[scatter_dim]] + shape[:scatter_dim] + shape[scatter_dim + 1:])]
    return out.view(shard_shape).movedim(0, scatter_dim)


reduce_scatter_tensor_autograd = reduce_scatter_tensor


# ---------------------------------------------------------------------------
# Coalesced family (torch parity: one groupStart/groupEnd window per call).
# ---------------------------------------------------------------------------


def _resolve_coalesced_group(group):
    if group is None:
        return _c10d_default_group()
    if isinstance(group, dist.ProcessGroup):
        return group
    if isinstance(group, str):
        return _c10d_resolve_group(group)
    raise ValueError(
        f"Unsupported group type: {type(group)}; expected ProcessGroup or None")


class AllReduceCoalescedWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, reduce_op, op_int, *tensors):
        from tensorplay._C import _distributed as _C

        ctx.group_name = group_name
        ctx.reduce_op = str(reduce_op).lower()
        pg = _resolve_coalesced_group(group_name)
        # The engine pads backward grad slots to the input arity with None;
        # keep per-input metas so unused outputs map to zero grads.
        outputs = []
        _C.group_start()
        try:
            for t in tensors:
                out = t.detach().clone().contiguous()
                _C.all_reduce(out, int(op_int), pg.comm)
                outputs.append(out)
        except BaseException:
            _C.group_end()
            raise
        _C.group_end()
        ctx._in_metas = [(list(o.shape), o.dtype, o.device) for o in outputs]
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.reduce_op != "sum":
            raise RuntimeError(
                "all_reduce_coalesced backward only supports 'sum' reduction, "
                f"got '{ctx.reduce_op}'")
        pg = _resolve_coalesced_group(ctx.group_name)
        from tensorplay._C import _distributed as _C

        grads = [g.contiguous() if g is not None else tp.zeros(*meta)
                 for g, meta in zip(grad_outputs, ctx._in_metas)]
        outs = []
        _C.group_start()
        try:
            for g in grads:
                out = g.clone()
                _C.all_reduce(out, int(dist.ReduceOp.SUM), pg.comm)
                outs.append(out)
        except BaseException:
            _C.group_end()
            raise
        _C.group_end()
        return (None, None, None, *outs)


def all_reduce_coalesced(self_tensor_list, reduce_op="sum", group=None, tag=""):
    """All-reduce a list of tensors in one coalesced group launch."""
    pg = _resolve_coalesced_group(group)
    op_int = _REDUCE_OPS.get(str(reduce_op).lower())
    if op_int is None:
        raise ValueError(f"Invalid reduce op: {reduce_op}")
    tensors = tuple(self_tensor_list)
    outputs = AllReduceCoalescedWithAutograd.apply(
        pg.group_name, str(reduce_op).lower(), int(op_int), *tensors)
    return list(outputs)


class AllGatherSingleCoalescedWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, group_size, *tensors):
        from tensorplay._C import _distributed as _C

        ctx.group_name = group_name
        ctx.group_size = group_size
        pg = _resolve_coalesced_group(group_name)
        outputs = []
        _C.group_start()
        try:
            for t in tensors:
                send_t = t.detach().contiguous()
                n = send_t.numel()
                flat = tp.zeros(group_size * n, dtype=send_t.dtype,
                                device=send_t.device)
                _C.all_gather(flat, send_t, pg.comm)
                outputs.append(flat.view([group_size] + list(send_t.shape)))
        except BaseException:
            _C.group_end()
            raise
        _C.group_end()
        ctx._in_metas = [(list(o.shape), o.dtype, o.device) for o in outputs]
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # torch parity: gather forward, scatter-sum backward.
        pg = _resolve_coalesced_group(ctx.group_name)
        from tensorplay._C import _distributed as _C

        grads = [g.contiguous() if g is not None else tp.zeros(*meta)
                 for g, meta in zip(grad_outputs, ctx._in_metas)]
        outs = []
        _C.group_start()
        try:
            for g in grads:
                shard = g[0] if g.dim() > 0 else g
                n = shard.numel()
                flat_g = g.reshape(-1)
                out = tp.zeros(n, dtype=flat_g.dtype, device=flat_g.device)
                _C.reduce_scatter(out, flat_g, int(dist.ReduceOp.SUM), pg.comm)
                outs.append(out.view(list(shard.shape)))
        except BaseException:
            _C.group_end()
            raise
        _C.group_end()
        return (None, None, *outs)


def all_gather_single_coalesced(self_tensor_list, group=None, tag=""):
    """All-gather a list of tensors (gather_dim=0) in one coalesced launch."""
    pg = _resolve_coalesced_group(group)
    tensors = tuple(self_tensor_list)
    outputs = AllGatherSingleCoalescedWithAutograd.apply(
        pg.group_name, pg.size(), *tensors)
    return list(outputs)


class ReduceScatterSingleCoalescedWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, group_size, reduce_op, op_int, *tensors):
        from tensorplay._C import _distributed as _C

        ctx.group_name = group_name
        ctx.group_size = group_size
        ctx.reduce_op = str(reduce_op).lower()
        pg = _resolve_coalesced_group(group_name)
        outputs = []
        _C.group_start()
        try:
            for t in tensors:
                send_t = t.detach().contiguous()
                n = send_t.numel() // group_size
                out = tp.zeros(n, dtype=send_t.dtype, device=send_t.device)
                _C.reduce_scatter(out, send_t, int(op_int), pg.comm)
                outputs.append(out)
        except BaseException:
            _C.group_end()
            raise
        _C.group_end()
        ctx._in_metas = [(list(o.shape), o.dtype, o.device) for o in outputs]
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.reduce_op != "sum":
            raise RuntimeError(
                "reduce_scatter_tensor_coalesced backward only supports "
                f"'sum' reduction, got '{ctx.reduce_op}'")
        # torch parity: scatter forward, gather backward (no division).
        pg = _resolve_coalesced_group(ctx.group_name)
        from tensorplay._C import _distributed as _C

        grads = [g.contiguous() if g is not None else tp.zeros(*meta)
                 for g, meta in zip(grad_outputs, ctx._in_metas)]
        outs = []
        _C.group_start()
        try:
            for g in grads:
                n = g.numel()
                flat = tp.zeros(ctx.group_size * n, dtype=g.dtype,
                                device=g.device)
                _C.all_gather(flat, g, pg.comm)
                outs.append(flat)
        except BaseException:
            _C.group_end()
            raise
        _C.group_end()
        return (None, None, None, None, *outs)


def reduce_scatter_single_coalesced(inputs, reduce_op, scatter_dim, group=None,
                                    tag=""):
    """Reduce-scatter a list of tensors in one coalesced group launch."""
    pg = _resolve_coalesced_group(group)
    op_int = _REDUCE_OPS.get(str(reduce_op).lower())
    if op_int is None:
        raise ValueError(f"Invalid reduce op: {reduce_op}")
    group_size = pg.size()
    inputs = list(inputs)
    if len(scatter_dim) != len(inputs):
        raise AssertionError(
            f"Length of scatter_dim ({len(scatter_dim)}) must equal length "
            f"of inputs ({len(inputs)})")
    for idx, (dim, tensor) in enumerate(zip(scatter_dim, inputs)):
        size_dim = tensor.shape[dim]
        if size_dim % group_size != 0:
            raise AssertionError(
                f"input dimension {dim} ({size_dim} must be a multiple of "
                f"group_size {group_size} for tensor at index {idx}")
        if dim != 0:
            inputs[idx] = tp.cat(tp.chunk(tensor, group_size, dim=dim))
    flat = [t.reshape(-1) for t in inputs]
    outputs = ReduceScatterSingleCoalescedWithAutograd.apply(
        pg.group_name, group_size, str(reduce_op).lower(), int(op_int),
        *flat)
    out_list = []
    for out, t in zip(outputs, inputs):
        shard_shape = list(t.shape)
        shard_shape[0] = shard_shape[0] // group_size
        out_list.append(out.view(shard_shape))
    return out_list


def all_gather_into_tensor_coalesced(self_tensor_list, group=None, tag=""):
    """Deprecated alias of :func:`all_gather_single_coalesced` (torch parity)."""
    warnings.warn(
        "`tensorplay.distributed._functional_collectives."
        "all_gather_into_tensor_coalesced` is deprecated. Please use "
        "`all_gather_single_coalesced` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return all_gather_single_coalesced(self_tensor_list, group, tag)


def reduce_scatter_tensor_coalesced(inputs, reduce_op, scatter_dim, group=None,
                                    tag=""):
    """Deprecated alias of :func:`reduce_scatter_single_coalesced` (torch)."""
    warnings.warn(
        "`tensorplay.distributed._functional_collectives."
        "reduce_scatter_tensor_coalesced` is deprecated. Please use "
        "`reduce_scatter_single_coalesced` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return reduce_scatter_single_coalesced(inputs, reduce_op, scatter_dim,
                                           group, tag)
