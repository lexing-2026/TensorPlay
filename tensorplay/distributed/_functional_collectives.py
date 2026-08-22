# Ported from torch/distributed/_functional_collectives.py.
#
# This module provides tensor-collectives that participate in autograd
# (``*_autograd`` variants use ``tensorplay.autograd.Function``), matching
# the public surface torch's async-TP and python-reducer DDP paths rely on.
#
# The *_coalesced family in torch is backed by the C++ coalescing manager;
# tp's group semantics cover p2p batching only, so those entries are listed
# in docs/gap_analysis.md as pending native work.

import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.autograd.function import Function


__all__ = [
    "wait_tensor",
    "broadcast",
    "all_reduce",
    "all_gather_tensor",
    "all_gather_tensor_autograd",
    "reduce_scatter_tensor",
    "reduce_scatter_tensor_autograd",
]


RANK_TYPES = "int | ProcessGroup | str"


def wait_tensor(tensor, timeout=None):
    """Wait until the collective that produced ``tensor`` completes."""
    if isinstance(tensor, tp.Tensor) and hasattr(tensor, "_wait_tensor"):
        tensor._wait_tensor(timeout)
    return tensor


def _resolve_group(group):
    if group is None:
        return dist._get_default_group()
    if isinstance(group, dist.ProcessGroup):
        return group
    if isinstance(group, int):  # global rank prefix of a sub-group
        return dist._get_default_group()
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
