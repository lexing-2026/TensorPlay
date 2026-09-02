"""Functional distributed collectives and their autograd rules."""

from __future__ import annotations

import contextlib
import threading

import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.autograd.function import Function
from tensorplay.overrides import _disable_tensorplay_function
from tensorplay.distributed import distributed_core as _core
from tensorplay.distributed.distributed_core import (
    _get_default_group as _core_default_group,
    _resolve_group as _core_resolve_group,
)


__all__ = [
    "wait_tensor",
    "wait_tensors",
    "broadcast",
    "all_reduce",
    "all_gather_single",
    "all_gather_single_autograd",
    "reduce_scatter_single",
    "reduce_scatter_single_autograd",
    "all_to_all_single",
    "all_to_all_single_autograd",
    "permute_tensor",
    "all_gather_tensor",
    "all_gather_tensor_autograd",
    "reduce_scatter_tensor",
    "reduce_scatter_tensor_autograd",
    "all_reduce_coalesced",
    "all_gather_single_coalesced",
    "reduce_scatter_single_coalesced",
    "all_gather_into_tensor_coalesced",
    "reduce_scatter_tensor_coalesced",
    "AsyncCollectiveTensor",
    "allow_inflight_collective_as_graph_input_ctx",
    "all_gather_tensor_inplace",
    "reduce_scatter_tensor_inplace",
    "all_reduce_inplace",
    "all_to_all_inplace",
    "all_gather_inplace",
    "reduce_scatter_inplace",
    "isend_inplace",
    "irecv_inplace",
    "batch_p2p_ops_inplace",
]


RANK_TYPES = "int | ProcessGroup | str"

_REDUCE_OPS = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "product": dist.ReduceOp.PRODUCT,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}

_VIEW_OPS = frozenset({
    "as_strided",
    "expand",
    "expand_as",
    "flatten",
    "movedim",
    "narrow",
    "permute",
    "reshape",
    "select",
    "slice",
    "squeeze",
    "t",
    "transpose",
    "unsqueeze",
    "view",
})


def _op_name(func):
    name = getattr(func, "__name__", None)
    if not isinstance(name, str):
        name = getattr(func, "name", None)
    return name.rsplit(".", 1)[-1] if isinstance(name, str) else ""


def _walk_async(value):
    if isinstance(value, AsyncCollectiveTensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_async(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_async(item)


def _map_tensor_result(value, callback):
    if isinstance(value, tp.Tensor):
        return callback(value)
    if isinstance(value, tuple):
        return tuple(_map_tensor_result(item, callback) for item in value)
    if isinstance(value, list):
        return [_map_tensor_result(item, callback) for item in value]
    return value


def _attach_async(tensor, work):
    if work is None or not isinstance(tensor, tp.Tensor):
        return tensor
    if type(tensor) is not tp.Tensor:
        if isinstance(tensor, AsyncCollectiveTensor):
            return tensor
        raise TypeError(
            "functional collectives require a base Tensor output"
        )
    tensor.__class__ = AsyncCollectiveTensor
    tensor._tp_collective_work = work
    tensor._tp_collective_completed = False
    tensor.completed = False
    return tensor


class AsyncCollectiveTensor(tp.Tensor):
    """Tensor storage with a pending collective completion handle."""

    def __new__(cls, elem, work=None):
        if not isinstance(elem, tp.Tensor):
            raise TypeError("elem must be a Tensor")
        if isinstance(elem, AsyncCollectiveTensor):
            return elem
        return _attach_async(elem, work)

    def __init__(self, elem, work=None):
        del elem, work

    def trigger_wait(self, timeout=None):
        if not getattr(self, "_tp_collective_completed", True):
            work = getattr(self, "_tp_collective_work", None)
            if work is not None:
                if work.wait(timeout) is False:
                    raise TimeoutError("collective wait timed out")
            self._tp_collective_completed = True
            self.completed = True
            self._tp_collective_work = None
        self.__class__ = tp.Tensor
        return self

    def wait(self):
        return self.trigger_wait()

    def _get_acs_underlying_tensor(self):
        return self

    def view(self, *shape):
        work = getattr(self, "_tp_collective_work", None)
        with _disable_tensorplay_function():
            result = tp.Tensor.view(self, *shape)
        if work is None or getattr(self, "_tp_collective_completed", True):
            return result
        return _attach_async(result, work)

    def tolist(self):
        return self.trigger_wait().tolist()

    def numpy(self):
        return self.trigger_wait().numpy()

    def __repr__(self):
        return repr(self.trigger_wait())

    def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        kwargs = kwargs or {}
        values = list(_walk_async(args)) + list(_walk_async(kwargs))
        if not values:
            return NotImplemented
        name = _op_name(func)
        if name in _VIEW_OPS:
            work = getattr(values[0], "_tp_collective_work", None)
            completed = all(
                getattr(value, "_tp_collective_completed", True)
                for value in values
            )
            with _disable_tensorplay_function():
                result = func(*args, **kwargs)
            if completed:
                return result
            return _map_tensor_result(
                result,
                lambda tensor: _attach_async(tensor, work),
            )
        seen = set()
        for value in values:
            if id(value) not in seen:
                seen.add(id(value))
                value.trigger_wait()
        with _disable_tensorplay_function():
            return func(*args, **kwargs)

    def __tensor_flatten__(self):
        self.trigger_wait()
        return [], (tuple(self.shape), tuple(self.stride()), self.dtype,
                    self.device)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        del inner_tensors, outer_stride
        shape, _, dtype, device = meta
        if outer_size is not None:
            shape = outer_size
        return tp.empty(shape, dtype=dtype, device=device)


_inflight_state = threading.local()


@contextlib.contextmanager
def allow_inflight_collective_as_graph_input_ctx(value: bool = True):
    previous = getattr(_inflight_state, "allowed", False)
    _inflight_state.allowed = bool(value)
    try:
        yield
    finally:
        _inflight_state.allowed = previous


def _group_from_ranks(ranks, tag=""):
    del tag
    if not ranks:
        raise ValueError("ranks must not be empty")
    rankset = tuple(int(rank) for rank in ranks)
    if len(set(rankset)) != len(rankset):
        raise ValueError("ranks must be unique")
    if not dist.is_initialized():
        raise RuntimeError("Default process group has not been initialized")
    default = _core_default_group()
    canonical = tuple(sorted(rankset))
    if tuple(sorted(default.ranks)) == canonical:
        return default
    for candidate in _core._groups.values():
        if tuple(sorted(candidate.ranks)) == canonical:
            return candidate
    result = dist.new_group(ranks=list(canonical))
    if result is dist.GroupMember.NON_GROUP_MEMBER:
        raise ValueError("current rank is not in the requested process group")
    return result


def _resolve_group(group, tag=""):
    if group is None:
        return _core_default_group()
    if isinstance(group, dist.ProcessGroup):
        return group
    if isinstance(group, str):
        return _core_resolve_group(group)
    try:
        from tensorplay.distributed.device_mesh import DeviceMesh
    except ImportError:
        DeviceMesh = ()
    if DeviceMesh and isinstance(group, DeviceMesh):
        if group.ndim() != 1:
            raise AssertionError(
                "only one-dimensional meshes can be used as a group"
            )
        return group.get_group()
    if isinstance(group, tuple) and len(group) == 2:
        mesh, dim = group
        if DeviceMesh and isinstance(mesh, DeviceMesh) and isinstance(dim, int):
            return mesh.get_group(dim)
    if isinstance(group, list):
        if group and isinstance(group[0], (list, tuple)):
            rows = [list(row) for row in group]
            widths = {len(row) for row in rows}
            if len(widths) != 1 or not rows[0]:
                raise ValueError("nested rank groups must be rectangular")
            current = dist.get_rank()
            rows = [row for row in rows if current in row]
            if len(rows) != 1:
                raise ValueError("current rank must occur in exactly one rank group")
            return _group_from_ranks(rows[0], tag)
        return _group_from_ranks(group, tag)
    raise ValueError(f"Unsupported group type: {type(group)}")


def _normalize_reduce_op(value):
    if isinstance(value, str):
        name = value.lower()
        if name in _REDUCE_OPS:
            return name, _REDUCE_OPS[name]
    if isinstance(value, int):
        for name, op in _REDUCE_OPS.items():
            if int(op) == value:
                return name, op
    operation = getattr(value, "op", None)
    if callable(operation):
        return _normalize_reduce_op(operation())
    raise ValueError(f"Unsupported reduction operation: {value!r}")


def _work_tensor(tensor, work):
    return _attach_async(tensor, work)


def _work_tensors(tensors, work):
    return [_work_tensor(tensor, work) for tensor in tensors]


def _gather_views(output, input_tensor, group_size):
    shape = tuple(input_tensor.shape)
    if not shape:
        return [output.narrow(0, rank, 1).reshape(())
                for rank in range(group_size)]
    chunk = int(shape[0])
    return [
        output.narrow(0, rank * chunk, chunk).reshape(shape)
        for rank in range(group_size)
    ]


def _gather_output_shape(input_tensor, group_size):
    return _gather_output_shape_from_shape(tuple(input_tensor.shape), group_size)


def _gather_output_shape_from_shape(input_shape, group_size):
    shape = list(input_shape)
    if shape:
        shape[0] *= group_size
    else:
        shape = [group_size]
    return shape


def _split_gather_output(output, input_shape, group_size):
    if not input_shape:
        return [output.narrow(0, rank, 1).reshape(())
                for rank in range(group_size)]
    chunk = int(input_shape[0])
    return [
        output.narrow(0, rank * chunk, chunk).reshape(input_shape)
        for rank in range(group_size)
    ]


def wait_tensor(tensor, timeout=None):
    """Wait until the collective that produced ``tensor`` completes."""
    if isinstance(tensor, AsyncCollectiveTensor):
        return tensor.trigger_wait(timeout)
    work = getattr(tensor, "_tp_collective_work", None)
    if work is not None:
        if work.wait(timeout) is False:
            raise TimeoutError("collective wait timed out")
        tensor._tp_collective_work = None
    return tensor


def wait_tensors(tensors, timeout=None):
    """Wait for a collection of tensors and return it as a list."""
    if not tensors:
        raise ValueError("wait_tensors requires at least one tensor")
    return [wait_tensor(tensor, timeout) for tensor in tensors]


class _CollectiveFunctionBase(Function):
    @staticmethod
    def _group_size(group):
        return max(1, group.size())


class AllReduceWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, reduce_op, tensor_input, op):
        ctx.group_name = group_name
        ctx.reduce_op = reduce_op
        output = tensor_input.detach().clone()
        pg = _resolve_group(group_name)
        work = dist.all_reduce(output, op=op, group=pg, async_op=True)
        if reduce_op in ("min", "max"):
            ctx.save_for_backward(tensor_input, output)
        return _work_tensor(output, work)

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.clone()
        pg = _resolve_group(ctx.group_name)
        if ctx.reduce_op not in ("sum", "avg", "min", "max"):
            raise RuntimeError(
                "all_reduce backward supports sum, avg, min, and max"
            )
        dist.all_reduce(grad_out, op=dist.ReduceOp.SUM, group=pg)
        if ctx.reduce_op == "avg":
            grad_out.div_(pg.size())
        elif ctx.reduce_op in ("min", "max"):
            input_tensor, output = ctx.saved_tensors
            mask = input_tensor == output
            grad_out = tp.where(mask, grad_out, tp.zeros_like(grad_out))
        return None, None, grad_out, None


def all_reduce(tensor_input, reduce_op="sum", group=None, *, op=None):
    """All-reduce the input across the entire group with autograd support."""
    if op is not None:
        reduce_op = op
    reduce_op, op_int = _normalize_reduce_op(reduce_op)
    pg = _resolve_group(group)
    return AllReduceWithAutograd.apply(
        pg.group_name, reduce_op, tensor_input, op_int
    )


class BroadcastWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, src_rank, group_name, tensor_input):
        ctx.src = int(src_rank)
        ctx.group_name = group_name
        pg = _resolve_group(group_name)
        output = tensor_input.detach().clone()
        src_global = dist.get_global_rank(pg, src_rank)
        work = dist.broadcast(output, src=src_global, group=pg, async_op=True)
        return _work_tensor(output, work)

    @staticmethod
    def backward(ctx, grad_output):
        pg = _resolve_group(ctx.group_name)
        grad_input = grad_output.clone()
        src_global = dist.get_global_rank(pg, ctx.src)
        dist.reduce(grad_input, dst=src_global, op=dist.ReduceOp.SUM, group=pg)
        if dist.get_rank(group=pg) != ctx.src:
            grad_input.zero_()
        return None, None, grad_input


def broadcast(tensor_input, src, group=None):
    """Broadcast the tensor from a rank to the whole group."""
    pg = _resolve_group(group)
    return BroadcastWithAutograd.apply(src, pg.group_name, tensor_input)


class AllGatherTensorWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, group_size, rank, tensor_input):
        ctx.group_name = group_name
        ctx.rank = rank
        ctx.input_shape = tuple(tensor_input.shape)
        ctx.group_size = group_size
        pg = _resolve_group(group_name)
        out_shape = list(ctx.input_shape)
        if out_shape:
            out_shape[0] *= group_size
        else:
            out_shape = [group_size]
        out = tp.empty(out_shape, dtype=tensor_input.dtype,
                       device=tensor_input.device)
        send_t = tensor_input.contiguous() if hasattr(tensor_input, "contiguous") \
            else tensor_input
        work = dist.all_gather_single(
            out, send_t, group=pg, async_op=True
        )
        return _work_tensor(out, work)

    @staticmethod
    def backward(ctx, grad_output):
        pg = _resolve_group(ctx.group_name)
        grad_in = tp.empty(
            ctx.input_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.reduce_scatter_single(
            grad_in,
            grad_output.contiguous(),
            op=dist.ReduceOp.SUM,
            group=pg,
        )
        return None, None, None, grad_in


def all_gather_single(tensor_input, gather_dim=0, group=None, tag=""):
    """All-gather a tensor along ``gather_dim`` with autograd support."""
    del tag
    pg = _resolve_group(group)
    size = pg.size()
    if tensor_input.ndim == 0:
        if gather_dim not in (0, -1):
            raise IndexError("gather_dim is out of range for a scalar")
        gather_dim = 0
    else:
        gather_dim = gather_dim % tensor_input.ndim
    out = AllGatherTensorWithAutograd.apply(
        pg.group_name, size, pg.rank(), tensor_input
    )
    if tensor_input.ndim == 0:
        return out
    if gather_dim == 0:
        return out
    grouped = out.reshape([size] + list(tensor_input.shape))
    return grouped.movedim(0, gather_dim).reshape(
        [d * size if i == gather_dim else d
         for i, d in enumerate(tensor_input.shape)]
    )


all_gather_single_autograd = all_gather_single


class ReduceScatterTensorWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, group_size, rank, reduce_op, tensor_input):
        ctx.group_name = group_name
        ctx.rank = rank
        ctx.group_size = group_size
        ctx.reduce_op = int(reduce_op)
        pg = _resolve_group(group_name)
        if tensor_input.numel() % group_size:
            raise ValueError("input elements must divide evenly by group size")
        shard = tensor_input.numel() // group_size
        out = tp.empty(shard, dtype=tensor_input.dtype,
                       device=tensor_input.device)
        send_t = tensor_input.contiguous() if hasattr(tensor_input, "contiguous") \
            else tensor_input
        work = dist.reduce_scatter_single(
            out, send_t, op=int(reduce_op), group=pg, async_op=True
        )
        return _work_tensor(out, work)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reduce_op != int(dist.ReduceOp.SUM):
            raise RuntimeError(
                "reduce_scatter_single backward only supports sum reduction"
            )
        pg = _resolve_group(ctx.group_name)
        n = grad_output.numel()
        inp = tp.empty(ctx.group_size * n, dtype=grad_output.dtype,
                       device=grad_output.device)
        dist.all_gather_single(inp, grad_output.contiguous(), group=pg)
        return None, None, None, None, inp


def reduce_scatter_single(tensor_input, reduce_op="sum", scatter_dim=0,
                          group=None, tag=""):
    """Reduce-scatter a tensor along ``scatter_dim`` with autograd support."""
    del tag
    pg = _resolve_group(group)
    reduce_op, op_int = _normalize_reduce_op(reduce_op)
    size = pg.size()
    shape = list(tensor_input.shape)
    if not shape:
        raise ValueError("reduce_scatter_single requires a non-scalar input")
    scatter_dim = scatter_dim % len(shape)
    assert shape[scatter_dim] % size == 0, (
        f"collective reduce_scatter_tensor requires the dim {scatter_dim} "
        f"to be equally split across the given group ({size}), but the "
        f"dim size {shape[scatter_dim]} is not divisible by {size}.")
    flat = tensor_input.movedim(scatter_dim, 0).reshape(-1)
    out = ReduceScatterTensorWithAutograd.apply(
        pg.group_name, size, pg.rank(), op_int, flat
    )
    shard_shape = [d // size if i == 0 else d for i, d in enumerate(
        [shape[scatter_dim]] + shape[:scatter_dim] + shape[scatter_dim + 1:])]
    return out.view(shard_shape).movedim(0, scatter_dim)


reduce_scatter_single_autograd = reduce_scatter_single


def all_gather_tensor(tensor_input, gather_dim=0, group=None, tag=""):
    return all_gather_single(tensor_input, gather_dim, group, tag)


all_gather_tensor_autograd = all_gather_single


def reduce_scatter_tensor(tensor_input, reduce_op="sum", scatter_dim=0,
                          group=None, tag=""):
    return reduce_scatter_single(tensor_input, reduce_op, scatter_dim, group,
                                 tag)


reduce_scatter_tensor_autograd = reduce_scatter_single


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _resolve_coalesced_group(group):
    return _resolve_group(group)


class AllReduceCoalescedWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, reduce_op, op_int, *tensors):
        ctx.group_name = group_name
        ctx.reduce_op = str(reduce_op).lower()
        pg = _resolve_coalesced_group(group_name)
        outputs = [t.detach().clone().contiguous() for t in tensors]
        work = dist.all_reduce_coalesced(
            outputs, op=int(op_int), group=pg, async_op=True
        )
        ctx._in_metas = [(tuple(o.shape), o.dtype, o.device) for o in outputs]
        return tuple(_work_tensors(outputs, work))

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.reduce_op not in ("sum", "avg"):
            raise RuntimeError(
                "all_reduce_coalesced backward supports sum and avg, "
                f"got '{ctx.reduce_op}'")
        pg = _resolve_coalesced_group(ctx.group_name)
        grads = []
        for grad, meta in zip(grad_outputs, ctx._in_metas):
            shape, dtype, device = meta
            grads.append(
                grad.contiguous()
                if grad is not None
                else tp.zeros(shape, dtype=dtype, device=device)
            )
        dist.all_reduce_coalesced(
            grads, op=dist.ReduceOp.SUM, group=pg
        )
        if ctx.reduce_op == "avg":
            for grad in grads:
                grad.div_(pg.size())
        outs = grads
        return (None, None, None, *outs)


def all_reduce_coalesced(self_tensor_list, reduce_op="sum", group=None, tag=""):
    """All-reduce a list of tensors in one coalesced group launch."""
    del tag
    pg = _resolve_coalesced_group(group)
    reduce_op, op_int = _normalize_reduce_op(reduce_op)
    tensors = tuple(self_tensor_list)
    if not tensors:
        return []
    outputs = AllReduceCoalescedWithAutograd.apply(
        pg.group_name, reduce_op, int(op_int), *tensors)
    return list(outputs)


class AllGatherSingleCoalescedWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, group_size, *tensors):
        ctx.group_name = group_name
        ctx.group_size = group_size
        pg = _resolve_coalesced_group(group_name)
        outputs = [
            tp.empty(_gather_output_shape(t, group_size), dtype=t.dtype,
                     device=t.device)
            for t in tensors
        ]
        work = dist.all_gather_single_coalesced(
            outputs,
            [tensor.contiguous() for tensor in tensors],
            group=pg,
            async_op=True,
        )
        ctx._in_metas = [(tuple(t.shape), t.dtype, t.device)
                         for t in tensors]
        return tuple(_work_tensors(outputs, work))

    @staticmethod
    def backward(ctx, *grad_outputs):
        pg = _resolve_coalesced_group(ctx.group_name)
        grads = []
        for grad, meta in zip(grad_outputs, ctx._in_metas):
            input_shape, dtype, device = meta
            shape = tuple(_gather_output_shape_from_shape(
                input_shape, ctx.group_size
            ))
            full = (grad.contiguous() if grad is not None else
                    tp.zeros(shape, dtype=dtype, device=device))
            grads.append(full)
        outputs = [
            tp.empty(input_shape, dtype=dtype, device=device)
            for input_shape, dtype, device in ctx._in_metas
        ]
        input_lists = [
            _split_gather_output(full, input_shape, ctx.group_size)
            for full, (input_shape, _, _) in zip(grads, ctx._in_metas)
        ]
        dist.reduce_scatter_coalesced(
            outputs, input_lists, op=dist.ReduceOp.SUM, group=pg
        )
        return (None, None, *outputs)


def all_gather_single_coalesced(self_tensor_list, group=None, tag=""):
    """All-gather a list of tensors (gather_dim=0) in one coalesced launch."""
    del tag
    pg = _resolve_coalesced_group(group)
    tensors = tuple(self_tensor_list)
    if not tensors:
        return []
    outputs = AllGatherSingleCoalescedWithAutograd.apply(
        pg.group_name, pg.size(), *tensors)
    return list(outputs)


class ReduceScatterSingleCoalescedWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, group_size, reduce_op, op_int, *tensors):
        ctx.group_name = group_name
        ctx.group_size = group_size
        ctx.reduce_op = str(reduce_op).lower()
        pg = _resolve_coalesced_group(group_name)
        outputs = []
        for tensor in tensors:
            if tensor.numel() % group_size:
                raise ValueError(
                    "input elements must divide evenly by group size"
                )
            chunk = tensor.numel() // group_size
            outputs.append(tp.empty(chunk, dtype=tensor.dtype,
                                    device=tensor.device))
        work = dist.reduce_scatter_single_coalesced(
            outputs,
            [tensor.contiguous() for tensor in tensors],
            op=int(op_int),
            group=pg,
            async_op=True,
        )
        ctx._in_metas = [(tuple(o.shape), o.dtype, o.device) for o in outputs]
        return tuple(_work_tensors(outputs, work))

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.reduce_op not in ("sum", "avg"):
            raise RuntimeError(
                "reduce_scatter_tensor_coalesced backward supports sum and "
                f"avg, got '{ctx.reduce_op}'")
        pg = _resolve_coalesced_group(ctx.group_name)
        grads = []
        output_lists = []
        for grad, meta in zip(grad_outputs, ctx._in_metas):
            shape, dtype, device = meta
            value = (grad.contiguous() if grad is not None else
                     tp.zeros(shape, dtype=dtype, device=device))
            grads.append(value)
            output_lists.append([
                tp.empty_like(value) for _ in range(ctx.group_size)
            ])
        dist.all_gather_coalesced(output_lists, grads, group=pg)
        outputs = [
            tp.cat(output_list).reshape(-1)
            for output_list in output_lists
        ]
        if ctx.reduce_op == "avg":
            for output in outputs:
                output.div_(ctx.group_size)
        return (None, None, None, None, *outputs)


def reduce_scatter_single_coalesced(inputs, reduce_op, scatter_dim, group=None,
                                    tag=""):
    """Reduce-scatter a list of tensors in one coalesced group launch."""
    del tag
    pg = _resolve_coalesced_group(group)
    reduce_op, op_int = _normalize_reduce_op(reduce_op)
    group_size = pg.size()
    inputs = list(inputs)
    scatter_dim = list(scatter_dim)
    if len(scatter_dim) != len(inputs):
        raise AssertionError(
            f"Length of scatter_dim ({len(scatter_dim)}) must equal length "
            f"of inputs ({len(inputs)})")
    for idx, (dim, tensor) in enumerate(zip(scatter_dim, inputs)):
        if not tensor.shape:
            raise ValueError("reduce_scatter_single requires non-scalar inputs")
        dim = dim % len(tensor.shape)
        scatter_dim[idx] = dim
        size_dim = tensor.shape[dim]
        if size_dim % group_size != 0:
            raise AssertionError(
                f"input dimension {dim} ({size_dim} must be a multiple of "
                f"group_size {group_size} for tensor at index {idx}")
        if dim != 0:
            inputs[idx] = tp.cat(tp.chunk(tensor, group_size, dim=dim))
    flat = [t.reshape(-1) for t in inputs]
    outputs = ReduceScatterSingleCoalescedWithAutograd.apply(
        pg.group_name, group_size, reduce_op, int(op_int),
        *flat)
    out_list = []
    for out, t in zip(outputs, inputs):
        shard_shape = list(t.shape)
        shard_shape[0] = shard_shape[0] // group_size
        out_list.append(out.view(shard_shape))
    return out_list


def all_gather_into_tensor_coalesced(self_tensor_list, group=None, tag=""):
    return all_gather_single_coalesced(self_tensor_list, group, tag)


def reduce_scatter_tensor_coalesced(inputs, reduce_op, scatter_dim, group=None,
                                    tag=""):
    return reduce_scatter_single_coalesced(
        inputs, reduce_op, scatter_dim, group, tag
    )


def _split_sizes(value, group_size, name):
    if value is None:
        return None
    result = [int(item) for item in value]
    if len(result) != group_size:
        raise ValueError(
            f"{name} must have one entry per rank in the process group"
        )
    if any(item < 0 for item in result):
        raise ValueError(f"{name} must contain non-negative values")
    return result


def _all_to_all_single_native(
    tensor_input, output_split_sizes, input_split_sizes, group
):
    pg = _resolve_group(group)
    if tensor_input.ndim == 0:
        raise ValueError("all_to_all_single requires a non-scalar input")
    size = pg.size()
    output_split_sizes = _split_sizes(output_split_sizes, size,
                                      "output_split_sizes")
    input_split_sizes = _split_sizes(input_split_sizes, size,
                                     "input_split_sizes")
    if output_split_sizes is None or input_split_sizes is None:
        if output_split_sizes is not None or input_split_sizes is not None:
            raise AssertionError(
                "output_split_sizes and input_split_sizes must either be "
                "specified together or both set to None"
            )
        default = int(tensor_input.shape[0])
        if default % size:
            raise ValueError(
                "the leading dimension must divide evenly by group size"
            )
        input_split_sizes = [default // size] * size
        output_split_sizes = list(input_split_sizes)
    if sum(input_split_sizes) != int(tensor_input.shape[0]):
        raise ValueError(
            "input_split_sizes must sum to the input leading dimension"
        )
    row_width = 1
    for dim in tensor_input.shape[1:]:
        row_width *= int(dim)
    output_shape = list(tensor_input.shape)
    output_shape[0] = sum(output_split_sizes)
    output = tp.empty(output_shape, dtype=tensor_input.dtype,
                      device=tensor_input.device)
    work = dist.all_to_all_single(
        output.reshape(-1),
        tensor_input.reshape(-1),
        output_split_sizes=[item * row_width for item in output_split_sizes],
        input_split_sizes=[item * row_width for item in input_split_sizes],
        group=pg,
        async_op=True,
    )
    return _work_tensor(output, work)


class AllToAllSingleWithAutograd(_CollectiveFunctionBase):
    @staticmethod
    def forward(ctx, group_name, output_split_sizes, input_split_sizes,
                tensor_input):
        ctx.group_name = group_name
        ctx.output_split_sizes = tuple(output_split_sizes)
        ctx.input_split_sizes = tuple(input_split_sizes)
        return _all_to_all_single_native(
            tensor_input,
            list(output_split_sizes),
            list(input_split_sizes),
            _resolve_group(group_name),
        )

    @staticmethod
    def backward(ctx, grad_output):
        result = _all_to_all_single_native(
            grad_output,
            list(ctx.input_split_sizes),
            list(ctx.output_split_sizes),
            _resolve_group(ctx.group_name),
        )
        non_tensor_inputs = (
            1 + len(ctx.output_split_sizes) + len(ctx.input_split_sizes)
        )
        return (None,) * non_tensor_inputs + (wait_tensor(result),)


def all_to_all_single(tensor_input, output_split_sizes=None,
                      input_split_sizes=None, group=None, tag=""):
    del tag
    pg = _resolve_group(group)
    size = pg.size()
    if output_split_sizes is None and input_split_sizes is None:
        if tensor_input.ndim == 0:
            raise ValueError("all_to_all_single requires a non-scalar input")
        default = int(tensor_input.shape[0])
        if default % size:
            raise ValueError(
                "the leading dimension must divide evenly by group size"
            )
        input_split_sizes = [default // size] * size
        output_split_sizes = list(input_split_sizes)
    output_split_sizes = _split_sizes(output_split_sizes, size,
                                      "output_split_sizes")
    input_split_sizes = _split_sizes(input_split_sizes, size,
                                     "input_split_sizes")
    if output_split_sizes is None or input_split_sizes is None:
        raise AssertionError(
            "output_split_sizes and input_split_sizes must either be "
            "specified together or both set to None"
        )
    return AllToAllSingleWithAutograd.apply(
        pg.group_name,
        tuple(output_split_sizes),
        tuple(input_split_sizes),
        tensor_input,
    )


all_to_all_single_autograd = all_to_all_single


def permute_tensor(tensor_input, src_dst, group=None, tag=""):
    del tag
    pg = _resolve_group(group)
    if len(src_dst) != pg.size():
        raise ValueError("src_dst must contain one destination per group rank")
    if sorted(int(value) for value in src_dst) != list(range(pg.size())):
        raise ValueError("src_dst must be a permutation of group ranks")
    rank = pg.rank()
    input_split_sizes = [0] * pg.size()
    output_split_sizes = [0] * pg.size()
    input_split_sizes[int(src_dst[rank])] = tensor_input.numel()
    for source, destination in enumerate(src_dst):
        if int(destination) == rank:
            output_split_sizes[source] = tensor_input.numel()
    return all_to_all_single(
        tensor_input.reshape(-1), output_split_sizes, input_split_sizes, pg
    )


def _require_sync(async_op):
    if async_op:
        raise AssertionError(
            "in-place remapping cannot return an asynchronous collective"
        )


def all_gather_tensor_inplace(output_tensor, input_tensor, group=None,
                              async_op=False, tag="", gather_dim=0):
    _require_sync(async_op)
    result = wait_tensor(all_gather_single(input_tensor, gather_dim, group, tag))
    return output_tensor.copy_(result)


def reduce_scatter_tensor_inplace(output, input, op="sum", group=None,
                                  async_op=False, scatter_dim=0, tag=""):
    _require_sync(async_op)
    result = wait_tensor(
        reduce_scatter_single(input, op, scatter_dim, group, tag)
    )
    return output.copy_(result)


def all_reduce_inplace(tensor_input, op="sum", group=None, async_op=False,
                       tag=""):
    _require_sync(async_op)
    result = wait_tensor(all_reduce(tensor_input, op, group=group))
    return tensor_input.copy_(result)


def all_to_all_inplace(output, input, output_split_sizes=None,
                       input_split_sizes=None, group=None, async_op=False,
                       tag=""):
    _require_sync(async_op)
    result = wait_tensor(all_to_all_single(
        input, output_split_sizes, input_split_sizes, group, tag
    ))
    return output.copy_(result)


def all_gather_inplace(tensor_list, tensor_input, group=None, async_op=False,
                       tag=""):
    _require_sync(async_op)
    if len(tensor_list) != _resolve_group(group).size():
        raise ValueError("all_gather output list must match group size")
    if tensor_input.ndim != 0:
        if any(target.shape != tensor_input.shape for target in tensor_list):
            raise ValueError("all_gather output tensors must have equal shapes")
    elif any(target.ndim != 0 for target in tensor_list):
        raise ValueError("scalar all_gather outputs must be scalar tensors")
    result = wait_tensor(all_gather_single(tensor_input, 0, group, tag))
    for rank, target in enumerate(tensor_list):
        view = (result.narrow(0, rank, 1).reshape(())
                if tensor_input.ndim == 0 else
                result.narrow(0, rank * tensor_input.shape[0],
                               tensor_input.shape[0]))
        target.copy_(view)
    return tensor_list


def reduce_scatter_inplace(output, input_list, op="sum", group=None,
                           async_op=False, tag=""):
    _require_sync(async_op)
    if len(input_list) != _resolve_group(group).size():
        raise ValueError("reduce_scatter input list must match group size")
    if any(t.shape != output.shape for t in input_list):
        raise ValueError(
            "reduce_scatter requires every input tensor to match output shape"
        )
    if output.ndim == 0:
        stacked = tp.stack(list(input_list), dim=0)
        result = wait_tensor(reduce_scatter_single(
            stacked, op, 0, group, tag
        ))
        return output.copy_(result.reshape(output.shape))
    result = wait_tensor(reduce_scatter_single(
        tp.cat(list(input_list)), op, 0, group, tag
    ))
    return output.copy_(result)


def _resolve_p2p_group(group):
    return _resolve_group(group)


def _tag_to_int(tag):
    if isinstance(tag, bool):
        raise ValueError("tag must be an integer")
    if isinstance(tag, int):
        return tag
    if isinstance(tag, str) and tag.isdigit():
        return int(tag)
    raise ValueError(f"tag must be an integer, got {tag!r}")


def isend_inplace(tensor, dst=None, tag=0, group=None, group_dst=-1):
    pg = _resolve_p2p_group(group)
    if group_dst != -1:
        if dst is not None:
            raise ValueError("dst and group_dst cannot both be specified")
        dst = dist.get_global_rank(pg, int(group_dst))
    elif dst is None:
        raise ValueError("either dst or group_dst must be specified")
    return dist.isend(tensor, dst=int(dst), group=pg, tag=_tag_to_int(tag))


def irecv_inplace(tensor, src=None, tag=0, group=None, group_src=-1):
    pg = _resolve_p2p_group(group)
    if group_src != -1:
        if src is not None:
            raise ValueError("src and group_src cannot both be specified")
        src = dist.get_global_rank(pg, int(group_src))
    elif src is None:
        raise ValueError("either src or group_src must be specified")
    return dist.irecv(tensor, src=int(src), group=pg, tag=_tag_to_int(tag))


def batch_p2p_ops_inplace(op_list, peer_list, tag_list, tensors, group_name):
    if not (len(op_list) == len(peer_list) == len(tag_list) == len(tensors)):
        raise ValueError(
            "op_list, peer_list, tag_list, and tensors must have equal lengths"
        )
    pg = _resolve_p2p_group(group_name)
    operations = []
    for operation, peer, tag, tensor in zip(
        op_list, peer_list, tag_list, tensors
    ):
        if operation == "isend":
            fn = dist.isend
        elif operation == "irecv":
            fn = dist.irecv
        else:
            raise ValueError(f"Unsupported point-to-point operation: {operation!r}")
        operations.append(
            dist.P2POp(
                fn, tensor, dist.get_global_rank(pg, int(peer)), pg,
                _tag_to_int(tag)
            )
        )
    return dist.batch_isend_irecv(operations)
