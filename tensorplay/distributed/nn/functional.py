from __future__ import annotations

import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.autograd.function import Function

ReduceOp = dist.ReduceOp

__all__ = [
    "broadcast",
    "gather",
    "scatter",
    "reduce",
    "reduce_scatter",
    "all_gather",
    "_all_gather_base",
    "all_to_all",
    "all_to_all_single",
    "all_reduce",
]


def _not_supported_under_compile(name: str, suggestion: str | None = None) -> None:
    message = f"tensorplay.distributed.nn.functional.{name} is not available during graph compilation"
    if suggestion:
        message += f"; use {suggestion}"
    raise RuntimeError(message)


def broadcast(tensor: tp.Tensor, src: int, group=None):
    return _Broadcast.apply(src, group, tensor)


def gather(tensor: tp.Tensor, dst: int = 0, group=None):
    return _Gather.apply(dst, group, tensor)


def scatter(tensors, src: int = 0, group=None):
    if tensors is None:
        raise ValueError("scatter requires a tensor sequence")
    return _Scatter.apply(src, group, *tuple(tensors))


def reduce(tensor: tp.Tensor, dst: int, op: int = ReduceOp.SUM, group=None):
    return _Reduce.apply(dst, op, group, tensor)


def reduce_scatter(output, input_list, op: int = ReduceOp.SUM, group=None):
    return _Reduce_Scatter.apply(op, group, output, *tuple(input_list))


def all_gather(tensor: tp.Tensor, group=None):
    return _AllGather.apply(group, tensor)


def _all_gather_base(output_tensor, input_tensor, group=None):
    return _AllGatherBase.apply(output_tensor, input_tensor, group)


def all_to_all(output_tensor_list, input_tensor_list, group=None):
    return _AlltoAll.apply(group, output_tensor_list, *tuple(input_tensor_list))


def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
):
    return _AlltoAllSingle.apply(
        group, output, output_split_sizes, input_split_sizes, input
    )


def all_reduce(tensor: tp.Tensor, op: int = ReduceOp.SUM, group=None):
    return _AllReduce.apply(op, group, tensor)


class _Broadcast(Function):
    @staticmethod
    def forward(ctx, src, group, tensor):
        ctx.src = src
        ctx.group = group
        ctx.global_rank = dist.get_rank(group)
        result = tensor.clone()
        dist.broadcast(result, src, group=group)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad = _Reduce.apply(ctx.src, ReduceOp.SUM, ctx.group, grad_output)
        if ctx.src != ctx.global_rank:
            grad.zero_()
        return None, None, grad


class _Gather(Function):
    @staticmethod
    def forward(ctx, dst, group, tensor):
        ctx.dst = dst
        ctx.group = group
        outputs = [
            tp.zeros_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]
        dist.gather(
            tensor.contiguous(),
            outputs if dist.get_rank(group=group) == dst else None,
            dst,
            group=group,
        )
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, _Scatter.apply(ctx.dst, ctx.group, *grad_outputs)


class _Scatter(Function):
    @staticmethod
    def forward(ctx, src, group, *tensors):
        if not tensors:
            raise ValueError("scatter requires at least one tensor")
        ctx.src = src
        ctx.group = group
        first = tensors[0]
        if any(tensor.shape != first.shape for tensor in tensors[1:]):
            raise ValueError("scatter tensors must have equal shapes")
        output = tp.zeros_like(first)
        dist.scatter(
            output,
            list(tensors) if dist.get_rank(group=group) == src else None,
            src,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, *_Gather.apply(ctx.src, ctx.group, grad_output)


class _Reduce(Function):
    @staticmethod
    def forward(ctx, src, op, group, tensor):
        ctx.src = src
        ctx.op = op
        ctx.group = group
        result = tensor.clone()
        dist.reduce(result, src, op=op, group=group)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, _Broadcast.apply(ctx.src, ctx.group, grad_output)


class _Reduce_Scatter(Function):
    @staticmethod
    def forward(ctx, op, group, output, *input_tensors):
        ctx.op = op
        ctx.group = group
        dist.reduce_scatter(
            output,
            [tensor.contiguous() for tensor in input_tensors],
            op=op,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, *_AllGather.apply(ctx.group, grad_output)


class _AllGather(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        ctx.group = group
        outputs = [
            tp.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(outputs, tensor.contiguous(), group=group)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        rank = dist.get_rank(group=ctx.group)
        if len(grad_outputs) == 1:
            return None, grad_outputs[0]
        result = grad_outputs[rank].clone()
        for index, value in enumerate(grad_outputs):
            if index != rank:
                result.add_(value)
        return None, result


class _AllGatherBase(Function):
    @staticmethod
    def forward(ctx, output_tensor, input_tensor, group):
        ctx.group = group
        operation = getattr(dist, "_allgather_base", None)
        if operation is None:
            raise RuntimeError("base all-gather is not available")
        operation(output_tensor, input_tensor.contiguous(), group=group)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        del ctx
        raise RuntimeError("base all-gather backward is not available")


class _AlltoAll(Function):
    @staticmethod
    def forward(ctx, group, output_tensor_list, *tensors):
        ctx.group = group
        ctx.input_sizes = [tensor.shape for tensor in tensors]
        dist.all_to_all(
            output_tensor_list,
            [tensor.contiguous() for tensor in tensors],
            group=group,
        )
        return tuple(output_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        outputs = [
            tp.empty(size, dtype=grad_outputs[0].dtype, device=grad_outputs[0].device)
            for size in ctx.input_sizes
        ]
        return (None, None, *_AlltoAll.apply(ctx.group, outputs, *grad_outputs))


class _AlltoAllSingle(Function):
    @staticmethod
    def forward(ctx, group, output, output_split_sizes, input_split_sizes, input):
        ctx.group = group
        ctx.input_size = input.shape
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensor = tp.empty_like(grad_output)
        return (
            None,
            None,
            None,
            None,
            _AlltoAllSingle.apply(
                ctx.group,
                tensor,
                ctx.output_split_sizes,
                ctx.input_split_sizes,
                grad_output.contiguous(),
            ),
        )


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.op = op
        ctx.group = group
        result = tensor.clone()
        dist.all_reduce(result, op=op, group=group)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output)
