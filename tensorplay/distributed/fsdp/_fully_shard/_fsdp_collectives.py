"""Collective adapters used by composable sharding."""

import math
from dataclasses import dataclass
from typing import Any, Sequence

import tensorplay as tp

from ... import distributed_core as dist
from ._fsdp_api import AllGather, Comm, ReduceScatter

__all__ = [
    "AllGatherResult",
    "DefaultAllGather",
    "ProcessGroupAllocAllGather",
    "SymmMemAllGather",
    "DefaultReduceScatter",
    "ProcessGroupAllocReduceScatter",
    "SymmMemReduceScatter",
    "all_gather_copy_in_meta",
    "all_gather_copy_in_cuda",
    "split_with_sizes_copy",
    "chunk_cat",
    "foreach_all_gather",
    "foreach_all_gather_copy_out",
    "foreach_reduce",
    "foreach_reduce_scatter_copy_in",
    "_get_param_all_gather_inputs",
    "_get_all_gather_input_metadatas",
    "_get_gradient_divide_factors",
    "_div_if_needed",
]


@dataclass
class AllGatherResult:
    output: Any
    work: Any = None

    def wait(self) -> Any:
        if self.work is not None:
            self.work.wait()
        return self.output


class DefaultAllocMixin(Comm):
    def allocate(self, size: Sequence[int], *, dtype: Any, device: Any) -> Any:
        return tp.empty(tuple(size), dtype=dtype, device=device)


class ProcessGroupAllocMixin(DefaultAllocMixin):
    def __init__(self, group: Any = None) -> None:
        self.group = group


class SymmMemAllocMixin(ProcessGroupAllocMixin):
    def __init__(self, group: Any = None, backend: Any = None) -> None:
        super().__init__(group)
        self.backend = backend


class DefaultAllGather(DefaultAllocMixin, AllGather):
    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, async_op: bool = False) -> Any:
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class ProcessGroupAllocAllGather(DefaultAllGather, ProcessGroupAllocMixin):
    def __init__(self, group: Any = None) -> None:
        ProcessGroupAllocMixin.__init__(self, group)


class SymmMemAllGather(ProcessGroupAllocAllGather, SymmMemAllocMixin):
    def __init__(self, group: Any = None, backend: Any = None) -> None:
        SymmMemAllocMixin.__init__(self, group, backend)


class DefaultReduceScatter(DefaultAllocMixin, ReduceScatter):
    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, op: Any, async_op: bool = False) -> Any:
        return dist.reduce_scatter_single(
            output_tensor,
            input_tensor,
            op=op,
            group=group,
            async_op=async_op,
        )


class ProcessGroupAllocReduceScatter(DefaultReduceScatter, ProcessGroupAllocMixin):
    def __init__(self, group: Any = None) -> None:
        ProcessGroupAllocMixin.__init__(self, group)


class SymmMemReduceScatter(ProcessGroupAllocReduceScatter, SymmMemAllocMixin):
    def __init__(self, group: Any = None, backend: Any = None) -> None:
        SymmMemAllocMixin.__init__(self, group, backend)


def all_gather_copy_in_meta(all_gather_inputs: Sequence[Any], all_gather_output: Any, inp_split_sizes: Sequence[int], all_gather_input_numel: int, rank: int) -> Any:
    del all_gather_inputs, inp_split_sizes
    if int(all_gather_input_numel) < 0 or int(rank) < 0:
        raise ValueError("all-gather input size and rank must be non-negative")
    start = int(all_gather_input_numel) * int(rank)
    end = start + int(all_gather_input_numel)
    if end > int(all_gather_output.numel()):
        raise ValueError("all-gather input slice exceeds the output buffer")
    return all_gather_output[start:end], all_gather_output


def all_gather_copy_in_cuda(all_gather_inputs: Sequence[Any], all_gather_output: Any, inp_split_sizes: Sequence[int], all_gather_input_numel: int, rank: int) -> Any:
    if int(all_gather_input_numel) < 0 or int(rank) < 0:
        raise ValueError("all-gather input size and rank must be non-negative")
    sizes = [int(size) for size in inp_split_sizes]
    if len(sizes) != len(all_gather_inputs):
        raise ValueError("input split sizes must match the input tensor list")
    if any(size < 0 for size in sizes) or sum(sizes) != int(all_gather_input_numel):
        raise ValueError("input split sizes do not match the input buffer size")
    start = int(all_gather_input_numel) * int(rank)
    end = start + int(all_gather_input_numel)
    if end > int(all_gather_output.numel()):
        raise ValueError("all-gather input slice exceeds the output buffer")
    all_gather_input = all_gather_output[start:end]
    offset = 0
    for value, size in zip(all_gather_inputs, sizes):
        if int(value.numel()) != size:
            raise ValueError("input split size does not match the tensor size")
        next_offset = offset + size
        all_gather_input[offset:next_offset].copy_(value.reshape(-1))
        offset = next_offset
    return all_gather_input, all_gather_output


def split_with_sizes_copy(all_gather_output: Any, all_gather_input_split_sizes: Sequence[int], dim: int, out: Sequence[Any]) -> Sequence[Any]:
    sizes = tuple(int(size) for size in all_gather_input_split_sizes)
    if len(out) != len(sizes):
        raise ValueError("output count must match the split-size count")
    if any(size < 0 for size in sizes):
        raise ValueError("split sizes must be non-negative")
    values = all_gather_output.split(sizes, dim=dim)
    for target, value in zip(out, values):
        if int(target.numel()) != int(value.numel()):
            raise ValueError("split output has an incompatible number of elements")
        target.copy_(value)
    return out


def chunk_cat(tensors: Sequence[Any], dim: int, num_chunks: int, out: Any = None) -> Any:
    num_chunks = int(num_chunks)
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if not tensors:
        raise ValueError("tensors must not be empty")
    chunks_by_rank = [value.chunk(num_chunks, dim=dim) for value in tensors]
    chunks = [
        tp.cat(tuple(chunks_by_rank[index][rank].reshape(-1) for index in range(len(tensors))), dim=0)
        for rank in range(num_chunks)
    ]
    if out is None:
        return tuple(chunks)
    if int(out.numel()) != sum(int(value.numel()) for value in chunks):
        raise ValueError("output has an incompatible number of elements")
    if int(out.dim()) == 2 and int(out.shape[0]) == num_chunks:
        row_width = int(out.shape[1])
        if any(int(value.numel()) != row_width for value in chunks):
            raise ValueError("chunked outputs must have equal flattened sizes")
        for rank, value in enumerate(chunks):
            out[rank].copy_(value.reshape(out[rank].shape))
    else:
        out.reshape(-1).copy_(tp.cat(tuple(chunks), dim=0))
    return out


def _get_param_all_gather_inputs(fsdp_params: Sequence[Any]) -> list[Any]:
    return [param.all_gather_inputs for param in fsdp_params]


def foreach_all_gather(fsdp_params: Sequence[Any], group: Any, async_op: bool, all_gather_copy_in_stream: Any, all_gather_stream: Any, device: Any, all_gather_comm: AllGather | None = None) -> list[AllGatherResult]:
    del all_gather_copy_in_stream, all_gather_stream, device
    comm = all_gather_comm or DefaultAllGather()
    results = []
    for param in fsdp_params:
        inputs = tuple(param.all_gather_inputs)
        if not inputs:
            raise ValueError("each sharded parameter needs an all-gather input")
        local = inputs[0] if len(inputs) == 1 else tp.cat(
            tuple(value.reshape(-1) for value in inputs), dim=0
        )
        width = int(local.numel()) * dist.get_world_size(group)
        output = local.new_empty(width)
        work = comm(output, local.reshape(-1), group, async_op)
        results.append(AllGatherResult(output, work))
    return results


def foreach_all_gather_copy_out(all_gather_result: Sequence[AllGatherResult], fsdp_params: Sequence[Any], group: Any) -> None:
    del group
    for result, param in zip(all_gather_result, fsdp_params):
        gathered = result.wait()
        gathered = param._attach_local_gradient_to_all_gather(gathered)
        param._use_unsharded_tensor(gathered)


def foreach_reduce(
    fsdp_params: Sequence[Any],
    unsharded_grads: list[Any],
    reduce_scatter_group: Any,
    reduce_scatter_stream: Any,
    reduce_scatter_comm: ReduceScatter | None,
    orig_dtype: Any,
    reduce_dtype: Any,
    device: Any,
    gradient_divide_factor: float | None,
    all_reduce_group: Any,
    all_reduce_stream: Any,
    all_reduce_grads: bool,
    partial_reduce_output: Any,
    all_reduce_hook: Any,
    force_sum_reduction_for_comms: bool,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    del reduce_scatter_stream, device, all_reduce_stream
    if len(fsdp_params) != len(unsharded_grads):
        raise ValueError("parameter and gradient lists must have the same length")
    if not fsdp_params:
        raise ValueError("reduce-scatter requires at least one parameter")
    grad_dtypes = {getattr(grad, "dtype", None) for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        raise ValueError(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = next(iter(grad_dtypes))
    if reduce_dtype is None:
        reduce_dtype = grad_dtype
    world_size = (
        dist.get_world_size(reduce_scatter_group)
        if reduce_scatter_group is not None
        else 1
    )
    predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op = (
        _get_gradient_divide_factors(
            reduce_scatter_group,
            all_reduce_group,
            reduce_dtype,
            getattr(getattr(device, "type", None), "lower", lambda: str(device).lower())(),
            gradient_divide_factor,
            force_sum_reduction_for_comms,
        )
    )

    prepared_grads: list[Any] = []
    padded_sizes: list[tuple[int, ...]] = []
    for fsdp_param, grad in zip(fsdp_params, unsharded_grads):
        value = grad.to(dtype=reduce_dtype) if grad.dtype != reduce_dtype else grad
        if int(value.dim()) == 0:
            value = value.reshape(1)
        if world_size > 1:
            placement = getattr(fsdp_param, "fsdp_placement", None)
            if placement is None:
                placement = getattr(fsdp_param, "_placement", None)
            shard_dim = int(getattr(placement, "dim", 0))
            if shard_dim < 0:
                shard_dim += int(value.dim())
            if shard_dim < 0 or shard_dim >= int(value.dim()):
                raise ValueError("shard dimension is outside the gradient")
            if shard_dim != 0:
                if int(value.shape[shard_dim]) % world_size:
                    raise ValueError("gradient shard dimension must divide evenly")
                value = tp.cat(
                    tuple(value.chunk(world_size, dim=shard_dim)), dim=0
                )
            padded_dim0 = int(
                math.ceil(int(value.shape[0]) / world_size) * world_size
            )
            if padded_dim0 != int(value.shape[0]):
                padded_shape = list(value.shape)
                padded_shape[0] = padded_dim0
                padded = value.new_zeros(tuple(padded_shape))
                padded.narrow(0, 0, int(value.shape[0])).copy_(value)
                value = padded
        padded_sizes.append(tuple(int(size) for size in value.shape))
        prepared_grads.append(value)

    reduce_scatter_input_numel = sum(
        math.prod(size) for size in padded_sizes
    )
    if reduce_scatter_input_numel % world_size:
        raise RuntimeError("reduce-scatter input size must divide by world size")
    comm = reduce_scatter_comm or DefaultReduceScatter()
    reduce_scatter_input = comm.allocate(
        (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
    )
    foreach_reduce_scatter_copy_in(
        prepared_grads, reduce_scatter_input, world_size
    )
    unsharded_grads.clear()
    reduce_scatter_input = _div_if_needed(
        reduce_scatter_input, predivide_factor
    )
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    reduce_output = comm.allocate(
        (reduce_scatter_output_numel,), dtype=reduce_dtype, device=device
    )
    if world_size > 1:
        comm(
            reduce_output,
            reduce_scatter_input,
            reduce_scatter_group,
            reduce_scatter_op,
            False,
        )
    else:
        reduce_output.copy_(reduce_scatter_input)

    all_reduce_input = None
    all_reduce_event = None
    all_reduce_world_size = (
        dist.get_world_size(all_reduce_group)
        if all_reduce_group is not None
        else 1
    )
    if all_reduce_group is not None and all_reduce_world_size > 1:
        if all_reduce_grads:
            dist.all_reduce(
                reduce_output, op=all_reduce_op, group=all_reduce_group, async_op=False
            )
            all_reduce_input = reduce_output
        elif partial_reduce_output is not None:
            reduce_output = partial_reduce_output + reduce_output
        elif not all_reduce_grads:
            partial_reduce_output = reduce_output

    if all_reduce_hook is not None:
        all_reduce_hook(reduce_output)
    reduce_output = _div_if_needed(reduce_output, postdivide_factor)
    if orig_dtype is not None and reduce_output.dtype != orig_dtype:
        reduce_output = reduce_output.to(dtype=orig_dtype)

    flat_grad_offset = 0
    for padded_size, fsdp_param in zip(padded_sizes, fsdp_params):
        sharded_size = tuple(int(size) for size in fsdp_param.sharded_size)
        new_sharded_grad = tp.as_strided(
            reduce_output,
            sharded_size,
            tuple(int(stride) for stride in fsdp_param.contiguous_sharded_stride),
            storage_offset=flat_grad_offset,
        )
        if getattr(fsdp_param, "offload_to_cpu", False):
            new_sharded_grad = new_sharded_grad.to("cpu")
        old_grad = getattr(fsdp_param, "_sharded_grad", None)
        if old_grad is not None:
            new_sharded_grad = old_grad + new_sharded_grad
        fsdp_param._set_sharded_grad(new_sharded_grad)
        flat_grad_offset += math.prod(padded_size) // world_size

    return (
        reduce_scatter_input,
        None,
        None,
        None,
        all_reduce_input,
        all_reduce_event,
        partial_reduce_output,
    )


def foreach_reduce_scatter_copy_in(unsharded_grads: Sequence[Any], reduce_scatter_input: Any, world_size: int) -> Any:
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    rows = [[] for _ in range(world_size)]
    for grad in unsharded_grads:
        value = grad.reshape(1) if int(grad.dim()) == 0 else grad
        if int(value.shape[0]) % world_size:
            raise ValueError("gradient leading dimension must divide evenly")
        chunks = value.chunk(world_size, dim=0)
        for rank, chunk in enumerate(chunks):
            rows[rank].append(chunk.reshape(-1))
    if rows and any(not row for row in rows):
        raise ValueError("reduce-scatter input cannot be empty")
    packed = [tp.cat(tuple(row), dim=0) if row else reduce_scatter_input.new_empty(0) for row in rows]
    expected = int(reduce_scatter_input.numel())
    if sum(int(value.numel()) for value in packed) != expected:
        raise ValueError("reduce-scatter input size does not match the gradients")
    offset = 0
    for value in packed:
        end = offset + int(value.numel())
        reduce_scatter_input[offset:end].copy_(value)
        offset = end
    return reduce_scatter_input


def _get_all_gather_input_metadatas(param_all_gather_inputs: Sequence[Sequence[Any]]) -> tuple[list[list[Any]], list[list[int]], Any]:
    if not param_all_gather_inputs or not param_all_gather_inputs[0]:
        raise ValueError("all-gather input metadata requires at least one tensor")
    dtypes: list[list[Any]] = []
    numels: list[list[int]] = []
    dtype = param_all_gather_inputs[0][0].dtype
    for inputs in param_all_gather_inputs:
        current_dtypes: list[Any] = []
        current_numels: list[int] = []
        for value in inputs:
            current_dtypes.append(value.dtype)
            current_numels.append(int(value.numel()))
            if value.dtype != dtype:
                dtype = getattr(tp, "uint8", "uint8")
        dtypes.append(current_dtypes)
        numels.append(current_numels)
    return dtypes, numels, dtype


def _get_gradient_divide_factors(reduce_scatter_group: Any, all_reduce_group: Any, reduce_dtype: Any, device_type: str, factor: float | None, force_sum_reduction_for_comms: bool) -> tuple[float | None, float | None, Any, Any]:
    if device_type == "mtia":
        force_sum_reduction_for_comms = True
    reduce_world = dist.get_world_size(reduce_scatter_group) if reduce_scatter_group is not None else 1
    all_reduce_world = dist.get_world_size(all_reduce_group) if all_reduce_group is not None else 1
    total = reduce_world * all_reduce_world
    dtype_name = str(reduce_dtype).lower()
    overflow_risk = not any(name in dtype_name for name in ("float32", "bfloat16"))
    if not overflow_risk and not force_sum_reduction_for_comms:
        if factor is None:
            if total == 1:
                return None, None, dist.ReduceOp.SUM, dist.ReduceOp.SUM
            return None, None, dist.ReduceOp.AVG, dist.ReduceOp.AVG
        factor = float(factor)
        reduce_op = dist.ReduceOp.AVG if reduce_world > 1 and factor == reduce_world else dist.ReduceOp.PREMUL_SUM
        if reduce_op == dist.ReduceOp.PREMUL_SUM:
            reduce_op = dist.ReduceOp.SUM
        return None, None, reduce_op, dist.ReduceOp.SUM
    divisor = float(total if factor is None else factor)
    if divisor <= 0:
        raise ValueError("gradient divide factor must be positive")
    if overflow_risk:
        pre = 1.0
        while divisor % pre == 0 and divisor / pre > pre:
            pre *= 2.0
        return pre, divisor / pre, dist.ReduceOp.SUM, dist.ReduceOp.SUM
    return None, divisor, dist.ReduceOp.SUM, dist.ReduceOp.SUM


def _div_if_needed(tensor: Any, div_factor: float | None) -> Any:
    return tensor if div_factor in (None, 1) else tensor / div_factor
