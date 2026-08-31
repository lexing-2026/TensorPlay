"""Collective adapters used by composable sharding."""

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
        world_size = dist.get_world_size(group)
        outputs = [output_tensor.new_empty(output_tensor.shape) for _ in range(world_size)]
        work = dist.all_gather(outputs, input_tensor, group=group, async_op=async_op)
        for index, value in enumerate(outputs):
            output_tensor[index].copy_(value)
        return work


class ProcessGroupAllocAllGather(DefaultAllGather, ProcessGroupAllocMixin):
    def __init__(self, group: Any = None) -> None:
        ProcessGroupAllocMixin.__init__(self, group)


class SymmMemAllGather(ProcessGroupAllocAllGather, SymmMemAllocMixin):
    def __init__(self, group: Any = None, backend: Any = None) -> None:
        SymmMemAllocMixin.__init__(self, group, backend)


class DefaultReduceScatter(DefaultAllocMixin, ReduceScatter):
    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, op: Any, async_op: bool = False) -> Any:
        world_size = dist.get_world_size(group)
        chunks = list(input_tensor.chunk(world_size, dim=0))
        return dist.reduce_scatter(output_tensor, chunks, op=op, group=group, async_op=async_op)


class ProcessGroupAllocReduceScatter(DefaultReduceScatter, ProcessGroupAllocMixin):
    def __init__(self, group: Any = None) -> None:
        ProcessGroupAllocMixin.__init__(self, group)


class SymmMemReduceScatter(ProcessGroupAllocReduceScatter, SymmMemAllocMixin):
    def __init__(self, group: Any = None, backend: Any = None) -> None:
        SymmMemAllocMixin.__init__(self, group, backend)


def all_gather_copy_in_meta(all_gather_inputs: Sequence[Any], all_gather_output: Any, inp_split_sizes: Sequence[int], all_gather_input_numel: int, rank: int) -> Any:
    del all_gather_inputs, inp_split_sizes, all_gather_input_numel, rank
    return all_gather_output


def all_gather_copy_in_cuda(all_gather_inputs: Sequence[Any], all_gather_output: Any, inp_split_sizes: Sequence[int], all_gather_input_numel: int, rank: int) -> Any:
    del inp_split_sizes, all_gather_input_numel, rank
    offset = 0
    for value in all_gather_inputs:
        end = offset + int(value.numel())
        all_gather_output[offset:end].copy_(value.reshape(-1))
        offset = end
    return all_gather_output


def split_with_sizes_copy(all_gather_output: Any, all_gather_input_split_sizes: Sequence[int], dim: int, out: Sequence[Any]) -> Sequence[Any]:
    for target, value in zip(out, all_gather_output.split(tuple(all_gather_input_split_sizes), dim=dim)):
        target.copy_(value)
    return out


def chunk_cat(tensors: Sequence[Any], dim: int, num_chunks: int, out: Any = None) -> Any:
    result = tp.cat(tuple(tensors), dim=dim)
    if out is not None:
        out.copy_(result)
        return out
    return result.chunk(num_chunks, dim=dim)


def _get_param_all_gather_inputs(fsdp_params: Sequence[Any]) -> list[Any]:
    return [param.all_gather_inputs() for param in fsdp_params]


def foreach_all_gather(fsdp_params: Sequence[Any], group: Any, async_op: bool, all_gather_copy_in_stream: Any, all_gather_stream: Any, device: Any, all_gather_comm: AllGather | None = None) -> list[AllGatherResult]:
    del all_gather_copy_in_stream, all_gather_stream, device
    comm = all_gather_comm or DefaultAllGather()
    results = []
    for param in fsdp_params:
        local = param._sharded_local_tensor()
        width = int(local.numel()) * dist.get_world_size(group)
        output = local.new_empty(width)
        work = comm(output, local.reshape(-1), group, async_op)
        results.append(AllGatherResult(output, work))
    return results


def foreach_all_gather_copy_out(all_gather_result: Sequence[AllGatherResult], fsdp_params: Sequence[Any], group: Any) -> None:
    del group
    for result, param in zip(all_gather_result, fsdp_params):
        param._use_unsharded_tensor(result.wait())


def foreach_reduce(fsdp_params: Sequence[Any], unsharded_grads: Sequence[Any], reduce_scatter_group: Any, reduce_scatter_stream: Any, reduce_scatter_comm: ReduceScatter | None, orig_dtype: Any, reduce_dtype: Any, device: Any, gradient_divide_factor: float, all_reduce_group: Any, all_reduce_stream: Any, all_reduce_grads: bool, partial_reduce_output: Any, all_reduce_hook: Any, force_sum_reduction_for_comms: bool) -> None:
    del reduce_scatter_stream, orig_dtype, device, all_reduce_stream, partial_reduce_output, all_reduce_hook, force_sum_reduction_for_comms
    for param, grad in zip(fsdp_params, unsharded_grads):
        value = grad.to(dtype=reduce_dtype) if reduce_dtype is not None else grad
        if all_reduce_grads:
            dist.all_reduce(value, group=all_reduce_group)
        elif reduce_scatter_comm is not None:
            local = value.new_empty(tuple(max(1, s // dist.get_world_size(reduce_scatter_group)) for s in value.shape))
            reduce_scatter_comm(local, value, reduce_scatter_group, dist.ReduceOp.SUM, False)
            value = local
        if gradient_divide_factor not in (None, 1):
            value = value / gradient_divide_factor
        param._set_sharded_grad(value)


def foreach_reduce_scatter_copy_in(unsharded_grads: Sequence[Any], reduce_scatter_input: Any, world_size: int) -> Any:
    offset = 0
    for grad in unsharded_grads:
        end = offset + int(grad.numel())
        reduce_scatter_input[offset:end].copy_(grad.reshape(-1))
        offset = end
    return reduce_scatter_input


def _get_all_gather_input_metadatas(param_all_gather_inputs: Sequence[Any]) -> list[tuple[Any, ...]]:
    return [tuple(value.shape) for value in param_all_gather_inputs]


def _get_gradient_divide_factors(reduce_scatter_group: Any, all_reduce_group: Any, reduce_dtype: Any, device_type: str, factor: float | None, force_sum_reduction_for_comms: bool) -> tuple[float, float]:
    del reduce_dtype, device_type, force_sum_reduction_for_comms
    return float(factor or 1), float(factor or 1)


def _div_if_needed(tensor: Any, div_factor: float | None) -> Any:
    return tensor if div_factor in (None, 1) else tensor / div_factor
