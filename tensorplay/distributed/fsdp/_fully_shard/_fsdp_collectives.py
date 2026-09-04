"""Collective adapters used by composable sharding."""

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import tensorplay as tp

from ... import distributed_core as dist
from ... import _symmetric_memory as symm_mem
from ._fsdp_api import AllGather, Comm, ReduceScatter
from ._fsdp_param import ShardedState

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
    event: Any = None
    work: Any = None
    param_all_gather_input_dtypes: list[list[Any]] = field(default_factory=list)
    param_all_gather_input_numels: list[list[int]] = field(default_factory=list)
    all_gather_input_split_sizes: list[int] = field(default_factory=list)

    def wait(self) -> Any:
        if self.work is not None:
            self.work.wait()
        return self.output


def _stream_context(stream: Any) -> Any:
    if stream is None:
        return contextlib.nullcontext()
    cuda = getattr(tp, "cuda", None)
    stream_context = getattr(cuda, "stream", None)
    return stream_context(stream) if callable(stream_context) else contextlib.nullcontext()


def _record_event(stream: Any) -> Any:
    record = getattr(stream, "record_event", None)
    return record() if callable(record) else None


def _current_stream(device: Any) -> Any:
    device_type = str(getattr(device, "type", device)).split(":", 1)[0].lower()
    cuda = getattr(tp, "cuda", None)
    current_stream = getattr(cuda, "current_stream", None)
    if device_type != "cuda" or not callable(current_stream):
        return None
    try:
        return current_stream(device)
    except (RuntimeError, TypeError):
        return None


def _wait_stream(dst: Any, src: Any) -> None:
    wait = getattr(dst, "wait_stream", None)
    if dst is not None and src is not None and dst != src and callable(wait):
        wait(src)


def _wait_event(event: Any, stream: Any = None) -> None:
    if event is None:
        return
    if stream is not None:
        wait = getattr(stream, "wait_event", None)
        if callable(wait):
            wait(event)
            return
    wait = getattr(event, "wait", None)
    if callable(wait):
        wait()


class DefaultAllocMixin(Comm):
    def allocate(self, size: Sequence[int], *, dtype: Any, device: Any) -> Any:
        return tp.empty(tuple(size), dtype=dtype, device=device)


class ProcessGroupAllocMixin:
    def __init__(self, group: Any = None) -> None:
        self.group = group

    def allocate(self, size: Sequence[int], *, dtype: Any, device: Any) -> Any:
        backend = None
        getter = getattr(self.group, "_get_backend", None)
        if callable(getter):
            try:
                backend = getter(device)
            except (AttributeError, RuntimeError, TypeError):
                backend = None
        if backend is None and self.group is not None:
            backend_name = str(getattr(self.group, "backend", "")).lower()
            backend = getattr(self.group, f"{backend_name}_pg", None)
        supports = getattr(backend, "supports_tensor_alloc", None)
        allocate = getattr(backend, "allocate_tensor", None)
        if callable(supports) and callable(allocate) and supports(device):
            return allocate(math.prod(int(value) for value in size), dtype=dtype, device=device)
        return tp.empty(tuple(size), dtype=dtype, device=device)


class SymmMemAllocMixin:
    def __init__(self, group: Any = None, backend: Any = None) -> None:
        self.group = group
        self.backend = backend
        if backend is not None:
            symm_mem.set_backend(backend)
        if group is not None:
            size = getattr(group, "size", None)
            world_size = int(size() if callable(size) else size or 1)
            if world_size > 1:
                dist.barrier(group=group)

    def allocate(self, size: Sequence[int], *, dtype: Any, device: Any) -> Any:
        return symm_mem.empty(tuple(size), dtype=dtype, device=device)


class DefaultAllGather(DefaultAllocMixin, AllGather):
    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, async_op: bool = False) -> Any:
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class ProcessGroupAllocAllGather(ProcessGroupAllocMixin, AllGather):
    def __init__(self, group: Any = None) -> None:
        ProcessGroupAllocMixin.__init__(self, group)

    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, async_op: bool = False) -> Any:
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class SymmMemAllGather(SymmMemAllocMixin, AllGather):
    def __init__(self, group: Any = None, backend: Any = None) -> None:
        SymmMemAllocMixin.__init__(self, group, backend)

    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, async_op: bool = False) -> Any:
        if group is not None:
            symm_mem.rendezvous(output_tensor, group)
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class DefaultReduceScatter(DefaultAllocMixin, ReduceScatter):
    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, op: Any, async_op: bool = False) -> Any:
        return dist.reduce_scatter_single(
            output_tensor,
            input_tensor,
            op=op,
            group=group,
            async_op=async_op,
        )


class ProcessGroupAllocReduceScatter(ProcessGroupAllocMixin, ReduceScatter):
    def __init__(self, group: Any = None) -> None:
        ProcessGroupAllocMixin.__init__(self, group)

    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, op: Any, async_op: bool = False) -> Any:
        return dist.reduce_scatter_single(
            output_tensor,
            input_tensor,
            op=op,
            group=group,
            async_op=async_op,
        )


class SymmMemReduceScatter(SymmMemAllocMixin, ReduceScatter):
    def __init__(self, group: Any = None, backend: Any = None) -> None:
        SymmMemAllocMixin.__init__(self, group, backend)

    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, op: Any, async_op: bool = False) -> Any:
        if group is not None:
            symm_mem.rendezvous(input_tensor, group)
            symm_mem.rendezvous(output_tensor, group)
        return dist.reduce_scatter_single(
            output_tensor,
            input_tensor,
            op=op,
            group=group,
            async_op=async_op,
        )


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
    def use_foreach_copy(param: Any) -> bool:
        local = param._sharded_local_tensor()
        return bool(
            getattr(param, "param_dtype", None) is not None
            and not getattr(param, "offload_to_cpu", False)
            and not callable(getattr(local, "fsdp_pre_all_gather", None))
        )

    param_all_gather_inputs: list[list[Any]] = [[] for _ in fsdp_params]
    foreach_copy_indices: list[int] = []
    foreach_copy_inputs: list[Any] = []
    foreach_copy_input_numels: list[int] = []

    for index, param in enumerate(fsdp_params):
        if use_foreach_copy(param):
            foreach_copy_indices.append(index)
            if param.sharded_state == ShardedState.SHARDED:
                value = param._sharded_param_data
            else:
                value = param._sharded_post_forward_param_data
            if value is None:
                raise RuntimeError("all-gather input storage is unavailable")
            foreach_copy_inputs.append(value)
            foreach_copy_input_numels.append(int(value.numel()))
        else:
            param_all_gather_inputs[index] = param.all_gather_inputs

    if foreach_copy_inputs:
        first = fsdp_params[foreach_copy_indices[0]]
        flat = tp.empty(
            (sum(foreach_copy_input_numels),),
            device=first.device,
            dtype=first.param_dtype,
        )
        splits = flat.split(tuple(foreach_copy_input_numels))
        tp._foreach_copy_(splits, foreach_copy_inputs)
        for index, split in zip(foreach_copy_indices, splits):
            param_all_gather_inputs[index] = [split]

    return param_all_gather_inputs


def foreach_all_gather(fsdp_params: Sequence[Any], group: Any, async_op: bool, all_gather_copy_in_stream: Any, all_gather_stream: Any, device: Any, all_gather_comm: AllGather | None = None) -> AllGatherResult:
    comm = all_gather_comm or DefaultAllGather()
    current_stream = _current_stream(device)
    copy_stream = all_gather_copy_in_stream or current_stream
    comm_stream = all_gather_stream or copy_stream
    with _stream_context(copy_stream):
        param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
        (
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
        if dtype == getattr(tp, "uint8", "uint8"):
            all_gather_inputs = [
                value.view(getattr(tp, "uint8", "uint8"))
                for inputs in param_all_gather_inputs
                for value in inputs
            ]
        else:
            all_gather_inputs = [
                value for inputs in param_all_gather_inputs for value in inputs
            ]
        inp_split_sizes = [int(value.numel()) for value in all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        output = comm.allocate(
            (all_gather_input_numel * world_size,),
            dtype=dtype,
            device=device,
        )
        all_gather_input, output = all_gather_copy_in_cuda(
            all_gather_inputs,
            output,
            inp_split_sizes,
            all_gather_input_numel,
            rank,
        )
    _wait_stream(comm_stream, copy_stream)
    with _stream_context(comm_stream):
        work = comm(
            output_tensor=output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        event = _record_event(comm_stream)
    return AllGatherResult(
        output,
        event,
        work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        inp_split_sizes,
    )


def foreach_all_gather_copy_out(all_gather_result: AllGatherResult, fsdp_params: Sequence[Any], group: Any) -> None:
    _wait_event(all_gather_result.event, _current_stream(all_gather_result.output.device))
    all_gather_result.wait()
    world_size = dist.get_world_size(group)
    output = all_gather_result.output
    input_sizes = all_gather_result.all_gather_input_split_sizes
    if not input_sizes:
        raise ValueError("all-gather result is missing input split sizes")
    if len(all_gather_result.param_all_gather_input_numels) != len(fsdp_params):
        raise ValueError("all-gather metadata does not match the parameter list")
    if len(all_gather_result.param_all_gather_input_dtypes) != len(fsdp_params):
        raise ValueError("all-gather dtype metadata does not match the parameter list")
    all_gather_outputs: list[Any] = []
    shard_i_copy_infos: list[tuple[Any, list[Any]]] = []
    for input_numels, input_dtypes, param in zip(
        all_gather_result.param_all_gather_input_numels,
        all_gather_result.param_all_gather_input_dtypes,
        fsdp_params,
    ):
        param.init_all_gather_outputs(input_numels, input_dtypes, world_size, output.device)
        param.alloc_all_gather_outputs()
        param_outputs = param.all_gather_outputs
        placement = getattr(param, "fsdp_placement", getattr(param, "_placement", None))
        if int(getattr(placement, "dim", 0)) != 0:
            param_outputs = [tp.empty_like(tensor) for tensor in param_outputs]
            shard_i_copy_infos.append((param, param_outputs))
        all_gather_outputs.extend(param_outputs)
    output_rows = output.view(world_size, -1)
    offset = 0
    output_dtype = getattr(tp, "uint8", "uint8")
    for target in all_gather_outputs:
        target_view = target
        if output.dtype == output_dtype:
            target_view = target.view(output_dtype)
        width = int(target_view.numel()) // world_size
        end = offset + width
        if end > int(output_rows.shape[1]):
            raise ValueError("all-gather output is smaller than the parameter outputs")
        target_view.view(world_size, -1).copy_(output_rows[:, offset:end])
        offset = end
    if offset != int(output_rows.shape[1]):
        raise ValueError("all-gather output has unused elements")
    for param, param_outputs in shard_i_copy_infos:
        placement = getattr(param, "fsdp_placement", getattr(param, "_placement", None))
        shard_dim = int(getattr(placement, "dim", 0))
        padded_size = tuple(
            int(size) for size in getattr(param, "padded_sharded_param_size", ())
        )
        sharded_state = getattr(param, "sharded_state", getattr(param, "_state", None))
        if getattr(sharded_state, "name", sharded_state) == "SHARDED_POST_FORWARD":
            post_data = getattr(param, "_sharded_post_forward_param_data", None)
            post_shape = getattr(param, "_post_forward_shape", None)
            if post_data is not None and post_shape is not None:
                padded_size = list(int(size) for size in post_shape)
                other_numel = math.prod(
                    size for index, size in enumerate(padded_size) if index != shard_dim
                )
                padded_size[shard_dim] = int(post_data.numel()) // max(other_numel, 1)
                padded_size = tuple(padded_size)
        if not padded_size:
            raise ValueError("all-gather parameter shape metadata is missing")
        pre_size = list(padded_size)
        pre_size[0] *= world_size
        post_size = list(padded_size)
        post_size[shard_dim] *= world_size
        for source, target in zip(param_outputs, param.all_gather_outputs):
            chunks = source.view(tuple(pre_size)).chunk(world_size, dim=0)
            target.view(tuple(post_size)).copy_(tp.cat(tuple(chunks), dim=shard_dim))


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
    comm_hook: Any = None,
    comm_hook_state: Any = None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
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
            str(getattr(device, "type", device)).split(":", 1)[0].lower(),
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

    reduce_scatter_input_numel = sum(math.prod(size) for size in padded_sizes)
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

    current_stream = _current_stream(device)
    _wait_stream(reduce_scatter_stream, current_stream)
    with _stream_context(reduce_scatter_stream):
        reduce_scatter_input = _div_if_needed(
            reduce_scatter_input, predivide_factor
        )
        reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
        reduce_output = comm.allocate(
            (reduce_scatter_output_numel,), dtype=reduce_dtype, device=device
        )
        if comm_hook is not None:
            comm_hook(comm_hook_state, reduce_scatter_input, reduce_output)
        elif world_size > 1:
            comm(
                reduce_output,
                reduce_scatter_input,
                reduce_scatter_group,
                reduce_scatter_op,
                False,
            )
        else:
            reduce_output.copy_(reduce_scatter_input)
        reduce_scatter_event = _record_event(reduce_scatter_stream)

    all_reduce_input = None
    all_reduce_event = None
    post_reduce_stream = reduce_scatter_stream
    if all_reduce_group is not None and comm_hook is None:
        if not all_reduce_grads:
            with _stream_context(reduce_scatter_stream):
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                post_reduce_event = _record_event(reduce_scatter_stream)
            return (
                reduce_scatter_input,
                reduce_scatter_event,
                reduce_scatter_stream,
                post_reduce_event,
                all_reduce_input,
                all_reduce_event,
                partial_reduce_output,
            )
        with _stream_context(reduce_scatter_stream):
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
        all_reduce_stream = all_reduce_stream or reduce_scatter_stream
        _wait_stream(all_reduce_stream, reduce_scatter_stream or current_stream)
        with _stream_context(all_reduce_stream):
            dist.all_reduce(
                reduce_output,
                op=all_reduce_op,
                group=all_reduce_group,
                async_op=False,
            )
            all_reduce_input = reduce_output
            all_reduce_event = _record_event(all_reduce_stream)
        post_reduce_stream = all_reduce_stream

    if all_reduce_hook is not None:
        hook_stream = all_reduce_stream or post_reduce_stream
        _wait_stream(hook_stream, reduce_scatter_stream or current_stream)
        with _stream_context(hook_stream):
            all_reduce_hook(reduce_output)
        post_reduce_stream = hook_stream

    with _stream_context(post_reduce_stream):
        reduce_output = _div_if_needed(reduce_output, postdivide_factor)
        if orig_dtype is not None and reduce_output.dtype != orig_dtype:
            reduce_output = reduce_output.to(dtype=orig_dtype)

        flat_grad_offset = 0
        for padded_size, fsdp_param in zip(padded_sizes, fsdp_params):
            sharded_size = tuple(int(size) for size in fsdp_param.sharded_size)
            new_sharded_grad = tp.as_strided(
                reduce_output,
                sharded_size,
                tuple(
                    int(stride) for stride in fsdp_param.contiguous_sharded_stride
                ),
                storage_offset=flat_grad_offset,
            )
            if getattr(fsdp_param, "offload_to_cpu", False):
                old_grad = getattr(fsdp_param, "_sharded_grad", None)
                has_post_accumulate_grad_hook = bool(
                    getattr(
                        fsdp_param._sharded_local_tensor(),
                        "_post_accumulate_grad_hooks",
                        None,
                    )
                )
                non_blocking = bool(
                    getattr(fsdp_param, "pin_memory", False)
                    and old_grad is None
                    and not has_post_accumulate_grad_hook
                )
                new_sharded_grad = new_sharded_grad.to(
                    "cpu", non_blocking=non_blocking
                )
                if non_blocking:
                    fsdp_param.grad_offload_event = _record_event(post_reduce_stream)
            else:
                old_grad = getattr(fsdp_param, "_sharded_grad", None)
            if old_grad is not None:
                new_sharded_grad = old_grad + new_sharded_grad
            fsdp_param._set_sharded_grad(new_sharded_grad)
            flat_grad_offset += math.prod(padded_size) // world_size
        post_reduce_event = _record_event(post_reduce_stream)

    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_stream,
        post_reduce_event,
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
    if div_factor in (None, 1):
        return tensor
    divide = getattr(tensor, "div_", None)
    if callable(divide):
        divide(div_factor)
        return tensor
    return tensor / div_factor
