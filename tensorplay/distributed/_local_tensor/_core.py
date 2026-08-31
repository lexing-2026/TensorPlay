from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable, Sequence
from typing import Any

from .._mesh_layout import _FlatLayout
from .. import distributed_core as dist
from . import LocalIntNode, LocalTensor, LocalRunnerMode, _zero_sized_like

__all__ = [
    "_prepare_collective_groups",
    "_local_functional_all_gather_into_tensor",
    "_local_functional_reduce_scatter_tensor",
    "_local_functional_shard_dim_alltoall",
    "_local_functional_all_to_all_single",
    "_local_broadcast_",
    "_local_reduce",
    "_local_all_reduce_",
    "_local_allreduce_coalesced_",
    "_local_reduce_scatter_tensor_coalesced_",
    "_local_allgather_base_",
    "_local_reduce_scatter_base_",
    "_local_all_gather_",
    "_local_allgather_into_tensor_coalesced_",
    "_local_gather_",
    "_local_scatter_",
    "_local_alltoall_",
    "_local_alltoall_base_",
    "_local_barrier",
    "_local_monitored_barrier_",
    "_local_send",
    "_local_recv_",
    "_local_recv_any_source_",
    "_attach_rank",
    "local_p2p_op",
    "wait_all",
]


def _gcd_list(numbers: Sequence[int]) -> int:
    return functools.reduce(math.gcd, numbers, 0)


def _indices_to_layout(indices: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(indices) <= 1:
        return (), ()
    differences = [indices[index] - indices[index - 1] for index in range(1, len(indices))]
    stride = _gcd_list(differences)
    if stride == 0:
        raise ValueError("indices must be unique")
    present = set(indices)
    starts = [indices[0]] + [value for value in indices[1:] if value - stride not in present]
    if len(indices) % len(starts):
        raise ValueError("indices do not form a regular layout")
    shape = len(indices) // len(starts)
    higher_shape, higher_stride = _indices_to_layout(starts)
    return higher_shape + (shape,), higher_stride + (stride,)


def _resolve_pg_or_name(group: Any) -> Any:
    if isinstance(group, dist.ProcessGroup):
        return group
    if group is None:
        return dist._get_default_group()
    if isinstance(group, str):
        return dist._resolve_group(group)
    return group


def _prepare_collective_groups(process_group: Any) -> tuple[list[int], list[int], int]:
    group = _resolve_pg_or_name(process_group)
    ranks = list(getattr(group, "ranks", []))
    if not ranks:
        try:
            ranks = list(dist.get_process_group_ranks(group))
        except Exception:
            ranks = [dist.get_rank(group)]
    if ranks != sorted(ranks):
        raise ValueError("local simulation requires sorted group ranks")
    offset = ranks[0]
    relative = [rank - offset for rank in ranks]
    shape, stride = _indices_to_layout(relative)
    layout = _FlatLayout(shape, stride)
    try:
        world_size = dist.get_world_size()
    except Exception:
        world_size = max(ranks) + 1
    offsets = layout.complement(world_size).all_ranks_from_zero()
    return relative, offsets, offset


class _LocalWork:
    def wait(self, timeout: Any = None) -> bool:
        del timeout
        return True

    def is_completed(self) -> bool:
        return True

    def get_future(self) -> Any:
        return None


def _groups(process_group: Any, local: LocalTensor) -> list[list[int]]:
    try:
        relative, offsets, base = _prepare_collective_groups(process_group)
        groups = [[offset + rank for rank in relative] for offset in offsets]
    except Exception:
        groups = [sorted(local._ranks)]
        base = groups[0][0] if groups[0] else 0
    del base
    return [group for group in groups if all(rank in local._local_tensors for rank in group)]


def _local_reduce(reduce_op: Any, tensors: list[Any]) -> Any:
    if not tensors:
        raise ValueError("cannot reduce an empty tensor list")
    name = getattr(reduce_op, "name", reduce_op)
    if isinstance(name, str):
        name = name.lower()
    if name in (0, "sum"):
        return functools.reduce(operator.add, tensors)
    if name in (1, "product", "prod"):
        return functools.reduce(operator.mul, tensors)
    if name in (2, "max"):
        result = tensors[0]
        for value in tensors[1:]:
            result = result.maximum(value) if hasattr(result, "maximum") else result
        return result
    if name in (3, "min"):
        result = tensors[0]
        for value in tensors[1:]:
            result = result.minimum(value) if hasattr(result, "minimum") else result
        return result
    if name in (4, "avg", "average"):
        return functools.reduce(operator.add, tensors) / len(tensors)
    raise NotImplementedError(f"reduce operation {reduce_op!r} is not supported")


def _local_functional_all_gather_into_tensor(tensor: LocalTensor, group_size: int, group_name: Any) -> LocalTensor:
    import tensorplay as tp

    del group_size
    output = {}
    for group in _groups(group_name, tensor):
        gathered = tp.cat([tensor._local_tensors[rank] for rank in group], dim=0)
        output.update({rank: gathered.clone() for rank in group})
    return LocalTensor(output)


def _local_functional_reduce_scatter_tensor(tensor: LocalTensor, reduce_op: Any, group_size: int, group_name: Any) -> LocalTensor:
    del group_size
    output = {}
    for group in _groups(group_name, tensor):
        reduced = _local_reduce(reduce_op, [tensor._local_tensors[rank] for rank in group])
        pieces = reduced.chunk(len(group), dim=0)
        output.update({rank: pieces[index].clone() for index, rank in enumerate(group)})
    return LocalTensor(output)


def _local_functional_shard_dim_alltoall(tensor: LocalTensor, gather_dim: int, shard_dim: int, group_name: Any) -> LocalTensor:
    import tensorplay as tp
    output = {}
    for group in _groups(group_name, tensor):
        gathered = tp.cat([tensor._local_tensors[rank] for rank in group], dim=gather_dim)
        pieces = gathered.chunk(len(group), dim=shard_dim)
        output.update({rank: pieces[index].clone() for index, rank in enumerate(group)})
    return LocalTensor(output)


def _local_functional_all_to_all_single(tensor: LocalTensor, output_split_sizes: Sequence[int], input_split_sizes: Sequence[int], group_name: Any) -> LocalTensor:
    import tensorplay as tp
    output = {}
    for group in _groups(group_name, tensor):
        parts = {rank: list(tp.split(tensor._local_tensors[rank], list(input_split_sizes), dim=0)) for rank in group}
        for destination_index, destination in enumerate(group):
            output[destination] = tp.cat([parts[source][destination_index] for source in group], dim=0)
    return LocalTensor(output)


def _local_broadcast_(tensors: list[LocalTensor], process_group_so: Any, root_rank: int, root_tensor: int, async_op: bool = True, timeout: int = -1):
    del root_tensor, async_op, timeout
    tensor = tensors[0]
    for group in _groups(process_group_so, tensor):
        source = tensor._local_tensors[group[root_rank % len(group)]]
        for rank in group:
            tensor._local_tensors[rank].copy_(source)
    return tensors, _LocalWork()


def _local_all_reduce_(tensors: list[LocalTensor], process_group_so: Any, reduce_op_so: Any, sparse_indices: Any = None, async_op: bool = True, timeout: int = -1):
    del sparse_indices, async_op, timeout
    tensor = tensors[0]
    op = getattr(reduce_op_so, "op", lambda: reduce_op_so)()
    for group in _groups(process_group_so, tensor):
        result = _local_reduce(op, [tensor._local_tensors[rank] for rank in group])
        for rank in group:
            tensor._local_tensors[rank].copy_(result)
    return tensors, _LocalWork()


def _local_allreduce_coalesced_(tensors: list[LocalTensor], process_group_so: Any, reduce_op_so: Any, async_op: bool = True, timeout: int = -1):
    for tensor in tensors:
        _local_all_reduce_([tensor], process_group_so, reduce_op_so, async_op=async_op, timeout=timeout)
    return _LocalWork()


def _local_reduce_scatter_tensor_coalesced_(output_tensors: list[LocalTensor], input_tensors: list[LocalTensor], process_group_so: Any, reduce_op_so: Any, async_op: bool = True, timeout: int = -1):
    del async_op, timeout
    op = getattr(reduce_op_so, "op", lambda: reduce_op_so)()
    for output, input_tensor in zip(output_tensors, input_tensors):
        for group in _groups(process_group_so, input_tensor):
            pieces = _local_reduce(op, [input_tensor._local_tensors[rank] for rank in group]).chunk(len(group), dim=0)
            for index, rank in enumerate(group):
                output._local_tensors[rank].copy_(pieces[index])
    return _LocalWork()


def _local_allgather_base_(output_tensor: LocalTensor, input_tensor: LocalTensor, process_group_so: Any, async_op: bool = True, timeout: int = -1):
    del async_op, timeout
    for group in _groups(process_group_so, input_tensor):
        import tensorplay as tp
        value = tp.cat([input_tensor._local_tensors[rank] for rank in group], dim=0)
        for rank in group:
            output_tensor._local_tensors[rank].copy_(value)
    return output_tensor, _LocalWork()


def _local_reduce_scatter_base_(output_tensor: LocalTensor, input_tensor: LocalTensor, process_group_so: Any, reduce_op_so: Any, async_op: bool = True, timeout: int = -1):
    del async_op, timeout
    op = getattr(reduce_op_so, "op", lambda: reduce_op_so)()
    for group in _groups(process_group_so, input_tensor):
        pieces = _local_reduce(op, [input_tensor._local_tensors[rank] for rank in group]).chunk(len(group), dim=0)
        for index, rank in enumerate(group):
            output_tensor._local_tensors[rank].copy_(pieces[index])
    return output_tensor, _LocalWork()


def _local_all_gather_(output_tensors: list[list[LocalTensor]], input_tensors: list[LocalTensor], process_group_so: Any, async_op: bool = True, timeout: int = -1):
    del async_op, timeout
    input_tensor = input_tensors[0]
    outputs = output_tensors[0]
    for group in _groups(process_group_so, input_tensor):
        for index, rank in enumerate(group):
            outputs[index]._local_tensors[rank].copy_(input_tensor._local_tensors[rank])
    return output_tensors, _LocalWork()


def _local_allgather_into_tensor_coalesced_(output_tensors: list[LocalTensor], input_tensors: list[LocalTensor], process_group_so: Any, async_op: bool = True):
    for output, input_tensor in zip(output_tensors, input_tensors):
        _local_allgather_base_(output, input_tensor, process_group_so, async_op=async_op)
    return _LocalWork()


def _local_gather_(output_tensors: Any, input_tensors: Any, process_group_so: Any, root_rank: int, async_op: bool = True, timeout: int = -1):
    del output_tensors, input_tensors, process_group_so, root_rank, async_op, timeout
    raise NotImplementedError("gather is not defined for a single-program local simulation")


def _local_scatter_(output_tensors: list[LocalTensor], input_tensors: list[list[LocalTensor]], process_group_so: Any, root_rank: int, async_op: bool = True, timeout: int = -1):
    del async_op, timeout
    output = output_tensors[0]
    inputs = input_tensors[0]
    for group in _groups(process_group_so, output):
        for index, rank in enumerate(group):
            source = inputs[index]._local_tensors[group[root_rank % len(group)]]
            output._local_tensors[rank].copy_(source)
    return output_tensors, _LocalWork()


def _local_alltoall_(output_tensors: list[LocalTensor], input_tensors: list[LocalTensor], process_group_so: Any, async_op: bool = True, timeout: int = -1):
    del async_op, timeout
    for destination_index, output in enumerate(output_tensors):
        for group in _groups(process_group_so, output):
            for source_index, source in enumerate(input_tensors):
                output._local_tensors[group[destination_index]].copy_(source._local_tensors[group[source_index]])
    return output_tensors, _LocalWork()


def _local_alltoall_base_(output_tensor: LocalTensor, input_tensor: LocalTensor, process_group_so: Any, output_split_sizes: Sequence[int], input_split_sizes: Sequence[int], async_op: bool = True, timeout: int = -1):
    del async_op, timeout
    result = _local_functional_all_to_all_single(input_tensor, output_split_sizes, input_split_sizes, process_group_so)
    for rank, value in result._local_tensors.items():
        output_tensor._local_tensors[rank].copy_(value)
    return _LocalWork()


def _local_barrier(tensor: LocalTensor, process_group_so: Any, device_ids: list[int], async_op: bool = True, timeout: int = -1):
    del tensor, process_group_so, device_ids, async_op, timeout
    return _LocalWork()


def _local_monitored_barrier_(tensor: LocalTensor, process_group_so: Any, device_ids: list[int], timeout: int, wait_all_ranks: bool) -> None:
    del tensor, process_group_so, device_ids, timeout, wait_all_ranks


def _local_send(tensors: list[LocalTensor], process_group_so: Any, dst: int, tag: int):
    del process_group_so, tag
    runner = LocalRunnerMode.current()
    if runner is None:
        raise RuntimeError("a LocalRunnerMode is required for point-to-point operations")
    tensor = tensors[0]
    source = getattr(tensor, "__src_rank__", min(tensor._ranks))
    runner._signal_send(source, dst, tensor._local_tensors[source])
    return _LocalWork()


def _local_recv_(tensors: list[LocalTensor], process_group_so: Any, src: int, tag: int):
    del process_group_so, tag
    runner = LocalRunnerMode.current()
    if runner is None:
        raise RuntimeError("a LocalRunnerMode is required for point-to-point operations")
    tensor = tensors[0]
    destination = getattr(tensor, "__src_rank__", min(tensor._ranks))
    value = runner._wait_recv(src, destination)
    if value is not None:
        tensor._local_tensors[destination].copy_(value)
    return _LocalWork()


def _local_recv_any_source_(tensors: list[LocalTensor], process_group_so: Any, tag: int):
    return _local_recv_(tensors, process_group_so, -1, tag)


def _attach_rank(tensor: Any, rank: int) -> Any:
    tensor.__src_rank__ = rank
    return tensor


def local_p2p_op(dst: Any, tensor: Any, op: Callable[[Any, int], Any]) -> Any:
    if isinstance(dst, LocalIntNode):
        return [op(_attach_rank(tensor, source), destination) for source, destination in dst._local_ints.items()]
    raise TypeError("destination must carry one value per source rank")


def wait_all(work: Any) -> None:
    if work is None:
        return
    values = work if isinstance(work, (list, tuple)) else [work]
    for item in values:
        if item is not None:
            item.wait()
