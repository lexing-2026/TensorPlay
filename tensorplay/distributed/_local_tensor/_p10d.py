from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable, Sequence
from typing import Any

import tensorplay as tp

from .. import distributed_core as dist
from .._mesh_layout import _FlatLayout

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
    return 0 if not numbers else functools.reduce(math.gcd, numbers)


def _indices_to_layout(indices: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(indices) <= 1:
        return (), ()
    differences = [indices[index] - indices[index - 1] for index in range(1, len(indices))]
    stride = _gcd_list(differences)
    if stride == 0:
        raise AssertionError("indices must be sorted and unique")
    present = set(indices)
    starts = [indices[0]] + [value for value in indices[1:] if value - stride not in present]
    if len(indices) % len(starts):
        raise AssertionError("indices do not form a regular layout")
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
    ranks = list(dist.get_process_group_ranks(group))
    if not ranks:
        raise AssertionError("process group must contain at least one rank")
    if ranks != sorted(ranks):
        raise AssertionError(ranks)
    offset = ranks[0]
    relative_ranks = [rank - offset for rank in ranks]
    shape, strides = _indices_to_layout(relative_ranks)
    layout = _FlatLayout(shape, strides)
    try:
        world_size = int(dist.get_world_size())
    except RuntimeError:
        world_size = max(ranks) + 1
    offsets = layout.complement(world_size).all_ranks_from_zero()
    return relative_ranks, offsets, offset


class _P10dWork:
    def __init__(self, done: Callable[[], Any] | None = None, tensors: Sequence[Any] = (), source_rank: int = -1) -> None:
        self._done = done
        self._tensors = list(tensors)
        self._source = int(source_rank)
        self._completed = done is None
        self._error: BaseException | None = None

    def wait(self, timeout: Any = None) -> bool:
        del timeout
        if not self._completed and self._done is not None:
            try:
                self._done()
            except BaseException as error:
                self._error = error
                self._completed = True
                raise
            self._completed = True
        if self._error is not None:
            raise self._error
        return True

    def is_completed(self) -> bool:
        return self._completed

    def get_future(self) -> Any:
        from tensorplay import futures

        future = futures.Future()

        def complete() -> None:
            try:
                self.wait()
                future.set_result(list(self._tensors))
            except Exception as error:
                future.set_exception(error)

        future._completer = complete
        return future

    def _result_tensors(self) -> list[Any]:
        return list(self._tensors)

    def _source_rank(self) -> int:
        return self._source

    def abort(self) -> None:
        if not self._completed:
            self._completed = True
            self._done = None


def _local_functional_all_gather_into_tensor(
    tensor: Any, group_size: int, group_name: Any
) -> Any:
    del group_size
    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(_resolve_pg_or_name(group_name))
    if not isinstance(tensor, LocalTensor):
        raise AssertionError("input must be a LocalTensor")
    output: dict[int, Any] = {}
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue
        gathered = tp.cat([tensor._local_tensors[rank] for rank in group_ranks], dim=0)
        output.update({rank: gathered.clone() for rank in group_ranks})
    return LocalTensor(output)


def _local_reduce(
    reduce_op: Any, tensors: list[Any]
) -> Any:
    if not tensors:
        raise ValueError("cannot reduce an empty tensor list")
    reduce_op = getattr(reduce_op, "op", lambda: reduce_op)()
    names = {
        getattr(dist.ReduceOp, "SUM", 0): "sum",
        getattr(dist.ReduceOp, "AVG", 1): "avg",
        getattr(dist.ReduceOp, "PRODUCT", 2): "product",
        getattr(dist.ReduceOp, "MIN", 3): "min",
        getattr(dist.ReduceOp, "MAX", 4): "max",
    }
    name = names.get(reduce_op, reduce_op)
    if isinstance(name, str):
        name = name.lower()
    if name in ("sum",):
        return functools.reduce(operator.add, tensors)
    if name in ("product", "prod"):
        return functools.reduce(operator.mul, tensors)
    if name == "max":
        return functools.reduce(lambda left, right: left.maximum(right), tensors)
    if name == "min":
        return functools.reduce(lambda left, right: left.minimum(right), tensors)
    if name in ("avg", "average"):
        return functools.reduce(operator.add, tensors) / len(tensors)
    if name in ("band", "bitwise_and"):
        return functools.reduce(tp.bitwise_and, tensors)
    if name in ("bor", "bitwise_or"):
        return functools.reduce(tp.bitwise_or, tensors)
    if name in ("bxor", "bitwise_xor"):
        return functools.reduce(tp.bitwise_xor, tensors)
    if name in ("premul_sum", "premulsum"):
        raise NotImplementedError("PREMUL_SUM requires a scaling factor")
    raise NotImplementedError(f"reduce operation {reduce_op!r} is not supported")


def _local_functional_reduce_scatter_tensor(
    tensor: Any, reduce_op: Any, group_size: int, group_name: Any
) -> Any:
    del group_size
    from . import LocalTensor, _zero_sized_like

    ranks, group_offsets, _offset = _prepare_collective_groups(_resolve_pg_or_name(group_name))
    if not isinstance(tensor, LocalTensor):
        raise AssertionError("input must be a LocalTensor")
    output: dict[int, Any] = {}
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue
        reduced = _local_reduce(reduce_op, [tensor._local_tensors[rank] for rank in group_ranks])
        pieces = tp.split(reduced, reduced.size(0) // len(group_ranks), dim=0)
        for index, rank in enumerate(group_ranks):
            output[rank] = pieces[index].clone() if index < len(pieces) else _zero_sized_like(reduced, 0)
    return LocalTensor(output)


def _local_functional_shard_dim_alltoall(
    tensor: Any, gather_dim: int, shard_dim: int, group_name: Any
) -> Any:
    from . import LocalTensor, _zero_sized_like

    ranks, group_offsets, _offset = _prepare_collective_groups(_resolve_pg_or_name(group_name))
    if not isinstance(tensor, LocalTensor):
        raise AssertionError("input must be a LocalTensor")
    output: dict[int, Any] = {}
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue
        gathered = tp.cat([tensor._local_tensors[rank] for rank in group_ranks], dim=gather_dim)
        pieces = tp.split(gathered, gathered.size(shard_dim) // len(group_ranks), dim=shard_dim)
        for index, rank in enumerate(group_ranks):
            output[rank] = pieces[index].clone() if index < len(pieces) else _zero_sized_like(gathered, shard_dim)
    return LocalTensor(output)


def _local_functional_all_to_all_single(
    tensor: Any, output_split_sizes: Sequence[Any], input_split_sizes: Sequence[Any], group_name: Any
) -> Any:
    from . import LocalIntNode, LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(_resolve_pg_or_name(group_name))
    if not isinstance(tensor, LocalTensor):
        raise AssertionError("input must be a LocalTensor")
    local_sizes: dict[int, list[int]] = {}
    for split_size in input_split_sizes:
        values = split_size._local_ints if isinstance(split_size, LocalIntNode) else {
            rank: int(split_size) for rank in tensor._local_tensors
        }
        for rank, size in values.items():
            local_sizes.setdefault(rank, []).append(size)
    local_splits = {
        rank: list(tp.split(tensor._local_tensors[rank], sizes))
        for rank, sizes in local_sizes.items()
    }
    output: dict[int, Any] = {}
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in local_splits for rank in group_ranks):
            continue
        for index, destination in enumerate(group_ranks):
            output[destination] = tp.cat([local_splits[source][index] for source in group_ranks])
    return LocalTensor(output)


def _local_broadcast_(
    tensors: list[Any], process_group: Any, root_rank: int, root_tensor: int,
    async_op: bool = True, timeout: int = -1,
) -> tuple[list[Any], _P10dWork]:
    del async_op, timeout
    from . import LocalTensor

    if len(tensors) != 1 or root_tensor != 0:
        raise AssertionError
    tensor = tensors[0]
    if not isinstance(tensor, LocalTensor):
        raise AssertionError("input must be a LocalTensor")
    ranks, group_offsets, offset = _prepare_collective_groups(process_group)
    relative_root = root_rank - offset
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue
        source = tensor._local_tensors[group_offset + relative_root]
        for rank in group_ranks:
            if rank != group_offset + relative_root:
                tensor._local_tensors[rank].copy_(source)
    return tensors, _P10dWork()


def _local_all_reduce_(
    tensors: list[Any], process_group: Any, reduce_op: Any, sparse_indices: Any = None,
    async_op: bool = True, timeout: int = -1,
) -> tuple[list[Any], _P10dWork]:
    del sparse_indices, async_op, timeout
    from . import LocalTensor

    if len(tensors) != 1:
        raise AssertionError
    tensor = tensors[0]
    if not isinstance(tensor, LocalTensor):
        raise AssertionError("input must be a LocalTensor")
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group)
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue
        result = _local_reduce(reduce_op, [tensor._local_tensors[rank] for rank in group_ranks])
        for rank in group_ranks:
            tensor._local_tensors[rank].copy_(result)
    return tensors, _P10dWork()


def _local_allreduce_coalesced_(
    tensors: list[Any], process_group: Any, reduce_op: Any,
    async_op: bool = True, timeout: int = -1,
) -> _P10dWork:
    for tensor in tensors:
        _local_all_reduce_([tensor], process_group, reduce_op, async_op=async_op, timeout=timeout)
    return _P10dWork()


def _local_reduce_scatter_tensor_coalesced_(
    output_tensors: list[Any], input_tensors: list[Any], process_group: Any, reduce_op: Any,
    async_op: bool = True, timeout: int = -1,
) -> _P10dWork:
    del async_op, timeout
    from . import LocalTensor

    if len(output_tensors) != len(input_tensors):
        raise AssertionError("input and output counts must match")
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group)
    for output, input_tensor in zip(output_tensors, input_tensors):
        if not isinstance(input_tensor, LocalTensor) or not isinstance(output, LocalTensor):
            raise AssertionError("collective tensors must be LocalTensor instances")
        for group_offset in group_offsets:
            group_ranks = [group_offset + rank for rank in ranks]
            if not all(rank in input_tensor._local_tensors for rank in group_ranks):
                continue
            if not all(rank in output._local_tensors for rank in group_ranks):
                continue
            reduced = _local_reduce(reduce_op, [input_tensor._local_tensors[rank] for rank in group_ranks])
            pieces = tp.split(reduced, reduced.size(0) // len(group_ranks), dim=0)
            for index, rank in enumerate(group_ranks):
                output._local_tensors[rank].copy_(pieces[index])
    return _P10dWork()


def _local_allgather_base_(
    output_tensor: Any, input_tensor: Any, process_group: Any,
    async_op: bool = True, timeout: int = -1,
) -> tuple[Any, _P10dWork]:
    del async_op, timeout
    from . import LocalTensor

    if not isinstance(output_tensor, LocalTensor) or not isinstance(input_tensor, LocalTensor):
        raise AssertionError("collective tensors must be LocalTensor instances")
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group)
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue
        gathered = tp.cat([input_tensor._local_tensors[rank] for rank in group_ranks], dim=0)
        for rank in group_ranks:
            output_tensor._local_tensors[rank].copy_(gathered)
    return output_tensor, _P10dWork()


def _local_reduce_scatter_base_(
    output_tensor: Any, input_tensor: Any, process_group: Any, reduce_op: Any,
    async_op: bool = True, timeout: int = -1,
) -> tuple[Any, _P10dWork]:
    del async_op, timeout
    from . import LocalTensor

    if not isinstance(output_tensor, LocalTensor) or not isinstance(input_tensor, LocalTensor):
        raise AssertionError("collective tensors must be LocalTensor instances")
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group)
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue
        reduced = _local_reduce(reduce_op, [input_tensor._local_tensors[rank] for rank in group_ranks])
        pieces = tp.split(reduced, reduced.size(0) // len(group_ranks), dim=0)
        for index, rank in enumerate(group_ranks):
            output_tensor._local_tensors[rank].copy_(pieces[index].clone())
    return output_tensor, _P10dWork()


def _local_all_gather_(
    output_tensors: list[list[Any]], input_tensors: list[Any], process_group: Any,
    async_op: bool = True, timeout: int = -1,
) -> tuple[list[list[Any]], _P10dWork]:
    del async_op, timeout
    from . import LocalTensor

    if len(output_tensors) != 1 or len(input_tensors) != 1:
        raise AssertionError
    outputs = output_tensors[0]
    input_tensor = input_tensors[0]
    if not all(isinstance(output, LocalTensor) for output in outputs):
        raise AssertionError("outputs must be LocalTensor instances")
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group)
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        for index, rank in enumerate(group_ranks):
            source = input_tensor._local_tensors[rank] if isinstance(input_tensor, LocalTensor) else input_tensor
            outputs[index].copy_(source)
    return [outputs], _P10dWork()


def _local_allgather_into_tensor_coalesced_(
    output_tensors: list[Any], input_tensors: list[Any], process_group: Any,
    async_op: bool = True,
) -> _P10dWork:
    del async_op
    if len(output_tensors) != len(input_tensors):
        raise AssertionError("input and output counts must match")
    for output, input_tensor in zip(output_tensors, input_tensors):
        _local_allgather_base_(output, input_tensor, process_group)
    return _P10dWork()


def _local_gather_(
    output_tensors: list[list[Any]], input_tensors: list[Any], process_group: Any,
    root_rank: int, async_op: bool = True, timeout: int = -1,
) -> tuple[list[list[Any]], _P10dWork]:
    del async_op, timeout
    from . import LocalTensor

    if len(input_tensors) != 1 or len(output_tensors) > 1:
        raise ValueError("gather accepts one input tensor and at most one output list")
    input_tensor = input_tensors[0]
    if not isinstance(input_tensor, LocalTensor):
        raise TypeError("input must be a LocalTensor")
    ranks, group_offsets, offset = _prepare_collective_groups(process_group)
    relative_root = int(root_rank) - offset
    if relative_root < 0 or relative_root >= len(ranks):
        raise ValueError("root rank is not in the process group")
    if output_tensors:
        outputs = output_tensors[0]
        if len(outputs) != len(ranks):
            raise ValueError("gather output list must match the process-group size")
    else:
        outputs = []
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        root = group_offset + relative_root
        if not outputs:
            continue
        for index, output in enumerate(outputs):
            source = input_tensor._local_tensors[group_ranks[index]]
            if isinstance(output, LocalTensor):
                if root in output._local_tensors:
                    output._local_tensors[root].copy_(source)
                elif len(output._local_tensors) == 1:
                    next(iter(output._local_tensors.values())).copy_(source)
                else:
                    raise ValueError("each gather output must contain the destination rank")
            else:
                output.copy_(source)
    return output_tensors, _P10dWork(tensors=outputs)


def _local_scatter_(
    output_tensors: list[Any], input_tensors: list[list[Any]], process_group: Any,
    root_rank: int, async_op: bool = True, timeout: int = -1,
) -> tuple[list[Any], _P10dWork]:
    del async_op, timeout
    from . import LocalTensor

    if len(output_tensors) != 1 or len(input_tensors) != 1:
        raise AssertionError
    output = output_tensors[0]
    inputs = input_tensors[0]
    if not isinstance(output, LocalTensor):
        raise AssertionError("output must be a LocalTensor")
    ranks, group_offsets, offset = _prepare_collective_groups(process_group)
    if len(ranks) != len(inputs):
        raise AssertionError((ranks, inputs))
    relative_root = root_rank - offset
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in output._local_tensors for rank in group_ranks):
            continue
        for index, rank in enumerate(group_ranks):
            source = inputs[index]
            if not isinstance(source, LocalTensor):
                raise AssertionError("inputs must be LocalTensor instances")
            output._local_tensors[rank].copy_(source._local_tensors[group_offset + relative_root])
    return output_tensors, _P10dWork()


def _local_alltoall_(
    output_tensors: list[Any], input_tensors: list[Any], process_group: Any,
    async_op: bool = True, timeout: int = -1,
) -> tuple[list[Any], _P10dWork]:
    del async_op, timeout
    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group)
    if not (len(input_tensors) == len(output_tensors) == len(ranks)):
        raise AssertionError("input, output, and process-group sizes must match")
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        for destination_index, destination_rank in enumerate(group_ranks):
            output = output_tensors[destination_index]
            if not isinstance(output, LocalTensor):
                raise AssertionError("outputs must be LocalTensor instances")
            if not all(rank in output._local_tensors for rank in group_ranks):
                continue
            for source_index, source_rank in enumerate(group_ranks):
                source = input_tensors[source_index]
                if not isinstance(source, LocalTensor):
                    raise AssertionError("inputs must be LocalTensor instances")
                if not all(rank in source._local_tensors for rank in group_ranks):
                    continue
                output._local_tensors[source_rank].copy_(source._local_tensors[destination_rank])
    return output_tensors, _P10dWork()


def _local_alltoall_base_(
    output_tensor: Any, input_tensor: Any, process_group: Any,
    output_split_sizes: Sequence[int] | None, input_split_sizes: Sequence[int] | None,
    async_op: bool = True, timeout: int = -1,
) -> _P10dWork:
    del async_op, timeout
    from . import LocalTensor

    if not isinstance(input_tensor, LocalTensor) or not isinstance(output_tensor, LocalTensor):
        raise AssertionError("collective tensors must be LocalTensor instances")
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group)
    output_sizes = list(output_split_sizes) if output_split_sizes else None
    input_sizes = list(input_split_sizes) if input_split_sizes else None
    for group_offset in group_offsets:
        group_ranks = [group_offset + rank for rank in ranks]
        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue
        for source_rank in group_ranks:
            source = input_tensor._local_tensors[source_rank]
            splits = (
                tp.split(source, input_sizes, dim=0)
                if input_sizes
                else tp.split(source, source.size(0) // len(group_ranks), dim=0)
            )
            for destination_index, destination_rank in enumerate(group_ranks):
                if destination_index >= len(splits):
                    continue
                piece = splits[destination_index]
                output = output_tensor._local_tensors[destination_rank]
                if output_sizes:
                    offset = sum(output_sizes[:source_rank - group_offset])
                    end = offset + output_sizes[source_rank - group_offset]
                else:
                    width = output.size(0) // len(group_ranks)
                    offset = (source_rank - group_offset) * width
                    end = min(offset + width, output.size(0))
                section = output[offset:end]
                if piece.numel() != section.numel():
                    raise ValueError(
                        f"all_to_all input split from rank {source_rank} to rank {destination_rank} "
                        f"has {piece.numel()} elements, but the output split has {section.numel()}"
                    )
                if section.numel():
                    section.copy_(piece.reshape(section.shape))
    return _P10dWork()


def _local_barrier(
    tensor: Any, process_group: Any, device_ids: list[int],
    async_op: bool = True, timeout: int = -1,
) -> _P10dWork:
    del process_group, device_ids, async_op, timeout
    from . import LocalTensor

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("barrier tensor must be a LocalTensor")
    return _P10dWork()


def _local_monitored_barrier_(
    tensor: Any, process_group: Any, device_ids: list[int], timeout: int,
    wait_all_ranks: bool,
) -> None:
    del process_group, device_ids, timeout, wait_all_ranks
    from . import LocalTensor

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("barrier tensor must be a LocalTensor")


def _local_send(tensors: list[Any], process_group: Any, dst: int, tag: int) -> _P10dWork:
    del process_group
    from . import LocalRunnerMode, LocalTensor

    if len(tensors) != 1 or not isinstance(tensors[0], LocalTensor):
        raise AssertionError("send expects one LocalTensor")
    tensor = tensors[0]
    source = int(getattr(tensor, "__src_rank__", min(tensor._ranks)))
    runner = LocalRunnerMode.current()
    if runner is None:
        raise RuntimeError("a LocalRunnerMode is required for point-to-point operations")
    runner._signal_send(source, dst, tensor._local_tensors[source], tag)
    return _P10dWork()


def _local_recv_(tensors: list[Any], process_group: Any, src: int, tag: int) -> _P10dWork:
    del process_group
    from . import LocalRunnerMode, LocalTensor

    if len(tensors) != 1 or not isinstance(tensors[0], LocalTensor):
        raise AssertionError("receive expects one LocalTensor")
    tensor = tensors[0]
    destination = int(getattr(tensor, "__src_rank__", min(tensor._ranks)))
    runner = LocalRunnerMode.current()
    if runner is None:
        raise RuntimeError("a LocalRunnerMode is required for point-to-point operations")
    value = runner._wait_recv(src, destination, tag=tag)
    if value is not None:
        tensor._local_tensors[destination].copy_(value)
    return _P10dWork(source_rank=getattr(runner._last_recv_source, "rank", src))


def _local_recv_any_source_(tensors: list[Any], process_group: Any, tag: int) -> _P10dWork:
    return _local_recv_(tensors, process_group, -1, tag)


def _attach_rank(tensor: Any, rank: int) -> Any:
    if hasattr(tensor, "_local_tensor") and not hasattr(tensor, "_local_tensors"):
        tensor = tensor._local_tensor
    tensor.__src_rank__ = rank
    return tensor


def local_p2p_op(dst: Any, tensor: Any, op: Callable[[Any, int], Any]) -> Any:
    dist._check_op(op)
    from . import LocalIntNode

    node = getattr(dst, "node", dst)
    if not isinstance(node, LocalIntNode):
        raise AssertionError("destination must carry one value per source rank")
    return [op(_attach_rank(tensor, source), destination) for source, destination in node._local_ints.items()]


def wait_all(work: Any) -> None:
    if work is None:
        return
    values = work if isinstance(work, (list, tuple)) else [work]
    for item in values:
        if item is not None:
            item.wait()
