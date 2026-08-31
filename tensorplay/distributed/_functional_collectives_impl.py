from __future__ import annotations

from typing import Any

import tensorplay.distributed as dist
from . import _functional_collectives as functional

__all__ = [
    "_broadcast",
    "_all_reduce",
    "_all_reduce_coalesced",
    "_all_gather_into_tensor",
    "_reduce_scatter_tensor",
    "_all_to_all_single",
    "_wait_tensor",
    "_isend",
    "_irecv",
    "_batch_p2p_ops",
]


def _group_from_ranks(ranks: list[int], tag: str | int) -> Any:
    resolver = getattr(dist, "_resolve_group_name_by_ranks_and_tag", None)
    if resolver is not None:
        return resolver(ranks, tag)
    if not dist.is_initialized():
        return None
    current = dist.get_process_group_ranks()
    if list(current) == list(ranks):
        return dist._get_default_group()
    return dist.new_group(ranks=list(ranks))


def _broadcast(input: Any, src: int, tag: str, ranks: list[int], group_size: int) -> Any:
    del group_size
    return functional.broadcast(input, src=src, group=_group_from_ranks(ranks, tag))


def _all_reduce(input: Any, reduce_op: str, tag: str, ranks: list[int], group_size: int) -> Any:
    del group_size
    return functional.all_reduce(input, reduce_op=reduce_op, group=_group_from_ranks(ranks, tag))


def _all_reduce_coalesced(inputs: list[Any], reduce_op: str, tag: str, ranks: list[int], group_size: int) -> Any:
    del group_size
    return functional.all_reduce_coalesced(inputs, reduce_op=reduce_op, group=_group_from_ranks(ranks, tag))


def _all_gather_into_tensor(input: Any, tag: str, ranks: list[int], group_size: int) -> Any:
    return functional.all_gather_tensor(input, gather_dim=0, group=_group_from_ranks(ranks, tag))


def _reduce_scatter_tensor(input: Any, reduce_op: str, tag: str, ranks: list[int], group_size: int) -> Any:
    return functional.reduce_scatter_tensor(input, reduce_op=reduce_op, group=_group_from_ranks(ranks, tag), scatter_dim=0)


def _all_to_all_single(
    input: Any,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    tag: str,
    ranks: list[int],
    group_size: int,
) -> Any:
    if output_split_sizes is None or input_split_sizes is None:
        if output_split_sizes is not None or input_split_sizes is not None:
            raise AssertionError("split sizes must be supplied together")
        if input.shape[0] % group_size:
            raise ValueError("the leading dimension must divide evenly")
        output_split_sizes = [input.shape[0] // group_size] * group_size
        input_split_sizes = list(output_split_sizes)
    group = _group_from_ranks(ranks, tag)
    return functional.all_to_all_single(
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )


def _wait_tensor(tensor: Any) -> Any:
    return functional.wait_tensor(tensor)


def _isend(tensor: Any, dst: int, tag: str, group_name: Any) -> Any:
    return dist.isend(tensor, dst=dst, group=group_name, tag=int(tag) if str(tag).isdigit() else 0)


def _irecv(tensor: Any, src: int, tag: str, group_name: Any) -> Any:
    return dist.irecv(tensor, src=src, group=group_name, tag=int(tag) if str(tag).isdigit() else 0)


def _batch_p2p_ops(op_list: list[str], peer_list: list[int], tag_list: list[int], tensors: list[Any], group_name: Any) -> Any:
    operations = []
    for op, peer, tag, tensor in zip(op_list, peer_list, tag_list, tensors):
        operations.append(dist.P2POp(op, tensor, peer, group_name, tag))
    return dist.batch_isend_irecv(operations)
