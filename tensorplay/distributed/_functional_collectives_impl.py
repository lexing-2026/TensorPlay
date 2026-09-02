from __future__ import annotations

from typing import Any

import tensorplay as tp
import tensorplay.distributed as dist

from . import _functional_collectives as functional
from . import distributed_core as _core

__all__ = [
    "_broadcast",
    "_all_reduce",
    "_all_reduce_coalesced",
    "_all_gather_into_tensor",
    "_all_gather_into_tensor_coalesced",
    "_reduce_scatter_tensor",
    "_reduce_scatter_tensor_coalesced",
    "_all_to_all_single",
    "_wait_tensor",
    "_isend",
    "_irecv",
    "_batch_p2p_ops",
]


_REDUCE_OPS = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "product": dist.ReduceOp.PRODUCT,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}
_REDUCE_OP_NAMES = {value: name for name, value in _REDUCE_OPS.items()}
_P2P_OPS = {
    "isend": dist.isend,
    "irecv": dist.irecv,
}


def _group_from_ranks(ranks: list[int], tag: str | int) -> Any:
    del tag
    if not ranks:
        raise ValueError("ranks must not be empty")
    rankset = tuple(int(rank) for rank in ranks)
    if len(rankset) != len(set(rankset)):
        raise ValueError("ranks must be unique")
    if not dist.is_initialized():
        raise RuntimeError(
            "Default process group has not been initialized"
        )

    default_group = dist._get_default_group()
    canonical = tuple(sorted(rankset))
    if tuple(sorted(default_group.ranks)) == canonical:
        return default_group

    for group in _core._groups.values():
        if tuple(sorted(group.ranks)) == canonical:
            return group
    group = dist.new_group(ranks=list(canonical))
    if group is dist.GroupMember.NON_GROUP_MEMBER:
        raise ValueError("current rank is not in the requested process group")
    return group


def _collective_group(
    ranks: list[int], tag: str | int, group_size: int
) -> Any:
    group = _group_from_ranks(ranks, tag)
    if group.size() != int(group_size):
        raise ValueError(
            f"group_size ({group_size}) does not match process group size "
            f"({group.size()})"
        )
    return group


def _reduce_op_name(reduce_op: Any) -> str:
    if isinstance(reduce_op, str):
        name = reduce_op.lower()
        if name in _REDUCE_OPS:
            return name
    elif isinstance(reduce_op, int) and reduce_op in _REDUCE_OP_NAMES:
        return _REDUCE_OP_NAMES[reduce_op]
    operation = getattr(reduce_op, "op", None)
    if callable(operation):
        return _reduce_op_name(operation())
    raise ValueError(f"Unsupported reduction operation: {reduce_op!r}")


def _reduce_op_value(reduce_op: Any) -> int:
    return int(_REDUCE_OPS[_reduce_op_name(reduce_op)])


def _all_gather_output(input: Any, group_size: int) -> Any:
    shape = list(input.shape)
    if shape:
        shape[0] *= group_size
    else:
        shape.append(group_size)
    return tp.empty(shape, dtype=input.dtype, device=input.device)


def _reduce_scatter_output(input: Any, group_size: int) -> Any:
    shape = list(input.shape)
    if not shape:
        raise ValueError(
            "reduce_scatter_tensor requires an input with a leading dimension"
        )
    if shape[0] % group_size:
        raise ValueError(
            "the leading dimension must divide evenly across the group"
        )
    shape[0] //= group_size
    return tp.empty(shape, dtype=input.dtype, device=input.device)


def _contiguous(input: Any) -> Any:
    if input.is_contiguous():
        return input
    output = tp.empty(input.shape, dtype=input.dtype, device=input.device)
    output.copy_(input)
    return output


def _all_gather_views(output: Any, input: Any, group_size: int) -> list[Any]:
    shape = tuple(input.shape)
    if shape:
        chunk = int(shape[0])
        return [
            output.narrow(0, rank * chunk, chunk).reshape(shape)
            for rank in range(group_size)
        ]
    return [
        output.narrow(0, rank, 1).reshape(shape)
        for rank in range(group_size)
    ]


def _reduce_scatter_views(input: Any, group_size: int) -> list[Any]:
    shape = tuple(input.shape)
    if not shape:
        raise ValueError(
            "reduce_scatter_tensor requires an input with a leading dimension"
        )
    chunk = int(shape[0]) // group_size
    return [
        input.narrow(0, rank * chunk, chunk).reshape(
            (chunk,) + shape[1:]
        )
        for rank in range(group_size)
    ]


def _broadcast(input: Any, src: int, tag: str, ranks: list[int], group_size: int) -> Any:
    group = _collective_group(ranks, tag, group_size)
    return functional.broadcast(input, src=src, group=group)


def _all_reduce(
    input: Any, reduce_op: str, tag: str, ranks: list[int], group_size: int
) -> Any:
    group = _collective_group(ranks, tag, group_size)
    return functional.all_reduce(
        input, reduce_op=_reduce_op_name(reduce_op), group=group
    )


def _all_reduce_coalesced(
    inputs: list[Any],
    reduce_op: str,
    tag: str,
    ranks: list[int],
    group_size: int,
) -> Any:
    group = _collective_group(ranks, tag, group_size)
    return functional.all_reduce_coalesced(
        inputs, reduce_op=_reduce_op_name(reduce_op), group=group
    )


def _all_gather_into_tensor(
    input: Any, tag: str, ranks: list[int], group_size: int
) -> Any:
    group = _collective_group(ranks, tag, group_size)
    return functional.all_gather_single(input, gather_dim=0, group=group,
                                        tag=tag)


def _all_gather_into_tensor_coalesced(
    inputs: list[Any], tag: str, ranks: list[int], group_size: int
) -> Any:
    group = _collective_group(ranks, tag, group_size)
    return functional.all_gather_single_coalesced(inputs, group=group, tag=tag)


def _reduce_scatter_tensor(
    input: Any,
    reduce_op: str,
    tag: str,
    ranks: list[int],
    group_size: int,
) -> Any:
    group = _collective_group(ranks, tag, group_size)
    return functional.reduce_scatter_single(
        input, reduce_op=reduce_op, group=group, scatter_dim=0, tag=tag
    )


def _reduce_scatter_tensor_coalesced(
    inputs: list[Any],
    reduce_op: str,
    tag: str,
    ranks: list[int],
    group_size: int,
) -> Any:
    group = _collective_group(ranks, tag, group_size)
    return functional.reduce_scatter_single_coalesced(
        inputs, reduce_op, [0] * len(inputs), group=group, tag=tag
    )


def _all_to_all_single(
    input: Any,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    tag: str,
    ranks: list[int],
    group_size: int,
) -> Any:
    group = _collective_group(ranks, tag, group_size)
    return functional._all_to_all_single_native(
        input, output_split_sizes, input_split_sizes, group
    )


def _wait_tensor(tensor: Any) -> Any:
    return functional.wait_tensor(tensor)


def _tag_value(tag: Any) -> int:
    if isinstance(tag, bool):
        raise ValueError("tag must be an integer")
    if isinstance(tag, int):
        return tag
    if isinstance(tag, str) and tag.isdigit():
        return int(tag)
    raise ValueError(f"tag must be an integer, got {tag!r}")


def _p2p_group(group_name: Any) -> Any:
    return _core._resolve_group(group_name)


def _isend(tensor: Any, dst: int, tag: str, group_name: Any) -> Any:
    group = _p2p_group(group_name)
    return functional.isend_inplace(
        tensor, group=group, group_dst=int(dst), tag=_tag_value(tag)
    )


def _irecv(tensor: Any, src: int, tag: str, group_name: Any) -> Any:
    group = _p2p_group(group_name)
    return functional.irecv_inplace(
        tensor, group=group, group_src=int(src), tag=_tag_value(tag)
    )


def _batch_p2p_ops(
    op_list: list[str],
    peer_list: list[int],
    tag_list: list[int],
    tensors: list[Any],
    group_name: Any,
) -> Any:
    if not (
        len(op_list) == len(peer_list)
        == len(tag_list)
        == len(tensors)
    ):
        raise ValueError(
            "op_list, peer_list, tag_list, and tensors must have equal lengths"
        )
    if not op_list:
        return []

    return functional.batch_p2p_ops_inplace(
        op_list, peer_list, tag_list, tensors, group_name
    )
