from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import tensorplay as tp

__all__ = [
    "non_functional_collectives",
    "functional_collectives",
    "sync_ops",
    "collective_ops",
    "CollectiveOp",
]


def _name(value: Any) -> str:
    return str(getattr(value, "__name__", getattr(value, "name", value))).lower()


non_functional_collectives: set[Any] = set()
functional_collectives: set[Any] = set()
sync_ops: set[Any] = set()
collective_ops: set[Any] = non_functional_collectives | functional_collectives


def _walk(value: Any):
    if isinstance(value, tp.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _walk(item)


def _tensor_bytes(value: Any) -> int:
    storage = getattr(value, "untyped_storage", None)
    if callable(storage):
        storage = storage()
        nbytes = getattr(storage, "nbytes", None)
        if callable(nbytes):
            return int(nbytes())
    nbytes = getattr(value, "nbytes", None)
    if callable(nbytes):
        return int(nbytes())
    element_size = getattr(value, "element_size", lambda: 1)
    return int(value.numel()) * int(element_size())


class CollectiveOp:
    """Memory and process-group helpers for collective operation records."""

    PG_ARG_1: set[Any] = set()
    PG_ARG_2: set[Any] = set()
    PG_ARG_3: set[Any] = set()
    PG_ARG_4: set[Any] = set()
    PG_ARG_5: set[Any] = set()
    WK_ARG_1: set[Any] = set()
    WK: set[Any] = set()
    COMM_TENSOR_ARG_0: set[Any] = set()
    COMM_TENSOR_ARG_1: set[Any] = set()
    COMM_TENSOR_ARG_RES: set[Any] = set()
    COMM_TENSOR_SINGLE_UNTYPED_STORAGE: set[Any] = set()
    COMM_TENSOR_ARG_0_AND_RES: set[Any] = set()
    COMM_TENSOR_RES_SUM: set[Any] = set()

    @staticmethod
    def sum_tensors(arg: Any) -> int:
        return sum(_tensor_bytes(value) for value in _walk(arg))

    @staticmethod
    def get_process_group(func: Any, args: Sequence[Any]) -> Any:
        for value in reversed(args):
            if hasattr(value, "allreduce") or hasattr(value, "size") and hasattr(value, "rank"):
                return value
        for value in args:
            if hasattr(value, "group"):
                return value.group
        raise TypeError(f"no process group found for {func!r}")

    @staticmethod
    def get_comm_tensor_size(func: Any, res: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> int:
        del kwargs
        name = _name(func)
        if "barrier" in name or "wait" in name:
            return 0
        if "gather" in name or "all_gather" in name:
            return max(CollectiveOp.sum_tensors(args), CollectiveOp.sum_tensors(res))
        if "all_to_all" in name or "alltoall" in name:
            return max(CollectiveOp.sum_tensors(args), CollectiveOp.sum_tensors(res))
        return CollectiveOp.sum_tensors(args) or CollectiveOp.sum_tensors(res)

    @staticmethod
    def get_work(func: Any, res: Any) -> Any:
        if res is not None and callable(getattr(res, "wait", None)):
            return res
        if "wait" in _name(func):
            return res
        return None
