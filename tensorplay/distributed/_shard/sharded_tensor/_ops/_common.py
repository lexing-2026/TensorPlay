"""Shared sharded operation dispatch utilities."""

from typing import Any, Callable

__all__ = ["_sharded_op_common", "_register_sharded_op_on_local_shards"]


def _sharded_op_common(op: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return op(*[_local(value) for value in args], **{key: _local(value) for key, value in kwargs.items()})


def _register_sharded_op_on_local_shards(op: Any, op_impl: Callable[..., Any]) -> Callable[..., Any]:
    from ..api import _SHARDED_OPS
    _SHARDED_OPS[op] = op_impl
    return op_impl


def _local(value: Any) -> Any:
    return value.to_local() if hasattr(value, "to_local") else value
