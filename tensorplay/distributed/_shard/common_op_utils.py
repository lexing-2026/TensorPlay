"""Operation registration helpers for sharded values."""

from typing import Any, Callable

__all__ = ["_basic_validation", "_register_default_op"]


def _basic_validation(types: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    del types, kwargs
    if not args:
        raise ValueError("a sharded operation requires an input")


def _register_default_op(op: Any, op_impl: Callable[..., Any]) -> Callable[..., Any]:
    from .sharded_tensor.api import _SHARDED_OPS
    _SHARDED_OPS[op] = op_impl
    return op_impl
