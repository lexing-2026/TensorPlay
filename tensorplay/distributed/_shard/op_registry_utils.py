"""Registry decorators for sharded operations."""

from typing import Any, Callable

__all__ = ["_register_op", "_decorator_func"]


def _register_op(op: Any, op_table: dict[Any, Callable[..., Any]]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        op_table[op] = fn
        return fn
    return decorator


def _decorator_func(fn: Callable[..., Any], op: Any, op_table: dict[Any, Callable[..., Any]]) -> Callable[..., Any]:
    op_table[op] = fn
    return fn
