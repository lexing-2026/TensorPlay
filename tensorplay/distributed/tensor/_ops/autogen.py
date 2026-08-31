"""Registration support for generated operation variants."""

from __future__ import annotations

from typing import Any, Callable

from .utils import register_op_strategy

__all__ = ["auto_register_op_variants"]


def auto_register_op_variants(operations: Any = ()) -> dict[Any, Callable[..., Any]]:
    registered: dict[Any, Callable[..., Any]] = {}
    for operation in operations:
        def strategy(schema: Any, _operation=operation) -> Any:
            del _operation
            return schema

        registered[operation] = register_op_strategy(operation)(strategy)
    return registered
