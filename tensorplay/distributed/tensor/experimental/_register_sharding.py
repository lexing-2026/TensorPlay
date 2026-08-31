"""User registration for custom placement rules."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["register_sharding"]

_SHARDING_RULES: dict[Any, Callable[..., Any]] = {}


def register_sharding(operation: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    operations = tuple(operation) if isinstance(operation, (list, tuple)) else (operation,)

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        for item in operations:
            _SHARDING_RULES[item] = function
        return function

    return decorator
