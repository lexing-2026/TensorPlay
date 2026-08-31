"""Registry for eager placement propagation rules."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["ShardingPropagator"]


class ShardingPropagator:
    def __init__(self) -> None:
        self._rules: dict[Any, Callable[..., Any]] = {}

    def register_op_strategy(self, operation: Any, rule: Callable[..., Any]) -> Callable[..., Any]:
        self._rules[operation] = rule
        return rule

    def propagate_op_sharding(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        rule = self._rules.get(operation)
        if rule is None:
            return None
        return rule(*args, **kwargs)

    def clear(self) -> None:
        self._rules.clear()
