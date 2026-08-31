"""Operation dispatch helpers for distributed tensor wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from ._api import DTensor

__all__ = ["OpDispatcher", "unwrap_dtensor", "wrap_dtensor"]


def unwrap_dtensor(value: Any) -> Any:
    if isinstance(value, DTensor):
        return value.to_local()
    if isinstance(value, tuple):
        return tuple(unwrap_dtensor(item) for item in value)
    if isinstance(value, list):
        return [unwrap_dtensor(item) for item in value]
    if isinstance(value, dict):
        return {key: unwrap_dtensor(item) for key, item in value.items()}
    return value


def wrap_dtensor(value: Any, template: DTensor | None) -> Any:
    if template is None:
        return value
    if hasattr(value, "shape"):
        return DTensor(value, template.device_mesh, template.placements, shape=template.shape)
    if isinstance(value, tuple):
        return tuple(wrap_dtensor(item, template) for item in value)
    if isinstance(value, list):
        return [wrap_dtensor(item, template) for item in value]
    return value


class OpDispatcher:
    """Apply an eager operation to local values and preserve one layout."""

    def __init__(self) -> None:
        self._rules: dict[Any, Callable[..., Any]] = {}

    def register(self, operation: Any, rule: Callable[..., Any]) -> Callable[..., Any]:
        self._rules[operation] = rule
        return rule

    def __call__(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        distributed = next(
            (value for value in _walk((args, kwargs)) if isinstance(value, DTensor)),
            None,
        )
        rule = self._rules.get(operation, operation)
        result = rule(*unwrap_dtensor(args), **unwrap_dtensor(kwargs))
        return wrap_dtensor(result, distributed)


def _walk(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)
    else:
        yield value
