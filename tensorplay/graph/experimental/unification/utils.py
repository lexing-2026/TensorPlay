from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any, TypeVar

from .variable import Var

_T = TypeVar("_T")

__all__ = [
    "freeze",
    "hashable",
    "raises",
    "reverse_dict",
    "transitive_get",
    "xfail",
    "_toposort",
]


def hashable(value: Any) -> bool:
    try:
        hash(value)
        return True
    except TypeError:
        return False


def transitive_get(key: Any, mapping: dict[Var, Any]) -> Any:
    seen: set[Any] = set()
    while hashable(key) and key in mapping and key not in seen:
        seen.add(key)
        key = mapping[key]
    return key


def freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((freeze(key), freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze(item) for item in value)
    return value


def raises(error: type[BaseException], function: Callable[[], object]) -> bool:
    try:
        function()
    except error:
        return True
    return False


def xfail(function: Callable[[], object]) -> None:
    try:
        function()
    except Exception:
        return
    raise RuntimeError("expected operation to fail")


def reverse_dict(mapping: Mapping[_T, Iterable[Any]]) -> OrderedDict[Any, tuple[_T, ...]]:
    result: OrderedDict[Any, tuple[_T, ...]] = OrderedDict()
    for key, values in mapping.items():
        for value in values:
            result[value] = result.get(value, ()) + (key,)
    return result


def _toposort(edges: Mapping[_T, Iterable[_T]]) -> list[_T]:
    incoming = reverse_dict(edges)
    pending = OrderedDict((key, None) for key in edges if key not in incoming)
    result: list[_T] = []
    while pending:
        key, _ = pending.popitem(last=False)
        result.append(key)
        for child in edges.get(key, ()):
            parents = set(incoming[child])
            parents.discard(key)
            incoming[child] = tuple(parents)
            if not parents:
                pending[child] = None
    if any(incoming.get(key) for key in edges):
        raise ValueError("input contains a cycle")
    return result
