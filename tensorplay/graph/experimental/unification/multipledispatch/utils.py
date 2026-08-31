from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar

_T = TypeVar("_T")

__all__ = ["_toposort", "expand_tuples", "groupby", "raises", "reverse_dict", "typename"]


def raises(error: type[BaseException], function: Callable[[], object]) -> bool:
    try:
        function()
    except error:
        return True
    return False


def expand_tuples(values: Sequence[type | tuple[type, ...]]) -> list[tuple[type, ...]]:
    if not values:
        return [()]
    rest = expand_tuples(values[1:])
    head = values[0] if isinstance(values[0], tuple) else (values[0],)
    return [(item, *tail) for item in head for tail in rest]


def reverse_dict(mapping: Mapping[_T, Iterable[_T]]) -> OrderedDict[_T, tuple[_T, ...]]:
    result: OrderedDict[_T, tuple[_T, ...]] = OrderedDict()
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
            remaining = set(incoming[child])
            remaining.discard(key)
            incoming[child] = tuple(remaining)
            if not remaining:
                pending[child] = None
    if any(incoming.get(key) for key in edges):
        raise ValueError("input contains a cycle")
    return result


def groupby(function: Callable[[_T], object], values: Iterable[_T]) -> OrderedDict[object, list[_T]]:
    result: OrderedDict[object, list[_T]] = OrderedDict()
    for value in values:
        result.setdefault(function(value), []).append(value)
    return result


def typename(value: type | tuple[type, ...]) -> str:
    if isinstance(value, type):
        return value.__name__
    if len(value) == 1:
        return typename(value[0])
    return "(" + ", ".join(typename(item) for item in value) + ")"
