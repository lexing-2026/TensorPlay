from __future__ import annotations

import collections
import operator
from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from typing import Any, TypeVar

_T = TypeVar("_T")

__all__ = [
    "assoc",
    "assoc_in",
    "dissoc",
    "first",
    "get_in",
    "getter",
    "groupby",
    "itemfilter",
    "itemmap",
    "keyfilter",
    "keymap",
    "merge",
    "merge_with",
    "update_in",
    "valfilter",
    "valmap",
]


def merge(*dicts: Mapping[Any, Any], **kwargs: Any) -> dict[Any, Any]:
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = tuple(dicts[0])  # type: ignore[assignment]
    factory = kwargs.pop("factory", dict)
    if kwargs:
        raise TypeError(f"merge() got an unexpected keyword argument {next(iter(kwargs))!r}")
    result: dict[Any, Any] = factory()
    for mapping in dicts:
        result.update(mapping)
    result.update(kwargs)
    return result


def assoc(mapping: Mapping[Any, Any], key: Any, value: Any, **kwargs: Any) -> dict[Any, Any]:
    factory = kwargs.pop("factory", dict)
    result = factory(mapping)
    result[key] = value
    result.update(kwargs)
    return result


def dissoc(mapping: Mapping[Any, Any], *keys: Any, **kwargs: Any) -> dict[Any, Any]:
    factory = kwargs.pop("factory", dict)
    result = factory(mapping)
    for key in keys:
        result.pop(key, None)
    result.update(kwargs)
    return result


def valmap(function: Callable[[Any], Any], mapping: Mapping[Any, Any]) -> dict[Any, Any]:
    return {key: function(value) for key, value in mapping.items()}


def merge_with(function: Callable[[list[Any]], Any], *dicts: Mapping[Any, Any], **kwargs: Any) -> dict[Any, Any]:
    factory = kwargs.pop("factory", dict)
    if kwargs:
        raise TypeError(f"merge_with() got an unexpected keyword argument {next(iter(kwargs))!r}")
    values: dict[Any, list[Any]] = {}
    for mapping in dicts:
        for key, value in mapping.items():
            values.setdefault(key, []).append(value)
    return factory((key, function(items)) for key, items in values.items())


def keymap(function: Callable[[Any], Any], mapping: Mapping[Any, Any], factory: type = dict) -> dict[Any, Any]:
    return factory((function(key), value) for key, value in mapping.items())


def itemmap(function: Callable[[tuple[Any, Any]], Any], mapping: Mapping[Any, Any], factory: type = dict) -> dict[Any, Any]:
    return factory(function(item) for item in mapping.items())


def valfilter(function: Callable[[Any], bool], mapping: Mapping[Any, Any], factory: type = dict) -> dict[Any, Any]:
    return factory((key, value) for key, value in mapping.items() if function(value))


def keyfilter(function: Callable[[Any], bool], mapping: Mapping[Any, Any], factory: type = dict) -> dict[Any, Any]:
    return factory((key, value) for key, value in mapping.items() if function(key))


def itemfilter(function: Callable[[tuple[Any, Any]], bool], mapping: Mapping[Any, Any], factory: type = dict) -> dict[Any, Any]:
    return factory(item for item in mapping.items() if function(item))


def groupby(key: Callable[[_T], Any] | Any, values: Iterable[_T]) -> dict[Any, list[_T]]:
    if not callable(key):
        key = getter(key)
    result: dict[Any, list[_T]] = collections.defaultdict(list)
    for value in values:
        result.setdefault(key(value), []).append(value)
    return dict(result)


def first(values: Iterable[_T]) -> _T:
    return next(iter(values))


def assoc_in(mapping: Mapping[Any, Any], keys: Iterable[Any], value: Any, factory: type = dict) -> Any:
    return update_in(mapping, keys, lambda _old: value, value, factory)


def update_in(mapping: Mapping[Any, Any], keys: Iterable[Any], function: Callable[[Any], Any], default: Any = None, factory: type = dict) -> Any:
    path = list(keys)
    if not path:
        return function(mapping)
    result = factory(mapping)
    cursor = result
    original: Any = mapping
    for key in path[:-1]:
        existing = original.get(key, {}) if isinstance(original, Mapping) else {}
        child = factory(existing) if isinstance(existing, Mapping) else factory()
        cursor[key] = child
        cursor = child
        original = existing
    leaf = path[-1]
    previous = original.get(leaf, default) if isinstance(original, Mapping) else default
    cursor[leaf] = function(previous)
    return result


def getter(index: Any) -> Callable[[Any], Any]:
    if isinstance(index, list):
        if len(index) == 1:
            return lambda value: (value[index[0]],)
        if index:
            return operator.itemgetter(*index)
        return lambda _value: ()
    return operator.itemgetter(index)


def get_in(*args: Any, default: Any = None, no_default: bool = False) -> Any:
    if len(args) < 2:
        raise TypeError("get_in requires keys and collection")
    if isinstance(args[0], Mapping) and not isinstance(args[1], Mapping):
        collection, keys = args[0], args[1]
    else:
        keys, collection = args[0], args[1]
    if len(args) >= 3:
        default = args[2]
    try:
        return reduce(operator.getitem, keys, collection)
    except (KeyError, IndexError, TypeError):
        if no_default:
            raise
        return default
