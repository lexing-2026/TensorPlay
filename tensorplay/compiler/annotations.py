"""Capture-state helpers.

These helpers describe or influence graph capture itself.  Capture runs
before any backend is selected, so none of them depend on, or reveal, which
backend will lower the captured graph.
"""

from __future__ import annotations

import builtins
import typing as _typing
from typing import Any


def annotate(annotation: Any, value: Any) -> Any:
    """Attach an annotation to a symbolic value and return the value."""

    from tensorplay.graph import Proxy
    from tensorplay.graph.annotate import annotate as annotate_graph_value

    if builtins.isinstance(value, Proxy):
        return annotate_graph_value(value, annotation)
    return value


def isinstance(value: Any, *types: Any) -> bool:
    """Test values against runtime types and parameterized containers."""

    if len(types) != 1:
        raise TypeError(
            "isinstance() takes exactly one type argument, got {}".format(len(types))
        )
    type_spec = types[0]
    origin = _typing.get_origin(type_spec)
    if origin is _typing.Union:
        members = list(_typing.get_args(type_spec))
        if type(None) in members:
            if value is None:
                return True
            members = [member for member in members if member is not type(None)]
            if not members:
                return False
            if len(members) == 1:
                return isinstance(value, members[0])
            return any(isinstance(value, member) for member in members)
        return any(isinstance(value, member) for member in members)
    if origin is not None:
        base = {
            list: list,
            set: set,
            frozenset: frozenset,
            dict: dict,
            tuple: tuple,
        }.get(origin, origin)
        try:
            return builtins.isinstance(value, base)
        except TypeError:
            return False
    if builtins.isinstance(type_spec, type):
        return builtins.isinstance(value, type_spec)
    return type_spec is _typing.Any or value is type_spec


def Final(value: Any) -> Any:
    """Marker the tracer treats as a frozen, non-reconstructible value."""

    return value


__all__ = ["Final", "annotate", "isinstance"]
