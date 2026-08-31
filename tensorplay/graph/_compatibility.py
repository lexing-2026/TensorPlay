"""Decorators for recording the stability level of graph APIs."""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T")

_BACK_COMPAT_OBJECTS: set[Any] = set()
_MARKED_WITH_COMPATIBILITY: set[Any] = set()


def compatibility(is_backward_compatible: bool) -> Callable[[_T], _T]:
    """Mark an API and retain the marker on the decorated object."""

    def mark(obj: _T) -> _T:
        _MARKED_WITH_COMPATIBILITY.add(obj)
        if is_backward_compatible:
            _BACK_COMPAT_OBJECTS.add(obj)
        doc = textwrap.dedent(getattr(obj, "__doc__", None) or "").rstrip()
        note = (
            "This API is stable for existing graph programs."
            if is_backward_compatible
            else "This API is experimental and may change."
        )
        try:
            obj.__doc__ = f"{doc}\n\n{note}\n"  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        try:
            setattr(obj, "__tensorplay_backward_compatible__", is_backward_compatible)
        except (AttributeError, TypeError):
            pass
        return obj

    return mark


__all__ = ["compatibility"]
