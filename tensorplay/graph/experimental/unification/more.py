from __future__ import annotations

from typing import Any

from .core import reify, unify

__all__ = ["reify_object", "unifiable", "unify_object"]


def unifiable(cls: type) -> type:
    cls.__unifiable__ = True
    return cls


def reify_object(value: Any, substitutions: dict[Any, Any]) -> Any:
    if not getattr(value, "__unifiable__", False):
        return value
    clone = object.__new__(type(value))
    for name, item in vars(value).items():
        setattr(clone, name, reify(item, substitutions))
    return clone


def unify_object(left: Any, right: Any, substitutions: dict[Any, Any]) -> dict[Any, Any] | bool:
    if type(left) is not type(right):
        return False
    return unify(vars(left), vars(right), substitutions)

