from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from .variable import Var, isvar

__all__ = ["reify", "unify"]


def _walk(value: Any, substitutions: dict[Var, Any]) -> Any:
    while isvar(value) and value in substitutions and substitutions[value] is not value:
        value = substitutions[value]
    return value


def reify(value: Any, substitutions: dict[Var, Any]) -> Any:
    value = _walk(value, substitutions)
    if isvar(value):
        return value
    if isinstance(value, tuple):
        return tuple(reify(item, substitutions) for item in value)
    if isinstance(value, list):
        return [reify(item, substitutions) for item in value]
    if isinstance(value, dict):
        return {key: reify(item, substitutions) for key, item in value.items()}
    if isinstance(value, slice):
        return slice(
            reify(value.start, substitutions),
            reify(value.stop, substitutions),
            reify(value.step, substitutions),
        )
    if hasattr(value, "__dict__") and getattr(value, "__unifiable__", False):
        clone = object.__new__(type(value))
        for name, item in vars(value).items():
            setattr(clone, name, reify(item, substitutions))
        return clone
    return value


def _occurs(variable: Var, value: Any, substitutions: dict[Var, Any]) -> bool:
    value = _walk(value, substitutions)
    if variable == value:
        return True
    if isinstance(value, (tuple, list)):
        return any(_occurs(variable, item, substitutions) for item in value)
    if isinstance(value, dict):
        return any(_occurs(variable, item, substitutions) for item in value.values())
    if isinstance(value, slice):
        return any(
            _occurs(variable, item, substitutions)
            for item in (value.start, value.stop, value.step)
        )
    return False


def _unify(left: Any, right: Any, substitutions: dict[Var, Any]) -> dict[Var, Any] | bool:
    left = _walk(left, substitutions)
    right = _walk(right, substitutions)
    if left == right:
        return substitutions
    if isvar(left):
        if _occurs(left, right, substitutions):
            return False
        substitutions[left] = right
        return substitutions
    if isvar(right):
        if _occurs(right, left, substitutions):
            return False
        substitutions[right] = left
        return substitutions
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if left.keys() != right.keys():
            return False
        for key in left:
            if _unify(left[key], right[key], substitutions) is False:
                return False
        return substitutions
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return False
        for first, second in zip(left, right):
            if _unify(first, second, substitutions) is False:
                return False
        return substitutions
    if isinstance(left, slice):
        return _unify(
            (left.start, left.stop, left.step),
            (right.start, right.stop, right.step),
            substitutions,
        )
    if getattr(left, "__unifiable__", False) and vars(left) and vars(right):
        return _unify(vars(left), vars(right), substitutions)
    return substitutions if left == right else False


def unify(left: Any, right: Any, substitutions: dict[Var, Any] | None = None) -> dict[Var, Any] | bool:
    return _unify(left, right, substitutions if substitutions is not None else {})

