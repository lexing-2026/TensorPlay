"""Eager-execution helpers for the JIT-compatible namespace.

Model code can use these helpers to guard eager paths. TensorPlay has no
``is_scripting()`` always returns False — exactly the semantics of an eager
execution environment.
"""

import builtins
import typing as _typing

from typing import Callable, TypeVar

T = TypeVar("T")


def is_scripting() -> bool:
    return False


def is_tracing() -> bool:
    return False


def is_exporting() -> bool:
    return False


def is_importing() -> bool:
    return False


def unused(fn: T) -> T:
    return fn


def script(fn=None, *args, **kwargs):
    """No-op: returns the function unchanged (eager execution)."""
    if fn is None:
        return lambda f: f
    return fn


def ignore(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


def _overload_method(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


def export(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


def interface(fn: T) -> T:
    return fn


def Final(value):
    return value


def isinstance(x, *args):
    """

    container generics (List[int], Dict[str, Tensor], Tuple[T, ...]).
    """
    if len(args) != 1:
        raise TypeError(
            "isinstance() takes exactly one type argument, got {}".format(len(args)))
    t = args[0]
    origin = _typing.get_origin(t)
    if origin is _typing.Union:
        members = list(_typing.get_args(t))
        if type(None) in members:
            if x is None:
                return True
            members = [m for m in members if m is not type(None)]
            if not members:
                return False
            rest = members[0] if len(members) == 1 else _typing.Union[tuple(members)]
            return isinstance(x, rest)
        return any(isinstance(x, m) for m in members)
    if origin is not None:
        base = {list: list, set: set, frozenset: frozenset,
                dict: dict, tuple: tuple}.get(origin, origin)
        try:
            return builtins.isinstance(x, base)
        except TypeError:
            return False
    if builtins.isinstance(t, type):
        return builtins.isinstance(x, t)
    # Non-type annotations without an origin (e.g. typing.Any) accept anything.
    return t is _typing.Any or x is t
