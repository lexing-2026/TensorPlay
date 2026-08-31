from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

__all__ = ["Var", "isvar", "var", "variables", "vars"]

_global_variables: set[Any] = set()


class Var:
    """A hashable logic variable used by shape refinement."""

    _counter = 0
    is_type_variable = True

    def __init__(self, *token: Any) -> None:
        if not token:
            type(self)._counter += 1
            token = (f"_{type(self)._counter}",)
        self.token = token[0] if len(token) == 1 else token

    def __str__(self) -> str:
        return "~" + str(self.token)

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Var) and self.token == other.token

    def __hash__(self) -> int:
        return hash((type(self), self.token))


def var(*tokens: Any) -> Var:
    return Var(*tokens)


def vars(count: int) -> list[Var]:
    if count < 0:
        raise ValueError("count must be non-negative")
    return [Var() for _ in range(count)]


def isvar(value: Any) -> bool:
    if isinstance(value, Var):
        return True
    try:
        return value in _global_variables
    except TypeError:
        return False


@contextmanager
def variables(*values: Any) -> Iterator[None]:
    old = set(_global_variables)
    _global_variables.update(values)
    try:
        yield
    finally:
        _global_variables.clear()
        _global_variables.update(old)
