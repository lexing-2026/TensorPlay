"""Gradual tensor shape annotations used by tracing and analysis passes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._compatibility import compatibility

__all__ = ["Dyn", "TensorType", "is_consistent", "is_more_precise"]


@compatibility(is_backward_compatible=False)
class TensorType:
    """Describe tensor rank and dimensions, allowing unknown dimensions."""

    def __init__(self, dim: Sequence[Any]) -> None:
        self.__origin__ = TensorType
        self.__args__ = tuple(dim)

    @property
    def dims(self) -> tuple[Any, ...]:
        return self.__args__

    def __repr__(self) -> str:
        return f"TensorType[{self.dims!r}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TensorType) and self.dims == other.dims

    def __hash__(self) -> int:
        return hash((TensorType, self.dims))

    @staticmethod
    def __class_getitem__(*args: object) -> "TensorType":
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return TensorType(args)


class _DynType:
    __name__ = "_DynType"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DynType)

    def __hash__(self) -> int:
        return hash(_DynType)

    def __str__(self) -> str:
        return "Dyn"

    def __repr__(self) -> str:
        return "Dyn"


Dyn = _DynType()


def _is_variable(value: object) -> bool:
    return bool(getattr(value, "is_type_variable", False))


@compatibility(is_backward_compatible=False)
def is_consistent(t1: object, t2: object) -> bool:
    """Return whether two gradual types can describe the same value."""

    if t1 == t2 or t1 is Dyn or t2 is Dyn:
        return True
    if _is_variable(t1) or _is_variable(t2):
        return True
    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.dims) == len(t2.dims) and all(
            is_consistent(left, right) for left, right in zip(t1.dims, t2.dims)
        )
    return False


@compatibility(is_backward_compatible=False)
def is_more_precise(t1: object, t2: object) -> bool:
    """Return whether ``t1`` contains at least as much shape information as ``t2``."""

    if t1 == t2:
        return True
    if t2 is Dyn:
        return True
    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.dims) == len(t2.dims) and all(
            is_more_precise(left, right) for left, right in zip(t1.dims, t2.dims)
        )
    return False
