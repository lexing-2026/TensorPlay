from __future__ import annotations

from typing import Any

__all__ = ["ConstantIntNode"]


class ConstantIntNode:
    """Integer scalar node used when a value must expose symbolic operations."""

    __slots__ = ("val",)

    def __init__(self, val: int) -> None:
        if type(val) is not int:
            raise TypeError(f"expected int, got {type(val).__name__}")
        self.val = val

    def is_constant(self) -> bool:
        return True

    def maybe_as_int(self) -> int:
        return self.val

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return False

    def is_symbolic(self) -> bool:
        return False

    def clone(self) -> "ConstantIntNode":
        return self

    def constant_int(self) -> int:
        return self.val

    def _str(self) -> str:
        return str(self.val)

    def __str__(self) -> str:
        return self._str()

    def __repr__(self) -> str:
        return self._str()

    def _graph_repr(self) -> str:
        return self._str()

    def add(self, other: Any) -> Any:
        return other.add(self)

    def sub(self, other: Any) -> Any:
        return other.neg().add(self.val)

    def mul(self, other: Any) -> Any:
        return other.mul(self)

    def eq(self, other: Any) -> Any:
        return other.eq(self)

    def ne(self, other: Any) -> Any:
        return other.ne(self)

    def gt(self, other: Any) -> Any:
        return other.lt(self)

    def lt(self, other: Any) -> Any:
        return other.gt(self)

    def le(self, other: Any) -> Any:
        return other.ge(self)

    def ge(self, other: Any) -> Any:
        return other.le(self)

    def guard_int(self, *_: Any) -> int:
        return self.val
