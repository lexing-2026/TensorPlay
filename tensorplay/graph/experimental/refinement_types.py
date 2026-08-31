from __future__ import annotations

__all__ = ["Equality"]


class Equality:
    """A small immutable-style record for a refinement equality."""

    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs: object, rhs: object) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def __repr__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Equality) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self) -> int:
        return hash((self.lhs, self.rhs))

