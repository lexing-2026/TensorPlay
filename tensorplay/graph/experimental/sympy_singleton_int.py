"""An integer that is one run of memory, rather than a number.

Some extents are known to be a single run of memory before anything is
allocated: a row of a tensor laid out with one row after another, a window
whose stride was worked out, a size that came from a stride.  Such an extent is
an address difference, and the arithmetic on it is linear with a coefficient
that is a power of two, not the general integer arithmetic a symbolic number
gets.

The distinction matters where the two meet.  Two of these are equal exactly
when their runs are the same length, which is a question about the run and not
a question about arithmetic, so it has an answer.  But one of these is less
than an ordinary number only when the run is short enough that the answer is
known without knowing the extents, and a run whose length is not settled has no
ordering against a plain number at all -- so that comparison raises rather than
answering.  Answering it anyway would let an unsound layout be chosen.
"""

from __future__ import annotations

from typing import Any

import sympy
from sympy.multipledispatch import dispatch

__all__ = ["SingletonInt"]


class SingletonInt(sympy.AtomicExpr):
    """One run of memory, of a length that is settled but not written down.

    Carries the length as an ordinary value and a coefficient it is scaled by,
    so that a run of ``coeff * val`` elements is expressible without a
    multiplication node, and so that equality can compare the two directly
    instead of comparing expression trees that happen to be equal.
    """

    _op_priority = 99999
    _val: int
    _coeff: int

    def __new__(
        cls, *args: Any, coeff: int | None = None, **kwargs: Any
    ) -> "SingletonInt":
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, val: int, *, coeff: int = 1) -> None:
        self._val = val
        self._coeff = coeff
        super().__init__()

    def _eval_Eq(self, other: sympy.Basic) -> Any:
        """Equality is a comparison of runs, not of expression trees."""

        if (
            isinstance(other, SingletonInt)
            and other._val == self._val
            and self._coeff == other._coeff
        ):
            return sympy.true
        return sympy.false

    @property
    def free_symbols(self) -> set[sympy.Symbol]:
        """Empty, so asking a run which symbols it depends on is answerable.

        A run's length is settled, so a run does not depend on anything, and
        the question has to have an answer for the operations that ask it of
        every expression they are handed.
        """

        return set()

    def __mul__(self, other: int) -> "SingletonInt":
        """Scaling a run gives a run of the scaled length."""

        if isinstance(other, SingletonInt):
            raise ValueError("a run cannot be scaled by another run")
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __rmul__(self, other: int) -> "SingletonInt":
        if isinstance(other, SingletonInt):
            raise ValueError("a run cannot be scaled by another run")
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __add__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("a run's length is not a number to add to")

    def __sub__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("a run's length is not a number to subtract")

    def __truediv__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("a run's length is not a number to divide")

    def __floordiv__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("a run's length is not a number to divide")

    def __mod__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("a run's length is not a number to take a remainder of")


@dispatch(sympy.Integer, SingletonInt)
def _eval_is_ge(a: sympy.Integer, b: SingletonInt) -> Any:
    """Whether a plain number is at least a run.

    Answerable only while the run is short enough that its length is below the
    number, whatever the number is.  A longer run is not less than a small
    number, and not known to be more either, so the comparison raises instead
    of picking one.
    """

    if a < 2:
        return sympy.false
    raise ValueError("a run's length is not settled: the comparison has no answer")


@dispatch(SingletonInt, sympy.Integer)  # type: ignore[no-redef]
def _eval_is_ge(a: SingletonInt, b: sympy.Integer) -> Any:
    """Whether a run is at least a plain number."""

    if b <= 2:
        return sympy.true
    raise ValueError("a run's length is not settled: the comparison has no answer")


@dispatch(SingletonInt, SingletonInt)  # type: ignore[no-redef]
def _eval_is_ge(a: SingletonInt, b: SingletonInt) -> Any:
    """Whether one run is at least another, which their lengths decide."""

    if a._val == b._val:
        return sympy.true if a._coeff >= b._coeff else sympy.false
    raise ValueError("a run's length is not settled: the comparison has no answer")
