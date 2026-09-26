"""The range of values an index expression can take, computed per operation.

A load's index is an expression over the region's extents, and the range that
expression can take is what says whether the load is inside the buffer without
a check being written for it.  This computes that range: the range of a symbol
comes from what the region knows about it, the range of an operation comes
from the ranges of its operands, and an operation whose range cannot be
computed is unbounded rather than wrong.

Only the operations that appear in an index are answered.  Everything else is
unbounded, because an operation nobody asked about does not need a range and
computing one anyway would cost more than it is worth.
"""

from __future__ import annotations

import operator
from typing import Any

import sympy

import tensorplay as tp

from tensorplay.graph.experimental.sympy_functions import (
    PowByNatural,
    SymPyValueRangeAnalysis,
    ValueRanges,
    int_oo,
)
from .ops_handler import DefaultHandler


class ValueRangeAnalysis(SymPyValueRangeAnalysis, DefaultHandler):
    """The range of each operation, given the ranges of its operands.

    An operation the analysis has no rule for is unbounded, which is the safe
    direction: a too-wide range costs a check in the generated code, while a
    too-narrow one would let a load read outside the buffer.
    """

    def __init__(self) -> None:
        self.name = "ValueRangeAnalysis"
        # A truth value can be either one, whatever its operands were, so these
        # four are answered directly rather than composed from their operands.
        boolean_operators = (
            "xor",
            "logical_and",
            "logical_or",
            "logical_not",
        )
        for op in boolean_operators:
            setattr(self, op, self.bool_handler)

    @staticmethod
    def bool_handler(*args: Any, **kwargs: Any):
        return ValueRanges(sympy.false, sympy.true)

    def _default(self, name, args, kwargs):
        return ValueRanges.unknown()

    def load(self, name: str, index):
        # What a buffer holds is not derivable from its extents, and a value
        # read out of one is therefore unbounded.
        return ValueRanges.unknown()

    def store(self, name: str, index, value, mode=None) -> None:
        return

    def reduction(self, dtype, src_dtype, reduction_type, value):
        return ValueRanges.unknown()

    @classmethod
    def index_expr(cls, index, dtype):
        if not isinstance(index, ValueRanges):
            raise AssertionError(f"expected ValueRanges, got {type(index)}")
        return cls.to_dtype(index, dtype)

    @classmethod
    def value_expr(cls, index, dtype):
        return cls.index_expr(index, dtype)

    @staticmethod
    def to_dtype(
        x,
        dtype,
        src_dtype=None,
        use_compute_types: bool = True,
    ):
        """The range of a value read as another type.

        A conversion to a truth value is a comparison against zero, so the
        range collapses to the two truth values unless the range says which one
        it is; a conversion between a number and a truth value turns the range
        into the numbers zero and one; a conversion between two kinds of number
        moves each end of the range, and an infinite end stays infinite,
        because infinity is not a number an integer can hold.
        """

        x = ValueRanges.wrap(x)

        if dtype == tp.bool:
            if x.is_singleton():
                return ValueRanges.wrap(x.lower != 0)
            elif x.is_bool:
                return x
            elif 0 not in x:
                return ValueRanges.wrap(sympy.true)
            else:
                return ValueRanges(sympy.false, sympy.true)

        def cast(x, dtype):
            if dtype.is_floating_point:
                return sympy.Float(x)
            else:
                if x in (int_oo, -int_oo):
                    return x
                try:
                    return sympy.Integer(x)
                except TypeError:
                    # An infinity cannot become an integer; leaving it as one
                    # is what makes the range unbounded rather than wrong.
                    return x

        if x.is_bool:
            if x.is_singleton():
                val = 1 if x.lower else 0
                return ValueRanges.wrap(cast(val, dtype))
            else:
                return ValueRanges(cast(0, dtype), cast(1, dtype))
        else:
            return ValueRanges(cast(x.lower, dtype), cast(x.upper, dtype))

    @staticmethod
    def square(x):
        return ValueRanges.convex_min_zero_map(x, lambda y: PowByNatural(y, 2))

    @staticmethod
    def neg(x):
        return ValueRanges.decreasing_map(x, operator.neg)

    @classmethod
    def truncdiv(cls, a, b):
        """A truncated division's range, from a real division then truncation.

        The truncation is at integer precision while the division is a real one,
        so the range this produces can be a little wider than the truth; it is
        used because it is cheap rather than because it is exact.
        """

        x = cls.truediv(a, b)
        if x == ValueRanges.unknown():
            return x

        return cls.trunc(x)

    @classmethod
    def sub(cls, a, b):
        return cls.add(a, cls.neg(b))
