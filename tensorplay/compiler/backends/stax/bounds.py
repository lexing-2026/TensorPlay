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

import operator
import sympy
from functools import partial

import tensorplay as tp

from tensorplay.graph.experimental.sympy_functions import (
    PowByNatural,
    SymPyValueRangeAnalysis,
    ValueRanges,
    int_oo,
)
from .codegen.index_expr import Expr
from tensorplay.graph.experimental.sympy_functions import bound_sympy
from .ops_handler import DefaultHandler
from .utils import cache_on_self, dominated_nodes


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


class BoundVars:
    """What range each value of a loop body can take.

    The body is a graph, and the question is what range each node's value is
    in, given the ranges of the values it was computed from.  That is answered
    by running the body with a value-range handler installed, so the answer
    comes from the same walk the values are produced by rather than from a
    second description of the body that could disagree with it.

    A value read out of memory has no range -- how far a load can read is not
    known here -- so every value computed from a load is given none either,
    rather than a range that would let an out-of-bounds read look in bounds.

    The analysis is per body.  A value one body returns and another reads would
    need the two analyses joined, which this does not do; what it does do is
    refuse to claim a range it has not established.
    """

    def __init__(self, loop_body) -> None:
        def upper_bound(value):
            # The length of a run of memory is its stride's multiple less the
            # stride, since the last element starts one stride short of the
            # end.  A value that is not an expression has no length to work
            # out, and is its own upper bound.  An index-algebra value is
            # translated into the symbolic language the range analysis uses.
            if isinstance(value, Expr):
                return bound_sympy(value.to_sympy()).upper
            if isinstance(value, sympy.Expr):
                return bound_sympy(value).upper
            return value

        self.loop_body = loop_body
        self.replacement_vals = {
            key: ValueRanges(0, upper_bound(value) - 1)
            for key, value in loop_body.var_ranges.items()
        }
        # Everything computed from a load, a reduction, or a piece of a list is
        # treated as unbounded, because a load's range is not known and the
        # others are not simple values.
        self.unbounded_vars = dominated_nodes(
            node
            for node in self.loop_body.get_nodes()
            if node.target in ("load", "reduction", operator.getitem)
            or isinstance(node.target, str) and "masked_subblock" in node.target
        )
        self._bounds: dict[Any, ValueRanges] = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loop_body={self.loop_body},\n "
            f"replacement_vals={self.replacement_vals}, \n"
            f"unbounded_vars={self.unbounded_vars}, \n"
            f"_bounds={self._bounds})"
        )

    @cache_on_self
    def get_bounds(self) -> dict[Any, ValueRanges]:
        """The range of every value in the body, by the value that stands for it.

        The unbounded values are seeded first, so that anything computed from
        one is computed from a value already known to have no range.
        """

        from .loop_body import InterpreterShim
        from .loops import V, set_ops_handler

        submodules = self.swap_submodules(self.loop_body.submodules)
        for node in self.unbounded_vars:
            # A masked sub-block and an indirect write are evaluated rather than
            # assumed: they establish what the values inside them are, and
            # skipping them would leave the block's own values unbounded when
            # they need not be.
            if not isinstance(node.target, str) or (
                "masked_subblock" not in node.target
                and "set_indirect" not in node.target
            ):
                self._bounds[node] = ValueRanges.unknown()

        with set_ops_handler(ValueRangeAnalysis()):
            interpreter = InterpreterShim(self.loop_body.root_block.graph, submodules)
            interpreter.run(V.get_ops_handler(), initial_env=self._bounds)
        return self._bounds

    def swap_submodules(self, submodules: dict) -> dict:
        """The sub-modules the analysis runs, with the ones it answers itself.

        An index has to be answered with a range rather than a value, and a
        block has to be run rather than called, so those are replaced; a scan is
        left alone, because it produces the same values it would otherwise.
        """

        result: dict = {}
        for key in submodules:
            if key == "get_index":
                result[key] = self.get_index
            elif "masked_subblock" in key:
                subblock = self.loop_body.subblocks[key]

                # Bound in a function of its own: a lambda written here would
                # close over the sub-block by reference, so every lambda made in
                # the loop would see the last one.
                def make_fn(subblock):
                    return lambda mask, value: self.masked_subblock(
                        subblock, self._bounds, mask, value, result
                    )

                result[key] = make_fn(subblock)
            elif "set_indirect" in key:
                index = int(key[len("set_indirect") :])
                var = self.loop_body.indirect_vars[index]
                result[key] = partial(self.set_indirect, var)
            else:
                if "scan" not in key:
                    raise AssertionError(
                        f"expected 'scan' in submodule key, got {key!r}"
                    )
                result[key] = submodules[key]
        return result

    def masked_subblock(
        self,
        subblock,
        env: dict,
        mask: Any,
        value: Any,
        submodules: dict,
    ) -> ValueRanges:
        """The range of a part of a body, as a value of the whole."""

        from .loop_body import InterpreterShim
        from .loops import V

        interp = InterpreterShim(subblock.graph, submodules)
        interp.run(V.get_ops_handler(), initial_env=env)
        output = [node for node in subblock.graph.nodes if node.target == "output"]
        if len(output) != 1:
            raise AssertionError(f"expected exactly 1 output node, got {len(output)}")
        # Not joined with the value it replaces: that value came out of a
        # masked read, whose range is not known, so joining would widen the
        # answer to nothing.
        return interp.env[output[0]]

    def set_indirect(self, old: Expr, new: ValueRanges) -> ValueRanges:
        """Record the range a value was last written with."""

        if not isinstance(new, ValueRanges):
            raise AssertionError(f"expected ValueRanges, got {type(new)}")
        self.replacement_vals[old] = new
        return new

    def get_index(self, name: str) -> ValueRanges:
        """The range of an index expression, given the ranges of its variables."""

        expr = self.loop_body.indexing_exprs[name]
        bound = self.replacement_vals.get(expr)
        if bound is None:
            # An index may be held as an expression already, or wrapped.
            to_sympy = getattr(expr, "to_sympy", None)
            bound = bound_sympy(to_sympy() if to_sympy is not None else expr, self.replacement_vals)
        self.replacement_vals[name] = bound
        return bound
