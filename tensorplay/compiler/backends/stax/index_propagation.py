"""Carrying positions through a body as expressions, instead of recomputing them.

An index a body computes is usually a simple expression over the loop variables
-- a sum, a difference, a division.  Recomputing it at every use is both slower
and, more to the point, hides the fact that two uses are the same position.  So
a value that can be written as an expression is carried as one, and the
simplification that follows from knowing it is an expression is available to
every use.

A position that came out of the data is a second case: it is not an expression
while the body is being captured, and once it is, a position that turns out to
be a plain expression over the loops can be used directly instead of being
gathered.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, overload

import sympy

import tensorplay as tp

from tensorplay.graph.experimental.sympy_functions import (
    bound_sympy,
    FloorDiv,
    Max,
    Min,
    ModularIndexing,
    ValueRanges,
)
from tensorplay.primitives.common import dtype_to_type, is_integer_dtype

from .codegen.index_expr import Where

from .loops import V
from .ops_handler import DefaultHandler
from .sizevars import statically_known_true
from .utils import generate_assert


_ExprType = sympy.Expr | float | int | bool


def _is_constant(val: _ExprType):
    if isinstance(val, sympy.Basic):
        return val.is_number
    return isinstance(val, (int, float, bool))


def upper_bound(val: _ExprType):
    return bound_sympy(val).upper if isinstance(val, sympy.Expr) else val


@dataclass
class TypedExpr:
    """An expression together with the type it is held in.

    The type is not decoration: an expression over a type narrower than itself
    is not the same value as the expression over the wider type, and where the
    two are the same the difference is a wrap that has to be applied.
    """

    expr: _ExprType
    dtype: Any

    def is_constant(self):
        return _is_constant(self.expr)

    def __post_init__(self):
        if _is_constant(self.expr):
            expr = self.expr
            if isinstance(expr, sympy.Expr):
                expr = expr.expand(identity=True)
            expr = dtype_to_type(self.dtype)(expr)
            if is_integer_dtype(self.dtype):
                bits = tp.iinfo(self.dtype).bits
                if self.dtype.is_signed:
                    expr = expr + 2 ** (bits - 1)
                expr = expr % 2**bits
                if self.dtype.is_signed:
                    expr = expr - 2 ** (bits - 1)
            self.expr = expr


class SymPyOps:
    """The operations a value can be put through while staying an expression.

    An operation that cannot be written as one has no method here, and saying so
    is what lets the caller know to fall back rather than to pretend.
    """

    @staticmethod
    def identity(value: Any) -> Any:
        return value

    @staticmethod
    def constant(value, dtype) -> TypedExpr:
        return TypedExpr(value, dtype)

    @staticmethod
    def index_expr(value, dtype) -> TypedExpr:
        return TypedExpr(value, dtype)

    @staticmethod
    def value_expr(value, dtype) -> TypedExpr:
        return TypedExpr(value, dtype)

    @staticmethod
    def to_dtype(
        value: TypedExpr,
        dtype,
        src_dtype=None,
        use_compute_types: bool = False,
    ) -> TypedExpr:
        return TypedExpr(value.expr, dtype)

    @staticmethod
    def abs(x: TypedExpr) -> TypedExpr:
        return TypedExpr(abs(x.expr), x.dtype)  # type: ignore[arg-type]

    @staticmethod
    def square(x: TypedExpr) -> TypedExpr:
        return TypedExpr(x.expr * x.expr, x.dtype)

    @staticmethod
    def add(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = tp.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr + y.expr, result_type)

    @staticmethod
    def sub(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = tp.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr - y.expr, result_type)

    @staticmethod
    def mul(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = tp.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr * y.expr, result_type)

    @staticmethod
    def neg(x: TypedExpr) -> TypedExpr:
        return TypedExpr(-x.expr, x.dtype)

    @staticmethod
    def floordiv(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = tp.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        return TypedExpr(FloorDiv(x.expr, y.expr), result_type)

    @staticmethod
    def mod(x: TypedExpr, y: TypedExpr):
        result_type = tp.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        result_expr = ModularIndexing(x.expr, sympy.S.One, y.expr)
        return TypedExpr(result_expr, result_type)

    @staticmethod
    def remainder(x: TypedExpr, y: TypedExpr):
        result_type = tp.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        x_expr = sympy.sympify(x.expr)
        y_expr = sympy.sympify(y.expr)
        # Where both are known to be non-negative, or the divisor known to be
        # positive, the two definitions of a remainder agree, so the remainder
        # can be written as a position within a group.
        if (
            x_expr.is_nonnegative is not None
            and x_expr.is_nonnegative == y_expr.is_positive
        ):
            result_expr = ModularIndexing(x.expr, sympy.S.One, y.expr)
            return TypedExpr(result_expr, result_type)
        return NotImplemented

    @staticmethod
    def minimum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = tp.promote_types(x.dtype, y.dtype)
        if result_type == tp.bool:
            return NotImplemented
        return TypedExpr(Min(x.expr, y.expr), result_type)

    @staticmethod
    def maximum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = tp.promote_types(x.dtype, y.dtype)
        if result_type == tp.bool:
            return NotImplemented
        return TypedExpr(Max(x.expr, y.expr), result_type)


@dataclass
class IndexPropVar:
    """A value, either as itself or as the expression it turned out to be."""

    value: Any  # Either an IR value, or TypedExpr if is_symbolic is true
    is_symbolic: bool = False

    @staticmethod
    def new_symbolic(expr: TypedExpr) -> "IndexPropVar":
        return IndexPropVar(expr, is_symbolic=True)

    def __post_init__(self):
        if not (not self.is_symbolic or isinstance(self.value, TypedExpr)):
            raise AssertionError("Symbolic IndexPropVar must contain a TypedExpr")


IndexPropResult: TypeAlias = IndexPropVar | tuple["IndexPropResult", ...]


class IndexPropagation(DefaultHandler):
    """Carries positions and constants through a body as expressions.

    Every value that can be written as an expression is carried as one, so that
    each use of it is a use of the same expression rather than a second
    computation, and so that what can be decided about the expression can be
    decided once.  A value that cannot is passed on as it is.
    """

    def __init__(
        self,
        inner: Any,
        iter_ranges: dict,
        indirect_var_ranges: dict,
    ) -> None:
        self._inner = inner
        self.shape_env = V.graph.sizevars.shape_env

        var_to_range = {
            k: ValueRanges(0, upper_bound(v) - 1) for k, v in iter_ranges.items()
        }
        self.var_to_range = tuple(
            itertools.chain(self.shape_env.var_to_range.items(), var_to_range.items())
        )
        # Kept as a reference on purpose, so that a caller can add to it.
        self.indirect_var_ranges = indirect_var_ranges

        axioms = []
        for x, s in iter_ranges.items():
            axioms.append(0 <= x)
            axioms.append(x < s)
        self.axioms = tuple(axioms) + self.shape_env.get_axioms()

    def materialize_expr(self, expr, dtype) -> Any:
        """Hand an expression on as the constant or position it has become."""

        if _is_constant(expr):
            val = dtype_to_type(dtype)(expr)
            return self._inner.constant(val, dtype)
        return self._inner.index_expr(expr, dtype)

    def value_expr(self, expr, dtype) -> "IndexPropResult":
        return self.wrap(self._inner.value_expr(expr, dtype))

    def unwrap(self, a) -> Any:
        """The value itself, in whichever form it can be handed on in.

        A value that is an expression is handed on as one wherever that is
        wanted, since a position that is known can be used directly; otherwise
        it is the value itself.
        """

        if isinstance(a, (list, tuple)):
            return tuple(self.unwrap(v) for v in a)

        if not isinstance(a, IndexPropVar):
            return a

        # The expression is preferred where there is one.
        if a.is_symbolic:
            return self.materialize_expr(a.value.expr, a.value.dtype)

        return a.value

    def wrap(self, a) -> "IndexPropResult":
        if isinstance(a, (list, tuple)):
            return tuple(self.wrap(v) for v in a)
        return IndexPropVar(a)

    @overload
    def fallback(
        self,
        name: Literal["indirect_indexing"],
        args: Sequence,
        kwargs: dict,
    ) -> IndexPropVar: ...

    @overload
    def fallback(self, name: str, args: Sequence, kwargs: dict) -> "IndexPropResult": ...

    def fallback(self, name: str, args: Sequence, kwargs: dict) -> "IndexPropResult":
        """The operation done by the handler underneath, with values unwrapped."""

        new_args = [self.unwrap(a) for a in args]
        new_kwargs = {k: self.unwrap(v) for k, v in kwargs.items()}
        return self.wrap(getattr(self._inner, name)(*new_args, **new_kwargs))

    def propagate_sympy(self, name: str, args: Sequence, kwargs: dict):
        """The operation done as an expression rather than as a computation."""

        def unwrap(a):
            if not isinstance(a, IndexPropVar):
                return a
            return a.value

        new_args = [unwrap(a) for a in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        try:
            new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
        except (OverflowError, ValueError):
            # An unbounded value asked to become a number raises, and there is
            # no expression to carry in that case.
            return self.fallback(name, args, kwargs)
        is_valid_expr = new_expr is not NotImplemented and (
            # A position is not expected to be a floating point one, but a
            # floating point constant is still worth carrying.
            new_expr.is_constant() or new_expr.expr.is_integer
        )
        if not is_valid_expr:
            return self.fallback(name, args, kwargs)
        return IndexPropVar.new_symbolic(new_expr)

    def _default(self, name: str, args: tuple, kwargs: dict):
        if not hasattr(SymPyOps, name):
            return self.fallback(name, args, kwargs)

        var_arguments = [
            a
            for a in itertools.chain(args, kwargs.values())
            if isinstance(a, IndexPropVar)
        ]
        if not all(v.is_symbolic for v in var_arguments):
            return self.fallback(name, args, kwargs)

        return self.propagate_sympy(name, args, kwargs)

    def statically_true(self, e):
        """Whether a relation about a position holds, from what is known of it.

        The extents of the loops are known, and so are the ones of the positions
        read out of the data, so a relation between them can often be settled
        without a guard.
        """

        var_to_range = (
            *self.var_to_range,
            *(
                (k, ValueRanges(0, upper_bound(v) - 1))
                for k, v in self.indirect_var_ranges.items()
            ),
        )
        return statically_known_true(self.shape_env, e, self.axioms, var_to_range)

    def indirect_indexing(
        self,
        index,
        size,
        check: bool = True,
        wrap_neg=True,
    ) -> Any:
        """Read a position out of a tensor, as an expression wherever it is one.

        A position that is already a plain expression over the loops does not
        have to be gathered at all, and using it directly is what turns an
        indirect read into an ordinary one.  A negative position is wrapped
        round by the extent, and whether that is needed at all is decided from
        what is known of the position rather than checked.
        """

        if isinstance(index, IndexPropVar) and index.is_symbolic:
            # A position that can be written as an expression is used as one.
            # It still has to be brought into range and checked, but doing that
            # here rather than in the kernel is the point: a kernel is not
            # fused across an indirect read.
            expr = sympy.sympify(index.value.expr)

            def wrap_expr(expr):
                # Positive, negative, or of neither sign.
                if self.statically_true(0 <= expr):
                    return expr
                elif self.statically_true(expr < 0):
                    return expr + size
                else:
                    return Where(expr < 0, expr + size, expr)

            # A lower bound of negative the extent only counts once a negative
            # position has been wrapped.
            can_prove_lower = self.statically_true(0 <= expr) or (
                wrap_neg and self.statically_true(-size <= expr)
            )
            can_prove_upper = self.statically_true(expr < size)
            if wrap_neg:
                expr = wrap_expr(expr)
            if generate_assert(check):
                self.fallback(
                    "check_bounds",
                    (expr, size),
                    dict(lower=not can_prove_lower, upper=not can_prove_upper),
                )
            return expr

        indirect_var = self.fallback(
            "indirect_indexing", (index, size, check, wrap_neg), {}
        ).value
        return indirect_var
