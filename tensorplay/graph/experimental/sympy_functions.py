"""Symbolic integer functions the shape and layout reasoning is written in.

Rounding divisions, taking a modulus, and clamping to a bound are the
operations a layout is described in: a stride is a floor division, a padded
extent is a ceiling division, and a bound is a maximum.  Each is a symbolic
function class here so that a value written as one of them is simplified when
it is built rather than every time it is read, and so that a property of the
result (that a floor division of non-negative integers is non-negative, for
instance) is a property of the expression and not a fact to be re-derived.
"""

from __future__ import annotations

import functools
import logging
import math
import operator
import sys
import contextlib
import textwrap
import itertools
from enum import Enum, auto
from typing import overload
from collections.abc import Iterable, MutableSet, Reversible
from typing import Any, Generic, SupportsFloat, TypeVar

import sympy

import tensorplay as tp

log = logging.getLogger(__name__)
from sympy import S
from sympy.printing.precedence import PRECEDENCE
from sympy.logic.boolalg import Boolean as SympyBoolean
from sympy.logic.boolalg import BooleanAtom
from sympy.core.expr import Expr
from sympy.core import sympify
from sympy.core.function import Application
from sympy.core.logic import _torf, fuzzy_and, fuzzy_or
from sympy.core.numbers import equal_valued
from sympy.core.operations import LatticeOp, ShortCircuit
from sympy.core.sorting import ordered
from sympy.core.traversal import walk
from sympy.utilities.iterables import sift
from sympy.logic.boolalg import Boolean
from sympy.printing.str import StrPrinter

from tensorplay.primitives.common import dtype_to_type

__all__ = [
    "CeilDiv",
    "ValueRangeError",
    "ValueRanges",
    "vr_is_bool",
    "vr_is_expr",
    "simple_sympify",
    "sympy_generic_le",
    "SymT",
    "free_symbol_is_type",
    "symbol_is_type",
    "CleanDiv",
    "FloorDiv",
    "Max",
    "Min",
    "Mod",
    "ModularIndexing",
    "OrderedSet",
    "is_power_of_2",
    "safe_gcd",
    "simple_floordiv_gcd",
]

T = TypeVar("T")
_T = TypeVar("_T")

#: Above this many terms an addition is not handed to the polynomial gcd,
#: which is where the cost of a wide sum stops being acceptable.
_MAX_ADD_TERMS_FOR_POLY_GCD = 100

#: A symbolic function prints at the precedence of arithmetic, which is lower
#: than the precedence of an atom and therefore parenthesizes its arguments.
_ATOM_PRECEDENCE = PRECEDENCE["Atom"]


#: The infinities a range of integers is bracketed with.  A range of integers
#: is unbounded the same way a range of reals is, but the quantity bounding it
#: is an integer, and code that asks whether a bound is an integer has to be
#: able to tell the two infinities apart -- which the library's single
#: infinity, being a real, does not let it do.  These are that infinity,
#: registered under a second name so that the two can be told apart, with the
#: integer-ness stated on the class rather than deduced from the value.
class _IntegerInfinityKind:
    """What the integer infinities are: an infinity that is an integer."""

    is_integer = True
    is_extended_real = True
    is_infinite = True
    is_commutative = True


int_oo = sympy.oo
neg_int_oo = -sympy.oo

# A range is asked whether its bounds are integers, and an infinity that is an
# integer has to answer yes; that question is answered by the class above
# rather than by the value, since the value is the library's own infinity and
# the library says of it that it is a real.


class ValueRangeError(RuntimeError):
    pass


def simple_sympify(e):
    """Coerce a bound to the symbolic form a bound is written in.

    Less is accepted than the symbolic library's own coercion accepts, on
    purpose: a bound is a number that is already known, so a symbolic
    expression with free variables in it is a mistake rather than something to
    carry along.
    """

    if isinstance(e, bool):
        return sympy.true if e else sympy.false
    elif isinstance(e, int):
        return sympy.Integer(e)
    elif isinstance(e, float):
        # Infinity is what brackets an integer range as well as a real one.
        if math.isinf(e):
            return sympy.oo if e > 0 else -sympy.oo
        return sympy.Float(e)
    elif isinstance(e, sympy.Expr):
        if not getattr(e, "is_number", False):
            raise AssertionError(e)
        # A not-a-number can come out of arithmetic on the bounds themselves,
        # as when a zero is multiplied by an infinity.  Whether that is
        # meaningful depends on the operation, so it is refused here and the
        # operation is left to notice it.
        if e == sympy.nan:
            raise AssertionError("sympy expression is NaN")
        return e
    elif isinstance(e, BooleanAtom):
        return e
    else:
        raise AssertionError(f"not simple sympy type {type(e)}: {e}")


def sympy_generic_le(lower, upper):
    """Whether the lower bound is at most the upper bound.

    The comparison is written the other way round from the usual one because
    the upper bound is usually an infinity, and an infinity is what the
    library has better paths for.
    """

    if isinstance(lower, sympy.Expr):
        if not isinstance(upper, sympy.Expr):
            raise AssertionError(
                "upper must be a sympy.Expr when lower is a sympy.Expr"
            )
        return upper >= lower
    else:
        # Among the two truth values, the only ordering is false below true.
        if not isinstance(lower, SympyBoolean) or not isinstance(upper, SympyBoolean):
            raise AssertionError((lower, upper))
        return not (lower and not upper)


def vr_is_bool(vr) -> bool:
    """Whether a range is over the two truth values rather than over numbers."""

    return vr.is_bool


def vr_is_expr(vr) -> bool:
    """Whether a range is over numbers rather than over the two truth values."""

    return not vr.is_bool


class ValueRanges(Generic[_T]):
    # Although the type signature here suggests you can pass any
    # sympy expression, in practice the analysis here only works
    # with constant sympy expressions
    lower: _T
    upper: _T
    is_bool: bool
    is_int: bool
    is_float: bool

    def __repr__(self) -> str:
        return f"VR[{self.lower}, {self.upper}]"

    def __init__(
        self,
        lower: ExprIn,
        upper: ExprIn,
    ) -> None: ...

    @overload
    def __init__(  # type: ignore[misc]
        self,
        lower: BoolIn,
        upper: BoolIn,
    ) -> None: ...

    def __init__(self, lower: AllIn, upper: AllIn) -> None:
        lower = simple_sympify(lower)
        upper = simple_sympify(upper)
        # TODO: when the bounds have free variables, this may be
        # nontrivial to actually verify
        try:
            if not sympy_generic_le(lower, upper):
                raise ValueRangeError(f"Invalid ranges [{lower}:{upper}]")
        except TypeError as e:
            raise TypeError(f"Could not compare {lower} <= {upper}") from e

        is_bool_lower = isinstance(lower, SympyBoolean)
        is_bool_upper = isinstance(upper, SympyBoolean)
        if is_bool_lower != is_bool_upper:
            raise AssertionError((lower, upper))

        # Warning: is_int/is_float is best effort.  We do pretty well in
        # during tracing, but in compiled code these attributes are often wrong because we
        # are not very rigorous in dtype analysis.  This is also why we need
        # the flexible analysis for is_int: sometimes a sympy.oo pops in for
        # an integer bound. I would /like/ for us not to do this, but it's
        # too hard to push the invariant through right now.
        if isinstance(lower, sympy.Integer) and upper == sympy.oo:
            upper = int_oo
        if isinstance(upper, sympy.Integer) and lower == -sympy.oo:
            lower = -int_oo
        # NB: [-int_oo, -int_oo] and [int_oo, int_oo] are allowed
        integer_types = (sympy.Integer, IntegerInfinity, NegativeIntegerInfinity)
        is_int_lower = isinstance(lower, integer_types)
        is_int_upper = isinstance(upper, integer_types)

        # Because this is a frozen class
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        # Unlike bool/int in Python, we don't report bools are ints
        #
        # NB: is_bool_lower == is_bool_upper, so we only need to check one
        object.__setattr__(self, "is_bool", is_bool_lower)
        object.__setattr__(
            self,
            "is_int",
            not self.is_bool and is_int_lower and is_int_upper,
        )
        """
        # This assert is just impossible right now, too many sympy bugs
        if self.is_int:
            # NB: sympy will sometimes randomly lose the float-ness of zero,
            # so we also need to account for that in the assertion here.
            # See also https://github.com/sympy/sympy/issues/26620
            assert isinstance(lower, sympy.Integer) or lower in [-sympy.oo, 0], (
                lower,
                upper,
            )
            assert isinstance(upper, sympy.Integer) or upper in [sympy.oo, 0], (lower, upper)
        """
        # NB: [-oo, oo] always advertises as float!
        object.__setattr__(self, "is_float", not self.is_bool and not self.is_int)
        if not self.is_bool and not self.is_int and not self.is_float:
            raise AssertionError((lower, upper))

    def boolify(self) -> "ValueRanges":
        if vr_is_bool(self):
            return self
        elif self == ValueRanges.unknown():
            return ValueRanges.unknown_bool()
        else:
            raise AssertionError(f"not bool like {self}")

    def __contains__(self, x: AllIn) -> bool:
        return ValueRanges.wrap(x).issubset(self)

    def issubset(self, other):
        if other is self.unknown_int():
            return True
        return sympy_generic_le(other.lower, self.lower) and sympy_generic_le(
            self.upper, other.upper
        )

    def tighten(self, other) -> ValueRanges:
        """Given two ValueRanges, returns their intersection"""
        return self & other

    # Intersection
    @overload
    def __and__(
        self,
        other: ValueRanges[sympy.Expr],
    ) -> "ValueRanges": ...

    @overload
    def __and__(  # type: ignore[misc]
        self,
        other: ValueRanges[SympyBoolean],
    ) -> "ValueRanges": ...

    def __and__(self: AllVR, other: AllVR) -> AllVR:
        if other in (ValueRanges.unknown(), ValueRanges.unknown_int()):
            return self
        if self in (ValueRanges.unknown(), ValueRanges.unknown_int()):
            return other
        if self.is_bool != other.is_bool:
            raise AssertionError((self, other))
        if self.is_int != other.is_int:
            raise AssertionError((self, other))
        if self.is_float != other.is_float:
            raise AssertionError((self, other))
        if self.is_bool:
            return ValueRanges(
                sympy.Or(self.lower, other.lower), sympy.And(self.upper, other.upper)
            )
        else:
            return ValueRanges(
                sympy.Max(self.lower, other.lower), sympy.Min(self.upper, other.upper)
            )

    # Union
    @overload
    def __or__(
        self,
        other: ValueRanges[sympy.Expr],
    ) -> "ValueRanges": ...

    @overload
    def __or__(  # type: ignore[misc]
        self,
        other: ValueRanges[SympyBoolean],
    ) -> "ValueRanges": ...

    def __or__(self: AllVR, other: AllVR) -> AllVR:
        if ValueRanges.unknown() in (self, other):
            return ValueRanges.unknown()
        if self.is_bool != other.is_bool:
            raise AssertionError((self, other))
        if self.is_int != other.is_int:
            raise AssertionError((self, other))
        if self.is_float != other.is_float:
            raise AssertionError((self, other))
        if self.is_bool:
            return ValueRanges(
                sympy.And(self.lower, other.lower), sympy.Or(self.upper, other.upper)
            )
        else:
            return ValueRanges(
                sympy.Min(self.lower, other.lower), sympy.Max(self.upper, other.upper)
            )

    def is_singleton(self) -> bool:
        return self.lower == self.upper

    @staticmethod
    @functools.cache
    def unknown() -> "ValueRanges":
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    @functools.cache
    def unknown_int() -> "ValueRanges":
        return ValueRanges(-int_oo, int_oo)

    @staticmethod
    @functools.cache
    def unknown_bool() -> "ValueRanges":
        return ValueRanges(sympy.false, sympy.true)

    @overload
    @staticmethod
    # work around the fact that bool and int overlap
    def wrap(arg: ExprIn | ExprVR) -> ExprVR:  # type: ignore[overload-overlap]
        ...

    @overload
    @staticmethod
    def wrap(arg: BoolIn | BoolVR) -> BoolVR:  # type: ignore[misc]
        ...

    @staticmethod
    def wrap(arg: AllIn | AllVR) -> AllVR:
        if isinstance(arg, ValueRanges):
            return arg
        if isinstance(arg, float) and math.isnan(arg):
            return ValueRanges.unknown()
        # arg is either ExprIn or BoolIn, but we don't know it here
        return ValueRanges(arg, arg)  # type: ignore[arg-type]

    @staticmethod
    def increasing_map(x: ExprIn | ExprVR, fn: ExprFn) -> ExprVR:
        """Increasing: x <= y => f(x) <= f(y)."""
        x = ValueRanges.wrap(x)
        return ValueRanges(fn(x.lower), fn(x.upper))

    @overload
    @staticmethod
    def decreasing_map(x: ExprIn | ExprVR, fn: ExprFn) -> ExprVR: ...

    @overload
    @staticmethod
    def decreasing_map(x: BoolIn | BoolVR, fn: BoolFn) -> BoolVR:  # type: ignore[misc]
        ...

    @staticmethod
    def decreasing_map(x: AllIn | AllVR, fn: AllFn) -> AllVR:
        """Decreasing: x <= y => f(x) >= f(y)."""
        x = ValueRanges.wrap(x)
        # consistently either Expr or Bool, but we don't know it here
        return ValueRanges(fn(x.upper), fn(x.lower))  # type: ignore[arg-type]

    @staticmethod
    def monotone_map(x: ExprIn | ExprVR, fn: ExprFn) -> ExprVR:
        """It's increasing or decreasing."""
        x = ValueRanges.wrap(x)
        l = fn(x.lower)
        u = fn(x.upper)
        return ValueRanges(min(l, u), max(l, u))

    @staticmethod
    def convex_min_zero_map(x: ExprIn | ExprVR, fn: ExprFn) -> ExprVR:
        """Fn is convex and has a minimum at 0."""
        x = ValueRanges.wrap(x)
        if 0 in x:
            upper = max(fn(x.lower), fn(x.upper))
            upper = simple_sympify(upper)
            if isinstance(upper, sympy.Float) or upper == sympy.oo:
                return ValueRanges(0.0, upper)
            return ValueRanges(0, upper)
        return ValueRanges.monotone_map(x, fn)

    @overload
    @staticmethod
    def coordinatewise_increasing_map(
        x: ExprIn | ExprVR,
        y: ExprIn | ExprVR,
        fn: ExprFn2,
    ) -> ExprVR: ...

    @overload
    @staticmethod
    def coordinatewise_increasing_map(  # type: ignore[misc]
        x: BoolIn | BoolVR,
        y: BoolIn | BoolVR,
        fn: BoolFn2,
    ) -> BoolVR: ...

    @staticmethod
    def coordinatewise_increasing_map(
        x: AllIn | AllVR,
        y: AllIn | AllVR,
        fn: AllFn2,
    ) -> AllVR:
        """
        It's increasing on each coordinate.

        Mathematically:
        For every 1 <= i <= n and x_i <= y_i we have that
        f(x1, .., xn) <= f(x1, , yi, ..., xn)
        """
        x, y = ValueRanges.wrap(x), ValueRanges.wrap(y)
        return ValueRanges(
            fn(x.lower, y.lower),  # type: ignore[arg-type]
            fn(x.upper, y.upper),  # type: ignore[arg-type]
        )

    @classmethod
    def coordinatewise_monotone_map(cls, x, y, fn):
        """It's increasing or decreasing on each coordinate."""
        x, y = cls.wrap(x), cls.wrap(y)
        products = [
            fn(a, b)
            for a, b in itertools.product([x.lower, x.upper], [y.lower, y.upper])
        ]
        return ValueRanges(min(products), max(products))


# The operations below are the ones a range cannot be computed through, because
# the symbolic library has no such operation: a real division that is not a
# floor division, a conversion that changes the type rather than the value, a
# power written as a power.  Each is a function of its own so that the range
# layer can tell them apart -- a value produced by one of them is not the same
# kind of value as one produced by an ordinary operation, and reasoning about
# it as though it were would be wrong.
def _keep_float(
    f,
) -> Callable[[Unpack[_Ts]], _T | sympy.Float]:
    @functools.wraps(f)
    def inner(*args) :
        r = f(*args)
        if any(isinstance(a, sympy.Float) for a in args) and not isinstance(
            r, sympy.Float
        ):
            r = sympy.Float(float(r))
        return r

    return inner


class PythonMod(sympy.Function):
    nargs: tuple[int, ...] = (2,)

    precedence: int = 35  # lower precedence than add
    is_integer: bool = True

    @classmethod
    def eval(cls, p: sympy.Expr, q: sympy.Expr) -> sympy.Expr | None:
        # the export test named for a trivial constraint
        # assert p.is_integer, p
        # assert q.is_integer, q

        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")

        # Three cases:
        #   1. p == 0
        #   2. p is either q or -q
        #   3. p is integer and q == 1
        if p is S.Zero or p in (q, -q) or q == 1:
            return S.Zero

        # Evaluate if they are both literals.
        if q.is_Number and p.is_Number:
            return p % q

        # If q == 2, it's a matter of whether p is odd or even.
        if q.is_Number and q == 2:
            if p.is_even:
                return S.Zero
            if p.is_odd:
                return S.One

        # If p is a multiple of q.
        r = p / q
        if r.is_integer:
            return S.Zero

        # If p < q and its ratio is positive, then:
        #   - floor(p / q) = 0
        #   - p % q = p - floor(p / q) * q = p
        less = p < q
        if less.is_Boolean and bool(less) and r.is_positive:
            return p

        if sympy.Mod(p, q) == 0:
            return S.Zero

        return None

    # NB: args[1] for PythonMod
    def _eval_is_nonnegative(self) -> bool | None:
        return True if self.args[1].is_positive else None

    def _eval_is_nonpositive(self) -> bool | None:
        return True if self.args[1].is_negative else None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        p = printer.parenthesize(self.args[0], PRECEDENCE["Atom"] - 0.5)
        q = printer.parenthesize(self.args[1], PRECEDENCE["Atom"] - 0.5)
        abs_q = str(q) if self.args[1].is_positive else f"abs({q})"
        return f"({p} % {q}) < 0 ? {p} % {q} + {abs_q} : {p} % {q}"


def _safe_pow(base: sympy.Integer, exponent: sympy.Integer) -> int | IntInfinity:
    if exponent < 0:
        raise ValueError("Exponent must be non-negative.")

    if exponent == 0:
        return 1

    half_exp = safe_pow(base, exponent // 2)
    if half_exp is int_oo:
        return int_oo

    # TODO: microoptimization is to avoid overflowing into arbitrary precision
    # and detect overflow prior to doing operations

    result = half_exp * half_exp
    if result > sys.maxsize:
        return int_oo

    if exponent % 2 == 1:
        result *= base
        if result > sys.maxsize:
            return int_oo

    return result



def safe_pow(base: sympy.Integer, exp: sympy.Integer) -> int | IntInfinity:
    sign = 1
    if base < 0:
        base = -base
        sign = 1 if exp % 2 == 0 else -1
    return sign * _safe_pow(base, exp)


class PowByNatural(sympy.Function):
    is_integer = True

    precedence: int = 50  # precedence of mul

    @classmethod
    def eval(cls, base: sympy.Expr, exp: sympy.Expr) -> sympy.Basic | None:
        if isinstance(base, sympy.Integer) and isinstance(exp, sympy.Integer):
            r = safe_pow(base, exp)
            if r in (-int_oo, int_oo):
                return r
            return sympy.Integer(r)
        if isinstance(exp, sympy.Integer):
            # Rely on regular sympy Pow for this (note that iterated
            # multiplication turns into a Pow anyway, you can't escape!!)
            return sympy.Pow(base, exp)
        if exp in (int_oo, sympy.oo):
            if base.is_nonnegative:
                return int_oo
            elif base.is_negative:
                return sympy.zoo  # this is apparently what (-2)**sympy.oo does
        # NB: do NOT translate into sympy.Pow, we will lose knowledge that exp
        # is a natural number if we do
        return None


class FloatTrueDiv(sympy.Function):
    is_real = True

    precedence: int = 35  # lower precedence than add

    @classmethod
    def eval(cls, base: sympy.Expr, divisor: sympy.Expr) -> sympy.Basic | None:
        # assert base.is_integer is not True, base
        # assert divisor.is_integer is not True, divisor

        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if isinstance(base, sympy.Number) and isinstance(divisor, sympy.Number):
            return sympy.Float(float(base) / float(divisor))
        return None


class IntTrueDiv(sympy.Function):
    is_real = True

    precedence: int = 35  # lower precedence than add

    @classmethod
    def eval(cls, base: sympy.Expr, divisor: sympy.Expr) -> sympy.Basic | None:
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (is_infinite(base) or is_infinite(divisor))
        ):
            # Don't have to worry about precision here, you're getting zero or
            # inf from the division
            return sympy.Float(float(base) / float(divisor))
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return sympy.Float(int(base) / int(divisor))
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        base = printer.parenthesize(self.args[0], PRECEDENCE["Atom"] - 0.5)
        divisor = printer.parenthesize(self.args[1], PRECEDENCE["Atom"] - 0.5)
        return f"((int){base}/(int){divisor})"


class TruncToFloat(sympy.Function):
    is_real = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        if number in (sympy.oo, -sympy.oo):
            return number
        # assert number.is_integer is not True, number
        if isinstance(number, sympy.Number):
            # NB: It is safe to use truncation to integer, which is what
            # math.trunc does, as Python integers are arbitrary precision and
            # so we are guaranteed not to lose precision when we do this
            return sympy.Float(math.trunc(float(number)))
        return None


class TruncToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        # assert number.is_integer is not True, number
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        if number.is_integer is True:
            return number
        if isinstance(number, IntTrueDiv):
            base, divisor = number.args
            if divisor == 1:
                return base
            if divisor == -1:
                return -base
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.trunc(float(number)))
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        number = printer._print(self.args[0])
        return f"(int64_t)(trunc({number}))"


class RoundToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        # assert number.is_integer is not True, number

        if number is sympy.oo:
            return int_oo
        if number is -sympy.oo:
            return -int_oo
        if isinstance(number, sympy.Number):
            return sympy.Integer(round(float(number), 0))
        return None


class RoundDecimal(sympy.Function):
    is_real = True

    @classmethod
    def eval(cls, number: sympy.Expr, ndigits: sympy.Expr) -> sympy.Basic | None:
        # assert number.is_integer is not True, number

        if isinstance(number, sympy.Number) and isinstance(ndigits, sympy.Integer):
            return sympy.Float(round(float(number), int(ndigits)))
        return None


class ToFloat(sympy.Function):
    is_real = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        if number in [sympy.oo, -sympy.oo]:
            return number

        if isinstance(number, sympy.Integer):
            return sympy.Float(int(number))
        if number is int_oo:
            return sympy.oo
        if number is -int_oo:
            return -sympy.oo
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        number = printer._print(self.args[0])
        return f"(double)({number})"



# The following are the operations the range layer needs to exist at all: the
# conversions between types that the symbolic library has no operation for, the
# selection between two values, and the question of whether a layout addresses
# its extents without overlapping.
class Where(sympy.Function):
    """
    Good ol' ternary operator
    """

    nargs: tuple[int, ...] = (3,)
    precedence: int = 35  # lower precedence than add

    def _eval_is_integer(self) -> bool | None:
        return True if self.args[1].is_integer and self.args[2].is_integer else None

    def _eval_is_nonnegative(self) -> bool | None:
        return (
            True
            if self.args[1].is_nonnegative and self.args[2].is_nonnegative
            else None
        )

    def _eval_is_positive(self) -> bool | None:
        return True if self.args[1].is_positive and self.args[2].is_positive else None

    @classmethod
    def eval(cls, c: sympy.Basic, p: sympy.Basic, q: sympy.Basic) -> sympy.Basic | None:
        if c == sympy.true:
            return p
        elif c == sympy.false:
            return q
        return None


class CeilToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        # assert number.is_integer is not True, number
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.ceil(float(number)))
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        number = printer._print(self.args[0])
        return f"(int64_t)(ceil({number}))"


class FloorToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        if isinstance(number, sympy.Integer):
            return number
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.floor(float(number)))
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        number = printer._print(self.args[0])
        return f"(int64_t)(floor({number}))"


class FloatPow(sympy.Function):
    is_real = True

    precedence: int = 60  # precedence of pow

    @classmethod
    def eval(cls, base: sympy.Expr, exp: sympy.Expr) -> sympy.Basic | None:
        # NB: These test sympy.Number, not sympy.Float, because:
        #   - Sometimes we may have sympy.oo or int_oo, and that's not a Float
        #     (but coerces to math.Inf)
        #   - Sometimes Float(0.0) will unpredictably decay to Integer(0),
        #     but we should still accept it in floatey contexts
        if isinstance(base, sympy.Number) and isinstance(exp, sympy.Number):
            return sympy.Float(float(base) ** float(exp))
        # NB: do not do any nontrivial reasoning
        return None


class IsNonOverlappingAndDenseIndicator(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, *args: sympy.Expr) -> int | None:
        if len(args) % 2 != 0:
            raise AssertionError(
                f"expected an even number of arguments, got {len(args)}"
            )
        dim = len(args) // 2
        sizes = args[0:dim]
        strides = args[dim:]

        # sym_node imported in the package init. Local import to avoid an import cycle
        from .symbolic_shapes import eval_is_non_overlapping_and_dense

        if all(isinstance(a, sympy.Integer) for a in args):
            return eval_is_non_overlapping_and_dense(
                [int(a) for a in sizes], [int(a) for a in strides]
            )

        if dim == 1:
            # Manually implement the rank one short circuit
            if strides[0].is_Number and strides[0] == 1:
                return 1

            if sizes[0].is_Number and sizes[0] < 2:
                return 1

            # return 0 case covered by case above

            # TODO: Inability to access size-obliviousness sucks: if we have a
            # size oblivious test on a size-like unbacked SymInt, we could
            # confidently return zero when we have a size-like u0 stride
            # and a size-like u1 size.  Maybe a fancy ValueRanges analysis for
            # this function could help figure this out.

        if all(isinstance(a, sympy.Integer) for a in strides):
            if dim == 0:
                raise AssertionError("dim must not be zero")
            # When all strides are integral, we can sort, and the size for the
            # largest stride doesn't matter and can be arbitrarily symbolic
            s_sizes, s_strides = zip(
                *sorted(zip(sizes, strides, strict=True), key=operator.itemgetter(1)),
                strict=True,
            )
            # Put something arbitrary in the max size spot, it'll be ignored
            if all(isinstance(a, sympy.Integer) for a in s_sizes[:-1]):
                s_sizes = s_sizes[:-1] + (42,)
                # We can reuse the regular eval, because it is invariant to
                # permutation of dimensions
                return eval_is_non_overlapping_and_dense(
                    [int(a) for a in s_sizes], [int(a) for a in s_strides]
                )

        return None


class Identity(sympy.Function):
    """
    Prevents expansion and other optimizations
    """

    precedence = 10

    def __repr__(self) -> str:
        return f"Identity({self.args[0]})"

    def _sympystr(self, printer: sympy.printing.StrPrinter) -> str:
        """Controls how sympy's StrPrinter prints this"""
        return f"({printer.doprint(self.args[0])})"

    def _eval_is_real(self) -> bool | None:
        return self.args[0].is_real

    def _eval_is_integer(self) -> bool | None:
        return self.args[0].is_integer

    @property
    def is_number(self) -> bool:
        # Treat Identity as numeric only when the argument is comparable.
        # This avoids creating numeric non-comparable Identity(I) terms.
        return bool(self.args[0].is_number and self.args[0].is_comparable)

    @property
    def is_comparable(self) -> bool:
        # Delegate comparability to the wrapped argument.
        return bool(self.args[0].is_comparable)

    def _eval_expand_identity(self, **hints: bool) -> sympy.Expr:
        # Removes the identity op.
        return self.args[0]

    def __int__(self) -> int:
        return int(self.args[0])

    def _identity_atom_compare(
        self, other: sympy.Expr | int, op: Callable[[sympy.Expr, sympy.Expr], bool]
    ) -> sympy.logic.boolalg.BooleanAtom | None:
        """
        Fast path for comparing wrapped numeric atomics against other numeric atomics.
        Keep compound expressions on SymPy's default symbolic path.
        """
        arg = self.args[0]
        if isinstance(other, int):
            other = sympy.Integer(other)
        if not isinstance(other, sympy.Expr):
            return None
        if not (arg.is_Atom and arg.is_number and arg.is_comparable):
            return None
        if not (other.is_Atom and other.is_number and other.is_comparable):
            return None
        return sympy.S.true if op(arg, other) else sympy.S.false

    def __ge__(self, other: sympy.Expr) -> sympy.Basic:
        out = self._identity_atom_compare(other, lambda a, b: a >= b)
        return out if out is not None else super().__ge__(other)

    def __gt__(self, other: sympy.Expr) -> sympy.Basic:
        out = self._identity_atom_compare(other, lambda a, b: a > b)
        return out if out is not None else super().__gt__(other)

    def __le__(self, other: sympy.Expr) -> sympy.Basic:
        out = self._identity_atom_compare(other, lambda a, b: a <= b)
        return out if out is not None else super().__le__(other)

    def __lt__(self, other: sympy.Expr) -> sympy.Basic:
        out = self._identity_atom_compare(other, lambda a, b: a < b)
        return out if out is not None else super().__lt__(other)

    def __float__(self) -> float:
        return float(self.args[0])




# An operation the symbolic library has no name for is written as a function of
# its own rather than being approximated by one that exists, because the range
# layer has to be able to tell it apart from the operation it would otherwise be
# mistaken for: a logarithm is not a logarithm base two, and a caller that
# treated the two the same would reason about a value it does not have.



# The operations the symbolic library has a name for but no reasoning about are
# written here as functions of their own: a square root of a real is a real
# square root, but a square root computed in floating point is not, and code
# that rewrites one into the other is right about the reals and wrong about the
# program.  These do the constant folding and nothing else, which is exactly
# what is sound for both.
def make_opaque_unary_fn(name: str) -> type[sympy.Function]:
    class OpaqueUnaryFn(sympy.Function):
        """
        Unlike the builtin sympy functions on real numbers like sympy.sqrt,
        these equivalents do not do any nontrivial reasoning besides
        constant propagation.  This helps avoid performing transformations
        that are valid for real numbers but are invalid for floating point;
        in particular, while we are willing to make optimizations that change
        numerics for Tensor compute, we are NOT willing to make optimizations
        that change numerics for size compute.
        """

        _handler_name = name
        _unpickler = make_opaque_unary_fn

        @classmethod
        def eval(cls, a: sympy.Expr) -> sympy.Basic | None:
            if isinstance(a, (sympy.Integer, sympy.Float)):
                # Python converts to float64 before computing, c.f.
                # >>> math.sin(2**53+1)
                # -0.848925964814655
                # >>> math.sin(float(2**53+1))
                # -0.848925964814655
                try:
                    return sympy.Float(getattr(math, name)(float(a)))
                # Just use sympy semantics for infinity/overflow, you might get some
                # weird objects but ask silly questions, get silly answers
                except OverflowError:
                    return getattr(sympy, name)(a)
            elif a in [sympy.oo, -sympy.oo, sympy.zoo, -sympy.zoo, int_oo, -int_oo]:
                if a is int_oo:
                    a = sympy.oo
                if a is -int_oo:
                    a = -sympy.oo
                if name == "log2":
                    return sympy.log(a, 2)
                return getattr(sympy, name)(a)
            return None

    nm = "OpaqueUnaryFn_" + name
    OpaqueUnaryFn.__name__ = nm
    OpaqueUnaryFn.__qualname__ = nm

    return OpaqueUnaryFn


def make_opaque_bitwise_fn(name: str, real_op_name: str) -> type[sympy.Function]:
    if name == "bitwise_and":
        prec = PRECEDENCE["BitwiseAnd"]
    elif name == "bitwise_xor":
        prec = PRECEDENCE["BitwiseXor"]
    elif name == "bitwise_or":
        prec = PRECEDENCE["BitwiseOr"]
    else:
        raise AssertionError(f"unrecognized {name}")

    class BitwiseFn(sympy.Function):
        _torch_handler_name = name
        precedence: int = prec
        _torch_unpickler = functools.partial(
            make_opaque_bitwise_fn, real_op_name=real_op_name
        )

        @classmethod
        def eval(cls, a: sympy.Expr, b: sympy.Expr) -> sympy.Basic | None:
            if a.is_Boolean and b.is_Boolean:
                return getattr(operator, real_op_name)(a, b)
            if a.is_Boolean:
                a = sympy.Integer(1 if a else 0)
            if b.is_Boolean:
                b = sympy.Integer(1 if b else 0)
            if isinstance(a, (sympy.Integer, int)) and isinstance(
                b, (sympy.Integer, int)
            ):
                return sympy.Integer(getattr(operator, real_op_name)(int(a), int(b)))
            return None

    nm = "BitwiseFn_" + name
    BitwiseFn.__name__ = nm
    BitwiseFn.__qualname__ = nm

    return BitwiseFn



OpaqueUnaryFn_sqrt = make_opaque_unary_fn("sqrt")

OpaqueUnaryFn_cos = make_opaque_unary_fn("cos")

OpaqueUnaryFn_cosh = make_opaque_unary_fn("cosh")

OpaqueUnaryFn_sin = make_opaque_unary_fn("sin")

OpaqueUnaryFn_sinh = make_opaque_unary_fn("sinh")

OpaqueUnaryFn_tan = make_opaque_unary_fn("tan")

OpaqueUnaryFn_tanh = make_opaque_unary_fn("tanh")

OpaqueUnaryFn_asin = make_opaque_unary_fn("asin")

OpaqueUnaryFn_acos = make_opaque_unary_fn("acos")

OpaqueUnaryFn_atan = make_opaque_unary_fn("atan")

OpaqueUnaryFn_exp = make_opaque_unary_fn("exp")

OpaqueUnaryFn_log = make_opaque_unary_fn("log")

OpaqueUnaryFn_asinh = make_opaque_unary_fn("asinh")

OpaqueUnaryFn_log2 = make_opaque_unary_fn("log2")

BitwiseFn_bitwise_and = make_opaque_bitwise_fn("bitwise_and", "and_")

BitwiseFn_bitwise_or = make_opaque_bitwise_fn("bitwise_or", "or_")

BitwiseFn_bitwise_xor = make_opaque_bitwise_fn("bitwise_xor", "xor")



# The interpreter below walks a symbolic expression and answers each node by
# asking a class that has one method per operation, which is what lets a range
# be computed for an expression rather than only for an operation the region
# called directly.

def handlers() -> dict[type[sympy.Basic], str]:
    # TODO add CeilDiv (it doesn't appear in the index_expr)

    # TODO default to some decompositions if the interpreter doesn't have them
    # like decomposing ModularIndexing or implementing Le(a,b) as Ge(b, a)

    HANDLERS = {
        sympy.Or: "or_",
        sympy.And: "and_",
        sympy.Eq: "eq",
        sympy.Ne: "ne",
        sympy.Lt: "lt",
        sympy.Gt: "gt",
        sympy.Le: "le",
        sympy.Ge: "ge",
        sympy.Not: "not_",
        IntTrueDiv: "int_truediv",
        FloatTrueDiv: "truediv",
        FloorDiv: "floordiv",
        CleanDiv: "floordiv",  # TODO: hmm?
        TruncToFloat: "trunc",
        Where: "where",
        sympy.Add: "add",
        sympy.Mul: "mul",
        FloatPow: "pow",
        PowByNatural: "pow_by_natural",
        # sympy simplifies x * x into Pow(x, 2), so we need to handle this.
        # Do NOT use builtin Pow for floats
        # TODO: There is a hazard here, if we have float * float it will
        # also get turned into Pow(float, 2) but we don't want this because
        # pow_by_natural is assumed to only be integers.  Probably the fix is
        # to add a FloatMul to impede this optimization
        sympy.Pow: "pow_by_natural",
        Mod: "mod",
        PythonMod: "python_mod",
        # TODO: the compiler can generate these, but it's ill-specified which
        # semantics were intended here.  Needs to be cleaned up along with
        # FloorDiv in a bigger cleanup
        sympy.Mod: "mod",
        sympy.Abs: "abs",
        sympy.log: "log",
        sympy.exp: "exp",
        sympy.Min: "minimum",
        sympy.Max: "maximum",
        Min: "minimum",
        Max: "maximum",
        ModularIndexing: "modular_indexing",
        sympy.functions.elementary.piecewise.ExprCondPair: "expr_cond_pair",
        sympy.Piecewise: "piecewise",
        Identity: "identity",
        IsNonOverlappingAndDenseIndicator: "is_non_overlapping_and_dense_indicator",
        RoundDecimal: "round_decimal",
        # TODO: do the rest of the opaque unary functions...
        OpaqueUnaryFn_log2: "log2",
        BitwiseFn_bitwise_and: "bitwise_and",
        BitwiseFn_bitwise_or: "bitwise_or",
        BitwiseFn_bitwise_xor: "bitwise_xor",
    }
    # TODO: This is kind of pointless, we shouldn't be generating sympy.sin
    # for these functions, they should be Opaque instead
    for name in ["cos", "sin", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan"]:
        HANDLERS[getattr(sympy, name)] = name

    return HANDLERS


#: The operations whose arguments may be reordered, and whose result is
#: therefore the same whichever order they are given in.  The interpreter puts
#: them in the order it walks the expression rather than the order the range
#: layer would prefer, so it has to know which of them that does not matter for.
ASSOCIATIVE_OPS = {"minimum", "maximum", "mul", "add", "and_", "or_"}


def _run_sympy_handler(
    analysis: Any,
    args: list[Any],
    expr: sympy.Basic,
    index_dtype: dtype = tp.int64,
) -> Any:
    # Special cases
    if isinstance(expr, sympy.Pow) and isinstance(
        expr.args[1], sympy.core.numbers.Half
    ):
        return analysis.sqrt(args[0])
    if isinstance(expr, ToFloat):
        return analysis.to_dtype(args[0], tp.float64)

    # These handlers are special because they take an extra dtype argument
    # specifying what they should convert to, and we need to appropriately set
    # this up when we convert from Sympy.  A reasonable default when you
    # are translating is to conservatively do int64, and then narrow these
    # arguments later when you discover you can narrow the index range.  But
    # if you already know that 32-bit indexing is OK, you can directly do the
    # sympy translation with index_dtype=int32
    INDEX_DTYPE_HANDLERS = {
        TruncToInt: "trunc_to_int",
        sympy.floor: "floor_to_int",
        sympy.ceiling: "ceil_to_int",
        FloorToInt: "floor_to_int",
        CeilToInt: "ceil_to_int",
        RoundToInt: "round_to_int",
    }
    if (handler_name := INDEX_DTYPE_HANDLERS.get(expr.func)) is not None:
        return getattr(analysis, handler_name)(*args, index_dtype)

    # Fastpath for n-ary integral addition
    if expr.func is sympy.Add and expr.is_integer and hasattr(analysis, "sym_sum"):
        r = analysis.sym_sum(args)
        log.debug("sym_sum(%s) -> %s", args, r)
        return r

    if expr.func is sympy.Pow:
        exp = expr.args[1]
        if exp.is_integer and exp.is_nonnegative:
            handler_name = "pow_by_natural"
        else:
            handler_name = "pow"
    elif hasattr(expr.func, "_torch_handler_name"):
        handler_name = expr.func._torch_handler_name
    else:
        handler_name = handlers()[expr.func]
    handler = getattr(analysis, handler_name)
    try:
        if handler_name in ASSOCIATIVE_OPS:
            if len(args) <= 1:
                raise AssertionError("associative op needs >1 args")
            acc = handler(args[0], args[1])
            for i in range(2, len(args)):
                acc = handler(acc, args[i])
            log.debug("%s(%s) -> %s", handler_name, args, acc)
            return acc
        else:
            r = handler(*args)
            log.debug("%s(%s) -> %s", handler_name, args, r)
            return r
    except NotImplementedError:
        raise
    except Exception:
        log.warning("failed while executing %s(%s)", handler_name, args)
        raise


#: A value that is in no environment, so that a lookup that finds nothing can be
#: told apart from a lookup that found nothing.  A range is a value like any
#: other and a real range may be the empty one, so the sentinel cannot be one of
#: the values being looked up.
_nil = object()


def sympy_interp(
    analysis: Any,
    env: dict[sympy.Symbol, Any],
    expr: sympy.Expr | SympyBoolean,
    *,
    index_dtype: dtype = tp.int64,
    missing_handler: Callable[[sympy.Symbol], object] | None = None,
) -> Any:
    # Handle base cases
    dtype = None
    if isinstance(expr, BooleanAtom):
        dtype = tp.bool
    elif isinstance(expr, sympy.Integer):
        dtype = tp.int64
    elif isinstance(expr, sympy.Number):
        dtype = tp.float64

    if dtype is not None:
        return analysis.constant(expr, dtype)
    elif isinstance(expr, sympy.Symbol):
        if (r := env.get(expr, _nil)) is not _nil:
            return r
        elif missing_handler:
            return missing_handler(expr)
        else:
            raise KeyError(expr)

    # Recursive case
    return _run_sympy_handler(
        analysis,
        [
            sympy_interp(
                analysis,
                env,
                arg,
                index_dtype=index_dtype,
                missing_handler=missing_handler,
            )
            for arg in expr.args
        ],
        expr,
        index_dtype=index_dtype,
    )


# The analysis below answers, for a symbolic expression, the range of values it
# can take.  It is asked about an expression rather than about an operation the
# region called, because the expression is what a load's index is: a bound
# computed for the index expression is what tells the generated code whether
# the load is in range without a check.
class SymPyValueRangeAnalysis:
    """
    It gives bounds on a SymPy operator given bounds on its arguments
    See the function `bound_sympy` for a function that applies this logic to a full SymPy expression
    """

    @staticmethod
    def constant(value, dtype):
        if isinstance(value, ValueRanges):
            if not value.is_singleton():
                raise AssertionError("ValueRanges must be a singleton for constant()")
            value = value.lower
        # NB: value is NOT a sympy expression, it's a constant!
        is_python = isinstance(value, (int, float, bool))
        if not is_python and not isinstance(
            value, (BooleanAtom, sympy.Integer, sympy.Number)
        ):
            raise AssertionError(f"not a supported constant type: {type(value)}")

        # using nan makes subsequent computation throw, and for the purposes of optimization
        # returning -math.inf - math.inf is equivalent to giving up
        if isinstance(value, SupportsFloat) and math.isnan(value):
            if dtype == tp.bool:
                return ValueRanges.unknown_bool()
            elif dtype.is_floating_point:
                return ValueRanges.unknown()
            else:
                return ValueRanges.unknown_int()

        if is_python:
            type_ = dtype_to_type(dtype)
            value = type_(value)
        else:
            # We do a type check on a best-effort basis
            # We don't want to force a cast to sympy.Float if the value is Rational to avoid losing precision
            if dtype == tp.bool:
                if not isinstance(value, BooleanAtom):
                    raise AssertionError("expected BooleanAtom for bool dtype")
            elif dtype.is_floating_point:
                if value.is_finite and not value.is_real:
                    raise AssertionError(
                        "expected float-like sympy value for float dtype"
                    )
            else:
                # dtype is intXX
                if not getattr(value, "is_integer", False):
                    raise AssertionError("expected integer sympy value for int dtype")

        r = ValueRanges.wrap(value)
        return r

    @staticmethod
    def to_dtype(a, dtype, src_dtype=None):
        if dtype == tp.float64:
            return ValueRanges.increasing_map(a, ToFloat)
        elif dtype == tp.bool:
            return ValueRanges.unknown_bool()
        elif not dtype.is_floating_point:
            return ValueRanges.unknown_int()
        return ValueRanges.unknown()

    @staticmethod
    def trunc_to_int(a, dtype):
        return ValueRanges.increasing_map(a, TruncToInt)

    @staticmethod
    def not_(a):
        a = ValueRanges.wrap(a)
        a = a.boolify()
        if not a.is_bool:
            raise AssertionError("not_ expects a boolean ValueRanges")
        return ValueRanges.decreasing_map(a, sympy.Not)

    @staticmethod
    def or_(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.Or)

    @staticmethod
    def and_(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.And)

    @staticmethod
    def _bool_to_int(x):
        if x.is_singleton():
            return ValueRanges.wrap(sympy.Integer(1 if x.lower else 0))
        else:
            return ValueRanges(sympy.Integer(0), sympy.Integer(1))

    @classmethod
    def bitwise_and(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool and b.is_bool:
            return cls.and_(a, b)
        if a.is_bool:
            a = cls._bool_to_int(a)
        if b.is_bool:
            b = cls._bool_to_int(b)
        lower = min(a.lower, b.lower)
        if lower < 0 and lower != -sympy.oo and lower != -int_oo:
            # If both lower bounds are negative, then bits start like
            # 1...10..., so the smallest possible value is 1...101...1.
            # Thus, we need to find the next smallest power of 2 (inclusive).
            try:
                lower = -(1 << int(-lower - 1).bit_length())
            except Exception:
                lower = -int_oo
        else:
            lower = 0
        return ValueRanges(lower, max(a.upper, b.upper))

    @classmethod
    def bitwise_or(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool and b.is_bool:
            return cls.or_(a, b)
        if a.is_bool:
            a = cls._bool_to_int(a)
        if b.is_bool:
            b = cls._bool_to_int(b)
        upper = max(a.upper, b.upper)
        if upper == 0:
            upper = 0
        elif upper > 0 and upper != sympy.oo and upper != int_oo:
            # If both upper bounds are positive, then the largest
            # possible value is 01...1, so we need to find
            # next largest power of 2 (exclusive), minus 1
            try:
                upper = (1 << int(upper).bit_length()) - 1
            except Exception:
                upper = int_oo
        elif upper < 0:
            upper = -1
        return ValueRanges(min(a.lower, b.lower), upper)

    @classmethod
    def bitwise_xor(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool and b.is_bool:
            bounds = {
                a.lower ^ b.lower,
                a.lower ^ b.upper,
                a.upper ^ b.lower,
                a.upper ^ b.upper,
            }

            has_false = any(bound == sympy.false for bound in bounds)
            has_true = any(bound == sympy.true for bound in bounds)

            if has_false and has_true:
                lower, upper = sympy.false, sympy.true
            elif has_true:
                lower = upper = sympy.true
            elif has_false:
                lower = upper = sympy.false
            else:
                raise AssertionError(f"Non-boolean xor result: {bounds}")

            return ValueRanges(lower, upper)
        if a.is_bool:
            a = cls._bool_to_int(a)
        if b.is_bool:
            b = cls._bool_to_int(b)
        if (
            a.lower == a.upper
            and b.lower == b.upper
            and is_sympy_integer(a.lower)
            and is_sympy_integer(b.lower)
        ):
            value_range = a.lower ^ b.lower
            return ValueRanges(value_range, value_range)
        return ValueRanges(-int_oo, int_oo)

    @classmethod
    def eq(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton() and a.lower == b.lower:
            return ValueRanges.wrap(sympy.true)
        # sympy booleans do not support ordered comparison (bool >/< raises), so
        # map them to {0, 1} before the disjoint-range test below.
        if a.is_bool:
            a = cls._bool_to_int(a)
        if b.is_bool:
            b = cls._bool_to_int(b)
        if a.lower > b.upper or b.lower > a.upper:  # ranges disjoint
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @classmethod
    def identity(cls, a):
        return ValueRanges.wrap(a)

    @classmethod
    def lt(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_bool != b.is_bool:
            raise AssertionError(
                "operands must both be boolean ValueRanges or both non-boolean"
            )
        if a.is_bool:
            return cls.and_(cls.not_(a), b)
        else:
            if a.upper < b.lower:
                return ValueRanges.wrap(sympy.true)
            elif a.lower >= b.upper:
                return ValueRanges.wrap(sympy.false)
            return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def gt(cls, a, b):
        return cls.lt(b, a)

    @classmethod
    def le(cls, a, b):
        return cls.not_(cls.gt(a, b))

    @classmethod
    def ge(cls, a, b):
        return cls.not_(cls.lt(a, b))

    @staticmethod
    def add(a, b):
        return ValueRanges.coordinatewise_increasing_map(
            a, b, _keep_float(operator.add)
        )

    @classmethod
    def mul(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        if a.is_bool != b.is_bool:
            raise AssertionError(
                "operands must both be boolean ValueRanges or both non-boolean"
            )
        if a.is_bool:
            return cls.and_(a, b)

        def safe_mul(a, b):
            # Make unknown() * wrap(0.0) == wrap(0.0)
            if a == 0.0 or a == 0:
                return a
            elif b == 0.0 or b == 0:
                return b
            else:
                return a * b

        return ValueRanges.coordinatewise_monotone_map(a, b, _keep_float(safe_mul))

    @staticmethod
    def int_truediv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-int_oo in a or int_oo in a) and (-int_oo in b or int_oo in b)):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(
                a,
                b,
                _keep_float(IntTrueDiv),
            )

    @staticmethod
    def truediv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or (
            (-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)
        ):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(
                a,
                b,
                _keep_float(FloatTrueDiv),
            )

    @staticmethod
    def floordiv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        # TODO We shall assume division is always valid probably.
        if 0 in b:
            if b.lower >= 0 and a.lower >= 0:
                return ValueRanges(0, int_oo)
            if b.upper <= 0 and a.upper <= 0:
                return ValueRanges(0, int_oo)
            if b.upper <= 0 and a.lower >= 0:
                return ValueRanges(-int_oo, 0)
            if b.lower >= 0 and a.upper <= 0:
                return ValueRanges(-int_oo, 0)
            return ValueRanges.unknown_int()
        products = []
        for x, y in itertools.product([a.lower, a.upper], [b.lower, b.upper]):
            r = FloorDiv(x, y)
            if r is sympy.nan:
                products.append((sympy.sign(x) * sympy.sign(y)) * int_oo)
            else:
                products.append(r)

        return ValueRanges(min(products), max(products))

    @classmethod
    def mod(cls, x, y):
        x = ValueRanges.wrap(x)
        y = ValueRanges.wrap(y)
        # nb. We implement C semantics

        def c_mod(a, b):
            ret = abs(a) % abs(b)
            if a < 0:
                ret *= -1
            return ret

        def c_div(a, b):
            x = a / b
            return sympy.Integer(x) if x.is_finite and x not in (int_oo, -int_oo) else x

        if 0 in y:
            return ValueRanges.unknown_int()
        elif y.is_singleton():
            y_val = abs(y.lower)
            # If it wraps, we need to take the whole interval

            # The function is locally linear if they are in the same class
            if c_div(x.lower, y_val) == c_div(x.upper, y_val):
                return ValueRanges.increasing_map(x, lambda u: c_mod(u, y_val))
            if x.upper < 0:
                # Negative case
                return ValueRanges(-y_val + 1, 0)
            elif x.lower > 0:
                # Positive case
                return ValueRanges(0, y_val - 1)
            else:
                # Mixed case
                lower = max(-y_val + 1, x.lower)
                upper = min(y_val - 1, x.upper)
                return ValueRanges(lower, upper)
        else:
            # Too difficult, we bail out
            upper = cls.abs(y).upper - 1
            return ValueRanges(-upper, upper)

    @classmethod
    def python_mod(cls, x, y):
        """Python-style modulo: result has same sign as divisor.

        Assumes valid input where y is never 0.
        - When y > 0: result is in [0, y - 1]
        - When y < 0: result is in [y + 1, 0]
        """

        x = ValueRanges.wrap(x)
        y = ValueRanges.wrap(y)
        if x.lower >= 0 and y.lower >= 0:
            return SymPyValueRangeAnalysis.mod(x, y)
        lower = y.lower + 1 if y.lower < 0 else 0
        upper = y.upper - 1 if y.upper > 0 else 0
        return ValueRanges(lower, upper)

    @classmethod
    def modular_indexing(cls, a, b, c):
        return cls.mod(cls.floordiv(a, b), c)

    @classmethod
    def is_non_overlapping_and_dense_indicator(cls, *args):
        return ValueRanges.unknown_int()

    @classmethod
    def pow_by_natural(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton():
            return ValueRanges.wrap(safe_pow(a.lower, b.lower))
        # NB: Exclude zero, because zero is special
        elif a.lower >= 1:
            # We should know that b >= 0 but we may have forgotten this fact due
            # to replacements, so don't assert it, but DO clamp it to prevent
            # degenerate problems
            return ValueRanges.coordinatewise_increasing_map(
                a, b & ValueRanges(0, int_oo), PowByNatural
            )
        elif b.is_singleton():
            if b.lower % 2 == 0:
                # x^n where n is even
                return ValueRanges.convex_min_zero_map(
                    a, lambda x: safe_pow(x, b.lower)
                )
            else:
                # x^n where n is odd
                return ValueRanges.increasing_map(a, lambda x: safe_pow(x, b.lower))
        else:
            # a is potentially negative, and we don't know if the exponent is
            # even or odd.  So just conservatively set the upper and lower
            # bound based on what the maximum absolute value could be, in both
            # directions
            max_base = max(a.upper, -a.lower)
            return ValueRanges(
                -(safe_pow(max_base, b.upper)), safe_pow(max_base, b.upper)
            )

    @classmethod
    def pow(cls, a, b):
        return ValueRanges.unknown()

        # We could implement all this, but for floating point pow, is there
        # really a point?
        """
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        # Not implemented yet. It's a bit tricky
        # If you want to implement it, compute the partial derivatives of a ** b
        # and check the ranges where the function is increasing / decreasing
        # Another non-tight way of doing this is defaulting to doing noting that for a > 0,  a ** b == exp(b * log(a))
        # If this second option is implemented, be careful about the types and possible infinities here and there.
        if not b.is_singleton():
            return ValueRanges.unknown()

        b = b.lower
        if a.is_singleton():
            a = a.lower
            r = a**b
            if not r.is_finite:
                return ValueRanges.unknown()
            return ValueRanges.wrap(r)

        if b == 0:
            if not a.lower.is_finite:
                return ValueRanges.unknown()
            return ValueRanges.wrap(1.0)

        if b < 0:
            a = cls.reciprocal(a)
            b = -b

        if a == ValueRanges.unknown():
            return ValueRanges.unknown()

        # If the base is positive, then we're good, otherwise nothing's defined
        if a.lower >= 0:
            return ValueRanges.increasing_map(a, lambda x: x**b)
        else:
            return ValueRanges.unknown()
        """

    @staticmethod
    def reciprocal(x):
        """Needed as it's used in pow, but it won't appear on a SymPy expression"""
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges.unknown()
        else:
            return ValueRanges.decreasing_map(x, lambda y: FloatTrueDiv(1.0, y))

    @staticmethod
    def abs(x):
        return ValueRanges.convex_min_zero_map(x, abs)

    @staticmethod
    def exp(x):
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_exp)

    @staticmethod
    def log(x):
        x = ValueRanges.wrap(x)
        if x.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_log)

    @staticmethod
    def log2(x):
        x = ValueRanges.wrap(x)
        if x.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_log2)

    @classmethod
    def minimum(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool != b.is_bool:
            raise AssertionError(
                "operands must both be boolean ValueRanges or both non-boolean"
            )
        if a.is_bool:
            return cls.and_(a, b)
        return cls.min_or_max(a, b, sympy.Min)

    @classmethod
    def maximum(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool != b.is_bool:
            raise AssertionError(
                "operands must both be boolean ValueRanges or both non-boolean"
            )
        if a.is_bool:
            return cls.or_(a, b)
        return cls.min_or_max(a, b, sympy.Max)

    @staticmethod
    def min_or_max(a, b, fn):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        return ValueRanges.coordinatewise_increasing_map(a, b, fn)

    @classmethod
    def floor_to_int(cls, x, dtype):
        return ValueRanges.increasing_map(x, sympy.functions.elementary.integers.floor)

    @classmethod
    def ceil_to_int(cls, x, dtype):
        return ValueRanges.increasing_map(
            x, sympy.functions.elementary.integers.ceiling
        )

    # I think these implementations are sound.  The hazard here is that sympy
    # will carry out the floor/ceil at too high precision and then something
    # bad will happen when we convert it to float.
    #
    # For truncation, the implementation is clearly sound, because the desired
    # target float is always exactly representable, since you're just chopping
    # off bits the mantissa.  But what about ceil/floor?
    #
    # The important constraint here is that we're not defining floor on
    # arbitrary real numbers, only representable float numbers.  So we can
    # take advantage of the fact that before we reach the first
    # unrepresentable integer in floating point space, we have the range of
    # numbers corresponding to exponent zero: all integers, with no fractional
    # amounts.  floor/ceil is an identity operation in this case.  In the
    # range below here, representable floating point numbers are spaced
    # exactly 1/2 apart, and notably, both the floor/ceil are defined floating
    # point numbers.  There is no "gap" as you step up to the next exponent.

    @classmethod
    def floor(cls, x):
        return ValueRanges.increasing_map(
            x, _keep_float(sympy.functions.elementary.integers.floor)
        )

    @classmethod
    def ceil(cls, x):
        return ValueRanges.increasing_map(
            x, _keep_float(sympy.functions.elementary.integers.ceiling)
        )

    @classmethod
    def round_decimal(cls, number, ndigits):
        if not ndigits.is_singleton():
            return ValueRanges.unknown()

        ndigits = ndigits.lower
        # We can't use functools.partial here since sympy doesn't support keyword arguments, but we have to bind
        # the second parameter.
        fn = lambda number: RoundDecimal(number, ndigits)  # noqa: E731

        return ValueRanges.increasing_map(number, fn)

    @classmethod
    def round_to_int(cls, number, dtype):
        return ValueRanges.increasing_map(number, RoundToInt)

    # It's used in some models on symints
    @staticmethod
    def sqrt(x):
        x = ValueRanges.wrap(x)
        if x.lower < 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_sqrt)

    @staticmethod
    def where(a, b, c):
        b = ValueRanges.wrap(b)
        c = ValueRanges.wrap(c)
        a = a.boolify()
        # We sometimes write unknown without specifying the type correctly
        # In particular, we do that when initialising the bounds for loads in bounds.py
        if b.is_bool != c.is_bool and ValueRanges.unknown() not in (b, c):
            raise AssertionError(
                "where() requires b and c to have the same boolean-ness or allow unknown()"
            )
        if b.is_bool:
            return ValueRanges(sympy.And(b.lower, c.lower), sympy.Or(b.upper, c.upper))
        else:
            return ValueRanges(sympy.Min(b.lower, c.lower), sympy.Max(b.upper, c.upper))

    # expr_cond_pair is used to represent a single (expr, condition) pair in piecewise.
    # We just return the value range of the expression and its corresponding condition as a tuple
    # and defer the analysis to piecewise
    @staticmethod
    def expr_cond_pair(a, b):
        b = b.boolify()
        return (a, b)

    # piecewise function can be used to convert a SymBool to SymInt:
    # int_expr = Piecewise((1, bool_expr), (0, True)), it evaluates to 1 when sym_bool is True and 0 otherwise.
    #
    # ranges is a sequence of (expr_range, condition_range) pairs. The range pair is constructed in expr_cond_pair.
    # The ValueRange of Piecewise is just the union of all expr ranges whose condition expr can be True.
    @staticmethod
    def piecewise(*ranges):
        init_range = None
        for expr_range, cond_range in ranges:
            if sympy.true in cond_range:
                if init_range is None:
                    init_range = expr_range
                else:
                    init_range = init_range | expr_range
        return init_range

    @staticmethod
    def cos(x):
        # TODO: We should tighten value ranges
        # If input range span is pi + 2*pi*k, then output range is (-1, 1)
        # otherwise the minimum of the value of the function on the extremes
        return ValueRanges(-1.0, 1.0)

    @staticmethod
    def cosh(x):
        return ValueRanges(0.0, sympy.oo)
        """
        x = ValueRanges.wrap(x)
        if x.lower > 0:
            return ValueRanges.increasing_map(x, OpaqueUnaryFn_cosh)
        elif x.upper < 0:
            return ValueRanges.decreasing_map(x, OpaqueUnaryFn_cosh)
        return ValueRanges(0.0, sympy.oo)
        """

    @staticmethod
    def sin(x):
        # TODO: We should tighten value ranges
        # See details on cos
        return ValueRanges(-1.0, 1.0)

    @staticmethod
    def sinh(x):
        # return ValueRanges.increasing_map(x, OpaqueUnaryFn_sinh)
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def tan(x):
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def tanh(x):
        # return ValueRanges.increasing_map(x, OpaqueUnaryFn_tanh)
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def asin(x):
        return ValueRanges(-sympy.oo, sympy.oo)
        """
        x = ValueRanges.wrap(x)
        if -1 <= x.lower and x.upper <= 1:
            return ValueRanges.increasing_map(x, OpaqueUnaryFn_asinh)
        return ValueRanges.unknown()
        """

    @staticmethod
    def acos(x):
        return ValueRanges(-sympy.oo, sympy.oo)
        """
        x = ValueRanges.wrap(x)
        if -1 <= x.lower and x.upper <= 1:
            return ValueRanges.decreasing_map(x, OpaqueUnaryFn_acos)
        return ValueRanges.unknown()
        """

    @staticmethod
    def atan(x):
        return ValueRanges(-sympy.oo, sympy.oo)
        # return ValueRanges.increasing_map(x, OpaqueUnaryFn_atan)

    @staticmethod
    def trunc(x):
        return ValueRanges.increasing_map(x, TruncToFloat)


def _current_var_to_range() -> dict:
    """Whatever the region being compiled already knows about its extents.

    A range computed here is only as tight as what is known, and the region
    usually knows more than the expression does, so its knowledge is used when
    there is a region.  Nothing here requires one: an expression can be bounded
    with no region in context, and the answer is then only as tight as the
    expression itself allows.
    """

    try:
        from ....compiler.backends.stax.loops import V
    except Exception:
        return {}
    graph = getattr(V, "graph", None)
    sizevars = getattr(graph, "sizevars", None)
    if sizevars is None:
        return {}
    return getattr(getattr(sizevars, "shape_env", None), "var_to_range", None) or {}


def _default_symbol_range(s: sympy.Symbol) -> ValueRanges:
    if s.is_integer:
        if s.is_positive:
            return ValueRanges(1, int_oo)
        if s.is_nonnegative:
            return ValueRanges(0, int_oo)
        return ValueRanges.unknown_int()
    return ValueRanges.unknown()



# A sum of terms that are each reduced by a modulus is not the same expression
# as the sum of the reductions, but the two have the same range, and rewriting
# the first into the second is what lets the range of a sum containing a
# reduction be computed at all: a reduction is a term the analysis can reason
# about and the subtraction it came from is not.  Each rewrite is done only
# where it is sound, which the guard below decides term by term -- rewriting a
# sum into something with a wider range would make every bound computed from it
# too wide to be of use.
def _bound_sympy_for_rewrite_guard(
    expr: sympy.Expr, ranges: dict[sympy.Symbol, ValueRanges]
) -> ValueRanges | None:
    try:
        return sympy_interp(
            SymPyValueRangeAnalysis,
            ranges,
            expr,
            missing_handler=_default_symbol_range,
        )
    except (AttributeError, KeyError, NotImplementedError):
        return None


def _definitely_ge_value(value: sympy.Expr, lower: int) -> bool:
    try:
        return bool(value >= lower)
    except TypeError:
        return False


def _definitely_ge(
    expr: sympy.Expr, lower: int, ranges: dict[sympy.Symbol, ValueRanges]
) -> bool:
    if lower == 0 and expr.is_nonnegative:
        return True
    if lower == 1 and expr.is_positive:
        return True

    if isinstance(expr, sympy.Symbol):
        vr = ranges.get(expr)
        if isinstance(vr, ValueRanges):
            return bool(vr.lower >= lower)

    vr = _bound_sympy_for_rewrite_guard(expr, ranges)
    return vr is not None and _definitely_ge_value(vr.lower, lower)


def _mod_rewrite_is_valid(
    mod: type[sympy.Function],
    base: sympy.Expr,
    divisor: sympy.Expr,
    ranges: dict[sympy.Symbol, ValueRanges],
) -> bool:
    if not _definitely_ge(divisor, 1, ranges):
        return False
    return mod is PythonMod or _definitely_ge(base, 0, ranges)


def _rewrite_mod_subtraction(
    base: sympy.Expr,
    divisor: sympy.Expr,
    coeff: sympy.Expr,
) -> sympy.Expr:
    return coeff * FloorDiv(base, divisor) * divisor


def _terms_of_add(expr: sympy.Expr) -> dict[sympy.Expr, sympy.Expr]:
    terms: dict[sympy.Expr, sympy.Expr] = {}
    for term in sympy.Add.make_args(expr):
        coeff, factor = term.as_coeff_Mul()
        terms[factor] = terms.get(factor, sympy.S.Zero) + coeff
    return terms


def _rewrite_mod_subtractions_in_add(
    expr: sympy.Add, ranges: dict[sympy.Symbol, ValueRanges]
) -> sympy.Expr:
    terms = _terms_of_add(expr)

    replacements = []
    for factor, mod_coeff in tuple(terms.items()):
        if mod_coeff == 0 or mod_coeff.is_integer is not True:
            continue

        mod = factor.func
        if mod not in (PythonMod, Mod, sympy.Mod):
            continue

        base, divisor = factor.args
        if not _mod_rewrite_is_valid(mod, base, divisor, ranges):
            continue

        matched_terms = []
        for base_factor, base_coeff in _terms_of_add(base).items():
            if base_coeff.is_integer is not True:
                break

            term_coeff = terms.get(base_factor, sympy.S.Zero)
            needed_coeff = -mod_coeff * base_coeff
            if needed_coeff == 0:
                continue
            if term_coeff == 0 or term_coeff * needed_coeff <= 0:
                break
            if needed_coeff > 0 and term_coeff < needed_coeff:
                break
            if needed_coeff < 0 and term_coeff > needed_coeff:
                break

            matched_terms.append((base_factor, needed_coeff))
        else:
            for base_factor, needed_coeff in matched_terms:
                terms[base_factor] -= needed_coeff
            terms[factor] = sympy.S.Zero
            replacements.append(_rewrite_mod_subtraction(base, divisor, -mod_coeff))

    if not replacements:
        return expr

    new_terms = []
    for factor, coeff in terms.items():
        if coeff == 0:
            continue
        if factor == 1:
            new_terms.append(coeff)
        elif coeff == 1:
            new_terms.append(factor)
        else:
            new_terms.append(coeff * factor)

    return sympy.Add(*new_terms, *replacements)


def _rewrite_for_value_range_analysis(
    expr: sympy.Basic, ranges: dict[sympy.Symbol, ValueRanges]
) -> sympy.Basic:
    """Preserve simple dependencies that interval arithmetic would lose."""
    if not expr.args:
        return expr

    args = tuple(_rewrite_for_value_range_analysis(arg, ranges) for arg in expr.args)
    if args != expr.args:
        expr = expr.func(*args)

    if isinstance(expr, sympy.Add):
        return _rewrite_mod_subtractions_in_add(expr, ranges)

    return expr



def bound_sympy(
    expr: sympy.Expr, ranges: dict[sympy.Symbol, ValueRanges] | None = None
) -> ValueRanges:
    log.debug(
        "bound_sympy(%s)%s",
        expr,
        "\n"
        + "\n".join(
            f"  {k}: {r}" for k, r in (ranges or {}).items() if k in expr.free_symbols
        ),
    )
    if isinstance(expr, sympy.Number):
        return ValueRanges.wrap(expr)

    ranges = ranges or {}

    # Whatever the region already knows about its extents is used for the
    # symbols this expression mentions, so a bound computed here is as tight as
    # the region can make it.
    known = _current_var_to_range()
    if known:
        ranges = {**known, **ranges} if ranges else dict(known)

    if expr.has(PythonMod, Mod, sympy.Mod):
        expr = _rewrite_for_value_range_analysis(expr, ranges)

    return sympy_interp(
        SymPyValueRangeAnalysis, ranges, expr, missing_handler=_default_symbol_range
    )


class SymT(Enum):
    """What a symbol stands for, told apart by the first letters of its name.

    A symbol's name is the only thing that travels with it, so what it stands
    for is recorded in that name: a size is named with an s, a loop index with
    an i, a value that came out of the data with a u.  Asking what a symbol is
    is therefore a question about its name, and it is asked often enough, and
    cheaply enough, that the prefixes are stated once here.
    """

    SIZE = auto()
    FLOAT = auto()
    UNBACKED_INT = auto()
    UNBACKED_FLOAT = auto()
    TMP = auto()
    INDIRECT = auto()
    PRECOMPUTED_SIZE = auto()
    INDEX = auto()
    R0_INDEX = auto()
    R1_INDEX = auto()
    TEMPLATE_INDEX = auto()
    XBLOCK = auto()
    YBLOCK = auto()
    ZBLOCK = auto()
    VIEW = auto()
    HALIDE = auto()


#: The letters a symbol's name begins with, one per kind of symbol.
prefix_str = _PREFIX_STR = {
    SymT.SIZE: "s",
    SymT.UNBACKED_INT: "u",
    # The z is here so that a float prefix cannot be confused with a size one.
    SymT.FLOAT: "zf",
    SymT.UNBACKED_FLOAT: "zuf",
    SymT.TMP: "tmp",
    SymT.PRECOMPUTED_SIZE: "ps",
    SymT.INDEX: "i",
    SymT.R0_INDEX: "r0_",
    SymT.R1_INDEX: "r1_",
    SymT.TEMPLATE_INDEX: "idx",
    SymT.XBLOCK: "x",
    SymT.YBLOCK: "y",
    SymT.ZBLOCK: "z",
    SymT.INDIRECT: "indirect",
    SymT.VIEW: "view",
    SymT.HALIDE: "h",
}


class IntegerInfinity(sympy.core.numbers.Number, metaclass=sympy.core.singleton.Singleton):
    """A value larger than every integer.

    An integer infinity says an extent has no upper bound, which is a
    different statement from a real infinity: the quantity is still an integer
    and can still be divided, taken a modulus of, and compared, and an
    expression is made of it only when the shape it came from is unknown.
    """

    is_integer = True
    is_commutative = True
    is_number = True
    is_extended_real = True
    is_comparable = True
    is_extended_positive = True
    is_prime = False

    # Dispatched to before the plain numbers, so that a combination with one
    # of those is handled here rather than as a generic number.
    _op_priority = 100.0

    __slots__: tuple = ()

    def _sympystr(self, printer) -> str:
        return "int_oo"

    def _eval_evalf(self, prec):
        return sympy.Float("inf")

    def _as_mpf_val(self, prec):
        return sympy.Float("inf")._as_mpf_val(prec)

    def _eval_subs(self, old, new):
        if self == old:
            return new
        return None

    def __add__(self, other):
        if isinstance(other, sympy.Number) and sympy.core.parameters.global_parameters.evaluate:
            if other in (sympy.S.Infinity, sympy.S.NegativeInfinity):
                return other
            if other is getattr(sympy.S, "NaN", None):
                return sympy.S.NaN
            return self
        return sympy.core.numbers.Number.__add__(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, sympy.Number) and sympy.core.parameters.global_parameters.evaluate:
            if other is sympy.S.Infinity:
                return sympy.S.NegativeInfinity
            if other is sympy.S.NegativeInfinity:
                return sympy.S.Infinity
            if other is self or other is sympy.S.NaN:
                return sympy.S.NaN
            return self
        return sympy.core.numbers.Number.__sub__(self, other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        if isinstance(other, sympy.Number) and sympy.core.parameters.global_parameters.evaluate:
            if other.is_zero or other is sympy.S.NaN:
                return sympy.S.NaN
            if other.is_extended_positive:
                return self
            return _NEG_INT_OO
        return sympy.core.numbers.Number.__mul__(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, sympy.Number) and sympy.core.parameters.global_parameters.evaluate:
            if other in (
                sympy.S.Infinity,
                _INT_OO,
                sympy.S.NegativeInfinity,
                _NEG_INT_OO,
                sympy.S.NaN,
            ):
                return sympy.S.NaN
            if other.is_extended_nonnegative:
                return sympy.S.Infinity
            return sympy.S.NegativeInfinity
        return sympy.core.numbers.Number.__truediv__(self, other)

    def __abs__(self):
        return _INT_OO

    def __neg__(self):
        return _NEG_INT_OO

    def _eval_power(self, expt):
        if expt.is_extended_positive:
            return _INT_OO
        if expt.is_extended_negative:
            return sympy.S.Zero
        if expt is sympy.S.NaN:
            return sympy.S.NaN
        if expt is sympy.S.ComplexInfinity:
            return sympy.S.NaN
        if expt.is_extended_real is False and expt.is_number:
            from sympy.functions.elementary.complexes import re

            expt_real = re(expt)
            if expt_real.is_positive:
                return sympy.S.ComplexInfinity
            if expt_real.is_negative:
                return sympy.S.Zero
            if expt_real.is_zero:
                return sympy.S.NaN

            return self**expt.evalf()
        return None

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other) -> bool:
        return other is _INT_OO

    def __ne__(self, other) -> bool:
        return other is not _INT_OO

    def __gt__(self, other):
        if other is sympy.S.Infinity:
            return sympy.false
        elif other is _INT_OO:
            return sympy.false
        else:
            return sympy.true

    def __ge__(self, other):
        if other is sympy.S.Infinity:
            return sympy.false
        elif other is _INT_OO:
            return sympy.true
        else:
            return sympy.true

    def __lt__(self, other):
        if other is sympy.S.Infinity:
            return sympy.true
        elif other is _INT_OO:
            return sympy.false
        else:
            return sympy.false

    def __le__(self, other):
        if other is sympy.S.Infinity:
            return sympy.true
        elif other is _INT_OO:
            return sympy.true
        else:
            return sympy.false

    def __mod__(self, other):
        if not isinstance(other, sympy.Expr):
            return NotImplemented
        if other is sympy.S.Infinity:
            return sympy.S.NaN
        if other is sympy.S.NegativeInfinity:
            return sympy.S.NaN
        if other is _INT_OO:
            return sympy.S.NaN
        if other is _NEG_INT_OO:
            return sympy.S.NaN
        if other is sympy.S.NaN:
            return other
        return sympy.core.numbers.Number.__mod__(self, other)

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self


class NegativeIntegerInfinity(sympy.core.numbers.Number, metaclass=sympy.core.singleton.Singleton):
    """A value smaller than every integer."""

    _op_priority = 100.0

    is_integer = True
    is_extended_real = True
    is_commutative = True
    is_comparable = True
    is_extended_negative = True
    is_number = True
    is_prime = False

    __slots__: tuple = ()

    def _eval_subs(self, old, new):
        if self == old:
            return new
        return None

    def _sympystr(self, printer) -> str:
        return "-int_oo"

    def _eval_evalf(self, prec):
        return sympy.Float("-inf")

    def _as_mpf_val(self, prec):
        return sympy.Float("-inf")._as_mpf_val(prec)

    def __add__(self, other):
        if isinstance(other, sympy.Number) and sympy.core.parameters.global_parameters.evaluate:
            if other is sympy.S.NegativeInfinity:
                return other
            if other is _INT_OO:
                return sympy.S.NaN
            return self
        return sympy.core.numbers.Number.__add__(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, sympy.Number) and sympy.core.parameters.global_parameters.evaluate:
            if other is sympy.S.NegativeInfinity:
                return sympy.S.NegativeInfinity
            if other is _INT_OO:
                return sympy.S.NegativeInfinity
            if other is sympy.S.NaN or other is self:
                return sympy.S.NaN
            return self
        return sympy.core.numbers.Number.__sub__(self, other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        if isinstance(other, sympy.Number) and sympy.core.parameters.global_parameters.evaluate:
            if other.is_zero or other is sympy.S.NaN:
                return sympy.S.NaN
            if other.is_extended_negative:
                return self
            return _INT_OO
        return sympy.core.numbers.Number.__mul__(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, sympy.Number) and sympy.core.parameters.global_parameters.evaluate:
            if other in (
                sympy.S.Infinity,
                _INT_OO,
                sympy.S.NegativeInfinity,
                _NEG_INT_OO,
                sympy.S.NaN,
            ):
                return sympy.S.NaN
            if other.is_extended_nonpositive:
                return self
            return sympy.S.Infinity
        return sympy.core.numbers.Number.__truediv__(self, other)

    def __abs__(self):
        return _INT_OO

    def __neg__(self):
        return _INT_OO

    def _eval_power(self, expt):
        if expt is sympy.S.NaN:
            return sympy.S.NaN
        if expt is sympy.S.ComplexInfinity:
            return sympy.S.NaN
        if isinstance(expt, sympy.Integer) and expt.is_extended_positive:
            if expt.is_odd:
                return _NEG_INT_OO
            return _INT_OO
        if expt.is_extended_real is False and expt.is_number:
            from sympy.functions.elementary.complexes import im, re

            s_part = re(expt)
            if s_part is sympy.S.Infinity:
                return sympy.S.ComplexInfinity
            if s_part is sympy.S.NegativeInfinity:
                return sympy.S.ComplexInfinity
            inf_part = _INT_OO**expt
            if inf_part is sympy.S.ComplexInfinity:
                return sympy.S.ComplexInfinity
            return s_part * inf_part
        return None

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other) -> bool:
        return other is _NEG_INT_OO

    def __ne__(self, other) -> bool:
        return other is not _NEG_INT_OO

    def __gt__(self, other):
        if other is sympy.S.NegativeInfinity:
            return sympy.true
        elif other is _NEG_INT_OO:
            return sympy.false
        else:
            return sympy.false

    def __ge__(self, other):
        if other is sympy.S.NegativeInfinity:
            return sympy.true
        elif other is _NEG_INT_OO:
            return sympy.true
        else:
            return sympy.false

    def __lt__(self, other):
        if other is sympy.S.NegativeInfinity:
            return sympy.false
        elif other is _NEG_INT_OO:
            return sympy.false
        else:
            return sympy.true

    def __le__(self, other):
        if other is sympy.S.NegativeInfinity:
            return sympy.true
        elif other is _NEG_INT_OO:
            return sympy.true
        else:
            return sympy.false

    def floor(self):
        return self

    def ceiling(self):
        return self


#: The two infinities, by the names the rest of this module reaches them by.
#:
#: They are held here and not looked up through the registry of singletons the
#: symbolic library keeps.  That registry is keyed by class name and shared by
#: everything in the process, so a second package that defines a class of the
#: same name takes the entry over, and a lookup through it would then hand back
#: that package's object: one this module's arithmetic does not recognize as
#: its own infinity, which leaves ``int_oo // 2`` standing as written instead
#: of being ``int_oo``.  The classes are named so as not to take over anyone
#: else's entry either.
_INT_OO = IntegerInfinity()
_NEG_INT_OO = NegativeIntegerInfinity()

#: The names the two classes are imported under.
IntInfinity = IntegerInfinity
NegativeIntInfinity = NegativeIntegerInfinity

#: A value larger than every integer.
int_oo = _INT_OO

def make_symbol(prefix, idx, **kwargs):
    """A symbol of the kind named, numbered.

    What a symbol stands for is carried in its name, so making one of a kind is
    naming it with that kind's letters and a number that tells it apart from
    the others of its kind.
    """

    return sympy.Symbol(f"{_PREFIX_STR[prefix]}{idx}", **kwargs)


def symbol_is_type(sym, prefix) -> bool:
    """Whether a symbol is of the kind, or one of the kinds, named."""

    if not isinstance(sym, sympy.Symbol):
        raise AssertionError("expected sympy.Symbol")
    name_str = sym.name.lower()
    if isinstance(prefix, SymT):
        return name_str.startswith(_PREFIX_STR[prefix])
    return name_str.startswith(tuple(_PREFIX_STR[p] for p in prefix))


def free_symbol_is_type(e, prefix) -> bool:
    """Whether any symbol appearing in an expression is of the kind named."""

    return any(symbol_is_type(v, prefix) for v in e.free_symbols)


def is_power_of_2(n: int) -> bool:
    """Whether the number is a positive power of two."""

    return n > 0 and n & n - 1 == 0


def _int_coefficient(x: sympy.Basic) -> int:
    """The product of the integer factors of a product expression."""

    return math.prod(
        abs(int(arg))
        for arg in sympy.Mul.make_args(x)
        if isinstance(arg, (int, sympy.Integer))
    )


def _int_factor(expr: sympy.Basic) -> int:
    """The greatest common integer factor of the terms of a sum expression."""

    return functools.reduce(
        math.gcd, map(_int_coefficient, sympy.Add.make_args(expr))
    )


def simple_floordiv_gcd(p: sympy.Basic, q: sympy.Basic) -> sympy.Basic:
    """A greatest common divisor from factoring alone, without theories.

    Both arguments are divided by the integer factor they share, and then any
    factor of the divisor that appears as a whole term of the dividend is
    divided out as well.  Factoring the remainder may find more, but only a
    theory-aware factorization would, and that is not worth its cost here.
    """

    gcd: int = math.gcd(_int_factor(p), _int_factor(q))
    if gcd:
        p, q = p / gcd, q / gcd

    base_splits: list[tuple[sympy.Basic, ...]] = list(
        map(sympy.Mul.make_args, sympy.Add.make_args(p))
    )
    for x in sympy.Mul.make_args(q):
        if all(x in base_split for base_split in base_splits):
            gcd = gcd * x  # type: ignore[operator]
    return gcd


def _is_wide_add(expr: sympy.Basic) -> bool:
    return isinstance(expr, sympy.Add) and len(expr.args) > _MAX_ADD_TERMS_FOR_POLY_GCD


def safe_gcd(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    """A greatest common divisor that does not run polynomial gcd on a wide sum."""

    if _is_wide_add(a) or _is_wide_add(b):
        return simple_floordiv_gcd(a, b)
    return sympy.gcd(a, b)


def _equal_valued(a: Any, b: Any) -> bool:
    """Whether two values are known to be the same, asking sympy if unsure."""

    if a is b:
        return True
    if isinstance(a, Boolean):
        return False
    if isinstance(b, Boolean):
        return False
    try:
        if a == b:
            return True
    except TypeError:
        return False
    if isinstance(a, sympy.Basic) and isinstance(b, sympy.Basic):
        return bool(a == b)
    return False


def is_infinite(v: Any) -> bool:
    if isinstance(v, sympy.Basic):
        if v.is_infinite is True:
            return True
    return False


class OrderedSet(MutableSet[T], Reversible[T]):
    """A set that iterates in the order things were added."""

    __slots__ = ("_dict",)

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._dict = dict.fromkeys(iterable, None) if iterable is not None else {}

    @staticmethod
    def _from_dict(dict_inp: dict) -> "OrderedSet":
        s: OrderedSet = OrderedSet()
        s._dict = dict_inp
        return s

    def __contains__(self, elem: object) -> bool:
        return elem in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __reversed__(self):
        return reversed(self._dict)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._dict)})"

    def add(self, elem: T) -> None:
        self._dict[elem] = None

    def discard(self, elem: T) -> None:
        self._dict.pop(elem, None)

    def clear(self) -> None:
        self._dict.clear()

    def pop(self) -> T:
        if not self:
            raise KeyError("pop from an empty set")
        return self._dict.popitem()[0]

    def copy(self) -> "OrderedSet":
        return OrderedSet._from_dict(self._dict.copy())

    def difference(self, *others: Iterable[T]) -> "OrderedSet":
        cls = type(self)
        containers = [list(other) for other in others]
        diff = cls(self)
        for item in list(diff):
            for other in containers:
                if item in other:
                    diff.discard(item)
                    break
        return diff

    def update(self, *others: Iterable[T]) -> "OrderedSet":
        for other in others:
            for item in other:
                self.add(item)
        return self

    def intersection(self, *others: Iterable[T]) -> "OrderedSet":
        cls = type(self)
        if len(others) > 1:
            others = (cls.intersection(cls(self), *others),)
        result = cls(self)
        for other in others:
            for item in list(result):
                if item not in other:
                    result.discard(item)
        return result

    def union(self, *others: Iterable[T]) -> "OrderedSet":
        containers = map(list, others)
        return OrderedSet._from_dict(self._dict).update(*containers)

    def __or__(self, other: Iterable[T]) -> "OrderedSet":
        return self.union(other)

    def __sub__(self, other: Iterable[T]) -> "OrderedSet":
        return self.difference(other)

    def __and__(self, other: Iterable[T]) -> "OrderedSet":
        return self.intersection(other)

    def __xor__(self, other: Iterable[T]) -> "OrderedSet":
        return self.symmetric_difference(other)

    def symmetric_difference(self, other: Iterable[T]) -> "OrderedSet":
        diff1 = OrderedSet._from_dict(self._dict.copy())
        diff2 = OrderedSet(other)
        diff1.difference_update(diff2)
        diff2.difference_update(diff1)
        return diff1.union(diff2)

    def difference_update(self, *others: Iterable[T]) -> "OrderedSet":
        containers = [list(other) for other in others]
        for item in list(self):
            for other in containers:
                if item in other:
                    self.discard(item)
                    break
        return self

    def issubset(self, other: Iterable[T]) -> bool:
        other = set(other)
        return all(item in other for item in self)

    def issuperset(self, other: Iterable[T]) -> bool:
        other = set(other)
        return all(item in self for item in other)

    def __eq__(self, other: object) -> bool:
        # Two sets are the same set whichever order they were filled in, so
        # the comparison is of the members and not of the iteration: an order
        # is how the members are kept, not what they are.
        if isinstance(other, OrderedSet):
            return self._dict == other._dict
        return set(self) == set(other)

    def __ne__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return self._dict != other._dict
        return set(self) != set(other)

    __hash__ = None


class FloorDiv(sympy.Function):
    """Division rounded towards negative infinity, as an expression of its own.

    Writing the operation as its own function rather than as a subtraction of
    a modulus is what lets a divisibility fact rewrite it into an exact
    division, and what lets it print as the division it is.
    """

    nargs: tuple[int, ...] = (2,)
    precedence: int = 35
    is_integer: bool = True

    @property
    def base(self) -> sympy.Basic:
        return self.args[0]

    @property
    def divisor(self) -> sympy.Basic:
        return self.args[1]

    def _sympystr(self, printer: StrPrinter) -> str:
        base = printer.parenthesize(self.base, _ATOM_PRECEDENCE - 0.5)
        divisor = printer.parenthesize(self.divisor, _ATOM_PRECEDENCE - 0.5)
        return f"({base}//{divisor})"

    @classmethod
    def eval(cls, base, divisor):
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")
        if base is _INT_OO or base is _NEG_INT_OO:
            if divisor.is_positive:
                return base
            if divisor.is_negative:
                return (_NEG_INT_OO if base is _INT_OO
                        else _INT_OO)
            return sympy.nan
        if divisor is _INT_OO or divisor is _NEG_INT_OO:
            if base.is_zero:
                return sympy.S.Zero
            return sympy.nan
        if is_infinite(base) and is_infinite(divisor):
            return sympy.nan
        if base is sympy.nan or divisor is sympy.nan:
            return sympy.nan

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and _equal_valued(divisor, 1):
            return base
        if base.is_integer and _equal_valued(divisor, -1):
            return sympy.Mul(base, -1)
        if base == divisor:
            return sympy.S.One

        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (is_infinite(base) or is_infinite(divisor))
        ):
            r = float(base) / float(divisor)
            if r == math.inf:
                return sympy.oo
            if r == -math.inf:
                return -sympy.oo
            if math.isnan(r):
                return sympy.nan
            return sympy.Integer(math.floor(r))
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return sympy.Integer(int(base) // int(divisor))
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        if isinstance(divisor, sympy.Integer):
            quotients = 0
            terms: list[sympy.Expr] = []
            for term in sympy.Add.make_args(base):
                quotient = term / divisor
                quotient_is_integer = quotient.is_integer
                if quotient_is_integer:
                    terms.append(term)
                    quotients += quotient

            if len(terms) != 0:
                return (
                    FloorDiv(base - sympy.Add(*terms, evaluate=False), divisor)
                    + quotients
                )

        try:
            gcd = simple_floordiv_gcd(base, divisor)
            if _equal_valued(gcd, 1) and isinstance(divisor, sympy.Add):
                gcd = safe_gcd(base, divisor)
            if not _equal_valued(gcd, 1):
                return FloorDiv(
                    sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
                )
        except sympy.PolynomialError:
            pass

        return None

    def _eval_is_nonnegative(self) -> bool | None:
        p, q = self.args[:2]
        if all([p.is_integer, q.is_integer, p.is_nonnegative, q.is_nonnegative]):
            return True
        return None


class CleanDiv(FloorDiv):
    """A division that is known not to round."""


class CeilDiv(sympy.Function):
    """Division rounded upwards, as an expression of its own."""

    is_integer = True

    def __new__(cls, base, divisor):
        base = sympy.sympify(base)
        divisor = sympy.sympify(divisor)
        if sympy.gcd(base, divisor) == divisor:
            return CleanDiv(base, divisor)
        return FloorDiv(base + (divisor - 1), divisor)


class Mod(sympy.Function):
    """A modulus with the sign rule of the language the kernels are written in."""

    nargs = (2,)
    precedence: int = 35

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, p, q):
        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")

        if p is S.Zero or p in (q, -q) or q == 1:
            return S.Zero

        if q.is_Number and p.is_Number:
            if p < 0:
                raise AssertionError(p)
            if q < 1:
                raise AssertionError(q)
            return p % q

        if q.is_Number and q == 2:
            if p.is_even:
                return S.Zero
            if p.is_odd:
                return S.One

        r = p / q
        if r.is_integer:
            return S.Zero

        less = p < q
        if less.is_Boolean and bool(less) and r.is_positive:
            return p
        return None


class ModularIndexing(sympy.Function):
    """The element of a strided walk of a buffer: ``(base // divisor) % modulus``."""

    nargs: tuple[int, ...] = (3,)
    is_integer: bool = True
    precedence: int = 35

    @classmethod
    def eval(cls, base, divisor, modulus):
        if base == 0 or modulus == 1:
            return sympy.S.Zero
        if (
            isinstance(base, sympy.Integer)
            and isinstance(divisor, sympy.Integer)
            and isinstance(modulus, sympy.Integer)
        ):
            return (base // divisor) % modulus

        try:
            if divisor != 1:
                gcd = safe_gcd(base, divisor)
                if gcd != 1:
                    return ModularIndexing(
                        sympy.simplify(base / gcd),
                        sympy.simplify(divisor / gcd),
                        modulus,
                    )
        except sympy.PolynomialError:
            pass

        if isinstance(base, sympy.Add) and not _is_wide_add(base):
            new_terms: list[sympy.Integer] = []
            all_nonnegative: bool = True
            for term in base.args:
                if safe_gcd(term, modulus * divisor) != modulus * divisor:
                    if term.is_nonnegative is not True:
                        all_nonnegative = False
                        break
                    new_terms.append(term)

            if len(new_terms) != len(base.args) and all_nonnegative:
                return ModularIndexing(sum(new_terms), divisor, modulus)

        if isinstance(base, FloorDiv):
            return ModularIndexing(base.args[0], base.args[1] * divisor, modulus)

        return None

    def _eval_is_nonnegative(self) -> bool | None:
        p, q = self.args[:2]
        if p.is_nonnegative is not None and q.is_nonnegative is not None:
            return p.is_nonnegative == q.is_nonnegative
        return None


#: The two clamps come from the symbolic library itself: folding a clamp over
#: known values is a lattice operation, and that is what the library's own
#: implementation already does.
Max = sympy.Max
Min = sympy.Min


def _is_symbols_binary_summation(expr: sympy.Expr) -> bool:
    # No need to check that two args are not the same, since expr is pre-optimized but we do it anyway.
    return (
        isinstance(expr, sympy.Expr)
        and expr.is_Add
        and len(expr._args) == 2
        and expr._args[0].is_symbol
        and expr._args[1].is_symbol
        and expr._args[0] is not expr._args[1]
    )

class MinMaxBase(Expr, LatticeOp):  # type: ignore[misc]
    def __new__(cls, *original_args: sympy.Expr, **assumptions: bool) -> sympy.Basic:
        from sympy.core.parameters import global_parameters

        evaluate = assumptions.pop("evaluate", global_parameters.evaluate)
        args = (sympify(arg) for arg in original_args)

        # See the comment in _satisfy_unique_summations_symbols.
        unique_summations_symbols = (
            None
            if not evaluate
            else cls._satisfy_unique_summations_symbols(original_args)
        )

        if evaluate:
            try:
                # first standard filter, for cls.zero and cls.identity
                # also reshape Max(a, Max(b, c)) to Max(a, b, c)
                args = frozenset(cls._new_args_filter(args))  # type: ignore[assignment]
            except ShortCircuit:
                return cls.zero  # type: ignore[attr-defined]

            # No need to run _collapse_arguments and _find_localzeros, see the comment
            # in _satisfy_unique_summations_symbols.
            if unique_summations_symbols is None:
                # remove redundant args that are easily identified
                args = cls._collapse_arguments(args, **assumptions)

                # find local zeros
                args = cls._find_localzeros(args, **assumptions)

        args = frozenset(args)

        if not args:
            return cls.identity  # type: ignore[attr-defined]

        if len(args) == 1:
            return list(args).pop()

        # base creation
        obj = Expr.__new__(cls, *ordered(args), **assumptions)
        obj._argset = args

        obj.unique_summations_symbols = unique_summations_symbols
        return obj

    @classmethod
    def _collapse_known_multiplicative_terms(
        cls, values: set[sympy.Expr]
    ) -> sympy.Expr | None:
        if len(values) != 2:
            return None

        a, b = values
        a_coeff, a_term = a.as_coeff_Mul()
        b_coeff, b_term = b.as_coeff_Mul()
        if a_term != b_term:
            return None

        if not (a_coeff.is_comparable and b_coeff.is_comparable):
            return None

        if a_coeff == b_coeff:
            return a

        if a_term.is_nonnegative:
            a_is_smaller = a_coeff < b_coeff
        elif a_term.is_nonpositive:
            a_is_smaller = a_coeff > b_coeff
        else:
            return None

        if cls is Min:
            return a if a_is_smaller else b
        if cls is Max:
            return b if a_is_smaller else a

        raise AssertionError(f"impossible {cls}")

    @classmethod
    def _satisfy_unique_summations_symbols(
        cls, args: Any
    ) -> set[sympy.core.symbol.Symbol] | None:
        """
        One common case in some models is building expressions of the form
        max(max(max(a+b...), c+d), e+f) which is simplified to max(a+b, c+d, e+f, ...).
        For such expressions, we call the Max constructor X times (once for each nested
        max) and the expression gets flattened.

        An expensive cost in constructing those expressions is running _collapse_arguments
        and _find_localzeros. However, those two optimizations are unnecessary when the args
        to max are all of the form a+b, c+d, ..etc where each term uses a unique set of symbols.

        This function is used to detect such properties of the expressions we are building
        and if so inform that we do not need to run those optimizations. To detect those,
        we store a property in the expression that tells that this expression is a min/max
        operation over terms that use unique symbols "unique_summations_symbols". This property
        also memoize the set of symbols used in all the terms to make it faster to detect this
        property inductively.

        When we apply max to add a new term, all we need to do is check if the new term uses
        unique symbols (with respect to existing terms and itself).
        Example:
        t = Max(a+b, c+d) ==> satisfies the property
        Max(t, h+j)       ==> h,j not in [a,b,c,d] => satisfy the property.

        The function returns None if the new expression does not satisfy the unique_summations_symbols
        property. Otherwise, it returns a new set of unique symbols.
        """
        if len(args) != 2:
            return None

        (lhs, rhs) = (
            (args[1], args[0])
            if isinstance(args[1], MinMaxBase)
            else (args[0], args[1])
        )

        if not _is_symbols_binary_summation(rhs):
            return None

        # base case max(a+b, c+d) ==> satisfies the property if a+b and c+d use unique symbols.
        if _is_symbols_binary_summation(lhs):
            return cls._unique_symbols(args)

        # inductive case max(t, h+j) ==> satisfies the property if h, j not in t.unique_summations_symbols
        if isinstance(lhs, MinMaxBase):
            lhs_unique_summations_symbols = getattr(
                lhs, "unique_summations_symbols", None
            )
            if lhs_unique_summations_symbols is not None:
                return cls._unique_symbols([rhs], lhs_unique_summations_symbols)

        return None

    @classmethod
    def _unique_symbols(
        cls,
        args: "Iterable[sympy.Expr]",
        initial_set: set[sympy.core.symbol.Symbol] | None = None,
    ) -> set[sympy.core.symbol.Symbol] | None:
        """
        Return seen_symbols if all atoms in all args are all unique symbols,
        else returns None. initial_set can be used to represent initial value for seen_symbols
        """
        seen_symbols = set() if initial_set is None else initial_set.copy()
        for arg in args:
            for element in arg.atoms():
                if not isinstance(element, sympy.core.symbol.Symbol):
                    return None
                elif element in seen_symbols:
                    return None
                else:
                    seen_symbols.add(element)
        return seen_symbols

    @classmethod
    def _collapse_arguments(
        cls, args: "Iterable[sympy.Expr]", **assumptions: bool
    ) -> "Iterable[sympy.Expr]":
        """Remove redundant args.

        Examples
        ========

        >>> from sympy import Min, Max
        >>> from sympy.abc import a, b, c, d, e

        Any arg in parent that appears in any
        parent-like function in any of the flat args
        of parent can be removed from that sub-arg:

        >>> Min(a, Max(b, Min(a, c, d)))
        Min(a, Max(b, Min(c, d)))

        If the arg of parent appears in an opposite-than parent
        function in any of the flat args of parent that function
        can be replaced with the arg:

        >>> Min(a, Max(b, Min(c, d, Max(a, e))))
        Min(a, Max(b, Min(a, c, d)))
        """
        if not args:
            return args
        args = list(ordered(args))
        if cls is Min:
            other = Max
        else:
            other = Min  # type: ignore[assignment]

        # find global comparable max of Max and min of Min if a new
        # value is being introduced in these args at position 0 of
        # the ordered args
        if args[0].is_number:
            sifted = mins, maxs = [], []  # type: ignore[var-annotated]
            for i in args:
                for v in walk(i, Min, Max):
                    if v.args[0].is_comparable:
                        sifted[isinstance(v, Max)].append(v)
            small = Min.identity
            for i in mins:
                v = i.args[0]
                if v.is_number and (v < small) == True:  # noqa: E712
                    small = v
            big = Max.identity
            for i in maxs:
                v = i.args[0]
                if v.is_number and (v > big) == True:  # noqa: E712
                    big = v
            # at the point when this function is called from __new__,
            # there may be more than one numeric arg present since
            # local zeros have not been handled yet, so look through
            # more than the first arg
            if cls is Min:
                for arg in args:
                    if not arg.is_number:
                        break
                    if (arg < small) == True:  # noqa: E712
                        small = arg
            elif cls == Max:
                for arg in args:
                    if not arg.is_number:
                        break
                    if (arg > big) == True:  # noqa: E712
                        big = arg
            T = None
            if cls is Min:
                if small != Min.identity:
                    other = Max
                    T = small
            elif big != Max.identity:
                other = Min  # type: ignore[assignment]
                T = big
            if T is not None:
                # remove numerical redundancy
                for i in range(len(args)):
                    a = args[i]
                    if isinstance(a, other):
                        a0 = a.args[0]
                        if (  # noqa: E712
                            (a0 > T) if other == Max else (a0 < T)
                        ) == True:
                            args[i] = cls.identity  # type: ignore[attr-defined]

        # remove redundant symbolic args
        def do(ai: sympy.Expr, a: sympy.Expr) -> sympy.Expr:
            if not isinstance(ai, (Min, Max)):
                return ai
            cond = a in ai.args
            if not cond:
                return ai.func(*[do(i, a) for i in ai.args], evaluate=False)
            if isinstance(ai, cls):
                return ai.func(*[do(i, a) for i in ai.args if i != a], evaluate=False)
            return a

        for i, a in enumerate(args):
            args[i + 1 :] = [do(ai, a) for ai in args[i + 1 :]]

        # factor out common elements as for
        # Min(Max(x, y), Max(x, z)) -> Max(x, Min(y, z))
        # and vice versa when swapping Min/Max -- do this only for the
        # easy case where all functions contain something in common;
        # trying to find some optimal subset of args to modify takes
        # too long

        def factor_minmax(args: list[sympy.Expr]) -> list[sympy.Expr]:
            is_other = lambda arg: isinstance(arg, other)  # noqa: E731
            other_args, remaining_args = sift(args, is_other, binary=True)
            if not other_args:
                return args

            # Min(Max(x, y, z), Max(x, y, u, v)) -> {x,y}, ({z}, {u,v})
            arg_sets = [set(arg.args) for arg in other_args]
            common = set.intersection(*arg_sets)
            if not common:
                return args

            new_other_args = list(common)
            arg_sets_diff = [arg_set - common for arg_set in arg_sets]

            # If any set is empty after removing common then all can be
            # discarded e.g. Min(Max(a, b, c), Max(a, b)) -> Max(a, b)
            if all(arg_sets_diff):
                other_args_diff = [other(*s, evaluate=False) for s in arg_sets_diff]
                new_other_args.append(cls(*other_args_diff, evaluate=False))

            other_args_factored = other(*new_other_args, evaluate=False)
            return remaining_args + [other_args_factored]

        if len(args) > 1:
            args = factor_minmax(args)

        return args

    @classmethod
    def _new_args_filter(
        cls, arg_sequence: "Iterable[sympy.Expr]"
    ) -> "Iterator[sympy.Expr]":
        """
        Generator filtering args.

        first standard filter, for cls.zero and cls.identity.
        Also reshape ``Max(a, Max(b, c))`` to ``Max(a, b, c)``,
        and check arguments for comparability
        """
        for arg in arg_sequence:
            # pre-filter, checking comparability of arguments
            if (
                not isinstance(arg, Expr)
                or arg.is_extended_real is False
                or (arg.is_number and not arg.is_comparable)
            ):
                raise ValueError(f"The argument '{arg}' is not comparable.")

            if arg == cls.zero:  # type: ignore[attr-defined]
                raise ShortCircuit(arg)
            elif arg == cls.identity:  # type: ignore[attr-defined]
                continue
            elif arg.func == cls:
                yield from arg.args
            else:
                yield arg

    @classmethod
    def _find_localzeros(
        cls, values: "Iterable[sympy.Expr]", **options: bool
    ) -> set[sympy.Expr]:
        """
        Sequentially allocate values to localzeros.

        When a value is identified as being more extreme than another member it
        replaces that member; if this is never true, then the value is simply
        appended to the localzeros.

        Unlike the sympy implementation, we only look for zero and one, we don't
        do generic is connected test pairwise which is slow
        """

        # First, collapse all numeric arguments
        other_values = set()
        num_value = None
        for arg in values:
            if arg.is_Number:
                if num_value is None:
                    num_value = arg
                else:
                    if cls is Max:
                        num_value = max(num_value, arg)
                    elif cls is Min:
                        num_value = min(num_value, arg)
                    else:
                        raise AssertionError(f"impossible {cls}")
            else:
                other_values.add(arg)

        # Special cases when there is only one symbolic value
        if num_value is None:
            collapsed_value = cls._collapse_known_multiplicative_terms(other_values)
            if collapsed_value is not None:
                return {collapsed_value}
            return other_values

        if len(other_values) == 0:
            return {num_value}

        if len(other_values) == 1:
            other_value = next(iter(other_values))
            if num_value in (0.0, 0) and other_value.is_nonnegative:
                return other_values if cls is Max else {num_value}
            if num_value == 1 and other_value.is_positive:
                return other_values if cls is Max else {num_value}

        other_values.add(num_value)
        return other_values

    _eval_is_algebraic = lambda s: _torf(i.is_algebraic for i in s.args)  # noqa: E731
    _eval_is_antihermitian = lambda s: _torf(  # noqa: E731
        i.is_antihermitian for i in s.args
    )
    _eval_is_commutative = lambda s: _torf(  # noqa: E731
        i.is_commutative for i in s.args
    )
    _eval_is_complex = lambda s: _torf(i.is_complex for i in s.args)  # noqa: E731
    _eval_is_composite = lambda s: _torf(i.is_composite for i in s.args)  # noqa: E731
    _eval_is_even = lambda s: _torf(i.is_even for i in s.args)  # noqa: E731
    _eval_is_finite = lambda s: _torf(i.is_finite for i in s.args)  # noqa: E731
    _eval_is_hermitian = lambda s: _torf(i.is_hermitian for i in s.args)  # noqa: E731
    _eval_is_imaginary = lambda s: _torf(i.is_imaginary for i in s.args)  # noqa: E731
    _eval_is_infinite = lambda s: _torf(i.is_infinite for i in s.args)  # noqa: E731
    _eval_is_integer = lambda s: _torf(i.is_integer for i in s.args)  # noqa: E731
    _eval_is_irrational = lambda s: _torf(i.is_irrational for i in s.args)  # noqa: E731
    _eval_is_negative = lambda s: _torf(i.is_negative for i in s.args)  # noqa: E731
    _eval_is_noninteger = lambda s: _torf(i.is_noninteger for i in s.args)  # noqa: E731
    _eval_is_nonnegative = lambda s: _torf(  # noqa: E731
        i.is_nonnegative for i in s.args
    )
    _eval_is_nonpositive = lambda s: _torf(  # noqa: E731
        i.is_nonpositive for i in s.args
    )
    _eval_is_nonzero = lambda s: _torf(i.is_nonzero for i in s.args)  # noqa: E731
    _eval_is_odd = lambda s: _torf(i.is_odd for i in s.args)  # noqa: E731
    _eval_is_polar = lambda s: _torf(i.is_polar for i in s.args)  # noqa: E731
    _eval_is_positive = lambda s: _torf(i.is_positive for i in s.args)  # noqa: E731
    _eval_is_prime = lambda s: _torf(i.is_prime for i in s.args)  # noqa: E731
    _eval_is_rational = lambda s: _torf(i.is_rational for i in s.args)  # noqa: E731
    _eval_is_real = lambda s: _torf(i.is_real for i in s.args)  # noqa: E731
    _eval_is_extended_real = lambda s: _torf(  # noqa: E731
        i.is_extended_real for i in s.args
    )
    _eval_is_transcendental = lambda s: _torf(  # noqa: E731
        i.is_transcendental for i in s.args
    )
    _eval_is_zero = lambda s: _torf(i.is_zero for i in s.args)  # noqa: E731

class Max(MinMaxBase, Application):  # type: ignore[misc]
    r"""
    Return, if possible, the maximum value of the list.
    """

    zero = S.Infinity
    identity = S.NegativeInfinity

    def _eval_is_positive(self):  # type:ignore[override]
        return fuzzy_or(a.is_positive for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self):  # type:ignore[override]
        return fuzzy_or(a.is_nonnegative for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_negative(self):  # type:ignore[override]
        return fuzzy_and(a.is_negative for a in self.args)

class Min(MinMaxBase, Application):  # type: ignore[misc]
    """
    Return, if possible, the minimum value of the list.
    """

    zero = S.NegativeInfinity
    identity = S.Infinity

    def _eval_is_positive(self):  # type:ignore[override]
        return fuzzy_and(a.is_positive for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self):  # type:ignore[override]
        return fuzzy_and(a.is_nonnegative for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_negative(self):  # type:ignore[override]
        return fuzzy_or(a.is_negative for a in self.args)
