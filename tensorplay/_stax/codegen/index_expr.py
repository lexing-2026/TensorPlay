"""Integer index expressions for kernel addressing.

A small closed algebra over non-negative integers used to describe the
address arithmetic of generated kernels: load/store offsets, broadcast
strides and loop-carried induction.  Expressions are immutable, hashable
and kept in canonical form so two derivations of the same address agree.

Node surface:
- ``Symbol``      free variable (loop induction, placeholder extent)
- ``Const``       integer literal
- ``Add``         canonical sum ``coeff*term + coeff*term + ...``
- ``Mul``         ``const * expr`` only (non-affine products stay unexpanded)
- ``FloorDiv``    ``floor(a / b)`` for ``b > 0`` (rounds toward -inf)
- ``ModularIndexing``  ``(a // div) % mod`` for ``div, mod > 0``
- ``Where``       split selection, kept opaque

Analyses:
- ``is_affine(expr, var)``  -- degree-1 membership with a constant offset
- ``linear_form``           -- ``(mult, offset)`` when affine
- ``ValueRange`` analysis   -- conservative [lo, hi] intervals, used to
  prove that an addressing chain cannot overflow the buffer it targets.
"""

from __future__ import annotations

import math
from typing import Any


class Expr:
    """Base class: immutable, structural equality, canonical repr."""

    __slots__ = ()

    def __add__(self, other: Any) -> "Expr":
        return _add(self, _lift(other))

    __radd__ = __add__

    def __mul__(self, other: Any) -> "Expr":
        return _mul(self, _lift(other))

    __rmul__ = __mul__

    def __sub__(self, other: Any) -> "Expr":
        return _add(self, _mul(Const(-1), _lift(other)))

    def __rsub__(self, other: Any) -> "Expr":
        return _add(_mul(Const(-1), self), _lift(other))

    def __neg__(self) -> "Expr":
        return _mul(Const(-1), self)

    def __floordiv__(self, other: Any) -> "Expr":
        return floordiv(self, _lift(other))

    def __mod__(self, other: Any) -> "Expr":
        other = _lift(other)
        if not isinstance(other, Const):
            raise TypeError("index-algebra moduli must be integer literals")
        return modular_indexing(self, 1, other.value)

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._key()))

    def _key(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._key()!r}"


class Symbol(Expr):
    """A free integer variable."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = str(name)

    def _key(self) -> Any:
        return self.name


class Const(Expr):
    """An integer literal."""

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def _key(self) -> Any:
        return self.value


_Zero = Const(0)
_One = Const(1)


class Add(Expr):
    """Canonical sum: ``coeff * term`` pairs, constants folded into offset."""

    __slots__ = ("terms", "offset")

    def __init__(self, terms: dict[Expr, int], offset: int) -> None:
        self.terms = terms
        self.offset = offset

    def _key(self) -> Any:
        return (
            tuple(sorted((t._key(), c) for t, c in self.terms.items())),
            self.offset,
        )


class Mul(Expr):
    """``scalar * expr``; factors beyond one scalar stay symbolic."""

    __slots__ = ("scalar", "operand")

    def __init__(self, scalar: int, operand: Expr) -> None:
        self.scalar = scalar
        self.operand = operand

    def _key(self) -> Any:
        return (self.scalar, self.operand)


class FloorDiv(Expr):
    """``floor(a / b)`` with positive divisor (rounds toward -inf)."""

    __slots__ = ("numerator", "divisor")

    def __init__(self, numerator: Expr, divisor: int) -> None:
        self.numerator = numerator
        self.divisor = divisor

    def _key(self) -> Any:
        return (self.numerator, self.divisor)


class ModularIndexing(Expr):
    """``(a // div) % mod`` with positive ``div``/``mod``."""

    __slots__ = ("base", "divisor", "modulus")

    def __init__(self, base: Expr, divisor: int, modulus: int) -> None:
        self.base = base
        self.divisor = divisor
        self.modulus = modulus

    def _key(self) -> Any:
        return (self.base, self.divisor, self.modulus)


class Where(Expr):
    """Opaque split: one of two arms selected by a runtime condition."""

    __slots__ = ("condition", "left", "right")

    def __init__(self, condition: Expr, left: Expr, right: Expr) -> None:
        self.condition = condition
        self.left = left
        self.right = right

    def _key(self) -> Any:
        return (self.condition, self.left, self.right)


def _lift(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        return Const(int(value))
    if isinstance(value, int):
        return Const(value)
    raise TypeError(f"index expression expected, got {type(value)!r}")


def _add(left: Expr, right: Expr) -> Expr:
    terms: dict[Expr, int] = {}
    offset = 0

    def absorb(expr: Expr, coeff: int) -> None:
        nonlocal offset
        if isinstance(expr, Const):
            offset += coeff * expr.value
        elif isinstance(expr, Add):
            for term, c in expr.terms.items():
                absorb(term, coeff * c)
            offset += coeff * expr.offset
        elif isinstance(expr, Mul) and expr.operand is not None:
            absorb(expr.operand, coeff * expr.scalar)
        else:
            terms[expr] = terms.get(expr, 0) + coeff

    absorb(left, 1)
    absorb(right, 1)
    terms = {t: c for t, c in terms.items() if c != 0}
    # Flatten identity (non-negative addressing): d*floor(x/d) + x mod d
    # reconstructs x, so a quotient term absorbs its matching remainder.
    quotients: dict[Expr, list[Expr]] = {}
    for term, coeff in terms.items():
        if isinstance(term, FloorDiv) and coeff == term.divisor:
            quotients.setdefault(term.numerator, []).append(term)
    for base, qterms in quotients.items():
        rem = ModularIndexing(base, 1, qterms[0].divisor)
        if rem in terms and terms[rem] == 1:
            del terms[rem]
            terms[base] = terms.get(base, 0) + 1
            del terms[qterms[0]]
    terms = {t: c for t, c in terms.items() if c != 0}
    if not terms:
        return Const(offset)
    if len(terms) == 1 and offset == 0:
        (term, coeff), = terms.items()
        if coeff == 1:
            return term
        return Mul(coeff, term)
    return Add(terms, offset)


def _mul(left: Expr, right: Expr) -> Expr:
    if isinstance(left, Const) and isinstance(right, Const):
        return Const(left.value * right.value)
    if isinstance(left, Const):
        scalar, operand = left.value, right
    elif isinstance(right, Const):
        scalar, operand = right.value, left
    else:
        # Addressing arithmetic never multiplies two free variables; a
        # quadratic term is a codegen bug, so fail loudly instead of
        # silently emitting wrong addresses.
        raise TypeError("symbol-by-symbol products are not address expressions")
    if scalar == 0:
        return Const(0)
    if scalar == 1:
        return operand
    if isinstance(operand, Const):
        return Const(scalar * operand.value)
    if isinstance(operand, Mul):
        return Mul(scalar * operand.scalar, operand.operand)
    return Mul(scalar, operand)


def floordiv(numerator: Expr, divisor: Expr) -> Expr:
    """Exact-division elimination plus nested-floor tightening."""
    if not isinstance(divisor, Const):
        raise TypeError("index-algebra divisors must be integer literals")
    d = divisor.value
    if d == 0:
        raise ZeroDivisionError("index expression divided by zero")
    if isinstance(numerator, Const):
        return Const(numerator.value // d)
    if d == 1:
        return numerator
    if isinstance(numerator, Mul) and isinstance(numerator.operand, Const):
        inner = numerator.operand.value
        if inner % d == 0:
            return Mul(numerator.scalar, Const(inner // d))
        if numerator.scalar % d == 0:
            return Mul(numerator.scalar // d, numerator.operand)
    return FloorDiv(numerator, d)


def modular_indexing(base: Expr, divisor: int, modulus: int) -> Expr:
    """``(base // divisor) % modulus`` with the same eliminations as floordiv."""
    if divisor <= 0 or modulus <= 0:
        raise ValueError("modular indexing needs positive divisor and modulus")
    if isinstance(base, Const):
        return Const((base.value // divisor) % modulus)
    if divisor == 1 and modulus == 1:
        return Const(0)
    if isinstance(base, ModularIndexing):
        d1, m1 = base.divisor, base.modulus
        d2 = divisor
        if d2 >= m1:
            # The inner remainder lives in [0, m1); dividing by a larger
            # modulus always lands on zero.
            return Const(0)
        if m1 == d2 * modulus:
            # floor(floor(a/d1) mod (d2*m2) / d2) mod m2 == floor(a/(d1*d2)) mod m2.
            return ModularIndexing(base.base, d1 * d2, modulus)
    return ModularIndexing(base, divisor, modulus)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def is_affine(expr: Expr, var: Symbol) -> bool:
    """True when ``expr`` is ``c*var + k`` with constant ``c, k``."""
    return linear_form(expr, var) is not None


def affine_coeff(expr: Expr, var: Symbol) -> int | None:
    """Degree of ``var`` in ``expr`` when affine, else ``None``.

    Unlike :func:`linear_form`, terms independent of ``var`` (loop-invariant
    addresses such as a lane's ``lane*W``) are tolerated and do not enter the
    result.  ``0`` means the address is loop-invariant, ``1`` means unit
    stride, ``k > 1`` a strided walk; ``None`` rejects the address as
    non-affine.
    """
    if isinstance(expr, Const):
        return 0
    if isinstance(expr, Symbol):
        return 1 if expr == var else 0
    if isinstance(expr, Mul):
        inner = affine_coeff(expr.operand, var)
        return None if inner is None else expr.scalar * inner
    if isinstance(expr, Add):
        mult = 0
        for term, coeff in expr.terms.items():
            inner = affine_coeff(term, var)
            if inner is None:
                return None
            if inner != 0:
                if mult != 0:
                    return None
                mult = coeff * inner
        return mult
    return None


def linear_form(expr: Expr, var: Symbol) -> tuple[int, int] | None:
    """Return ``(mult, offset)`` when ``expr == mult*var + offset``."""
    if isinstance(expr, Const):
        return (0, expr.value)
    if expr is var or expr == var:
        return (1, 0)
    if isinstance(expr, Mul):
        inner = linear_form(expr.operand, var)
        if inner is not None and inner[0] == 0:
            return None
        if inner is not None:
            return (expr.scalar * inner[0], expr.scalar * inner[1])
        return None
    if isinstance(expr, Add):
        mult = 0
        offset = expr.offset
        for term, coeff in expr.terms.items():
            inner = linear_form(term, var)
            if inner is None:
                return None
            if inner[0] != 0:
                if mult != 0:
                    return None
                mult = coeff * inner[0]
            offset += coeff * inner[1]
        return (mult, offset)
    return None


def free_symbols(expr: Expr) -> set[Symbol]:
    """Every symbol appearing in the expression."""
    seen: set[Symbol] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Symbol):
            seen.add(node)
        elif isinstance(node, Const):
            pass
        elif isinstance(node, Add):
            stack.append(Const(node.offset))
            stack.extend(node.terms)
        elif isinstance(node, Mul):
            stack.append(node.operand)
        elif isinstance(node, FloorDiv):
            stack.append(node.numerator)
        elif isinstance(node, ModularIndexing):
            stack.append(node.base)
        elif isinstance(node, Where):
            stack.extend((node.condition, node.left, node.right))
    return seen


class ValueRange:
    """Conservative closed interval ``[lo, hi]``; None marks an unknown bound."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: int | None, hi: int | None) -> None:
        self.lo = lo
        self.hi = hi

    def __repr__(self) -> str:
        return f"[{self.lo}, {self.hi}]"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, ValueRange)
            and self.lo == other.lo
            and self.hi == other.hi
        )

    def __hash__(self) -> int:
        return hash((self.lo, self.hi))


def value_range(expr: Expr, ranges: dict[Symbol, ValueRange]) -> ValueRange:
    """Interval propagation for addressing bounds.

    Division/modulo widen to the host range of the argument; ``Where``
    widens to the union of its arms.  Overflow of a buffer is disprovable
    whenever the result's upper bound stays below the buffer extent.
    """
    if isinstance(expr, Const):
        return ValueRange(expr.value, expr.value)
    if isinstance(expr, Symbol):
        known = ranges.get(expr)
        return known if known is not None else ValueRange(None, None)
    if isinstance(expr, Add):
        acc = ValueRange(expr.offset, expr.offset)
        for term, coeff in expr.terms.items():
            inner = value_range(term, ranges)
            if coeff > 0:
                lo = None if inner.lo is None else inner.lo * coeff
                hi = None if inner.hi is None else inner.hi * coeff
            else:
                lo = None if inner.hi is None else inner.hi * coeff
                hi = None if inner.lo is None else inner.lo * coeff
            acc = ValueRange(
                None if acc.lo is None or lo is None else acc.lo + lo,
                None if acc.hi is None or hi is None else acc.hi + hi,
            )
        return acc
    if isinstance(expr, Mul):
        inner = value_range(expr.operand, ranges)
        if expr.scalar > 0:
            lo = None if inner.lo is None else inner.lo * expr.scalar
            hi = None if inner.hi is None else inner.hi * expr.scalar
        else:
            lo = None if inner.hi is None else inner.hi * expr.scalar
            hi = None if inner.lo is None else inner.lo * expr.scalar
        return ValueRange(lo, hi)
    if isinstance(expr, FloorDiv):
        inner = value_range(expr.numerator, ranges)
        if expr.divisor == 1:
            return inner
        if inner.lo is not None and inner.hi is not None:
            lo = math.floor(inner.lo / expr.divisor)
            hi = math.floor(inner.hi / expr.divisor)
            return ValueRange(min(lo, hi), max(lo, hi))
        return ValueRange(None, None)
    if isinstance(expr, ModularIndexing):
        inner = value_range(expr.base, ranges)
        if inner.lo is not None and inner.hi is not None:
            lo = (inner.lo // expr.divisor) % expr.modulus
            hi = (inner.hi // expr.divisor) % expr.modulus
            return ValueRange(min(lo, hi), max(lo, hi))
        return ValueRange(0, expr.modulus - 1)
    if isinstance(expr, Where):
        left = value_range(expr.left, ranges)
        right = value_range(expr.right, ranges)
        return ValueRange(
            None if left.lo is None or right.lo is None else min(left.lo, right.lo),
            None if left.hi is None or right.hi is None else max(left.hi, right.hi),
        )
    return ValueRange(None, None)


def render(expr: Expr) -> str:
    """C-language rendering matching the algebra's semantics exactly.

    ``FloorDiv``/``ModularIndexing`` operate on possibly negative values, so
    they expand to a branchless floor-division pair rather than C's
    truncating operators.
    """
    if isinstance(expr, Const):
        return f"{expr.value}LL"
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, Add):
        parts = []
        if expr.offset:
            parts.append(f"{expr.offset}LL")
        for term, coeff in expr.terms.items():
            rendered = render(term)
            if coeff == 1:
                parts.append(rendered)
            elif coeff == -1:
                parts.append(f"-({rendered})")
            else:
                parts.append(f"{coeff}LL*({rendered})")
        return " + ".join(parts) if parts else "0LL"
    if isinstance(expr, Mul):
        return f"{expr.scalar}LL*({render(expr.operand)})"
    if isinstance(expr, FloorDiv):
        if expr.divisor == 1:
            return render(expr.numerator)
        n = render(expr.numerator)
        d = f"{expr.divisor}LL"
        # Floor division for possibly negative numerators: C's `/`
        # truncates toward zero, so adjust the negative side by d - 1.
        return f"(({n}) < 0 ? -((-({n}) - 1) / {d}) : ({n}) / {d})"
    if isinstance(expr, ModularIndexing):
        if expr.divisor == 1:
            base = render(expr.base)
        else:
            base = render(FloorDiv(expr.base, expr.divisor))
        m = f"{expr.modulus}LL"
        return f"(({base}) % {m} + {m}) % {m}"
    if isinstance(expr, Where):
        return (
            f"(({render(expr.condition)}) ? ({render(expr.left)})"
            f" : ({render(expr.right)}))"
        )
    raise TypeError(f"unrenderable index expression: {expr!r}")
