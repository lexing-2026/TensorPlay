"""Isolating one thing in an expression, where a general solver gives up.

What is wanted here is narrow: given something like ``a // b == c``, say what
``a`` has to be.  A general solver would manage, but slowly and with results
that are hard to check, so this does the few rearrangements that are needed and
gives up rather than guessing when they do not apply.

Everything here is symbolic arithmetic on shapes.  Nothing is evaluated and
nothing is assumed about values that are not written down.
"""

from __future__ import annotations

import logging

import sympy

from tensorplay.graph.experimental.sympy_functions import FloorDiv

log = logging.getLogger(__name__)

#: For each way of comparing two things, the comparison that says the same thing
#: with the two sides the other way round: greater becomes less.
_MIRROR_REL_OP: dict = {
    sympy.Eq: sympy.Eq,
    sympy.Ne: sympy.Ne,
    sympy.Ge: sympy.Le,
    sympy.Gt: sympy.Lt,
    sympy.Le: sympy.Ge,
    sympy.Lt: sympy.Gt,
}

INEQUALITY_TYPES = (sympy.Gt, sympy.Ge, sympy.Lt, sympy.Le)


def mirror_rel_op(type: type):
    """The comparison that says the same thing with the sides swapped."""

    return _MIRROR_REL_OP.get(type)


def try_solve(
    expr: sympy.Basic,
    thing: sympy.Basic,
    trials: int = 5,
    floordiv_inequality: bool = True,
):
    """Rearrange an expression until the thing wanted is alone on one side.

    Returns the rearranged expression and what the other side became, or
    nothing where the thing wanted cannot be got on its own.  Where the thing
    appears on both sides there is no answer to give, since which copy was meant
    is not written down.

    ``trials`` is how many rearrangements to attempt, since each one may
    expose another.  ``floordiv_inequality`` turns a whole-number division into
    the pair of comparisons that say the same thing, which is what lets a
    division be undone.
    """

    mirror = mirror_rel_op(type(expr))

    # Only a comparison can be rearranged, and only if the comparison has a
    # mirror -- anything else would be rearranged into something that does not
    # say the same thing.
    if not isinstance(expr, sympy.Rel) or mirror is None:
        log.debug("expression with unsupported type: %s", type(expr))
        return None

    lhs_has_thing = expr.lhs.has(thing)
    rhs_has_thing = expr.rhs.has(thing)

    if lhs_has_thing and rhs_has_thing:
        log.debug("thing (%s) found in both sides of expression: %s", thing, expr)
        return None

    # Both sides are tried, the second with the sides swapped, so that the thing
    # wanted ends up on the left either way.
    expressions = []

    if lhs_has_thing:
        expressions.append(expr)
    if rhs_has_thing:
        expressions.append(mirror(expr.rhs, expr.lhs))

    for e in expressions:
        if e is None:
            continue

        if not isinstance(e, sympy.Rel):
            raise AssertionError("expected sympy.Rel")

        for _ in range(trials):
            trial = _try_isolate_lhs(e, thing, floordiv_inequality=floordiv_inequality)
            # No change this time round means there is nothing more to try.
            if trial == e:
                break
            e = trial

        if isinstance(e, sympy.Rel) and e.lhs == thing:
            log.debug("solved: %s ---> %s", expr, e)
            return e, e.rhs

    return None


def _try_isolate_lhs(
    e: sympy.Basic, thing: sympy.Basic, floordiv_inequality: bool
) -> sympy.Basic:
    """One step towards getting the thing wanted alone on the left.

    Two things are done: whatever on the left does not involve the thing wanted
    is moved to the right, and then both sides are divided by whatever factor on
    the left does not involve it.
    """

    op = type(e)

    if isinstance(e, sympy.Rel):
        # Whatever is on the left that is not the thing wanted goes across,
        # subtracted from both sides so the comparison still says the same thing.
        lhs_not_thing = (
            sum(a for a in e.lhs.args if not a.has(thing))
            if isinstance(e.lhs, sympy.Add)
            else 0
        )
        e = op(e.lhs - lhs_not_thing, e.rhs - lhs_not_thing)

    # A product on the left is divided through, which cancels whatever factor
    # does not involve the thing wanted.
    if isinstance(e, sympy.Rel) and isinstance(e.lhs, sympy.Mul):
        lhs, rhs = e.args
        other = sympy.Mul(*[a for a in lhs.args if not a.has(thing)])

        # Whether the factor is negative decides whether dividing by it keeps
        # the comparison the right way round.  Where that is not known, nothing
        # is done, since the result would be a claim about the opposite of what
        # was written.  A right-hand side of zero cannot be divided by either.
        if not (isinstance(e, INEQUALITY_TYPES) and other.is_negative is None) and not (
            not isinstance(e, INEQUALITY_TYPES) and rhs.is_zero
        ):
            lhs = lhs / other
            rhs = rhs / other

            # Dividing an inequality by something negative reverses it.
            if isinstance(e, INEQUALITY_TYPES) and other.is_negative:
                op = mirror_rel_op(op)

            if op is None:
                raise AssertionError("expected op to be not None")
            e = op(lhs, rhs)

    # A whole-number division on the left is turned into the two comparisons
    # that bound it, which is what makes it possible to undo.  This only says
    # the same thing where the divisor is positive and the right-hand side is a
    # whole number.
    if (
        floordiv_inequality
        and isinstance(e, sympy.Rel)
        and isinstance(e.lhs, FloorDiv)
        and e.lhs.divisor.is_positive
        and e.rhs.is_integer
    ):
        # a // b == expr  =>  a is from b*expr up to but not including b*(expr+1)
        if isinstance(e, sympy.Eq):
            numerator, denominator = e.lhs.args
            return sympy.And(
                sympy.Ge(numerator, (e.rhs * denominator)),
                sympy.Lt(numerator, ((e.rhs + 1) * denominator)),
            )
        # a // b != expr  =>  a is outside that range
        if isinstance(e, sympy.Ne):
            numerator, denominator = e.lhs.args
            return sympy.Or(
                sympy.Lt(numerator, (e.rhs * denominator)),
                sympy.Ge(numerator, ((e.rhs + 1) * denominator)),
            )
        # These only hold where the divisor is positive, which is only known
        # for a number.
        # a // b > expr   =>  a >= b * (expr + 1)
        # a // b >= expr  =>  a >= b * expr
        if isinstance(e, (sympy.Gt, sympy.Ge)):
            quotient = e.rhs if isinstance(e, sympy.Ge) else (e.rhs + 1)
            return sympy.Ge(e.lhs.args[0], (quotient * e.lhs.args[1]))
        # a // b < expr   =>  a < b * expr
        # a // b <= expr  =>  a < b * (expr + 1)
        if isinstance(e, (sympy.Lt, sympy.Le)):
            quotient = e.rhs if isinstance(e, sympy.Lt) else (e.rhs + 1)
            return sympy.Lt(e.lhs.args[0], (quotient * e.lhs.args[1]))

    return e
