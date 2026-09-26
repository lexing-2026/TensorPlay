"""Writing a value out as text, faster than the general writer does.

The general writer simplifies as it goes and produces the tidiest form it can.
That is what code generation wants, but not what comparing two sizes wants: a
key that says whether two buffers are the same size only has to be equal when
the sizes are equal, and it is asked about many buffers.  So this writes the
expression as it stands, without simplifying, which is much quicker and
sometimes uglier.  A key built this way is not something to show a person.
"""

from __future__ import annotations

import sympy

from tensorplay.graph.experimental.sympy_functions import (
    CleanDiv,
    FloorDiv,
    Identity,
    ModularIndexing,
)


def sympy_str(expr: sympy.Expr) -> str:
    """Write the expression out as it stands, without simplifying it.

    Subtraction is written as a subtraction rather than as an addition of a
    negative, since that is how it was written; the general form of a longer
    chain of subtractions still appears as additions of negatives, which is the
    price of not simplifying.
    """

    def is_neg_lead(expr: sympy.Expr) -> bool:
        return (
            isinstance(expr, sympy.Mul) and len(expr.args) == 2 and expr.args[0] == -1
        )

    def sympy_str_add(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Add):
            # Special case 'a - b'. Note that 'a - b - c' will still appear as
            # 'a + -1 * b + -1 * c'.
            if len(expr.args) == 2 and is_neg_lead(expr.args[1]):
                return f"{sympy_str_mul(expr.args[0])} - {sympy_str_mul(expr.args[1].args[1])}"
            else:
                return " + ".join(map(sympy_str_mul, expr.args))
        else:
            return sympy_str_mul(expr)

    def sympy_str_mul(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Mul):
            if is_neg_lead(expr):
                # Special case '-a'. Note that 'a * -b' will still appear as
                # '-1 * a * b'.
                return f"-{sympy_str_atom(expr.args[1])}"
            else:
                return " * ".join(map(sympy_str_atom, expr.args))
        else:
            return sympy_str_atom(expr)

    def sympy_str_atom(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Symbol):
            return expr.name
        elif isinstance(expr, (sympy.Add, sympy.Mul)):
            return f"({sympy_str_add(expr)})"
        elif isinstance(expr, (ModularIndexing, CleanDiv, FloorDiv, Identity)):
            return f"{expr.func.__name__}({', '.join(map(sympy_str, expr.args))})"
        else:
            return str(expr)

    return sympy_str_add(expr)
