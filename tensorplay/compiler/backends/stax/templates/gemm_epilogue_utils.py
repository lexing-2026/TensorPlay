"""The symbolic-shape questions a gemm epilogue asks, in one place.

An epilogue is examined before anything is emitted, and the examination is a
sequence of questions about shapes: is this extent the same as that one, is it
worth a number yet, is this list the same length as that one.  A question the
answerer can answer yes to is a fact the epilogue may be planned around; a
question it cannot answer is not an error and is not made into one by guessing,
because guessing here would mean a plan that is only sometimes right.

So the questions are asked through a supplier rather than answered here.  That
also keeps the guarding out: an extent becomes a number only once something has
said it may, and the fact that it became one is remembered rather than
re-derived.
"""

from __future__ import annotations
import logging

#: Where this module's messages go.  A measurement that is
#: discarded is worth a line and a discarded one is worth none, so
#: the messages are here rather than printed.
log = logging.getLogger(__name__)

from typing import Any, Callable, Sequence

__all__ = [
    "guarded_int",
    "normalize_shape",
    "statically_known",
    "statically_known_equal",
    "statically_known_shape_equal",
]


def normalize_shape(shape: Any) -> Any:
    """Canonicalize a sequence-like shape to a tuple, and leave anything else.

    Three containers hold the same thing -- a list, a tuple, and the size type
    the framework's own tensors carry -- and a comparison that treats them as
    three different things is a comparison that is sometimes false.
    """

    if isinstance(shape, (list, tuple)) or type(shape).__name__ == "Size":
        return tuple(shape)
    return shape


def guarded_int(value: Any) -> int | None:
    """The number a value stands for, or ``None`` while it is still symbolic.

    A value backed by a symbol is only a number once something has said it may
    be taken as one, so a value nothing has vouched for is reported as not a
    number rather than guessed at.
    """

    return value if isinstance(value, int) and not isinstance(value, bool) else None


def statically_known(expr: Any, supplier: Callable[[Any], bool] | None = None) -> bool:
    """Whether a relation is known to hold, without asking for it to be made so.

    A question that cannot be answered is left alone rather than turned into a
    guard: guarding is a much stronger thing to do, and an epilogue planner that
    guards while it is only looking would recompile for shapes it never intended
    to specialise.
    """

    if isinstance(expr, bool):
        return expr
    if supplier is not None:
        return bool(supplier(expr))
    return False


def statically_known_equal(lhs: Any, rhs: Any,
                          supplier: Callable[[Any], bool] | None = None) -> bool:
    """Whether two extents are the same for every shape this will be handed."""

    return statically_known(lhs == rhs, supplier)


def statically_known_shape_equal(actual_shape: Sequence[Any],
                                expected_shape: Sequence[Any],
                                supplier: Callable[[Any], bool] | None = None) -> bool:
    """Whether two shapes agree, element for element, for every shape."""

    return len(actual_shape) == len(expected_shape) and all(
        statically_known_equal(actual, expected, supplier)
        for actual, expected in zip(actual_shape, expected_shape)
    )
