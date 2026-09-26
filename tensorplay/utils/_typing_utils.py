"""Small assertions about types and values that a caller relies on.

These are for the places where a value has already been established and the
code below cannot say what to do if it is not there.  Each one raises rather
than returning a substitute, because a substitute here would be a value the
caller goes on to use as if it were the one that was established.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def not_none(obj: T | None) -> T:
    """The value, which is not nothing.

    For a value that has just been looked up and is about to be used without
    a further check: the check is written once here rather than at each use,
    and the message says which invariant broke rather than what was indexed.
    """
    if obj is None:
        raise TypeError("Invariant encountered: value was None when it should not be")
    return obj


__all__ = ["not_none"]
