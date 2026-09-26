"""The values a capture was handed that are not tensors.

A captured region is normally called with tensors, and those are described by
their shape and stride and passed as such.  Some captures are handed something
else as well -- a memory pool to allocate out of, a stream, a handle owned by
the caller -- and a region that was given one has to be able to name it in the
code generated for it.

A generated function is handed its ordinary arguments positionally and has
nothing else to reach through, so the values that are not arguments are reached
by the position they were given in.  That is what this holds: the values a
capture was given, in order, so that a position recorded while capturing still
names the same value when the generated code runs.
"""

from __future__ import annotations

import threading
from typing import Any

#: The values the current capture was handed, in the order it was handed them.
#: Empty when nothing is being captured, so a generated function that asks for
#: one outside a capture is told there is none rather than handed a stale value
#: from whatever was captured before.
_current: list[Any] = []
_lock = threading.Lock()


def set_capture_inputs(values: list[Any]) -> None:
    """Record the values a capture was handed, in order.

    Replaces rather than appends: a second capture in the same process must not
    see the first one's values, and a generated function that outlives the
    capture it was generated for must not keep them.
    """

    global _current
    with _lock:
        _current = list(values)


def get_capture_inputs() -> list[Any]:
    """The values the current capture was handed, in order."""

    with _lock:
        return list(_current)


def get_external_object_by_index(index: int) -> Any:
    """The value the capture was handed at this position.

    Raises for a position that was not handed one, because a generated function
    that reached for a value that is not there would otherwise fail later, at the
    point the value is used, with nothing to say which position was wrong.
    """

    values = get_capture_inputs()
    if not 0 <= index < len(values):
        raise IndexError(
            f"the capture was handed {len(values)} value(s), so there is no "
            f"value at position {index}"
        )
    return values[index]


def clear_capture_inputs() -> None:
    """Forget the values a capture was handed."""

    set_capture_inputs([])


__all__ = [
    "clear_capture_inputs",
    "get_capture_inputs",
    "get_external_object_by_index",
    "set_capture_inputs",
]
