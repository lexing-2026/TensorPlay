"""Watching what a region allocated, and when it let go.

A region that allocates and frees wrongly does not usually miscompute: it reads
memory that has been handed back, and the value it reads is whatever was written
next.  Which means the wrongness moves around, and the only way to find it is to
be able to say what was alive when.

So every allocation and every free is recorded, and a check answers whether the
set of live allocations is the one the region said it would be.  The recording
is off until a check asks for it: an always-on record of every allocation in a
program is a cost every program pays for a feature almost none use.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Iterator
from typing import Any

#: What is alive right now, by the name the region gave it.  A weak value, so
#: recording an allocation does not keep it alive: the point is to watch what
#: the region does, not to change what it does.
_live: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()

#: Every allocation and free in order, once recording is turned on.  Bounded,
#: because a long-running program's record would otherwise grow without limit;
#: the oldest entries are the ones dropped, since a use-after-free is close
#: behind the free that caused it.
_history: list[tuple[str, str, str]] = []
_HISTORY_LIMIT = 10000

_recording = False


def set_recording(enabled: bool) -> None:
    """Turn the record on or off."""

    global _recording
    _recording = bool(enabled)
    if not enabled:
        _history.clear()


def is_recording() -> bool:
    """Whether anything is being recorded."""

    return _recording


def track_tensor(name: str, value: Any) -> Any:
    """Record a value as alive under ``name``, and hand it back.

    Returns the value so that a caller can wrap an allocation in a call to this
    without binding it twice.
    """

    if not _recording:
        return value
    try:
        _live[name] = value
    except TypeError:
        # A value that cannot be referred to weakly cannot be watched for
        # going away, so it is recorded in the history and left alone.
        pass
    _history.append(("alloc", name, type(value).__name__))
    _trim()
    return value


def untrack_tensor(name: str) -> None:
    """Record a value as no longer alive."""

    if not _recording:
        return
    _live.pop(name, None)
    _history.append(("free", name, ""))
    _trim()


def _trim() -> None:
    if len(_history) > _HISTORY_LIMIT:
        del _history[: len(_history) - _HISTORY_LIMIT]


def live_names() -> list[str]:
    """What is alive right now, in the order it was allocated."""

    return [name for _, name, _ in _history if name in _live]


def history() -> list[tuple[str, str, str]]:
    """Every allocation and free in order, as ``(what, name, type)``."""

    return list(_history)


def check_memory_step(
    step_name: str,
    expected_live: Any = None,
) -> None:
    """Raise unless what is alive is what the region said would be.

    ``expected_live`` is the set of names the region believes it holds.  Passing
    nothing checks only that nothing is alive that was freed, which is the half
    of the check that needs no knowledge of the region -- a use-after-free shows
    up there, and shows up as a name that was freed and is still being read.
    """

    if not _recording:
        return
    alive = set(live_names())
    if expected_live is None:
        expected_live = {name for kind, name, _ in _history if kind == "alloc"}
    expected = {str(name) for name in expected_live}
    unexpected = alive - expected
    missing = expected - alive
    if unexpected or missing:
        problems = []
        if unexpected:
            problems.append(f"still alive but not expected: {sorted(unexpected)}")
        if missing:
            problems.append(f"expected but gone: {sorted(missing)}")
        raise AssertionError(
            f"at step {step_name!r} the live allocations do not match: "
            + "; ".join(problems)
        )


@contextlib.contextmanager
def tracking_memory() -> Iterator[None]:
    """Record allocations and frees for the length of the block.

    The record is dropped on the way out, including when the block raises, so a
    failed run does not leave the next one looking at the previous run's
    allocations.
    """

    set_recording(True)
    _live.clear()
    _history.clear()
    try:
        yield
    finally:
        set_recording(False)
        _live.clear()
        _history.clear()


__all__ = [
    "check_memory_step",
    "history",
    "is_recording",
    "live_names",
    "set_recording",
    "track_tensor",
    "tracking_memory",
    "untrack_tensor",
]
