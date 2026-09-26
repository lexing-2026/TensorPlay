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

#: Names some step has already accounted for, so a later step is not asked
#: about them again.
_claimed: set[str] = set()


def set_recording(enabled: bool) -> None:
    """Turn the record on or off."""

    global _recording
    _recording = bool(enabled)
    _claimed.clear()
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
    step_name: Any = None,
    expected_live: Any = None,
    *,
    allocated: Any = None,
    freed: Any = None,
    is_final_step: bool = False,
) -> None:
    """Raise unless what is alive is what was said it would be.

    Asked which names a whole region should be holding, by giving
    ``expected_live``: what is alive then is compared against that set as a
    whole.  Passing nothing for it checks only that nothing is alive that was
    freed, which is the half of the check that needs no knowledge of the region
    -- a use-after-free shows up there, and shows up as a name that was freed and
    is still being read.

    Asked what changed during one step, by giving ``allocated`` and ``freed``:
    the names that came alive and the names that went away are compared against
    what changed, which is what a program laid out in steps can say and is a
    claim about the step rather than about the region.  This is the form the
    generated code asks in, because the step it is asking about is written into
    that code next to the call.

    The last step is the one place the two claims cannot both be made: a name
    that is handed back as a result is not freed, and nothing here can tell that
    apart from a name that was simply forgotten.  So a name said to be freed at
    the last step is not required to have gone -- the check would report a leak
    as a fault and would be wrong more often than not.
    """

    if not _recording:
        return
    problems: list[str] = []
    if allocated is not None or freed is not None:
        expected_alloc = {str(name) for name in (allocated or ())}
        expected_free = {str(name) for name in (freed or ())}
        # What should be alive once this step has run: what earlier steps left
        # behind, plus what this step brings up, less what it lets go of.  The
        # running expectation is carried rather than recounted, so a name is
        # accounted for at the step that dealt with it and not asked about
        # again at every step after.
        expected_after = (_claimed | expected_alloc) - expected_free
        actually_live = set(live_names())
        if is_final_step:
            # A name handed back as a result has not been freed, and nothing
            # here can tell that from one that was forgotten.  Insisting would
            # report a leak as a fault, and would be wrong more often than not.
            expected_after |= expected_free & actually_live
        extra = actually_live - expected_after
        missing = expected_after - actually_live
        if extra:
            problems.append(f"alive but not accounted for: {sorted(extra)}")
        if missing:
            problems.append(f"accounted for but not alive: {sorted(missing)}")
        _claimed.clear()
        _claimed.update(expected_after)
    else:
        alive = set(live_names())
        if expected_live is None:
            expected_live = {name for kind, name, _ in _history if kind == "alloc"}
        expected = {str(name) for name in expected_live}
        unexpected = alive - expected
        missing = expected - alive
        if unexpected:
            problems.append(f"still alive but not expected: {sorted(unexpected)}")
        if missing:
            problems.append(f"expected but gone: {sorted(missing)}")
    if problems:
        where = f" at step {step_name!r}" if step_name is not None else ""
        raise AssertionError(
            f"the live allocations do not match{where}: " + "; ".join(problems)
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
    _claimed.clear()
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
