"""Looking at every value a compiled region produced, without stopping it.

A region that produces forty intermediate values and then gets one of them
wrong is hard to narrow down: the wrong one is not named by anything the region
says.  So generated code can be asked to call back after each value it
produces, and the callback is what turns forty values into one line of output
naming the one that differs.

That is worth having only when it is asked for, so the call is behind a flag
and the list of callbacks starts empty.  An empty list makes the call a
length test, which the generated branch already knows the answer to, so leaving
it empty costs the run nothing measurable.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

#: Called as ``hook(name, value)`` after each value is produced.  A hook that
#: raises is the hook's problem, not the region's: a region that computed a
#: value correctly must not be reported as failing because something watching
#: it did.
_hooks: list[Callable[[str, Any], None]] = []
_hook_lock = threading.Lock()


def register_intermediate_hook(hook: Callable[[str, Any], None]) -> None:
    """Call ``hook(name, value)`` after every value the region produces.

    Returns nothing: a hook stays registered until it is removed, because a hook
    that applied to one region and then stopped would make the output of the
    next one look like a region that produces nothing.
    """

    with _hook_lock:
        _hooks.append(hook)


def remove_intermediate_hook(hook: Callable[[str, Any], None]) -> None:
    """Stop calling a hook registered earlier.

    Removing one that is not registered is not an error: two pieces of code
    that each arrange for a hook to go away should not have to agree on which
    of them registered it.
    """

    with _hook_lock:
        if hook in _hooks:
            _hooks.remove(hook)


def clear_intermediate_hooks() -> None:
    """Forget every hook."""

    with _hook_lock:
        _hooks.clear()


def has_intermediate_hooks() -> bool:
    """Whether anything is watching.

    Generated code asks this before making the call, so that a region nobody is
    watching does not pay for a call into an empty list.
    """

    return bool(_hooks)


def run_intermediate_hooks(name: str, value: Any) -> None:
    """Tell every hook which value was just produced.

    A hook that raises is logged and skipped rather than allowed to fail the
    region: the value was computed, and reporting that it was computed is not
    the region's business to get right.
    """

    if not _hooks:
        return
    import logging

    log = logging.getLogger(__name__)
    for hook in list(_hooks):
        try:
            hook(name, value)
        except Exception:  # noqa: BLE001 - a hook must not fail the region
            log.exception("an intermediate hook failed for %r", name)


@contextmanager
def collect_intermediates(
    sink: Callable[[str, Any], None] | None = None,
) -> Iterator[list[tuple[str, Any]]]:
    """Collect every value produced inside the block, as ``(name, value)``.

    The pairs come back in the order the values were produced, so a caller
    comparing two runs compares them position by position.  A ``sink`` is
    called for each as well, for a caller that would rather not hold them all.
    """

    collected: list[tuple[str, Any]] = []

    def hook(name: str, value: Any) -> None:
        collected.append((name, value))
        if sink is not None:
            sink(name, value)

    register_intermediate_hook(hook)
    try:
        yield collected
    finally:
        remove_intermediate_hook(hook)


__all__ = [
    "clear_intermediate_hooks",
    "collect_intermediates",
    "has_intermediate_hooks",
    "register_intermediate_hook",
    "remove_intermediate_hook",
    "run_intermediate_hooks",
]
