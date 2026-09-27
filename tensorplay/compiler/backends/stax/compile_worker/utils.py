"""Which process this is, where it matters.

A module built from generated source is registered in the module table and kept
in this process's cache -- or not, depending on whether this process is the one
the program is running in.  A process spawned to compile something is a worker:
what it builds belongs to the program, and registering it here would make it
reachable by name from this process, which is not where it will be used.

So the distinction is recorded once, when a process turns out to be a worker,
and read wherever the answer changes what is done.
"""

from __future__ import annotations

#: Whether this process is the one the program is running in.  A process that
#: was started to compile something is not, and says so by clearing this.
_IN_TOPLEVEL_PROCESS = True


def in_toplevel_process() -> bool:
    return _IN_TOPLEVEL_PROCESS


def mark_as_worker_process() -> None:
    """Record that this process was started to compile something.

    What this changes is only what is registered and kept here: a module built
    by a worker is used by the program that started it, not by the worker, so
    keeping it in the worker's own table would be keeping it somewhere it will
    not be asked for again.
    """

    global _IN_TOPLEVEL_PROCESS
    _IN_TOPLEVEL_PROCESS = False
