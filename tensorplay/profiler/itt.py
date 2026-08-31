"""Optional low-level instrumentation range helpers."""

from __future__ import annotations

import contextlib

import tensorplay._C as _C


def _native(name):
    function = getattr(_C, name, None)
    if function is None:
        raise RuntimeError("ITT instrumentation is not available")
    return function


def is_available():
    """Return whether low-level instrumentation calls are available."""
    return getattr(_C, "_profiler_itt_available", None) is not None


def range_push(message):
    """Start a nested instrumentation range."""
    return _native("_profiler_itt_range_push")(message)


def range_pop():
    """Finish the current instrumentation range."""
    return _native("_profiler_itt_range_pop")()


def mark(message):
    """Emit an instantaneous instrumentation marker."""
    return _native("_profiler_itt_mark")(message)


@contextlib.contextmanager
def range(message, *args, **kwargs):
    """Start a formatted instrumentation range for a context."""
    range_push(message.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()


__all__ = ["is_available", "mark", "range", "range_pop", "range_push"]
