"""Profiler import aliases.

The modern surface lives in :mod:`tensorplay.profiler`; this module keeps
the legacy import names available.
"""

from ..profiler import (  # noqa: F401
    EventList,
    emit_itt,
    emit_nvtx,
    profile,
    record_function,
    schedule,
)

__all__ = ["profile", "record_function", "emit_nvtx", "emit_itt",
           "schedule", "EventList"]
