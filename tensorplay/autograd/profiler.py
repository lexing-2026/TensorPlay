"""Compatibility alias module.

The profiler implementation lives in ``tensorplay.profiler``. This module
re-exports it under ``tensorplay.autograd`` so both import paths resolve to
the same classes.
"""

from __future__ import annotations

from tensorplay.profiler.profiler import (
    ProfilerActivity,
    _activity_name,
    _module_span_call,
    _start_module_tracing,
    _stop_module_tracing,
    emit_itt,
    emit_nvtx,
    profile,
    record_function,
    supported_activities,
    tensorboard_trace_handler,
)
from tensorplay.profiler._schedule import ProfilerAction

__all__ = [
    "ProfilerActivity",
    "ProfilerAction",
    "emit_itt",
    "emit_nvtx",
    "profile",
    "record_function",
    "supported_activities",
    "tensorboard_trace_handler",
]
