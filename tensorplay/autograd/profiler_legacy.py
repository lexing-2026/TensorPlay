"""Legacy profiler entry point backed by the native event recorder."""

from __future__ import annotations

import itertools
import warnings

import tensorplay
from .profiler_util import EventList, FunctionEvent
from .profiler import _start_module_tracing, _stop_module_tracing

__all__ = ["profile"]


class profile:
    """Collect CPU events using the compatibility profiler interface."""

    def __init__(
        self,
        enabled=True,
        *,
        use_cuda=False,
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    ):
        self.enabled = bool(enabled)
        self.use_cuda = bool(use_cuda)
        self.record_shapes = bool(record_shapes or with_flops)
        self.with_flops = bool(with_flops)
        self.profile_memory = bool(profile_memory)
        self.with_stack = bool(with_stack)
        self.with_modules = bool(with_modules)
        self.entered = False
        self.function_events = None
        self._modules_patched = False

        if self.use_cuda and not tensorplay.cuda.is_available():
            warnings.warn(
                "CUDA is not available; compatibility profiling will use CPU events",
                stacklevel=2,
            )
            self.use_cuda = False

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("profiler context manager is not reentrant")
        self.entered = True
        try:
            if self.with_modules:
                _start_module_tracing()
                self._modules_patched = True
            tensorplay._C._profiler_start(
                self.record_shapes,
                self.with_stack,
                self.use_cuda,
                False,
                self.profile_memory,
            )
        except Exception:
            if self._modules_patched:
                _stop_module_tracing()
                self._modules_patched = False
            self.entered = False
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return False
        try:
            if self.use_cuda:
                tensorplay.cuda.synchronize()
            raw_ops, raw_gpu, raw_mem = tensorplay._C._profiler_stop()
            del raw_gpu, raw_mem
            events = EventList(
                raw_ops,
                use_device="cuda" if self.use_cuda else None,
                profile_memory=self.profile_memory,
                with_flops=self.with_flops,
                with_stack=self.with_stack,
            )
            events._build_tree()
            self.function_events = events
        finally:
            if self._modules_patched:
                _stop_module_tracing()
                self._modules_patched = False
            self.entered = False
        return False

    def _check_finish(self):
        if self.function_events is None:
            raise RuntimeError("Profiler did not finish running")

    def __repr__(self):
        if self.function_events is None:
            return "<unfinished profiler_legacy.profile>"
        return repr(self.function_events)

    def __str__(self):
        if self.function_events is None:
            return "<unfinished profiler_legacy.profile>"
        return str(self.function_events)

    def table(self, sort_by=None, row_limit=100, **kwargs):
        self._check_finish()
        return self.function_events.table(sort_by=sort_by, row_limit=row_limit, **kwargs)

    def export_chrome_trace(self, path):
        self._check_finish()
        return self.function_events.export_chrome_trace(path)

    def export_stacks(self, path, metric="self_cpu_time_total"):
        self._check_finish()
        if not self.with_stack:
            raise AssertionError("export_stacks() requires with_stack=True")
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._check_finish()
        return self.function_events.key_averages(
            group_by_input_shape=group_by_input_shape,
            group_by_stack_n=group_by_stack_n,
        )

    def total_average(self):
        self._check_finish()
        return self.function_events.total_average()

    @property
    def self_cpu_time_total(self):
        self._check_finish()
        return self.function_events.self_cpu_time_total


def _parse_legacy_records(thread_records):
    """Flatten native per-thread records into event objects."""
    records = list(thread_records)
    if records and isinstance(records[0], (tuple, list)):
        first = records[0]
        if len(first) < 4 or not isinstance(first[0], str):
            records = list(itertools.chain.from_iterable(records))
    return [event for event in EventList(records) if isinstance(event, FunctionEvent)]
