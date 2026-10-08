"""Public profiler controller and instrumentation contexts."""

from __future__ import annotations

import os
import socket
import time
from contextlib import ContextDecorator
from enum import Enum

import tensorplay._C as _C

from ._chrome_trace_export import merge_distributed_traces
from ._memory_profiler import (
    export_memory_timeline,
    memory_summary,
)
from ._schedule import ProfilerAction, schedule
from ._utils import rank_world
from .profiler_util import EventList, device_activities
from .python_tracer import PySampler


DeviceType = _C.DeviceType


class ProfilerActivity(str, Enum):
    """Activity groups understood by the profiler."""

    CPU = "cpu"
    CUDA = "cuda"


def supported_activities():
    """Return activity groups available in this build."""
    activities = {ProfilerActivity.CPU}
    try:
        import tensorplay as tp

        if tp.cuda.is_available():
            activities.add(ProfilerActivity.CUDA)
    except Exception:
        pass
    return activities


def _activity_name(activity):
    if isinstance(activity, ProfilerActivity):
        return activity.value
    if isinstance(activity, str):
        return activity.lower()
    value = getattr(activity, "value", None)
    return str(value).lower() if value is not None else str(activity).lower()


def tensorboard_trace_handler(directory, worker_name=None, use_gzip=False):
    """Create a callback that writes one trace file per completed cycle."""
    directory = os.fspath(directory)

    def handler(prof):
        os.makedirs(directory, exist_ok=True)
        name = worker_name or f"{socket.gethostname()}_{os.getpid()}"
        filename = f"{name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            filename += ".gz"
        prof.export_chrome_trace(os.path.join(directory, filename), torch_compat=True)

    return handler


_module_original_call = None
_module_patch_depth = 0


def _start_module_tracing():
    """Patch Module.__call__ so every module forward brackets a span.

    The patch is reference-counted across nested sessions; only the
    outermost session installs it, and only the last exit restores.  While
    installed the span begin/end pair costs one atomic load when no session
    captures, and one extra Python frame per module call while capturing.
    """
    global _module_original_call, _module_patch_depth
    from tensorplay.nn.modules.module import Module

    if _module_patch_depth == 0:
        _module_original_call = Module.__call__
        Module.__call__ = _module_span_call
    _module_patch_depth += 1


def _stop_module_tracing():
    global _module_original_call, _module_patch_depth
    if _module_patch_depth == 0:
        return
    _module_patch_depth -= 1
    if _module_patch_depth == 0:
        from tensorplay.nn.modules.module import Module

        if Module.__call__ is _module_span_call:
            Module.__call__ = _module_original_call
        _module_original_call = None


def _module_span_call(module_self, *args, **kwargs):
    """Module.__call__ replacement active during with_modules sessions."""
    _C._profiler_user_begin("nn.Module: " + type(module_self).__name__)
    try:
        return _module_original_call(module_self, *args, **kwargs)
    finally:
        _C._profiler_user_end()


#: Whether any device activity record has ever been collected in this process.
#: The collector allocates its record buffers when it is first asked for them and
#: does not record what happens while it does, so the first session in a process
#: reports nothing; a caller that needs device records has to know whether the
#: session it just ran was that one.
_device_records_seen = False


def device_activity_ready() -> bool:
    """Whether a device-activity session started now would report records.

    False in a process that has not yet run one, and false only for the first
    session: the collector is ready from the second onwards.
    """

    return _device_records_seen


def _note_device_records(count: int) -> None:
    """Record that a session received this many device records."""

    global _device_records_seen
    if count:
        _device_records_seen = True


class profile:
    """Context manager that records dispatched operations and annotations."""

    def __init__(
        self,
        enabled=True,
        *,
        activities=None,
        schedule=None,
        on_trace_ready=None,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        use_device=None,
        gpu_timing=None,
        gpu_trace=None,
        with_samples=False,
    ):
        self.enabled = bool(enabled)
        self.activities = tuple(activities) if activities is not None else None
        self.record_shapes = bool(record_shapes)
        self.profile_memory = bool(profile_memory)
        self.with_stack = bool(with_stack)
        # FLOP estimation reads input shapes at aggregation time, so it
        # requires shape capture; enabling it implies record_shapes.
        self.with_flops = bool(with_flops)
        if self.with_flops and not self.record_shapes:
            self.record_shapes = True
        self.with_modules = bool(with_modules)
        self.use_device = use_device
        self.with_samples = bool(with_samples)
        self.schedule = schedule
        self.on_trace_ready = on_trace_ready

        activity_names = {_activity_name(activity) for activity in self.activities or ()}
        cuda_requested = (
            "cuda" in activity_names
            or (use_device is not None and _activity_name(use_device) == "cuda")
        )
        # GPU durations come from the CUPTI activity stream by default: it
        # collects kernel/memcpy records asynchronously on its own thread,
        # so the dispatch hot path pays no per-op driver calls.  An explicit
        # gpu_timing=True request keeps the legacy event-pair timing (it can
        # coexist with gpu_trace for exact per-op times); the session start
        # falls back to event pairs automatically when the CUPTI runtime
        # cannot load.
        self.gpu_timing = bool(gpu_timing)
        if gpu_trace is None:
            self.gpu_trace = cuda_requested and gpu_timing is None
        else:
            self.gpu_trace = bool(gpu_trace)

        self.step_num = 0
        self.events = None
        self.gpu_activities = []
        self.mem_events = []
        self._recording = False
        self._entered = False
        self._modules_patched = False
        self._t0 = 0
        self._sampler = None
        self.stop_ms = 0.0
        self.gpu_timed_events = 0
        self.gpu_resolved_events = 0
        self._device_results = None

    def _ensure_events(self):
        if self.events is None:
            self.events = EventList(
                (),
                base_ns=self._t0,
                gpu_activities=self.gpu_activities,
                mem_events=self.mem_events,
                with_stack=self.with_stack,
            )
            self.events.with_flops = self.with_flops
        return self.events

    def __enter__(self):
        if self._entered:
            raise RuntimeError("profiler context manager is not reentrant")
        self._entered = True
        self._t0 = time.perf_counter_ns()
        try:
            if self.enabled and self.with_modules:
                _start_module_tracing()
                self._modules_patched = True
            if self.enabled and self.with_samples:
                self._sampler = PySampler(lambda: self._recording)
                self._sampler.start()
            if self.enabled and self.schedule is None:
                self._start_session()
        except Exception:
            if self._sampler is not None:
                self._sampler.stop()
                self._sampler = None
            if self._modules_patched:
                _stop_module_tracing()
                self._modules_patched = False
            self._entered = False
            raise
        return self

    def _start_session(self):
        if not self.enabled or self._recording:
            return
        _C._profiler_start(
            self.record_shapes,
            self.with_stack,
            self.gpu_timing,
            self.gpu_trace,
            self.profile_memory,
        )
        self._recording = True
        # Custom operators emit their op spans only while a session is live;
        # the counter handles nested/scheduled sessions.
        from .. import library as _library
        _library._note_profiling_start()

    def _stop_session(self):
        if not self._recording:
            return
        from .. import library as _library
        _library._note_profiling_stop()
        stop_start = time.perf_counter_ns()
        try:
            raw_ops, raw_gpu, raw_mem = _C._profiler_stop()
            self.stop_ms += (time.perf_counter_ns() - stop_start) / 1e6
            self.gpu_timed_events += sum(
                1 for event in raw_ops if len(event) > 8 and event[8] is not None and event[8] >= 0
            )
            self.gpu_resolved_events = self.gpu_timed_events
            events = self._ensure_events()
            events.extend(raw_ops)
            if raw_gpu:
                self.gpu_activities.extend(raw_gpu)
            _note_device_records(len(raw_gpu))
            if raw_mem:
                self.mem_events.extend(raw_mem)
        finally:
            self._recording = False

    def _notify_trace_ready(self):
        if self.on_trace_ready is not None:
            self.on_trace_ready(self)

    def step(self):
        """Advance the configured step schedule."""
        if not self.enabled or self.schedule is None:
            return None
        if not self._entered:
            raise RuntimeError("profiler.step() called outside its context")
        action = self.schedule(self.step_num)
        self.step_num += 1
        if action in (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE):
            if not self._recording:
                self._start_session()
            if action == ProfilerAction.RECORD_AND_SAVE:
                self._stop_session()
                self._notify_trace_ready()
        else:
            was_recording = self._recording
            self._stop_session()
            if was_recording and action == ProfilerAction.NONE:
                self._notify_trace_ready()
        return action

    def __exit__(self, exc_type, exc, tb):
        try:
            self._stop_session()
        finally:
            if self._sampler is not None:
                self._sampler.stop()
            if self._modules_patched:
                _stop_module_tracing()
                self._modules_patched = False
            self._ensure_events()
            self._entered = False
        return False

    def __str__(self):
        """The recording as a table; printing a session prints what it saw."""

        return str(self._ensure_events())

    def kineto_results(self):
        """The device's own record of what it did, in a form meant to be read.

        Named for what it is rather than for how it was collected: a caller
        wants the device's work, wants to know which piece of it belongs to which
        operation, and wants the two joined by something exact.  What the
        collector gathered is exactly that, so it is handed over as records that
        answer those questions, and the join is already made rather than left to
        the caller to reconstruct.
        """

        if self._device_results is None:
            self._device_results = device_activities(self.gpu_activities)
        return self._device_results

    @property
    def device_time_by_span(self):
        """For each collected span, the device time of the work inside it.

        The per-span answer, for a caller that has a span and wants what it cost
        on the device rather than what it cost to ask for.
        """

        return self._ensure_events().device_time_by_span

    def key_averages(
        self,
        group_by_input_shape=False,
        group_by_stack_n=0,
        group_by_overload_name=False,
        include_python_functions=False,
        sort_by=None,
    ):
        events = self._ensure_events()
        if group_by_input_shape and not any(len(event) > 5 and event[5] is not None for event in events):
            raise ValueError("No shapes recorded: pass record_shapes=True to profile()")
        table = events.key_averages(
            group_by_input_shape,
            group_by_stack_n,
            group_by_overload_name,
            include_python_functions,
        )
        if sort_by is not None:
            table.sort(sort_by)
        return table

    def device_kernels(self, sort_by=None):
        """What the device ran, by the name the device knows it by."""

        return self._ensure_events().device_kernels(sort_by=sort_by)

    def table(
        self,
        sort_by=None,
        row_limit=100,
        header=None,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        top_level_events_only=False,
        time_unit=None,
    ):
        """Render the session's spans, one row per span.

        Every span gets a row rather than one row per name, so the per-span
        captures the session holds -- the shapes an operation was given, the
        frames it was called from, the memory it allocated -- can be printed
        against the span they describe.  :meth:`key_averages` is the aggregated
        view of the same recording.
        """

        return self._ensure_events().table(
            sort_by=sort_by,
            row_limit=row_limit,
            header=header,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            top_level_events_only=top_level_events_only,
            time_unit=time_unit,
        )

    def total_average(self):
        return self.key_averages().total_average()

    @property
    def self_cpu_time_total(self):
        return self.key_averages().self_cpu_time_total

    @property
    def current_action(self):
        if self.schedule is None:
            return None
        return self.schedule(self.step_num)

    def export_chrome_trace(self, path, torch_compat=False):
        events = self._ensure_events()
        if self._sampler is not None:
            events.samples = self._sampler.samples
        return events.export_chrome_trace(
            path,
            torch_compat=torch_compat,
            gpu_activities=self.gpu_activities,
            mem_events=self.mem_events,
            samples=events.samples,
        )

    def export_tensorboard_trace(self, directory, run_name=None):
        path = os.fspath(directory)
        os.makedirs(path, exist_ok=True)
        name = run_name or "tensorplay"
        rank, _world = rank_world()
        if rank is not None:
            name = f"{name}-rank{rank}"
        output = os.path.join(
            path,
            f"{name}.{socket.gethostname()}.{os.getpid()}.{time.time_ns()}.pt.trace.json",
        )
        self.export_chrome_trace(output, torch_compat=True)
        return output

    def export_memory_timeline(self, path):
        return export_memory_timeline(self.mem_events, path)

    def memory_summary(self):
        return memory_summary(self.mem_events, self._ensure_events())

    def export_stacks(self, path, metric="self_cpu_time_total"):
        if not self.with_stack:
            raise AssertionError("export_stacks() requires with_stack=True")
        return self._ensure_events().export_stacks(path, metric)


class record_function(ContextDecorator):
    """Mark a user-defined span in the active profiling session."""

    def __init__(self, name, args=None):
        self.name = str(name)
        self.args = args
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("record_function context manager is not reentrant")
        _C._profiler_user_begin(self.name)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._entered:
            _C._profiler_user_end()
            self._entered = False
        return False


class _EmitContext(ContextDecorator):
    _setter = None

    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = bool(enabled)
        self.record_shapes = bool(record_shapes)
        self._entered = False

    def __enter__(self):
        if not self.enabled:
            return self
        if self._entered:
            raise RuntimeError("annotation context manager is not reentrant")
        self._setter(True)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._entered:
            self._setter(False)
            self._entered = False
        return False


class emit_nvtx(_EmitContext):
    """Emit named ranges for active operations."""

    _setter = staticmethod(_C._profiler_emit_nvtx)


class emit_itt(_EmitContext):
    """Emit instrumentation ranges for active operations."""

    _setter = staticmethod(_C._profiler_emit_itt)


__all__ = [
    "DeviceType",
    "ProfilerActivity",
    "ProfilerAction",
    "emit_itt",
    "emit_nvtx",
    "merge_distributed_traces",
    "profile",
    "record_function",
    "schedule",
    "supported_activities",
    "tensorboard_trace_handler",
]
