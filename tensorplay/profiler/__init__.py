"""Runtime profiler for operator, autograd, memory, and GPU events.

Every dispatched operator is recorded once at the below-autograd redispatch
funnel. Composite inner calls remain visible, and the autograd engine emits a
``__backward__`` span around every backward()/grad() execution.

Typical use::

    with tp.profiler.profile() as prof:
        y = model(x)
        loss = criterion(y)
    print(prof.key_averages())
    prof.export_chrome_trace("trace.json")

The Chrome trace includes GPU lanes and accelerator-to-GPU flow arrows and
loads directly in ``chrome://tracing`` or Perfetto.


* ``gpu_trace=True`` -- CUPTI kernel-level GPU tracing (USE_CUDA builds):
  kernel/memcpy/memset rows on GPU lanes, CUDA runtime/driver API rows, and
  op->runtime->kernel correlation via CUPTI external correlation ids.
* ``profile_memory=True`` -- allocator-level alloc/free events (CPU and
  CUDA caching allocators) backing :meth:`profile.memory_summary` and
  :meth:`profile.export_memory_timeline`.
* ``with_samples=True`` -- a Python stack sampler (background thread) whose
  samples appear as ``python_function`` instant events in traces and feed
  :meth:`profile.export_stacks`.

Multi-process runs tag their traces with the ``RANK``/``WORLD_SIZE``
environment values; :func:`merge_distributed_traces` folds per-rank exports
into one file.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import sys
import threading
import time

import tensorplay._C as _C

__all__ = ["profile", "record_function", "EventList", "schedule",
           "emit_nvtx", "emit_itt", "merge_distributed_traces"]


def _rank_world():
    rank = os.environ.get("RANK", os.environ.get("TP_RANK"))
    world = os.environ.get("WORLD_SIZE", os.environ.get("TP_WORLD_SIZE"))
    try:
        return (int(rank) if rank is not None else None,
                int(world) if world is not None else None)
    except ValueError:
        return (None, None)


class _FunctionsTable:
    """Build an aggregated event table including self-CPU time.

    Each
    event's duration minus the time spent inside its same-thread children is
    computed with a per-thread sweep over the (properly nested) spans.  When
    a session captured GPU data (gpu_timing or gpu_trace), CUDA columns are
    appended (``Self CUDA`` / ``CUDA Total``) and the table sorts by them.
    """

    def __init__(self, events, group_by_input_shape=False):
        self_ns = _self_times(events)
        self_cuda = _self_cuda_us(events)
        agg = collections.OrderedDict()
        has_gpu = False
        for ev, self_dur, sc_us in zip(events, self_ns, self_cuda):
            name, kind, start_ns, end_ns = ev[0], ev[1], ev[2], ev[3]
            shapes = ev[5]
            kernel_count = ev[11] if len(ev) > 11 else 0
            gpu_ms = ev[8] if ev[8] is not None else -1.0
            if gpu_ms >= 0:
                has_gpu = True
            if end_ns <= start_ns:
                continue
            key = (name, kind)
            if group_by_input_shape and shapes is not None:
                key = key + (tuple(tuple(s) for s in shapes),)
            row = agg.get(key)
            if row is None:
                row = [0, 0, None, None, 0, 0.0, 0.0, 0]
                agg[key] = row
            dur = end_ns - start_ns
            row[0] += 1
            row[1] += dur
            row[2] = dur if row[2] is None else min(row[2], dur)
            row[3] = dur if row[3] is None else max(row[3], dur)
            row[4] += self_dur
            if gpu_ms >= 0:
                row[5] += gpu_ms * 1000.0  # total CUDA us
                row[6] += sc_us            # self CUDA us
                row[7] += kernel_count
        grand_total = sum(r[1] for r in agg.values()) or 1
        self.has_gpu = has_gpu
        self.rows = []
        for key, (cnt, total, mn, mx, stotal, cuda_us, self_cuda_us, nk) \
                in sorted(agg.items(), key=lambda kv: -kv[1][4]):
            name, kind = key[0], key[1]
            shapes_sig = key[2] if len(key) > 2 else None
            self.rows.append(_Row(name, kind, cnt, total // cnt, mn, mx,
                                  shapes_sig, stotal,
                                  stotal / grand_total * 100.0,
                                  cuda_us, nk, self_cuda_us))
        self.total_ns = grand_total
        self.total_cuda_us = sum(r.cuda_us for r in self.rows)

    def __str__(self):
        header = (f"{'Name':<28}{'Calls':>7}{'Self us':>10}"
                  f"{'Self %':>8}{'Total us':>10}")
        if self.has_gpu:
            header += f"{'Self CUDA us':>13}{'CUDA Total us':>15}{'#K':>6}"
        lines = [header, "-" * len(header)]
        for r in self.rows:
            line = (f"{r.name:<28}{r.count:>7}{r.self_us:>10.2f}"
                    f"{r.self_pct:>7.1f}%{r.total_us:>10.2f}")
            if self.has_gpu:
                line += (f"{r.self_cuda_us:>13.2f}{r.cuda_us:>15.2f}"
                         f"{r.kernel_count:>6}")
            lines.append(line)
        lines.append("-" * len(header))
        lines.append(f"Self CPU time total: {self.total_ns/1e3:.2f} us")
        if self.has_gpu:
            lines.append(
                f"Self CUDA time total: {self.total_cuda_us:.2f} us")
        return "\n".join(lines)

    def __repr__(self):
        return str(self)


def _self_times(events):
    """Per-event self duration (ns): own span minus same-thread children.

    Spans from one thread nest properly (RAII/LIFO), so a sweep with a stack
    attributes every child's duration to its immediate parent exactly once.
    """
    by_tid = collections.defaultdict(list)
    for idx, ev in enumerate(events):
        s, e, tid = ev[2], ev[3], ev[4]
        by_tid[tid].append((s, e, idx))
    self_ns = [0] * len(events)
    for lst in by_tid.values():
        lst.sort()
        stack = []  # (start, end, idx)
        for s, e, idx in lst:
            while stack and stack[-1][1] <= s:
                stack.pop()
            dur = e - s
            self_ns[idx] += dur
            if stack:
                # charge the full duration to whoever is on top; their own
                # self-time was pre-credited above and is reduced here
                self_ns[stack[-1][2]] -= dur
            stack.append((s, e, idx))
    return self_ns


def _self_cuda_us(events):
    """Per-event self CUDA time (us): op's correlated kernel time minus
    same-thread children's.  In gpu_trace mode kernels attach only to the
    innermost dispatch, so double counting is already avoided; the sweep
    keeps gpu_timing-mode pairs honest when ops nest."""
    by_tid = collections.defaultdict(list)
    for idx, ev in enumerate(events):
        gpu_ms = ev[8] if ev[8] is not None else -1.0
        cuda_us = gpu_ms * 1000.0 if gpu_ms >= 0 else 0.0
        by_tid[ev[4]].append((ev[2], ev[3], idx, cuda_us))
    self_us = [0.0] * len(events)
    for lst in by_tid.values():
        lst.sort()
        stack = []
        for s, _e, idx, cuda_us in lst:
            while stack and stack[-1][1] <= s:
                stack.pop()
            self_us[idx] += cuda_us
            if stack and cuda_us:
                self_us[stack[-1][2]] -= cuda_us
            stack.append((s, _e, idx))
    return [max(v, 0.0) for v in self_us]


# Factory ops whose output allocation volume is derivable from
# shapes x dtype (the fallback basis of the memory snapshot view when a
# session ran without allocator-level capture).
_FACTORY_PREFIXES = (
    "empty", "zeros", "ones", "full", "rand", "randn", "randint",
    "randperm", "arange", "linspace", "logspace", "eye", "tensor",
    "scalar_tensor",
)

_DTYPE_ITEMSIZE = {
    "float32": 4, "float64": 8, "float16": 2, "bfloat16": 2,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "bool": 1, "complex64": 8, "complex128": 16,
}


def _dtype_name(v):
    # dtypes ride as raw DType ints from the binding; map via tensorplay.
    try:
        import tensorplay as tp
        return str(tp.DType(v)).rsplit(".", 1)[-1]
    except Exception:
        return None


class _Row:
    def __init__(self, name, kind, count, avg_ns, min_ns, max_ns, shapes,
                 self_ns=0, self_pct=0.0, cuda_us=0.0, kernel_count=0,
                 self_cuda_us=0.0):
        self.name = name
        self.kind = kind
        self.count = count
        self.input_shapes = shapes
        self.avg_ns = avg_ns
        self.min_ns = min_ns
        self.max_ns = max_ns
        self.self_ns = self_ns
        self.self_pct = self_pct
        self.cuda_us = cuda_us
        self.kernel_count = kernel_count
        self.self_cuda_us = self_cuda_us

    @property
    def avg_us(self):
        return self.avg_ns / 1e3

    @property
    def min_us(self):
        return self.min_ns / 1e3

    @property
    def max_us(self):
        return self.max_ns / 1e3

    @property
    def self_us(self):
        return self.self_ns / 1e3

    @property
    def total_us(self):
        return self.avg_ns * self.count / 1e3

    @property
    def self_cuda_time_total(self):
        return self.cuda_us

    @property
    def cuda_time_total(self):
        return self.cuda_us


class _Action:
    NONE = "none"
    WARMUP = "warmup"
    RECORD = "record"
    WARMUP_AND_RECORD = "warmup_and_record"


def schedule(*, wait, warmup, active, repeat=0, skip_first=0):
    """

    The returned callable maps a monotonically increasing ``step`` number to
    one of the actions NONE / WARMUP / RECORD; ``profile.step()`` consults it
    to start and stop capture cycles: wait steps do nothing, warmup steps run
    uncaptured (allocator/cache warm-up), active steps record.
    """
    def schedule_fn(step):
        if step < skip_first:
            return _Action.NONE
        cycle = warmup + active
        pos = (step - skip_first)
        if repeat > 0 and pos >= repeat * (wait + cycle):
            return _Action.NONE
        pos %= wait + cycle if (wait + cycle) > 0 else 1
        if pos < wait:
            return _Action.NONE
        pos -= wait
        if pos < warmup:
            return _Action.WARMUP
        return _Action.RECORD
    return schedule_fn


# ---- Python stack sampler --------------------------------------------------
def _sample_interval_ms():
    raw = os.environ.get("TP_PROFILER_SAMPLE_MS", "")
    try:
        v = float(raw) if raw else 0.0
    except ValueError:
        v = 0.0
    return v if v > 0 else 5.0


class _PySampler:
    """Background Python stack sampler.

    Snapshots every thread's frame chain via ``sys._current_frames()`` at a
    configurable interval (default 5 ms, ``TP_PROFILER_SAMPLE_MS``).  A GIL
    must be taken per tick, so keep the interval modest on latency-critical
    workloads; the samples land as ``python_function`` instant events in the
    chrome trace and feed ``export_stacks``.
    """

    def __init__(self, should_run):
        self.should_run = should_run
        self.samples = []  # (ts_ns, os_tid, [frame strings, leaf first])
        self._stop = threading.Event()
        self._thread = None

    def _tick(self):
        frames_by_tid = sys._current_frames()
        ts = time.perf_counter_ns()
        for os_tid, frame in frames_by_tid.items():
            chain = []
            f = frame
            depth = 0
            while f is not None and depth < 64:
                code = f.f_code
                chain.append(f"{code.co_filename}:{f.f_lineno}"
                             f" ({code.co_name})")
                f = f.f_back
                depth += 1
            if chain:
                self.samples.append((ts, os_tid, chain))

    def _run(self):
        interval = _sample_interval_ms() / 1e3
        while not self._stop.wait(interval):
            try:
                if not self.should_run():
                    continue
                self._tick()
            except Exception:
                continue  # a sampling tick must never kill the thread

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="tp-profiler-sampler")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class profile:
    """Context manager recording ops executed inside the block.

    shapes+dtypes; ``with_stack`` captures the full Python frame chain of
    each op; ``schedule`` enables step-driven capture cycles via ``step()``;
    ``profile_memory`` records allocator-level alloc/free events;
    ``gpu_timing`` arms pool-backed cudaEvent pairs around dispatched CUDA
    work (validated on CUDA 12.x, sm_89); ``gpu_trace`` enables CUPTI
    kernel-level tracing with op->runtime->kernel correlation (USE_CUDA
    builds); ``with_samples`` runs the Python stack sampler.  Extra keyword
    working.
    """

    def __init__(self, *args, record_shapes=False, with_stack=False,
                 schedule=None, gpu_timing=False, gpu_trace=False,
                 profile_memory=False, with_samples=False, **kwargs):
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.gpu_timing = gpu_timing
        self.gpu_trace = gpu_trace
        self.profile_memory = profile_memory
        self.with_samples = with_samples
        self.schedule = schedule
        self.step_num = 0
        self.events = None
        self.gpu_activities = []
        self.mem_events = []
        self._recording = False
        self._t0 = 0
        self._sampler = None
        # Stop-time accounting is intentionally kept outside the event tuple:
        # it describes the profiler itself, not a user operation.  In CUDA
        # mode this is the complete event-query/reclaim path, so callers can
        # detect a regression without inferring it from wall-clock noise in
        # the profiled workload.
        self.stop_ms = 0.0
        self.gpu_timed_events = 0
        self.gpu_resolved_events = 0

    def __enter__(self):
        import time
        self._t0 = time.perf_counter_ns()
        if self.with_samples:
            self._sampler = _PySampler(lambda: self._recording)
            self._sampler.start()
        if self.schedule is None:
            self._start_session()
        return self

    def _start_session(self):
        _C._profiler_start(self.record_shapes, self.with_stack,
                           self.gpu_timing, self.gpu_trace,
                           self.profile_memory)
        self._recording = True

    def _stop_session(self):
        import time
        stop_t0 = time.perf_counter_ns()
        raw_ops, raw_gpu, raw_mem = _C._profiler_stop()
        self.stop_ms += (time.perf_counter_ns() - stop_t0) / 1e6
        self.gpu_timed_events += sum(1 for ev in raw_ops if ev[8] >= 0)
        # Keep the name explicit: the native binding returns all events, and
        # gpu_ms >= 0 is the success bit for a resolved CUDA event pair.
        self.gpu_resolved_events = self.gpu_timed_events
        self._recording = False
        if self.events is None:
            self.events = EventList([], base_ns=self._t0)
        self.events.extend(raw_ops)
        if raw_gpu:
            self.gpu_activities.extend(raw_gpu)
        if raw_mem:
            self.mem_events.extend(raw_mem)

    def step(self):
        """Advance the schedule; called once per training iteration."""
        if self.schedule is None:
            return None
        action = self.schedule(self.step_num)
        self.step_num += 1
        if action in (_Action.RECORD, _Action.WARMUP_AND_RECORD):
            if not self._recording:
                self._start_session()
        else:  # NONE / WARMUP
            if self._recording:
                self._stop_session()
        return action

    def __exit__(self, exc_type, exc, tb):
        if self._recording:
            self._stop_session()
        if self._sampler is not None:
            self._sampler.stop()
        if self.events is None:
            self.events = EventList([], base_ns=self._t0)
        return False

    def key_averages(self, group_by_input_shape=False, sort_by=None):
        if group_by_input_shape and not any(
                ev[5] is not None for ev in self.events):
            raise ValueError(
                "No shapes recorded: pass record_shapes=True to profile()")
        table = _FunctionsTable(self.events,
                                group_by_input_shape=group_by_input_shape)
        if sort_by:
            keys = {
                "self_cpu_time": lambda r: r.self_ns,
                "cpu_time": lambda r: r.total_us,
                "calls": lambda r: r.count,
                "name": lambda r: r.name,
                "self_cuda_time": lambda r: r.cuda_us,
                "self_cuda_time_total": lambda r: r.cuda_us,
                "cuda_time": lambda r: r.cuda_us,
                "cuda_time_total": lambda r: r.cuda_us,
            }
            if sort_by not in keys:
                raise ValueError(f"unsupported sort_by: {sort_by}")
            table.rows.sort(key=keys[sort_by], reverse=(sort_by != "name"))
        return table

    @property
    def current_action(self):
        if self.schedule is None:
            return None
        return self.schedule(self.step_num)

    def export_chrome_trace(self, path, torch_compat=False):
        if self.events is None:
            self.events = EventList([], base_ns=self._t0)
        self.events.export_chrome_trace(
            path, torch_compat=torch_compat,
            gpu_activities=self.gpu_activities,
            mem_events=self.mem_events,
            samples=self._sampler.samples if self._sampler else [])

    def export_tensorboard_trace(self, directory, run_name=None):
        """

        json``) and schema (``schemaVersion``/``deviceProperties``/
        ``baseTimeNanoseconds``/``distributedInfo`` plus process/thread
        metadata events), so TensorBoard's profiler plugin can open the
        file.  Returns the written path.
        """
        import socket
        os.makedirs(directory, exist_ok=True)
        name = run_name or "tensorplay"
        rank, _world = _rank_world()
        if rank is not None:
            name = f"{name}-rank{rank}"
        path = os.path.join(
            directory,
            f"{name}.{socket.gethostname()}.{os.getpid()}."
            f"{int(time.time() * 1000)}.pt.trace.json")
        self.export_chrome_trace(path, torch_compat=True)
        return path

    def export_memory_timeline(self, path):
        """CSV timeline of allocator-level live bytes.

        Requires ``profile_memory=True``.  One row per allocation or free:
        ``timestamp_ns,device,allocated_bytes`` where allocated_bytes is the
        running live total for that device at the event time.
        """
        live = collections.defaultdict(int)
        rows = []
        for ts, _ptr, nbytes, alloc, is_cuda, device, _stream, _tid \
                in sorted(self.mem_events, key=lambda e: e[0]):
            delta = nbytes if alloc else -nbytes
            live[(is_cuda, device)] = max(0, live[(is_cuda, device)] + delta)
            dev = f"cuda:{device}" if is_cuda else "cpu"
            total = sum(live.values())
            rows.append((ts, dev, total))
        with open(path, "w") as fh:
            fh.write("timestamp_ns,device,allocated_bytes\n")
            for ts, dev, total in rows:
                fh.write(f"{ts},{dev},{total}\n")
        return rows

    def memory_summary(self):
        """Allocation view: (total_allocated, peak_live, timeline).

        With ``profile_memory=True`` this is exact allocator-level
        accounting over all devices.  Otherwise it falls back to the
        shapes-derived factory-op estimate (requires ``record_shapes=True``;
        in-place resizes and view aliases are not visible there).
        """
        if self.events is None:
            self.events = EventList([], base_ns=self._t0)
        if self.mem_events:
            timeline = []
            total = 0
            peak = live = 0
            for ts, _ptr, nbytes, alloc, *_rest in sorted(
                    self.mem_events, key=lambda e: e[0]):
                delta = nbytes if alloc else -nbytes
                total += nbytes if alloc else 0
                live += delta
                live = max(live, 0)
                peak = max(peak, live)
                timeline.append((ts, nbytes, "alloc" if alloc else "free"))
            return total, peak, timeline
        return self._factory_memory_summary()

    def _factory_memory_summary(self):
        timeline = []
        for ev in self.events:
            name, start_ns, nbytes = ev[0], ev[2], ev[9]
            base = name.split(".")[0]
            if base not in _FACTORY_PREFIXES and not base.endswith("_like"):
                continue
            if nbytes is None or nbytes <= 0:
                continue
            timeline.append((start_ns, nbytes, name))
        timeline.sort()
        total = sum(b for _, b, _ in timeline)
        peak = live = 0
        for _ts, b, _n in timeline:
            live += b
            peak = max(peak, live)
        return total, peak, timeline

    def export_stacks(self, path):
        """

        Aggregates per-op self CPU time by the op's captured Python stack
        (``with_stack=True``) in flamegraph.pl format:
        ``frame1;frame2;... <self_us>``.
        """
        if self.events is None:
            self.events = EventList([], base_ns=self._t0)
        self_ns = _self_times(self.events)
        folded = collections.Counter()
        for ev, self_dur in zip(self.events, self_ns):
            stack = ev[10] if len(ev) > 10 else None
            if not stack or self_dur <= 0:
                continue
            folded[";".join(reversed(stack))] += self_dur / 1e3
        with open(path, "w") as fh:
            for stack, us in folded.most_common():
                fh.write(f"{stack} {us:.0f}\n")
        return folded


class _FunctionEvent:
    """Read-only view over one collected tuple (FunctionEvent subset)."""

    def __init__(self, ev):
        (self.name, self.kind, self.start_ns, self.end_ns, self.tid,
         self.shapes, self.dtypes, self.site, self.gpu_ms,
         self.out_bytes) = ev[:10]
        self.stack = ev[10] if len(ev) > 10 else None
        self.kernel_count = ev[11] if len(ev) > 11 else 0
        self.cpu_interval = (start_ns, end_ns) if False else \
            type("Interval", (), {"start": start_ns / 1e3,
                                  "end": end_ns / 1e3})()

    @property
    def cpu_time(self):
        return (self.end_ns - self.start_ns) / 1e3

    @property
    def input_shapes(self):
        return self.shapes

    def __repr__(self):
        return f"<FunctionEvent {self.name} {(self.end_ns-self.start_ns)/1e3:.2f}us>"


class EventList(list):
    """Collected op ``(name, kind, start_ns, end_ns, tid, shapes, dtypes,
    site, gpu_ms, out_bytes, stack|None, kernel_count)`` tuples plus the
    session's GPU activities, memory events and Python samples."""

    def __init__(self, raw, base_ns=0, gpu_activities=None, mem_events=None,
                 samples=None):
        super().__init__(raw)
        self.base_ns = base_ns
        self.gpu_activities = gpu_activities if gpu_activities is not None else []
        self.mem_events = mem_events if mem_events is not None else []
        self.samples = samples if samples is not None else []

    # ---- chrome trace ------------------------------------------------------
    def export_chrome_trace(self, path, torch_compat=False,
                            gpu_activities=None, mem_events=None,
                            samples=None):
        """

        Layout: CPU rows on ``pid=0`` lanes; kernel/memcpy/memset rows on
        ``pid=1000000+device`` stream lanes; CUDA runtime/driver API rows on
        the OS-thread lanes of ``pid=0``; ``ac2g`` flow arrows linking each
        op to the GPU activity it launched; ``[memory]`` rows for
        allocator-level alloc/free events; ``python_function`` instant rows
        for stack samples; process/thread metadata (``ph:"M"``) events.
        """
        gpu_activities = (self.gpu_activities if gpu_activities is None
                          else gpu_activities)
        mem_events = self.mem_events if mem_events is None else mem_events
        samples = self.samples if samples is None else samples

        pid_cpu = 0
        pid_gpu_base = 1000000
        tids = {}  # raw id (hashed thread id / OS tid / stream id) -> lane
        lane_meta = {}  # (pid, lane id) -> human name (thread_name metadata)

        def lane(raw_tid, pid=0, name=None):
            key = tids.setdefault(raw_tid, len(tids))
            if name is not None:
                lane_meta.setdefault((pid, key), name)
            return key

        cat_of = {"o": "cpu_op", "u": "user_annotation", "b": "backward"}
        if torch_compat:
            # user annotation there.
            cat_of["b"] = "user_annotation"
        gpu_cat = {"k": "kernel", "m": "gpu_memcpy", "s": "gpu_memset",
                   "r": "cuda_runtime", "d": "cuda_driver"}

        trace_events = []
        flow_arrows = []  # (id, start_row, end_row)
        op_rows = []      # (idx_in_trace_events, start_ns, tid_lane)

        rank, world = _rank_world()

        def rel(ns):
            return (ns - self.base_ns) / 1e3

        for ev in self:
            name, kind, start_ns, end_ns, tid = ev[0], ev[1], ev[2], ev[3], ev[4]
            shapes, dtypes, site, gpu_ms = ev[5], ev[6], ev[7], ev[8]
            out_bytes = ev[9] if len(ev) > 9 else None
            kernel_count = ev[11] if len(ev) > 11 else 0
            tid_key = lane(tid)
            args = {}
            if shapes is not None:
                args["Input Dims"] = [list(s) for s in shapes]
                if dtypes is not None:
                    args["Input type"] = [_dtype_name(d) for d in dtypes]
            if site:
                args["Call site"] = site
            if gpu_ms is not None and gpu_ms >= 0:
                args["GPU duration (ms)"] = round(gpu_ms, 4)
                if kernel_count:
                    args["Kernels"] = kernel_count
            if out_bytes is not None and out_bytes > 0:
                args["Output alloc (bytes)"] = out_bytes
            if rank is not None:
                args["rank"] = rank
            row = {
                "name": name,
                "cat": cat_of.get(kind, "cpu_op"),
                "ph": "X",
                "ts": rel(start_ns),  # us, relative
                "dur": max((end_ns - start_ns) / 1e3, 0.001),
                "pid": pid_cpu,
                "tid": tid_key,
                "args": args,
            }
            op_rows.append((len(trace_events), start_ns, tid_key))
            trace_events.append(row)

        op_by_slot = {}  # external slot -> (ts, tid lane)
        for slot, (row_idx, start_ns, tid_key) in enumerate(op_rows):
            op_by_slot[slot] = (start_ns, tid_key)

        for act in gpu_activities:
            (name, kind, start_ns, end_ns, device, stream, corr, ext,
             os_tid, _cbid, nbytes, copy_kind, _value) = act
            args = {"correlation": corr}
            if ext is not None and ext != 0xffffffffffffffff:
                args["External id"] = ext
            if nbytes:
                args["bytes"] = nbytes
            if kind == "m":
                args["copy kind"] = int(copy_kind)
            if kind in ("r", "d"):
                # runtime/driver rows live on OS-thread lanes of the CPU pid
                row_pid, row_tid = pid_cpu, lane(("api", os_tid), pid_cpu,
                                                 f"thread {os_tid}")
            else:
                row_pid = pid_gpu_base + max(device, 0)
                row_tid = lane(("stream", device, stream), row_pid,
                               f"stream {stream} ")
            trace_events.append({
                "name": name,
                "cat": gpu_cat.get(kind, "kernel"),
                "ph": "X",
                "ts": rel(start_ns),
                "dur": max((end_ns - start_ns) / 1e3, 0.001),
                "pid": row_pid,
                "tid": row_tid,
                "args": args,
            })
            # ac2g flow: start at the launching op, finish at the activity.
            if (kind not in ("r", "d") and ext is not None and
                    ext != 0xffffffffffffffff and ext in op_by_slot):
                op_ts, op_tid = op_by_slot[ext]
                flow_arrows.append((corr, op_ts, op_tid, pid_cpu,
                                    start_ns, row_tid, row_pid))

        for ts, ptr, nbytes, alloc, is_cuda, device, stream, tid in mem_events:
            dev = f"cuda:{device}" if is_cuda else "cpu"
            trace_events.append({
                "name": "[memory]",
                "cat": "user_annotation",
                "ph": "X",
                "ts": rel(ts),
                "dur": 0.001,
                "pid": pid_cpu,
                "tid": lane(tid),
                "args": {"Bytes": nbytes,
                         "Allocation Type": "alloc" if alloc else "free",
                         "Device": dev},
            })

        for ts, os_tid, chain in samples:
            trace_events.append({
                "name": chain[0],
                "cat": "python_function",
                "ph": "i",
                "s": "t",
                "ts": rel(ts),
                "pid": pid_cpu,
                "tid": lane(("api", os_tid), pid_cpu, f"thread {os_tid}"),
                "args": {"stack": chain},
            })

        for corr, op_ts, op_tid, op_pid, gpu_ts, gpu_tid, gpu_pid \
                in flow_arrows:
            trace_events.append({
                "ph": "s", "id": corr, "pid": op_pid, "tid": op_tid,
                "ts": rel(op_ts), "cat": "ac2g", "name": "ac2g",
            })
            trace_events.append({
                "ph": "f", "id": corr, "pid": gpu_pid, "tid": gpu_tid,
                "ts": rel(gpu_ts), "cat": "ac2g", "name": "ac2g", "bp": "e",
            })

        # after every row registered itself, so emit these first.
        used_gpu_pids = sorted({e["pid"] for e in trace_events
                                if isinstance(e.get("pid"), int) and
                                e["pid"] >= pid_gpu_base})
        first_ts = min((e["ts"] for e in trace_events if "ts" in e),
                       default=0.0)
        meta = []

        def _m(name, pid, tid, arg_key, arg_value):
            meta.append({"ph": "M", "name": name, "ts": first_ts,
                         "pid": pid, "tid": tid,
                         "args": {arg_key: arg_value}})

        _m("process_name", pid_cpu, 0, "name", "python")
        _m("process_labels", pid_cpu, 0, "labels", "CPU")
        _m("process_sort_index", pid_cpu, 0, "sort_index", pid_cpu)
        for d in used_gpu_pids:
            _m("process_name", d, 0, "name", "python")
            _m("process_labels", d, 0, "labels", f"GPU {d - pid_gpu_base}")
            _m("process_sort_index", d, 0, "sort_index", 5000000 + d)
        for (row_pid, lane_id), name in sorted(lane_meta.items()):
            _m("thread_name", row_pid, lane_id, "name", name)
            _m("thread_sort_index", row_pid, lane_id, "sort_index",
               lane_id)
        end_ts = max(
            (e["ts"] + e.get("dur", 0) for e in trace_events
             if e["ph"] == "X" and "ts" in e),
            default=0.0)
        trace_events.append({
            "ph": "i", "s": "g", "name": "Record Window End",
            "pid": "", "tid": "", "ts": end_ts + 1.0,
        })
        trace_events = meta + trace_events

        doc = self._chrome_doc(trace_events, torch_compat=torch_compat,
                               path=path)
        with open(path, "w") as fh:
            json.dump(doc, fh)

    def _device_properties(self):
        props = []
        try:
            import tensorplay as tp
            if getattr(tp.cuda, "is_available", lambda: False)():
                n = tp.cuda.device_count()
                for i in range(n):
                    p = tp.cuda.get_device_properties(i)
                    props.append({
                        "id": i,
                        "name": getattr(p, "name", "unknown"),
                        "totalGlobalMem": getattr(p, "total_memory", 0),
                        "computeMajor": getattr(p, "major", 0),
                        "computeMinor": getattr(p, "minor", 0),
                        "maxThreadsPerBlock":
                            getattr(p, "max_threads_per_block", 0),
                        "maxThreadsPerMultiprocessor":
                            getattr(p, "multi_processor_count", 0),
                        "regsPerMultiprocessor":
                            getattr(p, "regs_per_multiprocessor", 0),
                        "warpSize": getattr(p, "warp_size", 32),
                    })
        except Exception:
            pass
        return props

    def _chrome_doc(self, trace_events, torch_compat=False, path=""):
        rank, world = _rank_world()
        doc = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ns",
        }
        if torch_compat:
            import socket
            import uuid
            doc["schemaVersion"] = 1
            doc["deviceProperties"] = self._device_properties()
            doc["displayTimeUnit"] = "ms"
            doc["baseTimeNanoseconds"] = self.base_ns
            doc["with_flops"] = False
            doc["record_shapes"] = any(
                ev[5] is not None for ev in self)
            doc["with_stack"] = any(
                len(ev) > 10 and ev[10] for ev in self)
            doc["traceName"] = path
            doc["host_name"] = socket.gethostname()
            doc["trace_id"] = uuid.uuid4().hex
            # kineto stamps the CUDA stack versions into the trace header;
            try:
                runtime_ver = _C._cuda.get_version()
                driver_ver = _C._cuda.get_driver_version()
            except Exception:
                runtime_ver = driver_ver = 0
            try:
                cupti_ver = _C._profiler_cupti_version()
            except Exception:
                cupti_ver = 0
            if runtime_ver:
                doc["cuda_runtime_version"] = runtime_ver
            if driver_ver:
                doc["cuda_driver_version"] = driver_ver
            if cupti_ver:
                doc["cupti_version"] = cupti_ver
            if rank is not None:
                doc["distributedInfo"] = {
                    "rank": rank,
                    "world_size": world if world is not None else 0,
                    "backend": "tensorplay.distributed",
                }
        return doc


def merge_distributed_traces(paths, out_path):
    """Merges per-rank chrome traces into one multi-process trace.

    Each rank exports with the ``RANK``/``WORLD_SIZE`` environment set (its
    rank rides in ``distributedInfo`` and per-event args).  This helper
    re-bases every rank's ``pid`` into an exclusive lane
    (``rank * 10_000_000 + pid``) so the merged file shows one process lane
    group per rank in Perfetto / chrome://tracing.
    """
    merged = {"traceEvents": [], "displayTimeUnit": "ns",
              "distributedInfo": {"merged_ranks": []}}
    for path in paths:
        with open(path) as fh:
            doc = json.load(fh)
        rank = doc.get("distributedInfo", {}).get("rank")
        if rank is None:
            rank = len(merged["distributedInfo"]["merged_ranks"])
        merged["distributedInfo"]["merged_ranks"].append(rank)
        base = rank * 10_000_000
        for ev in doc.get("traceEvents", []):
            ev = dict(ev)
            if "pid" in ev:
                try:
                    ev["pid"] = base + int(ev["pid"])
                except (TypeError, ValueError):
                    pass
            merged["traceEvents"].append(ev)
    merged["traceName"] = out_path
    with open(out_path, "w") as fh:
        json.dump(merged, fh)
    return out_path


@contextlib.contextmanager
def record_function(name):
    _C._profiler_user_begin(name)
    try:
        yield
    finally:
        _C._profiler_user_end()


@contextlib.contextmanager
def emit_nvtx():
    """

    Makes every dispatched op (and every backward node, via the engine)
    emit a matching NVTX range so ``nsys profile`` timelines show
    TensorPlay operator names.  Works with or without a surrounding
    ``profile()`` session; requires libnvtx at runtime (CUDA toolkit),
    otherwise ranges are silently skipped.
    """
    _C._profiler_emit_nvtx(True)
    try:
        yield
    finally:
        _C._profiler_emit_nvtx(False)


@contextlib.contextmanager
def emit_itt():
    """

    Every dispatched op / backward node emits an ITT task under the
    "tensorplay" domain while active.  Requires libittnotify at runtime;
    silently skipped otherwise.
    """
    _C._profiler_emit_itt(True)
    try:
        yield
    finally:
        _C._profiler_emit_itt(False)


__all__ = ["profile", "record_function", "EventList", "schedule",
           "emit_nvtx", "emit_itt", "merge_distributed_traces"]
