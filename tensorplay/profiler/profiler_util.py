"""Event containers and summary views."""

from __future__ import annotations

import collections
from dataclasses import dataclass

import tensorplay

from ._utils import (
    event_gpu_us,
    nested_cuda_us,
    nested_spans,
    self_cuda_us,
    self_times,
)


@dataclass(frozen=True)
class Interval:
    """An interval expressed in microseconds."""

    start: float
    end: float


class FunctionEventAvg:
    """Aggregated statistics for one event group."""

    def __init__(
        self,
        name,
        kind,
        count,
        avg_ns,
        min_ns,
        max_ns,
        shapes,
        self_ns=0,
        self_pct=0.0,
        cuda_us=0.0,
        kernel_count=0,
        self_cuda_us=0.0,
        stack=None,
        flops=0,
    ):
        self.name = name
        self.key = name
        self.kind = kind
        self.count = count
        self.input_shapes = shapes
        self.stack = stack
        self.avg_ns = avg_ns
        self.min_ns = min_ns
        self.max_ns = max_ns
        self.self_ns = self_ns
        self.self_pct = self_pct
        self.cuda_us = cuda_us
        self.kernel_count = kernel_count
        self.self_cuda_us = self_cuda_us
        self.flops = flops

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
    def cpu_time(self):
        return self.avg_us

    @property
    def cpu_time_total(self):
        return self.total_us

    @property
    def self_cpu_time_total(self):
        return self.self_us

    @property
    def self_cuda_time_total(self):
        return self.self_cuda_us

    @property
    def cuda_time_total(self):
        return self.cuda_us

    @property
    def device_time_total(self):
        return self.cuda_time_total

    @property
    def self_device_time_total(self):
        return self.self_cuda_us

    def __repr__(self):
        return (
            f"<FunctionEventAvg key={self.key} "
            f"self_cpu_time={self.self_us:.2f}us "
            f"cpu_time={self.total_us:.2f}us count={self.count}>"
        )


def _demangle_all(names):
    """The same names with the code names in them undone, where that is possible.

    A kernel this project's own code emits is compiled code, so what it is
    called is a code name: hundreds of characters, mostly template arguments
    nobody reading a profile is looking for.  Undoing that is worth doing, and
    worth not failing over when the tool for it is not here: a name that cannot
    be read is still a name, and a profile that raised would be worth less than
    one that is merely hard to read.

    Done for the whole recording at once, because the work is one call for any
    number of names and a call each would cost more than the reading saves.
    """

    wanted = [n for n in dict.fromkeys(names) if isinstance(n, str) and n.startswith("_Z")]
    if not wanted:
        return {}
    demangle = getattr(getattr(tensorplay, "_C", None), "_demangle", None)
    out = {}
    if demangle is not None:
        for name in wanted:
            try:
                out[name] = demangle(name) or name
            except Exception:
                pass
        if out:
            return out
    try:
        import subprocess

        done = subprocess.run(
            ["c++filt"], input="\n".join(wanted), capture_output=True, text=True
        ).stdout.splitlines()
    except Exception:
        return {}
    if len(done) != len(wanted):
        return {}
    for name, readable in zip(wanted, done):
        out[name] = readable.strip() or name
    return out


#: What the device did rather than what the host asked it to do.  A kernel is
#: "k"; a copy between host and device or between devices is "m".
_DEVICE_WORK_KINDS = frozenset({"k", "m"})


class _DeviceKernelTable:
    """One row per piece of work the device actually ran, with the time it took.

    A row per operation answers which operation was dispatched.  It does not
    answer what ran for it, and the two are not the same question: one
    operation may run several kernels, and work may be launched without going
    through any operation that is named after it -- a kernel this project's own
    code emits, for one.  Joining device time to the operation that launched it
    cannot answer that, so what the device recorded about its own work is kept
    and offered here under its own names.

    Only the device's own work is here -- kernels and copies.  A call the host
    made to ask for that work is not itself work the device did, and one of
    them can be the largest entry in a recording while saying nothing about
    where the time went: a synchronisation waits for everything and is not
    itself the reason anything took as long as it did.
    """

    def __init__(self, gpu_activities=(), sort_by=None):
        self.has_gpu = True
        self.with_flops = False
        self.has_flops = False
        aggregate = collections.OrderedDict()
        for activity in gpu_activities or ():
            if activity is None or len(activity) < 4:
                continue
            name, kind, start_ns, end_ns = activity[0], activity[1], activity[2], activity[3]
            if kind not in _DEVICE_WORK_KINDS:
                continue
            if start_ns is None or end_ns is None or end_ns <= start_ns:
                continue
            row = aggregate.get((name, kind))
            if row is None:
                row = [0, 0, None, None]
                aggregate[(name, kind)] = row
            duration = end_ns - start_ns
            row[0] += 1
            row[1] += duration
            row[2] = duration if row[2] is None else min(row[2], duration)
            row[3] = duration if row[3] is None else max(row[3], duration)

        total_ns = sum(row[1] for row in aggregate.values())
        self.total_ns = total_ns
        readable = _demangle_all(key[0] for key in aggregate)
        self.rows = []
        for (name, kind), values in sorted(aggregate.items(), key=lambda item: -item[1][1]):
            count, total, minimum, maximum = values
            self.rows.append(
                FunctionEventAvg(
                    readable.get(name, name),
                    kind,
                    count,
                    total // count if count else 0,
                    minimum,
                    maximum,
                    None,
                    0,
                    0.0,
                    total / 1000.0,
                    count,
                    total / 1000.0,
                )
            )
        self.total_cuda_us = sum(row.cuda_us for row in self.rows)
        if sort_by is not None:
            self.sort(sort_by)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def sort(self, sort_by):
        self.rows.sort(key=lambda row: getattr(row, sort_by, 0), reverse=True)

    def key_averages(self, sort_by=None, row_limit=None):
        if sort_by is not None:
            self.sort(sort_by)
        return self.rows if row_limit is None else self.rows[:row_limit]

    def total_average(self):
        return self.total_cuda_us / 1000.0

    def table(self, sort_by=None, row_limit=-1, **_kwargs):
        if sort_by is not None:
            self.sort(sort_by)
        rows = self.rows if row_limit is None or row_limit < 0 else self.rows[:row_limit]
        header = f"{'Name':<40}{'Kind':>5}{'Calls':>7}{'Total us':>12}{'Avg us':>11}{'Min us':>11}{'Max us':>11}"
        lines = [header, "-" * len(header)]
        for row in rows:
            lines.append(
                f"{row.name:<40}{row.kind:>5}{row.count:>7}"
                f"{row.total_us:>12.2f}{row.avg_us / 1000.0:>11.2f}"
                f"{row.min_ns / 1000.0:>11.2f}{row.max_ns / 1000.0:>11.2f}"
            )
        lines.append("-" * len(header))
        lines.append(f"{'Total':<40}{'':>5}{len(self.rows):>7}{self.total_cuda_us:>12.2f}")
        return "\n".join(lines)


class _FunctionsTable:
    """Aggregate raw event tuples by name and optional input shape."""

    def __init__(
        self,
        events,
        group_by_input_shape=False,
        group_by_stack_n=0,
        sort_by=None,
        with_flops=False,
    ):
        events = list(events)
        self._events = events
        self._group_by_input_shape = group_by_input_shape
        self._group_by_stack_n = group_by_stack_n
        self_ns = self_times(events)
        self_cuda = self_cuda_us(events)
        aggregate = collections.OrderedDict()
        has_gpu = False
        has_flops = False

        for event, own_ns, own_cuda_us in zip(events, self_ns, self_cuda):
            if len(event) < 9:
                continue
            name, kind = event[0], event[1]
            start_ns, end_ns = event[2], event[3]
            shapes = event[5] if len(event) > 5 else None
            kernel_count = event[11] if len(event) > 11 else 0
            gpu_us = event_gpu_us(event)
            if len(event) > 8 and event[8] is not None and event[8] >= 0:
                has_gpu = True
            if end_ns <= start_ns:
                continue

            flops = event[12] if with_flops and len(event) > 12 else 0
            if flops:
                has_flops = True

            key = (name, kind)
            if group_by_input_shape:
                key += (tuple(tuple(shape) for shape in shapes) if shapes is not None else None,)
            stack = event[10] if len(event) > 10 else None
            if group_by_stack_n:
                key += (tuple(stack[-group_by_stack_n:]) if stack else None,)
            row = aggregate.get(key)
            if row is None:
                row = [0, 0, None, None, 0, 0.0, 0.0, 0, 0]
                aggregate[key] = row
            duration = end_ns - start_ns
            row[0] += 1
            row[1] += duration
            row[2] = duration if row[2] is None else min(row[2], duration)
            row[3] = duration if row[3] is None else max(row[3], duration)
            row[4] += own_ns
            row[5] += gpu_us
            row[6] += own_cuda_us
            row[7] += kernel_count
            row[8] += flops

        total_ns = sum(row[1] for row in aggregate.values())
        self.has_gpu = has_gpu
        self.with_flops = bool(with_flops)
        self.has_flops = has_flops
        self.total_ns = total_ns
        self.rows = []
        for key, values in sorted(aggregate.items(), key=lambda item: -item[1][4]):
            count, total, minimum, maximum, own_ns, cuda_us, own_cuda_us, kernels, flops = values
            shapes = key[2] if group_by_input_shape and len(key) > 2 else None
            stack_index = 2 + int(group_by_input_shape)
            stack = key[stack_index] if group_by_stack_n and len(key) > stack_index else None
            self.rows.append(
                FunctionEventAvg(
                    key[0],
                    key[1],
                    count,
                    total // count,
                    minimum,
                    maximum,
                    shapes,
                    own_ns,
                    own_ns / total_ns * 100.0 if total_ns else 0.0,
                    cuda_us,
                    kernels,
                    own_cuda_us,
                    stack,
                    flops,
                )
            )
        self.total_cuda_us = sum(row.cuda_us for row in self.rows)
        self.total_flops = sum(row.flops for row in self.rows)
        if sort_by is not None:
            self.sort(sort_by)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def sort(self, sort_by="self_cpu_time_total"):
        keys = {
            "self_cpu_time": lambda row: row.self_ns,
            "self_cpu_time_total": lambda row: row.self_ns,
            "cpu_time": lambda row: row.total_us,
            "cpu_time_total": lambda row: row.total_us,
            "calls": lambda row: row.count,
            "count": lambda row: row.count,
            "name": lambda row: row.name,
            "self_cuda_time": lambda row: row.self_cuda_us,
            "self_cuda_time_total": lambda row: row.self_cuda_us,
            "cuda_time": lambda row: row.cuda_us,
            "cuda_time_total": lambda row: row.cuda_us,
            "flops": lambda row: row.flops,
            "total_flops": lambda row: row.flops,
        }
        if sort_by not in keys:
            raise ValueError(f"unsupported sort_by: {sort_by}")
        self.rows.sort(key=keys[sort_by], reverse=sort_by != "name")
        return self

    def table(self, sort_by=None, row_limit=-1, **_kwargs):
        if sort_by is not None:
            self.sort(sort_by)
        rows = self.rows if row_limit is None or row_limit < 0 else self.rows[:row_limit]
        header = f"{'Name':<28}{'Calls':>7}{'Self us':>10}{'Self %':>8}{'Total us':>10}"
        show_flops = self.with_flops and self.has_flops
        if show_flops:
            header += f"{'Total Flops':>18}"
        if self.has_gpu:
            header += f"{'Self CUDA us':>13}{'CUDA Total us':>15}{'#K':>6}"
        lines = [header, "-" * len(header)]
        for row in rows:
            line = f"{row.name:<28}{row.count:>7}{row.self_us:>10.2f}{row.self_pct:>7.1f}%{row.total_us:>10.2f}"
            if show_flops:
                line += f"{row.flops:>18,}"
            if self.has_gpu:
                line += f"{row.self_cuda_us:>13.2f}{row.cuda_us:>15.2f}{row.kernel_count:>6}"
            lines.append(line)
        lines.append("-" * len(header))
        lines.append(f"Self CPU time total: {self.total_ns / 1e3:.2f} us")
        if show_flops:
            lines.append(f"Total Flops: {self.total_flops:,}")
        if self.has_gpu:
            lines.append(f"Self CUDA time total: {self.total_cuda_us:.2f} us")
        return "\n".join(lines)

    def total_average(self):
        if not self.rows:
            return FunctionEventAvg("Total", "o", 0, 0, 0, 0, None)
        count = sum(row.count for row in self.rows)
        total_ns = sum(row.total_us for row in self.rows) * 1e3
        own_ns = sum(row.self_ns for row in self.rows)
        return FunctionEventAvg(
            "Total",
            "o",
            count,
            int(total_ns / count) if count else 0,
            min(row.min_ns for row in self.rows),
            max(row.max_ns for row in self.rows),
            None,
            own_ns,
            100.0 if total_ns else 0.0,
            sum(row.cuda_us for row in self.rows),
            sum(row.kernel_count for row in self.rows),
            sum(row.self_cuda_us for row in self.rows),
            flops=sum(row.flops for row in self.rows),
        )

    @property
    def self_cpu_time_total(self):
        return sum(row.self_cpu_time_total for row in self.rows)

    @property
    def cpu_time_total(self):
        return sum(row.cpu_time_total for row in self.rows)

    def __str__(self):
        return self.table()

    __repr__ = __str__


class FunctionEvent:
    """Read-only object view over one collected event tuple."""

    def __init__(self, event, self_ns=0, self_cuda_us=0.0, children=(), device_us=None):
        values = list(event) + [None] * max(0, 14 - len(event))
        (
            self.name,
            self.kind,
            self.start_ns,
            self.end_ns,
            self.tid,
            self.shapes,
            self.dtypes,
            self.site,
            self.gpu_ms,
            self.out_bytes,
            self.stack,
            self.kernel_count,
            self.flops,
            self.slot_id,
        ) = values[:14]
        self.flops = self.flops or 0
        self.id = self.slot_id
        # What this span dispatched, in dispatch order.  A span is timed on the
        # device by adding up the device work of everything under it, and adding
        # that up needs to know what is under it rather than what merely overlaps
        # it: two spans on one thread overlap whenever one waits, and a wait is
        # not a use of the device.
        self.cpu_children = list(children)
        # The device time of the work dispatched inside this span, which for a
        # span that dispatches work of its own is its own and for a span that
        # only encloses work is the whole of what was asked about.  Left unset
        # rather than zero when no device activity was collected, so that "no
        # activity" is not read as "no device work".
        self.device_us = device_us
        self.thread = self.tid
        self.input_shapes = self.shapes
        self.input_dtypes = self.dtypes
        self.device_type = _device_type("cpu")
        self.is_async = False
        self.scope = {"o": 0, "u": 0, "b": 1}.get(self.kind, 0)
        self.cpu_interval = Interval(self.start_ns / 1e3, self.end_ns / 1e3)
        self.self_ns = self_ns
        self.self_cuda_us = self_cuda_us

    @property
    def key(self):
        return self.name

    @property
    def cpu_time(self):
        return (self.end_ns - self.start_ns) / 1e3

    @property
    def cpu_time_total(self):
        return self.cpu_time

    @property
    def self_cpu_time_total(self):
        return self.self_ns / 1e3

    @property
    def cuda_time_total(self):
        return max(float(self.gpu_ms or 0.0), 0.0) * 1000.0

    @property
    def self_cuda_time_total(self):
        return self.self_cuda_us

    @property
    def device_time_total(self):
        """Device time for this span and for the work dispatched inside it.

        The span's own device time when it dispatched any, and otherwise the sum
        over what it encloses; a span that did both is both, because a caller
        asking what a region cost is asking about the region and not about the
        one call inside it that happened to be the timed one.  An annotation
        around a piece of work dispatches nothing itself, so for the span whose
        time is actually wanted this is the whole answer rather than zero.
        """

        if self.device_us is not None:
            return self.device_us
        return self.cuda_time_total

    @property
    def self_device_time_total(self):
        return self.self_cuda_time_total

    def __repr__(self):
        return f"<FunctionEvent {self.name} {self.cpu_time:.2f}us>"


def _device_type(name):
    """A device kind as the value a reader compares against.

    Compared by identity of the kind rather than by the spelling of the device,
    so that a record saying which kind it is on and a caller saying which kind it
    expected are talking about the same thing rather than about two strings that
    happen to look alike.
    """

    import tensorplay as tp

    kind = getattr(tp.DeviceType, str(name).upper(), None)
    return name if kind is None else kind


class DeviceActivity:
    """One piece of work the device did, as the collector recorded it.

    Read through methods rather than fields because a caller asking which
    operation a piece of device work belongs to should not have to know that
    the join runs through the correlation id the launch record shares with it,
    and should not have to know which of the two ends of that id is the one
    naming the operation.
    """

    #: A record the collector gathered with no operation being dispatched, which
    #: is every record whose launch could not be tied to one.
    NO_EXTERNAL_ID = 0xFFFFFFFFFFFFFFFF

    def __init__(self, activity, launch_to_slot=None, device_name="cuda"):
        values = list(activity) + [None] * max(0, 13 - len(activity))
        (
            self._name,
            self._kind,
            self._start_ns,
            self._end_ns,
            self._device,
            self._stream,
            self._correlation,
            self._external_id,
            self._thread_id,
            self._cbid,
            self._bytes,
            self._copy_kind,
            self._value,
        ) = values[:13]
        self._launch_to_slot = launch_to_slot or {}
        # The kind of device, which the index alone does not say: index 0 is the
        # first device of whichever kind this is.
        self._device_name = device_name

    def name(self):
        return self._name

    def correlation_id(self):
        """The id shared with the launch record for this same piece of work."""
        return self._correlation

    def linked_correlation_id(self):
        """The launch this work belongs to, or zero when it belongs to none.

        Zero rather than a sentinel of this code's own, and zero rather than the
        operation, because a caller that cannot find this id among the launches it
        cares about needs to be able to say so, and a value that could also be a
        real one would let it mistake a coincidence for an answer.  Zero is used
        because it is not a launch: the collector numbers launches from one, so
        zero names none of them.

        To ask which *operation* the work belongs to, ask :meth:`op_slot`, which
        resolves the launch rather than reporting it.
        """

        if self._external_id != self.NO_EXTERNAL_ID:
            return self._external_id
        return self._launch_to_slot.get(self._correlation, 0)

    def op_slot(self):
        """The operation that asked for this work, or None when none did.

        A kernel record names the launch that issued it, not the operation that
        asked for the launch, so the operation is reached through the launch:
        either this record carries it, or the record sharing this correlation
        does.  Which one is available depends on whether the collector saw the
        launch, so both are asked.
        """

        if self._external_id != self.NO_EXTERNAL_ID:
            return self._external_id
        return self._launch_to_slot.get(self._correlation)

    def activity_type(self):
        """What kind of record this is, in the spelling a reader expects.

        The device's own annotations are excluded by a caller that wants only the
        device's work, and they arrive here indistinguishable from a launch, so
        the distinction is made here where the kinds are known.
        """

        if self._kind == "k":
            return "gpu_kernel"
        if self._kind in ("m", "s"):
            return "gpu_memcpy"
        if self._kind == "r":
            return "cuda_runtime"
        if self._kind == "d":
            return "cuda_driver"
        return "gpu_user_annotation"

    def device_type(self):
        """Which kind of device this record is work on.

        A launch or a driver call is work the processor did while setting the
        device up, and its duration is not device time however much of it
        happened to be issued from a device queue; a kernel, a copy or a fill is
        the device's own work and its duration is.  The two are told apart here
        so that a reader adding up device time does not add up the launching.
        """

        if self._kind in ("k", "m", "s"):
            return _device_type(self._device_name or "cuda")
        return _device_type("cpu")

    def start_ns(self):
        return self._start_ns

    def end_ns(self):
        return self._end_ns

    def duration_us(self):
        return (self._end_ns - self._start_ns) / 1000.0

    def __repr__(self):
        return f"<DeviceActivity {self._kind} {str(self._name)[:32]}>"


def device_activities(gpu_activities, device_name="cuda"):
    """The device's work as records a reader can ask questions of.

    The correlation-to-operation map is built once here and shared by every
    record, because answering "which operation is this" for one record means
    looking through all of them, and doing that per record would make asking
    about a session cost a pass over the session per record.
    """

    launch_to_slot: dict[int, int] = {}
    for activity in gpu_activities:
        if len(activity) > 7 and activity[7] != DeviceActivity.NO_EXTERNAL_ID:
            launch_to_slot.setdefault(activity[6], activity[7])
    return [DeviceActivity(a, launch_to_slot, device_name) for a in gpu_activities]


class EventList(list):
    """Raw profiler events with analysis and export helpers."""

    def __init__(
        self,
        raw=(),
        base_ns=0,
        gpu_activities=None,
        mem_events=None,
        samples=None,
        with_stack=False,
    ):
        super().__init__(raw)
        self.base_ns = base_ns
        self.gpu_activities = gpu_activities if gpu_activities is not None else []
        self.mem_events = mem_events if mem_events is not None else []
        self.samples = samples if samples is not None else []
        self.with_stack = with_stack
        self.with_flops = False
        self._function_events = None
        self._device_time = None

    def _invalidate(self):
        self._function_events = None

    def append(self, value):
        super().append(value)
        self._invalidate()

    def extend(self, values):
        super().extend(values)
        self._invalidate()

    def insert(self, index, value):
        super().insert(index, value)
        self._invalidate()

    def clear(self):
        super().clear()
        self._invalidate()

    def __call__(self):
        return self.function_events

    @property
    def function_events(self):
        if self._function_events is None:
            own_ns = self_times(self)
            own_cuda_us = self_cuda_us(self)
            children = nested_spans(self)
            device_us = nested_cuda_us(self, self.gpu_activities)
            # Every event is built before any is given its children, because a
            # child is an event and an event is what a list of children holds.
            built = [
                FunctionEvent(event, cpu_ns, cuda_us, (), dev)
                for event, cpu_ns, cuda_us, dev in zip(
                    self, own_ns, own_cuda_us, device_us
                )
            ]
            for index, kids in enumerate(children):
                built[index].cpu_children = [built[c] for c in kids]
            self._function_events = built
        return self._function_events

    @property
    def device_time_by_span(self):
        """For each collected span, the device time of the work inside it.

        Kept alongside the events because the two answer different questions: a
        span's own event says how long the span took on the host, and this says
        how long the device was busy for it, which for a span that only encloses
        work is the whole of what was being asked.
        """

        if self._device_time is None:
            self._device_time = nested_cuda_us(self, self.gpu_activities)
        return self._device_time

    def key_averages(
        self,
        group_by_input_shape=False,
        group_by_stack_n=0,
        group_by_overload_name=False,
        include_python_functions=False,
        with_flops=False,
    ):
        del group_by_overload_name, include_python_functions
        return _FunctionsTable(
            self,
            group_by_input_shape=group_by_input_shape,
            group_by_stack_n=group_by_stack_n,
            with_flops=bool(with_flops) or bool(self.with_flops),
        )

    def device_kernels(self, sort_by=None):
        """What the device ran, by the name the device knows it by.

        Separate from the per-operation view because a kernel is not an
        operation: work this project's own code launches belongs to whatever
        operation happened to be in flight, and naming it is the only way to
        find out it ran at all.
        """

        return _DeviceKernelTable(self.gpu_activities, sort_by=sort_by)

    def table(self, sort_by=None, row_limit=100, **kwargs):
        return self.key_averages().table(sort_by=sort_by, row_limit=row_limit, **kwargs)

    def export_chrome_trace(self, path, torch_compat=False, **kwargs):
        from ._chrome_trace_export import export_chrome_trace

        return export_chrome_trace(
            self,
            path,
            torch_compat=torch_compat,
            **kwargs,
        )

    def export_stacks(self, path, metric="self_cpu_time_total"):
        if metric not in {
            "self_cpu_time_total",
            "self_cpu_time",
            "self_cuda_time_total",
            "self_cuda_time",
        }:
            raise ValueError(f"unsupported stack metric: {metric}")
        cpu_metric = metric.startswith("self_cpu")
        own_times = self_times(self) if cpu_metric else self_cuda_us(self)
        folded = collections.Counter()
        for event, own_ns in zip(self, own_times):
            stack = event[10] if len(event) > 10 else None
            if not stack or own_ns <= 0:
                continue
            value = own_ns / 1e3 if cpu_metric else own_ns
            folded[";".join(reversed(stack))] += value
        with open(path, "w") as file:
            for stack, value in folded.most_common():
                file.write(f"{stack} {value:.0f}\n")
        return folded

    def total_average(self):
        return self.key_averages().total_average()

    @property
    def self_cpu_time_total(self):
        return self.key_averages().self_cpu_time_total


__all__ = [
    "EventList",
    "FunctionEvent",
    "FunctionEventAvg",
    "Interval",
]
