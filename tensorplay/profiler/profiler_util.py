"""Event containers and summary views."""

from __future__ import annotations

import collections
from dataclasses import dataclass

from ._utils import event_gpu_us, self_cuda_us, self_times


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
        return self.cuda_us

    @property
    def self_device_time_total(self):
        return self.self_cuda_us

    def __repr__(self):
        return (
            f"<FunctionEventAvg key={self.key} "
            f"self_cpu_time={self.self_us:.2f}us "
            f"cpu_time={self.total_us:.2f}us count={self.count}>"
        )


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

    def __init__(self, event, self_ns=0, self_cuda_us=0.0):
        values = list(event) + [None] * max(0, 13 - len(event))
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
        ) = values[:13]
        self.flops = self.flops or 0
        self.thread = self.tid
        self.input_shapes = self.shapes
        self.input_dtypes = self.dtypes
        self.device_type = "cpu"
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
        return self.cuda_time_total

    @property
    def self_device_time_total(self):
        return self.self_cuda_time_total

    def __repr__(self):
        return f"<FunctionEvent {self.name} {self.cpu_time:.2f}us>"


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
            self._function_events = [
                FunctionEvent(event, cpu_ns, cuda_us) for event, cpu_ns, cuda_us in zip(self, own_ns, own_cuda_us)
            ]
        return self._function_events

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
