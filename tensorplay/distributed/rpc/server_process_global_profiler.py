from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

__all__: list[str] = []
_events_lock = threading.RLock()
_events: list["_Event"] = []


@dataclass(frozen=True)
class _Event:
    name: str
    start: float
    end: float
    caller: str
    destination: str

    @property
    def cpu_time_total(self) -> float:
        return (self.end - self.start) * 1_000_000


@dataclass(frozen=True)
class _EventSummary:
    key: str
    count: int
    cpu_time_total: float

    @property
    def self_cpu_time_total(self) -> float:
        return self.cpu_time_total

    @property
    def cpu_time_avg(self) -> float:
        return self.cpu_time_total / self.count if self.count else 0.0


class _EventList(list[_Event]):
    def key_averages(self, *args: Any, **kwargs: Any) -> list[_EventSummary]:
        del args, kwargs
        grouped: dict[str, list[_Event]] = defaultdict(list)
        for event in self:
            grouped[event.name].append(event)
        return [
            _EventSummary(name, len(values), sum(item.cpu_time_total for item in values))
            for name, values in sorted(grouped.items())
        ]

    def table(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return "\n".join(
            f"{item.key}\t{item.count}\t{item.cpu_time_total:.3f}"
            for item in self.key_averages()
        )


def _record_server_event(name: str, start: float, end: float, caller: str, destination: str) -> None:
    with _events_lock:
        _events.append(_Event(name, start, end, caller, destination))


class _server_process_global_profile:
    def __init__(self, enabled: bool = True, use_cuda: bool = False, record_shapes: bool = False, profile_memory: bool = False, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.enabled = bool(enabled)
        self.use_cuda = bool(use_cuda)
        self.record_shapes = bool(record_shapes)
        self.profile_memory = bool(profile_memory)
        self.entered = False
        self.function_events = _EventList()
        self.process_global_function_events: list[list[_Event]] = []
        self._start_index = 0

    def __enter__(self) -> "_server_process_global_profile | None":
        if not self.enabled:
            return None
        if self.entered:
            raise RuntimeError("profiler contexts are not reentrant")
        with _events_lock:
            self._start_index = len(_events)
        self.entered = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback_value: Any) -> bool:
        del exc_type, exc_value, traceback_value
        if not self.enabled:
            return False
        with _events_lock:
            selected = list(_events[self._start_index :])
        self.function_events = _EventList(selected)
        self.process_global_function_events = [selected]
        self.entered = False
        return False
