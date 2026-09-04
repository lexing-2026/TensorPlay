"""Bounded queues for communication completion events."""

from collections import deque
from typing import Any

__all__ = ["_FreeEventQueue"]


class _FreeEventQueue:
    def __init__(self, max_num: int = 2) -> None:
        self.max_num = int(max_num)
        self._max_num_inflight_all_gathers = self.max_num
        self._queue: deque[Any] = deque()

    def enqueue(self, event: Any) -> None:
        self._queue.append(event)

    def dequeue_if_needed(self) -> Any:
        if self._max_num_inflight_all_gathers and len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()
        return None

    def _dequeue(self) -> Any:
        return self._queue.popleft() if self._queue else None

    def push(self, event: Any) -> None:
        self.enqueue(event)
        old = self.dequeue_if_needed()
        if old is not None:
            wait = getattr(old, "wait", None)
            if callable(wait):
                wait()

    def record(self, stream: Any) -> None:
        record = getattr(stream, "record_event", None)
        if record is not None:
            self.push(record())

    def drain(self) -> None:
        while self._queue:
            event = self._queue.popleft()
            wait = getattr(event, "wait", None)
            if wait is not None:
                wait()
