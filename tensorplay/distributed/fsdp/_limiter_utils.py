"""Bounded queues for communication completion events."""

from collections import deque
from typing import Any

__all__ = ["_FreeEventQueue"]


class _FreeEventQueue:
    def __init__(self, max_num: int = 0) -> None:
        self.max_num = int(max_num)
        self._queue: deque[Any] = deque()

    def push(self, event: Any) -> None:
        self._queue.append(event)
        while self.max_num and len(self._queue) > self.max_num:
            old = self._queue.popleft()
            wait = getattr(old, "wait", None)
            if wait is not None:
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
