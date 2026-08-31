from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Empty
from typing import Any

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer

__all__ = ["LocalTimerClient", "MultiprocessingRequestQueue", "LocalTimerServer"]

logger = logging.getLogger(__name__)


@dataclass
class _ProcessTimerRequest(TimerRequest):
    worker_id: int = 0


class LocalTimerClient(TimerClient):
    def __init__(self, mp_queue) -> None:
        self._mp_queue = mp_queue

    def start_timer(self, request: TimerRequest) -> None:
        self._mp_queue.put(request)

    def acquire_scope(self, scope: str, expiration_time: datetime) -> None:
        self.acquire(scope, expiration_time.timestamp())

    def cancel_scope(self, scope: str) -> None:
        self.release(scope)

    def acquire(self, scope_id: str, expiration_time: float) -> None:
        expire = expiration_time if isinstance(expiration_time, datetime) else datetime.fromtimestamp(float(expiration_time))
        self._mp_queue.put(_ProcessTimerRequest(scope=scope_id, expire_time=expire, worker_id=os.getpid()))

    def release(self, scope_id: str) -> None:
        self._mp_queue.put(_ProcessTimerRequest(scope=scope_id, expire_time=datetime.fromtimestamp(0), worker_id=os.getpid()))


class MultiprocessingRequestQueue(RequestQueue):
    def __init__(self, mp_queue: mp.Queue) -> None:
        self._mp_queue = mp_queue

    def size(self) -> int:
        try:
            return self._mp_queue.qsize()
        except (NotImplementedError, AttributeError):
            return 0

    def get(self, size: int | None = None, timeout: float = 0) -> TimerRequest | list[TimerRequest] | None:
        if size is None:
            try:
                return self._mp_queue.get(timeout=timeout)
            except Empty:
                return None
        values: list[TimerRequest] = []
        deadline = time.monotonic() + max(0.0, timeout)
        for _ in range(max(0, size)):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                values.append(self._mp_queue.get(timeout=remaining))
            except Empty:
                break
        return values

    def put(self, request: TimerRequest) -> None:
        self._mp_queue.put(request)


class LocalTimerServer(TimerServer):
    def __init__(self, mp_queue: mp.Queue, max_interval: float = 60, daemon: bool = True) -> None:
        super().__init__(MultiprocessingRequestQueue(mp_queue), max_interval, daemon)
        self._process_timers: dict[tuple[int, str], _ProcessTimerRequest] = {}

    def _process_waiting_timers(self) -> None:
        queue = self._request_queue
        while True:
            request = queue.get()
            if request is None:
                return
            if not isinstance(request, _ProcessTimerRequest):
                request = _ProcessTimerRequest(scope=request.scope, expire_time=request.expire_time, worker_id=os.getpid())
            if request.expire_time.timestamp() <= 0:
                self._process_timers.pop((request.worker_id, request.scope), None)
                with self._lock:
                    self._timers.pop(request.scope, None)
            else:
                self._process_timers[(request.worker_id, request.scope)] = request
                self.register_timer(request)

    def _handle_timer(self, request: TimerRequest) -> bool:
        worker_id = getattr(request, "worker_id", None)
        if worker_id is None:
            return True
        try:
            os.kill(int(worker_id), signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            logger.exception("unable to terminate worker %s", worker_id)
            return False
        return True
