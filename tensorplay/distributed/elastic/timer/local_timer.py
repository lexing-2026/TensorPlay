from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
from datetime import datetime
from queue import Empty
from typing import Any

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer

__all__ = ["LocalTimerClient", "MultiprocessingRequestQueue", "LocalTimerServer"]

logger = logging.getLogger(__name__)


class _ProcessTimerRequest(TimerRequest):
    def __init__(self, scope: str, expire_time, worker_id: int = 0) -> None:
        super().__init__(worker_id, scope, expire_time)


class LocalTimerClient(TimerClient):
    def __init__(self, mp_queue) -> None:
        self._mp_queue = mp_queue

    def start_timer(self, request: TimerRequest) -> None:
        self._mp_queue.put(request)

    def acquire_scope(self, scope: str, expiration_time: datetime) -> None:
        value = (
            expiration_time.timestamp()
            if isinstance(expiration_time, datetime)
            else expiration_time
        )
        self.acquire(scope, float(value))

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
        self._timers: dict[tuple[int, str], _ProcessTimerRequest] = {}

    def register_timers(self, timer_requests: list[TimerRequest]) -> None:
        for request in timer_requests:
            key = (request.worker_id, request.scope_id)
            if float(request.expiration_time) < 0:
                self._timers.pop(key, None)
            else:
                self._timers[key] = request

    def clear_timers(self, worker_ids: set[int]) -> None:
        for key in list(self._timers):
            if key[0] in worker_ids:
                self._timers.pop(key, None)

    def get_expired_timers(self, deadline: float) -> dict[int, list[TimerRequest]]:
        expired: dict[int, list[TimerRequest]] = {}
        for request in self._timers.values():
            if float(request.expiration_time) <= deadline:
                expired.setdefault(request.worker_id, []).append(request)
        return expired

    def _reap_worker(self, worker_id: int) -> bool:
        try:
            os.kill(worker_id, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return True

    def _process_waiting_timers(self) -> None:
        queue = self._request_queue
        while True:
            request = queue.get()
            if request is None:
                return
            if not isinstance(request, _ProcessTimerRequest):
                request = _ProcessTimerRequest(scope=request.scope, expire_time=request.expire_time, worker_id=os.getpid())
            if request.expire_time.timestamp() <= 0:
                self._timers.pop((request.worker_id, request.scope_id), None)
            else:
                self._timers[(request.worker_id, request.scope_id)] = request

    def _handle_timer(self, request: TimerRequest) -> bool:
        worker_id = getattr(request, "worker_id", None)
        if worker_id is None:
            return True
        try:
            return self._reap_worker(int(worker_id))
        except Exception:
            logger.exception("unable to terminate worker %s", worker_id)
            return False
