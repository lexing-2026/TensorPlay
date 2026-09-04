"""Deadline contracts for long-running elastic phases.

Workers hold a :class:`TimerClient` to register scope deadlines; a
:class:`TimerServer` running next to the agent watches deadlines and
interrupts workers that overstay them.
"""
import abc
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from inspect import getframeinfo, stack
from typing import Any

logger = logging.getLogger(__name__)


class TimerRequest:
    """One deadline registration and its compatibility aliases."""

    __slots__ = ["worker_id", "scope_id", "expiration_time"]

    def __init__(
        self,
        *args,
        worker_id: Any = None,
        scope_id: str | None = None,
        expiration_time: float | datetime | None = None,
        scope: str | None = None,
        expire_time: float | datetime | None = None,
    ) -> None:
        if len(args) >= 3:
            worker_id, scope_id, expiration_time = args[:3]
        elif len(args) == 2:
            scope, expire_time = args
        elif len(args) == 1:
            raise TypeError("TimerRequest requires a scope and expiration")
        scope_id = scope_id if scope_id is not None else scope
        expiration_time = (
            expiration_time if expiration_time is not None else expire_time
        )
        if scope_id is None or expiration_time is None:
            raise TypeError("TimerRequest requires a scope and expiration")
        self.worker_id = worker_id
        self.scope_id = str(scope_id)
        self.expiration_time = (
            expiration_time.timestamp()
            if isinstance(expiration_time, datetime)
            else expiration_time
        )

    @property
    def scope(self) -> str:
        return self.scope_id

    @property
    def expire_time(self) -> datetime:
        return datetime.fromtimestamp(float(self.expiration_time))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, TimerRequest)
            and self.worker_id == other.worker_id
            and self.scope_id == other.scope_id
            and self.expiration_time == other.expiration_time
        )

    def __repr__(self) -> str:
        return (
            f"TimerRequest(worker_id={self.worker_id!r}, "
            f"scope_id={self.scope_id!r}, expiration_time={self.expiration_time!r})"
        )


class TimerClient(abc.ABC):
    """Client side of the timer contract."""

    @abc.abstractmethod
    def acquire(self, scope_id: str, expiration_time: float) -> None:
        ...

    @abc.abstractmethod
    def release(self, scope_id: str) -> None:
        ...

    def start_timer(self, request: TimerRequest) -> None:
        self.acquire(request.scope_id, float(request.expiration_time))

    def acquire_scope(self, scope: str, expiration_time: datetime) -> None:
        value = expiration_time.timestamp() if isinstance(expiration_time, datetime) else expiration_time
        self.acquire(scope, float(value))

    def cancel_scope(self, scope: str) -> None:
        self.release(scope)


class RequestQueue(abc.ABC):
    """FIFO transport of timer requests between clients and a server."""

    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def get(self, size: int = 1, timeout: float = 0):
        ...

    @abc.abstractmethod
    def put(self, request: TimerRequest) -> None:
        ...


class TimerServer(abc.ABC):
    """Watches outstanding deadlines and reacts when they expire."""

    def __init__(self, request_queue: RequestQueue, max_interval: float, daemon: bool = True) -> None:
        self._request_queue = request_queue
        self._max_interval = max_interval
        self._daemon = daemon
        self._watchdog_thread: threading.Thread | None = None
        self._stop_signaled = False

    @abc.abstractmethod
    def register_timers(self, timer_requests: list[TimerRequest]) -> None:
        ...

    @abc.abstractmethod
    def clear_timers(self, worker_ids: set[Any]) -> None:
        ...

    @abc.abstractmethod
    def get_expired_timers(self, deadline: float) -> dict[Any, list[TimerRequest]]:
        ...

    @abc.abstractmethod
    def _reap_worker(self, worker_id: Any) -> bool:
        ...

    def _reap_worker_no_throw(self, worker_id: Any) -> bool:
        try:
            return self._reap_worker(worker_id)
        except Exception:
            logger.exception("Uncaught exception while reaping worker %s", worker_id)
            return True

    def _get_scopes(self, timer_requests):
        return [request.scope_id for request in timer_requests]

    def _run_watchdog(self) -> None:
        batch_size = max(1, self._request_queue.size())
        requests = self._request_queue.get(batch_size, self._max_interval)
        if requests is None:
            requests = []
        if isinstance(requests, TimerRequest):
            requests = [requests]
        self.register_timers(requests)
        reaped = set()
        for worker_id in self.get_expired_timers(time.time()):
            if self._reap_worker_no_throw(worker_id):
                reaped.add(worker_id)
        self.clear_timers(reaped)

    def _watchdog_loop(self) -> None:
        while not self._stop_signaled:
            try:
                self._run_watchdog()
            except Exception:
                logger.exception("Error running timer watchdog")

    def start(self) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._stop_signaled = False
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=self._daemon
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._stop_signaled = True
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=self._max_interval * 2)
            self._watchdog_thread = None

    def is_running(self) -> bool:
        return self._watchdog_thread is not None and self._watchdog_thread.is_alive()


_default_timer_client: TimerClient | None = None


def configure(timer_client: TimerClient) -> None:
    """Set the process-wide default :class:`TimerClient`."""
    global _default_timer_client
    _default_timer_client = timer_client


@contextmanager
def expires(
    after: float,
    scope: str | None = None,
    client: TimerClient | None = None,
):
    """Context manager asserting the block finishes within ``after`` seconds.

    Registers the deadline with ``client`` (default: the configured client)
    and cancels it on clean exit. If the deadline lapses, the server side
    reacts (typically by raising ``SignalException`` in the worker).
    """
    client = client or _default_timer_client
    if client is None:
        raise RuntimeError("Configure timer client before using countdown timers.")
    if scope is None:
        caller = getframeinfo(stack()[1][0])
        scope = f"{caller.filename}#{caller.lineno}"
    expiration = time.time() + after
    client.acquire(scope, expiration)
    try:
        yield
    finally:
        client.release(scope)
