"""Deadline contracts for long-running elastic phases.

Workers hold a :class:`TimerClient` to register scope deadlines; a
:class:`TimerServer` running next to the agent watches deadlines and
interrupts workers that overstay them.
"""
import abc
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TimerRequest:
    """One deadline registration: ``scope`` must finish before ``expire_time``."""

    scope: str
    expire_time: datetime

    def __repr__(self) -> str:
        return f"TimerRequest(scope={self.scope!r}, expire_time={self.expire_time.isoformat()!r})"


class TimerClient(abc.ABC):
    """Client side of the timer contract."""

    @abc.abstractmethod
    def start_timer(self, request: TimerRequest) -> None:
        ...

    @abc.abstractmethod
    def acquire_scope(self, scope: str, expiration_time: datetime) -> None:
        ...

    @abc.abstractmethod
    def cancel_scope(self, scope: str) -> None:
        ...


class RequestQueue(abc.ABC):
    """FIFO transport of timer requests between clients and a server."""

    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def get(self) -> TimerRequest | None:
        ...

    @abc.abstractmethod
    def put(self, request: TimerRequest) -> None:
        ...


class TimerServer(abc.ABC):
    """Watches outstanding deadlines and reacts when they expire.

    Subclasses implement :meth:`_handle_timer` and :meth:`_process_waiting_timers`
    semantics; the watchdog loop calls them periodically.
    """

    def __init__(self, request_queue: RequestQueue, max_interval: float, daemon: bool = True) -> None:
        self._request_queue = request_queue
        self._max_interval = max_interval
        self._daemon = daemon
        self._watchdog_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._timers: dict[str, TimerRequest] = {}
        self._reaped_scopes: dict[str, TimerRequest] = {}
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        """Whether the watchdog loop is alive."""
        return self._watchdog_thread is not None and self._watchdog_thread.is_alive()

    def start(self) -> None:
        """Start the watchdog loop once."""
        if self.is_running():
            return
        self._stop.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=self._daemon)
        self._watchdog_thread.start()

    def stop(self) -> None:
        """Stop the watchdog loop and join it."""
        self._stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=self._max_interval * 2)
            self._watchdog_thread = None

    @abc.abstractmethod
    def _handle_timer(self, request: TimerRequest) -> bool:
        """React to an expired timer; return True once fully processed."""
        ...

    @abc.abstractmethod
    def _process_waiting_timers(self) -> None:
        """Poll the queue for new registrations."""
        ...

    def _watchdog(self) -> None:
        while not self._stop.is_set():
            try:
                self._process_waiting_timers()
                now = datetime.now()
                with self._lock:
                    expired = [
                        (scope, req)
                        for scope, req in list(self._timers.items())
                        if req.expire_time <= now
                    ]
                    for scope, req in expired:
                        del self._timers[scope]
                        self._reaped_scopes[scope] = req
                for scope, req in expired:
                    self._handle_timer(req)
                with self._lock:
                    for scope, req in list(self._reaped_scopes.items()):
                        if self._handle_timer(req):
                            del self._reaped_scopes[scope]
                        else:
                            self._timers[scope] = req
            except Exception:  # pragma: no cover - watchdog must not die
                pass
            self._stop.wait(self._max_interval)

    def register_timer(self, request: TimerRequest) -> None:
        """Add or replace a deadline."""
        with self._lock:
            self._timers[request.scope] = request


_default_timer_client: TimerClient | None = None


def configure(timer_client: TimerClient) -> None:
    """Set the process-wide default :class:`TimerClient`."""
    global _default_timer_client
    _default_timer_client = timer_client


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

    @contextmanager
    def _expires():
        actual_scope = scope or f"expires@{id(_expires)}"
        if client is not None:
            client.acquire_scope(actual_scope, datetime.now() + timedelta(seconds=after))
        try:
            yield
        finally:
            if client is not None:
                client.cancel_scope(actual_scope)

    return _expires()
