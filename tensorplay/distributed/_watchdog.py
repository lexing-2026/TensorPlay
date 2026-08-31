from __future__ import annotations

import faulthandler
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Generator

logger = logging.getLogger(__name__)

__all__ = ["_get_watchdog", "init", "shutdown", "stream_timeout", "cpu_timeout", "op_timeout"]


def _default_timeout_callback() -> None:
    logger.error("watchdog timeout; dumping thread stacks")
    faulthandler.dump_traceback()


@dataclass
class _StreamMonitor:
    event: object
    deadline: float
    callback: Callable[[], None]
    cancelled: bool = False
    monitor_id: int = field(default=0, init=False)


class _CancelHandle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._timer: threading.Timer | None = None
        self._monitor: _StreamMonitor | None = None

    def _set_timer_handle(self, timer_handle: threading.Timer) -> None:
        with self._lock:
            if self._cancelled:
                timer_handle.cancel()
            else:
                self._timer = timer_handle

    def _set_stream_monitor(self, monitor: _StreamMonitor) -> None:
        with self._lock:
            if self._cancelled:
                monitor.cancelled = True
            else:
                self._monitor = monitor

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._monitor is not None:
                self._monitor.cancelled = True
                self._monitor = None

    @property
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def cancelled(self) -> bool:
        return self.is_cancelled


class _Watchdog:
    def __init__(self, poll_interval: float | None = None, health_interval: float | None = None, stuck_action: str | None = None) -> None:
        self._poll_interval = float(poll_interval if poll_interval is not None else os.getenv("TP_WATCHDOG_POLL_INTERVAL_SECS", "1.0"))
        self._health_interval = float(health_interval if health_interval is not None else os.getenv("TP_WATCHDOG_HEALTH_INTERVAL_SECS", "30.0"))
        self._stuck_action = (stuck_action or os.getenv("TP_WATCHDOG_STUCK_ACTION", "log")).lower()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._monitors: dict[int, tuple[_StreamMonitor, _CancelHandle]] = {}
        self._next_id = 0
        self._thread = threading.Thread(target=self._run_loop, name="tp-watchdog", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        while not self._stop.wait(self._poll_interval):
            self._poll_monitors()

    def stream_timeout(self, timeout: float | timedelta, callback: Callable[[], None] | None = None) -> _CancelHandle:
        handle = _CancelHandle()
        if callback is None:
            callback = _default_timeout_callback
        seconds = timeout.total_seconds() if isinstance(timeout, timedelta) else float(timeout)
        try:
            import tensorplay as tp

            event = tp.cuda.Event(enable_timing=False)
            event.record()
        except Exception as error:
            raise RuntimeError("stream timeout requires an active accelerator stream") from error
        monitor = _StreamMonitor(event, time.monotonic() + seconds, callback)
        handle._set_stream_monitor(monitor)
        self._add_monitor(monitor, handle)
        return handle

    def _add_monitor(self, monitor: _StreamMonitor, handle: _CancelHandle) -> None:
        with self._lock:
            monitor.monitor_id = self._next_id
            self._next_id += 1
            self._monitors[monitor.monitor_id] = (monitor, handle)

    def _schedule_poll(self) -> None:
        return None

    def _poll_monitors(self) -> None:
        now = time.monotonic()
        expired = []
        with self._lock:
            for monitor_id, (monitor, handle) in list(self._monitors.items()):
                if handle.cancelled or monitor.cancelled:
                    del self._monitors[monitor_id]
                    continue
                completed = False
                try:
                    completed = bool(monitor.event.query())
                except AttributeError:
                    completed = False
                if completed:
                    del self._monitors[monitor_id]
                elif now >= monitor.deadline:
                    del self._monitors[monitor_id]
                    expired.append(monitor)
        for monitor in expired:
            self._fire_callback(monitor.callback, "stream", monitor.monitor_id)

    def cpu_timeout(self, timeout: float | timedelta, callback: Callable[[], None] | None = None) -> _CancelHandle:
        seconds = timeout.total_seconds() if isinstance(timeout, timedelta) else float(timeout)
        callback = callback or _default_timeout_callback
        handle = _CancelHandle()
        timer = threading.Timer(seconds, self._fire_callback, args=(callback, "cpu", -1))
        timer.daemon = True
        handle._set_timer_handle(timer)
        timer.start()
        return handle

    def _register_cpu_timeout(self, callback: Callable[[], None], timeout: float, handle: _CancelHandle) -> None:
        del callback, timeout, handle

    def _fire_callback(self, callback: Callable[[], None], kind: str, monitor_id: int) -> None:
        logger.warning("watchdog %s timeout fired (id=%d)", kind, monitor_id)
        if callback is _default_timeout_callback:
            callback()
            return
        try:
            callback()
        except Exception:
            logger.exception("watchdog callback failed")

    def _health_watchdog_loop(self) -> None:
        return None

    def _handle_stuck_loop(self) -> None:
        if self._stuck_action == "abort":
            os.abort()
        if self._stuck_action == "exit":
            raise SystemExit(1)

    def _drain_del_queue(self) -> int:
        return 0

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._health_interval))


_watchdog: _Watchdog | None = None
_watchdog_lock = threading.Lock()


def _get_watchdog() -> _Watchdog:
    global _watchdog
    with _watchdog_lock:
        if _watchdog is None:
            _watchdog = _Watchdog()
        return _watchdog


def shutdown() -> None:
    global _watchdog
    with _watchdog_lock:
        current, _watchdog = _watchdog, None
    if current is not None:
        current.shutdown()


def init(*, poll_interval: float | None = None, health_interval: float | None = None, stuck_action: str | None = None) -> None:
    global _watchdog
    with _watchdog_lock:
        old = _watchdog
        _watchdog = _Watchdog(poll_interval, health_interval, stuck_action)
    if old is not None:
        old.shutdown()


def stream_timeout(timeout: float | timedelta, callback: Callable[[], None] | None = None) -> _CancelHandle:
    return _get_watchdog().stream_timeout(timeout, callback)


def cpu_timeout(timeout: float | timedelta, callback: Callable[[], None] | None = None) -> _CancelHandle:
    return _get_watchdog().cpu_timeout(timeout, callback)


@contextmanager
def op_timeout(timeout: float | timedelta, callback: Callable[[], None] | None = None) -> Generator[None, None, None]:
    handle = cpu_timeout(timeout, callback)
    try:
        yield
    finally:
        handle.cancel()
