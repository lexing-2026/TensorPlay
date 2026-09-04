"""File-based timer queue shared between workers and a local timer server.

Clients drop JSON request files into ``request_dir``; the server scans the
directory, tracks deadlines per scope, and re-invokes the handler for
expired scopes until the handler reports completion.
"""
import json
import io
import os
import select
import signal
import tempfile
import threading
import time
from functools import wraps
from datetime import datetime
from typing import Callable

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer
from .debug_info_logging import log_debug_info_for_expired_timers


def _retry(max_retries: int, sleep_time: float):
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt + 1 == max_retries:
                        raise
                    time.sleep(sleep_time)

        return wrapped

    return decorate


class FileTimerRequest(TimerRequest):
    """A :class:`TimerRequest` that serializes to/from a JSON file."""

    def __init__(
        self,
        *args,
        worker_pid: int | None = None,
        scope_id: str | None = None,
        expiration_time: float | datetime | None = None,
        signal: int = 0,
        scope: str | None = None,
        expire_time: float | datetime | None = None,
    ) -> None:
        if args:
            worker_pid, scope_id, expiration_time = (list(args) + [0, "", -1])[:3]
            if len(args) > 3:
                signal = args[3]
        super().__init__(
            worker_id=os.getpid() if worker_pid is None else worker_pid,
            scope_id=scope_id or scope,
            expiration_time=expiration_time
            if expiration_time is not None
            else expire_time,
        )
        self.version = 1
        self.signal = int(signal)

    @property
    def worker_pid(self) -> int:
        return int(self.worker_id)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FileTimerRequest)
            and super().__eq__(other)
            and self.version == other.version
            and self.signal == other.signal
        )

    @staticmethod
    def from_json(json_data: str) -> "FileTimerRequest":
        data = json.loads(json_data)
        if "pid" in data:
            return FileTimerRequest(
                worker_pid=data["pid"],
                scope_id=data["scope_id"],
                expiration_time=data["expiration_time"],
                signal=data.get("signal", 0),
            )
        return FileTimerRequest(
            worker_pid=data.get("worker_id", 0),
            scope_id=data["scope"],
            expiration_time=data["expire_time"],
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "pid": self.worker_pid,
                "scope_id": self.scope_id,
                "expiration_time": self.expiration_time,
                "signal": self.signal,
            }
        )


class FileTimerRequestQueue(RequestQueue):
    """Directory-backed :class:`RequestQueue` (one JSON file per request)."""

    def __init__(self, request_dir: str) -> None:
        self.request_dir = request_dir
        os.makedirs(self.request_dir, exist_ok=True)

    def size(self) -> int:
        return len(self._list_files())

    def get(self, size: int = 1, timeout: float = 0):
        values = []
        deadline = time.monotonic() + max(0.0, timeout)
        while len(values) < max(1, size):
            found = False
            for path in self._list_files():
                found = True
                try:
                    with open(path) as f:
                        request = FileTimerRequest.from_json(f.read())
                    os.unlink(path)
                    values.append(request)
                    if len(values) >= max(1, size):
                        break
                except (OSError, ValueError, KeyError):
                    continue
            if values or found or time.monotonic() >= deadline:
                break
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return values

    def put(self, request: TimerRequest) -> None:
        fd, path = tempfile.mkstemp(dir=self.request_dir, suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(request.to_json())
        os.rename(path, os.path.join(self.request_dir, f"timer_{os.getpid()}_{request.scope}.json"))

    def _list_files(self) -> list[str]:
        return [
            os.path.join(self.request_dir, name)
            for name in sorted(os.listdir(self.request_dir))
            if name.endswith(".json")
        ]


class FileTimerClient(TimerClient):
    """Client writing deadline registrations into a shared directory."""

    def __init__(
        self,
        request_queue: FileTimerRequestQueue | str | None = None,
        file_path: str | None = None,
        signal: int = getattr(signal, "SIGKILL", 9),
    ) -> None:
        if isinstance(request_queue, str) and file_path is None:
            file_path = request_queue
            request_queue = None
        self._request_queue = request_queue
        self._file_path = file_path
        self.signal = signal

    def start_timer(self, request: TimerRequest) -> None:
        if self._request_queue is not None:
            self._request_queue.put(request)
        else:
            self._send_request(
                FileTimerRequest(
                    worker_pid=os.getpid(),
                    scope_id=request.scope_id,
                    expiration_time=request.expiration_time,
                    signal=self.signal,
                )
            )

    def acquire_scope(self, scope: str, expiration_time: datetime) -> None:
        value = (
            expiration_time.timestamp()
            if isinstance(expiration_time, datetime)
            else expiration_time
        )
        self.acquire(scope, float(value))

    def cancel_scope(self, scope: str) -> None:
        self.release(scope)

    @_retry(max_retries=10, sleep_time=0.1)
    def _open_non_blocking(self):
        if self._file_path is None:
            raise BrokenPipeError("File timer path is not configured")
        fd = os.open(self._file_path, os.O_WRONLY | os.O_NONBLOCK)
        return os.fdopen(fd, "wt")

    def _send_request(self, request: FileTimerRequest) -> None:
        try:
            stream = self._open_non_blocking()
        except OSError as exc:
            raise BrokenPipeError("File timer server is not available") from exc
        with stream:
            payload = request.to_json()
            if len(payload) > select.PIPE_BUF:
                raise RuntimeError("File timer request is too large")
            stream.write(payload + "\n")

    def acquire(self, scope_id: str, expiration_time: float) -> None:
        if self._request_queue is not None:
            self._request_queue.put(
                FileTimerRequest(
                    worker_pid=os.getpid(),
                    scope_id=scope_id,
                    expiration_time=expiration_time,
                    signal=self.signal,
                )
            )
            return
        self._send_request(
            FileTimerRequest(
                worker_pid=os.getpid(),
                scope_id=scope_id,
                expiration_time=expiration_time,
                signal=self.signal,
            )
        )

    def release(self, scope_id: str) -> None:
        if self._request_queue is not None:
            self._request_queue.put(
                FileTimerRequest(os.getpid(), scope_id, -1, 0)
            )
        else:
            self._send_request(FileTimerRequest(os.getpid(), scope_id, -1, 0))


class FileTimerServer(TimerServer):
    """Timer server watching a shared request directory.

    ``signal`` is delivered to the owning process when a deadline lapses
    (default SIGTERM); pass ``None`` to leave reaction to the handler.
    """

    def __init__(
        self,
        request_dir: str | None = None,
        signal: signal.Signals = signal.SIGTERM,
        max_interval: float = 60,
        daemon: bool = True,
        scope_signal_map: dict[str, signal.Signals] | None = None,
        file_path: str | None = None,
        run_id: str = "",
        log_event: Callable[[str, FileTimerRequest | None], None] | None = None,
    ) -> None:
        if isinstance(signal, str) and run_id == "":
            run_id = signal
            signal = __import__("signal").SIGTERM
            file_path = request_dir
            request_dir = None
        if file_path is None and run_id and request_dir is not None:
            file_path, request_dir = request_dir, None
        self._file_path = file_path
        self._run_id = run_id
        self._log_event = log_event or (lambda name, request: None)
        self._is_client_started = False
        self._run_once = False
        if file_path is not None:
            if os.path.exists(file_path):
                os.remove(file_path)
            os.mkfifo(file_path)
            queue = FileTimerRequestQueue(tempfile.mkdtemp(prefix="tp_timer_"))
        else:
            if request_dir is None:
                raise TypeError("request_dir or file_path is required")
            queue = FileTimerRequestQueue(request_dir)
        super().__init__(queue, max_interval, daemon)
        self.signal = signal
        self.scope_signal_map = scope_signal_map or {}
        self._timers: dict[tuple[int, str], FileTimerRequest] = {}
        self._last_progress_time = int(time.time())
        self._request_count = 0

    def register_timers(self, timer_requests: list[TimerRequest]) -> None:
        for request in timer_requests:
            key = (int(request.worker_id), request.scope_id)
            if float(request.expiration_time) < 0:
                self._timers.pop(key, None)
            else:
                if not isinstance(request, FileTimerRequest):
                    request = FileTimerRequest(
                        worker_pid=key[0],
                        scope_id=key[1],
                        expiration_time=request.expiration_time,
                    )
                self._timers[key] = request
            self._request_count += 1

    def clear_timers(self, worker_ids: set[int]) -> None:
        for key in list(self._timers):
            if key[0] in worker_ids or not self.is_process_running(key[0]):
                self._timers.pop(key, None)

    def get_expired_timers(self, deadline: float) -> dict[int, list[TimerRequest]]:
        expired: dict[int, list[TimerRequest]] = {}
        for request in self._timers.values():
            if float(request.expiration_time) <= deadline:
                expired.setdefault(request.worker_pid, []).append(request)
        return expired

    def _get_scopes(self, timer_requests: list[TimerRequest]) -> list[str]:
        return [request.scope_id for request in timer_requests]

    def _reap_worker(self, worker_id: int, reap_signal: int | None = None) -> bool:
        requests = [
            request for request in self._timers.values() if request.worker_pid == worker_id
        ]
        sig = self.signal
        if reap_signal is not None:
            sig = reap_signal
        for request in requests:
            if request.signal > 0:
                sig = request.signal
                break
        if sig is None or int(sig) <= 0:
            return True
        try:
            os.kill(worker_id, sig)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        self._log_event("kill worker process", requests[0] if requests else None)
        return True

    def _process_waiting_timers(self) -> None:
        while True:
            requests = self._request_queue.get()
            if not requests:
                break
            self.register_timers(requests if isinstance(requests, list) else [requests])

    def _handle_timer(self, request: TimerRequest) -> bool:
        sig = self.scope_signal_map.get(request.scope, self.signal)
        if sig is not None:
            try:
                os.kill(os.getpid(), sig)
            except OSError:
                pass
        return True

    def _get_requests(self, fd: io.TextIOBase, timeout: float) -> list[FileTimerRequest]:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return []
        requests = []
        for line in fd:
            if line.strip():
                requests.append(FileTimerRequest.from_json(line))
        self._is_client_started = True
        return requests

    def _watchdog_loop(self) -> None:
        if self._file_path is None:
            super()._watchdog_loop()
            return
        try:
            with open(self._file_path) as fd:
                self._is_client_started = True
                while not self._stop_signaled:
                    run_once = self._run_once
                    try:
                        self._run_watchdog(fd)
                    except Exception:
                        if not self._stop_signaled:
                            raise
                    if run_once:
                        break
                    self._last_progress_time = int(time.time())
        except (OSError, ValueError):
            if not self._stop_signaled:
                raise

    def _run_watchdog(self, fd=None) -> None:
        if fd is None:
            super()._run_watchdog()
            return
        self.register_timers(self._get_requests(fd, self._max_interval))
        expired = self.get_expired_timers(time.time())
        log_debug_info_for_expired_timers(
            self._run_id,
            {
                pid: [request.to_json() for request in requests]
                for pid, requests in expired.items()
            },
        )
        reaped: set[int] = set()
        kill_process = False
        reap_signal = 0
        for worker_id, requests in expired.items():
            reaped.add(worker_id)
            requests.sort(key=lambda request: float(request.expiration_time))
            selected = None
            for request in requests:
                self._log_event("timer expired", request)
                if request.signal > 0:
                    selected = request
                    break
            if selected is None:
                continue
            if self._reap_worker(worker_id, selected.signal):
                self._log_event("kill worker process", selected)
                kill_process = True
                reap_signal = selected.signal
        if kill_process and reap_signal > 0:
            self._reap_worker(os.getpid(), reap_signal)
        self.clear_timers(reaped)

    def _get_requests(
        self, fd: io.TextIOBase, timeout: float
    ) -> list[FileTimerRequest]:
        start = time.time()
        requests: list[FileTimerRequest] = []
        while not self._stop_signaled or self._run_once:
            line = fd.readline()
            if not line:
                if self._run_once:
                    break
                time.sleep(min(timeout, 1.0))
            else:
                requests.append(FileTimerRequest.from_json(line))
            if time.time() - start > timeout:
                break
        return requests

    def run_once(self) -> None:
        self._run_once = True
        if self._watchdog_thread is not None:
            self._watchdog_thread.join()
            self._watchdog_thread = None

    @staticmethod
    def is_process_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def get_last_progress_time(self) -> int:
        return self._last_progress_time if self._is_client_started else int(time.time())

    def start(self) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._stop_signaled = False
        self._run_once = False
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=self._daemon
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._stop_signaled = True
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=self._max_interval * 2)
            self._watchdog_thread = None
        if self._file_path is not None and os.path.exists(self._file_path):
            os.remove(self._file_path)
