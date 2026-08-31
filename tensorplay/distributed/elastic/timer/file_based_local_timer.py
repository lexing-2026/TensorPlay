"""File-based timer queue shared between workers and a local timer server.

Clients drop JSON request files into ``request_dir``; the server scans the
directory, tracks deadlines per scope, and re-invokes the handler for
expired scopes until the handler reports completion.
"""
import json
import os
import signal
import tempfile
from datetime import datetime

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer


class FileTimerRequest(TimerRequest):
    """A :class:`TimerRequest` that serializes to/from a JSON file."""

    @staticmethod
    def from_json(json_data: str) -> "FileTimerRequest":
        data = json.loads(json_data)
        return FileTimerRequest(
            scope=data["scope"],
            expire_time=datetime.fromtimestamp(data["expire_time"]),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "scope": self.scope,
                "expire_time": self.expire_time.timestamp(),
            }
        )


class FileTimerRequestQueue(RequestQueue):
    """Directory-backed :class:`RequestQueue` (one JSON file per request)."""

    def __init__(self, request_dir: str) -> None:
        self.request_dir = request_dir
        os.makedirs(self.request_dir, exist_ok=True)

    def size(self) -> int:
        return len(self._list_files())

    def get(self) -> TimerRequest | None:
        for path in self._list_files():
            try:
                with open(path) as f:
                    request = FileTimerRequest.from_json(f.read())
                os.unlink(path)
                return request
            except (OSError, ValueError, KeyError):
                continue
        return None

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

    def __init__(self, request_queue: FileTimerRequestQueue) -> None:
        self._request_queue = request_queue

    def start_timer(self, request: TimerRequest) -> None:
        self._request_queue.put(request)

    def acquire_scope(self, scope: str, expiration_time: datetime) -> None:
        self.start_timer(FileTimerRequest(scope=scope, expire_time=expiration_time))

    def cancel_scope(self, scope: str) -> None:
        # Cancellation is implicit: the server forgets scopes whose owning
        # pid is gone; scope files are consumed on first read.
        return


class FileTimerServer(TimerServer):
    """Timer server watching a shared request directory.

    ``signal`` is delivered to the owning process when a deadline lapses
    (default SIGTERM); pass ``None`` to leave reaction to the handler.
    """

    def __init__(
        self,
        request_dir: str,
        signal: signal.Signals = signal.SIGTERM,
        max_interval: float = 60,
        daemon: bool = True,
        scope_signal_map: dict[str, signal.Signals] | None = None,
    ) -> None:
        super().__init__(FileTimerRequestQueue(request_dir), max_interval, daemon)
        self.signal = signal
        self.scope_signal_map = scope_signal_map or {}

    def _process_waiting_timers(self) -> None:
        while True:
            request = self._request_queue.get()
            if request is None:
                break
            self.register_timer(request)

    def _handle_timer(self, request: TimerRequest) -> bool:
        sig = self.scope_signal_map.get(request.scope, self.signal)
        if sig is not None:
            try:
                os.kill(os.getpid(), sig)
            except OSError:
                pass
        return True
