"""Worker-side liveness control over a Unix-domain socket."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import socketserver
import stat
import threading
import time
from collections.abc import Generator

from .multiprocessing.errors import record

__all__ = ["worker_main"]

WORKER_SERVER_SOCKET_ENV = "TP_WORKER_SERVER_SOCKET"


class _ControlState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = "STARTING"
        self._updated_at = time.time()
        self._error: str | None = None

    def update(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._status = str(status).upper()
            self._updated_at = time.time()
            self._error = error

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            result: dict[str, object] = {
                "status": self._status,
                "pid": os.getpid(),
                "updated_at": self._updated_at,
            }
            if self._error is not None:
                result["error"] = self._error
            return result


class _StatusRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        state = getattr(server, "control_state", None)
        if state is None:
            self._send({"status": "ERROR", "error": "control state is unavailable"})
            return
        self.connection.settimeout(1.0)
        try:
            request = self.rfile.readline(4096).decode("utf-8", errors="replace").strip()
        except OSError:
            request = ""
        if request:
            try:
                payload = json.loads(request)
            except json.JSONDecodeError:
                self._send({"status": "ERROR", "error": "request must be JSON"})
                return
            if not isinstance(payload, dict):
                self._send({"status": "ERROR", "error": "request must be an object"})
                return
            command = str(payload.get("command", "status")).lower()
            if command not in {"status", "ping"}:
                self._send({"status": "ERROR", "error": "unsupported control command"})
                return
        self._send(state.snapshot())

    def _send(self, payload: dict[str, object]) -> None:
        try:
            self.wfile.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
            self.wfile.flush()
        except OSError:
            return


class _WorkerServer:
    """Threaded status endpoint with an idempotent shutdown operation."""

    def __init__(self, socket_path: str) -> None:
        if not isinstance(socket_path, str) or not socket_path:
            raise ValueError("socket path must be a non-empty string")
        if len(os.fsencode(socket_path)) >= 108:
            raise ValueError("socket path is too long")
        self.socket_path = socket_path
        self.control_state = _ControlState()
        parent = os.path.dirname(socket_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.lexists(socket_path):
            if not stat.S_ISSOCK(os.stat(socket_path).st_mode):
                raise FileExistsError(f"socket path is not a socket: {socket_path}")
            os.unlink(socket_path)
        server = socketserver.ThreadingUnixStreamServer(
            socket_path, _StatusRequestHandler
        )
        server.daemon_threads = True
        server.control_state = self.control_state
        self._server = server
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="tensorplay-worker-control",
            daemon=True,
        )
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread.start()

    def set_status(self, status: str, error: str | None = None) -> None:
        self.control_state.update(status, error)

    def shutdown(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.control_state.update("STOPPING")
            self._server.shutdown()
            self._server.server_close()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        with contextlib.suppress(OSError):
            os.unlink(self.socket_path)

@contextlib.contextmanager
def _worker_server(socket_path: str) -> Generator[_WorkerServer, None, None]:
    server = _WorkerServer(socket_path)
    server.set_status("RUNNING")
    try:
        yield server
    except BaseException as error:
        server.set_status("FAILED", f"{type(error).__name__}: {error}")
        raise
    else:
        server.set_status("SUCCEEDED")
    finally:
        server.shutdown()


@record
@contextlib.contextmanager
def worker_main() -> Generator[None, None, None]:
    """Wrap a worker entrypoint with failure recording and liveness status."""
    with contextlib.ExitStack() as stack:
        socket_path = os.environ.get(WORKER_SERVER_SOCKET_ENV)
        if socket_path is not None:
            stack.enter_context(_worker_server(socket_path))
        yield
