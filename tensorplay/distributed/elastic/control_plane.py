"""Worker-side status server exposing liveness over a unix socket.

Workers wrapped in :func:`worker_main` accept status probes through a unix
socket whose path arrives via the ``TP_WORKER_SERVER_SOCKET`` environment
variable, letting operators introspect a live job.
"""
import contextlib
import json
import os
import socket
import socketserver
import threading
from collections.abc import Generator

from .multiprocessing.errors import record

__all__ = ["worker_main"]

WORKER_SERVER_SOCKET_ENV = "TP_WORKER_SERVER_SOCKET"


class _StatusRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            payload = json.dumps({"status": "RUNNING", "pid": os.getpid()})
            self.request.sendall(payload.encode() + b"\n")
        except OSError:
            pass


class _WorkerServer:
    """Single-connection-at-a-time unix-socket status endpoint."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        self._server = socketserver.UnixStreamServer(socket_path, _StatusRequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        with contextlib.suppress(OSError):
            os.unlink(self.socket_path)


@contextlib.contextmanager
def _worker_server(socket_path: str) -> Generator[None, None, None]:
    server = _WorkerServer(socket_path)
    try:
        yield
    finally:
        server.shutdown()


@record
@contextlib.contextmanager
def worker_main() -> Generator[None, None, None]:
    """Wrap a worker entrypoint with failure recording and the status server.

    Usage::

        @worker_main()
        def main():
            ...
    """
    with contextlib.ExitStack() as stack:
        socket_path = os.environ.get(WORKER_SERVER_SOCKET_ENV)
        if socket_path is not None:
            stack.enter_context(_worker_server(socket_path))
        yield
