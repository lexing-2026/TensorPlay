from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

__all__ = ["HealthCheckServer", "create_healthcheck_server"]


class HealthCheckServer:
    def __init__(self, alive_callback: Callable[[], int], port: int, timeout: int) -> None:
        self._alive_callback = alive_callback
        self._port = int(port)
        self._timeout = int(timeout)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def alive_callback(self) -> Callable[[], int]:
        return self._alive_callback

    @property
    def port(self) -> int:
        return self._server.server_port if self._server is not None else self._port

    def start(self) -> None:
        with self._lock:
            if self._server is not None:
                return
            callback = self._alive_callback
            timeout = self._timeout

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self) -> None:
                    try:
                        last_progress = int(callback())
                        age = max(0, int(time.time()) - last_progress)
                        healthy = age <= timeout
                        body = json.dumps({"healthy": healthy, "last_progress": last_progress}).encode()
                        self.send_response(200 if healthy else 503)
                    except Exception as exc:
                        body = json.dumps({"healthy": False, "error": repr(exc)}).encode()
                        self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, fmt: str, *args) -> None:
                    log.debug("health endpoint: " + fmt, *args)

            self._server = ThreadingHTTPServer(("", self._port), Handler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="tp_health_check")
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            server, thread = self._server, self._thread
            self._server = None
            self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)


def create_healthcheck_server(alive_callback: Callable[[], int], port: int, timeout: int) -> HealthCheckServer:
    return HealthCheckServer(alive_callback, port, timeout)
