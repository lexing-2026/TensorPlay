"""Interactive debug server for stuck distributed jobs.

``start_debug_server`` opens a JSON-line socket endpoint backed by
pluggable handlers; clients connect and ask a named handler for a report.
Built-in handlers expose the handler index, live thread stacks, and store
state inspection. Out-of-tree packages can plug in additional handlers via
:func:`register_handler`.
"""
import json
import socketserver
import threading

from .handlers import DebugHandler, get_handler, list_handlers, register_handler
from .index import IndexHandler
from .stacks import StacksHandler
from .store import StoreDumpHandler

__all__ = [
    "DebugHandler",
    "IndexHandler",
    "StacksHandler",
    "StoreDumpHandler",
    "start_debug_server",
    "stop_debug_server",
    "default_handlers",
    "register_handler",
    "list_handlers",
]

_server_lock = threading.Lock()
_server = None
_server_thread = None


def default_handlers() -> list[DebugHandler]:
    """Fresh instances of the built-in handler set."""
    return [
        IndexHandler(),
        StacksHandler(),
        StoreDumpHandler(),
    ]


class _RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        f = self.request.makefile("r")
        while True:
            line = f.readline()
            if not line:
                return
            try:
                request = json.loads(line)
                handler = get_handler(request.get("handler", ""))
                if handler is None:
                    response = {
                        "error": f"unknown handler {request.get('handler')!r}",
                        "handlers": list_handlers(),
                    }
                else:
                    response = handler.handle_request(request.get("args", {}))
            except Exception as e:
                response = {"error": repr(e)}
            try:
                self.request.sendall(json.dumps(response, default=str).encode() + b"\n")
            except OSError:
                return


def start_debug_server(
    host: str = "localhost",
    port: int = 0,
    handlers: list[DebugHandler] | None = None,
) -> int:
    """Start the debug server; returns the bound port (idempotent)."""
    global _server, _server_thread
    with _server_lock:
        if _server is not None:
            return _server.server_address[1]
        for handler in handlers if handlers is not None else default_handlers():
            register_handler(handler)
        _server = socketserver.ThreadingTCPServer((host, port), _RequestHandler)
        _server.daemon_threads = True
        _server_thread = threading.Thread(
            target=_server.serve_forever, daemon=True, name="tp_debug_server"
        )
        _server_thread.start()
        return _server.server_address[1]


def stop_debug_server() -> None:
    """Stop the debug server if running."""
    global _server, _server_thread
    with _server_lock:
        if _server is None:
            return
        _server.shutdown()
        _server.server_close()
        _server = None
        _server_thread = None
