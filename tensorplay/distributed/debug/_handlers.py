from __future__ import annotations

import json
import pathlib
import tempfile
import time

_HANDLERS: dict[str, object] = {}


class _Request:
    def __init__(self, query: dict[str, list[str]] | None = None) -> None:
        self.query = query or {}

    def get_param(self, name: str, default: str | None = None) -> str | None:
        values = self.query.get(name)
        return default if not values else values[0]


class _Response:
    def __init__(self) -> None:
        self.status_code = 200
        self.content = b""
        self.content_type = "application/json"

    def set_content(self, content: bytes, content_type: str = "application/json") -> None:
        self.content = content
        self.content_type = content_type

    def set_status(self, status: int) -> None:
        self.status_code = int(status)


def register_handler(name: str, handler) -> None:
    _HANDLERS[name] = handler


def get_handler(name: str):
    return _HANDLERS.get(name)


def handle_request(name: str, query: dict[str, list[str]] | None = None) -> _Response:
    response = _Response()
    handler = get_handler(name)
    if handler is None:
        response.set_status(404)
        response.set_content(json.dumps({"error": f"unknown handler: {name}"}).encode())
        return response
    try:
        handler(_Request(query), response)
    except Exception as exc:
        response.set_status(500)
        response.set_content(json.dumps({"error": repr(exc)}).encode())
    return response


def _torch_profile(req: _Request, resp: _Response) -> None:
    duration = float(req.get_param("duration", "1.0") or 1.0)
    if duration < 0:
        raise ValueError("duration must be non-negative")
    started = time.time()
    time.sleep(duration)
    finished = time.time()
    payload = {
        "traceEvents": [
            {"name": "debug_profile", "ph": "X", "ts": started * 1e6, "dur": (finished - started) * 1e6, "pid": 0, "tid": 0}
        ],
        "displayTimeUnit": "ms",
    }
    with tempfile.NamedTemporaryFile(prefix="tp_debug_", suffix=".json", delete=False) as stream:
        stream.write(json.dumps(payload).encode())
        path = pathlib.Path(stream.name)
    try:
        resp.set_content(path.read_bytes(), "application/json")
    finally:
        path.unlink(missing_ok=True)
    resp.set_status(200)


register_handler("debug_profile", _torch_profile)

__all__ = ["_Request", "_Response", "_torch_profile", "get_handler", "handle_request", "register_handler"]
