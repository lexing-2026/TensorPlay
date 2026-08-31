from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from ._store import get_world_size, tcpstore_client

logger = logging.getLogger(__name__)
_DEFAULT_FETCH_TIMEOUT = 60.0


@dataclass(slots=True)
class Response:
    status_code: int
    text: str

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def json(self):
        return json.loads(self.text)


@dataclass(slots=True)
class NavLink:
    path: str
    label: str


@dataclass(slots=True)
class Route:
    path: str
    handler: Callable[["HTTPRequestHandler"], bytes]


class DebugHandler(ABC):
    fetch_timeout: float = _DEFAULT_FETCH_TIMEOUT

    @abstractmethod
    def routes(self) -> list[Route]:
        raise NotImplementedError

    @abstractmethod
    def nav_links(self) -> list[NavLink]:
        raise NotImplementedError

    def templates(self) -> dict[str, str]:
        return {}

    def dump(self) -> str | None:
        return None

    def dump_filename(self) -> str:
        return type(self).__name__.lower()


def _post(url: str, timeout: float) -> Response:
    request = Request(url, data=b"", method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return Response(int(response.status), body)
    except TimeoutError as exc:
        return Response(408, f"Timeout: {exc}")
    except OSError as exc:
        return Response(503, f"ConnectionError: {exc}")
    except Exception as exc:
        return Response(502, f"{type(exc).__name__}: {exc}")


def fetch_thread_pool(urls: list[str], timeout: float) -> list[Response]:
    with ThreadPoolExecutor(max_workers=max(1, min(20, len(urls)))) as pool:
        return list(pool.map(lambda url: _post(url, timeout), urls))


def fetch_aiohttp(urls: list[str], timeout: float) -> list[Response]:
    import aiohttp

    async def one(session, url: str) -> Response:
        try:
            async with session.post(url) as response:
                body = await response.text()
                return Response(response.status, body)
        except asyncio.TimeoutError as exc:
            return Response(408, f"Timeout: {exc}")
        except aiohttp.ClientError as exc:
            return Response(503, f"ConnectionError: {exc}")
        except Exception as exc:
            return Response(502, f"{type(exc).__name__}: {exc}")

    async def gather() -> list[Response]:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            return list(await asyncio.gather(*(one(session, url) for url in urls)))

    return asyncio.run(gather())


def _store_values(store, keys: list[str]) -> list[bytes]:
    values: list[bytes] = []
    for key in keys:
        value = store.get(key, timeout=5.0)
        values.append(value if isinstance(value, bytes) else str(value).encode())
    return values


def fetch_all(
    endpoint: str, args: str = "", *, timeout: float = _DEFAULT_FETCH_TIMEOUT
) -> tuple[list[str], list[Response]]:
    try:
        store = tcpstore_client()
        addresses = _store_values(store, [f"rank{rank}" for rank in range(get_world_size())])
    except Exception as exc:
        return [], [Response(503, f"debug workers unavailable: {exc}")]
    suffix = f"?{args}" if args else ""
    urls = [f"{address.decode(errors='replace')}/handler/{endpoint}{suffix}" for address in addresses]
    try:
        responses = fetch_aiohttp(urls, timeout)
    except ImportError:
        responses = fetch_thread_pool(urls, timeout)
    return urls, responses


def format_json(blob: str) -> str:
    try:
        return json.dumps(json.loads(blob), indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return blob


def format_fetch_summary(addrs: list[str], resps: list[Response]) -> str | None:
    failed = [(rank, response) for rank, response in enumerate(resps) if response.status_code != 200]
    if not failed:
        return None
    lines = [f"PARTIAL DATA: {len(resps) - len(failed)}/{len(addrs)} workers responded"]
    lines.extend(f"  Rank {rank}: {response.text}" for rank, response in failed)
    return "\n".join(lines)


BASE_TEMPLATE = """<!doctype html>
<html><head><meta charset='utf-8'><title>{{ title }}</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem}nav a{margin-right:1rem}pre{white-space:pre-wrap}</style>
</head><body><nav>{{ nav_links }}</nav><main>{{ content }}</main></body></html>"""


class PeriodicDumper:
    def __init__(self, handlers: list[DebugHandler], output_dir: str, interval_seconds: float = 60.0) -> None:
        self._handlers = handlers
        self._output_dir = output_dir
        self._interval_seconds = max(0.01, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        os.makedirs(self._output_dir, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tp_debug_dumper")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds + 1.0))
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            for handler in self._handlers:
                try:
                    content = handler.dump()
                    if content is None:
                        continue
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(self._output_dir, f"{handler.dump_filename()}_{stamp}.txt")
                    with open(path, "w", encoding="utf-8") as stream:
                        stream.write(content)
                except Exception:
                    logger.exception("failed to write debug dump for %s", type(handler).__name__)
            self._stop.wait(self._interval_seconds)


class _IPv6HTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    request_queue_size = 1024


class HTTPRequestHandler(BaseHTTPRequestHandler):
    frontend: "FrontendServer"

    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s %s", self.client_address[0], fmt % args)

    def do_GET(self) -> None:
        self.frontend._handle_request(self)

    def do_POST(self) -> None:
        self.frontend._handle_request(self)

    def get_path(self) -> str:
        return urlparse(self.path).path

    def get_raw_query(self) -> str:
        return urlparse(self.path).query

    def get_query(self) -> dict[str, list[str]]:
        return parse_qs(self.get_raw_query())

    def get_query_arg(self, name: str, default=None, type: type = str):
        values = self.get_query().get(name)
        return default if not values else type(values[0])


class FrontendServer:
    def __init__(self, port: int, handlers: list[DebugHandler] | None = None) -> None:
        if handlers is None:
            from ._debug_handlers import default_handlers

            handlers = default_handlers()
        self._handlers = list(handlers)
        self._routes = {route.path: route.handler for handler in self._handlers for route in handler.routes()}
        links = [link for handler in self._handlers for link in handler.nav_links()]
        self._nav = " ".join(
            f"<a href='{html.escape(link.path, quote=True)}'>{html.escape(link.label)}</a>" for link in links
        )
        self._server = self._make_server(port)
        request_class = type("TPHTTPRequestHandler", (HTTPRequestHandler,), {"frontend": self})
        self._server.RequestHandlerClass = request_class
        self._thread = threading.Thread(target=self._serve, daemon=True, name="tp_debug_frontend")
        self._thread.start()

    @staticmethod
    def _make_server(port: int):
        try:
            return _IPv6HTTPServer(("::", port), HTTPRequestHandler)
        except OSError:
            return ThreadingHTTPServer(("", port), HTTPRequestHandler)

    @property
    def port(self) -> int:
        return int(self._server.server_port)

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception:
            logger.exception("debug frontend stopped unexpectedly")

    def join(self) -> None:
        self._thread.join()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def _handle_request(self, request: HTTPRequestHandler) -> None:
        handler = self._routes.get(request.get_path())
        if handler is None:
            request.send_error(404, "debug route not found")
            return
        try:
            body = handler(request)
        except BaseException as exc:
            logger.exception("debug route failed: %s", request.get_path())
            request.send_error(500, str(exc))
            return
        if isinstance(body, str):
            body = body.encode()
        request.send_response(200)
        request.send_header("Content-Type", "text/html; charset=utf-8")
        request.send_header("Content-Length", str(len(body)))
        request.end_headers()
        request.wfile.write(body)

    def render_template(self, template: str, **values: object) -> bytes:
        title = str(values.pop("title", template))
        content = values.pop("content", None)
        if content is None:
            content = "\n".join(f"<h2>{html.escape(str(key))}</h2><pre>{html.escape(str(value))}</pre>" for key, value in values.items())
        rendered = BASE_TEMPLATE.replace("{{ title }}", html.escape(title))
        rendered = rendered.replace("{{ nav_links }}", self._nav).replace("{{ content }}", str(content))
        return rendered.encode("utf-8")


def main(
    port: int,
    dump_dir: str | None,
    dump_interval: float,
    handlers: list[DebugHandler],
    enabled_dumps: set[str],
    fetch_timeout: float = _DEFAULT_FETCH_TIMEOUT,
) -> None:
    for handler in handlers:
        handler.fetch_timeout = fetch_timeout
    server = FrontendServer(port, handlers)
    dumper = None
    if dump_dir is not None:
        selected = [handler for handler in handlers if not enabled_dumps or handler.dump_filename() in enabled_dumps]
        dumper = PeriodicDumper(selected, dump_dir, dump_interval)
        dumper.start()
    try:
        server.join()
    finally:
        if dumper is not None:
            dumper.stop()


__all__ = [
    "DebugHandler",
    "FrontendServer",
    "HTTPRequestHandler",
    "NavLink",
    "PeriodicDumper",
    "Response",
    "Route",
    "fetch_aiohttp",
    "fetch_all",
    "fetch_thread_pool",
    "format_fetch_summary",
    "format_json",
    "main",
]
