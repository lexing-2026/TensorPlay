from __future__ import annotations

import json
import traceback
from collections.abc import Iterable

from ._frontend import (
    DebugHandler,
    FrontendServer,
    NavLink,
    Request,
    Response,
    Route,
    fetch_all,
    format_fetch_summary,
    format_json,
)
from ._store import tcpstore_client
from .stacks import StacksHandler as LocalStacksHandler

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def _html_text(title: str, text: str) -> bytes:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<h1>{title}</h1><pre>{escaped}</pre>".encode()


def _render_responses(server: FrontendServer, title: str, addrs: list[str], responses: list[Response], json_mode: bool = False) -> bytes:
    sections: list[str] = []
    summary = format_fetch_summary(addrs, responses)
    if summary:
        sections.append(summary)
    for index, response in enumerate(responses):
        body = format_json(response.text) if json_mode and response.status_code == 200 else response.text
        sections.append(f"Rank {index}: {addrs[index] if index < len(addrs) else 'local'}\n{body}")
    return server.render_template(title.lower().replace(" ", "_") + ".html", title=title, content="\n\n".join(sections))


class IndexHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/", "Home")]

    def _handle(self, req) -> bytes:
        return req.frontend.render_template("index.html", title="Debug", content="<h1>Debug handlers</h1>")


class StacksHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/stacks", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/stacks", "Stacks")]

    def _local_dump(self) -> str:
        payload = LocalStacksHandler().handle_request({})
        return json.dumps(payload, indent=2, sort_keys=True)

    def _handle(self, req) -> bytes:
        addrs, responses = fetch_all("dump_traceback", timeout=self.fetch_timeout)
        if not addrs:
            return _html_text("Stacks", self._local_dump())
        return _render_responses(req.frontend, "Stacks", addrs, responses)

    def dump(self) -> str | None:
        return self._local_dump()

    def dump_filename(self) -> str:
        return "stacks"


class PySpyHandler(StacksHandler):
    def routes(self) -> list[Route]:
        return [Route("/pyspy_dump", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/pyspy_dump", "Python stacks")]

    def _handle(self, req) -> bytes:
        args = "nonblocking=1"
        if req.get_raw_query():
            args = f"{args}&{req.get_raw_query()}"
        addrs, responses = fetch_all("pyspy_dump", args, timeout=self.fetch_timeout)
        return _render_responses(req.frontend, "Python stacks", addrs, responses)

    def dump_filename(self) -> str:
        return "pyspy_dump"


class FlightRecorderHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [
            Route("/fr_trace", self._handle_text),
            Route("/fr_trace_json", self._handle_json),
            Route("/fr_dump_file", self._handle_text),
        ]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/fr_trace", "Flight recorder"), NavLink("/fr_trace_json", "JSON")]

    def _fetch(self, req):
        backend = req.get_query_arg("backend", "default")
        return fetch_all("fr_trace_json", f"backend={backend}", timeout=self.fetch_timeout)

    def _handle_text(self, req) -> bytes:
        addrs, responses = self._fetch(req)
        return _render_responses(req.frontend, "Flight recorder", addrs, responses)

    def _handle_json(self, req) -> bytes:
        addrs, responses = self._fetch(req)
        return _render_responses(req.frontend, "Flight recorder JSON", addrs, responses, json_mode=True)

    def dump(self) -> str | None:
        addrs, responses = fetch_all("fr_trace_json", timeout=self.fetch_timeout)
        if not addrs:
            return None
        parts = [format_fetch_summary(addrs, responses) or ""]
        parts.extend(response.text for response in responses)
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "fr_trace"


class ProfilerHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/profile", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/profile", "Profile")]

    def _handle(self, req) -> bytes:
        duration = req.get_query_arg("duration", 1.0, float)
        addrs, responses = fetch_all("debug_profile", f"duration={duration}", timeout=self.fetch_timeout + duration)
        return _render_responses(req.frontend, "Profile", addrs, responses, json_mode=True)


class WaitCountersHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/wait_counters", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/wait_counters", "Wait counters")]

    def _handle(self, req) -> bytes:
        addrs, responses = fetch_all("wait_counter_values", timeout=self.fetch_timeout)
        return _render_responses(req.frontend, "Wait counters", addrs, responses, json_mode=True)

    def dump(self) -> str | None:
        addrs, responses = fetch_all("wait_counter_values", timeout=self.fetch_timeout)
        if not addrs:
            return None
        return "\n\n".join(response.text for response in responses)

    def dump_filename(self) -> str:
        return "wait_counters"


class TCPStoreHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/tcpstore", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/tcpstore", "Store")]

    def _handle(self, req) -> bytes:
        try:
            store = tcpstore_client(prefix="")
            keys = getattr(store, "list_keys", lambda: [])()
            content = "\n".join(f"{key}: {store.get(key, timeout=1)!r}" for key in sorted(keys))
        except Exception as exc:
            content = repr(exc)
        return _html_text("Store", content)

    def dump(self) -> str | None:
        return self._handle(type("Request", (), {"frontend": None})()).decode(errors="replace")

    def dump_filename(self) -> str:
        return "tcpstore"


class TorchCommsFlightRecorderHandler(FlightRecorderHandler):
    def routes(self) -> list[Route]:
        return [Route("/torchcomms_fr_trace", self._handle_text), Route("/torchcomms_fr_trace_json", self._handle_json)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/torchcomms_fr_trace", "Comms recorder")]

    def dump_filename(self) -> str:
        return "torchcomms_fr_trace"


class TorchCommsHealthCheckHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/torchcomms_health_check", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/torchcomms_health_check", "Comms health")]

    def _handle(self, req) -> bytes:
        addrs, responses = fetch_all("torchcomms_health_check", timeout=self.fetch_timeout)
        return _render_responses(req.frontend, "Comms health", addrs, responses, json_mode=True)

    def dump(self) -> str | None:
        addrs, responses = fetch_all("torchcomms_health_check", timeout=self.fetch_timeout)
        return format_fetch_summary(addrs, responses)

    def dump_filename(self) -> str:
        return "torchcomms_health_check"


def default_handlers() -> list[DebugHandler]:
    return [
        IndexHandler(),
        StacksHandler(),
        PySpyHandler(),
        FlightRecorderHandler(),
        ProfilerHandler(),
        WaitCountersHandler(),
        TCPStoreHandler(),
        TorchCommsFlightRecorderHandler(),
        TorchCommsHealthCheckHandler(),
    ]


__all__ = [
    "FlightRecorderHandler",
    "IndexHandler",
    "ProfilerHandler",
    "PySpyHandler",
    "StacksHandler",
    "TCPStoreHandler",
    "TorchCommsFlightRecorderHandler",
    "TorchCommsHealthCheckHandler",
    "WaitCountersHandler",
    "default_handlers",
]
