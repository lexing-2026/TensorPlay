"""Pluggable destinations for structured events.

Handlers receive an :class:`~tensorplay.distributed.elastic.events.Event` and
forward it to a backend. ``null`` drops events; ``console`` prints them via
the standard logging module. Out-of-tree packages may register additional
handler names.
"""
import json
import logging
from collections.abc import Callable

from .api import Event

_event_handlers: dict[str, Callable] = {}


class NullEventHandler:
    """Discard every event."""

    def record(self, event: Event) -> None:
        return


class LoggingEventHandler:
    """Emit events as JSON through the ``tp_elastic_events`` logger."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("tp_elastic_events")

    def record(self, event: Event) -> None:
        payload = {
            "source": event.source.value,
            "event_type": event.event_type,
            "metadata": event.metadata,
            "created": event.created,
        }
        self.logger.info(json.dumps(payload, default=str))


def get_logging_handler(destination: str = "null") -> Callable:
    """Return the handler registered under ``destination`` (default: null)."""
    if destination not in _event_handlers:
        if destination == "console":
            _event_handlers[destination] = LoggingEventHandler()
        else:
            _event_handlers[destination] = NullEventHandler()
    return _event_handlers[destination]


def register_event_handler(destination: str, handler) -> None:
    """Register ``handler`` for ``destination``; later registrations win."""
    _event_handlers[destination] = handler
