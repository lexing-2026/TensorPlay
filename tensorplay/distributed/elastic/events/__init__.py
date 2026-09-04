"""Elastic event recording entry points."""
import logging
import os
import socket
from typing import Any

from .api import Event, EventSource, NodeState, RdzvEvent
from .handlers import get_logging_handler

__all__ = [
    "Event",
    "EventSource",
    "NodeState",
    "RdzvEvent",
    "record",
    "record_rdzv_event",
    "construct_and_record_rdzv_event",
]

_events_loggers: dict[str, logging.Logger] = {}


def _get_or_create_logger(destination: str = "null") -> logging.Logger:
    """Return a non-propagating logger with the requested event destination."""
    logger = _events_loggers.get(destination)
    if logger is not None:
        return logger
    logger = logging.getLogger(f"tp_elastic_events-{destination}")
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logger.propagate = False
    handler = get_logging_handler(destination)
    if isinstance(handler, logging.Handler):
        logger.addHandler(handler)
    else:
        class _EventHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                handler.record(record.getMessage())

        logger.addHandler(_EventHandler())
    _events_loggers[destination] = logger
    return logger


def record(event: Event, destination: str = "null") -> None:
    """Dispatch ``event`` to the handler registered under ``destination``."""
    handler = get_logging_handler(destination)
    handler.record(event)


def record_rdzv_event(event: RdzvEvent) -> None:
    """Log a rendezvous event through the ``tp_elastic_rdzv`` logger."""
    logger = get_logging_handler("rdzv_logger")
    if hasattr(logger, "record"):
        logger.record(event)
    else:
        logging.getLogger("tp_elastic_rdzv").info(str(event))


def construct_and_record_rdzv_event(
    node_state: NodeState, run_id: str, message: str, **kwargs: Any
) -> None:
    """Build an :class:`RdzvEvent` from current host context and record it."""
    event = RdzvEvent(
        run_id=run_id,
        message=message,
        node_state=node_state,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        master_endpoint=kwargs.get("master_endpoint", ""),
        rank=kwargs.get("rank"),
        local_rank=kwargs.get("local_rank"),
        rendezvous_id=kwargs.get("rendezvous_id", run_id),
    )
    record_rdzv_event(event)
