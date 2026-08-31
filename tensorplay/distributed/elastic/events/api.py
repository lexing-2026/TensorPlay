"""Structured event records emitted by the agent and workers."""
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union


__all__ = ["EventSource", "Event", "NodeState", "RdzvEvent"]


class EventSource(str, Enum):
    """Producer of an event: the agent process, a worker process, or the rendezvous layer."""

    AGENT = "AGENT"
    WORKER = "WORKER"
    RDZV = "RDZV"


class NodeState(str, Enum):
    """Lifecycle state carried by events."""

    INITIALIZED = "INITIALIZED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"


EventMetadataValue = Union[str, int, float, bool, None]


@dataclass
class Event:
    """A single agent or worker lifecycle event.

    ``event_type`` is a free-form string (typically a :class:`NodeState`
    value) and ``metadata`` holds structured context such as rank ids.
    """

    source: EventSource
    event_type: str
    metadata: dict[str, EventMetadataValue] = field(default_factory=dict)
    created: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class RdzvEvent:
    """A rendezvous progress event, rendered as JSON by the logging handler."""

    run_id: str
    message: str
    node_state: NodeState
    hostname: str
    pid: int
    master_endpoint: str
    rank: int | None
    local_rank: int | None
    rendezvous_id: str
    created: int = field(default_factory=lambda: int(time.time() * 1000) // 1000)

    def __str__(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "message": self.message,
                "node_state": self.node_state.value,
                "hostname": self.hostname,
                "pid": self.pid,
                "master_endpoint": self.master_endpoint,
                "rank": self.rank,
                "local_rank": self.local_rank,
                "rendezvous_id": self.rendezvous_id,
                "created": self.created,
            }
        )
