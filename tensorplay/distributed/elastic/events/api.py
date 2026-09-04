"""Structured event records emitted by the agent and workers."""

from __future__ import annotations
import json
import time
from dataclasses import asdict, dataclass, field
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


@dataclass(init=False)
class Event:
    """A single agent or worker lifecycle event.

    ``event_type`` is a free-form string (typically a :class:`NodeState`
    value) and ``metadata`` holds structured context such as rank ids.
    """

    name: str
    source: EventSource
    timestamp: int
    metadata: dict[str, EventMetadataValue]

    def __init__(
        self,
        *args: Any,
        name: str | None = None,
        source: EventSource | str | None = None,
        timestamp: int | None = None,
        metadata: dict[str, EventMetadataValue] | None = None,
        event_type: str | None = None,
        created: int | None = None,
    ) -> None:
        if args:
            if isinstance(args[0], EventSource):
                source = args[0]
                if len(args) > 1:
                    event_type = args[1]
                if len(args) > 2:
                    metadata = args[2]
                if len(args) > 3:
                    created = args[3]
            else:
                name = args[0]
                if len(args) > 1:
                    source = args[1]
                if len(args) > 2:
                    timestamp = args[2]
                if len(args) > 3:
                    metadata = args[3]
        if source is None:
            raise TypeError("source is required")
        if isinstance(source, str):
            source = EventSource[source] if source in EventSource.__members__ else EventSource(source)
        self.name = name if name is not None else event_type or ""
        self.source = source
        self.timestamp = (
            timestamp if timestamp is not None else created
            if created is not None
            else int(time.time() * 1000)
        )
        self.metadata = dict(metadata or {})

    @property
    def event_type(self) -> str:
        return self.name

    @event_type.setter
    def event_type(self, value: str) -> None:
        self.name = value

    @property
    def created(self) -> int:
        return self.timestamp

    @created.setter
    def created(self, value: int) -> None:
        self.timestamp = value

    def __str__(self) -> str:
        return self.serialize()

    @staticmethod
    def deserialize(data: str | "Event") -> "Event":
        if isinstance(data, Event):
            return data
        payload = json.loads(data)
        if "name" not in payload:
            payload["name"] = payload.pop("event_type", "")
        if "timestamp" not in payload:
            payload["timestamp"] = payload.pop("created", 0)
        source = payload.get("source")
        if isinstance(source, str):
            payload["source"] = (
                EventSource[source]
                if source in EventSource.__members__
                else EventSource(source)
            )
        return Event(**payload)

    def serialize(self) -> str:
        return json.dumps(asdict(self))


@dataclass(init=False)
class RdzvEvent:
    """A rendezvous progress event, rendered as JSON by the logging handler."""

    name: str
    run_id: str
    message: str
    hostname: str
    pid: int
    node_state: NodeState
    master_endpoint: str
    rank: int | None
    local_id: int | None
    error_trace: str

    def __init__(
        self,
        *args: Any,
        name: str | None = None,
        run_id: str = "",
        message: str = "",
        hostname: str = "",
        pid: int = 0,
        node_state: NodeState | str = NodeState.RUNNING,
        master_endpoint: str = "",
        rank: int | None = None,
        local_id: int | None = None,
        error_trace: str = "",
        rendezvous_id: str | None = None,
        local_rank: int | None = None,
        created: int | None = None,
    ) -> None:
        if args:
            names = [
                "run_id", "message", "node_state", "hostname", "pid",
                "master_endpoint", "rank", "local_rank", "rendezvous_id", "created",
            ]
            values = dict(zip(names, args))
            run_id = values.get("run_id", run_id)
            message = values.get("message", message)
            node_state = values.get("node_state", node_state)
            hostname = values.get("hostname", hostname)
            pid = values.get("pid", pid)
            master_endpoint = values.get("master_endpoint", master_endpoint)
            rank = values.get("rank", rank)
            local_rank = values.get("local_rank", local_rank)
            rendezvous_id = values.get("rendezvous_id", rendezvous_id)
            created = values.get("created", created)
        if isinstance(node_state, str):
            node_state = (
                NodeState[node_state]
                if node_state in NodeState.__members__
                else NodeState(node_state)
            )
        self.name = name if name is not None else rendezvous_id or ""
        self.run_id = run_id
        self.message = message
        self.hostname = hostname
        self.pid = int(pid)
        self.node_state = node_state
        self.master_endpoint = master_endpoint
        self.rank = rank
        self.local_id = local_id if local_id is not None else local_rank
        self.error_trace = error_trace
        self._created = (
            created if created is not None else int(time.time() * 1000) // 1000
        )

    @property
    def rendezvous_id(self) -> str:
        return self.name

    @rendezvous_id.setter
    def rendezvous_id(self, value: str) -> None:
        self.name = value

    @property
    def local_rank(self) -> int | None:
        return self.local_id

    @local_rank.setter
    def local_rank(self, value: int | None) -> None:
        self.local_id = value

    @property
    def created(self) -> int:
        return self._created

    @created.setter
    def created(self, value: int) -> None:
        self._created = value

    def __str__(self) -> str:
        return self.serialize()

    @staticmethod
    def deserialize(data: str | "RdzvEvent") -> "RdzvEvent":
        if isinstance(data, RdzvEvent):
            return data
        payload = json.loads(data)
        if "name" not in payload:
            payload["name"] = payload.pop("rendezvous_id", "")
        if "local_id" not in payload:
            payload["local_id"] = payload.pop("local_rank", None)
        payload.pop("created", None)
        if "node_state" in payload and isinstance(payload["node_state"], str):
            value = payload["node_state"]
            payload["node_state"] = (
                NodeState[value] if value in NodeState.__members__ else NodeState(value)
            )
        return RdzvEvent(**payload)

    def serialize(self) -> str:
        return json.dumps(asdict(self))
