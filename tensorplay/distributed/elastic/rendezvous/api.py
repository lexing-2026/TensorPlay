"""Rendezvous contract shared by all rendezvous backends.

A rendezvous produces, for every participating agent, a shared key/value
store plus its rank within the agent group. Handlers implement the
:class:`RendezvousHandler` interface; parameters and timeouts are carried by
:class:`RendezvousParameters` and :class:`RendezvousTimeout`.
"""
import abc
import socket
from dataclasses import dataclass, field
from datetime import timedelta

from tensorplay.distributed import Store


class RendezvousError(Exception):
    """Base class for all rendezvous failures."""


class RendezvousClosedError(RendezvousError):
    """The rendezvous has been closed and accepts no more participants."""


class RendezvousTimeoutError(RendezvousError):
    """A rendezvous phase exceeded its time budget."""


class RendezvousConnectionError(RendezvousError):
    """The rendezvous backend store could not be reached."""


class RendezvousStateError(RendezvousError):
    """The rendezvous state is corrupt or the backend rejected an update."""


class RendezvousGracefulExitError(RendezvousError):
    """Raised to unwind an agent when it is the last node leaving."""


class RendezvousExhaustedError(RendezvousError):
    """The rendezvous join window elapsed without reaching min nodes."""

    def __init__(self, config, timeout) -> None:
        super().__init__(
            f"The rendezvous join window elapsed without reaching "
            f"min_nodes={config.min_nodes} within {timeout}."
        )
        self.config = config
        self.timeout = timeout


@dataclass
class RendezvousStoreInfo:
    """Connection information for the bootstrap store handed to workers."""

    master_addr: str
    port: int

    @property
    def addr(self) -> str:
        """Alias of :attr:`master_addr`."""
        return self.master_addr

    @classmethod
    def build(cls, rank: int, store: Store, local_addr: str | None = None, server_port: int | None = None):
        """Derive store info for ``rank`` from the rendezvous store.

        Rank 0 owns the address/port pair: it uses the store's own listening
        endpoint when available and otherwise publishes one through the
        store so every other rank can read it.
        """
        if rank == 0:
            addr = local_addr or socket.getfqdn()
            port = server_port
            endpoint = getattr(store, "master_addr_port", None)
            if endpoint is not None and port is None:
                if callable(endpoint):
                    endpoint = endpoint()
                if isinstance(endpoint, tuple):
                    host, port = endpoint
                    if not local_addr:
                        addr = host
            if port is None:
                raise RendezvousStateError(
                    "The rendezvous store does not expose a listening endpoint; "
                    "pass server_port explicitly."
                )
            store.set("tp_elastic/rdzv/master_addr", addr)
            store.set("tp_elastic/rdzv/master_port", str(port))
            return cls(master_addr=addr, port=int(port))
        end = _deadline(300)
        while True:
            try:
                addr = store.get("tp_elastic/rdzv/master_addr", timeout=5).decode()
                port = int(store.get("tp_elastic/rdzv/master_port", timeout=5).decode())
                return cls(master_addr=addr, port=port)
            except Exception:
                if _now() >= end:
                    raise RendezvousTimeoutError(
                        "Timed out waiting for the bootstrap store address"
                    )


def _now() -> float:
    import time

    return time.monotonic()


def _deadline(seconds: float) -> float:
    return _now() + seconds


class RendezvousInfo:
    """Outcome of a successful rendezvous for one agent."""

    def __init__(
        self,
        store: Store,
        rank: int,
        world_size: int,
        bootstrap_store_info: RendezvousStoreInfo | None = None,
        participants: list | None = None,
        wait_list: list | None = None,
    ) -> None:
        self._store = store
        self._rank = rank
        self._world_size = world_size
        self._bootstrap_store_info = bootstrap_store_info
        self._participants = participants or []
        self._wait_list = wait_list or []

    @property
    def store(self) -> Store:
        """Shared store for this rendezvous round."""
        return self._store

    @property
    def rank(self) -> int:
        """This agent's rank within the rendezvous."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Number of participating agents."""
        return self._world_size

    @property
    def bootstrap_store_info(self) -> RendezvousStoreInfo | None:
        """Connection info for the worker-facing bootstrap store."""
        return self._bootstrap_store_info

    @property
    def participants(self) -> list:
        """Nodes participating in this round."""
        return self._participants

    @property
    def wait_list(self) -> list:
        """Nodes waiting for a future round."""
        return self._wait_list


class RendezvousHandler(abc.ABC):
    """Algorithmic interface of one rendezvous backend."""

    @abc.abstractmethod
    def get_backend(self) -> str:
        """Backend name (e.g. ``static`` or ``core``)."""
        ...

    @abc.abstractmethod
    def use_agent_store(self) -> bool:
        """Whether workers can reuse the agent-held store as their bootstrap store."""
        ...

    @abc.abstractmethod
    def next_rendezvous(self) -> RendezvousInfo:
        """Join or re-join the rendezvous; blocks until this round completes."""
        ...

    @abc.abstractmethod
    def is_closed(self) -> bool:
        """Whether the rendezvous has been marked closed."""
        ...

    @abc.abstractmethod
    def set_closed(self) -> None:
        """Mark the rendezvous closed; new joiners are rejected."""
        ...

    @abc.abstractmethod
    def num_nodes_waiting(self) -> int:
        """Number of nodes queued for the next round (pending scale-up)."""
        ...

    @abc.abstractmethod
    def get_run_id(self) -> str:
        """Run identifier of this rendezvous."""
        ...

    @abc.abstractmethod
    def shutdown(self) -> bool:
        """Release rendezvous resources; returns True when the backend shut down."""
        ...

    def pre_shutdown(self) -> None:
        """Optional hook invoked before :meth:`shutdown` during agent teardown."""


@dataclass
class RendezvousParameters:
    """Parameters describing one rendezvous request.

    ``endpoint`` is ``host:port`` of the rendezvous store, ``local_addr`` the
    address of this node (autodetected when empty), and ``config`` carries
    backend-specific options accessible through :meth:`get`.
    """

    backend: str
    endpoint: str
    run_id: str
    local_addr: str | None
    node_rank: int
    local_world_size: int
    config: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)

    def __init__(
        self,
        backend: str,
        endpoint: str,
        run_id: str,
        local_addr: str | None = None,
        node_rank: int = 0,
        local_world_size: int = 1,
        config: dict | None = None,
        **kwargs,
    ) -> None:
        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.local_addr = local_addr
        self.node_rank = int(node_rank)
        self.local_world_size = int(local_world_size)
        self.config = config or {}
        self.kwargs = kwargs

    def get(self, key: str, default=None):
        """Return backend parameter ``key`` or ``default``."""
        if key in self.config:
            return self.config[key]
        return self.kwargs.get(key, default)

    def get_as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return ``key`` coerced to bool (accepts 1/0/true/false)."""
        if key not in self.config and key not in self.kwargs:
            return default
        value = self.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"Parameter '{key}' is not a boolean: {value!r}")

    def get_as_int(self, key: str, default: int | None = None) -> int | None:
        """Return ``key`` coerced to int."""
        if key not in self.config and key not in self.kwargs:
            return default
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Parameter '{key}' is not an integer: {value!r}") from e

    @property
    def min_nodes(self) -> int:
        return self.get_as_int("min_nodes", 1)

    @property
    def max_nodes(self) -> int:
        return self.get_as_int("max_nodes", self.min_nodes)

    @property
    def timeout(self) -> "RendezvousTimeout":
        """Timeouts assembled from ``join/last_call/close/heartbeat`` params (seconds)."""
        return RendezvousTimeout.from_params(self)


@dataclass
class RendezvousTimeout:
    """Time budgets of the rendezvous phases."""

    join: timedelta = timedelta(seconds=600)
    last_call: timedelta = timedelta(seconds=30)
    close: timedelta = timedelta(seconds=30)
    heartbeat: timedelta = timedelta(seconds=5)

    @staticmethod
    def from_params(params: RendezvousParameters) -> "RendezvousTimeout":
        def _seconds(key: str, default: timedelta) -> timedelta:
            raw = params.get_as_int(key)
            return timedelta(seconds=raw) if raw is not None else default

        return RendezvousTimeout(
            join=_seconds("join_timeout", RendezvousTimeout.join),
            last_call=_seconds("last_call_timeout", RendezvousTimeout.last_call),
            close=_seconds("close_timeout", RendezvousTimeout.close),
            heartbeat=_seconds("heartbeat_timeout", RendezvousTimeout.heartbeat),
        )


class RendezvousHandlerRegistry:
    """Name-to-creator registry of rendezvous backends."""

    def __init__(self) -> None:
        self._handlers: dict[str, callable] = {}

    def register(self, backend: str, creator: callable) -> None:
        """Register ``creator`` under ``backend``; later registrations win."""
        if not backend:
            raise ValueError("Backend name must be non-empty")
        self._handlers[backend] = creator

    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        """Instantiate the handler for ``params.backend``."""
        try:
            creator = self._handlers[params.backend]
        except KeyError:
            raise RendezvousError(
                f"No rendezvous handler for backend '{params.backend}'. "
                f"Registered backends: {sorted(self._handlers)}"
            ) from None
        return creator(params)


_create_handler_registry: RendezvousHandlerRegistry | None = None


def get_registry() -> RendezvousHandlerRegistry:
    """Process-wide handler registry (with default backends pre-registered)."""
    global _create_handler_registry
    if _create_handler_registry is None:
        from .registry import _register_default_handlers

        _create_handler_registry = RendezvousHandlerRegistry()
        _register_default_handlers(_create_handler_registry)
    return _create_handler_registry


def create_handler(params: RendezvousParameters) -> RendezvousHandler:
    """Create a handler from the process-wide registry."""
    return get_registry().create_handler(params)
