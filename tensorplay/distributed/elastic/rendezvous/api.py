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

    @property
    def master_port(self) -> int:
        return self.port

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
                from ..utils.distributed import get_free_port

                port = get_free_port()
            store.set("tp_elastic/rdzv/master_addr", addr.encode())
            store.set("tp_elastic/rdzv/master_port", str(port).encode())
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

    @property
    def use_agent_store(self) -> bool:
        """Whether workers can reuse the agent-held store as their bootstrap store."""
        return False

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
        *args,
        min_nodes: int | None = None,
        max_nodes: int | None = None,
        local_addr: str | None = None,
        node_rank: int = 0,
        local_world_size: int = 1,
        config: dict | None = None,
        **kwargs,
    ) -> None:
        if len(args) >= 2 and all(isinstance(value, int) for value in args[:2]):
            positional_min_nodes, positional_max_nodes = int(args[0]), int(args[1])
            if len(args) > 2:
                local_addr = args[2]
            if len(args) > 3:
                raise TypeError("too many positional rendezvous parameters")
        else:
            positional_min_nodes, positional_max_nodes = 1, 1
            if args:
                local_addr = args[0]
            if len(args) > 1:
                node_rank = args[1]
            if len(args) > 2:
                local_world_size = args[2]
            if len(args) > 3:
                config = args[3]
            if len(args) > 4:
                raise TypeError("too many positional rendezvous parameters")
        merged_config = dict(config or {})
        merged_config.update(kwargs)
        self._min_nodes = int(
            positional_min_nodes if min_nodes is None else min_nodes
        )
        self._max_nodes = int(
            positional_max_nodes if max_nodes is None else max_nodes
        )
        if not backend:
            raise ValueError("The rendezvous backend name must be non-empty")
        if self._min_nodes < 1:
            raise ValueError("min_nodes must be greater than zero")
        if self._max_nodes < self._min_nodes:
            raise ValueError("max_nodes must be greater than or equal to min_nodes")
        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.local_addr = local_addr
        self.node_rank = int(node_rank)
        self.local_world_size = int(local_world_size)
        self.config = merged_config
        self.kwargs = {}

    def get(self, key: str, default=None):
        """Return backend parameter ``key`` or ``default``."""
        if key in self.config:
            return self.config[key]
        return self.kwargs.get(key, default)

    def get_as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return ``key`` coerced to bool (accepts 1/0/true/false)."""
        value = self.get(key, default)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value == 1:
                return True
            if value == 0:
                return False
        elif isinstance(value, str):
            text = value.lower()
            if text in ("1", "true", "t", "yes", "y"):
                return True
            if text in ("0", "false", "f", "no", "n"):
                return False
        raise ValueError(
            f"The rendezvous configuration option '{key}' does not represent a valid boolean value."
        )

    def get_as_int(self, key: str, default: int | None = None) -> int | None:
        """Return ``key`` coerced to int."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"The rendezvous configuration option '{key}' does not represent a valid integer value."
            ) from e

    @property
    def min_nodes(self) -> int:
        return self.get_as_int("min_nodes", self._min_nodes)

    @property
    def max_nodes(self) -> int:
        return self.get_as_int("max_nodes", self._max_nodes)

    @property
    def timeout(self) -> "RendezvousTimeout":
        """Timeouts assembled from ``join/last_call/close/heartbeat`` params (seconds)."""
        return RendezvousTimeout.from_params(self)


class RendezvousTimeout:
    """Hold the timeout configuration of a rendezvous."""

    _ZERO = timedelta(0)
    _DEFAULT_TIMEOUTS = {
        "join": timedelta(seconds=600),
        "last_call": timedelta(seconds=30),
        "close": timedelta(seconds=30),
        "heartbeat": timedelta(seconds=5),
    }

    def __init__(
        self,
        join: timedelta | None = None,
        last_call: timedelta | None = None,
        close: timedelta | None = None,
        heartbeat: timedelta | None = None,
    ) -> None:
        self._set_timeouts(
            join=join, last_call=last_call, close=close, heartbeat=heartbeat
        )

    @property
    def join(self) -> timedelta:
        return self._join

    @property
    def last_call(self) -> timedelta:
        return self._last_call

    @property
    def close(self) -> timedelta:
        return self._close

    @property
    def heartbeat(self) -> timedelta:
        return self._heartbeat

    def _set_timeouts(self, **timeouts: timedelta | None) -> None:
        for name, timeout in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            if timeout <= self._ZERO:
                raise ValueError(f"The {name} timeout ({timeout}) must be positive.")
            setattr(self, "_" + name, timeout)

    @staticmethod
    def from_params(params: RendezvousParameters) -> "RendezvousTimeout":
        def _seconds(key: str, default: timedelta) -> timedelta:
            raw = params.get_as_int(key)
            return timedelta(seconds=raw) if raw is not None else default

        return RendezvousTimeout(
            join=_seconds("join_timeout", RendezvousTimeout().join),
            last_call=_seconds("last_call_timeout", RendezvousTimeout().last_call),
            close=_seconds("close_timeout", RendezvousTimeout().close),
            heartbeat=_seconds("heartbeat_timeout", RendezvousTimeout().heartbeat),
        )


class RendezvousHandlerRegistry:
    """Name-to-creator registry of rendezvous backends."""

    def __init__(self) -> None:
        self._handlers: dict[str, callable] = {}

    def register(self, backend: str, creator: callable) -> None:
        """Register ``creator`` under ``backend``."""
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")
        current_creator = self._handlers.get(backend)
        if current_creator is not None and current_creator != creator:
            raise ValueError(
                f"The rendezvous backend '{backend}' is already registered with a different creator."
            )
        self._handlers[backend] = creator

    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        """Instantiate the handler for ``params.backend``."""
        try:
            creator = self._handlers[params.backend]
        except KeyError as e:
            raise ValueError(
                f"The rendezvous backend '{params.backend}' is not registered."
            ) from e
        handler = creator(params)
        if handler.get_backend() != params.backend:
            raise RuntimeError(
                f"The rendezvous backend '{handler.get_backend()}' does not match the requested "
                f"backend '{params.backend}'."
            )
        return handler


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
