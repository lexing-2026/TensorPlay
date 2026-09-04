"""Dynamic rendezvous state machine and store-backed handler."""

from __future__ import annotations

import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from tensorplay.distributed import PrefixStore, Store
from ..events import NodeState, construct_and_record_rdzv_event

from .api import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousGracefulExitError,
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousStoreInfo,
    RendezvousTimeout,
    RendezvousTimeoutError,
)
from .core_rendezvous_backend import RendezvousBackend, Token, create_backend
from .utils import _delay, _PeriodicTimer

__all__ = [
    "RendezvousBackend",
    "RendezvousTimeout",
    "RendezvousSettings",
    "DynamicRendezvousHandler",
    "create_handler",
]

logger = logging.getLogger(__name__)


def get_method_name(depth: int = 2) -> str:
    """Return the caller name when it is available."""
    import inspect

    stack = inspect.stack()
    try:
        return stack[depth].function if len(stack) > depth else "no_method_name"
    finally:
        del stack


@dataclass(init=False)
class RendezvousSettings:
    """Configuration shared by the rendezvous state machine."""

    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int

    def __init__(
        self,
        *args: Any,
        join: timedelta | None = None,
        last_call: timedelta | None = None,
        close: timedelta | None = None,
        heartbeat: timedelta | None = None,
        min_nodes: int = 1,
        max_nodes: int = 1,
        run_id: str = "",
        timeout: RendezvousTimeout | None = None,
        keep_alive_interval: timedelta | None = None,
        keep_alive_max_attempt: int = 3,
    ) -> None:
        if args:
            if isinstance(args[0], str):
                run_id = args[0]
                min_nodes = args[1]
                max_nodes = args[2]
                timeout = args[3] if len(args) > 3 else timeout
                keep_alive_interval = args[4] if len(args) > 4 else keep_alive_interval
                keep_alive_max_attempt = args[5] if len(args) > 5 else keep_alive_max_attempt
            else:
                values = list(args) + [None] * 6
                join, last_call, close, heartbeat = values[:4]
                min_nodes = values[4] if values[4] is not None else min_nodes
                max_nodes = values[5] if values[5] is not None else max_nodes
        self.run_id = run_id
        self.min_nodes = int(min_nodes)
        self.max_nodes = int(max_nodes)
        self.timeout = timeout or RendezvousTimeout(
            join=join, last_call=last_call, close=close, heartbeat=heartbeat
        )
        self.keep_alive_interval = keep_alive_interval or (
            self.timeout.heartbeat / 3
        )
        self.keep_alive_max_attempt = int(keep_alive_max_attempt)

    @property
    def join(self) -> timedelta:
        return self.timeout.join

    @property
    def last_call(self) -> timedelta:
        return self.timeout.last_call

    @property
    def close(self) -> timedelta:
        return self.timeout.close

    @property
    def heartbeat(self) -> timedelta:
        return self.timeout.heartbeat


@dataclass(frozen=True, order=True)
class _NodeDesc:
    """Identify one handler instance."""

    addr: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        return f"{self.addr}_{self.pid}_{self.local_id}"


class _NodeDescGenerator:
    """Generate process-unique node descriptors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local_id = 0

    def generate(self, local_addr: str | None = None) -> _NodeDesc:
        with self._lock:
            local_id = self._local_id
            self._local_id += 1
        return _NodeDesc(local_addr or socket.getfqdn(), os.getpid(), local_id)


def _node_desc(
    addr: str | None = None,
    pid: int | None = None,
    local_id: str | None = None,
) -> _NodeDesc:
    """Build a descriptor for callers using the legacy helper."""
    value = int(local_id, 16) if local_id else 0
    return _NodeDesc(addr or "localhost", pid or os.getpid(), value)


class _RendezvousState:
    """Serializable state for one rendezvous round."""

    def __init__(self) -> None:
        self.round = 0
        self.complete = False
        self.deadline: datetime | None = None
        self.closed = False
        self.participants: dict[_NodeDesc, int] = {}
        self.wait_list: set[_NodeDesc] = set()
        self.redundancy_list: set[_NodeDesc] = set()
        self.last_heartbeats: dict[_NodeDesc, datetime] = {}

    def prune_dead_nodes(self, expire_ms: float) -> list[_NodeDesc]:
        now = datetime.now(timezone.utc)
        expire = timedelta(milliseconds=expire_ms)
        dead = [
            node
            for node, heartbeat in self.last_heartbeats.items()
            if now - heartbeat > expire
        ]
        for node in dead:
            self.last_heartbeats.pop(node, None)
            self.participants.pop(node, None)
            self.wait_list.discard(node)
            self.redundancy_list.discard(node)
        return dead


def _remove_participant_epilogue(
    state: _RendezvousState, settings: RendezvousSettings
) -> None:
    if state.complete:
        if not state.participants:
            state.complete = False
            state.round += 1
    elif len(state.participants) < settings.min_nodes:
        state.deadline = None


class _RendezvousStateHolder(ABC):
    @property
    @abstractmethod
    def state(self) -> _RendezvousState:
        raise NotImplementedError

    @abstractmethod
    def sync(self) -> bool | None:
        raise NotImplementedError

    @abstractmethod
    def mark_dirty(self) -> None:
        raise NotImplementedError


class _BackendRendezvousStateHolder(_RendezvousStateHolder):
    """Cache, sanitize, and conditionally persist rendezvous state."""

    def __init__(
        self,
        backend: RendezvousBackend,
        settings: RendezvousSettings,
        cache_duration: int = 1,
    ) -> None:
        self._backend = backend
        self._state = _RendezvousState()
        self._settings = settings
        self._cache_duration = max(0, int(cache_duration))
        self._token: Token | None = None
        self._dirty = False
        self._last_sync_time = -1.0
        self._dead_nodes: list[_NodeDesc] = []

    def _record(self, message: str, node_state: NodeState = NodeState.RUNNING) -> None:
        construct_and_record_rdzv_event(
            node_state=node_state,
            run_id=self._settings.run_id,
            message=message,
        )

    @property
    def state(self) -> _RendezvousState:
        return self._state

    def sync(self) -> bool | None:
        state_bits: bytes | None = None
        token: Token | None = None
        has_set: bool | None
        if self._dirty:
            candidate = pickle.dumps(self._state)
            response = self._backend.set_state(candidate, self._token)
            has_set = False
            if response is not None:
                if len(response) == 3:
                    state_bits, token, has_set = response
                else:
                    state_bits, token = response
                    has_set = state_bits == candidate
        else:
            has_set = None
            if self._cache_duration and self._last_sync_time >= max(
                time.monotonic() - self._cache_duration, 0
            ):
                return None
            response = self._backend.get_state()
            if response is not None:
                state_bits, token = response

        if state_bits is None:
            self._state = _RendezvousState()
        else:
            try:
                value = pickle.loads(state_bits)
            except (pickle.PickleError, EOFError, AttributeError, ValueError) as exc:
                raise RendezvousStateError(
                    "The rendezvous state is corrupt. See inner exception for details."
                ) from exc
            if not isinstance(value, _RendezvousState):
                raise RendezvousStateError("The rendezvous state has an invalid type")
            self._state = value

        self._token = token
        self._dirty = False
        self._last_sync_time = time.monotonic()
        self._sanitize()
        return has_set

    def _sanitize(self) -> None:
        expire = self._settings.keep_alive_interval * self._settings.keep_alive_max_attempt
        now = datetime.now(timezone.utc)
        self._dead_nodes = [
            node
            for node, heartbeat in self._state.last_heartbeats.items()
            if heartbeat < now - expire
        ]
        removed = False
        for node in self._dead_nodes:
            self._state.last_heartbeats.pop(node, None)
            if node in self._state.participants:
                del self._state.participants[node]
                removed = True
            self._state.wait_list.discard(node)
            self._state.redundancy_list.discard(node)
        if removed:
            _remove_participant_epilogue(self._state, self._settings)
            self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True


class _Action(Enum):
    KEEP_ALIVE = 1
    ADD_TO_PARTICIPANTS = 2
    ADD_TO_WAIT_LIST = 3
    ADD_TO_REDUNDANCY_LIST = 4
    REMOVE_FROM_PARTICIPANTS = 5
    REMOVE_FROM_WAIT_LIST = 6
    REMOVE_FROM_REDUNDANCY_LIST = 7
    MARK_RENDEZVOUS_COMPLETE = 8
    MARK_RENDEZVOUS_CLOSED = 9
    SYNC = 10
    ERROR_CLOSED = 11
    ERROR_TIMEOUT = 12
    FINISH = 13


@dataclass
class _RendezvousContext:
    node: _NodeDesc
    state: _RendezvousState
    settings: RendezvousSettings

    def __init__(
        self, node: _NodeDesc, state: _RendezvousState, settings: RendezvousSettings
    ) -> None:
        self.node = node
        self.state = state
        self.settings = settings


class _RendezvousOpExecutor(ABC):
    @abstractmethod
    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
        update_deadline: Callable[[timedelta], float] | None = None,
    ) -> None:
        raise NotImplementedError


class _DistributedRendezvousOpExecutor(_RendezvousOpExecutor):
    """Apply state transitions until an operation reaches its terminal state."""

    def __init__(
        self,
        node: _NodeDesc,
        state_holder: _RendezvousStateHolder,
        settings: RendezvousSettings,
    ) -> None:
        self._node = node
        self._state_holder = state_holder
        self._settings = settings

    def _record(self, message: str, node_state: NodeState = NodeState.RUNNING) -> None:
        construct_and_record_rdzv_event(
            node_state=node_state,
            run_id=self._settings.run_id,
            message=message,
            hostname=self._node.addr,
            pid=self._node.pid,
            local_id=self._node.local_id,
        )

    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
        update_deadline: Callable[[timedelta], float] | None = None,
    ) -> None:
        action: _Action | None = None
        while action != _Action.FINISH:
            self._state_holder.sync()
            ctx = _RendezvousContext(
                self._node, self._state_holder.state, self._settings
            )
            action = state_handler(ctx, deadline)
            if action == _Action.FINISH:
                continue
            if action == _Action.ERROR_CLOSED:
                raise RendezvousClosedError
            if action == _Action.ERROR_TIMEOUT:
                raise RendezvousTimeoutError
            if action == _Action.SYNC:
                _delay(1)
                continue
            if action == _Action.KEEP_ALIVE:
                self._keep_alive()
            elif action == _Action.ADD_TO_PARTICIPANTS:
                self._add_to_participants()
            elif action == _Action.ADD_TO_WAIT_LIST:
                self._add_to_wait_list()
            elif action == _Action.ADD_TO_REDUNDANCY_LIST:
                self._add_to_redundancy_list()
            elif action == _Action.REMOVE_FROM_PARTICIPANTS:
                self._remove_from_participants()
            elif action == _Action.REMOVE_FROM_WAIT_LIST:
                self._remove_from_wait_list()
            elif action == _Action.REMOVE_FROM_REDUNDANCY_LIST:
                self._remove_from_redundancy_list()
                if update_deadline:
                    deadline = update_deadline(self._settings.timeout.join)
            elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
                self._mark_rendezvous_complete()
            elif action == _Action.MARK_RENDEZVOUS_CLOSED:
                self._mark_rendezvous_closed()
            self._state_holder.mark_dirty()

    def _keep_alive(self) -> None:
        self._state_holder.state.last_heartbeats[self._node] = datetime.now(
            timezone.utc
        )

    def _add_to_participants(self) -> None:
        state = self._state_holder.state
        state.wait_list.discard(self._node)
        state.participants[self._node] = 0
        self._keep_alive()
        if len(state.participants) == self._settings.min_nodes:
            state.deadline = datetime.now(timezone.utc) + self._settings.timeout.last_call
        if len(state.participants) == self._settings.max_nodes:
            self._mark_rendezvous_complete()

    def _add_to_wait_list(self) -> None:
        state = self._state_holder.state
        state.redundancy_list.discard(self._node)
        state.wait_list.add(self._node)
        self._keep_alive()

    def _add_to_redundancy_list(self) -> None:
        state = self._state_holder.state
        state.redundancy_list.add(self._node)
        self._keep_alive()

    def _remove_from_participants(self) -> None:
        state = self._state_holder.state
        state.participants.pop(self._node, None)
        state.last_heartbeats.pop(self._node, None)
        _remove_participant_epilogue(state, self._settings)

    def _remove_from_wait_list(self) -> None:
        state = self._state_holder.state
        state.wait_list.discard(self._node)
        state.last_heartbeats.pop(self._node, None)

    def _remove_from_redundancy_list(self) -> None:
        state = self._state_holder.state
        state.redundancy_list.discard(self._node)
        state.last_heartbeats.pop(self._node, None)

    def _mark_rendezvous_complete(self) -> None:
        state = self._state_holder.state
        state.complete = True
        state.deadline = None
        for rank, node in enumerate(sorted(state.participants)):
            state.participants[node] = rank

    def _mark_rendezvous_closed(self) -> None:
        self._state_holder.state.closed = True


def _should_keep_alive(ctx: _RendezvousContext) -> bool:
    heartbeat = ctx.state.last_heartbeats.get(ctx.node)
    return heartbeat is not None and heartbeat <= datetime.now(timezone.utc) - ctx.settings.keep_alive_interval


class _RendezvousExitOp:
    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if ctx.node not in ctx.state.participants:
            return _Action.FINISH
        return _Action.REMOVE_FROM_PARTICIPANTS if time.monotonic() <= deadline else _Action.ERROR_TIMEOUT


class _RendezvousJoinOp:
    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        state = ctx.state
        if state.closed:
            if ctx.node in state.redundancy_list:
                raise RendezvousGracefulExitError
            return _Action.ERROR_CLOSED
        if ctx.node in state.redundancy_list:
            if len(state.participants) == ctx.settings.max_nodes:
                return _Action.KEEP_ALIVE if _should_keep_alive(ctx) else _Action.SYNC
            return _Action.REMOVE_FROM_REDUNDANCY_LIST
        is_participant = ctx.node in state.participants
        if state.complete and is_participant:
            return _Action.FINISH
        now = time.monotonic()
        if now > deadline:
            if now <= deadline + 5:
                if is_participant:
                    return _Action.REMOVE_FROM_PARTICIPANTS
                if ctx.node in state.wait_list:
                    return _Action.REMOVE_FROM_WAIT_LIST
            return _Action.ERROR_TIMEOUT
        if state.complete:
            if len(state.participants) < ctx.settings.max_nodes:
                if ctx.node not in state.wait_list:
                    return _Action.ADD_TO_WAIT_LIST
            elif ctx.node not in state.redundancy_list and ctx.node not in state.wait_list:
                return _Action.ADD_TO_REDUNDANCY_LIST
        elif is_participant:
            if (
                ctx.settings.min_nodes <= len(state.participants) <= ctx.settings.max_nodes
                and state.deadline is not None
            ):
                if state.deadline < datetime.now(timezone.utc):
                    return _Action.MARK_RENDEZVOUS_COMPLETE
        else:
            return _Action.ADD_TO_PARTICIPANTS
        return _Action.KEEP_ALIVE if _should_keep_alive(ctx) else _Action.SYNC


class _RendezvousCloseOp:
    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if ctx.state.closed:
            return _Action.FINISH
        return _Action.ERROR_TIMEOUT if time.monotonic() > deadline else _Action.MARK_RENDEZVOUS_CLOSED


class _RendezvousKeepAliveOp:
    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if not _should_keep_alive(ctx):
            return _Action.FINISH
        return _Action.KEEP_ALIVE if time.monotonic() <= deadline else _Action.ERROR_TIMEOUT


class DynamicRendezvousHandler(RendezvousHandler):
    """Coordinate membership, ranks, heartbeats, and round transitions."""

    _node_desc_generator = _NodeDescGenerator()

    @classmethod
    def from_backend(cls, run_id: str, *args: Any, **kwargs: Any) -> "DynamicRendezvousHandler":
        source_style = len(args) >= 4 and not isinstance(
            args[0], (str, bytes, os.PathLike)
        )
        if not source_style and "store" in kwargs and "backend" in kwargs:
            args = (
                kwargs.pop("store"),
                kwargs.pop("backend"),
                kwargs.pop("min_nodes"),
                kwargs.pop("max_nodes"),
                kwargs.pop("local_addr", None),
                kwargs.pop("timeout", None),
                kwargs.pop("keep_alive_interval", 5),
                kwargs.pop("keep_alive_max_attempt", 3),
            )
            source_style = True
        if source_style:
            store = args[0]
            backend = args[1]
            min_nodes = args[2]
            max_nodes = args[3]
            local_addr = args[4] if len(args) > 4 else None
            timeout = args[5] if len(args) > 5 else None
            keep_alive_interval = args[6] if len(args) > 6 else 5
            keep_alive_max_attempt = args[7] if len(args) > 7 else 3
            settings = RendezvousSettings(
                run_id=run_id,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                timeout=timeout,
                keep_alive_interval=timedelta(seconds=keep_alive_interval)
                if not isinstance(keep_alive_interval, timedelta)
                else keep_alive_interval,
                keep_alive_max_attempt=keep_alive_max_attempt,
            )
            return cls(
                backend=backend,
                settings=settings,
                local_addr=local_addr,
                node_rank=0,
                run_id=run_id,
                store=store,
                backend_name=backend.name,
            )

        endpoint = kwargs.pop("endpoint", args[0] if args else "")
        settings = kwargs.pop("settings", None)
        local_addr = kwargs.pop("local_addr", None)
        node_rank = int(kwargs.pop("node_rank", 0))
        store = kwargs.pop("store", None)
        backend = kwargs.pop("backend", None)
        store_type = kwargs.pop("store_type", "tcp")
        if settings is None:
            settings = RendezvousSettings(
                run_id=run_id,
                min_nodes=int(kwargs.pop("min_nodes", 1)),
                max_nodes=int(kwargs.pop("max_nodes", 1)),
                timeout=kwargs.pop("timeout", None),
            )
        else:
            settings.run_id = run_id
        if backend is None:
            params = RendezvousParameters(
                backend="core",
                endpoint=endpoint,
                run_id=run_id,
                local_addr=local_addr,
                node_rank=node_rank,
                local_world_size=int(kwargs.pop("local_world_size", 1)),
                config=dict(kwargs.pop("config", {})),
                **kwargs,
            )
            backend, created_store = create_backend(params, store_type=store_type)
            store = store or created_store
        return cls(
            backend=backend,
            settings=settings,
            local_addr=local_addr,
            node_rank=node_rank,
            run_id=run_id,
            store=store,
            backend_name=backend.name,
        )

    def __init__(
        self,
        backend: RendezvousBackend | _NodeDesc | None = None,
        settings: RendezvousSettings | None = None,
        local_addr: str | None = None,
        node_rank: int = 0,
        run_id: str = "",
        store: Store | None = None,
        backend_name: str = "core",
        state_holder: _RendezvousStateHolder | None = None,
        node: _NodeDesc | None = None,
    ) -> None:
        if isinstance(backend, _NodeDesc):
            node = backend
            backend_name = str(local_addr)
            store = node_rank if isinstance(node_rank, Store) else store
            state_holder = run_id if isinstance(run_id, _RendezvousStateHolder) else state_holder
            backend = getattr(state_holder, "_backend", None)
            local_addr = node.addr
            run_id = ""
            node_rank = 0
        if backend is None or settings is None:
            raise TypeError("backend and settings are required")
        if node is not None:
            self._this_node = node
        if not run_id:
            run_id = settings.run_id
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")
        if settings.min_nodes < 1:
            raise ValueError("The minimum number of nodes must be greater than zero.")
        if settings.max_nodes < settings.min_nodes:
            raise ValueError("The maximum number of nodes must be >= min_nodes.")
        settings.run_id = run_id
        self._backend = backend
        self._backend_name = backend_name
        self._settings = settings
        self._node_rank = int(node_rank)
        self._run_id = run_id
        self._store = store if store is not None else getattr(backend, "_store", None)
        if self._store is None:
            raise ValueError("A rendezvous store is required")
        if node is None:
            self._this_node = self._node_desc_generator.generate(local_addr)
        self._node = self._this_node
        self._local_addr = local_addr
        self._state_holder = state_holder or _BackendRendezvousStateHolder(
            backend, settings
        )
        self._executor = _DistributedRendezvousOpExecutor(
            self._this_node, self._state_holder, settings
        )
        self._op_executor = self._executor
        self._heartbeat_lock = threading.Lock()
        self._keep_alive_timer: _PeriodicTimer | None = None
        self._bootstrap_store_info: RendezvousStoreInfo | None = None
        self._shared_tcp_store_server: Store | None = None

    @property
    def settings(self) -> RendezvousSettings:
        return self._settings

    def get_backend(self) -> str:
        return self._backend_name

    def _record(
        self,
        message: str,
        node_state: NodeState = NodeState.RUNNING,
        rank: int | None = None,
    ) -> None:
        construct_and_record_rdzv_event(
            node_state=node_state,
            run_id=self._settings.run_id,
            message=message,
            hostname=self._this_node.addr,
            pid=self._this_node.pid,
            local_id=self._this_node.local_id,
            rank=rank,
        )

    def _create_tcp_store_server(self, master_addr: str, master_port: int):
        from tensorplay.distributed import TCPStore

        return TCPStore(
            master_addr,
            master_port,
            world_size=1,
            is_master=True,
            wait_for_workers=False,
        )

    @property
    def use_agent_store(self) -> bool:
        return os.getenv("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "0") != "1"

    def next_rendezvous(self) -> RendezvousInfo:
        self._stop_heartbeats()
        if self._state_holder.state.round == 0:
            _delay((0, 0.3))
        deadline = self._get_deadline(self._settings.timeout.join)
        self._executor.run(_RendezvousExitOp(), deadline)
        self._executor.run(_RendezvousJoinOp(), deadline, self._get_deadline)
        self._start_heartbeats()
        rank, world_size = self._get_world()
        store = self._get_store() if self.use_agent_store else self._store
        if not self.use_agent_store:
            bootstrap_store_info = RendezvousStoreInfo.build(
                rank, self._store, local_addr=self._this_node.addr
            )
            return RendezvousInfo(
                store=store,
                rank=rank,
                world_size=world_size,
                bootstrap_store_info=bootstrap_store_info,
                participants=sorted(self._state_holder.state.participants),
                wait_list=sorted(self._state_holder.state.wait_list),
            )
        if self._bootstrap_store_info is None or (
            rank == 0 and self._shared_tcp_store_server is None
        ):
            server_port = 0
            if rank == 0:
                self._shared_tcp_store_server = self._create_tcp_store_server(
                    self._this_node.addr, server_port
                )
                server_port = getattr(self._shared_tcp_store_server, "port", 0)
            try:
                self._bootstrap_store_info = RendezvousStoreInfo.build(
                    rank,
                    self._store,
                    local_addr=self._this_node.addr,
                    server_port=server_port,
                )
            except RendezvousError:
                self._bootstrap_store_info = None
        return RendezvousInfo(
            store=store,
            rank=rank,
            world_size=world_size,
            bootstrap_store_info=self._bootstrap_store_info,
            participants=sorted(self._state_holder.state.participants),
            wait_list=sorted(self._state_holder.state.wait_list),
        )

    def is_closed(self) -> bool:
        with self._heartbeat_lock:
            self._state_holder.sync()
            return self._state_holder.state.closed

    def set_closed(self) -> None:
        with self._heartbeat_lock:
            self._close()

    def num_nodes_waiting(self) -> int:
        with self._heartbeat_lock:
            self._state_holder.sync()
            return len(self._state_holder.state.wait_list)

    def get_run_id(self) -> str:
        return self._run_id

    def shutdown(self) -> bool:
        self._stop_heartbeats()
        try:
            self._close()
        except RendezvousError:
            return False
        return True

    def _close(self) -> None:
        deadline = self._get_deadline(self._settings.timeout.close)
        self._executor.run(_RendezvousCloseOp(), deadline)

    @staticmethod
    def _keep_alive_weak(
        weak_self: weakref.ReferenceType["DynamicRendezvousHandler"],
    ) -> None:
        instance = weak_self()
        if instance is not None:
            instance._keep_alive()

    def _keep_alive(self) -> None:
        with self._heartbeat_lock:
            deadline = self._get_deadline(self._settings.timeout.heartbeat)
            self._executor.run(_RendezvousKeepAliveOp(), deadline)

    def _start_heartbeats(self) -> None:
        self._stop_heartbeats()
        weak_self = weakref.ref(self)
        timer = _PeriodicTimer(
            self._settings.keep_alive_interval,
            lambda: self._keep_alive_weak(weak_self),
            name=f"RendezvousKeepAliveTimer_{self._this_node.local_id}",
        )
        self._keep_alive_timer = timer
        timer.start()

    def _stop_heartbeats(self) -> None:
        if self._keep_alive_timer is not None:
            self._keep_alive_timer.cancel()
            self._keep_alive_timer = None

    def _get_world(self) -> tuple[int, int]:
        state = self._state_holder.state
        return state.participants[self._this_node], len(state.participants)

    def _wrap_store(self, store: Store) -> Store:
        return PrefixStore(
            f"tp_elastic/rendezvous/{self._settings.run_id}/{self._state_holder.state.round}",
            store,
        )

    def _get_store(self) -> Store:
        return self._wrap_store(self._store)

    def _get_deadline(self, timeout: timedelta) -> float:
        return time.monotonic() + timeout.total_seconds()


def _get_timeout(params: RendezvousParameters, key: str) -> timedelta | None:
    value = params.get_as_int(f"{key}_timeout")
    return timedelta(seconds=value) if value is not None else None


def create_handler(*args: Any) -> DynamicRendezvousHandler:
    """Create a dynamic handler from rendezvous parameters."""
    if len(args) == 1:
        params = args[0]
        store = backend = None
    elif len(args) == 3:
        store, backend, params = args
    else:
        raise TypeError("create_handler expects parameters or store, backend, parameters")
    timeout = RendezvousTimeout(
        join=_get_timeout(params, "join"),
        last_call=_get_timeout(params, "last_call"),
        close=_get_timeout(params, "close"),
        heartbeat=_get_timeout(params, "heartbeat"),
    )
    settings = RendezvousSettings(
        run_id=params.run_id,
        min_nodes=params.min_nodes,
        max_nodes=params.max_nodes,
        timeout=timeout,
        keep_alive_interval=timedelta(
            seconds=params.get_as_int("keep_alive_interval", 5) or 5
        ),
        keep_alive_max_attempt=params.get_as_int("keep_alive_max_attempt", 3) or 3,
    )
    if store is not None and backend is not None:
        return DynamicRendezvousHandler.from_backend(
            params.run_id,
            store,
            backend,
            params.min_nodes,
            params.max_nodes,
            params.local_addr,
            timeout,
            keep_alive_interval=params.get_as_int("keep_alive_interval", 5),
            keep_alive_max_attempt=params.get_as_int("keep_alive_max_attempt", 3),
        )
    return DynamicRendezvousHandler.from_backend(
        params.run_id,
        endpoint=params.endpoint,
        settings=settings,
        local_addr=params.local_addr,
        node_rank=params.node_rank,
        local_world_size=params.local_world_size,
        store_type=str(params.get("store_type", "tcp")),
        config=dict(params.config),
        **params.kwargs,
    )
