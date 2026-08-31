"""Elastic agent contract: worker specs, group state, and the run loop.

The agent owns one homogeneous group of local workers. Its run loop
rendezvous with peer agents, starts workers with per-rank environments,
monitors them, restarts on failure (bounded by ``max_restarts``), and
re-rendezvous when peer agents report queued nodes (scale-up).
"""
import abc
import json
import os
import signal
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from tensorplay.distributed import Store

from ...events import Event, EventSource, NodeState, record
from ...metrics import prof
from ...multiprocessing.errors import ProcessFailure, SignalException
from ...rendezvous import (
    RendezvousGracefulExitError,
    RendezvousHandler,
    RendezvousInfo,
)
from ...utils.logging import get_logger
from ...utils.store import barrier

__all__ = [
    "WorkerSpec",
    "Worker",
    "WorkerState",
    "WorkerGroup",
    "RunResult",
    "ElasticAgent",
    "SimpleElasticAgent",
]

DEFAULT_ROLE = "default"
logger = get_logger(__name__)

_TERMINAL_STATE_SYNC_ID = "tp_elastic/agent/terminal_state"


@dataclass
class WorkerSpec:
    """Blueprint of the worker group this agent manages.

    Every node runs the same spec: the same ``role`` name, the same
    ``local_world_size``, and the same entrypoint semantics, so global rank
    arithmetic across agents stays consistent.
    """

    role: str
    local_world_size: int
    rdzv_handler: RendezvousHandler
    entrypoint: Callable | str | None = None
    args: tuple = ()
    max_restarts: int = 3
    monitor_interval: float = 0.1
    master_port: int | None = None
    master_addr: str | None = None
    local_addr: str | None = None
    event_log_handler: str = "null"
    logs_specs: Any | None = None
    start_method: str = "spawn"
    redirects: Any = None
    tee: Any = None
    log_dir: str | None = None

    def __post_init__(self) -> None:
        if self.local_world_size <= 0:
            raise ValueError("local_world_size must be positive")
        if self.monitor_interval <= 0:
            raise ValueError("monitor_interval must be positive")

    def get_entrypoint_name(self) -> str:
        """Human-readable name of the entrypoint (module path or command)."""
        if self.entrypoint is None:
            return "None"
        if isinstance(self.entrypoint, str):
            return os.path.basename(self.entrypoint)
        return getattr(self.entrypoint, "__qualname__", str(self.entrypoint))


@dataclass
class Worker:
    """One logical worker slot with its rank assignments."""

    id: Any = None
    local_rank: int = -1
    role_rank: int = -1
    global_rank: int = -1
    role_world_size: int = -1
    world_size: int = -1

    def __str__(self) -> str:
        return (
            f"Worker(id={self.id}, local_rank={self.local_rank}, "
            f"global_rank={self.global_rank}, role_rank={self.role_rank}, "
            f"world_size={self.world_size}, role_world_size={self.role_world_size})"
        )

    def __repr__(self) -> str:
        return str(self)


class WorkerState(str, Enum):
    """State of the worker group in the agent run loop."""

    INITIALIZED = "INITIALIZED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"

    @staticmethod
    def is_running(state: "WorkerState") -> bool:
        return state in {WorkerState.HEALTHY, WorkerState.UNHEALTHY, WorkerState.STARTING}


class WorkerGroup:
    """Mutable group state driven by the agent."""

    def __init__(self, spec: WorkerSpec) -> None:
        self.spec = spec
        self.workers: list[Worker] = []
        self.store: Store | None = None
        self.group_rank = -1
        self.group_world_size = -1
        self.master_addr = ""
        self.master_port = -1
        self.state = WorkerState.INITIALIZED


class _RoleInstanceInfo:
    """Agent role descriptor exchanged through the store for rank assignment."""

    def __init__(self, role: str, local_world_size: int) -> None:
        self.role = role
        self.local_world_size = local_world_size

    def serialize(self) -> bytes:
        return json.dumps({"role": self.role, "local_world_size": self.local_world_size}).encode()

    @staticmethod
    def deserialize(data: bytes) -> "_RoleInstanceInfo":
        payload = json.loads(data.decode())
        return _RoleInstanceInfo(payload["role"], payload["local_world_size"])


@dataclass
class RunResult:
    """Terminal outcome of the agent run for one role."""

    state: WorkerState = WorkerState.UNHEALTHY
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    return_values: dict[int, Any] = field(default_factory=dict)
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)

    def is_failed(self) -> bool:
        return bool(self.failures)


def _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


class ElasticAgent(abc.ABC):
    """Agent interface for one worker-group role."""

    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]:
        ...

    @abc.abstractmethod
    def _stop_workers(self, worker_group: WorkerGroup, is_restart: bool = False) -> None:
        ...

    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        ...

    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        """Run the worker group to a terminal state."""
        raise NotImplementedError

    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        """Return the managed group."""
        raise NotImplementedError


class SimpleElasticAgent(ElasticAgent):
    """Reusable agent run loop over one worker group."""

    def __init__(self, exit_barrier_timeout: float = 300) -> None:
        self._worker_group: WorkerGroup | None = None
        self._store: Store | None = None
        self._exit_barrier_timeout = exit_barrier_timeout
        self._remaining_restarts = 0
        self._shutdown_timeout = 30
        self._total_execution_time = 0

    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        if self._worker_group is None:
            raise RuntimeError("The agent has no worker group yet")
        return self._worker_group

    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]:
        ...

    @abc.abstractmethod
    def _stop_workers(self, worker_group: WorkerGroup, is_restart: bool = False) -> None:
        ...

    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        ...

    def _shutdown(self, death_sig: "signal.Signals | None" = None, timeout: int = 30) -> None:
        """Stop workers and release the rendezvous."""
        if self._worker_group is None:
            return
        try:
            if self._worker_group.state != WorkerState.STOPPED:
                self._stop_workers(self._worker_group)
            self._worker_group.state = WorkerState.CLOSED
        finally:
            if self._worker_group.spec.rdzv_handler is not None:
                try:
                    self._worker_group.spec.rdzv_handler.shutdown()
                except Exception:
                    logger.warning("Rendezvous shutdown failed", exc_info=True)

    @prof
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        """Rendezvous and assign ranks; refreshes all group state."""
        spec = worker_group.spec
        rdzv_info: RendezvousInfo = spec.rdzv_handler.next_rendezvous()
        store = rdzv_info.store
        group_rank = rdzv_info.rank
        group_world_size = rdzv_info.world_size
        bootstrap = rdzv_info.bootstrap_store_info
        master_addr = spec.master_addr or (bootstrap.master_addr if bootstrap else "")
        master_port = spec.master_port or (bootstrap.port if bootstrap else -1)
        self._store = store
        workers = self._assign_worker_ranks(store, group_rank, group_world_size, spec)
        worker_group.workers = workers
        worker_group.store = store
        worker_group.group_rank = group_rank
        worker_group.group_world_size = group_world_size
        worker_group.master_addr = master_addr
        worker_group.master_port = master_port
        restart_count = spec.max_restarts - self._remaining_restarts
        logger.info(
            "[%s] Rendezvous complete: restart_count=%s master_addr=%s master_port=%s "
            "group_rank=%s group_world_size=%s global_ranks=%s world_size=%s",
            spec.role,
            restart_count,
            master_addr,
            master_port,
            group_rank,
            group_world_size,
            [w.global_rank for w in workers],
            workers[0].world_size if workers else -1,
        )

    @prof
    def _assign_worker_ranks(
        self, store: Store, group_rank: int, group_world_size: int, spec: WorkerSpec
    ) -> list[Worker]:
        """Determine global/role ranks for local workers.

        Each agent publishes its role descriptor; the rank-0 agent aggregates
        them and writes base ranks per agent; every agent then derives its
        workers' ranks from its own base.
        """
        role_info_prefix = "tp_elastic/role_info/"
        assigned_prefix = "tp_elastic/assigned_ranks/"
        agent_role_info = _RoleInstanceInfo(spec.role, spec.local_world_size)
        store.set(f"{role_info_prefix}{group_rank}", agent_role_info.serialize().decode())
        if group_rank == 0:
            role_infos = []
            for i in range(group_world_size):
                raw = store.get(f"{role_info_prefix}{i}")
                role_infos.append(_RoleInstanceInfo.deserialize(raw))
            role_sizes: dict[str, int] = {}
            global_size = 0
            for info in role_infos:
                role_sizes[info.role] = role_sizes.get(info.role, 0) + info.local_world_size
                global_size += info.local_world_size
            base_global_rank = 0
            role_ranks: dict[str, int] = {}
            for i, info in enumerate(role_infos):
                payload = json.dumps(
                    [
                        base_global_rank,
                        global_size,
                        role_ranks.get(info.role, 0),
                        role_sizes[info.role],
                    ]
                )
                store.set(f"{assigned_prefix}{i}", payload)
                base_global_rank += info.local_world_size
                role_ranks[info.role] = role_ranks.get(info.role, 0) + info.local_world_size
        assigned = json.loads(store.get(f"{assigned_prefix}{group_rank}").decode())
        base_global_rank, global_world_size, base_role_rank, role_world_size = assigned
        workers = []
        for local_rank in range(spec.local_world_size):
            workers.append(
                Worker(
                    local_rank=local_rank,
                    global_rank=base_global_rank + local_rank,
                    role_rank=base_role_rank + local_rank,
                    world_size=global_world_size,
                    role_world_size=role_world_size,
                )
            )
        return workers

    @prof
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        """Rendezvous then start a fresh set of workers."""
        role = worker_group.spec.role
        logger.info("[%s] Rendezvous'ing worker group", role)
        self._rendezvous(worker_group)
        logger.info("[%s] Starting worker group", role)
        worker_ids = self._start_workers(worker_group)
        for local_rank, worker_id in worker_ids.items():
            worker_group.workers[local_rank].id = worker_id
        worker_group.state = WorkerState.HEALTHY

    @prof
    def _restart_workers(self, worker_group: WorkerGroup) -> None:
        """Stop, re-rendezvous, and start the group."""
        role = worker_group.spec.role
        logger.info("[%s] Stopping worker group for restart", role)
        self._stop_workers(worker_group, is_restart=True)
        worker_group.state = WorkerState.STOPPED
        self._initialize_workers(worker_group)

    def _record_worker_events(self, result: RunResult) -> None:
        group = self._worker_group
        if group is None:
            return
        for worker in group.workers:
            state = self._get_worker_state(worker, result)
            record(
                Event(
                    source=EventSource.WORKER,
                    event_type=state,
                    metadata={"global_rank": worker.global_rank, "local_rank": worker.local_rank},
                ),
                group.spec.event_log_handler,
            )

    def _get_worker_state(self, worker: Worker, result: RunResult) -> str:
        if worker.global_rank in result.return_values:
            return NodeState.SUCCEEDED.value
        if worker.global_rank in result.failures:
            return NodeState.FAILED.value
        return result.state.value

    @prof
    def _exit_barrier(self) -> None:
        """Wait for all agents to reach the exit point before tearing down."""
        if self._store is None or self._worker_group is None:
            return
        if self._worker_group.group_world_size <= 1:
            return
        with barrier(
            self._store,
            self._worker_group.group_world_size,
            key_prefix=f"{_TERMINAL_STATE_SYNC_ID}/{self._worker_group.spec.role}",
            timeout=self._exit_barrier_timeout,
        ):
            logger.info("Exit barrier reached for all %s agents", self._worker_group.group_world_size)

    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        """Run the agent to a terminal state, shutting down on any exit path."""
        start_time = time.monotonic()
        shutdown_called = False
        try:
            result = self._invoke_run(role)
            self._total_execution_time = int(time.monotonic() - start_time)
            self._record_worker_events(result)
            return result
        except RendezvousGracefulExitError as e:
            logger.info("Rendezvous gracefully exited: %s", e)
            return RunResult(state=WorkerState.SUCCEEDED)
        except SignalException as e:
            logger.warning("Received %s death signal, shutting down workers", e.sigval)
            self._shutdown(e.sigval, timeout=self._shutdown_timeout)
            shutdown_called = True
            raise
        finally:
            if not shutdown_called:
                self._shutdown(timeout=self._shutdown_timeout)

    @abc.abstractmethod
    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        ...

