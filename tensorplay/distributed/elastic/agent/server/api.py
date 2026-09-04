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
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from tensorplay.distributed import Store

from ...events import Event, EventSource, NodeState, record
from ...metrics import prof, put_metric
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
    fn: Callable | None = None
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
    virtual_local_rank: bool = False
    numa_options: Any = None
    duplicate_stdout_filters: list[str] | None = None
    duplicate_stderr_filters: list[str] | None = None

    def __post_init__(self) -> None:
        if self.local_world_size <= 0:
            raise AssertionError
        if self.monitor_interval <= 0:
            raise AssertionError
        if self.fn:
            warnings.warn(
                "WorkerSpec.fn is deprecated; use WorkerSpec.entrypoint instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self.entrypoint = self.fn
        if not self.entrypoint:
            raise AssertionError

    def get_entrypoint_name(self) -> str:
        """Human-readable name of the entrypoint (module path or command)."""
        if isinstance(self.entrypoint, str):
            return os.path.basename(self.entrypoint)
        if self.entrypoint is None:
            raise AssertionError
        return self.entrypoint.__qualname__


@dataclass(init=False)
class Worker:
    """One logical worker slot with its rank assignments."""

    id: Any = None
    local_rank: int = -1
    role_rank: int = -1
    global_rank: int = -1
    role_world_size: int = -1
    world_size: int = -1

    def __init__(
        self,
        local_rank: int = -1,
        global_rank: int = -1,
        role_rank: int = -1,
        world_size: int = -1,
        role_world_size: int = -1,
        id: Any = None,
    ) -> None:
        self.id = id
        self.local_rank = local_rank
        self.role_rank = role_rank
        self.global_rank = global_rank
        self.role_world_size = role_world_size
        self.world_size = world_size

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

    UNKNOWN = "UNKNOWN"
    INIT = "INIT"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    @staticmethod
    def is_running(state: "WorkerState") -> bool:
        return state in {WorkerState.HEALTHY, WorkerState.UNHEALTHY}


class WorkerGroup:
    """Mutable group state driven by the agent."""

    def __init__(self, spec: WorkerSpec) -> None:
        self.spec = spec
        self.workers: list[Worker] = [
            Worker(local_rank=i) for i in range(spec.local_world_size)
        ]
        self.store: Store | None = None
        self.group_rank = None
        self.group_world_size = None
        self.master_addr = None
        self.master_port = None
        self.state = WorkerState.INIT


class _RoleInstanceInfo:
    """Agent role descriptor exchanged through the store for rank assignment."""

    def __init__(
        self, role: str, rank: int | None = None, local_world_size: int | None = None
    ) -> None:
        self.role = role
        if local_world_size is None:
            local_world_size = rank
            rank = 0
        self.rank = int(rank or 0)
        self.local_world_size = int(local_world_size or 0)

    def serialize(self) -> bytes:
        return json.dumps(
            {
                "role": self.role,
                "rank": self.rank,
                "local_world_size": self.local_world_size,
            }
        ).encode()

    @staticmethod
    def deserialize(data: bytes) -> "_RoleInstanceInfo":
        payload = json.loads(data.decode())
        return _RoleInstanceInfo(
            payload["role"], payload.get("rank", 0), payload["local_world_size"]
        )

    @staticmethod
    def compare(obj1: "_RoleInstanceInfo", obj2: "_RoleInstanceInfo") -> int:
        if obj1.role == obj2.role:
            return obj1.rank - obj2.rank
        return 1 if obj1.role > obj2.role else -1

    @staticmethod
    def find_role_boundaries(
        roles_infos: list["_RoleInstanceInfo"], role: str
    ) -> tuple[int, int]:
        start_idx, end_idx = -1, -1
        for idx, role_info in enumerate(roles_infos):
            if role_info.role == role:
                start_idx = idx if start_idx == -1 else start_idx
                end_idx = idx
        return start_idx, end_idx


@dataclass
class RunResult:
    """Terminal outcome of the agent run for one role."""

    state: WorkerState
    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)

    def is_failed(self) -> bool:
        return self.state == WorkerState.FAILED


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

    def __init__(
        self,
        spec: WorkerSpec | None = None,
        exit_barrier_timeout: float = 300,
        shutdown_timeout: int = 30,
    ) -> None:
        self._worker_group: WorkerGroup | None = WorkerGroup(spec) if spec else None
        self._store: Store | None = None
        self._exit_barrier_timeout = exit_barrier_timeout
        self._remaining_restarts = spec.max_restarts if spec else 0
        self._shutdown_timeout = shutdown_timeout
        self._total_execution_time = 0
        self._in_exit_barrier = False

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
        if os.environ.get("TORCH_ELASTIC_WORKER_IDENTICAL", "0") == "1":
            global_world_size = group_world_size * spec.local_world_size
            base_global_rank = group_rank * spec.local_world_size
            base_role_rank = base_global_rank
            role_world_size = global_world_size
        else:
            role_info_prefix = "tp_elastic/role_info/"
            assigned_prefix = "tp_elastic/assigned_ranks/"
            agent_role_info = _RoleInstanceInfo(
                spec.role, group_rank, spec.local_world_size
            )
            store.set(
                f"{role_info_prefix}{group_rank}",
                agent_role_info.serialize().decode(),
            )
            if group_rank == 0:
                role_infos = []
                for i in range(group_world_size):
                    raw = store.get(f"{role_info_prefix}{i}")
                    role_infos.append(_RoleInstanceInfo.deserialize(raw))
                role_sizes: dict[str, int] = {}
                global_size = 0
                for info in role_infos:
                    role_sizes[info.role] = (
                        role_sizes.get(info.role, 0) + info.local_world_size
                    )
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
                    role_ranks[info.role] = (
                        role_ranks.get(info.role, 0) + info.local_world_size
                    )
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
            record(
                self._construct_event(
                    state=NodeState.STARTING.value,
                    source=EventSource.WORKER,
                    worker=worker_group.workers[local_rank],
                ),
                worker_group.spec.event_log_handler,
            )
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
            record(
                self._construct_event(
                    state=self._get_worker_state(worker, result),
                    source=EventSource.WORKER,
                    worker=worker,
                    raw_error=(
                        json.dumps(result.failures[worker.global_rank].error_file_data)
                        if worker.global_rank in result.failures
                        else None
                    ),
                    exit_code=(
                        result.failures[worker.global_rank].exitcode
                        if worker.global_rank in result.failures
                        else None
                    ),
                    worker_pid=(
                        result.failures[worker.global_rank].pid
                        if worker.global_rank in result.failures
                        else None
                    ),
                ),
                group.spec.event_log_handler,
            )

    def get_event_failed(self) -> Event:
        return self._construct_event(
            state=NodeState.FAILED.value,
            source=EventSource.AGENT,
            raw_error=traceback.format_exc(),
        )

    def get_event_succeeded(self) -> Event:
        return self._construct_event(
            state=NodeState.SUCCEEDED.value,
            source=EventSource.AGENT,
        )

    @contextmanager
    def record_duration(self, state: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            group = self._worker_group
            if group is not None:
                record(
                    self._construct_event(
                        state=state,
                        source=EventSource.AGENT,
                        duration_ms=duration_ms,
                    ),
                    group.spec.event_log_handler,
                )

    def _construct_event(
        self,
        state: str,
        source: EventSource,
        worker: Worker | None = None,
        raw_error: str | None = None,
        duration_ms: float | None = None,
        exit_code: int | None = None,
        worker_pid: int | None = None,
    ) -> Event:
        group = self._worker_group
        if group is None:
            raise RuntimeError("The agent has no worker group yet")
        spec = group.spec
        metadata: dict[str, Any] = {
            "group_world_size": group.group_world_size,
            "entry_point": spec.get_entrypoint_name(),
            "run_id": spec.rdzv_handler.get_run_id(),
            "group_rank": group.group_rank,
            "role": spec.role,
            "hostname": _get_fq_hostname(),
            "state": state,
            "total_run_time": self._total_execution_time,
            "rdzv_backend": spec.rdzv_handler.get_backend(),
            "raw_error": raw_error,
            "agent_restarts": spec.max_restarts - self._remaining_restarts,
            "duration_ms": duration_ms,
        }
        if worker is not None:
            metadata.update(
                {
                    "global_rank": worker.global_rank,
                    "worker_id": str(worker.id),
                }
            )
        worker_metadata = {
            "group_world_size": group.group_world_size,
            "entry_point": spec.get_entrypoint_name(),
        }
        if worker is not None:
            worker_metadata.update(
                {
                    "local_rank": (worker.local_rank,),
                    "role_rank": (worker.role_rank,),
                    "role_world_size": (worker.role_world_size,),
                    "exit_code": (exit_code,),
                    "worker_pid": (worker_pid,),
                }
            )
        metadata["metadata"] = json.dumps(worker_metadata)
        return Event(
            source=source,
            event_type=f"tp_elastic.worker.status.{state}",
            metadata=metadata,
        )

    def _record_metric_with_condition(self, metric_name: str, condition: bool) -> None:
        group = self._worker_group
        if group is None:
            return
        put_metric(
            f"workers.{group.spec.role}.{metric_name}",
            int(condition),
        )

    def _record_flakiness_metric(self, is_failed: bool = False) -> None:
        group = self._worker_group
        if group is None:
            return
        spec = group.spec
        if is_failed:
            flakiness = 100.0
        else:
            flakiness = 100.0 - 100.0 * (self._remaining_restarts + 1) / (
                spec.max_restarts + 1
            )
        put_metric(f"workers.{spec.role}.flakiness", int(flakiness))

    def _record_metrics(self, group_results: RunResult) -> None:
        group = self._worker_group
        if group is None:
            return
        failed = group_results.is_failed()
        spec = group.spec
        restarted = self._remaining_restarts != spec.max_restarts
        self._record_flakiness_metric(failed)
        put_metric(f"workers.{spec.role}.run_total", 1)
        self._record_metric_with_condition(
            "run_success_with_retries", not failed and restarted
        )
        self._record_metric_with_condition(
            "run_success_no_retries", not failed and not restarted
        )
        self._record_metric_with_condition(
            "run_failed_with_retries", failed and restarted
        )
        self._record_metric_with_condition(
            "run_failed_no_retries", failed and not restarted
        )

    def _get_worker_state(self, worker: Worker, result: RunResult) -> str:
        failure = result.failures.get(worker.global_rank)
        if result.state in {WorkerState.UNHEALTHY, WorkerState.FAILED} and not failure:
            return "TERMINATED"
        if failure or worker.global_rank in result.return_values:
            return result.state.value
        raise ValueError(f"Unknown worker: {worker.global_rank}")

    @prof
    def _exit_barrier(self) -> None:
        """Wait for all agents to reach the exit point before tearing down."""
        if self._store is None or self._worker_group is None:
            return
        if self._worker_group.group_world_size <= 1:
            return
        self._in_exit_barrier = True
        try:
            with barrier(
                self._store,
                self._worker_group.group_world_size,
                key_prefix=f"{_TERMINAL_STATE_SYNC_ID}/{self._worker_group.spec.role}",
                timeout=self._exit_barrier_timeout,
            ):
                logger.info(
                    "Exit barrier reached for all %s agents",
                    self._worker_group.group_world_size,
                )
        finally:
            self._in_exit_barrier = False

    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        """Run the agent to a terminal state, shutting down on any exit path."""
        start_time = time.monotonic()
        shutdown_called = False
        try:
            result = self._invoke_run(role)
            self._total_execution_time = int(time.monotonic() - start_time)
            self._record_metrics(result)
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
            self._total_execution_time = int(time.monotonic() - start_time)

    @abc.abstractmethod
    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        ...
