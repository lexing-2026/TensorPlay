"""Agent run-loop implementation with restarts and scale-up handling."""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import uuid
from string import Template
from typing import Any, Callable

from ..server.api import DEFAULT_ROLE, RunResult, SimpleElasticAgent, WorkerGroup, WorkerState
from ... import events, timer
from ...events import EventSource
from ...agent.server.health_check_server import create_healthcheck_server
from ...multiprocessing.api import start_processes
from ...utils.process_state import is_uninterruptible_state, read_proc_state
from ...metrics import prof
from ...utils.api import macros
from ...utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "LocalElasticAgent",
    "TORCHELASTIC_ENABLE_FILE_TIMER",
    "TORCHELASTIC_TIMER_FILE",
    "TORCHELASTIC_HEALTH_CHECK_PORT",
    "TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT",
]

TORCHELASTIC_ENABLE_FILE_TIMER = "TORCHELASTIC_ENABLE_FILE_TIMER"
TORCHELASTIC_TIMER_FILE = "TORCHELASTIC_TIMER_FILE"
TORCHELASTIC_HEALTH_CHECK_PORT = "TORCHELASTIC_HEALTH_CHECK_PORT"
TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT = (
    "TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT"
)


def _resolve_uninterruptible_state_timeout(explicit: float | None) -> float:
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            return 0.0
    raw = os.environ.get(TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT, "")
    try:
        return max(0.0, float(raw)) if raw else 0.0
    except ValueError:
        return 0.0


class _AliveCallbackProxy:
    def __init__(self) -> None:
        self._delegate: Callable[[], int] | None = None

    def __call__(self) -> int:
        return self._delegate() if self._delegate is not None else int(time.time())

    def set_delegate(self, delegate: Callable[[], int]) -> None:
        self._delegate = delegate


class LocalElasticAgent(SimpleElasticAgent):
    """Agent managing workers on the local node.

    Workers are launched through ``start_processes`` (subprocesses for
    command entrypoints, multiprocessing for callables) with per-rank
    environments carrying the elastic contract; failures and restarts are
    handled by the base-class run loop.
    """

    def __init__(
        self,
        spec,
        logs_specs=None,
        start_method: str = "spawn",
        exit_barrier_timeout: float = 300,
        log_line_prefix_template: str | None = None,
        shutdown_timeout: int = 30,
        health_check_server=None,
        uninterruptible_state_timeout: float | None = None,
        log_dir: str | None = None,
    ) -> None:
        from ...multiprocessing.api import DefaultLogsSpecs

        if isinstance(logs_specs, (str, bytes, os.PathLike)) and log_dir is None:
            log_dir = os.fspath(logs_specs)
            logs_specs = None
        super().__init__(spec, exit_barrier_timeout, shutdown_timeout)
        self._start_method = (
            start_method
            if start_method != "spawn" or not hasattr(spec, "start_method")
            else spec.start_method
        )
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="tp_elastic_agent_")
        self._logs_specs = logs_specs or DefaultLogsSpecs(log_dir=self._log_dir)
        self._pcontext = None
        self._worker_group = WorkerGroup(spec)
        self._remaining_restarts = spec.max_restarts
        self._rdzv_handler = spec.rdzv_handler
        self._log_line_prefix_template = log_line_prefix_template
        self._worker_watchdog = None
        self._health_check_server = health_check_server
        self._uninterruptible_state_timeout = _resolve_uninterruptible_state_timeout(
            uninterruptible_state_timeout
        )
        self._uninterruptible_state_first_seen: dict[int, float] = {}

    @property
    def log_dir(self) -> str:
        return self._log_dir

    def _setup_local_watchdog(self, envs: dict[int, dict[str, str]]) -> None:
        enabled = os.getenv(TORCHELASTIC_ENABLE_FILE_TIMER) == "1"
        path = os.getenv(TORCHELASTIC_TIMER_FILE)
        if enabled:
            if path is None:
                path = os.path.join(
                    tempfile.gettempdir(), f"watchdog_timer_{uuid.uuid4()}"
                )
            run_id = envs.get(0, {}).get("TORCHELASTIC_RUN_ID", "")
            self._worker_watchdog = timer.FileTimerServer(
                file_path=path,
                run_id=run_id,
                max_interval=0.1,
                daemon=True,
                log_event=self._log_watchdog_event,
            )
            self._worker_watchdog.start()
        if path is not None:
            for worker_env in envs.values():
                worker_env[TORCHELASTIC_TIMER_FILE] = path

    @staticmethod
    def _get_current_time_secs() -> int:
        return int(time.time())

    def _get_alive_time(self) -> int:
        if self._in_exit_barrier:
            return int(time.time())
        if self._worker_watchdog is not None:
            return self._worker_watchdog.get_last_progress_time()
        return int(time.time())

    def _setup_healthcheck(self) -> None:
        raw_port = os.getenv(TORCHELASTIC_HEALTH_CHECK_PORT)
        if raw_port is None or self._health_check_server is not None:
            return
        try:
            port = int(raw_port)
        except ValueError:
            logger.info("Invalid health check port: %s", raw_port)
            return
        callback = self._get_alive_time
        self._health_check_server = create_healthcheck_server(
            alive_callback=callback, port=port, timeout=60
        )
        self._health_check_server.start()

    def _get_fq_hostname(self) -> str:
        return socket.getfqdn(socket.gethostname())

    def _log_watchdog_event(
        self, name: str, request: timer.FileTimerRequest | None
    ) -> None:
        group = self._worker_group
        if group is None:
            return
        metadata: dict[str, Any] = {
            "run_id": group.spec.rdzv_handler.get_run_id(),
            "global_rank": None,
            "group_rank": group.group_rank,
            "worker_id": None,
            "role": group.spec.role,
            "hostname": self._get_fq_hostname(),
            "state": "RUNNING",
            "total_run_time": self._total_execution_time,
            "rdzv_backend": group.spec.rdzv_handler.get_backend(),
            "raw_error": None,
            "metadata": json.dumps(
                {
                    "watchdog_event": name,
                    **(
                        {
                            "worker_pid": request.worker_pid,
                            "scope_id": request.scope_id,
                            "expiration_time": request.expiration_time,
                            "signal": request.signal,
                        }
                        if request is not None
                        else {}
                    ),
                }
            ),
            "agent_restarts": group.spec.max_restarts - self._remaining_restarts,
        }
        events.record(
            events.Event(name=name, source=EventSource.AGENT, metadata=metadata),
            group.spec.event_log_handler,
        )

    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, int | None]:
        spec = worker_group.spec
        restart_count = spec.max_restarts - self._remaining_restarts
        use_agent_store = bool(spec.rdzv_handler.use_agent_store)
        envs = {}
        log_line_prefixes = {} if self._log_line_prefix_template else None
        hostname = socket.gethostname()
        args_by_rank: dict[int, tuple] = {}
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "MASTER_ADDR": worker_group.master_addr,
                "MASTER_PORT": str(worker_group.master_port),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_ERROR_FILE": os.path.join(
                    self._log_dir, f"error_{worker.global_rank}.json"
                ),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.getenv(
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"
                ),
            }
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
            self._set_local_rank_env(worker_env, local_rank, spec)
            envs[local_rank] = worker_env
            if callable(spec.entrypoint):
                args_by_rank[local_rank] = tuple(
                    macros.substitute(list(spec.args), str(local_rank))
                )
            if log_line_prefixes is not None:
                log_line_prefixes[local_rank] = Template(
                    self._log_line_prefix_template
                ).safe_substitute(
                    role_name=spec.role,
                    local_rank=local_rank,
                    rank=worker.global_rank,
                    hostname=hostname,
                )
        self._setup_local_watchdog(envs)
        self._setup_healthcheck()
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args_by_rank if callable(spec.entrypoint) else tuple(spec.args),
            envs=envs,
            log_dir=self._log_dir,
            start_method=self._start_method,
            logs_specs=spec.logs_specs or self._logs_specs,
            redirects=spec.redirects,
            tee=spec.tee,
            log_line_prefixes=log_line_prefixes,
            numa_options=getattr(spec, "numa_options", None),
            duplicate_stdout_filters=getattr(spec, "duplicate_stdout_filters", None),
            duplicate_stderr_filters=getattr(spec, "duplicate_stderr_filters", None),
        )
        return {local_rank: pid for local_rank, pid in self._pcontext.pids().items()}

    def _set_local_rank_env(
        self, worker_env: dict[str, str], local_rank: int, spec
    ) -> None:
        if getattr(spec, "virtual_local_rank", False):
            worker_env["LOCAL_RANK"] = "0"
            visible = os.getenv("CUDA_VISIBLE_DEVICES")
            devices = visible.split(",") if visible is not None else []
            if devices and local_rank >= len(devices):
                raise ValueError(
                    f"local_rank {local_rank} exceeds available devices"
                )
            worker_env["CUDA_VISIBLE_DEVICES"] = (
                devices[local_rank].strip() if devices else str(local_rank)
            )
        else:
            worker_env["LOCAL_RANK"] = str(local_rank)
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                worker_env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

    def _stop_workers(self, worker_group: WorkerGroup, is_restart: bool = False) -> None:
        if self._worker_watchdog is not None:
            self._worker_watchdog.stop()
            self._worker_watchdog = None
        if self._health_check_server is not None:
            self._health_check_server.stop()
            self._health_check_server = None
        if self._pcontext is not None:
            from ...multiprocessing.api import _get_default_signal

            self._pcontext.close(death_sig=_get_default_signal())
            self._pcontext = None

    def _check_uninterruptible_state_timeout(
        self, worker_group: WorkerGroup, timeout: float
    ) -> RunResult | None:
        if self._pcontext is None:
            return None
        live_pids = set(self._pcontext.pids().values())
        for pid in list(self._uninterruptible_state_first_seen):
            if pid not in live_pids:
                self._uninterruptible_state_first_seen.pop(pid, None)
        for pid in live_pids:
            elapsed = self._update_uninterruptible_dwell(
                pid, worker_group.spec.role, timeout
            )
            if elapsed is not None and elapsed >= timeout:
                self._remaining_restarts = 0
                return RunResult(state=WorkerState.UNHEALTHY)
        return None

    def _update_uninterruptible_dwell(
        self, pid: int, role: str, timeout: float
    ) -> float | None:
        state = read_proc_state(pid)
        if state is None:
            return None
        if not is_uninterruptible_state(state):
            self._uninterruptible_state_first_seen.pop(pid, None)
            return None
        first = self._uninterruptible_state_first_seen.setdefault(pid, time.monotonic())
        return max(0.0, time.monotonic() - first)

    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        if self._pcontext is None:
            return RunResult(state=WorkerState.FAILED)
        worker_pids = {worker.id for worker in worker_group.workers}
        if worker_pids != set(self._pcontext.pids().values()):
            return RunResult(state=WorkerState.UNKNOWN)
        result = self._pcontext.wait(0)
        if result is None:
            if self._uninterruptible_state_timeout > 0:
                unhealthy = self._check_uninterruptible_state_timeout(
                    worker_group, self._uninterruptible_state_timeout
                )
                if unhealthy is not None:
                    return unhealthy
            return RunResult(state=WorkerState.HEALTHY)
        run_result = RunResult(
            state=WorkerState.SUCCEEDED if not result.failures else WorkerState.FAILED,
            failures={
                worker_group.workers[local_rank].global_rank: failure
                for local_rank, failure in result.failures.items()
            },
            return_values={
                worker_group.workers[local_rank].global_rank: value
                for local_rank, value in result.return_values.items()
            },
            stdouts=dict(result.stdouts),
            stderrs=dict(result.stderrs),
        )
        return run_result

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        spec = self._worker_group.spec
        role = spec.role
        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        while True:
            assert self._worker_group.state == WorkerState.HEALTHY
            result = self._monitor_workers(self._worker_group)
            self._worker_group.state = result.state
            from ...metrics import put_metric

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{result.state.name.lower()}", 1)
            if result.state == WorkerState.SUCCEEDED:
                logger.info("[%s] Worker group succeeded", role)
                try:
                    self._exit_barrier()
                except Exception:
                    logger.warning("Exit barrier timed out or failed", exc_info=True)
                return result
            elif result.state in {WorkerState.FAILED, WorkerState.UNHEALTHY}:
                if self._remaining_restarts > 0:
                    logger.info(
                        "[%s] Worker group failed; %s restarts remaining",
                        role,
                        self._remaining_restarts,
                    )
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group, is_restart=False)
                    self._worker_group.state = WorkerState.FAILED
                    return result
            elif result.state == WorkerState.HEALTHY:
                if self._worker_group.spec.rdzv_handler.num_nodes_waiting() > 0:
                    logger.info("[%s] Scale-up detected; re-rendezvous", role)
                    self._restart_workers(self._worker_group)
                time.sleep(monitor_interval)
            elif result.state == WorkerState.UNKNOWN:
                raise RuntimeError(f"Unexpected worker group state {result.state}")
            else:
                raise RuntimeError(f"Unexpected worker group state {result.state}")

    def _shutdown(self, death_sig=None, timeout: int = 30) -> None:
        from ...multiprocessing.api import _get_default_signal

        if self._worker_watchdog is not None:
            self._worker_watchdog.stop()
            self._worker_watchdog = None
        if self._health_check_server is not None:
            self._health_check_server.stop()
            self._health_check_server = None
        if self._pcontext is not None:
            self._pcontext.close(death_sig=death_sig or _get_default_signal(), timeout=timeout)
            self._pcontext = None
        super()._shutdown(death_sig, timeout=timeout)
