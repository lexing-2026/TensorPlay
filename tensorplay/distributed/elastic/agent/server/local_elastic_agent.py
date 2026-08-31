"""Agent run-loop implementation with restarts and scale-up handling."""
import os

from ..server.api import DEFAULT_ROLE, RunResult, SimpleElasticAgent, WorkerGroup, WorkerState
from ...utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["LocalElasticAgent"]


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
        log_dir: str | None = None,
        logs_specs=None,
        exit_barrier_timeout: float = 300,
        start_method: str = "spawn",
    ) -> None:
        import tempfile

        super().__init__(exit_barrier_timeout=exit_barrier_timeout)
        self._start_method = start_method
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="tp_elastic_agent_")
        self._logs_specs = logs_specs
        self._pcontext = None
        self._worker_group = WorkerGroup(spec)
        self._remaining_restarts = spec.max_restarts

    @property
    def log_dir(self) -> str:
        return self._log_dir

    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, int | None]:
        spec = worker_group.spec
        restart_count = spec.max_restarts - self._remaining_restarts
        envs = {}
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
            }
            envs[local_rank] = worker_env
        from ...multiprocessing.api import start_processes

        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=spec.args,
            envs=envs,
            log_dir=self._log_dir,
            start_method=spec.start_method if hasattr(spec, "start_method") else self._start_method,
            logs_specs=spec.logs_specs or self._logs_specs,
            redirects=spec.redirects,
            tee=spec.tee,
        )
        return {local_rank: pid for local_rank, pid in self._pcontext.pids().items()}

    def _stop_workers(self, worker_group: WorkerGroup, is_restart: bool = False) -> None:
        if self._pcontext is not None:
            from ...multiprocessing.api import _get_default_signal

            self._pcontext.close(death_sig=_get_default_signal())
            self._pcontext = None

    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        if self._pcontext is None:
            return RunResult(state=WorkerState.FAILED)
        result = self._pcontext.wait(0)
        if result is None:
            return RunResult(state=WorkerState.HEALTHY)
        run_result = RunResult(
            state=WorkerState.SUCCEEDED if not result.failures else WorkerState.FAILED,
            failures=dict(result.failures),
            return_values=dict(result.return_values),
            stdouts=dict(result.stdouts),
            stderrs=dict(result.stderrs),
        )
        return run_result

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        import time

        spec = self._worker_group.spec
        role = spec.role
        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        while True:
            assert self._worker_group.state == WorkerState.HEALTHY
            result = self._monitor_workers(self._worker_group)
            if result.state == WorkerState.SUCCEEDED:
                logger.info("[%s] Worker group succeeded", role)
                try:
                    self._exit_barrier()
                except Exception:
                    logger.warning("Exit barrier timed out or failed", exc_info=True)
                return result
            elif result.state == WorkerState.FAILED:
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
            else:
                raise RuntimeError(f"Unexpected worker group state {result.state}")

    def _shutdown(self, death_sig=None, timeout: int = 30) -> None:
        from ...multiprocessing.api import _get_default_signal

        if self._pcontext is not None:
            self._pcontext.close(death_sig=death_sig or _get_default_signal(), timeout=timeout)
            self._pcontext = None
        super()._shutdown(death_sig, timeout=timeout)

