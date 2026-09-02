"""Launcher configuration and the agent entry point.

``LaunchConfig`` captures everything needed to run an elastic job on this
node: rendezvous settings, restart policy, log routing, and the worker
count. :class:`elastic_launch` turns it into a running job, and
:func:`launch_agent` is the programmatic entry behind it.
"""
import logging
import os
import socket
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

from tensorplay.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from tensorplay.distributed.elastic.agent.server.api import WorkerSpec
from tensorplay.distributed.elastic.multiprocessing.errors import (
    ChildFailedError,
    record,
)
from tensorplay.distributed.elastic.rendezvous import (
    RendezvousParameters,
    create_handler,
)
from tensorplay.distributed.elastic.utils.logging import get_logger

__all__ = ["LaunchConfig", "elastic_launch", "launch_agent"]

logger = get_logger(__name__)


@dataclass
class LaunchConfig:
    """Elastic launch settings for one job.

    ``min_nodes``/``max_nodes`` bound the job size; when they differ the job
    is elastic and nodes may join or leave within those bounds. ``rdzv_*``
    fields select and configure the rendezvous backend.
    """

    min_nodes: int = 1
    max_nodes: int = 1
    nproc_per_node: int = 1
    run_id: str = ""
    role: str = "default_role"
    rdzv_endpoint: str = ""
    rdzv_backend: str = "static"
    rdzv_configs: dict[str, Any] = field(default_factory=dict)
    rdzv_timeout: int = -1
    max_restarts: int = 0
    monitor_interval: float = 0.1
    start_method: str = "spawn"
    log_dir: str | None = None
    redirects: Any = None
    tee: Any = None
    metrics_cfg: dict[str, str] = field(default_factory=dict)
    local_addr: str | None = None
    node_rank: int = 0
    master_addr: str | None = None
    master_port: int | None = None

    def __post_init__(self) -> None:
        if self.max_nodes < self.min_nodes:
            raise ValueError(
                f"max_nodes ({self.max_nodes}) must be >= min_nodes ({self.min_nodes})"
            )
        if self.nproc_per_node <= 0:
            raise ValueError(f"nproc_per_node ({self.nproc_per_node}) must be positive")
        if self.min_nodes <= 0:
            raise ValueError(f"min_nodes ({self.min_nodes}) must be positive")
        if self.max_restarts < 0:
            raise ValueError(f"max_restarts ({self.max_restarts}) must be non-negative")
        if self.monitor_interval < 0:
            raise ValueError("monitor_interval must be non-negative")
        if self.node_rank < 0:
            raise ValueError(f"node_rank ({self.node_rank}) must be non-negative")
        if self.node_rank >= self.max_nodes:
            raise ValueError(
                f"node_rank ({self.node_rank}) must be smaller than max_nodes ({self.max_nodes})"
            )
        if not self.run_id:
            self.run_id = "none"


def _get_entrypoint_name(entrypoint: Callable | str | None, args: list[Any]) -> str:
    if entrypoint is None:
        return "None"
    if isinstance(entrypoint, str):
        return os.path.basename(entrypoint)
    return getattr(entrypoint, "__qualname__", str(entrypoint))


def _setup_logs(config: LaunchConfig) -> None:
    """Apply ``config.log_dir`` to agent and workers; default to stdout logs."""
    if config.log_dir:
        os.makedirs(config.log_dir, exist_ok=True)
        log_handler = logging.FileHandler(os.path.join(config.log_dir, "agent.log"))
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logging.getLogger("tensorplay.distributed.elastic").addHandler(log_handler)


def _get_addr_and_port(
    rdzv_handler,
    port: int | None = None,
) -> tuple[str, int]:
    """Reserve a free address/port for the rendezvous endpoint."""
    addr = "127.0.0.1"
    if port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = int(sock.getsockname()[1])
    return addr, port


def _make_rdzv_handler(config: LaunchConfig):
    """Build the rendezvous handler for ``config``."""
    endpoint = config.rdzv_endpoint
    if config.rdzv_backend == "static" and not endpoint:
        addr, port = _get_addr_and_port(None, config.master_port)
        endpoint = f"{config.master_addr or addr}:{port}"
    rdzv_configs = dict(config.rdzv_configs)
    rdvz_timeout = config.rdzv_timeout
    if rdvz_timeout > 0:
        rdzv_configs.setdefault("join_timeout", rdvz_timeout)
    rdzv_configs.setdefault("min_nodes", config.min_nodes)
    rdzv_configs.setdefault("max_nodes", config.max_nodes)
    if config.rdzv_backend == "static":
        rdzv_configs.setdefault("rank", config.node_rank)
    params = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=endpoint,
        run_id=config.run_id,
        local_addr=config.local_addr,
        node_rank=config.node_rank,
        local_world_size=config.nproc_per_node,
        config=rdzv_configs,
    )
    return create_handler(params)


@record
def launch_agent(
    config: LaunchConfig,
    entrypoint: Callable | str | None,
    args: list[Any],
) -> Any:
    """Start the local agent for ``entrypoint`` and run it to completion.

    Raises :class:`ChildFailedError` when any worker fails after exhausting
    the restart budget.
    """
    _setup_logs(config)
    rdzv_handler = _make_rdzv_handler(config)
    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        rdzv_handler=rdzv_handler,
        entrypoint=entrypoint,
        args=tuple(args),
        max_restarts=config.max_restarts,
        monitor_interval=config.monitor_interval,
        master_addr=config.master_addr,
        master_port=config.master_port,
        local_addr=config.local_addr,
        logs_specs=None,
        start_method=config.start_method,
        redirects=config.redirects,
        tee=config.tee,
        log_dir=config.log_dir,
    )
    agent = LocalElasticAgent(
        spec=spec,
        log_dir=config.log_dir,
    )
    try:
        result = agent.run()
    except ChildFailedError:
        raise
    except Exception as e:
        if rdzv_handler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    rdzv_handler.shutdown()
                except Exception:
                    pass
        raise
    if result.is_failed():
        raise ChildFailedError(
            [(config.role, failure) for failure in result.failures.values()]
        )
    return result.return_values


class elastic_launch:
    """Callable wrapper around :func:`launch_agent`.

    Usage::

        elastic_launch(config=LaunchConfig(...), entrypoint="train.py")("--epoch", "10")
    """

    def __init__(
        self,
        config: LaunchConfig,
        entrypoint: Callable | str | None,
    ) -> None:
        self._config = config
        self._entrypoint = entrypoint

    def __call__(self, *args: Any) -> Any:
        return launch_agent(self._config, self._entrypoint, list(args))
