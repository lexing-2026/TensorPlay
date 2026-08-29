#
# Adaptation: tp implements a single-node elastic launcher backed by worker
# subprocesses and the pure-Python env:// rendezvous (TCPStore hosted by
# local rank 0). Multi-node rendezvous backends (c10d/etcd agents) are part
import argparse
import os
import socket
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable


__all__ = ["LaunchConfig", "elastic_launch", "run", "main"]


def _get_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


@dataclass
class LaunchConfig:

    min_nodes: int = 1
    max_nodes: int = 1
    nproc_per_node: int = 1
    run_id: str = "none"
    role: str = "default_role"
    max_restarts: int = 0
    monitor_interval: float = 0.1
    start_method: str = "subprocess"
    redirects: Any = field(default_factory=lambda: {})
    tee: Any = field(default_factory=lambda: {})
    log_dir: str | None = None
    rdzv_backend: str = "static"
    rdzv_endpoint: str = ""
    rdzv_configs: dict[str, Any] = field(default_factory=dict)
    rdzv_timeout: float = 900
    metrics_cfg: dict[str, Any] = field(default_factory=dict)
    local_addr: str | None = None

    @staticmethod
    def from_args(args_dict: dict[str, Any]) -> "LaunchConfig":
        return LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=int(args_dict.get("nproc_per_node", 1)),
            rdzv_backend=args_dict.get("rdzv_backend", "static"),
            run_id=args_dict.get("run_id", "none"),
            role=args_dict.get("role", "default_role"),
            max_restarts=args_dict.get("max_restarts", 0),
            monitor_interval=args_dict.get("monitor_interval", 0.1),
            start_method="subprocess",
            log_dir=args_dict.get("log_dir"),
            redirects=args_dict.get("redirects", {}),
            tee=args_dict.get("tee", {}),
            metrics_cfg=args_dict.get("metrics_cfg", {}),
            local_addr=getattr(args_dict, "local_addr", None),
        )


class elastic_launch:

    def __init__(self, config: LaunchConfig, entrypoint: Callable | str | None):
        self._config = config
        self._entrypoint = entrypoint

    def __call__(self, *args):
        return _launch_local(self._config, self._entrypoint, args)


def determine_local_world_size(nproc_per_node: str | int) -> int:
    if isinstance(nproc_per_node, int):
        return nproc_per_node
    if isinstance(nproc_per_node, str):
        if nproc_per_node == "gpu":
            import tensorplay as tp

            if not tp.cuda.is_available():
                raise ValueError(
                    'tp.distributed.run uses nproc_per_node="gpu" but CUDA is '
                    "not available."
                )
            return tp.cuda.device_count()
        if nproc_per_node == "auto":
            num_proc = len(os.sched_getaffinity(0))
            return num_proc
        return int(nproc_per_node)
    raise ValueError(f"Unsupported nproc_per_node value: {nproc_per_node}")


def config_from_args(args) -> tuple[LaunchConfig, Callable | str, list[str]]:
    config = LaunchConfig.from_args(vars(args)) if hasattr(args, "__dict__") \
        else LaunchConfig(nproc_per_node=determine_local_world_size(args))
    entrypoint = args.training_script
    cmd_args = list(args.training_script_args)
    return config, entrypoint, cmd_args


def _launch_local(config: LaunchConfig, entrypoint, script_args):
    """Spawn ``nproc_per_node`` workers joined by an env:// rendezvous."""
    import datetime as dt

    from tensorplay.distributed.rendezvous import TCPStore

    nproc = determine_local_world_size(config.nproc_per_node)
    master_addr = "127.0.0.1"
    master_port = (
        int(config.rdzv_configs.get("port", 0)) or _get_free_port()
    )
    store = TCPStore(master_addr, master_port, nproc, is_master=True,
                     timeout=300.0)
    port = store.port

    procs: list[subprocess.Popen] = []
    try:
        for local_rank in range(nproc):
            env = dict(os.environ)
            env.update({
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(nproc),
                "LOCAL_WORLD_SIZE": str(nproc),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(port),
                "TORCHELASTIC_RUN_ID": config.run_id,
                # Rank 0 hosts the TCPStore server.
                "TP_START_DAEMON": "1" if local_rank == 0 else "0",
            })
            if entrypoint == sys.executable or (
                isinstance(entrypoint, str) and entrypoint.endswith(".py")
            ):
                cmd = [sys.executable, "-u", entrypoint, *map(str, script_args)]
            else:
                cmd = [entrypoint, *map(str, script_args)] if isinstance(
                    entrypoint, str) else None
            if cmd is None:
                # Function entrypoint: execute via tp.multiprocessing-style
                # pickling in child processes is not supported by the
                # subprocess launcher; instruct users accordingly.
                raise ValueError(
                    "elastic_launch(function) requires a spawn-based agent "
                    "(pending); use a script path as the entrypoint."
                )
            procs.append(subprocess.Popen(cmd, env=env))

        failed = []
        for local_rank, proc in enumerate(procs):
            ret = proc.wait()
            if ret != 0:
                failed.append((local_rank, ret))
        if failed:
            local_rank, ret = failed[0]
            raise RuntimeError(
                f"Worker rank {local_rank} exited with error code {ret}."
            )
        return [None] * nproc
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()


def run(args=None):
    args = main_args(args)
    with warnings.catch_warnings(record=True) as caught_warnings:
        config, cmd, cmd_args = config_from_args(args)
    for w in caught_warnings:
        warnings.warn(w.message)
    if not args.run_path:
        assert os.path.isfile(cmd), f"{cmd} is not a valid file path"
    elastic_launch(config=config, entrypoint=cmd)(*cmd_args)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        action="store",
        default="auto",
        help="Number of workers per node; support auto/gpu/int.",
    )
    parser.add_argument(
        "--master-addr", "--master_addr", action="store", default="127.0.0.1"
    )
    parser.add_argument("--master-port", "--master_port", action="store", default="0")
    parser.add_argument("--run-path", "--run_path", action="store_true")
    parser.add_argument("--rdzv-backend", "--rdzv_backend", action="store",
                        default="static")
    parser.add_argument("--rdzv-endpoint", "--rdzv_endpoint", action="store",
                        default="")
    parser.add_argument("--rdzv-id", "--rdzv_id", action="store", default="none")
    parser.add_argument("--max-restarts", "--max_restarts", action="store",
                        default=0)
    parser.add_argument("--monitor-interval", "--monitor_interval",
                        action="store", default=0.1)
    parser.add_argument("--role", action="store", default="default_role")
    parser.add_argument("-m", "--module", action="store_true",
                        help="Change each process to interpret the launch "
                        "script as a python module.")
    parser.add_argument("--no-python", "--no_python", action="store_true",
                        help="Skip prepending the training script with python.")
    parser.add_argument("training_script", type=str)
    parser.add_argument("training_script_args", nargs="*")
    return parser


def main_args(args=None):
    return get_args_parser().parse_args(args)


def main(args=None):
    args = main_args(args)
    run(args)
