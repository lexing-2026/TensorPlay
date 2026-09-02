#
# The agent-based elastic launcher. ``main`` is the console entry point
# (tensorrun-style); it parses CLI arguments into a
# ``tensorplay.distributed.launcher.LaunchConfig`` and delegates to the
# elastic agent, which handles rendezvous, worker monitoring, restarts, and
# scale-up/scale-down re-rendezvous. Single-node jobs run through the same
# path as multi-node ones.
#
import argparse
import os
import runpy
import sys
import warnings
from typing import Callable

from tensorplay.distributed.launcher import LaunchConfig, elastic_launch
from tensorplay.distributed.elastic.multiprocessing.redirects import Redirects, Std

__all__ = ["LaunchConfig", "elastic_launch", "run", "main"]


def determine_local_world_size(nproc_per_node: str | int) -> int:
    """Resolve ``auto``/``gpu``/``cpu``/int into the local worker count."""
    if isinstance(nproc_per_node, int):
        return nproc_per_node
    text = str(nproc_per_node).strip().lower()
    if text.isdigit():
        return int(text)
    if text == "cpu":
        return os.cpu_count() or 1
    if text == "gpu" or text == "auto":
        try:
            from tensorplay import cuda

            return cuda.device_count()
        except Exception:
            return 1
    raise ValueError(f"Cannot resolve nproc_per_node value: {nproc_per_node!r}")


def _parse_redirects(value: str | None) -> Std | dict[int, Std] | None:
    if value is None or value == "":
        return None
    return Redirects.from_str(value).stdouts


def _parse_tee(value: str | None) -> Std | dict[int, Std] | None:
    if value is None or value == "":
        return None
    return Redirects.from_str(value).stderrs


def config_from_args(args) -> tuple[LaunchConfig, Callable | str, list[str]]:
    """Translate parsed CLI arguments into a config plus entrypoint."""
    if args.nnodes:
        nnodes = str(args.nnodes)
        if ":" in nnodes:
            min_nodes, _, max_nodes = nnodes.partition(":")
            min_nodes = int(min_nodes)
            max_nodes = int(max_nodes) if max_nodes else min_nodes
        else:
            min_nodes = max_nodes = int(nnodes)
    else:
        min_nodes = max_nodes = 1
    if max_nodes < min_nodes:
        raise ValueError(f"max_nodes ({max_nodes}) must be >= min_nodes ({min_nodes})")

    nproc = determine_local_world_size(args.nproc_per_node)
    if nproc > 1 and "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"
        warnings.warn(
            f"Setting OMP_NUM_THREADS environment variable for each process to be "
            f"1: setting OMP_NUM_THREADS=1 when using {nproc} processes, "
            f"otherwise CPU contention may slow the job down.",
            UserWarning,
            stacklevel=2,
        )

    rdzv_endpoint = args.rdzv_endpoint
    if args.rdzv_backend == "static":
        endpoint = f"{args.master_addr}:{args.master_port}"
        if rdzv_endpoint:
            endpoint = rdzv_endpoint
    else:
        endpoint = rdzv_endpoint or "localhost:0"

    config = LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc,
        run_id=args.rdzv_id,
        role=args.role,
        rdzv_endpoint=endpoint,
        rdzv_backend=args.rdzv_backend,
        rdzv_configs=_parse_rdzv_conf(args.rdzv_conf),
        rdzv_timeout=args.rdzv_timeout,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
        start_method=args.start_method,
        log_dir=args.log_dir,
        redirects=_parse_redirects(args.redirects),
        tee=_parse_tee(args.tee),
        local_addr=args.local_addr,
        node_rank=args.node_rank,
        master_addr=args.master_addr if args.master_addr != "127.0.0.1" else None,
        master_port=args.master_port if args.master_port else None,
    )

    if args.run_path:
        entrypoint: Callable | str = _run_script
        cmd_args = [(args.training_script, *args.training_script_args)] * config.nproc_per_node
    elif args.module:
        entrypoint = sys.executable
        cmd_args = ["-u", "-m", args.training_script, *args.training_script_args]
    elif args.no_python:
        entrypoint = args.training_script
        cmd_args = list(args.training_script_args)
    else:
        entrypoint = sys.executable
        cmd_args = ["-u", args.training_script, *args.training_script_args]
    return config, entrypoint, cmd_args


def _parse_rdzv_conf(conf: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in filter(None, (conf or "").split(",")):
        key, sep, value = token.partition("=")
        if not sep:
            raise ValueError(f"Malformed rdzv_conf entry '{token}'; expected key=value")
        out[key.strip()] = value.strip()
    return out


def _run_script(script: str, *script_args: str) -> int:
    """Function entrypoint body for ``--run-path``: exec the script in-place."""
    sys.argv = [script, *script_args]
    runpy.run_path(script, run_name="__main__")
    return 0


def run(args=None):
    """Parse CLI arguments (when ``args`` is None) and launch the job.

    ``args`` may be a list of CLI strings or a pre-parsed namespace.
    """
    if args is None or isinstance(args, (list, tuple)):
        args = main_args(args)
    with warnings.catch_warnings(record=True) as caught_warnings:
        config, cmd, cmd_args = config_from_args(args)
    for w in caught_warnings:
        warnings.warn(w.message, w.category, stacklevel=2)
    if not args.run_path and not args.module:
        assert os.path.isfile(args.training_script), (
            f"{args.training_script} is not a valid file path"
        )
    elastic_launch(config=config, entrypoint=cmd)(*cmd_args)


def get_args_parser() -> argparse.ArgumentParser:
    """Argument parser mirroring the documented launcher CLI."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--nnodes",
        action="store",
        default="1:1",
        help="Number of nodes, either a plain count (e.g. 2) or a "
        "min:max range (e.g. 1:4) for elastic jobs.",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        action="store",
        default="auto",
        help="Number of workers per node; support auto/gpu/cpu/int.",
    )
    parser.add_argument(
        "--rdzv-backend", "--rdzv_backend", action="store", default="core",
        help="Rendezvous backend: core (elastic) or static (fixed world size).",
    )
    parser.add_argument(
        "--rdzv-endpoint", "--rdzv_endpoint", action="store", default="",
        help="Rendezvous endpoint host:port; defaults to localhost with an "
        "ephemeral port for single-node runs.",
    )
    parser.add_argument("--rdzv-id", "--rdzv_id", action="store", default="none")
    parser.add_argument(
        "--rdzv-conf", "--rdzv_conf", action="store", default="",
        help="Comma-separated backend options: key=value,key2=value2.",
    )
    parser.add_argument(
        "--rdzv-timeout", "--rdzv_timeout", action="store", default=900, type=int,
        help="Rendezvous join timeout in seconds.",
    )
    parser.add_argument("--max-restarts", "--max_restarts", action="store", default=0, type=int)
    parser.add_argument("--monitor-interval", "--monitor_interval",
                        action="store", default=0.1, type=float)
    parser.add_argument("--role", action="store", default="default_role")
    parser.add_argument("--log-dir", "--log_dir", action="store", default=None)
    parser.add_argument("--redirects", action="store", default=None,
                        help="Std redirection spec: 0/1/2/3 or per-rank map.")
    parser.add_argument("--tee", action="store", default=None,
                        help="Tee spec like redirects; duplicates worker output to the console.")
    parser.add_argument("--local-addr", "--local_addr", action="store", default=None,
                        help="Address of the local node advertised to peers.")
    parser.add_argument("--start-method", "--start_method", action="store", default="spawn",
                        help="Multiprocessing start method for function entrypoints.")
    parser.add_argument("--run-path", "--run_path", action="store_true",
                        help="Run the training script with runpy in workers.")
    parser.add_argument("--master-addr", "--master_addr", action="store", default="127.0.0.1")
    parser.add_argument("--master-port", "--master_port", action="store", default="0", type=int)
    parser.add_argument("-m", "--module", action="store_true",
                        help="Interpret the launch script as a python module.")
    parser.add_argument("--no-python", "--no_python", action="store_true",
                        help="Skip prepending the interpreter to the script command.")
    parser.add_argument("--node-rank", "--node_rank", action="store", default=0, type=int,
                        help="Rank of this node among static nodes.")
    parser.add_argument("training_script", type=str)
    parser.add_argument("training_script_args", nargs="*")
    return parser


def main_args(args=None):
    return get_args_parser().parse_args(args)


def main(args=None):
    """Console entry point."""
    args = main_args(args)
    run(args)


if __name__ == "__main__":
    main()
