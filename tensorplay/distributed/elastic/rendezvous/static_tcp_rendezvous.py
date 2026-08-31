"""Fixed-size rendezvous over a pre-agreed store endpoint.

All participants agree on ``host:port`` out of band; the node with
``node_rank == 0`` hosts the store server and everyone else connects. The
resulting ranks mirror the node ranks, which makes this backend suitable for
homogeneous, statically-sized jobs.
"""
from tensorplay.distributed import TCPStore

from .api import (
    RendezvousConnectionError,
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStoreInfo,
)
from .utils import parse_rendezvous_endpoint


class StaticTCPRendezvous(RendezvousHandler):
    """One-shot rendezvous over a static TCPStore endpoint."""

    def __init__(self, params: RendezvousParameters) -> None:
        self._params = params
        self._endpoint = params.endpoint
        self._run_id = params.run_id
        host, port = parse_rendezvous_endpoint(self._endpoint, default_port=29400)
        if not self._endpoint:
            port = 0
        self._host = host
        self._port = port
        self._num_nodes = params.get_as_int("max_nodes", 1)
        join_timeout = params.timeout.join
        timeout = params.get_as_int("read_timeout", int(join_timeout.total_seconds()))
        is_master = params.node_rank == 0 and params.get_as_bool("start_daemon", True)
        try:
            self._store = TCPStore(
                self._host,
                self._port,
                world_size=self._num_nodes,
                is_master=is_master,
                timeout=float(timeout),
                wait_for_workers=False,
            )
        except OSError as e:
            raise RendezvousConnectionError(
                f"Failed to bind or connect to the static rendezvous endpoint "
                f"{self._host}:{self._port}: {e}"
            ) from e

    def get_backend(self) -> str:
        return "static"

    def use_agent_store(self) -> bool:
        # The store lives in the agent process of node 0.
        return True

    def next_rendezvous(self) -> RendezvousInfo:
        bootstrap = RendezvousStoreInfo.build(
            rank=self._params.node_rank,
            store=self._store,
            local_addr=self._params.local_addr,
        )
        return RendezvousInfo(
            store=self._store,
            rank=self._params.node_rank,
            world_size=self._num_nodes,
            bootstrap_store_info=bootstrap,
        )

    def is_closed(self) -> bool:
        return False

    def set_closed(self) -> None:
        return

    def num_nodes_waiting(self) -> int:
        return 0

    def get_run_id(self) -> str:
        return self._run_id

    def shutdown(self) -> bool:
        return False


def create_handler(params: RendezvousParameters) -> StaticTCPRendezvous:
    """Registry entry point for the ``static`` backend."""
    return StaticTCPRendezvous(params)
