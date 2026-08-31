"""Store bootstrap helpers for agent/worker startup."""
import socket
import time

from tensorplay.distributed import FileStore, Store, TCPStore

from .logging import get_logger

__all__ = ["create_core_store", "get_free_port", "get_socket_with_port"]

logger = get_logger(__name__)


def _now() -> float:
    return time.monotonic()


def _deadline(timeout: float) -> float:
    return _now() + timeout


def create_core_store(
    host_name: str,
    port: int,
    rank: int,
    world_size: int,
    timeout: float = 300,
    use_libuv: bool | None = None,
    retries: int = 0,
) -> Store:
    """Bootstrap the shared store for a worker group.

    Rank 0 hosts the TCPStore server; every other rank connects. ``retries``
    adds grace for slow bind/connect races on co-located agents. When
    ``host_name`` is a filesystem path the store falls back to a FileStore.
    """
    if host_name.startswith("file://"):
        path = host_name[len("file://") :]
        return FileStore(path, world_size)
    last_error = None
    attempts = max(0, retries) + 1
    for attempt in range(attempts):
        try:
            store = TCPStore(
                host_name,
                port,
                world_size=world_size,
                is_master=rank == 0,
                timeout=timeout,
                wait_for_workers=False,
            )
            _check_full_rank(store, world_size, timeout)
            return store
        except OSError as e:
            last_error = e
            logger.warning(
                "Store bootstrap attempt %s/%s failed: %s", attempt + 1, attempts, e
            )
    raise last_error if last_error else RuntimeError("Store bootstrap failed")


def _check_full_rank(store: Store, world_size: int, timeout: float) -> None:
    """Block until every rank has checked in, then reset the counter."""
    if world_size <= 1:
        return
    key = "tp_elastic/store_full_rank_check"
    num = store.add(key, 1)
    end = _deadline(timeout)
    while num < world_size:
        if _now() >= end:
            store.add(key, -num)
            raise TimeoutError(
                f"Only {num}/{world_size} ranks joined the store within {timeout}s"
            )
        num = int(store.get(key, timeout=1))
    if num == world_size:
        store.delete_key(key)


def get_free_port() -> int:
    """Reserve an ephemeral TCP port and release it for reuse."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def get_socket_with_port() -> socket.socket:
    """Create a socket already bound to an ephemeral port (caller closes)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    sock.listen(1)
    return sock
