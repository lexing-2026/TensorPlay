"""Store-backed synchronization primitives for agent-side coordination.

All routines expect the shared key/value store contract of
``tensorplay.distributed``: blocking ``get`` with timeout, ``add`` returning
the new value, and key-scoped namespacing via string prefixes.
"""
import contextlib
import time
from datetime import timedelta

from tensorplay.distributed import Store, StoreTimeoutError

from .logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT: timedelta = timedelta(seconds=600)


def _to_seconds(timeout: float | timedelta | None) -> float:
    if timeout is None:
        return DEFAULT_TIMEOUT.total_seconds()
    if isinstance(timeout, timedelta):
        return timeout.total_seconds()
    return float(timeout)


@contextlib.contextmanager
def store_timeout(store: Store, timeout: float | timedelta | None = None):
    """Temporarily override the store's blocking-call timeout."""
    seconds = _to_seconds(timeout)
    prev = getattr(store, "timeout", None)
    try:
        store.timeout = seconds
        yield store
    finally:
        if prev is not None:
            store.timeout = prev


def get_all(
    store: Store, rank: int, prefix: str, world_size: int, timeout: float = 600
) -> list[bytes]:
    """Fetch ``[prefix + str(i) for i in range(world_size)]`` blocking until all keys exist."""
    if rank >= world_size:
        raise ValueError(f"rank {rank} is out of range for world size {world_size}")
    keys = [f"{prefix}{i}" for i in range(world_size)]
    store.wait(keys, timeout=timeout)
    return [store.get(key) for key in keys]


def _barrier_nonblocking(store: Store, world_size: int, key_prefix: str) -> str:
    key = f"{key_prefix}/num_members"
    last_member = f"{key_prefix}/last_member"
    if store.add(key, 1) == world_size:
        store.set(last_member, b"1")
    return last_member


def synchronize(
    store: Store,
    data: bytes,
    rank: int,
    world_size: int,
    key_prefix: str,
    timeout: float = 600,
) -> list[bytes]:
    """Exchange ``data`` among ``world_size`` peers; returns data of all ranks.

    Each rank publishes its payload under ``key_prefix + rank`` then blocks
    until every peer has published, which makes the call an all-gather with
    barrier semantics.
    """
    store.set(f"{key_prefix}{rank}", data)
    return get_all(store, rank, key_prefix, world_size, timeout=timeout)


def _try_detecting_missing_ranks(
    store: Store, world_size: int, key_prefix: str, timeout: float
) -> list[int]:
    """Best-effort detection of ranks that never checked in, for diagnostics."""
    missing: list[int] = []
    for i in range(world_size):
        try:
            store.get(f"{key_prefix}{i}", timeout=timeout)
        except StoreTimeoutError:
            missing.append(i)
    return missing


@contextlib.contextmanager
def barrier(
    store: Store,
    world_size: int,
    key_prefix: str,
    timeout: float | timedelta | None = None,
):
    """Context-manager barrier over the store.

    On entry each participant increments the arrival counter under
    ``key_prefix`` and blocks until all ``world_size`` ranks have arrived.
    On exit the counters are reset so the prefix can be reused for the next
    barrier. If the barrier times out, a best-effort scan reports which
    ranks never arrived.
    """
    seconds = _to_seconds(timeout)
    arrived_key = f"{key_prefix}ARRIVED"
    done_key = f"{key_prefix}DONE"
    departed_key = f"{key_prefix}DEPARTED"
    try:
        num_arrived = store.add(arrived_key, 1)
        if num_arrived == world_size:
            store.set(done_key, "1")
        elif num_arrived > world_size:
            raise RuntimeError(
                f"Barrier under {key_prefix} counted {num_arrived} arrivals "
                f"for world size {world_size}"
            )
        else:
            end = time.monotonic() + seconds
            while True:
                try:
                    store.wait([done_key], timeout=0.5)
                    break
                except StoreTimeoutError:
                    pass
                if time.monotonic() >= end:
                    missing = _try_detecting_missing_ranks(
                        store, world_size, key_prefix, timeout=seconds
                    )
                    raise StoreTimeoutError(
                        f"Barrier under {key_prefix} timed out after {seconds}s; "
                        f"ranks that never arrived: {missing}"
                    )
        yield
    finally:
        # Cleanup is best-effort: peers may have torn the shared store down
        # already once everyone passed the barrier.
        try:
            num_departed = store.add(departed_key, 1)
            if num_departed == world_size:
                store.delete_key(arrived_key)
                store.delete_key(done_key)
                store.delete_key(departed_key)
        except Exception:
            pass
