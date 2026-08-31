from __future__ import annotations

import os

from tensorplay.distributed import FileStore, Store, TCPStore

__all__ = ["get_rank", "get_world_size", "tcpstore_client"]


class _PrefixStore(Store):
    def __init__(self, prefix: str, store: Store) -> None:
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.store = store

    def _key(self, key) -> str:
        text = key.decode() if isinstance(key, bytes) else str(key)
        return self.prefix + text

    def set(self, key, value) -> None:
        self.store.set(self._key(key), value)

    def get(self, key, timeout=None) -> bytes:
        return self.store.get(self._key(key), timeout=timeout)

    def add(self, key, amount: int) -> int:
        return self.store.add(self._key(key), amount)

    def compare_set(self, key, expected, value) -> bytes:
        return self.store.compare_set(self._key(key), expected, value)

    def compare_and_swap(self, key, expected: bytes, value: bytes):
        return self.store.compare_and_swap(self._key(key), expected, value)

    def has(self, key) -> bool:
        return self.store.has(self._key(key))

    def delete_key(self, key) -> None:
        self.store.delete_key(self._key(key))

    def wait(self, keys, timeout=None) -> bool:
        return self.store.wait([self._key(key) for key in keys], timeout=timeout)


def get_rank() -> int:
    value = os.environ.get("RANK")
    if value is not None:
        return int(value)
    try:
        from tensorplay.distributed import distributed_core as dist

        return int(dist.get_rank())
    except Exception:
        return 0


def get_world_size() -> int:
    value = os.environ.get("WORLD_SIZE")
    if value is not None:
        return int(value)
    try:
        from tensorplay.distributed import distributed_core as dist

        return int(dist.get_world_size())
    except Exception:
        return 1


def tcpstore_client(prefix: str = "debug_server") -> Store:
    host = os.environ.get("MASTER_ADDR")
    port = os.environ.get("MASTER_PORT")
    if host and port:
        store: Store = TCPStore(host, int(port), world_size=get_world_size(), is_master=False)
    else:
        path = os.environ.get("TP_DEBUG_STORE", os.path.join(os.getcwd(), ".tp_debug_store"))
        store = FileStore(path, world_size=get_world_size())
    return _PrefixStore(prefix, store) if prefix else store
