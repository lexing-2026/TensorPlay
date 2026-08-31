"""Key/value stores for rendezvous and cross-process synchronization.

The authoritative implementations live in the C++ layer (bound under
``tensorplay._C._distributed``); the classes here are thin Python shims that
preserve the seconds-based timeout convention and translate expiries into
:class:`StoreTimeoutError`.
"""

from __future__ import annotations

from typing import Optional

from tensorplay._C import _distributed as _C

__all__ = [
    "Store",
    "StoreTimeoutError",
    "FileStore",
    "TCPStore",
    "PrefixStore",
    "HashStore",
]

_DEFAULT_TIMEOUT_SECONDS = 300.0


class StoreTimeoutError(RuntimeError):
    """Raised when a blocking lookup or rendezvous wait runs past its deadline."""


class Store:
    """Abstract store surface. Every value is an arbitrary byte string."""

    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    def get(self, key: str, timeout: Optional[float] = None) -> bytes:
        raise NotImplementedError

    def add(self, key: str, amount: int) -> int:
        raise NotImplementedError

    def compare_set(self, key: str, expected: str, value: str) -> bytes:
        raise NotImplementedError

    def compare_and_swap(
        self, key: str, expected: bytes, value: bytes
    ) -> tuple[bool, bytes]:
        raise NotImplementedError

    def has(self, key: str) -> bool:
        raise NotImplementedError

    def delete_key(self, key: str) -> None:
        raise NotImplementedError

    def wait(self, keys: list[str], timeout: Optional[float] = None) -> bool:
        raise NotImplementedError


def _timeout_ms(timeout: Optional[float]) -> int:
    seconds = _DEFAULT_TIMEOUT_SECONDS if timeout is None else float(timeout)
    return int(seconds * 1000)


def _rethrow_timeout(error: RuntimeError) -> RuntimeError:
    if "timed out" in str(error):
        return StoreTimeoutError(str(error))
    return error


def _key_str(key: "str | bytes") -> str:
    return key if isinstance(key, str) else key.decode("utf-8")


class _BytesStoreShim:
    """Coerces `str | bytes` inputs to bytes and converts timeout expiries."""

    def _as_bytes(self, value: "str | bytes") -> bytes:
        return value.encode("utf-8") if isinstance(value, str) else bytes(value)

    def get(self, key, timeout: Optional[float] = None) -> bytes:
        try:
            return _C.Store.get(self, _key_str(key), _timeout_ms(timeout))
        except RuntimeError as error:
            raise _rethrow_timeout(error) from error

    def set(self, key, value) -> None:
        _C.Store.set(self, _key_str(key), self._as_bytes(value))

    def add(self, key, amount: int) -> int:
        return _C.Store.add(self, _key_str(key), int(amount))

    def compare_set(self, key, expected, value) -> bytes:
        key_s = _key_str(key)
        expected_b = self._as_bytes(expected)
        value_b = self._as_bytes(value)
        swapped, current = self.compare_and_swap(key_s, expected_b, value_b)
        return value_b if swapped else current

    def compare_and_swap(self, key, expected: bytes, value: bytes):
        return _C.Store.compare_and_swap(
            self, _key_str(key), self._as_bytes(expected), self._as_bytes(value)
        )

    def has(self, key) -> bool:
        return _C.Store.has(self, _key_str(key))

    def delete_key(self, key) -> None:
        _C.Store.delete_key(self, _key_str(key))

    def wait(self, keys, timeout: Optional[float] = None) -> bool:
        return _C.Store.wait(
            self, [_key_str(k) for k in keys], _timeout_ms(timeout)
        )


class FileStore(_BytesStoreShim, _C.FileStore):
    """Flock-protected append-log store kept in a single file."""

    def __init__(self, file_name: str, world_size: int = -1) -> None:
        _C.FileStore.__init__(self, file_name)


class TCPStore(_BytesStoreShim, _C.TCPStore):
    """Client-server store over TCP; `is_master` starts the server thread."""

    def __init__(
        self,
        host_name: str,
        port: int = 0,
        world_size: int = -1,
        is_master: bool = False,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        wait_for_workers: bool = True,
    ) -> None:
        self._timeout = timeout
        _C.TCPStore.__init__(
            self,
            host_name,
            int(port),
            bool(is_master),
            _timeout_ms(timeout),
        )
        # The historical API exposes the bound endpoint as plain attributes.
        self.host = host_name
        self.port = _C.TCPStore.port(self)

    def stop(self) -> None:
        """Shuts the server thread down before garbage collection."""
        _C.TCPStore.stop(self)

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def master_addr_port(self) -> tuple[str, int]:
        return self.host, self.port


class PrefixStore(_BytesStoreShim, _C.PrefixStore):
    """Namespaces every key of the wrapped store behind a prefix."""

    def __init__(self, prefix: str, store) -> None:
        _C.PrefixStore.__init__(self, prefix, store)


class HashStore(_BytesStoreShim, _C.HashStore):
    """In-process memory store, one table per instance."""

    def __init__(self) -> None:
        _C.HashStore.__init__(self)
