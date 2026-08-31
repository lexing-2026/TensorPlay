from __future__ import annotations

import datetime
import random
import time
from base64 import b64decode, b64encode

from tensorplay.distributed import Store, StoreTimeoutError

try:
    import etcd as _etcd
except ImportError:
    from . import _etcd_stub as _etcd

__all__ = ["EtcdStore", "cas_delay"]


def cas_delay() -> None:
    time.sleep(random.uniform(0.001, 0.02))


class EtcdStore(Store):
    def __init__(self, etcd_client, etcd_store_prefix: str, timeout: datetime.timedelta | None = None):
        self.client = etcd_client
        self.prefix = etcd_store_prefix.rstrip("/") + "/"
        self.timeout = timeout or datetime.timedelta(seconds=300)

    def _encode(self, value) -> str:
        if isinstance(value, bytes):
            return b64encode(value).decode()
        if isinstance(value, str):
            return b64encode(value.encode()).decode()
        raise ValueError("value must be str or bytes")

    def _decode(self, value) -> bytes:
        if isinstance(value, bytes):
            return b64decode(value)
        if isinstance(value, str):
            return b64decode(value.encode())
        raise ValueError("value must be str or bytes")

    def set(self, key, value) -> None:
        self.client.write(self.prefix + self._encode(key), self._encode(value))

    def get(self, key, timeout=None) -> bytes:
        deadline = time.monotonic() + (self.timeout if timeout is None else datetime.timedelta(seconds=float(timeout))).total_seconds()
        encoded_key = self.prefix + self._encode(key)
        while True:
            try:
                return self._decode(self.client.read(encoded_key).value)
            except _etcd.EtcdKeyNotFound:
                if time.monotonic() >= deadline:
                    raise StoreTimeoutError(f"key {key!r} was not published")
                time.sleep(0.01)

    def add(self, key, num: int) -> int:
        encoded_key = self.prefix + self._encode(key)
        while True:
            try:
                node = self.client.read(encoded_key)
                current = int(self._decode(node.value))
                new_value = self._encode(str(current + int(num)))
                result = self.client.test_and_set(encoded_key, new_value, node.value)
                return int(self._decode(result.value))
            except _etcd.EtcdKeyNotFound:
                try:
                    result = self.client.write(encoded_key, self._encode(str(num)), prevExist=False)
                    return int(self._decode(result.value))
                except _etcd.EtcdAlreadyExist:
                    pass
            except _etcd.EtcdCompareFailed:
                cas_delay()

    def wait(self, keys, override_timeout: datetime.timedelta | None = None) -> None:
        timeout = self.timeout if override_timeout is None else override_timeout
        deadline = time.monotonic() + timeout.total_seconds()
        for key in keys:
            remaining = max(0.0, deadline - time.monotonic())
            self.get(key, timeout=remaining)

    def check(self, keys) -> bool:
        try:
            for key in keys:
                self.get(key, timeout=0.001)
            return True
        except (StoreTimeoutError, LookupError):
            return False

    def _try_wait_get(self, keys, override_timeout=None):
        try:
            return {key: self.client.read(key).value for key in keys}
        except _etcd.EtcdKeyNotFound:
            return None
