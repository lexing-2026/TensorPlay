from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


class EtcdStubError(ImportError):
    pass


class EtcdAlreadyExist(Exception):
    pass


class EtcdCompareFailed(Exception):
    pass


class EtcdKeyNotFound(Exception):
    pass


class EtcdWatchTimedOut(Exception):
    pass


class EtcdEventIndexCleared(Exception):
    pass


class EtcdException(Exception):
    pass


@dataclass
class EtcdResult:
    key: str
    value: str | None
    etcd_index: int
    modifiedIndex: int
    createdIndex: int
    dir: bool = False
    ttl: int | None = None
    children: list["EtcdResult"] | None = None

    @property
    def value_bytes(self) -> bytes:
        return (self.value or "").encode()


@dataclass
class _Entry:
    value: str | None
    created: int
    modified: int
    directory: bool
    expires: float | None
    ttl: int | None


class _Database:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.entries: dict[str, _Entry] = {}
        self.index = 0

    def purge(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self.entries.items() if entry.expires is not None and entry.expires <= now]
        for key in expired:
            self.entries.pop(key, None)
        if expired:
            self.index += 1
            self.cond.notify_all()


_DATABASES: dict[tuple[str, int], _Database] = {}
_DATABASES_LOCK = threading.Lock()


def _database(host: str, port: int) -> _Database:
    key = (host, int(port))
    with _DATABASES_LOCK:
        return _DATABASES.setdefault(key, _Database())


class Client:
    def __init__(self, host: str = "localhost", port: int = 2379, **kwargs: Any) -> None:
        self.host = host
        self.port = int(port)
        self.read_timeout = float(kwargs.get("read_timeout", 60.0))
        self.machines = [f"{host}:{port}"]
        self._db = _database(host, self.port)

    @property
    def version(self) -> str:
        return "tp-etcd-1"

    def _result(self, key: str, entry: _Entry) -> EtcdResult:
        return EtcdResult(key, entry.value, self._db.index, entry.modified, entry.created, entry.directory, entry.ttl)

    def read(self, key: str) -> EtcdResult:
        return self.get(key)

    def get(self, key: str) -> EtcdResult:
        with self._db.lock:
            self._db.purge()
            entry = self._db.entries.get(key)
            if entry is not None:
                return self._result(key, entry)
            prefix = key.rstrip("/") + "/"
            children = [self._result(child_key, child) for child_key, child in self._db.entries.items() if child_key.startswith(prefix) and "/" not in child_key[len(prefix):].rstrip("/")]
            if children:
                return EtcdResult(key, None, self._db.index, self._db.index, self._db.index, True, children=children)
        raise EtcdKeyNotFound(key)

    def _write_locked(self, key: str, value: Any, ttl: int | None, directory: bool, previous: _Entry | None = None) -> EtcdResult:
        self._db.index += 1
        now_index = self._db.index
        entry = _Entry(None if value is None else str(value), previous.created if previous else now_index, now_index, directory, time.monotonic() + ttl if ttl else None, ttl)
        self._db.entries[key] = entry
        self._db.cond.notify_all()
        return self._result(key, entry)

    def write(self, key: str, value: Any = None, ttl: int | None = None, **kwargs: Any) -> EtcdResult:
        with self._db.lock:
            self._db.purge()
            current = self._db.entries.get(key)
            if kwargs.get("prevExist") is False and current is not None:
                raise EtcdAlreadyExist(key)
            if kwargs.get("prevExist") is True and current is None:
                raise EtcdKeyNotFound(key)
            prev_index = kwargs.get("prevIndex")
            if prev_index is not None and (current is None or current.modified != int(prev_index)):
                raise EtcdCompareFailed(key)
            if kwargs.get("dir"):
                value = None
            return self._write_locked(key, value, ttl, bool(kwargs.get("dir")), current)

    def set(self, key: str, value: Any = None, ttl: int | None = None, **kwargs: Any) -> EtcdResult:
        return self.write(key, value, ttl=ttl, **kwargs)

    def update(self, result: EtcdResult) -> EtcdResult:
        return self.write(result.key, result.value, ttl=result.ttl, prevIndex=result.modifiedIndex)

    def test_and_set(self, key: str, value: Any, prev_value: Any, ttl: int | None = None) -> EtcdResult:
        with self._db.lock:
            self._db.purge()
            current = self._db.entries.get(key)
            current_value = None if current is None else current.value
            if current is None or str(current_value) != str(prev_value):
                raise EtcdCompareFailed(key)
            return self._write_locked(key, value, ttl, current.directory, current)

    def refresh(self, key: str, ttl: int | None = None) -> EtcdResult:
        with self._db.lock:
            entry = self._db.entries.get(key)
            if entry is None:
                raise EtcdKeyNotFound(key)
            self._db.index += 1
            entry.expires = time.monotonic() + ttl if ttl else entry.expires
            entry.ttl = ttl or entry.ttl
            entry.modified = self._db.index
            self._db.cond.notify_all()
            return self._result(key, entry)

    def watch(self, key: str, index: int = 0, timeout: float | None = None, **kwargs: Any) -> EtcdResult:
        end = time.monotonic() + (self.read_timeout if timeout is None else max(0.0, float(timeout)))
        with self._db.cond:
            while True:
                self._db.purge()
                if self._db.index > int(index):
                    try:
                        return self.get(key)
                    except EtcdKeyNotFound:
                        return EtcdResult(key, None, self._db.index, self._db.index, self._db.index)
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise EtcdWatchTimedOut(key)
                self._db.cond.wait(min(remaining, 0.25))
