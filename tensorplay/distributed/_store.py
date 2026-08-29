"""Key-value stores for rendezvous and in-group coordination.

semantics: blocking ``get``, atomic ``add``, ``compare_set``, ``wait`` with
timeout. Pure Python by design so the rendezvous layer stays transparent.
"""

from __future__ import annotations

import fcntl
import os
import socket
import threading
import time
from typing import Optional


class StoreTimeoutError(RuntimeError):
    pass


class Store:
    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    def get(self, key: str, timeout: Optional[float] = None) -> bytes:
        raise NotImplementedError

    def add(self, key: str, amount: int) -> int:
        raise NotImplementedError

    def compare_set(self, key: str, expected: str, value: str) -> bytes:
        raise NotImplementedError

    def delete_key(self, key: str) -> None:
        raise NotImplementedError

    def wait(self, keys: list[str], timeout: Optional[float] = None) -> bool:
        deadline = time.monotonic() + (timeout if timeout is not None else 300.0)
        remaining = deadline - time.monotonic()
        while True:
            try:
                for key in keys:
                    self.get(key, timeout=max(remaining, 0.001))
                return True
            except StoreTimeoutError:
                if time.monotonic() >= deadline:
                    return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, max(remaining, 0.001)))


def _deadline(timeout: Optional[float]) -> float:
    return time.monotonic() + (timeout if timeout is not None else 300.0)


def _enc_key(key: str | bytes) -> bytes:
    return key.encode("utf-8") if isinstance(key, str) else key


def _enc_value(value: str | bytes) -> bytes:
    return value.encode("utf-8") if isinstance(value, str) else value


class FileStore(Store):
    """Flock-based append-log store in a single file."""

    def __init__(self, file_name: str, world_size: int = -1) -> None:
        self.path = file_name
        self.world_size = world_size
        parent = os.path.dirname(os.path.abspath(file_name))
        os.makedirs(parent, exist_ok=True)
        # Create the file eagerly so all ranks agree on the path.
        with open(self.path, "a"):
            pass

    def _locked_read(self) -> dict[str, list[bytes]]:
        data: dict[str, list[bytes]] = {}
        if not os.path.exists(self.path):
            return data
        with open(self.path, "rb") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.rstrip(b"\n")
                    if not line:
                        continue
                    key_b, _, value_b = line.partition(b"\t")
                    data.setdefault(key_b, []).append(value_b)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data

    def _locked_append(self, key: bytes, value: bytes) -> None:
        with open(self.path, "ab") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(key + b"\t" + value + b"\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def set(self, key, value) -> None:
        self._locked_append(_enc_key(key), _enc_value(value))

    def get(self, key, timeout: Optional[float] = None) -> bytes:
        key_b = _enc_key(key)
        end = _deadline(timeout)
        while True:
            entries = self._locked_read().get(key_b)
            if entries:
                return entries[-1]
            if time.monotonic() >= end:
                raise StoreTimeoutError(
                    f"FileStore: timed out waiting for key {key!r}"
                )
            time.sleep(0.01)

    def add(self, key, amount: int) -> int:
        key_b = _enc_key(key)
        with open(self.path, "a+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                current = 0
                if os.fstat(f.fileno()).st_size > 0:
                    f.seek(0)
                    for line in f:
                        k, _, v = line.rstrip(b"\n").partition(b"\t")
                        if k == key_b:
                            current = int(v)
                new_value = current + amount
                f.seek(0, os.SEEK_END)
                f.write(key_b + b"\t" + str(new_value).encode() + b"\n")
                f.flush()
                os.fsync(f.fileno())
                return new_value
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def compare_set(self, key, expected, value) -> bytes:
        key_b = _enc_key(key)
        expected_b = _enc_value(expected)
        value_b = _enc_value(value)
        with open(self.path, "a+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # A missing key compares equal to the empty string
                current = b""
                if os.fstat(f.fileno()).st_size > 0:
                    f.seek(0)
                    for line in f:
                        k, _, v = line.rstrip(b"\n").partition(b"\t")
                        if k == key_b:
                            current = v
                if current == expected_b:
                    f.seek(0, os.SEEK_END)
                    f.write(key_b + b"\t" + value_b + b"\n")
                    f.flush()
                    os.fsync(f.fileno())
                    return value_b
                return current
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def delete_key(self, key) -> None:
        self._locked_append(_enc_key(key), b"\x00__deleted__")


def _recv_line(sock: socket.socket) -> bytes:
    buf = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if b"\n" in buf:
            break
    line, _, rest = bytes(buf).partition(b"\n")
    if rest:
        raise RuntimeError("TCPStore protocol error: unexpected extra data")
    return line


class _TCPServer(threading.Thread):
    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self.data: dict[str, list[bytes]] = {}
        self.cond = threading.Condition()
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(128)
        self.port = self.sock.getsockname()[1]
        self.stopped = False

    def run(self) -> None:
        while not self.stopped:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def stop(self) -> None:
        self.stopped = True
        try:
            self.sock.close()
        except OSError:
            pass

    def _serve(self, conn: socket.socket) -> None:
        try:
            while True:
                line = _recv_line(conn)
                if not line:
                    return
                op, _, rest = line.decode("utf-8").partition(" ")
                if op == "SET":
                    key, _, value = rest.partition(" ")
                    with self.cond:
                        self.data[key] = [value.encode("utf-8")]
                        self.cond.notify_all()
                    conn.sendall(b"OK\n")
                elif op == "GET":
                    key = rest
                    deadline = time.monotonic() + 300.0
                    with self.cond:
                        while key not in self.data or not self.data[key]:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                conn.sendall(b"TIMEOUT\n")
                                return
                            self.cond.wait(min(remaining, 1.0))
                        value = self.data[key][-1]
                    conn.sendall(b"VALUE " + value.hex().encode() + b"\n")
                elif op == "ADD":
                    key, _, amount = rest.partition(" ")
                    with self.cond:
                        current = int(self.data[key][-1]) if self.data.get(key) else 0
                        new_value = current + int(amount)
                        self.data[key] = [str(new_value).encode()]
                        self.cond.notify_all()
                    conn.sendall(f"VALUE {new_value}\n".encode())
                elif op == "DEL":
                    with self.cond:
                        self.data.pop(rest, None)
                    conn.sendall(b"OK\n")
                elif op == "CHECK":
                    keys = rest.split(" ") if rest else []
                    with self.cond:
                        ok = all(self.data.get(k) for k in keys)
                    conn.sendall(b"READY\n" if ok else b"NOT_READY\n")
                else:
                    conn.sendall(b"ERR unknown op\n")
                    return
        except (OSError, ConnectionError):
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass


class TCPStore(Store):
    """
    subset: set/get/add/delete/check as used by rendezvous and barriers)."""

    def __init__(
        self,
        host_name: str,
        port: int = 0,
        world_size: int = -1,
        is_master: bool = False,
        timeout: float = 300.0,
        wait_for_workers: bool = True,
    ) -> None:
        self.host = host_name
        self.timeout = timeout
        self.world_size = world_size
        if is_master:
            self._server = _TCPServer(host_name, port)
            self._server.start()
            self.port = self._server.port
        else:
            self._server = None
            self.port = port

    @property
    def master_addr_port(self) -> tuple[str, int]:
        return self.host, self.port

    def _connect(self) -> socket.socket:
        last_err = None
        end = _deadline(self.timeout)
        while True:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                return sock
            except OSError as e:
                last_err = e
                if time.monotonic() >= end:
                    raise StoreTimeoutError(
                        f"TCPStore: could not connect to {self.host}:{self.port}: {last_err}"
                    )
                time.sleep(0.05)

    def _request(self, payload: str) -> bytes:
        with self._connect() as sock:
            sock.sendall(payload.encode("utf-8") + b"\n")
            return _recv_line(sock)

    def _key_str(self, key) -> str:
        return key.decode("utf-8") if isinstance(key, bytes) else key

    def set(self, key, value) -> None:
        resp = self._request(f"SET {self._key_str(key)} {self._key_str(value)}")
        if resp != b"OK":
            raise RuntimeError(f"TCPStore SET failed: {resp!r}")

    def get(self, key, timeout: Optional[float] = None) -> bytes:
        key = self._key_str(key)
        end = _deadline(timeout)
        while True:
            resp = self._request(f"GET {key}")
            if resp.startswith(b"VALUE "):
                return bytes.fromhex(resp[len(b"VALUE "):].decode())
            if time.monotonic() >= end:
                raise StoreTimeoutError(f"TCPStore: timed out waiting for key {key!r}")
            time.sleep(0.01)

    def add(self, key, amount: int) -> int:
        resp = self._request(f"ADD {self._key_str(key)} {amount}")
        if not resp.startswith(b"VALUE "):
            raise RuntimeError(f"TCPStore ADD failed: {resp!r}")
        return int(resp[len(b"VALUE "):])

    def compare_set(self, key, expected, value) -> bytes:
        # Not needed by our rendezvous; emulate via get+set (documented).
        current = self.get(key, timeout=0.001) if self._has(key) else b""
        if current == _enc_value(expected):
            self.set(key, value)
            return _enc_value(value)
        return current

    def _has(self, key: str) -> bool:
        try:
            self.get(key, timeout=0.001)
            return True
        except StoreTimeoutError:
            return False

    def delete_key(self, key) -> None:
        self._request(f"DEL {self._key_str(key)}")

    def wait(self, keys, timeout: Optional[float] = None) -> bool:
        end = _deadline(timeout if timeout is not None else self.timeout)
        keys = [self._key_str(k) for k in keys]
        while True:
            resp = self._request("CHECK " + " ".join(keys))
            if resp == b"READY":
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(0.05)
