from __future__ import annotations

import atexit
import os
import shutil
import socket
import subprocess
import tempfile
import time

from . import _etcd_stub

__all__ = ["EtcdServer", "find_free_port", "stop_etcd"]


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 0))
    sock.listen(1)
    return sock


def stop_etcd(process, data_dir: str | None = None):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if data_dir:
        shutil.rmtree(data_dir, ignore_errors=True)


class EtcdServer:
    def __init__(self, data_dir: str | None = None):
        self._host = "localhost"
        self._port = -1
        self._base_data_dir = data_dir or tempfile.mkdtemp(prefix="tp_etcd_data_")
        self._etcd_binary_path = os.environ.get("TORCHELASTIC_ETCD_BINARY_PATH", "etcd")
        self._etcd_proc: subprocess.Popen | None = None
        self._started = False

    def _get_etcd_server_process(self):
        if self._etcd_proc is None:
            raise RuntimeError("etcd server is not running")
        return self._etcd_proc

    def get_port(self) -> int:
        return self._port

    def get_host(self) -> str:
        return self._host

    def get_endpoint(self) -> str:
        return f"{self._host}:{self._port}"

    def start(self, timeout: int = 60, num_retries: int = 3, stderr=None) -> None:
        del timeout, num_retries, stderr
        if self._started:
            return
        sock = find_free_port()
        self._port = sock.getsockname()[1]
        sock.close()
        binary = shutil.which(self._etcd_binary_path) if not os.path.isfile(self._etcd_binary_path) else self._etcd_binary_path
        if binary:
            data_dir = os.path.join(self._base_data_dir, "0")
            os.makedirs(data_dir, exist_ok=True)
            self._etcd_proc = subprocess.Popen(
                [binary, "--data-dir", data_dir, "--listen-client-urls", f"http://{self._host}:{self._port}", "--advertise-client-urls", f"http://{self._host}:{self._port}"],
                close_fds=True,
            )
        self._started = True
        atexit.register(self.stop)

    def get_client(self):
        if not self._started:
            raise RuntimeError("call start before get_client")
        return _etcd_stub.Client(self._host, self._port)

    def _wait_for_ready(self, timeout: int = 60) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.get_client().version
                return
            except Exception:
                time.sleep(0.05)
        raise TimeoutError("timed out waiting for etcd server")

    def stop(self) -> None:
        if not self._started:
            return
        stop_etcd(self._etcd_proc, self._base_data_dir if self._etcd_proc is not None else None)
        self._etcd_proc = None
        self._started = False
