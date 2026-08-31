from __future__ import annotations

from datetime import timedelta

DEFAULT_RPC_TIMEOUT_SEC: float = 60.0
DEFAULT_INIT_METHOD: str = "env://"
DEFAULT_SHUTDOWN_TIMEOUT: float = 0.0
DEFAULT_NUM_WORKER_THREADS: int = 16
DEFAULT_PROCESS_GROUP_TIMEOUT: timedelta = timedelta(milliseconds=2**31 - 1)
UNSET_RPC_TIMEOUT: float = -1.0

__all__: list[str] = []
