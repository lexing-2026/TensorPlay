from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from ._async_executor import _AsyncCheckpointExecutor


def save_wrapper(save_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return save_fn(*args, **kwargs)


class _ThreadBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)

    def execute_save(self, save_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        return self._executor.submit(save_wrapper, save_fn, *args, **kwargs)

    def close(self) -> None:
        self._executor.shutdown(wait=True)
