from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
import os
from typing import Any

from ._async_executor import _AsyncCheckpointExecutor


class _CheckpointSaveProcessControlOpts(Enum):
    START = auto()
    STOP = auto()
    SAVE = auto()


@dataclass(frozen=True)
class _CheckpointRequestIdentifier:
    checkpoint_id: str | os.PathLike[str] | None


@dataclass
class _AsyncCheckpointRequest:
    request_id: _CheckpointRequestIdentifier
    state_dict: Any


@dataclass
class _ProcessGroupInitInfo:
    process_group: Any | None = None


class _AsyncCheckpointProcess:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def _send(self, data: Any) -> None:
        self._last_data = data

    def _wait_for_response(self, timeout: float | None = None) -> Any:
        del timeout
        return getattr(self, "_last_data", None)

    def save(self, save_fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
        return self._executor.submit(save_fn, *args, **kwargs)

    def _execute_save(self, save_fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
        return self.save(save_fn, *args, **kwargs)

    def _checkpointing_subprocess(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _ProcessBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._process = _AsyncCheckpointProcess()

    def _execute_save_impl(self, save_fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
        return self._process.save(save_fn, *args, **kwargs)

    def execute_save(self, save_fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
        return self._execute_save_impl(save_fn, *args, **kwargs)
