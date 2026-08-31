from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

from .checkpoint_writer import CheckpointWriter
from .types import RankInfo, STATE_DICT


@dataclass
class CheckpointProcessConfig:
    process_start_method: str = "spawn"


class RequestType(Enum):
    WRITE = auto()
    CLOSE = auto()


@dataclass
class WorkerRequest:
    request_type: RequestType
    payload: dict[str, Any]


@dataclass
class WorkerResponse:
    success: bool
    payload: Any = None


class CheckpointProcess:
    def __init__(self, rank_info: RankInfo, config: CheckpointProcessConfig, subprocess_init_fn: Callable[..., None], subprocess_init_args: tuple[Any, ...], checkpoint_writer_init_fn: Callable[..., CheckpointWriter], checkpoint_writer_init_args: dict[str, Any]) -> None:
        del config
        subprocess_init_fn(*subprocess_init_args)
        self._writer = checkpoint_writer_init_fn(rank_info, **checkpoint_writer_init_args)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _create_subprocess(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
    def _subprocess(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
    def _send(self, request_type: RequestType, payload: dict[str, Any]) -> None:
        self._last_request = WorkerRequest(request_type, payload)
    def _recv(self) -> dict[str, Any] | None:
        return getattr(self, "_last_request", None).__dict__ if hasattr(self, "_last_request") else None
    def write(self, path: str, state_dict: STATE_DICT, **kwargs: Any) -> Future[Any]:
        return self._executor.submit(self._writer.write, path, state_dict, **kwargs)
    def _write(self, path: str, state_dict: STATE_DICT, **kwargs: Any) -> Any:
        return self._writer.write(path, state_dict, **kwargs)
    def close(self) -> None:
        self._writer.close()
        self._executor.shutdown(wait=True)
