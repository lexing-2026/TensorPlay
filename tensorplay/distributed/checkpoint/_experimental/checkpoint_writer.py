from __future__ import annotations

import abc
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorplay as tp

from .barriers import Barrier
from .types import RankInfo, STATE_DICT


class WriterHook(abc.ABC):
    @abc.abstractmethod
    def pre_commit(self, path: str, **kwargs: Any) -> None: ...
    @abc.abstractmethod
    def post_commit(self, path: str, **kwargs: Any) -> None: ...


@dataclass
class CheckpointWriterConfig:
    write_barrier_timeout_secs: int = 600


class CheckpointWriter:
    def __init__(self, config: CheckpointWriterConfig, rank_info: RankInfo, barrier: Barrier | None = None, commit_hook: WriterHook | None = None) -> None:
        self._config = config
        self._rank_info = rank_info
        self._barrier = barrier
        self._commit_hook = commit_hook

    def write(self, path: str, state_dict: STATE_DICT, **kwargs: Any) -> Future[None] | None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        tp.save(state_dict, target / f"checkpoint_{self._rank_info.global_rank}.tp")
        if self._commit_hook is not None:
            self._commit_hook.pre_commit(path, **kwargs)
        if self._barrier is not None:
            self._barrier.execute_barrier()
        if self._commit_hook is not None:
            self._commit_hook.post_commit(path, **kwargs)
        return None

    def close(self) -> None:
        return None
