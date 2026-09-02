from __future__ import annotations

import abc
from concurrent.futures import Future
from dataclasses import dataclass
import os
from pathlib import Path
import threading
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
        if config.write_barrier_timeout_secs <= 0:
            raise ValueError("write_barrier_timeout_secs must be positive")
        if rank_info.global_rank < 0 or rank_info.global_world_size <= rank_info.global_rank:
            raise ValueError("rank information is invalid")
        self._config = config
        self._rank_info = rank_info
        self._barrier = barrier
        self._commit_hook = commit_hook
        self._closed = False
        self._close_lock = threading.Lock()

    def write(self, path: str, state_dict: STATE_DICT, **kwargs: Any) -> Future[None] | None:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("checkpoint path must be path-like")
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint state_dict must be a dictionary")
        with self._close_lock:
            if self._closed:
                raise RuntimeError("checkpoint writer is closed")
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        final_path = target / f"checkpoint_{self._rank_info.global_rank}.pt"
        temporary_path = target / (
            f".{final_path.stem}.tmp.{os.getpid()}.{threading.get_ident()}.pt"
        )
        try:
            tp.save(state_dict, temporary_path)
            os.replace(temporary_path, final_path)
        except BaseException:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise
        if self._commit_hook is not None:
            self._commit_hook.pre_commit(str(target), **kwargs)
        if self._barrier is not None:
            self._barrier.execute_barrier()
        if self._commit_hook is not None:
            self._commit_hook.post_commit(str(target), **kwargs)
        return None

    def close(self) -> None:
        with self._close_lock:
            self._closed = True
