from __future__ import annotations

import abc
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .checkpoint_process import CheckpointProcess
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter
from .staging import CheckpointStager
from .types import STATE_DICT


class Checkpointer(abc.ABC):
    @abc.abstractmethod
    def save(self, path: str, state_dict: STATE_DICT, **kwargs: Any) -> tuple[Future[Any], Future[Any]] | None: ...
    @abc.abstractmethod
    def load(self, path: str, state_dict: STATE_DICT | None = None, *, default_map_location: Any = None, strict: bool = False, **kwargs: Any) -> STATE_DICT: ...
    @abc.abstractmethod
    def close(self) -> None: ...


class SyncCheckpointer(Checkpointer):
    def __init__(self, writer: CheckpointWriter, reader: CheckpointReader) -> None:
        self._writer, self._reader = writer, reader

    def save(self, path: str, state_dict: STATE_DICT, **kwargs: Any) -> None:
        self._writer.write(path, state_dict, **kwargs)

    def load(self, path: str, state_dict: STATE_DICT | None = None, *, default_map_location: Any = None, strict: bool = False, **kwargs: Any) -> STATE_DICT:
        loaded, missing = self._reader.read(path, state_dict, map_location=default_map_location, **kwargs)
        if strict and missing:
            raise RuntimeError(f"checkpoint is missing keys: {missing}")
        return loaded

    def close(self) -> None:
        self._writer.close()


class AsyncCheckpointer(Checkpointer):
    def __init__(self, checkpoint_stager: CheckpointStager, checkpoint_process: CheckpointProcess, reader: CheckpointReader) -> None:
        self._stager, self._process, self._reader = checkpoint_stager, checkpoint_process, reader
        self._executor = ThreadPoolExecutor(max_workers=1)

    def save(self, path: str, state_dict: STATE_DICT, **kwargs: Any) -> tuple[Future[Any], Future[Any]]:
        staged = self._stager.stage(state_dict)
        if isinstance(staged, Future):
            stage_future = staged
        else:
            stage_future = Future(); stage_future.set_result(staged)
        write_future = self._executor.submit(lambda: self._process.write(path, stage_future.result(), **kwargs).result())
        return stage_future, write_future

    def load(self, path: str, state_dict: STATE_DICT | None = None, *, default_map_location: Any = None, strict: bool = False, **kwargs: Any) -> STATE_DICT:
        loaded, missing = self._reader.read(path, state_dict, map_location=default_map_location, **kwargs)
        if strict and missing:
            raise RuntimeError(f"checkpoint is missing keys: {missing}")
        return loaded

    def close(self) -> None:
        self._stager.close(); self._process.close(); self._executor.shutdown(wait=True)
