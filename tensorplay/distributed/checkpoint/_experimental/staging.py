from __future__ import annotations

import abc
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import tensorplay as tp

from .._state_dict_stager import StateDictStager
from .types import STATE_DICT

T = TypeVar("T")


class CheckpointStager(abc.ABC):
    @abc.abstractmethod
    def stage(self, state_dict: STATE_DICT, **kwargs: Any) -> STATE_DICT | Future[STATE_DICT]:
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...


@dataclass
class CheckpointStagerConfig:
    use_pinned_memory: bool = True
    use_shared_memory: bool = True
    use_async_staging: bool = True
    use_non_blocking_copy: bool = True


class DefaultStager(CheckpointStager):
    def __init__(self, config: CheckpointStagerConfig = CheckpointStagerConfig()) -> None:
        self._config = config
        self._stager = StateDictStager(config.use_pinned_memory, config.use_shared_memory)
        if config.use_non_blocking_copy and not tp.cuda.is_available():
            raise AssertionError(
                "Non-blocking copy requires an available accelerator"
            )
        self._executor = ThreadPoolExecutor(max_workers=1) if config.use_async_staging else None

    def stage(self, state_dict: STATE_DICT, **kwargs: Any) -> STATE_DICT | Future[STATE_DICT]:
        if self._executor is None:
            return self._stage(state_dict, **kwargs)
        return self._executor.submit(self._stage, state_dict, **kwargs)

    def _stage(self, state_dict: STATE_DICT, **kwargs: Any) -> STATE_DICT:
        return self._stager.stage(
            state_dict,
            non_blocking=self._config.use_non_blocking_copy,
            **kwargs,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._stager.close()
