from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ._state_dict_stager import StateDictStager

__all__ = ["AsyncStager", "BlockingAsyncStager", "DefaultStager", "StagingOptions"]


@runtime_checkable
class AsyncStager(Protocol):
    _synchronize_after_execute: bool = True
    @property
    def should_synchronize_after_execute(self) -> bool: ...
    def stage(self, state_dict: dict[str, Any]) -> Future[dict[str, Any]] | dict[str, Any]: ...
    def synchronize_staging(self) -> None: ...
    def close(self) -> None: ...


@dataclass
class StagingOptions:
    use_pinned_memory: bool = True
    use_shared_memory: bool = True
    use_async_staging: bool = True
    use_non_blocking_copy: bool = False


class DefaultStager:
    def __init__(self, config: StagingOptions = StagingOptions()) -> None:
        self._config = config
        self._stager = StateDictStager(config.use_pinned_memory, config.use_shared_memory)
        self._executor = ThreadPoolExecutor(max_workers=1) if config.use_async_staging else None
        self._staging_future: Future[dict[str, Any]] | None = None

    @property
    def should_synchronize_after_execute(self) -> bool:
        return True

    def _stage(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._stager.stage(state_dict, **kwargs)

    def stage(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any] | Future[dict[str, Any]]:
        if self._executor is None:
            return self._stage(state_dict, **kwargs)
        self._staging_future = self._executor.submit(self._stage, state_dict, **kwargs)
        return self._staging_future

    def synchronize_staging(self) -> None:
        if self._staging_future is not None:
            self._staging_future.result()

    def close(self) -> None:
        self.synchronize_staging()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._stager.close()


class BlockingAsyncStager(DefaultStager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        options = kwargs.pop("config", None) or (args[0] if args else StagingOptions())
        options.use_async_staging = False
        super().__init__(options)
