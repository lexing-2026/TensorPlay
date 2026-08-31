from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .state_dict_loader import load
from .state_dict_saver import save

__all__: list[str] = []


class _Checkpointer:
    def __init__(self, checkpoint_id: str | None = None, storage_writer: Any = None, storage_reader: Any = None, planner: Any = None) -> None:
        self.checkpoint_id = checkpoint_id
        self.storage_writer = storage_writer
        self.storage_reader = storage_reader
        self.planner = planner
        self._executor = ThreadPoolExecutor(max_workers=1)

    def save(self, state_dict: dict[str, Any], **kwargs: Any) -> Any:
        return save(state_dict, checkpoint_id=kwargs.pop("checkpoint_id", self.checkpoint_id), storage_writer=kwargs.pop("storage_writer", self.storage_writer), planner=kwargs.pop("planner", self.planner), **kwargs)

    def async_save(self, state_dict: dict[str, Any], **kwargs: Any) -> Future[Any]:
        return self._executor.submit(self.save, state_dict, **kwargs)

    def load(self, state_dict: dict[str, Any]) -> None:
        load(state_dict, checkpoint_id=self.checkpoint_id, storage_reader=self.storage_reader, planner=self.planner)
