from __future__ import annotations

import abc
import os
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from .metadata import Metadata, MetadataIndex, StorageMeta
from .planner import LoadPlan, LoadPlanner, SavePlan, SavePlanner

__all__ = ["WriteResult", "StorageWriter", "StorageReader"]


@dataclass(frozen=True)
class WriteResult:
    index: MetadataIndex
    size_in_bytes: int
    storage_data: Any


class StorageWriter(abc.ABC):
    @abc.abstractmethod
    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None: ...
    @abc.abstractmethod
    def set_up_storage_writer(self, is_coordinator: bool, *args: Any, **kwargs: Any) -> None: ...
    @abc.abstractmethod
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan: ...
    @abc.abstractmethod
    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]: ...
    @abc.abstractmethod
    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[list[WriteResult]]: ...
    @abc.abstractmethod
    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None: ...
    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool: ...
    def storage_meta(self) -> StorageMeta | None:
        return None

    def abort(self) -> None:
        return None


class StorageReader(abc.ABC):
    @abc.abstractmethod
    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None: ...
    @abc.abstractmethod
    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata: ...
    @abc.abstractmethod
    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any) -> None: ...
    @abc.abstractmethod
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan: ...
    @abc.abstractmethod
    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]: ...
    @abc.abstractmethod
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]: ...
    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool: ...
