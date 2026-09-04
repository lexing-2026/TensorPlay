from __future__ import annotations

import abc
import io
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import tensorplay as tp

from .metadata import ChunkStorageMetadata, Metadata, MetadataIndex, StorageMeta, TensorProperties

__all__ = ["WriteItemType", "LoadItemType", "BytesIOWriteData", "TensorWriteData", "WriteItem", "ReadItem", "SavePlan", "LoadPlan", "SavePlanner", "LoadPlanner"]


class WriteItemType(Enum):
    TENSOR = auto()
    SHARD = auto()
    BYTE_IO = auto()


class LoadItemType(Enum):
    TENSOR = auto()
    BYTE_IO = auto()


@dataclass(frozen=True)
class BytesIOWriteData:
    nbytes: int


@dataclass(frozen=True)
class TensorWriteData:
    chunk: ChunkStorageMetadata
    properties: TensorProperties
    size: tuple[int, ...]


@dataclass(frozen=True)
class WriteItem:
    index: MetadataIndex
    type: WriteItemType
    bytes_io_data: BytesIOWriteData | None = None
    tensor_data: TensorWriteData | None = None

    def tensor_storage_size(self) -> int | None:
        if self.tensor_data is None:
            return None
        size = math.prod(self.tensor_data.size)
        element_size = getattr(self.tensor_data.properties.dtype, "itemsize", 1)
        return int(size) * int(element_size)


@dataclass(frozen=True)
class ReadItem:
    type: LoadItemType
    dest_index: MetadataIndex
    dest_offsets: tuple[int, ...]
    storage_index: MetadataIndex
    storage_offsets: tuple[int, ...]
    lengths: tuple[int, ...]


@dataclass(frozen=True)
class SavePlan:
    items: list[WriteItem]
    storage_data: Any = None
    planner_data: Any = None
    usable: bool = True


@dataclass
class LoadPlan:
    items: list[ReadItem]
    storage_data: Any = None
    planner_data: Any = None


class SavePlanner(abc.ABC):
    _cached_save_plan: dict[str, SavePlan] = {}
    _cached_final_save_plan: dict[str, SavePlan] = {}
    _cached_all_plans: dict[str, list[SavePlan]] = {}
    _cached_global_plan: dict[str, list[SavePlan]] = {}
    _cached_metadata: dict[str, Metadata] = {}

    @abc.abstractmethod
    def set_up_planner(self, state_dict: dict[str, Any], storage_meta: StorageMeta | None = None, is_coordinator: bool = False) -> None: ...
    @abc.abstractmethod
    def create_local_plan(self) -> SavePlan: ...
    @abc.abstractmethod
    def create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]: ...
    @abc.abstractmethod
    def finish_plan(self, new_plan: SavePlan) -> SavePlan: ...
    @abc.abstractmethod
    def resolve_data(self, write_item: WriteItem) -> tp.Tensor | io.BytesIO: ...


class LoadPlanner(abc.ABC):
    @abc.abstractmethod
    def set_up_planner(self, state_dict: dict[str, Any], metadata: Metadata | None = None, is_coordinator: bool = False) -> None: ...
    @abc.abstractmethod
    def create_local_plan(self) -> LoadPlan: ...
    @abc.abstractmethod
    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]: ...
    @abc.abstractmethod
    def finish_plan(self, central_plan: LoadPlan) -> LoadPlan: ...
    @abc.abstractmethod
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None: ...
    def resolve_bytes(self, read_item: ReadItem) -> io.BytesIO:
        raise NotImplementedError
    @abc.abstractmethod
    def resolve_tensor(self, read_item: ReadItem) -> tp.Tensor: ...
    @abc.abstractmethod
    def commit_tensor(self, read_item: ReadItem, tensor: tp.Tensor) -> None: ...
