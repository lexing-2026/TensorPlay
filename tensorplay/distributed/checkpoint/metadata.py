from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import tensorplay as tp

__all__ = ["ChunkStorageMetadata", "TensorStorageMetadata", "BytesStorageMetadata", "Metadata", "MetadataIndex", "TensorProperties", "StorageMeta"]


@dataclass
class ChunkStorageMetadata:
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]


class _MEM_FORMAT_ENCODING(Enum):
    CONTIGUOUS = 0
    CHANNELS_LAST = 1
    PRESERVE = 2


@dataclass
class TensorProperties:
    dtype: Any = field(default_factory=lambda: tp.get_default_dtype())
    layout: Any = None
    requires_grad: bool = False
    memory_format: Any = None
    pin_memory: bool = False

    @staticmethod
    def create_from_tensor(tensor: tp.Tensor) -> "TensorProperties":
        return TensorProperties(tensor.dtype, getattr(tensor, "layout", None), bool(tensor.requires_grad), None, False)


@dataclass
class TensorStorageMetadata:
    properties: TensorProperties
    size: tuple[int, ...]
    chunks: list[ChunkStorageMetadata]


@dataclass
class BytesStorageMetadata:
    pass


@dataclass
class StorageMeta:
    checkpoint_id: str | os.PathLike[str] | None = None
    save_id: str | None = None
    load_id: str | None = None
    modules: list[str] = field(default_factory=list)


@dataclass
class Metadata:
    state_dict_metadata: dict[str, TensorStorageMetadata | BytesStorageMetadata]
    planner_data: Any = None
    storage_data: Any = None
    storage_meta: StorageMeta | None = None
    version: str | None = None


@dataclass(frozen=True, init=False)
class MetadataIndex:
    fqn: str
    offset: tuple[int, ...] | None = None
    index: int | None = field(default=None, compare=False, hash=False)

    def __init__(self, fqn: str, offset: Sequence[int] | None = None, index: int | None = None) -> None:
        object.__setattr__(self, "fqn", fqn)
        object.__setattr__(self, "offset", tuple(offset) if offset is not None else None)
        object.__setattr__(self, "index", index)
