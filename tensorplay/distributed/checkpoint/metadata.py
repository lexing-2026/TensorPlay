from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Sequence
from typing import Any

import tensorplay as tp

__all__ = [
    "ChunkStorageMetadata",
    "TensorStorageMetadata",
    "BytesStorageMetadata",
    "Metadata",
    "MetadataIndex",
    "TensorProperties",
    "StorageMeta",
]


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
    layout: Any = field(default_factory=lambda: getattr(tp, "strided", None))
    requires_grad: bool = False
    memory_format: Any = field(default_factory=lambda: tp.contiguous_format)
    pin_memory: bool = False

    def __getstate__(self) -> tuple[Any, Any, bool, _MEM_FORMAT_ENCODING, bool]:
        memory_format = self.memory_format
        if memory_format is None or memory_format == tp.contiguous_format:
            encoding = _MEM_FORMAT_ENCODING.CONTIGUOUS
        elif memory_format == tp.channels_last:
            encoding = _MEM_FORMAT_ENCODING.CHANNELS_LAST
        elif memory_format == tp.preserve_format:
            encoding = _MEM_FORMAT_ENCODING.PRESERVE
        else:
            raise RuntimeError(f"invalid memory format: {memory_format}")
        return (
            self.dtype,
            self.layout,
            self.requires_grad,
            encoding,
            self.pin_memory,
        )

    def __setstate__(self, state: tuple[Any, Any, bool, _MEM_FORMAT_ENCODING, bool]) -> None:
        self.dtype, self.layout, self.requires_grad, encoding, self.pin_memory = state
        if encoding is _MEM_FORMAT_ENCODING.CONTIGUOUS:
            self.memory_format = tp.contiguous_format
        elif encoding is _MEM_FORMAT_ENCODING.CHANNELS_LAST:
            self.memory_format = tp.channels_last
        elif encoding is _MEM_FORMAT_ENCODING.PRESERVE:
            self.memory_format = tp.preserve_format
        else:
            raise RuntimeError(f"invalid memory format encoding: {encoding}")

    @staticmethod
    def create_from_tensor(tensor: tp.Tensor) -> "TensorProperties":
        is_pinned = getattr(tensor, "is_pinned", None)
        return TensorProperties(
            dtype=tensor.dtype,
            layout=getattr(tensor, "layout", None),
            requires_grad=bool(tensor.requires_grad),
            memory_format=tp.contiguous_format,
            pin_memory=bool(is_pinned()) if callable(is_pinned) else False,
        )


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


STORAGE_TYPES = TensorStorageMetadata | BytesStorageMetadata
STATE_DICT_TYPE = dict[str, Any]
