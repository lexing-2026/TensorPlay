"""Metadata properties for sharded tensor values."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..metadata import ShardMetadata

__all__ = ["MEM_FORMAT_ENCODING", "TensorProperties", "ShardedTensorMetadata"]


class MEM_FORMAT_ENCODING(Enum):
    CONTIGUOUS = 0
    CHANNELS_LAST = 1
    PRESERVE = 2


@dataclass
class TensorProperties:
    dtype: Any = None
    layout: Any = None
    requires_grad: bool = False
    memory_format: Any = None
    pin_memory: bool = False

    @staticmethod
    def create_from_tensor(tensor: Any) -> "TensorProperties":
        return TensorProperties(tensor.dtype, getattr(tensor, "layout", None), bool(tensor.requires_grad), None, False)


@dataclass
class ShardedTensorMetadata:
    shards_metadata: list[ShardMetadata] = field(default_factory=list)
    size: tuple[int, ...] = ()
    tensor_properties: TensorProperties = field(default_factory=TensorProperties)
