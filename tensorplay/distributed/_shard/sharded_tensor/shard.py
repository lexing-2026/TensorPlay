"""Local shard containers."""

from dataclasses import dataclass
from typing import Any

from ..metadata import ShardMetadata

__all__ = ["Shard"]


@dataclass
class Shard:
    tensor: Any
    metadata: ShardMetadata

    def __post_init__(self) -> None:
        if tuple(self.tensor.shape) != tuple(self.metadata.shard_sizes):
            raise ValueError("local shard shape does not match its metadata")

    @classmethod
    def from_tensor_and_offsets(cls, tensor: Any, shard_offsets: list[int], rank: int) -> "Shard":
        device_type = getattr(getattr(tensor, "device", None), "type", "cpu")
        index = getattr(getattr(tensor, "device", None), "index", None)
        suffix = f":{index}" if index is not None else ""
        return cls(tensor, ShardMetadata(list(shard_offsets), list(tensor.shape), f"rank:{rank}/{device_type}{suffix}"))
