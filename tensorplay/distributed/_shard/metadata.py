"""Metadata records for explicitly placed shards."""

from dataclasses import dataclass
from typing import Any

from ..remote_device import _remote_device

__all__ = ["ShardMetadata"]


@dataclass(frozen=True)
class ShardMetadata:
    shard_offsets: list[int]
    shard_sizes: list[int]
    placement: Any = None

    def __post_init__(self) -> None:
        if len(self.shard_offsets) != len(self.shard_sizes):
            raise ValueError("shard offsets and sizes must have the same rank")
        if any(int(value) < 0 for value in self.shard_offsets + self.shard_sizes):
            raise ValueError("shard offsets and sizes must be non-negative")
        if isinstance(self.placement, str):
            object.__setattr__(self, "placement", _remote_device(self.placement))

    def __hash__(self) -> int:
        return hash((tuple(self.shard_offsets), tuple(self.shard_sizes), self.placement))
