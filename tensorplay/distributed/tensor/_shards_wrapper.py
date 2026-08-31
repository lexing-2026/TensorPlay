"""Containers for local shard metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

__all__ = ["ShardMetadata", "ShardsWrapper"]


@dataclass(frozen=True)
class ShardMetadata:
    shard: Any
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    placement: Any = None


class ShardsWrapper:
    def __init__(self, shards: list[ShardMetadata] | None = None) -> None:
        self._shards = list(shards or [])

    def append(self, metadata: ShardMetadata) -> None:
        self._shards.append(metadata)

    def __iter__(self) -> Iterator[ShardMetadata]:
        return iter(self._shards)

    def __len__(self) -> int:
        return len(self._shards)

    def __getitem__(self, index: int) -> ShardMetadata:
        return self._shards[index]

    def tensors(self) -> tuple[Any, ...]:
        return tuple(item.shard for item in self._shards)
