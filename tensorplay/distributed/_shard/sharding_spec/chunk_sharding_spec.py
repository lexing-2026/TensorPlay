"""Chunk-based sharding specifications."""

from dataclasses import dataclass
from typing import Any, Iterable

from ..metadata import ShardMetadata
from ...remote_device import _remote_device
from ._internals import get_chunked_dim_size, get_split_size
from .api import ShardingSpec

__all__ = ["ChunkShardingSpec"]


@dataclass
class ChunkShardingSpec(ShardingSpec):
    dim: int | str
    placements: list[Any]

    def __post_init__(self) -> None:
        if not isinstance(self.dim, int):
            raise TypeError("chunk sharding dimension must be an integer")
        if not self.placements:
            raise ValueError("placements must not be empty")
        self.placements = [
            placement
            if isinstance(placement, _remote_device)
            else _remote_device(placement)
            for placement in self.placements
        ]

    def build_metadata(self, tensor_sizes: Iterable[int], tensor_properties: Any) -> Any:
        shape = tuple(int(value) for value in tensor_sizes)
        dim = self.dim if self.dim >= 0 else self.dim + len(shape)
        if dim < 0 or dim >= len(shape):
            raise ValueError("chunk sharding dimension is outside tensor rank")
        split_size = get_split_size(shape[dim], len(self.placements))
        shards = []
        for index, placement in enumerate(self.placements):
            size = list(shape)
            size[dim] = get_chunked_dim_size(shape[dim], split_size, index)
            offset = [0] * len(shape)
            offset[dim] = index * split_size
            shards.append(ShardMetadata(offset, size, placement))
        from ..sharded_tensor.metadata import ShardedTensorMetadata
        return ShardedTensorMetadata(shards, shape, tensor_properties)

    def shard(self, tensor: Any, src_rank: int = 0, process_group: Any = None) -> Any:
        from ..sharded_tensor.api import ShardedTensor

        return ShardedTensor._scatter_from_global_tensor(
            self,
            tensor,
            process_group=process_group,
            src_rank=src_rank,
        )
