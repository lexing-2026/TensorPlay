"""Validation and chunk arithmetic for sharding specifications."""

import math
from typing import Any, Iterable

from ..metadata import ShardMetadata

__all__ = ["check_tensor", "get_split_size", "get_chunked_dim_size", "validate_non_overlapping_shards_metadata", "get_chunk_sharding_params"]


def _ranges_overlap(left: ShardMetadata, right: ShardMetadata) -> bool:
    return all(a < b + size_b and b < a + size_a for a, size_a, b, size_b in zip(left.shard_offsets, left.shard_sizes, right.shard_offsets, right.shard_sizes))


def validate_non_overlapping_shards_metadata(shards: Iterable[ShardMetadata]) -> None:
    values = list(shards)
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            if _ranges_overlap(left, right):
                raise ValueError("shard metadata contains overlapping regions")


def check_tensor(shards: Iterable[ShardMetadata], tensor_size: Iterable[int]) -> None:
    shape = tuple(int(value) for value in tensor_size)
    for shard in shards:
        if len(shard.shard_offsets) != len(shape):
            raise ValueError("shard rank does not match tensor rank")
        if any(offset + size > limit for offset, size, limit in zip(shard.shard_offsets, shard.shard_sizes, shape)):
            raise ValueError("shard lies outside tensor bounds")


def get_split_size(dim_size: int, chunks: int) -> int:
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    return (int(dim_size) + int(chunks) - 1) // int(chunks)


def get_chunked_dim_size(dim_size: int, split_size: int, idx: int) -> int:
    return max(0, min(split_size, int(dim_size) - idx * split_size))


def get_chunk_sharding_params(tensor_size: Iterable[int], dim: int, placements: Iterable[Any]) -> tuple[int, int, list[int]]:
    shape = tuple(tensor_size)
    dim = dim if dim >= 0 else dim + len(shape)
    count = len(list(placements))
    split_size = get_split_size(shape[dim], count)
    return dim, split_size, [get_chunked_dim_size(shape[dim], split_size, index) for index in range(count)]


def _find_1d_overlapping_shards(*args: Any, **kwargs: Any) -> list[Any]:
    del args, kwargs
    return []


def _find_nd_overlapping_shards(*args: Any, **kwargs: Any) -> list[Any]:
    del args, kwargs
    return []

