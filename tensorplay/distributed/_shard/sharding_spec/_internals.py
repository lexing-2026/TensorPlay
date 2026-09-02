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
    if any(value < 0 for value in shape):
        raise ValueError("tensor dimensions must be non-negative")
    values = list(shards)
    validate_non_overlapping_shards_metadata(values)
    total_volume = 1
    for value in shape:
        total_volume *= value
    covered_volume = 0
    for shard in values:
        if len(shard.shard_offsets) != len(shape):
            raise ValueError("shard rank does not match tensor rank")
        if any(
            offset < 0
            or size < 0
            or offset + size > limit
            for offset, size, limit in zip(
                shard.shard_offsets, shard.shard_sizes, shape
            )
        ):
            raise ValueError("shard lies outside tensor bounds")
        volume = 1
        for size in shard.shard_sizes:
            volume *= int(size)
        covered_volume += volume
    if covered_volume != total_volume:
        raise ValueError("shards do not cover the complete tensor")


def get_split_size(dim_size: int, chunks: int) -> int:
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    return (int(dim_size) + int(chunks) - 1) // int(chunks)


def get_chunked_dim_size(dim_size: int, split_size: int, idx: int) -> int:
    return max(0, min(split_size, int(dim_size) - idx * split_size))


def get_chunk_sharding_params(tensor_size: Iterable[int], dim: int, placements: Iterable[Any]) -> tuple[int, int, list[int]]:
    shape = tuple(tensor_size)
    dim = dim if dim >= 0 else dim + len(shape)
    if dim < 0 or dim >= len(shape):
        raise ValueError("sharding dimension is outside tensor rank")
    count = len(list(placements))
    if count <= 0:
        raise ValueError("placements must not be empty")
    split_size = get_split_size(shape[dim], count)
    return dim, split_size, [get_chunked_dim_size(shape[dim], split_size, index) for index in range(count)]


def _find_1d_overlapping_shards(
    shards: list[ShardMetadata], dim: int
) -> tuple[int, int] | None:
    intervals = sorted(
        (
            int(shard.shard_offsets[dim]),
            int(shard.shard_offsets[dim]) + int(shard.shard_sizes[dim]) - 1,
            index,
        )
        for index, shard in enumerate(shards)
    )
    for left, right in zip(intervals, intervals[1:]):
        if left[1] >= right[0]:
            return left[2], right[2]
    return None


def _find_nd_overlapping_shards(
    shards: list[ShardMetadata], sharded_dims: list[int]
) -> tuple[int, int] | None:
    intervals = [
        [
            (
                int(shard.shard_offsets[dim]),
                int(shard.shard_offsets[dim])
                + int(shard.shard_sizes[dim])
                - 1,
            )
            for dim in sharded_dims
        ]
        for shard in shards
    ]
    for left_index, left in enumerate(intervals):
        for right_index in range(left_index + 1, len(intervals)):
            right = intervals[right_index]
            if all(
                left_interval[0] <= right_interval[1]
                and right_interval[0] <= left_interval[1]
                for left_interval, right_interval in zip(left, right)
            ):
                return left_index, right_index
    return None
