"""Tensor slicing helpers for shard metadata."""

from typing import Any

from .metadata import ShardMetadata

__all__ = ["narrow_tensor_by_index", "narrow_tensor"]


def narrow_tensor_by_index(tensor: Any, index: int, dim: int, length: int) -> Any:
    return tensor.narrow(dim, index, length)


def narrow_tensor(tensor: Any, shard_metadata: ShardMetadata) -> Any:
    result = tensor
    for dim, (offset, size) in enumerate(zip(shard_metadata.shard_offsets, shard_metadata.shard_sizes)):
        result = result.narrow(dim, offset, size)
    return result
