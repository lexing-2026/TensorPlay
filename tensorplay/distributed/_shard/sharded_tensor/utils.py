"""Utility functions for local shard metadata."""

import math
from typing import Any, Iterable

from ..metadata import ShardMetadata
from ...remote_device import _remote_device

__all__ = ["_parse_and_validate_remote_device", "_validate_output_tensor_for_gather", "_flatten_tensor_size", "_raise_if_mismatch", "build_metadata_from_local_shards", "build_global_metadata", "recalc_global_sharded_tensor_metadata"]


def _parse_and_validate_remote_device(process_group: Any, remote_device: Any) -> tuple[int, str]:
    del process_group
    value = remote_device if isinstance(remote_device, _remote_device) else _remote_device(str(remote_device))
    return value.rank(), value.device()


def _validate_output_tensor_for_gather(output: Any, size: Iterable[int]) -> None:
    if tuple(output.shape) != tuple(size):
        raise ValueError("gather output shape does not match the global shape")


def _flatten_tensor_size(size: Any) -> tuple[int, ...]:
    return tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(int(value) for value in size)


def _raise_if_mismatch(left: Any, right: Any, message: str = "values do not match") -> None:
    if left != right:
        raise ValueError(message)


def build_metadata_from_local_shards(local_shards: Iterable[Any]) -> list[ShardMetadata]:
    return [shard.metadata for shard in local_shards]


def build_global_metadata(shards_metadata: Iterable[ShardMetadata], size: Iterable[int], tensor_properties: Any) -> Any:
    from .metadata import ShardedTensorMetadata
    return ShardedTensorMetadata(list(shards_metadata), tuple(size), tensor_properties)


def recalc_global_sharded_tensor_metadata(metadata: Any, size: Iterable[int]) -> Any:
    metadata.size = tuple(size)
    return metadata
