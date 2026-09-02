"""Sharding specification interfaces and generic enumerated layouts."""

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..metadata import ShardMetadata
from ...remote_device import _remote_device
from ._internals import check_tensor, validate_non_overlapping_shards_metadata

__all__ = ["PlacementSpec", "DevicePlacementSpec", "ShardingSpec", "EnumerableShardingSpec", "custom_sharding_spec_op"]


class PlacementSpec(ABC):
    pass


@dataclass
class DevicePlacementSpec(PlacementSpec):
    device: Any


class ShardingSpec(ABC):
    @abstractmethod
    def build_metadata(self, tensor_sizes: Iterable[int], tensor_properties: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def shard(self, tensor: Any, src_rank: int = 0, process_group: Any = None) -> Any:
        raise NotImplementedError


_CUSTOM_SHARDING_SPEC_OPS: dict[str, dict[Callable[..., Any], Callable[..., Any]]] = {}


def _has_custom_op(sharding_spec: Any, op: Callable[..., Any]) -> bool:
    return op in _CUSTOM_SHARDING_SPEC_OPS.get(type(sharding_spec).__qualname__, {})


def _dispatch_custom_op(sharding_spec: Any, op: Callable[..., Any], types: Any, args: Any, kwargs: Any, process_group: Any) -> Any:
    try:
        fn = _CUSTOM_SHARDING_SPEC_OPS[type(sharding_spec).__qualname__][op]
    except KeyError as exc:
        raise RuntimeError("custom operation is not registered for this specification") from exc
    return fn(types, args, kwargs, process_group)


def custom_sharding_spec_op(sharding_spec_class: type, func: Callable[..., Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    table = _CUSTOM_SHARDING_SPEC_OPS.setdefault(sharding_spec_class.__qualname__, {})
    def decorator(implementation: Callable[..., Any]) -> Callable[..., Any]:
        table[func] = implementation
        return implementation
    return decorator


@dataclass
class EnumerableShardingSpec(ShardingSpec):
    shards: list[ShardMetadata]

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("shards must not be empty")
        self.shards = [
            shard
            if isinstance(shard.placement, _remote_device)
            else ShardMetadata(
                list(shard.shard_offsets),
                list(shard.shard_sizes),
                _remote_device(shard.placement),
            )
            for shard in self.shards
        ]
        validate_non_overlapping_shards_metadata(self.shards)

    def build_metadata(self, tensor_sizes: Iterable[int], tensor_properties: Any) -> Any:
        shape = tuple(int(value) for value in tensor_sizes)
        check_tensor(self.shards, shape)
        from ..sharded_tensor.metadata import ShardedTensorMetadata
        return ShardedTensorMetadata(list(self.shards), shape, tensor_properties)

    def shard(self, tensor: Any, src_rank: int = 0, process_group: Any = None) -> Any:
        from ..sharded_tensor.api import ShardedTensor

        return ShardedTensor._scatter_from_global_tensor(
            self,
            tensor,
            process_group=process_group,
            src_rank=src_rank,
        )


def _infer_sharding_spec_from_shards_metadata(shards_metadata: list[ShardMetadata]) -> ShardingSpec:
    if not shards_metadata:
        raise ValueError("shards metadata must not be empty")
    dims = {index for shard in shards_metadata for index, offset in enumerate(shard.shard_offsets) if offset}
    if len(dims) == 1:
        from .chunk_sharding_spec import ChunkShardingSpec
        dim = next(iter(dims))
        return ChunkShardingSpec(dim, [shard.placement for shard in shards_metadata])
    return EnumerableShardingSpec(shards_metadata)
