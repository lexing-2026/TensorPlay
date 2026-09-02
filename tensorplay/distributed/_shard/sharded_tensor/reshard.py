"""Collective reshaping helpers for explicitly sharded tensors."""

import math
from typing import Any

import tensorplay as tp

from ... import distributed_core as dist
from ..metadata import ShardMetadata
from .metadata import TensorProperties
from .shard import Shard

__all__ = [
    "get_idx_from_placements",
    "build_reshard_metadata",
    "reshuffle_local_shard",
    "reshard_local_shard",
]


def _rank_of(placement: Any) -> int:
    rank = placement.rank() if hasattr(placement, "rank") else None
    if rank is None:
        text = str(placement)
        if not text.startswith("rank:"):
            raise ValueError(f"placement {placement!r} does not identify a rank")
        rank = int(text.split(":", 1)[1].split("/", 1)[0])
    return int(rank)


def get_idx_from_placements(placements: Any, current_rank: int) -> int:
    for index, placement in enumerate(placements):
        if _rank_of(placement) == int(current_rank):
            return index
    raise RuntimeError("current rank is not present in the placements")


def build_reshard_metadata(
    st_size: Any,
    sharding_spec: Any,
    world_size: int,
) -> tuple[list[ShardMetadata], list[int]]:
    shape = tuple(int(value) for value in st_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    placements = list(getattr(sharding_spec, "placements", ()))
    if len(placements) != world_size:
        raise ValueError("placement count must equal world_size")
    dimension = int(sharding_spec.dim)
    if dimension < 0:
        dimension += len(shape)
    if dimension < 0 or dimension >= len(shape):
        raise ValueError("sharding dimension is outside the tensor rank")
    width = (shape[dimension] + world_size - 1) // world_size
    metadata_by_rank: list[ShardMetadata | None] = [None] * world_size
    ranks: list[int] = []
    for index, placement in enumerate(placements):
        rank = _rank_of(placement)
        if rank < 0 or rank >= world_size:
            raise ValueError("placement rank is outside world_size")
        if metadata_by_rank[rank] is not None:
            raise ValueError("placements must use unique ranks")
        size = list(shape)
        size[dimension] = max(0, min(width, shape[dimension] - index * width))
        offset = [0] * len(shape)
        offset[dimension] = index * width
        metadata_by_rank[rank] = ShardMetadata(offset, size, placement)
        ranks.append(rank)
    if any(item is None for item in metadata_by_rank):
        raise ValueError("placements must cover every rank")
    return [item for item in metadata_by_rank if item is not None], ranks


def _group_rank(process_group: Any, global_rank: int) -> int:
    if process_group is None:
        return int(global_rank)
    return int(dist.get_group_rank(process_group, int(global_rank)))


def _source_metadata(
    local_tensor: Any,
    st_size: Any,
    sharding_spec: Any,
) -> list[ShardMetadata]:
    metadata = sharding_spec.build_metadata(
        tuple(int(value) for value in st_size),
        TensorProperties.create_from_tensor(local_tensor),
    )
    return list(metadata.shards_metadata)


def _gather_full_tensor(
    local_tensor: Any,
    st_size: Any,
    sharding_spec: Any,
    process_group: Any,
) -> Any:
    shape = tuple(int(value) for value in st_size)
    metadata = _source_metadata(local_tensor, shape, sharding_spec)
    if not dist.is_initialized() or dist.get_world_size(process_group) == 1:
        result = tp.zeros(shape, dtype=local_tensor.dtype, device=local_tensor.device)
        current_rank = 0 if not dist.is_initialized() else dist.get_rank()
        for item in metadata:
            if _rank_of(item.placement) == current_rank:
                _copy_into_global(result, local_tensor, item)
        return result

    group_size = dist.get_world_size(process_group)
    current_global_rank = dist.get_rank()
    current_meta = next(
        (
            item
            for item in metadata
            if _rank_of(item.placement) == current_global_rank
        ),
        None,
    )
    if current_meta is None:
        raise ValueError("current rank does not own a shard")
    count = int(math.prod(current_meta.shard_sizes))
    max_count = max(
        1, max(int(math.prod(item.shard_sizes)) for item in metadata)
    )
    packed = tp.zeros(max_count, dtype=local_tensor.dtype, device=local_tensor.device)
    if count:
        packed.narrow(0, 0, count).copy_(local_tensor.reshape(-1))
    gathered = [
        tp.empty((max_count,), dtype=local_tensor.dtype, device=local_tensor.device)
        for _ in range(group_size)
    ]
    dist.all_gather(gathered, packed, group=process_group)
    result = tp.zeros(shape, dtype=local_tensor.dtype, device=local_tensor.device)
    for item in metadata:
        group_rank = _group_rank(process_group, _rank_of(item.placement))
        item_count = int(math.prod(item.shard_sizes))
        values = gathered[group_rank].narrow(0, 0, item_count).reshape(
            tuple(item.shard_sizes)
        )
        _copy_into_global(result, values, item)
    return result


def _copy_into_global(destination: Any, source: Any, metadata: ShardMetadata) -> None:
    view = destination
    for dimension, (offset, size) in enumerate(
        zip(metadata.shard_offsets, metadata.shard_sizes)
    ):
        view = view.narrow(dimension, int(offset), int(size))
    view.copy_(source.reshape(tuple(metadata.shard_sizes)))


def _make_target_shard(
    full_tensor: Any,
    st_size: Any,
    resharding_spec: Any,
    process_group: Any,
) -> tuple[list[Shard], list[ShardMetadata]]:
    shape = tuple(int(value) for value in st_size)
    properties = TensorProperties.create_from_tensor(full_tensor)
    metadata = list(resharding_spec.build_metadata(shape, properties).shards_metadata)
    current_rank = 0 if not dist.is_initialized() else dist.get_rank()
    local_metadata = [
        item for item in metadata if _rank_of(item.placement) == current_rank
    ]
    if len(local_metadata) != 1:
        raise ValueError("every rank must own one target shard")
    item = local_metadata[0]
    local = full_tensor
    for dimension, (offset, size) in enumerate(
        zip(item.shard_offsets, item.shard_sizes)
    ):
        local = local.narrow(dimension, int(offset), int(size))
    local = local.detach().clone()
    local.requires_grad_(bool(full_tensor.requires_grad))
    return [Shard(local, item)], metadata


def _reshard(
    local_tensor: Any,
    st_size: Any,
    sharding_spec: Any,
    resharding_spec: Any,
    process_group: Any,
) -> tuple[list[Shard], list[ShardMetadata]]:
    full = _gather_full_tensor(local_tensor, st_size, sharding_spec, process_group)
    return _make_target_shard(full, st_size, resharding_spec, process_group)


def reshuffle_local_shard(
    local_shard: Any,
    st_size: Any,
    sharding_spec: Any,
    resharding_spec: Any,
    pg: Any,
) -> tuple[list[Shard], list[ShardMetadata]]:
    return _reshard(local_shard, st_size, sharding_spec, resharding_spec, pg)


def reshard_local_shard(
    local_tensor: Any,
    st_size: Any,
    sharding_spec: Any,
    resharding_spec: Any,
    pg: Any,
) -> tuple[list[Shard], list[ShardMetadata]]:
    return _reshard(local_tensor, st_size, sharding_spec, resharding_spec, pg)
