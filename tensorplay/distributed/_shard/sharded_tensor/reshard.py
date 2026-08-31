"""Metadata and local-shard reshaping helpers."""

from typing import Any

__all__ = ["get_idx_from_placements", "build_reshard_metadata", "reshuffle_local_shard", "reshard_local_shard"]


def get_idx_from_placements(placements: Any, current_rank: int) -> int:
    for index, placement in enumerate(placements):
        rank = placement.rank() if hasattr(placement, "rank") else int(str(placement).split(":")[1].split("/")[0])
        if rank == current_rank:
            return index
    return -1


def build_reshard_metadata(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return []


def reshuffle_local_shard(local_shard: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return local_shard


def reshard_local_shard(local_shard: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return local_shard
