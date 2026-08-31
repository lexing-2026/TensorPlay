"""Collective helpers for context-parallel attention."""

from __future__ import annotations

from typing import Any

from .... import distributed_core as dist

__all__ = ["flex_cp_allgather"]


def flex_cp_allgather(value: Any, group: Any = None, **kwargs: Any) -> Any:
    del kwargs
    if group is None or dist.get_world_size(group) <= 1:
        return value
    outputs = [value.new_empty(value.shape) for _ in range(dist.get_world_size(group))]
    dist.all_gather(outputs, value, group=group)
    import tensorplay

    return tensorplay.cat(outputs, dim=0)
