"""Collective building blocks used by layout conversions."""

from __future__ import annotations

from typing import Any, Sequence

import tensorplay

from .. import distributed_core as dist

__all__ = [
    "all_gather_tensor",
    "fill_empty_tensor_to_shards",
    "mesh_broadcast",
    "mesh_scatter",
    "pad_tensor",
    "shard_dim_alltoall",
    "unpad_tensor",
]


def _group_ranks(mesh: Any, mesh_dim: int) -> tuple[Any, list[int]]:
    group = mesh.get_group(mesh_dim)
    return group, dist.get_process_group_ranks(group)


def pad_tensor(value: Any, dim: int, pad_size: int) -> Any:
    if pad_size <= 0:
        return value
    shape = list(value.shape)
    shape[dim] = pad_size
    padding = value.new_zeros(shape)
    return tensorplay.cat((value, padding), dim=dim)


def unpad_tensor(value: Any, dim: int, pad_size: int) -> Any:
    if pad_size <= 0:
        return value
    shape = int(value.shape[dim]) - pad_size
    if shape < 0:
        raise ValueError("padding exceeds the tensor dimension")
    slices = [slice(None)] * value.dim()
    slices[dim] = slice(0, shape)
    return value[tuple(slices)]


def mesh_broadcast(value: Any, mesh: Any, mesh_dim: int, group_src: int = 0) -> Any:
    group, ranks = _group_ranks(mesh, mesh_dim)
    if group_src < 0 or group_src >= len(ranks):
        raise ValueError("group_src is outside the mesh dimension")
    dist.broadcast(value, src=ranks[group_src], group=group)
    return value


def mesh_scatter(
    output: Any,
    scatter_list: Sequence[Any] | None,
    mesh: Any,
    mesh_dim: int,
    group_src: int = 0,
) -> Any:
    group = mesh.get_group(mesh_dim)
    dist.scatter(output, scatter_list=scatter_list, src=group_src, group=group)
    return output


def all_gather_tensor(value: Any, mesh: Any, mesh_dim: int) -> Any:
    group = mesh.get_group(mesh_dim)
    outputs = [value.new_empty(value.shape) for _ in range(mesh.size(mesh_dim))]
    dist.all_gather(outputs, value, group=group)
    return tensorplay.cat(outputs, dim=0)


def fill_empty_tensor_to_shards(value: Any, shard_sizes: Sequence[int], dim: int) -> Any:
    width = max(shard_sizes, default=0)
    if int(value.shape[dim]) == width:
        return value
    return pad_tensor(value, dim, width - int(value.shape[dim]))


def shard_dim_alltoall(value: Any, mesh: Any, mesh_dim: int, source_dim: int, target_dim: int) -> Any:
    if source_dim == target_dim or mesh.size(mesh_dim) <= 1:
        return value
    moved = value.movedim(source_dim, target_dim)
    group = mesh.get_group(mesh_dim)
    outputs = [moved.new_empty(moved.shape) for _ in range(mesh.size(mesh_dim))]
    dist.all_gather(outputs, moved, group=group)
    return tensorplay.cat(outputs, dim=target_dim)
