"""Shard construction utilities shared by FSDP entrypoints."""

import math
from typing import Any

from ..tensor import DTensor, Replicate, Shard, distribute_tensor

__all__ = ["_get_remote_device_str", "_create_chunk_sharded_tensor", "_create_chunk_dtensor", "_all_gather_dtensor"]


def _get_remote_device_str(rank: int, device_type: str, num_devices_per_node: int) -> str:
    if device_type.lower() == "cpu":
        return f"rank:{rank}/{device_type}"
    return f"rank:{rank}/{device_type}:{rank % max(1, num_devices_per_node)}"


def _create_chunk_sharded_tensor(tensor: Any, rank: int, world_size: int, num_devices_per_node: int, pg: Any, device: Any = None) -> Any:
    del num_devices_per_node, pg, device
    chunks = tensor.chunk(world_size, dim=0)
    return chunks[rank].clone() if rank < len(chunks) else tensor.new_empty((0,) + tuple(tensor.shape[1:]))


def _create_chunk_dtensor(tensor: Any, rank: int, device_mesh: Any) -> DTensor:
    del rank
    placements = [Replicate() for _ in range(device_mesh.ndim())]
    placements[-1] = Shard(0)
    return distribute_tensor(tensor.detach(), device_mesh, placements)


def _all_gather_dtensor(tensor: DTensor, root_mesh: Any = None) -> Any:
    if root_mesh is not None and root_mesh != tensor.device_mesh:
        raise ValueError("tensor mesh does not match root mesh")
    placements = list(tensor.placements)
    placements[-1] = Replicate()
    return tensor.redistribute(placements=placements).to_local()
