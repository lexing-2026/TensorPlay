"""Shard construction utilities shared by FSDP entrypoints."""

import itertools
import math
from typing import Any

from ..tensor import DTensor, Replicate, Shard, distribute_tensor
from .._shard.metadata import ShardMetadata
from .._shard.sharded_tensor import (
    Shard as LocalShard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)

__all__ = ["_get_remote_device_str", "_create_chunk_sharded_tensor", "_create_chunk_dtensor", "_all_gather_dtensor"]


def _get_remote_device_str(rank: int, device_type: str, num_devices_per_node: int) -> str:
    if device_type.lower() == "cpu":
        return f"rank:{rank}/{device_type}"
    return f"rank:{rank}/{device_type}:{rank % max(1, num_devices_per_node)}"


def _create_chunk_sharded_tensor(tensor: Any, rank: int, world_size: int, num_devices_per_node: int, pg: Any, device: Any = None) -> Any:
    device_type = str(getattr(device, "type", None) or getattr(getattr(tensor, "device", None), "type", "cpu"))
    chunks = tensor.chunk(world_size, dim=0)
    local_shards = []
    if rank < len(chunks):
        local = chunks[rank].detach().clone()
        local_shards.append(
            LocalShard(
                local,
                ShardMetadata(
                    [math.ceil(int(tensor.shape[0]) / world_size) * rank]
                    + [0] * (int(tensor.dim()) - 1),
                    list(local.shape),
                    _get_remote_device_str(rank, device_type, num_devices_per_node),
                ),
            )
        )
    offsets = [0] * (int(tensor.dim()) - 1)
    chunk_offsets = [
        [sum(int(part.shape[0]) for part in chunks[:index])] + offsets
        for index in range(len(chunks))
    ]
    metadata = ShardedTensorMetadata(
        [
            ShardMetadata(
                offset,
                list(chunk.shape),
                _get_remote_device_str(index, device_type, num_devices_per_node),
            )
            for index, (offset, chunk) in enumerate(zip(chunk_offsets, chunks))
        ],
        tuple(int(value) for value in tensor.shape),
        TensorProperties.create_from_tensor(tensor),
    )
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards, metadata, process_group=pg
    )


def _mesh_ndim(mesh: Any) -> int:
    value = getattr(mesh, "ndim", None)
    value = value() if callable(value) else value
    if value is None:
        value = len(getattr(mesh, "shape"))
    return int(value)


def _create_chunk_dtensor(tensor: Any, rank: int, device_mesh: Any) -> DTensor:
    del rank
    root_getter = getattr(device_mesh, "_get_root_mesh", None)
    root_mesh = root_getter() if callable(root_getter) else device_mesh
    root_ndim = _mesh_ndim(root_mesh)
    if root_ndim < 2:
        raise RuntimeError("FSDP DTensor chunking requires a parent mesh with at least 2 dimensions")

    tensor = tensor.detach().clone()
    if not isinstance(tensor, DTensor):
        replicate = [Replicate() for _ in range(root_ndim)]
        sharded = [Replicate() for _ in range(root_ndim)]
        sharded[0] = Shard(0)
        return distribute_tensor(tensor, root_mesh, replicate).redistribute(
            device_mesh=root_mesh,
            placements=sharded,
        )

    tp_placement = tensor.placements[0]
    local_tensor = tensor.to_local()
    replicate = [Replicate() for _ in range(root_ndim)]
    replicate[-1] = tp_placement
    sharded = [Replicate() for _ in range(root_ndim)]
    sharded[-2] = Shard(0)
    sharded[-1] = tp_placement
    return DTensor.from_local(
        local_tensor,
        root_mesh,
        replicate,
        run_check=False,
    ).redistribute(
        device_mesh=root_mesh,
        placements=sharded,
    )


def _all_gather_dtensor(tensor: DTensor, root_mesh: Any = None) -> Any:
    if root_mesh != tensor.device_mesh:
        raise ValueError("tensor mesh does not match root mesh")
    placements = list(tensor.placements)
    for mesh_dim in range(max(0, len(placements) - 1)):
        placements[mesh_dim] = Replicate()
    return tensor.redistribute(
        device_mesh=tensor.device_mesh, placements=placements
    ).to_local()
