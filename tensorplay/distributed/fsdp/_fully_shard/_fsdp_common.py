"""Mesh and tensor utilities for composable sharding."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import tensorplay as tp

from .._common_utils import TrainingState
from .._flat_param import FlatParamShardMetadata
from ...tensor import Replicate, Shard

__all__ = [
    "DataParallelMeshInfo",
    "FSDPMeshInfo",
    "DDPMeshInfo",
    "HSDPMeshInfo",
    "TrainingState",
    "ShardPlacementResult",
    "resolve_shard_placement",
]


def _dynamo_disable(function: Callable[..., Any]) -> Callable[..., Any]:
    return function


def _disable_functorch_if_active(function: Callable[..., Any]) -> Callable[..., Any]:
    return function


@dataclass
class DataParallelMeshInfo:
    mesh: Any
    shard_mesh_dim: int | str = 0
    replicate_mesh_dim: int | str | None = None

    @property
    def shard_mesh(self) -> Any:
        return self.mesh

    @property
    def shard_world_size(self) -> int:
        return int(self.mesh.size(self.shard_mesh_dim))

    @property
    def replicate_world_size(self) -> int:
        return 1 if self.replicate_mesh_dim is None else int(self.mesh.size(self.replicate_mesh_dim))


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    pass


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    shard_mesh_dim: int | str = 0


@dataclass
class HSDPMeshInfo(DataParallelMeshInfo):
    replicate_mesh_dim: int | str = 0


def _raise_assert_with_print(message: str) -> None:
    raise AssertionError(message)


def _is_composable_with_fsdp(module: Any) -> bool:
    return not bool(getattr(module, "_fsdp_state", None))


def _get_dim0_padded_size(tensor_size: int, dim0_factor: int) -> int:
    return ((int(tensor_size) + dim0_factor - 1) // dim0_factor) * dim0_factor


def _chunk_with_empty(tensor: Any, num_chunks: int, dim: int) -> list[Any]:
    chunks = list(tensor.chunk(num_chunks, dim=dim))
    width = (int(tensor.shape[dim]) + num_chunks - 1) // num_chunks
    while len(chunks) < num_chunks:
        shape = list(tensor.shape)
        shape[dim] = 0
        chunks.append(tensor.new_empty(shape))
    return chunks


def _get_dim_chunked_size(chunk: Any, unchunked_size: int, dim: int) -> int:
    del unchunked_size
    return int(chunk.shape[dim])


def _from_local_no_grad(local_tensor: Any, sharding_spec: Any) -> Any:
    return sharding_spec(local_tensor) if callable(sharding_spec) else local_tensor


def _to_dtype_if_needed(tensor: Any, dtype: Any) -> Any:
    return tensor if dtype is None or tensor.dtype == dtype else tensor.to(dtype=dtype)


def _cast_fp_tensor(dtype: Any, value: Any) -> Any:
    if dtype is None or not getattr(value, "is_floating_point", lambda: False)():
        return value
    return value.to(dtype=dtype)


def is_bw() -> bool:
    return False


@dataclass
class ShardPlacementResult:
    placement: Shard
    mesh: Any | None = None


def resolve_shard_placement(result: Any, default_mesh_info: DataParallelMeshInfo) -> tuple[Shard, Any]:
    if result is None:
        return Shard(0), default_mesh_info.mesh
    if isinstance(result, Shard):
        return result, default_mesh_info.mesh
    if isinstance(result, ShardPlacementResult):
        return result.placement, result.mesh or default_mesh_info.mesh
    raise TypeError("shard placement must be a Shard or ShardPlacementResult")
