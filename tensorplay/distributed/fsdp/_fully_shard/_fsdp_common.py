"""Mesh and tensor utilities for composable sharding."""

import functools
from dataclasses import dataclass, field
from typing import Any, Callable

import tensorplay as tp

from ... import distributed_core as dist
from .._common_utils import TrainingState
from ...tensor import DTensor, Replicate, Shard

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
    disabled = tp.compiler.disable(
        function,
        recursive=True,
        reason="FSDP hooks run eagerly",
    )

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return disabled(*args, **kwargs)

    return wrapper


def _disable_functorch_if_active(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        c_api = getattr(tp, "_C", None)
        active_check = getattr(c_api, "_are_functorch_transforms_active", None)
        disable_context = getattr(c_api, "_DisableFuncTorch", None)
        try:
            active = bool(active_check()) if callable(active_check) else False
        except (AttributeError, RuntimeError):
            active = False
        if active and callable(disable_context):
            with disable_context():
                return function(*args, **kwargs)
        return function(*args, **kwargs)

    return wrapper


def _mesh_ndim(mesh: Any) -> int:
    value = getattr(mesh, "ndim", None)
    if value is None:
        shape = getattr(mesh, "shape", None)
        if shape is None:
            raise TypeError("mesh must expose ndim or shape")
        return len(shape)
    value = value() if callable(value) else value
    result = int(value)
    if result <= 0:
        raise ValueError("mesh must have at least one dimension")
    return result


def _mesh_dim_index(mesh: Any, mesh_dim: int | str | None) -> int | None:
    if mesh_dim is None:
        return None
    if isinstance(mesh_dim, bool):
        raise TypeError("mesh dimension must be an integer or string")
    if isinstance(mesh_dim, str):
        names = getattr(mesh, "mesh_dim_names", None)
        if names is None or mesh_dim not in names:
            raise KeyError(mesh_dim)
        return int(names.index(mesh_dim))
    dim = int(mesh_dim)
    ndim = _mesh_ndim(mesh)
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("mesh dimension is outside the mesh")
    return dim


def _mesh_group(mesh: Any, mesh_dim: int | str | None) -> Any:
    dim = _mesh_dim_index(mesh, mesh_dim)
    if dim is None or not dist.is_initialized():
        return None
    size = int(mesh.size(dim))
    if size <= 1:
        return None
    getter = getattr(mesh, "get_group", None)
    if not callable(getter):
        raise TypeError("mesh must provide get_group for distributed sharding")
    return getter(dim)


def _mesh_rank(mesh: Any, mesh_dim: int | str | None, group: Any) -> int:
    if group is not None:
        try:
            return int(dist.get_rank(group))
        except (AttributeError, RuntimeError, ValueError):
            rank_method = getattr(group, "rank", None)
            if callable(rank_method):
                return int(rank_method())
    getter = getattr(mesh, "get_local_rank", None)
    if callable(getter):
        try:
            return int(getter(mesh_dim))
        except (RuntimeError, ValueError, KeyError):
            pass
    return 0


@dataclass
class DataParallelMeshInfo:
    mesh: Any
    shard_mesh_dim: int | str | None = 0
    replicate_mesh_dim: int | str | None = None
    dp_mesh_dims: Any = None
    spmd_mesh: Any = field(default=None, repr=False)
    is_spmd_mesh: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mesh is None:
            raise ValueError("mesh cannot be None")
        _mesh_ndim(self.mesh)
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise ValueError("at least one data-parallel mesh dimension is required")
        self.shard_mesh_dim = _mesh_dim_index(self.mesh, self.shard_mesh_dim)
        self.replicate_mesh_dim = _mesh_dim_index(self.mesh, self.replicate_mesh_dim)
        if (
            self.shard_mesh_dim is not None
            and self.replicate_mesh_dim is not None
            and self.shard_mesh_dim == self.replicate_mesh_dim
        ):
            raise ValueError("shard and replicate dimensions must be different")
        self.is_spmd_mesh = self.dp_mesh_dims is not None

    @property
    def shard_mesh(self) -> Any:
        return self.mesh

    @property
    def shard_world_size(self) -> int:
        if self.shard_mesh_dim is None:
            return 1
        return int(self.mesh.size(self.shard_mesh_dim))

    @property
    def replicate_world_size(self) -> int:
        return 1 if self.replicate_mesh_dim is None else int(self.mesh.size(self.replicate_mesh_dim))


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise ValueError("FSDP requires a shard mesh dimension")
        self.shard_mesh_size = int(self.mesh.size(self.shard_mesh_dim))
        self.shard_process_group = _mesh_group(self.mesh, self.shard_mesh_dim)
        self.shard_mesh_rank = _mesh_rank(
            self.mesh, self.shard_mesh_dim, self.shard_process_group
        )
        self.reduce_scatter_process_group = None


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    shard_mesh_dim: int | str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise ValueError("DDP requires a replicate mesh dimension")
        self.replicate_mesh_size = int(self.mesh.size(self.replicate_mesh_dim))
        self.replicate_process_group = _mesh_group(
            self.mesh, self.replicate_mesh_dim
        )
        self.replicate_mesh_rank = _mesh_rank(
            self.mesh, self.replicate_mesh_dim, self.replicate_process_group
        )


@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    replicate_mesh_dim: int | str = 0


def _raise_assert_with_print(message: str) -> None:
    raise AssertionError(message)


def _is_composable_with_fsdp(module: Any) -> bool:
    if getattr(module, "_fsdp_state", None) is not None:
        return False
    from ..._composable.contract import _get_registry

    registry = _get_registry(module)
    if registry is None:
        return True
    return not any(
        key in registry
        for key in (
            "replicate",
            "__replicate_state_key__",
            "__replicate_with_fsdp_state__",
        )
    )


def _get_dim0_padded_size(tensor_size: int, dim0_factor: int) -> int:
    return ((int(tensor_size) + dim0_factor - 1) // dim0_factor) * dim0_factor


def _chunk_with_empty(tensor: Any, num_chunks: int, dim: int) -> list[Any]:
    if isinstance(num_chunks, bool) or num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    chunks = list(tensor.chunk(num_chunks, dim=dim))
    while len(chunks) < num_chunks:
        shape = list(tensor.shape)
        shape[dim] = 0
        chunks.append(tensor.new_empty(shape))
    return chunks


def _get_dim_chunked_size(chunk: Any, unchunked_size: Any, dim: int) -> Any:
    if int(getattr(chunk, "numel", lambda: 0)()) != 0:
        return chunk.shape[dim]
    if isinstance(unchunked_size, int):
        return 0
    shape = list(unchunked_size)
    shape[dim] = 0
    try:
        return type(unchunked_size)(shape)
    except (TypeError, ValueError):
        return tuple(shape)


def _from_local_no_grad(local_tensor: Any, sharding_spec: Any) -> Any:
    if callable(sharding_spec):
        return sharding_spec(local_tensor)
    mesh = getattr(sharding_spec, "mesh", None)
    placements = getattr(sharding_spec, "placements", None)
    if mesh is None or placements is None:
        return local_tensor
    shape = getattr(sharding_spec, "shape", None)
    stride = getattr(sharding_spec, "stride", None)
    return DTensor(
        local_tensor,
        mesh,
        placements,
        shape=shape,
        stride=stride,
    )


def _to_dtype_if_needed(tensor: Any, dtype: Any) -> Any:
    return tensor if dtype is None or tensor.dtype == dtype else tensor.to(dtype=dtype)


def _cast_fp_tensor(dtype: Any, value: Any) -> Any:
    if dtype is None or not getattr(value, "is_floating_point", lambda: False)():
        return value
    return value.to(dtype=dtype)


def is_bw() -> bool:
    current_graph_task = getattr(getattr(tp, "_C", None), "_current_graph_task_id", None)
    if not callable(current_graph_task):
        return False
    try:
        return int(current_graph_task()) != -1
    except (AttributeError, RuntimeError, TypeError):
        return False


@dataclass
class ShardPlacementResult:
    placement: Shard | Replicate | None = None
    mesh: Any | None = None
    mesh_info: DataParallelMeshInfo | None = None


def resolve_shard_placement(result: Any, default_mesh_info: DataParallelMeshInfo) -> tuple[Any, Any]:
    if result is None:
        return Shard(0), default_mesh_info.mesh
    if isinstance(result, (Shard, Replicate)):
        return result, default_mesh_info.mesh
    if isinstance(result, ShardPlacementResult):
        placement = result.placement if result.placement is not None else Shard(0)
        if result.mesh_info is not None:
            return placement, result.mesh_info.mesh
        if isinstance(result.mesh, DataParallelMeshInfo):
            return placement, result.mesh.mesh
        return placement, result.mesh or default_mesh_info.mesh
    raise TypeError("shard placement must be a Shard or ShardPlacementResult")
