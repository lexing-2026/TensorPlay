"""Collective primitives and layout-cost helpers for distributed tensors."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import tensorplay

from .. import distributed_core as dist
from ._dtensor_spec import DTensorSpec
from .placement_types import Partial, Replicate, Shard, _is_shard_like

__all__ = [
    "MeshTopoInfo",
    "all_gather_tensor",
    "allgather_cost",
    "allreduce_cost",
    "check_tensor_meta",
    "fill_empty_tensor_to_shards",
    "mesh_broadcast",
    "mesh_scatter",
    "one_step_redistribute_cost",
    "pad_tensor",
    "redistribute_cost",
    "reduce_scatter_cost",
    "shard_dim_alltoall",
    "spec_to_bytes",
    "unpad_tensor",
]


def _mesh_get_process_group_impl(mesh: Any, dim: int) -> Any:
    return mesh.get_group(dim)


def _mesh_get_process_group_fake(mesh: Any, dim: int) -> Any:
    return mesh.get_group(dim)


def _shard_dim_alltoall_meta(
    value: Any, gather_dim: int, shard_dim: int, group_name: Any
) -> Any:
    from .. import _functional_collectives as funcol

    group = funcol._resolve_group(group_name)
    group_size = int(group.size())
    gathered = tensorplay.cat(
        tuple(tensorplay.empty_like(value) for _ in range(group_size)),
        dim=int(gather_dim),
    )
    chunk_size = int(gathered.shape[int(shard_dim)]) // group_size
    return gathered.narrow(
        int(shard_dim), int(group.rank()) * chunk_size, chunk_size
    ).contiguous()


def _shard_dim_alltoall_setup_context(
    context: Any, inputs: tuple[Any, ...], output: Any
) -> None:
    del output
    _, gather_dim, shard_dim, group_name = inputs
    context.gather_dim = gather_dim
    context.shard_dim = shard_dim
    context.group_name = group_name


def _shard_dim_alltoall_backward(context: Any, grad_output: Any) -> tuple[Any, None, None, None]:
    return (
        _shard_dim_alltoall_impl(
            grad_output,
            context.shard_dim,
            context.gather_dim,
            context.group_name,
        ),
        None,
        None,
        None,
    )


def _shard_dim_alltoall_impl(
    value: Any, gather_dim: int, shard_dim: int, group_name: Any
) -> Any:
    from .. import _functional_collectives as funcol

    group = funcol._resolve_group(group_name)
    gather_dim = _tensor_dim(value, int(gather_dim))
    shard_dim = _tensor_dim(value, int(shard_dim))
    group_size = int(group.size())
    if gather_dim == shard_dim or group_size <= 1:
        return value
    source_shards, _ = Shard(shard_dim)._split_tensor(
        value, group_size, with_padding=True, contiguous=True
    )
    outputs = [value.new_empty(tuple(shard.shape)) for shard in source_shards]
    work = dist.all_to_all(outputs, source_shards, group=group)
    if work is not None and hasattr(work, "wait"):
        work.wait()
    return tensorplay.cat(tuple(outputs), dim=gather_dim).contiguous()


def _make_shard_dim_alltoall_function() -> Any:
    from tensorplay.autograd.function import Function

    class ShardDimAllToAll(Function):
        @staticmethod
        def forward(context: Any, value: Any, gather_dim: int, shard_dim: int, group_name: Any) -> Any:
            _shard_dim_alltoall_setup_context(
                context, (value, gather_dim, shard_dim, group_name), None
            )
            return _shard_dim_alltoall_impl(
                value, gather_dim, shard_dim, group_name
            )

        @staticmethod
        def backward(context: Any, grad_output: Any) -> tuple[Any, None, None, None]:
            return _shard_dim_alltoall_backward(context, grad_output)

    return ShardDimAllToAll


_ShardDimAllToAll = _make_shard_dim_alltoall_function()


def _mesh_ndim(mesh: Any) -> int:
    value = getattr(mesh, "ndim")
    return int(value() if callable(value) else value)


def _mesh_dim(mesh: Any, mesh_dim: int | str) -> int:
    if isinstance(mesh_dim, str):
        names = getattr(mesh, "mesh_dim_names", None)
        if names is None:
            raise KeyError(mesh_dim)
        try:
            mesh_dim = names.index(mesh_dim)
        except ValueError as error:
            raise KeyError(mesh_dim) from error
    dim = int(mesh_dim)
    if dim < 0:
        dim += _mesh_ndim(mesh)
    if dim < 0 or dim >= _mesh_ndim(mesh):
        raise ValueError(f"mesh dimension {mesh_dim} is outside the mesh")
    return dim


def _tensor_dim(value: Any, dim: int) -> int:
    ndim = int(value.dim())
    original = dim
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"tensor dimension {original} is outside tensor rank {ndim}")
    return dim


def _is_meta(value: Any) -> bool:
    return bool(getattr(value, "is_meta", False))


def _participates(mesh: Any) -> bool:
    coordinate = getattr(mesh, "get_coordinate", None)
    return coordinate is None or coordinate() is not None


def _group(mesh: Any, mesh_dim: int) -> Any:
    return mesh.get_group(_mesh_dim(mesh, mesh_dim))


def _group_size(mesh: Any, mesh_dim: int) -> int:
    return int(mesh.size(_mesh_dim(mesh, mesh_dim)))


def _validate_group_src(group_src: int | None, group_size: int) -> None:
    if group_src is None:
        return
    if type(group_src) is not int or group_src < 0 or group_src >= group_size:
        raise ValueError("group_src must be a valid rank within the mesh dimension")


def pad_tensor(value: Any, pad_dim: int, pad_size: int) -> Any:
    """Append zero rows along one dimension."""
    pad_dim = _tensor_dim(value, int(pad_dim))
    pad_size = int(pad_size)
    if pad_size < 0:
        raise ValueError("pad_size must be non-negative")
    if pad_size == 0:
        return value
    shape = list(value.shape)
    shape[pad_dim] = pad_size
    return tensorplay.cat((value, value.new_zeros(tuple(shape))), dim=pad_dim)


def unpad_tensor(value: Any, pad_dim: int, pad_size: int) -> Any:
    """Remove zero padding from one dimension."""
    pad_dim = _tensor_dim(value, int(pad_dim))
    pad_size = int(pad_size)
    if pad_size < 0:
        raise ValueError("pad_size must be non-negative")
    if pad_size == 0:
        return value
    length = int(value.shape[pad_dim]) - pad_size
    if length < 0:
        raise ValueError("padding exceeds the tensor dimension")
    narrow = getattr(value, "narrow", None)
    if narrow is not None:
        return narrow(pad_dim, 0, length)
    slices = [slice(None)] * int(value.dim())
    slices[pad_dim] = slice(0, length)
    return value[tuple(slices)]


def mesh_scatter(
    output: Any,
    scatter_list: Sequence[Any] | None,
    mesh: Any,
    mesh_dim: int = 0,
    async_op: bool = False,
    *,
    group_src: int | None = 0,
) -> Any:
    """Scatter one tensor per mesh rank using a mesh-relative source rank."""
    if _is_meta(output) or group_src is None or not _participates(mesh):
        return None
    mesh_dim = _mesh_dim(mesh, mesh_dim)
    group_size = _group_size(mesh, mesh_dim)
    _validate_group_src(group_src, group_size)
    group = _group(mesh, mesh_dim)
    local_rank = int(dist.get_rank(group))
    source_list = scatter_list if local_rank == group_src else None
    if local_rank == group_src:
        if source_list is None or len(source_list) != group_size:
            raise ValueError("scatter_list must contain one tensor per mesh rank")
    ranks = dist.get_process_group_ranks(group)
    return dist.scatter(
        output,
        scatter_list=source_list,
        src=int(ranks[int(group_src)]),
        group=group,
        async_op=async_op,
    )


def mesh_broadcast(
    value: Any,
    mesh: Any,
    mesh_dim: int = 0,
    async_op: bool = False,
    *,
    group_src: int | None = 0,
) -> Any:
    """Broadcast in a mesh dimension using a mesh-relative source rank."""
    if _is_meta(value) or group_src is None or not _participates(mesh):
        return None
    mesh_dim = _mesh_dim(mesh, mesh_dim)
    group_size = _group_size(mesh, mesh_dim)
    _validate_group_src(group_src, group_size)
    group = _group(mesh, mesh_dim)
    ranks = dist.get_process_group_ranks(group)
    return dist.broadcast(
        value,
        src=int(ranks[int(group_src)]),
        group=group,
        async_op=async_op,
    )


def all_gather_tensor(
    value: Any,
    mesh: Any,
    mesh_dim: int,
    gather_dim: int = 0,
) -> Any:
    """Gather equal-shaped local tensors and concatenate one tensor dimension."""
    if not _participates(mesh):
        return value
    mesh_dim = _mesh_dim(mesh, mesh_dim)
    gather_dim = _tensor_dim(value, int(gather_dim))
    group_size = _group_size(mesh, mesh_dim)
    if group_size <= 1:
        return value
    group = _group(mesh, mesh_dim)
    outputs = [value.new_empty(tuple(value.shape)) for _ in range(group_size)]
    dist.all_gather(outputs, value, group=group)
    return tensorplay.cat(tuple(outputs), dim=gather_dim)


def shard_dim_alltoall(
    value: Any,
    gather_dim: int,
    shard_dim: int,
    mesh: Any,
    mesh_dim: int,
) -> Any:
    """Exchange equal chunks so a shard moves from one tensor dimension to another."""
    if not _participates(mesh):
        return value
    gather_dim = _tensor_dim(value, int(gather_dim))
    shard_dim = _tensor_dim(value, int(shard_dim))
    mesh_dim = _mesh_dim(mesh, mesh_dim)
    group_size = _group_size(mesh, mesh_dim)
    if gather_dim == shard_dim or group_size <= 1:
        return value
    group = _mesh_get_process_group_impl(mesh, mesh_dim)
    return _ShardDimAllToAll.apply(
        value, gather_dim, shard_dim, group.group_name
    )


def fill_empty_tensor_to_shards(
    shards: list[Any], shard_dim: int, num_empty_tensors: int
) -> list[Any]:
    """Append zero-length tensors when a gather has fewer logical shards."""
    if type(num_empty_tensors) is not int or num_empty_tensors < 0:
        raise ValueError("num_empty_tensors must be non-negative")
    if num_empty_tensors == 0:
        return shards
    if not shards:
        raise ValueError("at least one tensor is required to infer shard metadata")
    shard_dim = _tensor_dim(shards[0], int(shard_dim))
    shape = list(shards[0].shape)
    shape[shard_dim] = 0
    empty = shards[0].new_zeros(tuple(shape))
    shards.extend(empty for _ in range(num_empty_tensors))
    return shards


def check_tensor_meta(
    local_tensor: Any, check_shape_stride: bool = False
) -> None:
    """Verify metadata that must be identical before a distributed tensor is used."""
    metadata: dict[str, Any] = {
        "dtype": local_tensor.dtype,
        "requires_grad": bool(local_tensor.requires_grad),
    }
    if check_shape_stride:
        metadata.update(
            {
                "shape": tuple(int(value) for value in local_tensor.shape),
                "stride": tuple(int(value) for value in local_tensor.stride()),
            }
        )
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, metadata)
    if not all(item == metadata for item in gathered):
        raise ValueError("inconsistent local tensor metadata across ranks")


def _dtype_itemsize(dtype: Any) -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if callable(itemsize):
        itemsize = itemsize()
    if itemsize is not None:
        return int(itemsize)
    name = str(dtype).rsplit(".", 1)[-1].lower()
    sizes = {
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "complex32": 4,
        "int64": 8,
        "uint64": 8,
        "float64": 8,
        "complex64": 8,
        "complex128": 16,
    }
    try:
        return sizes[name]
    except KeyError as error:
        raise TypeError(f"cannot determine the element size of {dtype!r}") from error


def spec_to_bytes(spec: DTensorSpec) -> int:
    if spec.tensor_meta is None or spec.shape is None:
        raise AssertionError("tensor metadata is required for a cost estimate")
    return _dtype_itemsize(spec.tensor_meta.dtype) * math.prod(spec.shape)


@dataclass
class MeshTopoInfo:
    """Mesh dimensions and the communication parameters used by the cost model."""

    mesh: Any
    mesh_dim_devices: list[int]
    mesh_dim_bandwidth: list[float]
    mesh_dim_latency: list[float]

    @staticmethod
    @lru_cache(None)
    def build_from_mesh(mesh: Any) -> "MeshTopoInfo":
        ndim = _mesh_ndim(mesh)
        local_devices = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0)
        if local_devices <= 0:
            local_devices = int(mesh.size())
        devices: list[int] = [1] * ndim
        bandwidth: list[float] = [87.7] * ndim
        latency: list[float] = [0.6] * ndim
        total = 1
        for dim in reversed(range(ndim)):
            count = int(mesh.size(dim))
            devices[dim] = count
            total *= count
            if total > local_devices:
                bandwidth[dim] *= 0.22
                latency[dim] = 2.7
        return MeshTopoInfo(mesh, devices, bandwidth, latency)


def _mesh_cost_args(mesh_topo: MeshTopoInfo, mesh_dim: int) -> tuple[int, float, float]:
    mesh_dim = _mesh_dim(mesh_topo.mesh, mesh_dim)
    devices = int(mesh_topo.mesh_dim_devices[mesh_dim])
    if devices <= 0:
        raise ValueError("mesh dimensions must contain at least one device")
    return devices, float(mesh_topo.mesh_dim_bandwidth[mesh_dim]), float(
        mesh_topo.mesh_dim_latency[mesh_dim]
    )


def allgather_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    devices, bandwidth, latency_per_hop = _mesh_cost_args(mesh_topo, mesh_dim)
    hops = devices - 1
    latency = 6.6 + hops * latency_per_hop
    bandwidth_cost = (bytes_gb * hops / devices) / bandwidth
    return latency + bandwidth_cost * 1e6


def allreduce_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    devices, bandwidth, latency_per_hop = _mesh_cost_args(mesh_topo, mesh_dim)
    hops = 2 * (devices - 1)
    latency = 6.6 + hops * latency_per_hop
    bandwidth_cost = (bytes_gb * hops / devices) / bandwidth
    return latency + bandwidth_cost * 1e6


def reduce_scatter_cost(
    bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int
) -> float:
    devices, bandwidth, latency_per_hop = _mesh_cost_args(mesh_topo, mesh_dim)
    hops = devices - 1
    latency = 6.6 + hops * latency_per_hop
    bandwidth_cost = (bytes_gb * hops / devices) / bandwidth
    return latency + bandwidth_cost * 1e6


def _compute_placement_transition_cost(
    current_placement: Any,
    target_placement: Any,
    mesh_topo: MeshTopoInfo,
    mesh_dim: int,
    comm_bytes_gb: float,
) -> tuple[float, float]:
    if current_placement == target_placement:
        return 0.0, comm_bytes_gb
    devices, _, _ = _mesh_cost_args(mesh_topo, mesh_dim)
    current_shard = _is_shard_like(current_placement)
    target_shard = _is_shard_like(target_placement)
    current_partial = isinstance(current_placement, Partial)
    target_partial = isinstance(target_placement, Partial)
    current_replicate = isinstance(current_placement, Replicate)
    target_replicate = isinstance(target_placement, Replicate)

    if current_shard and target_replicate:
        comm_bytes_gb *= devices
        return allgather_cost(comm_bytes_gb, mesh_topo, mesh_dim), comm_bytes_gb
    if current_shard and target_shard:
        return allgather_cost(comm_bytes_gb, mesh_topo, mesh_dim) + 1.0, comm_bytes_gb
    if current_partial and target_replicate:
        return allreduce_cost(comm_bytes_gb, mesh_topo, mesh_dim), comm_bytes_gb
    if current_partial and target_shard:
        cost = reduce_scatter_cost(comm_bytes_gb, mesh_topo, mesh_dim)
        return cost, comm_bytes_gb / devices
    if current_shard and target_partial:
        return float("inf"), comm_bytes_gb
    if current_partial and target_partial:
        return float("inf"), comm_bytes_gb
    if current_replicate and target_shard:
        return 0.0, comm_bytes_gb / devices
    if current_replicate and target_partial:
        return 0.0, comm_bytes_gb
    if current_replicate and target_replicate:
        return 0.0, comm_bytes_gb
    return float("inf"), comm_bytes_gb


def _spec_num_shards(spec: DTensorSpec) -> int:
    value = getattr(spec, "num_shards", None)
    if value is not None:
        return max(1, int(value() if callable(value) else value))
    result = 1
    for mesh_dim, placement in enumerate(spec.placements):
        if _is_shard_like(placement):
            result *= int(spec.mesh.size(mesh_dim))
    return max(1, result)


def _is_spec_replicated(spec: DTensorSpec) -> bool:
    value = getattr(spec, "is_replicated", False)
    return bool(value() if callable(value) else value)


def one_step_redistribute_cost(
    current_spec: DTensorSpec, target_spec: DTensorSpec
) -> float:
    if current_spec.mesh != target_spec.mesh:
        return float("inf")
    if current_spec.placements == target_spec.placements:
        return 0.0
    differing = [
        (mesh_dim, current, target)
        for mesh_dim, (current, target) in enumerate(
            zip(current_spec.placements, target_spec.placements)
        )
        if current != target
    ]
    if len(differing) != 1:
        raise ValueError(
            "one_step_redistribute_cost expects one placement to differ"
        )
    mesh_dim, current, target = differing[0]
    topo = MeshTopoInfo.build_from_mesh(current_spec.mesh)
    bytes_gb = spec_to_bytes(current_spec) / _spec_num_shards(current_spec)
    bytes_gb /= 1024**3
    return _compute_placement_transition_cost(
        current, target, topo, mesh_dim, bytes_gb
    )[0]


def redistribute_cost(
    current_spec: DTensorSpec, target_spec: DTensorSpec
) -> float:
    if current_spec.mesh != target_spec.mesh:
        return float("inf")
    if _is_spec_replicated(current_spec):
        return 0.0
    if current_spec.placements == target_spec.placements:
        return 0.0
    coordinate = getattr(current_spec.mesh, "get_coordinate", None)
    if coordinate is not None and coordinate() is None:
        return 0.0
    if len(current_spec.placements) != len(target_spec.placements):
        return float("inf")
    topo = MeshTopoInfo.build_from_mesh(current_spec.mesh)
    bytes_gb = spec_to_bytes(current_spec) / _spec_num_shards(current_spec)
    bytes_gb /= 1024**3
    cost = 0.0
    current = list(current_spec.placements)
    target = tuple(target_spec.placements)
    for mesh_dim in reversed(range(len(current))):
        if current[mesh_dim] == target[mesh_dim]:
            continue
        step, bytes_gb = _compute_placement_transition_cost(
            current[mesh_dim], target[mesh_dim], topo, mesh_dim, bytes_gb
        )
        if math.isinf(step):
            return step
        cost += step
        current[mesh_dim] = target[mesh_dim]
    return cost
