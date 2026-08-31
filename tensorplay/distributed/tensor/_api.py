"""Distributed tensor values and layout conversion routines."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import tensorplay

from .. import distributed_core as dist
from ..device_mesh import DeviceMesh, _MeshEnv
from .placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)

__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "from_local",
    "ones",
    "empty",
    "full",
    "linspace",
    "logspace",
    "rand",
    "randn",
    "zeros",
]


def _current_mesh() -> DeviceMesh:
    stack = _MeshEnv.get().mesh_stack
    if not stack:
        raise RuntimeError("a DeviceMesh is required when no mesh was provided")
    return stack[-1]


def _normalize_mesh(device_mesh: DeviceMesh | None) -> DeviceMesh:
    return device_mesh if device_mesh is not None else _current_mesh()


def _normalize_placements(
    placements: Sequence[Placement] | None, mesh: DeviceMesh, ndim: int | None = None
) -> tuple[Placement, ...]:
    result = tuple(placements) if placements is not None else tuple(
        Replicate() for _ in range(mesh.ndim())
    )
    if len(result) != mesh.ndim():
        raise ValueError(
            "placements must have the same length as device_mesh.ndim; "
            f"got {len(result)} and {mesh.ndim()}"
        )
    normalized: list[Placement] = []
    for placement in result:
        if not isinstance(placement, Placement):
            raise TypeError(f"invalid placement {placement!r}")
        if _is_shard_like(placement):
            if ndim is not None and not -ndim <= placement.dim < ndim:
                raise ValueError(
                    f"shard dimension {placement.dim} is outside tensor rank {ndim}"
                )
            dim = placement.dim if placement.dim >= 0 or ndim is None else placement.dim + ndim
            if isinstance(placement, _StridedShard):
                normalized.append(_StridedShard(dim, placement.split_factor))
            else:
                normalized.append(Shard(dim))
        else:
            normalized.append(placement)
    partial_ops = {p.reduce_op for p in normalized if isinstance(p, Partial)}
    if len(partial_ops) > 1:
        raise ValueError("all Partial placements must use the same reduction")
    return tuple(normalized)


def _group(mesh: DeviceMesh, mesh_dim: int) -> Any:
    if mesh.size(mesh_dim) <= 1:
        return None
    return mesh.get_group(mesh_dim)


def _local_rank(mesh: DeviceMesh, mesh_dim: int) -> int:
    if mesh.size(mesh_dim) <= 1:
        return 0
    return mesh.get_local_rank(mesh_dim)


def _global_source_rank(mesh: DeviceMesh, mesh_dim: int, source: int) -> int:
    group = _group(mesh, mesh_dim)
    if group is None:
        return source
    ranks = dist.get_process_group_ranks(group)
    if source < 0 or source >= len(ranks):
        raise ValueError(f"source rank {source} is outside mesh dimension {mesh_dim}")
    return ranks[source]


def _shape_with_dimension(shape: Sequence[int], dim: int, size: int) -> tuple[int, ...]:
    result = list(shape)
    result[dim] = int(size)
    return tuple(result)


def _copy_into_slice(destination: Any, source: Any, dim: int, start: int = 0) -> None:
    slices = [slice(None)] * destination.dim()
    slices[dim] = slice(start, start + int(source.shape[dim]))
    destination[tuple(slices)].copy_(source)


def _pad_shards(shards: Sequence[Any], dim: int, width: int) -> list[Any]:
    result: list[Any] = []
    for shard in shards:
        current = int(shard.shape[dim])
        if current == width:
            result.append(shard.contiguous() if hasattr(shard, "contiguous") else shard)
            continue
        shape = _shape_with_dimension(shard.shape, dim, width)
        padded = shard.new_zeros(shape)
        _copy_into_slice(padded, shard, dim)
        result.append(padded.contiguous() if hasattr(padded, "contiguous") else padded)
    return result


def _distribute_shard(
    value: Any,
    placement: Shard,
    mesh: DeviceMesh,
    mesh_dim: int,
    src_data_rank: int | None,
) -> Any:
    chunks, pads = placement._split_tensor(value, mesh.size(mesh_dim), with_padding=True)
    rank = _local_rank(mesh, mesh_dim)
    if src_data_rank is None or mesh.size(mesh_dim) <= 1:
        return placement._maybe_unpad_tensor_with_sizes(
            placement.dim, chunks[rank], pads, rank, True
        )
    group = _group(mesh, mesh_dim)
    width = int(chunks[0].shape[placement.dim])
    output = value.new_empty(
        _shape_with_dimension(value.shape, placement.dim, width)
    )
    dist.scatter(output, scatter_list=chunks, src=src_data_rank, group=group)
    return placement._maybe_unpad_tensor_with_sizes(
        placement.dim, output, pads, rank, True
    )


def _replicate(
    value: Any, mesh: DeviceMesh, mesh_dim: int, src_data_rank: int | None
) -> Any:
    if mesh.size(mesh_dim) <= 1 or src_data_rank is None:
        return value
    group = _group(mesh, mesh_dim)
    source = _global_source_rank(mesh, mesh_dim, src_data_rank)
    dist.broadcast(value, src=source, group=group)
    return value


def _reduce(value: Any, placement: Partial, mesh: DeviceMesh, mesh_dim: int) -> Any:
    if mesh.size(mesh_dim) <= 1:
        return value
    group = _group(mesh, mesh_dim)
    op = {
        "sum": dist.ReduceOp.SUM,
        "avg": dist.ReduceOp.AVG,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "product": dist.ReduceOp.PRODUCT,
    }[placement.reduce_op]
    dist.all_reduce(value, op=op, group=group)
    return value


class DTensor:
    """A logical tensor represented by a local value and a mesh placement."""

    __array_priority__ = 1000

    def __init__(
        self,
        local_tensor: Any,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        *,
        shape: Sequence[int] | None = None,
        stride: Sequence[int] | None = None,
    ) -> None:
        if isinstance(local_tensor, DTensor):
            raise TypeError("local_tensor must be a plain tensor")
        self._local_tensor = local_tensor
        self._device_mesh = device_mesh
        self._placements = _normalize_placements(placements, device_mesh, local_tensor.dim())
        if shape is None:
            inferred = list(local_tensor.shape)
            for mesh_dim, placement in enumerate(self._placements):
                if _is_shard_like(placement):
                    inferred[placement.dim] *= device_mesh.size(mesh_dim)
            shape = tuple(inferred)
        self._shape = tuple(int(value) for value in shape)
        self._stride = tuple(stride) if stride is not None else tuple(local_tensor.stride())

    @classmethod
    def from_local(
        cls,
        local_tensor: Any,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        run_check: bool = False,
        shape: Sequence[int] | None = None,
        stride: Sequence[int] | None = None,
        grad_placements: Sequence[Placement] | None = None,
    ) -> "DTensor":
        del run_check, grad_placements
        if isinstance(local_tensor, DTensor):
            raise TypeError("from_local expects a plain local tensor")
        mesh = _normalize_mesh(device_mesh)
        normalized = _normalize_placements(placements, mesh, local_tensor.dim())
        return cls(local_tensor, mesh, normalized, shape=shape, stride=stride)

    @property
    def device_mesh(self) -> DeviceMesh:
        return self._device_mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        return self._placements

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def dim(self) -> int:
        return self.ndim

    @property
    def dtype(self) -> Any:
        return self._local_tensor.dtype

    @property
    def device(self) -> Any:
        return self._local_tensor.device

    def numel(self) -> int:
        return math.prod(self._shape)

    def to_local(self) -> Any:
        return self._local_tensor

    def full_tensor(self) -> Any:
        value = self._local_tensor
        shard_positions = [
            index for index, placement in enumerate(self._placements) if _is_shard_like(placement)
        ]
        for mesh_dim in reversed(range(len(self._placements))):
            placement = self._placements[mesh_dim]
            if isinstance(placement, Partial):
                value = _reduce(value, placement, self._device_mesh, mesh_dim)
                continue
            if not _is_shard_like(placement) or self._device_mesh.size(mesh_dim) <= 1:
                continue
            prior_factor = math.prod(
                self._device_mesh.size(index)
                for index in shard_positions
                if index < mesh_dim and self._placements[index].dim == placement.dim
            )
            target_size = (self._shape[placement.dim] + prior_factor - 1) // prior_factor
            width = (target_size + self._device_mesh.size(mesh_dim) - 1) // self._device_mesh.size(mesh_dim)
            padded = value.new_zeros(_shape_with_dimension(value.shape, placement.dim, width))
            _copy_into_slice(padded, value, placement.dim)
            gathered = [padded.new_empty(padded.shape) for _ in range(self._device_mesh.size(mesh_dim))]
            dist.all_gather(gathered, padded, group=_group(self._device_mesh, mesh_dim))
            value = tensorplay.cat(gathered, dim=placement.dim)
            if int(value.shape[placement.dim]) > target_size:
                slices = [slice(None)] * value.dim()
                slices[placement.dim] = slice(0, target_size)
                value = value[tuple(slices)]
        return value

    def redistribute(
        self,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        async_op: bool = False,
    ) -> "DTensor":
        if async_op:
            raise NotImplementedError("asynchronous redistribution is unavailable")
        mesh = device_mesh or self._device_mesh
        target = _normalize_placements(placements, mesh, self.ndim)
        if mesh == self._device_mesh and target == self._placements:
            return self
        return distribute_tensor(self.full_tensor(), mesh, target, src_data_rank=0)

    def detach(self) -> "DTensor":
        return type(self)(self._local_tensor.detach(), self._device_mesh, self._placements, shape=self._shape, stride=self._stride)

    def clone(self) -> "DTensor":
        return type(self)(self._local_tensor.clone(), self._device_mesh, self._placements, shape=self._shape, stride=self._stride)

    def __repr__(self) -> str:
        return (
            f"DTensor(local_tensor={self._local_tensor!r}, "
            f"device_mesh={self._device_mesh!r}, placements={self._placements!r})"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._local_tensor, name)

    def _binary(self, other: Any, operation: Callable[[Any, Any], Any]) -> "DTensor":
        if isinstance(other, DTensor):
            if other.device_mesh != self.device_mesh or other.placements != self.placements:
                raise ValueError("DTensor operands must have identical layouts")
            other = other.to_local()
        return type(self)(operation(self.to_local(), other), self.device_mesh, self.placements, shape=self.shape)

    def __add__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left + right)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left - right)

    def __rsub__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: right - left)

    def __mul__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left * right)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left / right)

    def __matmul__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left @ right)


def from_local(
    local_tensor: Any,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
    **kwargs: Any,
) -> DTensor:
    return DTensor.from_local(local_tensor, device_mesh, placements, **kwargs)


def distribute_tensor(
    tensor: Any,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
    *,
    src_data_rank: int | None = 0,
) -> DTensor:
    if isinstance(tensor, DTensor):
        mesh = _normalize_mesh(device_mesh)
        target = _normalize_placements(placements, mesh, tensor.ndim)
        if mesh != tensor.device_mesh or target != tensor.placements:
            return tensor.redistribute(mesh, target)
        return tensor
    mesh = _normalize_mesh(device_mesh)
    normalized = _normalize_placements(placements, mesh, tensor.dim())
    value = tensor.detach() if hasattr(tensor, "detach") else tensor
    for mesh_dim, placement in enumerate(normalized):
        if _is_shard_like(placement):
            value = _distribute_shard(value, placement, mesh, mesh_dim, src_data_rank)
        elif isinstance(placement, Replicate):
            value = _replicate(value, mesh, mesh_dim, src_data_rank)
    return DTensor(value, mesh, normalized, shape=tensor.shape, stride=tensor.stride())


def distribute_module(
    module: Any,
    device_mesh: DeviceMesh | None = None,
    partition_fn: Callable[[str, Any, DeviceMesh], Any] | None = None,
    input_fn: Callable[[Any, Any, DeviceMesh], Any] | None = None,
    output_fn: Callable[[Any, Any, DeviceMesh], Any] | None = None,
) -> Any:
    mesh = _normalize_mesh(device_mesh)
    if partition_fn is not None:
        for name, child in module.named_modules():
            partition_fn(name, child, mesh)
    for child in module.modules():
        for name, value in list(child._parameters.items()):
            if value is not None and not isinstance(value, DTensor):
                child._parameters[name] = distribute_tensor(value, mesh)
        for name, value in list(child._buffers.items()):
            if value is not None and not isinstance(value, DTensor):
                child._buffers[name] = distribute_tensor(value, mesh)

    if input_fn is not None:
        def pre_hook(current: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            result = input_fn(current, inputs, mesh)
            return inputs if result is None else result

        module.register_forward_pre_hook(pre_hook)
    if output_fn is not None:
        def post_hook(current: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            del inputs
            result = output_fn(current, output, mesh)
            return output if result is None else result

        module.register_forward_hook(post_hook)
    return module


def _factory(
    operation: Callable[..., Any],
    *size: Any,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
    **kwargs: Any,
) -> DTensor:
    mesh = _normalize_mesh(device_mesh)
    value = operation(*size, **kwargs)
    return distribute_tensor(value, mesh, placements)


def ones(*size: Any, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    return _factory(tensorplay.ones, *size, device_mesh=device_mesh, placements=placements, **kwargs)


def empty(*size: Any, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    return _factory(tensorplay.empty, *size, device_mesh=device_mesh, placements=placements, **kwargs)


def full(*size: Any, fill_value: Any = None, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    if fill_value is None:
        if len(size) < 2:
            raise TypeError("full expects a size and fill value")
        size, fill_value = size[:-1], size[-1]
    return _factory(tensorplay.full, *size, fill_value=fill_value, device_mesh=device_mesh, placements=placements, **kwargs)


def linspace(start: Any, end: Any, steps: int, *, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    return _factory(tensorplay.linspace, start, end, steps, device_mesh=device_mesh, placements=placements, **kwargs)


def logspace(start: Any, end: Any, steps: int, *, base: float = 10.0, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    return _factory(tensorplay.logspace, start, end, steps, base=base, device_mesh=device_mesh, placements=placements, **kwargs)


def rand(*size: Any, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    return _factory(tensorplay.rand, *size, device_mesh=device_mesh, placements=placements, **kwargs)


def randn(*size: Any, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    return _factory(tensorplay.randn, *size, device_mesh=device_mesh, placements=placements, **kwargs)


def zeros(*size: Any, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    return _factory(tensorplay.zeros, *size, device_mesh=device_mesh, placements=placements, **kwargs)
