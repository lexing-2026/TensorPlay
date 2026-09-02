"""Immutable metadata records for distributed tensor layouts."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Any

from .placement_types import Partial, Placement, Replicate, Shard, _is_shard_like

__all__ = ["DTensorSpec", "ShardOrderEntry", "TensorMeta"]


class _DimMap(list[int]):
    def __call__(self) -> tuple[int, ...]:
        return tuple(self)


class _BoolValue(int):
    def __new__(cls, value: bool) -> "_BoolValue":
        return int.__new__(cls, bool(value))

    def __bool__(self) -> bool:
        return int(self) != 0

    def __call__(self) -> bool:
        return bool(self)


_TensorMetaBase = namedtuple("_TensorMetaBase", ("shape", "stride", "dtype"))


class TensorMeta(_TensorMetaBase):
    __slots__ = ()
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Any

    def __new__(cls, shape: Any, stride: Any, dtype: Any) -> "TensorMeta":
        return super().__new__(
            cls,
            tuple(shape),
            tuple(stride),
            dtype,
        )


@dataclass(frozen=True)
class ShardOrderEntry:
    mesh_dim: int
    tensor_dim: int


@dataclass(frozen=True)
class DTensorSpec:
    mesh: Any
    placements: tuple[Placement, ...]
    tensor_meta: TensorMeta | None = None
    shard_order: tuple[ShardOrderEntry, ...] = ()

    def __init__(
        self,
        mesh: Any,
        placements: Any,
        tensor_meta: TensorMeta | None = None,
        shard_order: Any = (),
    ) -> None:
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "placements", tuple(placements))
        object.__setattr__(self, "tensor_meta", tensor_meta)
        object.__setattr__(self, "shard_order", tuple(shard_order))

    @property
    def ndim(self) -> int:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return len(self.tensor_meta.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.shape

    @property
    def stride(self) -> tuple[int, ...]:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.stride

    @property
    def num_shards(self) -> int:
        result = 1
        for index, placement in enumerate(self.placements):
            if placement.is_shard():
                result *= int(self.mesh.size(index))
        return result

    @property
    def device_mesh(self) -> Any:
        return self.mesh

    @property
    def is_replicated(self) -> bool:
        return _BoolValue(all(placement.is_replicate() for placement in self.placements))

    @property
    def is_sharded(self) -> bool:
        return _BoolValue(any(_is_shard_like(placement) for placement in self.placements))

    @property
    def is_partial(self) -> bool:
        return _BoolValue(any(isinstance(placement, Partial) for placement in self.placements))

    @property
    def dim_map(self) -> list[int]:
        result = _DimMap([-1] * self.ndim)
        for mesh_dim, placement in enumerate(self.placements):
            if isinstance(placement, Shard):
                if result[placement.dim] != -1:
                    raise ValueError(
                        f"tensor dimension {placement.dim} is sharded on multiple mesh dimensions"
                    )
                result[placement.dim] = mesh_dim
        return result

    @property
    def num_shards_map(self) -> list[int]:
        result = [1] * self.ndim
        for index, placement in enumerate(self.placements):
            if placement.is_shard():
                result[placement.dim] *= int(self.mesh.size(index))
        return result

    @property
    def sums(self) -> list[int]:
        return [
            index
            for index, placement in enumerate(self.placements)
            if placement.is_partial()
        ]

    @classmethod
    def from_dim_map(
        cls,
        mesh: Any,
        dim_map: list[int],
        sums: list[int],
        tensor_meta: TensorMeta | None = None,
    ) -> "DTensorSpec":
        mesh_ndim_value = getattr(mesh, "ndim")
        mesh_ndim = int(mesh_ndim_value() if callable(mesh_ndim_value) else mesh_ndim_value)
        placements: list[Placement] = [Replicate() for _ in range(mesh_ndim)]
        for mesh_dim in sums:
            mesh_dim = int(mesh_dim)
            if mesh_dim < 0 or mesh_dim >= mesh_ndim:
                raise ValueError(f"sum mesh dimension {mesh_dim} is outside the mesh")
            placements[mesh_dim] = Partial()
        for tensor_dim, mesh_dim in enumerate(dim_map):
            mesh_dim = int(mesh_dim)
            if mesh_dim < 0:
                continue
            if mesh_dim >= mesh_ndim:
                raise ValueError(f"mesh dimension {mesh_dim} is outside the mesh")
            previous = placements[mesh_dim]
            if isinstance(previous, Shard):
                raise RuntimeError(
                    f"mesh dimension {mesh_dim} cannot shard two tensor dimensions"
                )
            if isinstance(previous, Partial):
                raise RuntimeError(
                    f"mesh dimension {mesh_dim} cannot be both sharded and partial"
                )
            placements[mesh_dim] = Shard(tensor_dim)
        return cls(mesh, tuple(placements), tensor_meta=tensor_meta)

    def shallow_copy_with_tensor_meta(
        self, tensor_meta: TensorMeta | None
    ) -> "DTensorSpec":
        if tensor_meta is None:
            raise ValueError("tensor_meta is required")
        return type(self)(
            self.mesh,
            self.placements,
            tensor_meta=tensor_meta,
            shard_order=self.shard_order,
        )

    def __hash__(self) -> int:
        meta = self.tensor_meta
        return hash(
            (
                self.mesh,
                self.placements,
                None if meta is None else meta.shape,
                None if meta is None else meta.stride,
                None if meta is None else meta.dtype,
                self.shard_order,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DTensorSpec):
            return NotImplemented
        return (
            self.mesh == other.mesh
            and self.placements == other.placements
            and self.tensor_meta == other.tensor_meta
            and self.shard_order == other.shard_order
        )

    def __str__(self) -> str:
        placement = self.placements[0] if len(self.placements) == 1 else self.placements
        shape = "unknown shape" if self.tensor_meta is None else tuple(self.tensor_meta.shape)
        return f"Spec({placement} on {shape})"

    def __repr__(self) -> str:
        return f"DTensorSpec(mesh={self.mesh!r}, placements={self.placements!r}, tensor_meta={self.tensor_meta!r})"
