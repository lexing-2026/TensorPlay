"""Immutable metadata records for distributed tensor layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .placement_types import Partial, Placement, Shard, _is_shard_like

__all__ = ["DTensorSpec", "ShardOrderEntry", "TensorMeta"]


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Any

    def __init__(self, shape: Any, stride: Any, dtype: Any) -> None:
        object.__setattr__(self, "shape", tuple(int(value) for value in shape))
        object.__setattr__(self, "stride", tuple(int(value) for value in stride))
        object.__setattr__(self, "dtype", dtype)


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
        return self.mesh.ndim()

    @property
    def shape(self) -> tuple[int, ...] | None:
        return None if self.tensor_meta is None else self.tensor_meta.shape

    @property
    def is_replicated(self) -> bool:
        return all(placement.is_replicate() for placement in self.placements)

    @property
    def is_sharded(self) -> bool:
        return any(_is_shard_like(placement) for placement in self.placements)

    @property
    def is_partial(self) -> bool:
        return any(isinstance(placement, Partial) for placement in self.placements)

    def dim_map(self) -> tuple[int, ...]:
        result = [-1] * (len(self.shape) if self.shape is not None else 0)
        for mesh_dim, placement in enumerate(self.placements):
            if isinstance(placement, Shard):
                result[placement.dim] = mesh_dim
        return tuple(result)

    def __repr__(self) -> str:
        return f"DTensorSpec(mesh={self.mesh!r}, placements={self.placements!r}, tensor_meta={self.tensor_meta!r})"
