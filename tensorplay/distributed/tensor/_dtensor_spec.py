"""Metadata records for distributed tensor layouts."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Any, NamedTuple

from .placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)

__all__ = ["DTensorSpec", "ShardOrderEntry", "TensorMeta"]


class _StridedShardNotDecodableError(ValueError):
    pass


class ShardOrderEntry(NamedTuple):
    tensor_dim: int
    mesh_dims: tuple[int, ...]


ShardOrder = tuple[ShardOrderEntry, ...]


_TensorMetaBase = namedtuple("_TensorMetaBase", ("shape", "stride", "dtype"))


class TensorMeta(_TensorMetaBase):
    __slots__ = ()
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Any

    def __new__(cls, shape: Any, stride: Any, dtype: Any) -> "TensorMeta":
        return super().__new__(cls, tuple(shape), tuple(stride), dtype)


@dataclass
class DTensorSpec:
    mesh: Any
    placements: tuple[Placement, ...]
    tensor_meta: TensorMeta | None = None
    shard_order: ShardOrder | None = None
    use_strided_shard_as_shard_order: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.placements, tuple):
            self.placements = tuple(self.placements)
        if self.use_strided_shard_as_shard_order is None:
            self.use_strided_shard_as_shard_order = any(
                isinstance(placement, _StridedShard)
                for placement in self.placements
            )
        if self.use_strided_shard_as_shard_order:
            if self.shard_order is not None:
                raise ValueError(
                    "DTensorSpec does not allow shard_order when "
                    "use_strided_shard_as_shard_order is True"
                )
        elif self.shard_order is None:
            self.shard_order = self.compute_default_shard_order(self.placements)
        self._hash: int | None = None

    @staticmethod
    def _normalize_placements_into_shard_order(
        placements: tuple[Placement, ...],
        mesh: Any,
        use_strided_shard_as_shard_order: bool = True,
    ) -> tuple[tuple[Placement, ...], ShardOrder]:
        if use_strided_shard_as_shard_order:
            shard_order = DTensorSpec._maybe_convert_StridedShard_to_shard_order(
                placements, mesh
            )
            if shard_order is None:
                raise _StridedShardNotDecodableError(
                    f"_StridedShard placements {placements} cannot be decoded "
                    "into a corresponding shard_order"
                )
            normalized_placements = tuple(
                Shard(placement.dim)
                if isinstance(placement, _StridedShard)
                else placement
                for placement in placements
            )
            return normalized_placements, shard_order
        return placements, DTensorSpec.compute_default_shard_order(placements)

    @staticmethod
    def compute_default_shard_order(
        placements: tuple[Placement, ...],
    ) -> ShardOrder:
        tensor_dim_to_mesh_dims: defaultdict[int, list[int]] = defaultdict(list)
        for mesh_dim, placement in enumerate(placements):
            if _is_shard_like(placement):
                if placement.dim < 0:
                    raise AssertionError(
                        f"Shard dim {placement.dim} in placements {placements} must be normalized"
                    )
                tensor_dim_to_mesh_dims[placement.dim].append(mesh_dim)
        return tuple(
            ShardOrderEntry(tensor_dim, tuple(mesh_dims))
            for tensor_dim, mesh_dims in sorted(tensor_dim_to_mesh_dims.items())
        )

    @staticmethod
    def _convert_shard_order_to_StridedShard(
        shard_order: ShardOrder,
        placements: tuple[Placement, ...],
        mesh: Any,
    ) -> tuple[Placement, ...]:
        placements_list = list(placements)
        for entry in shard_order:
            for index, mesh_dim in enumerate(entry.mesh_dims):
                if type(placements[mesh_dim]) is not Shard:
                    raise ValueError(
                        "Only Shard placement can be converted to _StridedShard, "
                        f"found {placements[mesh_dim]} in placements={placements}."
                    )
                split_factor = math.prod(
                    mesh.size(indexed_mesh_dim)
                    for indexed_mesh_dim in entry.mesh_dims[:index]
                    if indexed_mesh_dim > mesh_dim
                )
                placements_list[mesh_dim] = (
                    Shard(entry.tensor_dim)
                    if split_factor == 1
                    else _StridedShard(entry.tensor_dim, split_factor=split_factor)
                )
        return tuple(placements_list)

    @staticmethod
    def _maybe_convert_StridedShard_to_shard_order(
        placements: tuple[Placement, ...],
        mesh: Any,
    ) -> ShardOrder | None:
        if not any(isinstance(placement, _StridedShard) for placement in placements):
            return DTensorSpec.compute_default_shard_order(placements)
        shard_placements = [
            placement for placement in placements if _is_shard_like(placement)
        ]
        if not shard_placements:
            return ()
        max_tensor_dim = max(placement.dim for placement in shard_placements) + 1
        tensor_dim_to_mesh_dims_order: list[list[int]] = [
            [] for _ in range(max_tensor_dim)
        ]
        for mesh_dim in reversed(range(len(placements))):
            placement = placements[mesh_dim]
            if _is_shard_like(placement):
                tensor_dim = placement.dim
                mesh_dims_order = tensor_dim_to_mesh_dims_order[tensor_dim]
                split_factor = (
                    placement.split_factor
                    if isinstance(placement, _StridedShard)
                    else 1
                )
                accumulated_factor = 1
                found = False
                for index in range(len(mesh_dims_order) + 1):
                    if accumulated_factor == split_factor:
                        mesh_dims_order.insert(index, mesh_dim)
                        found = True
                        break
                    if index < len(mesh_dims_order):
                        accumulated_factor *= mesh.size(mesh_dims_order[index])
                if not found:
                    return None
            elif not isinstance(placement, (Replicate, Partial)):
                raise ValueError(
                    f"Unsupported placement type {type(placement)} encountered in "
                    f"{placements}; expected Replicate or Partial."
                )
        return tuple(
            ShardOrderEntry(tensor_dim, tuple(mesh_dims))
            for tensor_dim, mesh_dims in enumerate(tensor_dim_to_mesh_dims_order)
            if mesh_dims
        )

    def _verify_shard_order(self, shard_order: ShardOrder) -> None:
        if any(isinstance(placement, _StridedShard) for placement in self.placements):
            return
        total_shard = 0
        previous_tensor_dim = -1
        for entry in shard_order:
            if not entry.mesh_dims:
                raise AssertionError(
                    f"shard_order {shard_order} has empty mesh dimension"
                )
            if entry.tensor_dim < 0:
                raise AssertionError(
                    f"shard_order {shard_order} has invalid tensor dimension"
                )
            if entry.tensor_dim <= previous_tensor_dim:
                raise AssertionError("tensor dimensions must be sorted in shard_order")
            previous_tensor_dim = entry.tensor_dim
            total_shard += len(entry.mesh_dims)
            for mesh_dim in entry.mesh_dims:
                if not 0 <= mesh_dim < len(self.placements):
                    raise AssertionError(
                        f"shard_order {shard_order} has invalid mesh dimension"
                    )
                if self.placements[mesh_dim] != Shard(entry.tensor_dim):
                    raise AssertionError(
                        f"placement[{mesh_dim}] does not match shard_order"
                    )
        if total_shard != sum(
            isinstance(placement, Shard) for placement in self.placements
        ):
            raise AssertionError

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "shard_order" and value is not None:
            self._verify_shard_order(value)
        super().__setattr__(attr, value)
        if hasattr(self, "_hash") and attr in {
            "mesh",
            "placements",
            "tensor_meta",
            "shard_order",
        }:
            self._hash = None
        if attr == "tensor_meta" and value is not None:
            try:
                from tensorplay.graph.passes.shape_prop import TensorMetadata
            except ImportError:
                TensorMetadata = ()
            allowed = (TensorMeta, TensorMetadata) if TensorMetadata else (TensorMeta,)
            if not isinstance(value, allowed):
                raise AssertionError(repr(value))

    def _hash_key(self) -> tuple[Any, ...]:
        if self.tensor_meta is not None:
            return (
                self.mesh,
                self.placements,
                self.shard_order,
                self.tensor_meta.shape,
                self.tensor_meta.stride,
                self.tensor_meta.dtype,
            )
        return self.mesh, self.placements, self.shard_order

    def _hash_impl(self) -> int:
        return hash(self._hash_key())

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def _stable_hash(self) -> str:
        stable_hash = getattr(self.mesh, "_stable_hash", None)
        mesh_key = stable_hash() if callable(stable_hash) else repr(self.mesh)
        key = self._hash_key()
        stable_key = (mesh_key,) + key[1:]
        return hashlib.blake2b(repr(stable_key).encode(), digest_size=16).hexdigest()

    def _check_equals(self, other: object, skip_shapes: bool = False) -> bool:
        if not (
            isinstance(other, DTensorSpec)
            and self.mesh == other.mesh
            and self.placements == other.placements
            and self.shard_order == other.shard_order
        ):
            return False
        if self.tensor_meta is None or other.tensor_meta is None:
            return self.tensor_meta == other.tensor_meta
        if skip_shapes:
            return self.tensor_meta.dtype == other.tensor_meta.dtype
        return (
            self.tensor_meta.shape == other.tensor_meta.shape
            and self.tensor_meta.stride == other.tensor_meta.stride
            and self.tensor_meta.dtype == other.tensor_meta.dtype
        )

    def __eq__(self, other: object, /) -> bool:
        return self._check_equals(other)

    def __str__(self) -> str:
        placement_str = self.format_shard_order_str(self.placements, self.shard_order)
        if self.tensor_meta is None:
            return f"Spec(unknown shape({placement_str}))"
        dtype = getattr(self.tensor_meta.dtype, "name", str(self.tensor_meta.dtype))
        shape = tuple(self.tensor_meta.shape)
        return f"Spec({dtype}{shape}({placement_str}))"

    @staticmethod
    def is_default_device_order(shard_order: ShardOrder | None) -> bool:
        if shard_order is None:
            return False
        return all(
            all(previous < current for previous, current in itertools.pairwise(entry.mesh_dims))
            for entry in shard_order
        )

    @staticmethod
    def format_shard_order_str(
        placements: tuple[Placement, ...],
        shard_order: ShardOrder | None = None,
    ) -> str:
        result = ""
        for mesh_dim, placement in enumerate(placements):
            if _is_shard_like(placement) and shard_order is not None:
                for entry in shard_order:
                    if placement.dim != entry.tensor_dim:
                        continue
                    if mesh_dim not in entry.mesh_dims:
                        raise AssertionError
                    if len(entry.mesh_dims) > 1:
                        result += f"{placement}[{entry.mesh_dims.index(mesh_dim)}]"
                    else:
                        result += str(placement)
                    break
            else:
                result += str(placement)
        return result

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
    def ndim(self) -> int:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return len(self.tensor_meta.shape)

    @property
    def num_shards(self) -> int:
        result = 1
        for mesh_dim, placement in enumerate(self.placements):
            if _is_shard_like(placement):
                result *= int(self.mesh.size(mesh_dim))
        return result

    @property
    def device_mesh(self) -> Any:
        return self.mesh

    @property
    def dim_map(self) -> list[int]:
        result = [-1] * self.ndim
        for mesh_dim, placement in enumerate(self.placements):
            if _is_shard_like(placement):
                if result[placement.dim] != -1:
                    raise ValueError(
                        f"tensor dimension {placement.dim} is sharded on multiple mesh dimensions"
                    )
                result[placement.dim] = mesh_dim
        return result

    @property
    def num_shards_map(self) -> list[int]:
        result = [1] * self.ndim
        for mesh_dim, placement in enumerate(self.placements):
            if _is_shard_like(placement):
                result[placement.dim] *= int(self.mesh.size(mesh_dim))
        return result

    @property
    def sums(self) -> list[int]:
        return [
            mesh_dim
            for mesh_dim, placement in enumerate(self.placements)
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
            placements[mesh_dim] = Partial()
        for tensor_dim, mesh_dim in enumerate(dim_map):
            if mesh_dim < 0:
                continue
            placement = placements[mesh_dim]
            if placement.is_shard():
                raise RuntimeError(
                    f"mesh dimension {mesh_dim} cannot shard two tensor dimensions"
                )
            if placement.is_partial():
                raise RuntimeError(
                    f"mesh dimension {mesh_dim} cannot be both sharded and partial"
                )
            placements[mesh_dim] = Shard(tensor_dim)
        return cls(mesh, tuple(placements), tensor_meta=tensor_meta)

    def is_replicated(self) -> bool:
        return all(placement.is_replicate() for placement in self.placements)

    def is_sharded(self) -> bool:
        return any(_is_shard_like(placement) for placement in self.placements)

    def shallow_copy_with_tensor_meta(
        self, tensor_meta: TensorMeta | None
    ) -> "DTensorSpec":
        if tensor_meta is None:
            raise AssertionError("shallow copy with no tensor_meta!")
        return DTensorSpec(
            self.mesh,
            self.placements,
            tensor_meta=tensor_meta,
            use_strided_shard_as_shard_order=self.use_strided_shard_as_shard_order,
        )
