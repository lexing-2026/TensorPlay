"""Explicit sharded tensor values backed by local tensor shards."""

import copy
import math
from typing import Any, Callable, Iterable

import tensorplay as tp

from ... import distributed_core as dist
from ..metadata import ShardMetadata
from .metadata import ShardedTensorMetadata, TensorProperties
from .shard import Shard

__all__ = [
    "ShardedTensorBase",
    "ShardedTensor",
    "Shard",
    "ShardedTensorMetadata",
    "TensorProperties",
    "_SHARDED_OPS",
    "_CUSTOM_SHARDED_OPS",
    "_register_remote_shards",
]

_SHARDED_OPS: dict[Callable[..., Any], Callable[..., Any]] = {}
_CUSTOM_SHARDED_OPS: dict[Callable[..., Any], Callable[..., Any]] = {}


def _register_remote_shards(sharded_tensor_id: int, rrefs: list[Any], rpc_rank: int) -> None:
    del sharded_tensor_id, rrefs, rpc_rank


class ShardedTensorBase:
    def __init__(self, sharding_spec: Any, *size: Any, dtype: Any = None, requires_grad: bool = False, process_group: Any = None, **kwargs: Any) -> None:
        shape = _normalize_size(size)
        self._sharding_spec = sharding_spec
        self._process_group = process_group
        self._metadata = sharding_spec.build_metadata(shape, TensorProperties(dtype=dtype, requires_grad=requires_grad, layout=kwargs.get("layout")))
        self._local_shards: list[Shard] = []
        self._populate_empty_local_shards()

    def _populate_empty_local_shards(self) -> None:
        rank = _current_rank(self._process_group)
        for metadata in self._metadata.shards_metadata:
            if _placement_rank(metadata.placement) != rank:
                continue
            props = self._metadata.tensor_properties
            value = tp.empty(tuple(metadata.shard_sizes), dtype=props.dtype, requires_grad=props.requires_grad)
            self._local_shards.append(Shard(value, metadata))

    @classmethod
    def _init_from_local_shards_and_global_metadata(cls, local_shards: list[Shard], sharded_tensor_metadata: ShardedTensorMetadata, sharding_spec: Any = None, process_group: Any = None) -> "ShardedTensorBase":
        obj = cls.__new__(cls)
        obj._sharding_spec = sharding_spec
        obj._metadata = sharded_tensor_metadata
        obj._local_shards = list(local_shards)
        obj._process_group = process_group
        return obj

    def metadata(self) -> ShardedTensorMetadata:
        return self._metadata

    def local_shards(self) -> list[Shard]:
        return list(self._local_shards)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._metadata.size)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return self.ndim

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    @property
    def dtype(self) -> Any:
        return self._metadata.tensor_properties.dtype

    @property
    def device(self) -> Any:
        return self._local_shards[0].tensor.device if self._local_shards else None

    def to_local(self) -> Any:
        if len(self._local_shards) != 1:
            raise RuntimeError("the current rank does not own one unique local shard")
        return self._local_shards[0].tensor

    def gather(self, dst: int = 0, out: Any = None) -> Any:
        if not isinstance(dst, int) or dst < 0:
            raise ValueError("dst must be a non-negative integer")
        result = self._gather_full_tensor()
        if out is not None:
            if tuple(out.shape) != self.shape:
                raise ValueError("out does not match the global tensor shape")
            out.copy_(result)
            return out
        return result

    def _gather_full_tensor(self) -> Any:
        local = self._local_tensor()
        if local is None:
            raise RuntimeError("the current rank does not own a local shard")
        if not dist.is_initialized() or dist.get_world_size(self._process_group) == 1:
            result = tp.empty(self.shape, dtype=local.dtype, device=local.device)
            for shard in self._local_shards:
                _copy_shard(result, shard)
            return result

        group = self._process_group
        world_size = dist.get_world_size(group)
        local_flat = local.reshape(-1).contiguous()
        max_numel = max(
            1,
            max(
                int(math.prod(metadata.shard_sizes))
                for metadata in self._metadata.shards_metadata
            ),
        )
        packed = tp.zeros(max_numel, dtype=local.dtype, device=local.device)
        if local_flat.numel():
            packed.narrow(0, 0, local_flat.numel()).copy_(local_flat)
        gathered = [tp.empty((max_numel,), dtype=local.dtype, device=local.device) for _ in range(world_size)]
        dist.all_gather(gathered, packed, group=group)
        result = tp.zeros(self.shape, dtype=local.dtype, device=local.device)
        for metadata in self._metadata.shards_metadata:
            global_rank = _placement_rank(metadata.placement)
            group_rank = _group_rank(group, global_rank)
            count = int(math.prod(metadata.shard_sizes))
            values = gathered[group_rank].narrow(0, 0, count).reshape(tuple(metadata.shard_sizes))
            _copy_shard(result, Shard(values, metadata))
        return result

    def _local_tensor(self) -> Any:
        return self._local_shards[0].tensor if self._local_shards else None

    def clone(self) -> "ShardedTensorBase":
        shards = [Shard(item.tensor.clone(), copy.deepcopy(item.metadata)) for item in self._local_shards]
        return type(self)._init_from_local_shards_and_global_metadata(shards, copy.deepcopy(self._metadata), self._sharding_spec, self._process_group)

    def detach(self) -> "ShardedTensorBase":
        shards = [Shard(item.tensor.detach(), copy.deepcopy(item.metadata)) for item in self._local_shards]
        return type(self)._init_from_local_shards_and_global_metadata(shards, copy.deepcopy(self._metadata), self._sharding_spec, self._process_group)

    def requires_grad_(self, requires_grad: bool = True) -> "ShardedTensorBase":
        for shard in self._local_shards:
            shard.tensor.requires_grad = requires_grad
        self._metadata.tensor_properties.requires_grad = requires_grad
        return self

    def reshard(self, sharding_spec: Any) -> "ShardedTensorBase":
        from .reshard import reshard_local_shard

        if not isinstance(sharding_spec, type(self._sharding_spec)):
            raise TypeError("reshard requires a compatible sharding specification")
        local = self._local_tensor()
        if local is None:
            raise RuntimeError("the current rank does not own a local shard")
        local_shards, metadata = reshard_local_shard(
            local,
            self.shape,
            self._sharding_spec,
            sharding_spec,
            self._process_group,
        )
        return type(self)._init_from_local_shards_and_global_metadata(
            local_shards,
            type(self._metadata)(metadata, self.shape, self._metadata.tensor_properties),
            sharding_spec,
            self._process_group,
        )

    def __getattr__(self, name: str) -> Any:
        local = self.__dict__.get("_local_shards")
        if local:
            return getattr(local[0].tensor, name)
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"ShardedTensor(shape={self.shape}, local_shards={len(self._local_shards)})"


class ShardedTensor(ShardedTensorBase):
    @classmethod
    def _init_from_global_tensor(cls, sharding_spec: Any, tensor: Any, process_group: Any = None) -> "ShardedTensor":
        return cls._scatter_from_global_tensor(
            sharding_spec, tensor, process_group=process_group, src_rank=0
        )

    @classmethod
    def _scatter_from_global_tensor(
        cls,
        sharding_spec: Any,
        tensor: Any,
        *,
        process_group: Any = None,
        src_rank: int = 0,
    ) -> "ShardedTensor":
        properties = TensorProperties.create_from_tensor(tensor)
        metadata = sharding_spec.build_metadata(tuple(tensor.shape), properties)
        placements = metadata.shards_metadata
        if not placements:
            raise ValueError("sharding specification produced no shards")
        if not dist.is_initialized():
            if src_rank != 0:
                raise ValueError("src_rank must be zero without a process group")
            local = [
                Shard(_slice_by_metadata(tensor, item).detach().clone(), item)
                for item in placements
                if _placement_rank(item.placement) == 0
            ]
            return cls._init_from_local_shards_and_global_metadata(
                local, metadata, sharding_spec, process_group
            )

        group = process_group
        world_size = dist.get_world_size(group)
        if len(placements) != world_size:
            raise ValueError(
                "the number of shard placements must equal the process group size"
            )
        current_group_rank = dist.get_rank(group)
        current_global_rank = dist.get_rank()
        if src_rank < 0 or src_rank >= world_size:
            raise ValueError("src_rank is outside the process group")
        local_metadata = next(
            (
                item
                for item in placements
                if _placement_rank(item.placement) == current_global_rank
            ),
            None,
        )
        if local_metadata is None:
            raise ValueError("every process-group rank must own one shard")

        counts = [int(math.prod(item.shard_sizes)) for item in placements]
        max_count = max(1, max(counts))
        scatter_list = None
        if current_group_rank == src_rank:
            scatter_list = [None] * world_size
            for item, count in zip(placements, counts):
                global_rank = _placement_rank(item.placement)
                target_rank = _group_rank(group, global_rank)
                values = _slice_by_metadata(tensor, item).detach().clone().reshape(-1)
                packed = tp.zeros(max_count, dtype=tensor.dtype, device=tensor.device)
                if count:
                    packed.narrow(0, 0, count).copy_(values)
                scatter_list[target_rank] = packed
            if any(value is None for value in scatter_list):
                raise ValueError("shard placements must map to every process-group rank")

        received = tp.empty((max_count,), dtype=tensor.dtype, device=tensor.device)
        dist.scatter(
            received,
            scatter_list=scatter_list,
            group_src=src_rank,
            group=group,
        )
        count = int(math.prod(local_metadata.shard_sizes))
        value = received.narrow(0, 0, count).reshape(tuple(local_metadata.shard_sizes)).detach()
        value.requires_grad_(bool(tensor.requires_grad))
        local = [Shard(value, local_metadata)]
        return cls._init_from_local_shards_and_global_metadata(
            local, metadata, sharding_spec, process_group
        )


def _normalize_size(size: tuple[Any, ...]) -> tuple[int, ...]:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    return tuple(int(value) for value in size)


def _placement_rank(placement: Any) -> int:
    if placement is None:
        return 0
    rank = placement.rank() if hasattr(placement, "rank") else int(str(placement).split(":")[1].split("/")[0])
    if rank is None:
        raise ValueError(f"placement {placement!r} does not identify a process rank")
    return int(rank)


def _current_rank(process_group: Any = None) -> int:
    try:
        return dist.get_rank(process_group)
    except Exception:
        return 0


def _group_rank(process_group: Any, global_rank: int) -> int:
    if process_group is None:
        return int(global_rank)
    return int(dist.get_group_rank(process_group, int(global_rank)))


def _slice_by_metadata(tensor: Any, metadata: ShardMetadata) -> Any:
    slices = tuple(slice(offset, offset + size) for offset, size in zip(metadata.shard_offsets, metadata.shard_sizes))
    return tensor[slices]


def _copy_shard(destination: Any, shard: Shard) -> None:
    slices = tuple(slice(offset, offset + size) for offset, size in zip(shard.metadata.shard_offsets, shard.metadata.shard_sizes))
    destination[slices].copy_(shard.tensor)
