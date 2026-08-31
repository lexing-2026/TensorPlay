"""Explicit sharded tensor values backed by local tensor shards."""

import copy
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
        del dst
        result = tp.empty(self.shape, dtype=self.dtype, device=self.device)
        for shard in self._local_shards:
            _copy_shard(result, shard)
        if out is not None:
            out.copy_(result)
            return out
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
        return sharding_spec.shard(self.gather(), process_group=self._process_group)

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
        metadata = sharding_spec.build_metadata(tuple(tensor.shape), TensorProperties.create_from_tensor(tensor))
        rank = _current_rank(process_group)
        local = []
        for item in metadata.shards_metadata:
            if _placement_rank(item.placement) != rank:
                continue
            value = _slice_by_metadata(tensor, item).detach().clone()
            local.append(Shard(value, item))
        return cls._init_from_local_shards_and_global_metadata(local, metadata, sharding_spec, process_group)


def _normalize_size(size: tuple[Any, ...]) -> tuple[int, ...]:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    return tuple(int(value) for value in size)


def _placement_rank(placement: Any) -> int:
    if placement is None:
        return 0
    return placement.rank() if hasattr(placement, "rank") else int(str(placement).split(":")[1].split("/")[0])


def _current_rank(process_group: Any = None) -> int:
    try:
        return dist.get_rank(process_group)
    except Exception:
        return 0


def _slice_by_metadata(tensor: Any, metadata: ShardMetadata) -> Any:
    slices = tuple(slice(offset, offset + size) for offset, size in zip(metadata.shard_offsets, metadata.shard_sizes))
    return tensor[slices]


def _copy_shard(destination: Any, shard: Shard) -> None:
    slices = tuple(slice(offset, offset + size) for offset, size in zip(shard.metadata.shard_offsets, shard.metadata.shard_sizes))
    destination[slices].copy_(shard.tensor)
