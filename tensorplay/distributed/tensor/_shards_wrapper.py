from __future__ import annotations

import math
from typing import Any

import tensorplay as tp

from ..checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from ..checkpoint.planner import TensorWriteData, WriteItem, WriteItemType
from tensorplay.distributed import _functional_collectives as funcol

__all__ = ["LocalShardsWrapper"]


class LocalShardsWrapper:
    def __init__(
        self, local_shards: list[Any], local_offsets: list[tuple[int, ...]]
    ) -> None:
        if len(local_shards) != len(local_offsets):
            raise ValueError("local shard and offset counts must match")
        if local_shards and any(
            tensor.device != local_shards[0].device for tensor in local_shards[1:]
        ):
            raise AssertionError
        self._local_shards = list(local_shards)
        if not local_shards:
            properties = TensorProperties()
            size = (0, 0)
            chunks = [ChunkStorageMetadata((0, 0), (0, 0))]
        else:
            shape = list(local_shards[0].shape)
            if len(local_shards) > 1 and local_shards[0].ndim == 2:
                shape[1] += sum(int(shard.shape[1]) for shard in local_shards[1:])
            if len(local_shards) > 1 and local_shards[0].ndim == 1:
                shape[0] += sum(int(shard.shape[0]) for shard in local_shards[1:])
            properties = TensorProperties.create_from_tensor(local_shards[0])
            size = tuple(int(value) for value in shape)
            chunks = [
                ChunkStorageMetadata(
                    tuple(int(value) for value in offset),
                    tuple(int(value) for value in shard.shape),
                )
                for shard, offset in zip(local_shards, local_offsets)
            ]
        self._storage_meta = TensorStorageMetadata(properties, size, chunks)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._storage_meta.size)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return self.ndim

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def dtype(self) -> Any:
        return self._storage_meta.properties.dtype

    @property
    def layout(self) -> Any:
        return self._storage_meta.properties.layout

    @property
    def requires_grad(self) -> bool:
        return bool(self._storage_meta.properties.requires_grad)

    @property
    def device(self) -> Any:
        if self._local_shards:
            return self._local_shards[0].device
        return tp.device("meta")

    @property
    def is_meta(self) -> bool:
        return bool(self._local_shards[0].is_meta) if self._local_shards else True

    def is_pinned(self) -> bool:
        return bool(self._storage_meta.properties.pin_memory)

    def requires_grad_(self, requires_grad: bool = True) -> "LocalShardsWrapper":
        self._storage_meta.properties.requires_grad = bool(requires_grad)
        for shard in self._local_shards:
            shard.requires_grad_(requires_grad)
        return self

    def local_shards(self) -> list[Any]:
        return self._local_shards

    def local_sizes(self) -> list[tuple[int, ...]]:
        return [chunk.sizes for chunk in self._storage_meta.chunks]

    def local_offsets(self) -> list[tuple[int, ...]]:
        return [chunk.offsets for chunk in self._storage_meta.chunks]

    @property
    def local_chunks(self) -> list[ChunkStorageMetadata]:
        return self._storage_meta.chunks

    def storage_metadata(self) -> TensorStorageMetadata:
        return self._storage_meta

    def is_empty_shard(self) -> bool:
        return all(int(value) == 0 for value in self.shape)

    @staticmethod
    def handle_all_gather_into_tensor(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        wrapper = args[0]
        if not wrapper.local_shards():
            return tp.empty((0,), dtype=wrapper.dtype, device=wrapper.device)
        width = (
            wrapper.local_sizes()[0][1]
            if wrapper.local_shards()[0].ndim == 2
            else None
        )
        flattened = tp.cat(
            [shard.reshape(-1) for shard in wrapper.local_shards()], dim=0
        )
        if width is not None:
            flattened = flattened.reshape(-1, width)
        return funcol.all_gather_single(flattened, 0, group=kwargs.get("group"))

    @staticmethod
    def handle_wait_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        del kwargs
        return funcol.wait_tensor(args[0])

    @staticmethod
    def handle_to_copy(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> "LocalShardsWrapper":
        wrapper = args[0]
        return LocalShardsWrapper(
            [shard.to(*args[1:], **kwargs) for shard in wrapper.local_shards()],
            wrapper.local_offsets(),
        )

    @staticmethod
    def handle_view(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> "LocalShardsWrapper":
        wrapper = args[0]
        target = args[1:]
        if len(wrapper.local_shards()) > 1:
            if wrapper.ndim not in (1, 2):
                raise NotImplementedError("view supports at most two local dimensions")
            requested = tuple(int(value) for value in target)
            if requested != wrapper.shape:
                raise AssertionError
            shards = [
                shard.view(shard.shape, **kwargs)
                for shard in wrapper.local_shards()
            ]
        else:
            shards = [
                shard.view(*target, **kwargs) for shard in wrapper.local_shards()
            ]
        return LocalShardsWrapper(shards, wrapper.local_offsets())

    @staticmethod
    def handle_equal(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        del kwargs
        left, right = args[:2]
        if not isinstance(right, LocalShardsWrapper):
            return False
        return (
            len(left.local_shards()) == len(right.local_shards())
            and all(
                bool(tp.equal(a, b))
                for a, b in zip(left.local_shards(), right.local_shards())
            )
            and left.storage_metadata() == right.storage_metadata()
        )

    @staticmethod
    def handle_detach(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> "LocalShardsWrapper":
        del kwargs
        wrapper = args[0]
        wrapper._local_shards = [shard.detach() for shard in wrapper.local_shards()]
        wrapper._storage_meta.properties.requires_grad = False
        return wrapper

    @staticmethod
    def handle_clone(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> "LocalShardsWrapper":
        wrapper = args[0]
        memory_format = kwargs.get("memory_format")
        if memory_format is not None and memory_format != tp.preserve_format:
            raise NotImplementedError(f"{memory_format} is not supported")
        shards = [
            shard.clone(memory_format=memory_format)
            for shard in wrapper.local_shards()
        ]
        return LocalShardsWrapper(shards, wrapper.local_offsets())

    @staticmethod
    def handle_new_empty(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> "LocalShardsWrapper":
        del kwargs
        wrapper = args[0]
        return LocalShardsWrapper(
            [shard.new_empty(shard.shape) for shard in wrapper.local_shards()],
            wrapper.local_offsets(),
        )

    def to(self, *args: Any, **kwargs: Any) -> "LocalShardsWrapper":
        return self.handle_to_copy((self, *args), kwargs)

    def view(self, *shape: Any, **kwargs: Any) -> "LocalShardsWrapper":
        return self.handle_view((self, *shape), kwargs)

    def equal(self, other: Any) -> bool:
        return self.handle_equal((self, other), {})

    def detach(self) -> "LocalShardsWrapper":
        return self.handle_detach((self,), {})

    def clone(self, **kwargs: Any) -> "LocalShardsWrapper":
        return self.handle_clone((self,), kwargs)

    def new_empty(self) -> "LocalShardsWrapper":
        return self.handle_new_empty((self,), {})

    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunk.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(chunk.offsets, chunk.sizes),
                    properties=self._storage_meta.properties,
                    size=tuple(int(value) for value in object.shape),
                ),
            )
            for chunk in self.local_chunks
        ]

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        return self._storage_meta.chunks

    def __get_tensor_shard__(self, index: MetadataIndex) -> Any:
        if index.index is not None and index.index < len(self._local_shards):
            if self.local_chunks[index.index].offsets == index.offset:
                return self._local_shards[index.index]
        if index.offset is not None:
            for shard, chunk in zip(self.local_shards(), self.local_chunks):
                if chunk.offsets == index.offset:
                    return shard
        if not self.local_shards() and self.is_empty_shard():
            return tp.empty(0)
        raise ValueError(f"could not find shard at {index.offset!r} for {index.fqn!r}")

    def _get_tensor_size_bytes(self) -> int:
        return sum(
            shard.numel() * shard.element_size() for shard in self.local_shards()
        )

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self.equal(other)

    def __repr__(self) -> str:
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

    __str__ = __repr__
