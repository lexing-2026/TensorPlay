from __future__ import annotations

from typing import Protocol, TypeGuard, runtime_checkable

import tensorplay as tp

from .metadata import ChunkStorageMetadata, MetadataIndex

__all__ = ["CheckpointableTensor"]


@runtime_checkable
class CheckpointableTensor(Protocol):
    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]


def _is_checkpointable_tensor(obj: object) -> TypeGuard[CheckpointableTensor]:
    return isinstance(obj, tp.Tensor) and isinstance(obj, CheckpointableTensor)


def _copy_checkpointable_tensor_metadata(
    src: CheckpointableTensor, dst: tp.Tensor
) -> None:
    dst.global_shape = src.global_shape
    dst.global_offsets = src.global_offsets
    dst.local_offsets = src.local_offsets
    dst.local_sizes = src.local_sizes


def _get_checkpointable_tensor_chunks(
    tensor: CheckpointableTensor,
) -> list[ChunkStorageMetadata]:
    _validate_checkpointable_tensor_metadata(tensor)
    return [
        ChunkStorageMetadata(tuple(global_offset), tuple(local_size))
        for global_offset, local_size in zip(
            tensor.global_offsets, tensor.local_sizes
        )
    ]


def _get_checkpointable_tensor_shard(
    tensor: CheckpointableTensor, index: MetadataIndex
) -> tp.Tensor:
    _validate_checkpointable_tensor_metadata(tensor)
    if index.offset is None:
        if len(tensor.global_offsets) != 1:
            raise ValueError(
                f"Cannot locate {index.fqn!r} with multiple checkpointable shards"
            )
        shard_index = 0
    elif (
        index.index is not None
        and index.index < len(tensor.global_offsets)
        and tuple(tensor.global_offsets[index.index]) == tuple(index.offset)
    ):
        shard_index = index.index
    else:
        try:
            shard_index = next(
                position
                for position, offset in enumerate(tensor.global_offsets)
                if tuple(offset) == tuple(index.offset)
            )
        except StopIteration as error:
            raise ValueError(
                f"Could not find checkpointable shard at {index.offset!r} "
                f"for {index.fqn!r}"
            ) from error

    local_offset = tensor.local_offsets[shard_index]
    local_size = tensor.local_sizes[shard_index]
    if not local_offset:
        return tensor  # type: ignore[return-value]
    return tensor[
        tuple(
            slice(offset, offset + size)
            for offset, size in zip(local_offset, local_size)
        )
    ]  # type: ignore[index, return-value]


def _validate_checkpointable_tensor_metadata(
    tensor: CheckpointableTensor,
) -> None:
    count = len(tensor.global_offsets)
    if len(tensor.local_offsets) != count or len(tensor.local_sizes) != count:
        raise ValueError("checkpointable tensor metadata has inconsistent shard counts")
    tensor_shape = tuple(getattr(tensor, "shape", ()))
    for position, (global_offset, local_offset, local_size) in enumerate(
        zip(tensor.global_offsets, tensor.local_offsets, tensor.local_sizes)
    ):
        if len(global_offset) != len(tensor.global_shape):
            raise ValueError(f"global_offsets[{position}] has an invalid rank")
        if len(local_offset) != len(tensor_shape):
            raise ValueError(f"local_offsets[{position}] has an invalid rank")
        if len(local_size) != len(tensor.global_shape) or len(local_size) != len(tensor_shape):
            raise ValueError(f"local_sizes[{position}] has an invalid rank")
        for offset, size, dimension in zip(global_offset, local_size, tensor.global_shape):
            if offset < 0 or size < 0 or offset + size > dimension:
                raise ValueError(f"global shard {position} is outside global_shape")
        for offset, size, dimension in zip(local_offset, local_size, tensor_shape):
            if offset < 0 or size < 0 or offset + size > dimension:
                raise ValueError(f"local shard {position} is outside tensor shape")
