from __future__ import annotations

import io
from typing import Any

import tensorplay as tp

from ._traverse import get_element, traverse_state_dict
from .metadata import BytesStorageMetadata, ChunkStorageMetadata, Metadata, MetadataIndex, TensorProperties, TensorStorageMetadata
from .planner import BytesIOWriteData, LoadItemType, ReadItem, SavePlan, TensorWriteData, WriteItem, WriteItemType
from .resharding import _check_shard_metadata_pair_overlap, _shards_get_overlap_region_wrt_saved_tensor

__all__ = ["create_read_items_for_chunk_list"]


def _tensor_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in value.shape)


def _create_chunk_from_tensor(tensor: tp.Tensor) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(tuple(0 for _ in tensor.shape), _tensor_shape(tensor))


def _create_chunk_from_dtensor(tensor: Any) -> ChunkStorageMetadata:
    from ..tensor._utils import compute_local_shape_and_global_offset

    sizes, offsets = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    return ChunkStorageMetadata(
        tuple(int(value) for value in offsets),
        tuple(int(value) for value in sizes),
    )


def _create_write_item_for_dtensor(fqn: str, tensor: Any) -> WriteItem:
    local = tensor.to_local()
    chunk = _create_chunk_from_dtensor(tensor)
    return WriteItem(
        MetadataIndex(fqn, chunk.offsets),
        WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk,
            TensorProperties.create_from_tensor(local),
            _tensor_shape(tensor),
        ),
    )


def _create_write_item_for_tensor(fqn: str, tensor: tp.Tensor) -> WriteItem:
    chunk = _create_chunk_from_tensor(tensor)
    return WriteItem(MetadataIndex(fqn), WriteItemType.TENSOR, tensor_data=TensorWriteData(chunk, TensorProperties.create_from_tensor(tensor), _tensor_shape(tensor)))


def _create_write_item_for_bytesio(fqn: str, value: Any) -> WriteItem:
    return WriteItem(MetadataIndex(fqn), WriteItemType.BYTE_IO, bytes_io_data=BytesIOWriteData(len(value.getvalue()) if isinstance(value, io.BytesIO) else 0))


def _create_write_items(fqn: str, value: Any) -> list[WriteItem]:
    create_items = getattr(value, "__create_write_items__", None)
    if callable(create_items):
        return list(create_items(fqn, value))
    return [_create_write_item_for_tensor(fqn, value)] if isinstance(value, tp.Tensor) else [_create_write_item_for_bytesio(fqn, value)]


def _create_chunk_list(value: Any) -> list[ChunkStorageMetadata]:
    create_chunks = getattr(value, "__create_chunk_list__", None)
    if callable(create_chunks):
        return list(create_chunks())
    if isinstance(value, tp.Tensor):
        return [_create_chunk_from_tensor(value)]
    raise ValueError(f"unsupported tensor value {type(value)!r}")


def _create_write_items_from_state_dict(state_dict: dict[str, Any]) -> list[WriteItem]:
    items: list[WriteItem] = []
    traverse_state_dict(state_dict, lambda path, value: items.extend(_create_write_items(".".join(map(str, path)), value)))
    return items


def _create_default_metadata_only_plan(state_dict: dict[str, Any]) -> SavePlan:
    return SavePlan(_create_write_items_from_state_dict(state_dict))


def create_default_local_save_plan(state_dict: dict[str, Any], is_coordinator: bool = True) -> SavePlan:
    del is_coordinator
    return SavePlan(_create_write_items_from_state_dict(state_dict))


def create_default_global_save_plan(all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
    metadata: dict[str, Any] = {}
    for plan in all_plans:
        for item in plan.items:
            if item.tensor_data is not None:
                existing = metadata.get(item.index.fqn)
                if isinstance(existing, TensorStorageMetadata):
                    existing.chunks.append(item.tensor_data.chunk)
                else:
                    metadata[item.index.fqn] = TensorStorageMetadata(
                        item.tensor_data.properties,
                        item.tensor_data.size,
                        [item.tensor_data.chunk],
                    )
            else:
                metadata.setdefault(item.index.fqn, BytesStorageMetadata())
    return all_plans, Metadata(metadata)


def _create_read_item_for_tensor(fqn: str, md: TensorStorageMetadata, obj: Any) -> list[ReadItem]:
    chunks = _create_chunk_list(obj)
    return create_read_items_for_chunk_list(fqn, md, chunks)


def _create_read_item_for_bytesio(fqn: str) -> ReadItem:
    return ReadItem(LoadItemType.BYTE_IO, MetadataIndex(fqn), (), MetadataIndex(fqn), (), ())


def create_read_items_for_chunk_list(
    fqn: str, checkpoint_md_or_chunks: Any, local_chunks_or_obj: Any
) -> list[ReadItem]:
    if isinstance(checkpoint_md_or_chunks, TensorStorageMetadata):
        checkpoint_md = checkpoint_md_or_chunks
        local_chunks = list(local_chunks_or_obj)
        result: list[ReadItem] = []
        for index, local_chunk in enumerate(local_chunks):
            for storage_index, storage_chunk in enumerate(checkpoint_md.chunks):
                if not _check_shard_metadata_pair_overlap(local_chunk, storage_chunk):
                    continue
                storage_offsets: list[int] = []
                dest_offsets: list[int] = []
                lengths: list[int] = []
                for _, saved_offset, current_offset, length in _shards_get_overlap_region_wrt_saved_tensor(
                    saved_shard=storage_chunk,
                    current_shard=local_chunk,
                ):
                    storage_offsets.append(int(saved_offset))
                    dest_offsets.append(int(current_offset))
                    lengths.append(int(length))
                result.append(
                    ReadItem(
                        LoadItemType.TENSOR,
                        MetadataIndex(fqn, local_chunk.offsets, index),
                        tuple(dest_offsets),
                        MetadataIndex(fqn, storage_chunk.offsets, storage_index),
                        tuple(storage_offsets),
                        tuple(lengths),
                    )
                )
        return result

    chunks = list(checkpoint_md_or_chunks)
    obj = local_chunks_or_obj
    if isinstance(obj, tp.Tensor):
        result = []
        for chunk in chunks:
            size = tuple(getattr(chunk, "sizes", getattr(chunk, "size", obj.shape)))
            offset = tuple(getattr(chunk, "offsets", (0,) * len(size)))
            result.append(ReadItem(LoadItemType.TENSOR, MetadataIndex(fqn, offset), tuple(0 for _ in size), MetadataIndex(fqn, offset), offset, size))
        return result
    return [_create_read_item_for_bytesio(fqn)]


def create_default_local_load_plan(state_dict: dict[str, Any], metadata: Metadata, strict: bool = True):
    items: list[ReadItem] = []
    for key, value in state_dict.items():
        md = metadata.state_dict_metadata.get(key)
        if md is None:
            if strict:
                raise KeyError(f"checkpoint is missing {key}")
            continue
        if isinstance(md, TensorStorageMetadata):
            try:
                items.extend(_create_read_item_for_tensor(key, md, value))
            except (AttributeError, TypeError, ValueError):
                if strict:
                    raise
                continue
        else:
            items.append(_create_read_item_for_bytesio(key))
    return __import__("tensorplay.distributed.checkpoint.planner", fromlist=["LoadPlan"]).LoadPlan(items)


def create_default_global_load_plan(plans: list[Any]) -> list[Any]:
    return plans
