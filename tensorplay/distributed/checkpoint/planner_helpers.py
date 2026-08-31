from __future__ import annotations

import io
from typing import Any

import tensorplay as tp

from ._traverse import get_element, traverse_state_dict
from .metadata import BytesStorageMetadata, ChunkStorageMetadata, Metadata, MetadataIndex, TensorProperties, TensorStorageMetadata
from .planner import BytesIOWriteData, LoadItemType, ReadItem, SavePlan, TensorWriteData, WriteItem, WriteItemType

__all__ = ["create_read_items_for_chunk_list"]


def _tensor_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in value.shape)


def _create_chunk_from_tensor(tensor: tp.Tensor) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(tuple(0 for _ in tensor.shape), _tensor_shape(tensor))


def _create_write_item_for_tensor(fqn: str, tensor: tp.Tensor) -> WriteItem:
    chunk = _create_chunk_from_tensor(tensor)
    return WriteItem(MetadataIndex(fqn), WriteItemType.TENSOR, tensor_data=TensorWriteData(chunk, TensorProperties.create_from_tensor(tensor), _tensor_shape(tensor)))


def _create_write_item_for_bytesio(fqn: str, value: Any) -> WriteItem:
    return WriteItem(MetadataIndex(fqn), WriteItemType.BYTE_IO, bytes_io_data=BytesIOWriteData(len(value.getvalue()) if isinstance(value, io.BytesIO) else 0))


def _create_write_items(fqn: str, value: Any) -> list[WriteItem]:
    return [_create_write_item_for_tensor(fqn, value)] if isinstance(value, tp.Tensor) else [_create_write_item_for_bytesio(fqn, value)]


def _create_write_items_from_state_dict(state_dict: dict[str, Any]) -> list[WriteItem]:
    items: list[WriteItem] = []
    traverse_state_dict(state_dict, lambda path, value: items.extend(_create_write_items(".".join(map(str, path)), value)))
    return items


def _create_default_metadata_only_plan(state_dict: dict[str, Any]) -> SavePlan:
    return SavePlan(_create_write_items_from_state_dict(state_dict))


def create_default_local_save_plan(state_dict: dict[str, Any], is_coordinator: bool = True) -> SavePlan:
    items = _create_write_items_from_state_dict(state_dict) if is_coordinator else []
    return SavePlan(items)


def create_default_global_save_plan(all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
    metadata: dict[str, Any] = {}
    for plan in all_plans:
        for item in plan.items:
            if item.tensor_data is not None:
                metadata[item.index.fqn] = TensorStorageMetadata(item.tensor_data.properties, item.tensor_data.size, [item.tensor_data.chunk])
            else:
                metadata[item.index.fqn] = BytesStorageMetadata()
    return all_plans, Metadata(metadata)


def _create_read_item_for_tensor(fqn: str, md: TensorStorageMetadata, obj: tp.Tensor) -> list[ReadItem]:
    return [ReadItem(LoadItemType.TENSOR, MetadataIndex(fqn), tuple(0 for _ in obj.shape), MetadataIndex(fqn), tuple(0 for _ in md.size), tuple(md.size))]


def _create_read_item_for_bytesio(fqn: str) -> ReadItem:
    return ReadItem(LoadItemType.BYTE_IO, MetadataIndex(fqn), (), MetadataIndex(fqn), (), ())


def create_read_items_for_chunk_list(fqn: str, chunks: list[Any], obj: Any) -> list[ReadItem]:
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
        if isinstance(md, TensorStorageMetadata) and isinstance(value, tp.Tensor):
            items.extend(_create_read_item_for_tensor(key, md, value))
        else:
            items.append(_create_read_item_for_bytesio(key))
    return __import__("tensorplay.distributed.checkpoint.planner", fromlist=["LoadPlan"]).LoadPlan(items)


def create_default_global_load_plan(plans: list[Any]) -> list[Any]:
    return plans
