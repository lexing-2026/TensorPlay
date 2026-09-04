from __future__ import annotations

import io
import itertools
import math
from bisect import bisect_right, insort
from dataclasses import replace
from typing import Any, Callable

import tensorplay as tp

from ._traverse import get_element, traverse_state_dict
from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STORAGE_TYPES,
    TensorProperties,
    TensorStorageMetadata,
)
from .planner import BytesIOWriteData, LoadItemType, ReadItem, SavePlan, TensorWriteData, WriteItem, WriteItemType
from .protocol import (
    _get_checkpointable_tensor_chunks,
    _get_checkpointable_tensor_shard,
    _is_checkpointable_tensor,
)
from .resharding import _check_shard_metadata_pair_overlap, _shards_get_overlap_region_wrt_saved_tensor

__all__ = ["create_read_items_for_chunk_list"]


def _is_dtensor(value: Any) -> bool:
    return callable(getattr(value, "__create_write_items__", None)) and hasattr(value, "device_mesh") and hasattr(value, "placements")


def _is_sharded_tensor(value: Any) -> bool:
    return callable(getattr(value, "local_shards", None)) and callable(getattr(value, "metadata", None))


def _compare_save_plans(plan: SavePlan, other_plan: SavePlan) -> bool:
    if plan.usable != other_plan.usable:
        return False
    if len(plan.items) != len(other_plan.items):
        return False
    for plan_item, other_plan_item in zip(plan.items, other_plan.items):
        if plan_item.type != other_plan_item.type:
            return False
        plan_index = plan_item.index
        other_index = other_plan_item.index
        if (
            plan_index.fqn != other_index.fqn
            or plan_index.offset != other_index.offset
            or plan_index.index != other_index.index
        ):
            return False
        tensor_data = plan_item.tensor_data
        other_tensor_data = other_plan_item.tensor_data
        if (tensor_data is None) != (other_tensor_data is None):
            return False
        if tensor_data is not None and other_tensor_data is not None:
            if tensor_data.size != other_tensor_data.size:
                return False
            chunk = tensor_data.chunk
            other_chunk = other_tensor_data.chunk
            if (chunk is None) != (other_chunk is None):
                return False
            if chunk is not None and other_chunk is not None:
                if chunk.offsets != other_chunk.offsets or chunk.sizes != other_chunk.sizes:
                    return False
    return True


def _contains_usable_plan(delta_plans: list[SavePlan]) -> bool:
    return any(delta_plan and delta_plan.usable for delta_plan in delta_plans)


def _merge_delta_local_plans(
    cached_plans: list[SavePlan], delta_plans: list[SavePlan]
) -> list[SavePlan]:
    return [
        cached_plan if delta_plan and not delta_plan.usable else delta_plan
        for cached_plan, delta_plan in zip(cached_plans, delta_plans)
    ]


def _tensor_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in value.shape)


def _create_chunk_from_tensor(tensor: tp.Tensor) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(tuple(0 for _ in tensor.shape), _tensor_shape(tensor))


def _chunk_for_shard(shard_md: Any) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        tuple(int(value) for value in shard_md.shard_offsets),
        tuple(int(value) for value in shard_md.shard_sizes),
    )


def _sharded_tensor_metadata(sharded_tensor: Any, shard_md: Any) -> TensorWriteData:
    properties = sharded_tensor.metadata().tensor_properties
    return TensorWriteData(
        chunk=_chunk_for_shard(shard_md),
        properties=TensorProperties(
            dtype=properties.dtype,
            layout=properties.layout,
            requires_grad=properties.requires_grad,
            memory_format=properties.memory_format,
            pin_memory=properties.pin_memory,
        ),
        size=tuple(int(value) for value in sharded_tensor.metadata().size),
    )


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


def _create_write_items_for_dtensor(fqn: str, tensor: Any) -> WriteItem:
    return _create_write_item_for_dtensor(fqn, tensor)


def _create_write_item_for_shard(
    fqn: str, sharded_tensor: Any, shard_md: Any
) -> WriteItem:
    return WriteItem(
        MetadataIndex(fqn, _chunk_for_shard(shard_md).offsets),
        WriteItemType.SHARD,
        tensor_data=_sharded_tensor_metadata(sharded_tensor, shard_md),
    )


def _create_write_item_for_tensor(fqn: str, tensor: tp.Tensor) -> WriteItem:
    chunk = _create_chunk_from_tensor(tensor)
    return WriteItem(
        MetadataIndex(fqn, chunk.offsets),
        WriteItemType.TENSOR,
        tensor_data=TensorWriteData(
            chunk, TensorProperties.create_from_tensor(tensor), _tensor_shape(tensor)
        ),
    )


def _get_checkpointable_tensor_write_items(fqn: str, tensor: Any) -> list[WriteItem]:
    if not isinstance(tensor, tp.Tensor):
        raise TypeError("checkpointable tensor must also be a tensor")
    properties = TensorProperties.create_from_tensor(tensor)
    global_shape = tuple(int(value) for value in tensor.global_shape)
    return [
        WriteItem(
            MetadataIndex(fqn, chunk.offsets, index),
            WriteItemType.SHARD,
            tensor_data=TensorWriteData(chunk, properties, global_shape),
        )
        for index, chunk in enumerate(_get_checkpointable_tensor_chunks(tensor))
    ]


def _create_write_item_for_bytesio(fqn: str, bytes: Any) -> WriteItem:
    return WriteItem(MetadataIndex(fqn), WriteItemType.BYTE_IO)


def _create_write_items(fqn: str, object: Any) -> list[WriteItem]:
    create_items = getattr(object, "__create_write_items__", None)
    if callable(create_items):
        return list(create_items(fqn, object))
    if _is_sharded_tensor(object):
        return [
            _create_write_item_for_shard(fqn, object, shard.metadata)
            for shard in object.local_shards()
        ]
    if _is_checkpointable_tensor(object):
        return _get_checkpointable_tensor_write_items(fqn, object)
    return [_create_write_item_for_tensor(fqn, object)] if isinstance(object, tp.Tensor) else [_create_write_item_for_bytesio(fqn, object)]


def _create_chunk_list(tensor: Any) -> list[ChunkStorageMetadata]:
    create_chunks = getattr(tensor, "__create_chunk_list__", None)
    if callable(create_chunks):
        return list(create_chunks())
    if _is_checkpointable_tensor(tensor):
        return list(_get_checkpointable_tensor_chunks(tensor))
    if _is_sharded_tensor(tensor):
        return [_chunk_for_shard(shard.metadata) for shard in tensor.local_shards()]
    if isinstance(tensor, tp.Tensor):
        return [_create_chunk_from_tensor(tensor)]
    raise ValueError(f"unsupported tensor value {type(tensor)!r}")


def _create_write_items_from_state_dict(state_dict: dict[str, Any]) -> list[WriteItem]:
    items: list[WriteItem] = []
    traverse_state_dict(state_dict, lambda path, value: items.extend(_create_write_items(".".join(map(str, path)), value)))
    return items


def _create_default_metadata_only_plan(state_dict: dict[str, Any]) -> SavePlan:
    return SavePlan(_create_write_items_from_state_dict(state_dict))


def create_default_local_save_plan(state_dict: dict[str, Any], is_coordinator: bool = True) -> SavePlan:
    del is_coordinator
    return SavePlan(_create_write_items_from_state_dict(state_dict))


def create_default_global_save_plan(
    all_plans: list[SavePlan], rewrite_index_hints: bool = True
) -> tuple[list[SavePlan], Metadata]:
    metadata: dict[str, Any] = {}
    new_plans: list[SavePlan] = []
    for plan in all_plans:
        new_items: list[WriteItem] = []
        for item in plan.items:
            if item.type is not WriteItemType.SHARD and item.index.fqn in metadata:
                raise AssertionError("item.index.fqn not in metadata")
            if item.type is WriteItemType.BYTE_IO:
                metadata[item.index.fqn] = BytesStorageMetadata()
                new_items.append(item)
                continue
            if item.tensor_data is None:
                raise AssertionError("item.tensor_data is not None")
            tensor_metadata = metadata.setdefault(
                item.index.fqn,
                TensorStorageMetadata(
                    properties=item.tensor_data.properties,
                    size=item.tensor_data.size,
                    chunks=[],
                ),
            )
            if not isinstance(tensor_metadata, TensorStorageMetadata):
                raise AssertionError("tensor metadata has an incompatible type")
            new_item = item
            if rewrite_index_hints:
                new_item = replace(
                    item,
                    index=replace(item.index, index=len(tensor_metadata.chunks)),
                )
            new_items.append(new_item)
            tensor_metadata.chunks.append(item.tensor_data.chunk)
        new_plans.append(replace(plan, items=new_items))
    return new_plans, Metadata(metadata)


def _create_read_items_for_object(fqn: str, md: TensorStorageMetadata, obj: Any) -> list[ReadItem]:
    chunks = _create_chunk_list(obj)
    return create_read_items_for_chunk_list(fqn, md, chunks)


def _create_read_item_for_bytesio(fqn: str) -> ReadItem:
    return ReadItem(LoadItemType.BYTE_IO, MetadataIndex(fqn), (0,), MetadataIndex(fqn), (0,), (0,))


def _create_read_item_for_byteio(
    dest_index: MetadataIndex,
    dest_offset: int,
    storage_index: MetadataIndex,
    storage_offset: int,
    length: int,
) -> ReadItem:
    return ReadItem(
        LoadItemType.BYTE_IO,
        dest_index,
        (int(dest_offset),),
        storage_index,
        (int(storage_offset),),
        (int(length),),
    )


def _create_read_item_for_tensor(
    dest_index: MetadataIndex,
    dest_offsets: Any,
    storage_index: MetadataIndex,
    storage_offsets: Any,
    lengths: Any,
) -> ReadItem:
    return ReadItem(
        LoadItemType.TENSOR,
        dest_index,
        tuple(int(value) for value in dest_offsets),
        storage_index,
        tuple(int(value) for value in storage_offsets),
        tuple(int(value) for value in lengths),
    )


def create_read_items_for_chunk_list(
    fqn: str, checkpoint_md: TensorStorageMetadata, local_chunks: list[ChunkStorageMetadata]
) -> list[ReadItem]:
        local_chunks = list(local_chunks)
        saved_chunks = checkpoint_md.chunks
        if not local_chunks or not saved_chunks:
            return []
        dimensions = len(local_chunks[0].offsets)
        sweep_dimension = 0
        if dimensions > 1:
            sweep_dimension = max(
                range(dimensions),
                key=lambda dimension: max(
                    chunk.offsets[dimension] + chunk.sizes[dimension]
                    for chunk in itertools.chain(local_chunks, saved_chunks)
                ),
            )
        if dimensions == 0:
            saved_bounds = [(0, 1)] * len(saved_chunks)
            local_bounds = [(0, 1)] * len(local_chunks)
        else:
            saved_bounds = [
                (chunk.offsets[sweep_dimension], chunk.offsets[sweep_dimension] + chunk.sizes[sweep_dimension])
                for chunk in saved_chunks
            ]
            local_bounds = [
                (chunk.offsets[sweep_dimension], chunk.offsets[sweep_dimension] + chunk.sizes[sweep_dimension])
                for chunk in local_chunks
            ]
        saved_order = sorted(range(len(saved_chunks)), key=lambda index: saved_bounds[index][0])
        local_order = sorted(range(len(local_chunks)), key=lambda index: local_bounds[index][0])
        active: list[tuple[int, int]] = []
        saved_cursor = 0
        result: list[ReadItem] = []
        for local_index in local_order:
            local_chunk = local_chunks[local_index]
            local_start, local_end = local_bounds[local_index]
            cutoff = bisect_right(active, (local_start, -1))
            if cutoff:
                del active[:cutoff]
            while saved_cursor < len(saved_order):
                saved_index = saved_order[saved_cursor]
                saved_start, saved_end = saved_bounds[saved_index]
                if saved_start >= local_end:
                    break
                insort(active, (saved_end, saved_index))
                saved_cursor += 1
            for _, saved_index in active:
                saved_chunk = saved_chunks[saved_index]
                if not _check_shard_metadata_pair_overlap(local_chunk, saved_chunk):
                    continue
                storage_offsets: list[int] = []
                dest_offsets: list[int] = []
                lengths: list[int] = []
                for _, saved_offset, current_offset, length in _shards_get_overlap_region_wrt_saved_tensor(
                    saved_shard=saved_chunk, current_shard=local_chunk
                ):
                    storage_offsets.append(int(saved_offset))
                    dest_offsets.append(int(current_offset))
                    lengths.append(int(length))
                result.append(
                    _create_read_item_for_tensor(
                        MetadataIndex(fqn, local_chunk.offsets, local_index),
                        dest_offsets,
                        MetadataIndex(fqn, saved_chunk.offsets, saved_index),
                        storage_offsets,
                        lengths,
                    )
                )
        return result


def _create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> list[ReadItem]:
    if isinstance(md, BytesStorageMetadata):
        return [_create_read_item_for_bytesio(fqn)]
    try:
        local_chunks = _create_chunk_list(obj)
    except ValueError as error:
        raise ValueError(
            f"invalid checkpoint metadata for {fqn}: expected a tensor-compatible value"
        ) from error
    return create_read_items_for_chunk_list(fqn, md, local_chunks)


def create_default_local_load_plan(state_dict: dict[str, Any], metadata: Metadata, strict: bool = True):
    from .planner import LoadPlan

    items: list[ReadItem] = []
    for key, value in state_dict.items():
        if key not in metadata.state_dict_metadata:
            if strict:
                raise RuntimeError(f"missing key in checkpoint state dictionary: {key}")
            continue
        md = metadata.state_dict_metadata[key]
        if (
            isinstance(md, TensorStorageMetadata)
            and getattr(value, "shape", None) is not None
            and not _is_checkpointable_tensor(value)
        ):
            expected = tuple(int(item) for item in value.shape)
            if tuple(md.size) != expected:
                raise ValueError(f"size mismatch for {key}: saved {md.size}, current {expected}")
        coordinate = getattr(getattr(value, "device_mesh", None), "get_coordinate", None)
        if _is_dtensor(value) and callable(coordinate) and coordinate() is None:
            continue
        try:
            items.extend(_create_read_items(key, md, value))
        except (AttributeError, TypeError, ValueError):
            if strict:
                raise
    return LoadPlan(items)


def _create_default_local_metadata(state_dict: dict[str, Any]) -> Metadata:
    _, metadata = create_default_global_save_plan([_create_default_metadata_only_plan(state_dict)])
    return metadata


def _iterate_state_dict(
    iter_object: Any,
    dtensor_func: Callable[[Any], Any],
    sharded_tensor_func: Callable[[Any], Any],
    tensor_func: Callable[[Any], Any],
) -> Any:
    if _is_dtensor(iter_object):
        return dtensor_func(iter_object)
    if _is_sharded_tensor(iter_object):
        return sharded_tensor_func(iter_object)
    if isinstance(iter_object, tp.Tensor):
        return tensor_func(iter_object)
    if isinstance(iter_object, dict):
        for key, child in iter_object.items():
            iter_object[key] = _iterate_state_dict(child, dtensor_func, sharded_tensor_func, tensor_func)
        return iter_object
    if isinstance(iter_object, list):
        return [
            _iterate_state_dict(child, dtensor_func, sharded_tensor_func, tensor_func)
            for child in iter_object
        ]
    if isinstance(iter_object, tuple):
        return tuple(
            _iterate_state_dict(child, dtensor_func, sharded_tensor_func, tensor_func)
            for child in iter_object
        )
    return iter_object


def _init_state_dict(state_dict: dict[str, Any]) -> Any:
    def replace_tensor(value: Any) -> Any:
        if bool(getattr(value, "is_meta", False)):
            return tp.empty_like(value)
        return value

    def replace_dtensor(value: Any) -> Any:
        if bool(getattr(value, "is_meta", False)):
            local = value.to_local()
            return type(value)(
                tp.empty_like(local),
                value.device_mesh,
                value.placements,
                shape=value.shape,
                stride=getattr(value, "stride", lambda: None)(),
            )
        return value

    def replace_sharded(value: Any) -> Any:
        if bool(getattr(value, "is_meta", False)):
            raise RuntimeError(f"unsupported meta value type {type(value)}")
        return value

    return _iterate_state_dict(state_dict, replace_dtensor, replace_sharded, replace_tensor)


def create_default_global_load_plan(plans: list[Any]) -> list[Any]:
    return plans
