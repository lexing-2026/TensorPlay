from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import tensorplay as tp
import tensorplay.distributed as dist

from ._nested_dict import flatten_state_dict, unflatten_state_dict
from .default_planner import DefaultLoadPlanner
from .metadata import (
    BytesStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from .planner import LoadPlan, LoadPlanner
from .planner_helpers import _create_read_items, create_read_items_for_chunk_list
from .state_dict_loader import load as load_state_dict
from .utils import _element_wise_add, _element_wise_sub, _normalize_device_info

__all__ = ["load_sharded_optimizer_state_dict"]

STATE_DICT_2D_LAYOUT = dict[str, tuple[Sequence[int] | None, Sequence[int]]]


def _gen_rank_device(global_rank: int, device_type: str = "cuda") -> str:
    """Return a stable device placement for a process rank."""
    if device_type == "cpu":
        return "cpu"
    device_module = getattr(tp, device_type, None)
    is_available = getattr(device_module, "is_available", None)
    if callable(is_available) and not is_available():
        return "cpu"
    count = getattr(device_module, "device_count", None)
    try:
        device_count = int(count()) if callable(count) else 1
    except Exception:
        device_count = 1
    return _normalize_device_info(device_type, int(global_rank) % max(device_count, 1))


def _group_device_type(process_group: Any = None) -> str:
    if process_group is not None:
        backend = str(getattr(process_group, "backend", "")).lower()
        if "cuda" in backend or "nccl" in backend:
            return "cuda"
    return "cpu"


def _group_global_ranks(process_group: Any = None) -> list[int]:
    if not dist.is_initialized():
        return [0]
    if process_group is None:
        return list(range(dist.get_world_size()))
    return [
        int(dist.get_global_rank(process_group, index))
        for index in range(dist.get_world_size(process_group))
    ]


def _create_colwise_spec(pg: Any = None) -> Any:
    """Build a dimension-zero sharding specification for a process group."""
    from .._shard.sharding_spec import ChunkShardingSpec

    device_type = _group_device_type(pg)
    placements = [
        f"rank:{rank}/{_gen_rank_device(rank, device_type)}"
        for rank in _group_global_ranks(pg)
    ]
    return ChunkShardingSpec(dim=0, placements=placements)


def _is_nested_tensor(val: Any) -> bool:
    """Detect unsupported nested distributed tensor containers."""
    from .._shard.sharded_tensor import ShardedTensor
    from ..tensor import DTensor

    if type(val) is ShardedTensor:
        local_shards = val.local_shards()
        if not local_shards:
            return False
        local = local_shards[0].tensor
        if type(local) is ShardedTensor:
            return True
        if type(local) is DTensor:
            raise ValueError("nested distributed tensor state is not supported")
        return False
    if type(val) is DTensor:
        local = val.to_local()
        if type(local) is DTensor or type(local) is ShardedTensor:
            raise ValueError("nested distributed tensor state is not supported")
    return False


def _alloc_tensor(
    props: TensorProperties,
    size: Sequence[int],
    device_type: Any = "cuda",
) -> tp.Tensor:
    if device_type is None:
        device_type = "cpu"
    if not isinstance(device_type, str):
        device = device_type
    elif device_type == "cpu":
        device = "cpu"
    else:
        device_module = getattr(tp, device_type, None)
        is_available = getattr(device_module, "is_available", None)
        if callable(is_available) and not is_available():
            device = "cpu"
        else:
            current_device = getattr(device_module, "current_device", None)
            device_index = int(current_device()) if callable(current_device) else 0
            device = _normalize_device_info(device_type, device_index)
    kwargs: dict[str, Any] = {
        "dtype": props.dtype,
        "requires_grad": bool(props.requires_grad),
        "pin_memory": bool(props.pin_memory),
        "device": device,
    }
    return tp.empty(tuple(int(value) for value in size), **kwargs)


def _layout_for_value(value: Any) -> tuple[tuple[int, ...] | None, tuple[int, ...], Any]:
    """Return the local offset, local shape and communication group."""
    from .._shard.sharded_tensor import ShardedTensor
    from ..tensor import DTensor

    if isinstance(value, DTensor):
        chunks = value.__create_chunk_list__()
        if not chunks:
            return None, tuple(int(item) for item in value.shape), None
        process_group = None
        for mesh_dim, placement in enumerate(value.placements):
            if getattr(placement, "is_shard", lambda: False)():
                process_group = value.device_mesh.get_group(mesh_dim)
                break
        chunk = chunks[0]
        return tuple(chunk.offsets), tuple(chunk.sizes), process_group

    if isinstance(value, ShardedTensor):
        local_shards = value.local_shards()
        if len(local_shards) > 1:
            raise ValueError("one local optimizer shard is required per rank")
        if not local_shards:
            return None, tuple(int(item) for item in value.shape), value._process_group
        metadata = local_shards[0].metadata
        return (
            tuple(int(item) for item in metadata.shard_offsets),
            tuple(int(item) for item in metadata.shard_sizes),
            value._process_group,
        )

    if isinstance(value, tp.Tensor):
        return None, tuple(int(item) for item in value.shape), None
    return None, (), None


def _get_state_dict_2d_layout(
    state_dict: Mapping[str, Any],
) -> tuple[dict[str, tuple[Sequence[int] | None, Sequence[int]]], Any]:
    """Collect local slices used by optimizer tensors."""
    specs: dict[str, tuple[Sequence[int] | None, Sequence[int]]] = {}
    process_group = None
    from .._shard.sharded_tensor import ShardedTensor

    for key, value in state_dict.items():
        if not hasattr(value, "shape"):
            continue
        specs[str(key)] = (None, tuple(int(item) for item in value.shape))
        if _is_nested_tensor(value):
            local_shards = value.local_shards()
            if len(local_shards) != 1:
                raise AssertionError("one local optimizer shard is required")
            if not isinstance(value, ShardedTensor):
                raise AssertionError("nested optimizer values must be sharded tensors")
            shard = local_shards[0]
            specs[str(key)] = (
                tuple(int(item) for item in shard.metadata.shard_offsets),
                tuple(int(item) for item in shard.metadata.shard_sizes),
            )
            process_group = getattr(shard.tensor, "_process_group", None)
        else:
            offset, size, group = _layout_for_value(value)
            if offset is not None:
                specs[str(key)] = (offset, size)
            if process_group is None and group is not None:
                process_group = group
    return specs, process_group


def _metadata_path(metadata: Metadata, key: str) -> tuple[Any, ...]:
    planner_data = metadata.planner_data
    if isinstance(planner_data, Mapping):
        path = planner_data.get(key)
        if isinstance(path, (tuple, list)) and path:
            return tuple(path)
    return tuple(key.split("."))


def _find_model_name(
    path: tuple[Any, ...],
    optimizer_key: str | None,
    model_values: Mapping[str, Any],
    full_key: str,
) -> str | None:
    if optimizer_key is not None and path and str(path[0]) == optimizer_key:
        relative = path[1:]
    else:
        prefix = f"{optimizer_key}." if optimizer_key else ""
        relative_text = (
            full_key[len(prefix):]
            if prefix and full_key.startswith(prefix)
            else full_key
        )
        relative = tuple(relative_text.split("."))

    if len(relative) >= 3 and str(relative[0]) == "state":
        candidate = str(relative[1])
        if candidate in model_values:
            return candidate

    if optimizer_key is not None and full_key.startswith(f"{optimizer_key}.state."):
        remainder = full_key[len(f"{optimizer_key}.state."):]
        candidates = [
            name
            for name in model_values
            if remainder == name or remainder.startswith(f"{name}.")
        ]
        if candidates:
            return max(candidates, key=len)
    return None


def _template_for_optimizer_key(
    key: str,
    metadata: Metadata,
    optimizer_key: str | None,
    model_values: Mapping[str, Any],
) -> Any:
    name = _find_model_name(
        _metadata_path(metadata, key), optimizer_key, model_values, key
    )
    return model_values.get(name) if name is not None else None


def _empty_like_layout(
    template: Any,
    properties: TensorProperties,
    size: tuple[int, ...],
) -> Any:
    """Allocate a local optimizer state with the model's distributed layout."""
    from .._shard.sharded_tensor import Shard, ShardedTensor
    from ..tensor import DTensor

    if (
        not hasattr(template, "shape")
        or tuple(int(item) for item in template.shape) != size
    ):
        device = getattr(template, "device", None) if template is not None else None
        return _alloc_tensor(properties, size, device)

    if isinstance(template, DTensor):
        local = template.to_local()
        result = _alloc_tensor(
            properties, tuple(int(item) for item in local.shape), local.device
        )
        return DTensor.from_local(
            result,
            template.device_mesh,
            template.placements,
            shape=template.shape,
            stride=template.stride(),
        )

    if isinstance(template, ShardedTensor):
        local_shards = []
        for shard in template.local_shards():
            local = _alloc_tensor(
                properties,
                tuple(int(item) for item in shard.tensor.shape),
                shard.tensor.device,
            )
            local_shards.append(Shard(local, copy.deepcopy(shard.metadata)))
        return type(template)._init_from_local_shards_and_global_metadata(
            local_shards,
            copy.deepcopy(template.metadata()),
            getattr(template, "_sharding_spec", None),
            getattr(template, "_process_group", None),
        )

    return _alloc_tensor(properties, size, getattr(template, "device", None))


def _make_flat_destination(
    metadata: Metadata,
    optimizer_key: str | None,
    model_state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    model_values, _ = flatten_state_dict(model_state_dict)
    destination: dict[str, Any] = {}
    for key, description in metadata.state_dict_metadata.items():
        path = _metadata_path(metadata, key)
        belongs = (
            optimizer_key is None
            or (path and str(path[0]) == optimizer_key)
            or key == optimizer_key
            or key.startswith(f"{optimizer_key}.")
        )
        if not belongs:
            continue
        if isinstance(description, BytesStorageMetadata):
            destination[key] = None
            continue
        if not isinstance(description, TensorStorageMetadata):
            raise TypeError(f"unsupported metadata for {key}: {type(description)!r}")
        size = tuple(int(item) for item in description.size)
        template = _template_for_optimizer_key(
            key, metadata, optimizer_key, model_values
        )
        destination[key] = _empty_like_layout(template, description.properties, size)
    return destination


def _optimizer_result(
    values: Mapping[str, Any], metadata: Metadata, optimizer_key: str | None
) -> dict[str, Any]:
    paths: dict[str, tuple[Any, ...]] = {}
    for key in values:
        path = _metadata_path(metadata, key)
        if optimizer_key is not None and (
            not path or str(path[0]) != optimizer_key
        ):
            if key == optimizer_key:
                path = (optimizer_key,)
            elif key.startswith(f"{optimizer_key}."):
                path = (optimizer_key,) + tuple(
                    key[len(optimizer_key) + 1 :].split(".")
                )
        paths[key] = path
    return unflatten_state_dict(values, paths)


def _load_checkpoint_values(
    model_state_dict: Mapping[str, Any],
    optimizer_key: str | None,
    storage_reader: Any,
    planner: LoadPlanner | None,
) -> tuple[dict[str, Any], Metadata]:
    if not hasattr(storage_reader, "read_metadata"):
        raise TypeError("storage_reader must provide read_metadata()")
    metadata = storage_reader.read_metadata()
    if not isinstance(metadata, Metadata):
        raise TypeError("checkpoint metadata must be a Metadata object")
    destination = _make_flat_destination(metadata, optimizer_key, model_state_dict)
    if not destination:
        selected = optimizer_key or "optimizer state"
        raise KeyError(f"checkpoint does not contain {selected}")
    load_planner = planner or DefaultLoadPlanner(flatten_state_dict=False)
    load_state_dict(
        destination,
        storage_reader=storage_reader,
        planner=load_planner,
        process_group=None,
    )
    return destination, metadata


def load_sharded_optimizer_state_dict(
    model_state_dict: dict[str, Any],
    optimizer_key: str,
    storage_reader: Any,
    planner: LoadPlanner | None = None,
) -> dict[str, Any]:
    if not isinstance(model_state_dict, Mapping):
        raise TypeError("model_state_dict must be a mapping")
    metadata = storage_reader.read_metadata()
    layout_specs, dp_pg = _get_state_dict_2d_layout(model_state_dict)
    dp_pg_device_type = _group_device_type(dp_pg)
    device_module = getattr(tp, dp_pg_device_type, None)
    device_count = getattr(device_module, "device_count", None)
    num_devices_per_node = (
        max(int(device_count()), 1) if callable(device_count) else 1
    )
    if dist.is_initialized():
        world_size = dist.get_world_size(dp_pg)
        current_rank = dist.get_rank(dp_pg)
        current_global_rank = dist.get_rank()
    else:
        world_size = 1
        current_rank = 0
        current_global_rank = 0

    from .._shard.sharded_tensor import Shard, ShardedTensor
    from .._shard.sharded_tensor import TensorProperties as ShardTensorProperties
    from .._shard.sharding_spec import ChunkShardingSpec
    from ..fsdp._shard_utils import _create_chunk_sharded_tensor

    if dp_pg is None:
        placements = [
            f"rank:{rank}/{_gen_rank_device(rank, dp_pg_device_type)}"
            for rank in range(world_size)
        ]
        sharding_spec = ChunkShardingSpec(dim=0, placements=placements)
    else:
        placements = [
            f"rank:{rank}/{_gen_rank_device(rank, dp_pg_device_type)}"
            for rank in _group_global_ranks(dp_pg)
        ]
        sharding_spec = ChunkShardingSpec(dim=0, placements=placements)

    state_dict: dict[str, Any] = {}
    fqn_to_offset: dict[str, Sequence[int]] = {}
    planner_data = metadata.planner_data or {}
    for key, description in metadata.state_dict_metadata.items():
        key_path = planner_data.get(key, tuple(str(key).split(".")))
        if not key_path or key_path[0] != optimizer_key:
            continue
        if isinstance(description, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"
            continue
        if not isinstance(description, TensorStorageMetadata):
            raise TypeError(f"unsupported metadata for {key}: {type(description)!r}")
        size = tuple(int(item) for item in description.size)
        if math.prod(size) == 1:
            state_dict[key] = _alloc_tensor(
                description.properties, size, dp_pg_device_type
            )
        elif dp_pg is None:
            state_dict[key] = _create_chunk_sharded_tensor(
                _alloc_tensor(description.properties, size, dp_pg_device_type),
                rank=current_rank,
                world_size=world_size,
                num_devices_per_node=num_devices_per_node,
                pg=dist._get_default_group() if dist.is_initialized() else None,
            )
        else:
            spec_key = key_path[2] if len(key_path) > 2 else key
            alloc_size = layout_specs.get(spec_key, (None, size))[1]
            properties = ShardTensorProperties(
                dtype=description.properties.dtype,
                layout=description.properties.layout,
                requires_grad=description.properties.requires_grad,
                memory_format=description.properties.memory_format,
                pin_memory=description.properties.pin_memory,
            )
            sharded_metadata = sharding_spec.build_metadata(alloc_size, properties)
            local_shards = []
            for shard_metadata in sharded_metadata.shards_metadata:
                placement_rank = shard_metadata.placement.rank()
                if int(placement_rank) != int(current_global_rank):
                    continue
                local_shards.append(
                    Shard(
                        tensor=_alloc_tensor(
                            description.properties,
                            shard_metadata.shard_sizes,
                            dp_pg_device_type,
                        ),
                        metadata=shard_metadata,
                    )
                )
            state_dict[key] = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards,
                sharded_metadata,
                sharding_spec,
                dp_pg,
            )
            if spec_key in layout_specs and layout_specs[spec_key][0] is not None:
                fqn_to_offset[key] = layout_specs[spec_key][0]

    load_state_dict(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=_ReaderWithOffset(fqn_to_offset) if dp_pg is not None else planner,
        process_group=dp_pg,
    )
    return unflatten_state_dict(state_dict, planner_data)


class _ReaderWithOffset(DefaultLoadPlanner):
    """Translate destination offsets for a local distributed tensor slice."""

    def __init__(self, fqn_to_offset: Mapping[str, Sequence[int]]) -> None:
        super().__init__(flatten_state_dict=False)
        self.fqn_to_offset = {
            key: tuple(int(item) for item in offset)
            for key, offset in fqn_to_offset.items()
        }
        self.translation: dict[MetadataIndex, MetadataIndex] = {}

    def create_local_plan(self) -> LoadPlan:
        from .._shard.sharded_tensor import ShardedTensor

        self.translation = {}
        requests = []
        for fqn, value in self.state_dict.items():
            metadata = self.metadata.state_dict_metadata[fqn]
            if not isinstance(value, ShardedTensor):
                requests.extend(_create_read_items(fqn, metadata, value))
                continue
            if fqn not in self.fqn_to_offset:
                requests.extend(_create_read_items(fqn, metadata, value))
                continue
            offset = self.fqn_to_offset[fqn]
            local_shards = value.local_shards()
            if len(local_shards) != 1:
                raise AssertionError("one local optimizer shard is required")
            original_shard = local_shards[0]
            local_chunks = [
                ChunkStorageMetadata(
                    offsets=tuple(
                        _element_wise_add(original_shard.metadata.shard_offsets, offset)
                    ),
                    sizes=tuple(original_shard.metadata.shard_sizes),
                )
            ]
            read_items = create_read_items_for_chunk_list(
                fqn, metadata, local_chunks
            )
            for read_item in read_items:
                if read_item.dest_index.offset is None:
                    raise AssertionError("dest_index.offset must not be None")
                original_offset = _element_wise_sub(
                    read_item.dest_index.offset, offset
                )
                original_index = replace(
                    read_item.dest_index,
                    offset=tuple(original_offset),
                )
                self.translation[read_item.dest_index] = original_index
            requests.extend(read_items)
        return LoadPlan(requests)

    def lookup_tensor(self, index: MetadataIndex) -> tp.Tensor:
        return super().lookup_tensor(self.translation.get(index, index))
