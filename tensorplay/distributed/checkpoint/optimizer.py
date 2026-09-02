from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
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
from .state_dict_loader import load as load_state_dict

__all__ = ["load_sharded_optimizer_state_dict"]


def _gen_rank_device(global_rank: int, device_type: str = "cuda") -> str:
    """Return a stable device placement for a process rank."""
    if device_type == "cpu":
        return "cpu"
    device_module = getattr(tp, device_type, None)
    count = getattr(device_module, "device_count", None)
    try:
        device_count = int(count()) if callable(count) else 1
    except Exception:
        device_count = 1
    return f"{device_type}:{int(global_rank) % max(device_count, 1)}"


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


def _create_colwise_spec(process_group: Any = None) -> Any:
    """Build a dimension-zero sharding specification for a process group."""
    from .._shard.sharding_spec import ChunkShardingSpec

    device_type = _group_device_type(process_group)
    placements = [
        f"rank:{rank}/{_gen_rank_device(rank, device_type)}"
        for rank in _group_global_ranks(process_group)
    ]
    return ChunkShardingSpec(dim=0, placements=placements)


def _is_nested_tensor(value: Any) -> bool:
    """Detect unsupported nested distributed tensor containers."""
    from .._shard.sharded_tensor import ShardedTensor
    from ..tensor import DTensor

    if isinstance(value, ShardedTensor):
        local_shards = value.local_shards()
        if not local_shards:
            return False
        local = local_shards[0].tensor
        if isinstance(local, (ShardedTensor, DTensor)):
            raise ValueError("nested distributed tensor state is not supported")
        return False
    if isinstance(value, DTensor) and isinstance(
        value.to_local(), (ShardedTensor, DTensor)
    ):
        raise ValueError("nested distributed tensor state is not supported")
    return False


def _alloc_tensor(
    properties: TensorProperties,
    size: Sequence[int],
    device: Any = None,
) -> tp.Tensor:
    """Allocate a tensor using checkpoint properties."""
    kwargs: dict[str, Any] = {
        "dtype": properties.dtype,
        "requires_grad": bool(properties.requires_grad),
    }
    if device is not None:
        kwargs["device"] = device
    if properties.pin_memory:
        kwargs["pin_memory"] = True
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
    for key, value in state_dict.items():
        if not hasattr(value, "shape"):
            continue
        _is_nested_tensor(value)
        offset, size, group = _layout_for_value(value)
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
    if isinstance(storage_reader, (str, bytes)):
        from .filesystem import FileSystemReader

        storage_reader = FileSystemReader(storage_reader)
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


def _load_legacy_optimizer_state(
    model: Any,
    optimizer: Any,
    checkpoint_id: Any,
    process_group: Any = None,
) -> dict[str, Any]:
    del process_group
    if not callable(getattr(model, "state_dict", None)):
        raise TypeError("model must provide state_dict()")
    if not callable(getattr(optimizer, "state_dict", None)):
        raise TypeError("optimizer must provide state_dict()")
    from .filesystem import FileSystemReader

    loaded, metadata = _load_checkpoint_values(
        model.state_dict(), "optimizer", FileSystemReader(checkpoint_id), None
    )
    result = _optimizer_result(loaded, metadata, "optimizer")
    optimizer_state = result.get("optimizer", result)
    load_optimizer = getattr(optimizer, "load_state_dict", None)
    if callable(load_optimizer):
        load_optimizer(optimizer_state)
    return result


def load_sharded_optimizer_state_dict(
    model_state_dict: Any = None,
    optimizer_key: Any = None,
    storage_reader: Any = None,
    planner: LoadPlanner | None = None,
    **legacy_kwargs: Any,
) -> dict[str, Any]:
    """Load the optimizer portion of a distributed checkpoint.

    The primary interface accepts a model state dictionary, an optimizer key,
    and a storage reader.  A convenience form accepting ``model``,
    ``optimizer`` and ``checkpoint_id`` is also accepted.
    """
    if model_state_dict is None and "model" in legacy_kwargs:
        model_state_dict = legacy_kwargs.pop("model")
        optimizer = legacy_kwargs.pop("optimizer", None)
        checkpoint_id = legacy_kwargs.pop("checkpoint_id", None)
        if optimizer is None or checkpoint_id is None:
            raise TypeError("model, optimizer and checkpoint_id are required")
        return _load_legacy_optimizer_state(
            model_state_dict,
            optimizer,
            checkpoint_id,
            legacy_kwargs.pop("process_group", None),
        )

    if not isinstance(optimizer_key, str):
        if optimizer_key is None and "optimizer" in legacy_kwargs:
            optimizer_key = legacy_kwargs.pop("optimizer")
        if not isinstance(optimizer_key, str):
            if storage_reader is None:
                raise TypeError("optimizer_key must be a string")
            return _load_legacy_optimizer_state(
                model_state_dict,
                optimizer_key,
                storage_reader,
                legacy_kwargs.pop("process_group", None),
            )

    if legacy_kwargs:
        unexpected = next(iter(legacy_kwargs))
        raise TypeError(f"unexpected argument {unexpected!r}")
    if not isinstance(model_state_dict, Mapping):
        raise TypeError("model_state_dict must be a mapping")
    if storage_reader is None:
        raise TypeError("storage_reader is required")
    values, metadata = _load_checkpoint_values(
        model_state_dict, optimizer_key, storage_reader, planner
    )
    return _optimizer_result(values, metadata, optimizer_key)


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
        plan = super().create_local_plan()
        self.translation = {}
        for item in plan.items:
            offset = self.fqn_to_offset.get(item.dest_index.fqn)
            if offset is None or item.dest_index.offset is None:
                continue
            original = tuple(
                int(left) - int(right)
                for left, right in zip(item.dest_index.offset, offset)
            )
            self.translation[item.dest_index] = MetadataIndex(
                item.dest_index.fqn, original, item.dest_index.index
            )
        return plan

    def lookup_tensor(self, index: MetadataIndex) -> tp.Tensor:
        return super().lookup_tensor(self.translation.get(index, index))
