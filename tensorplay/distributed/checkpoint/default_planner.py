from __future__ import annotations

import io
import pickle
from typing import Any

import tensorplay as tp

from ._nested_dict import flatten_state_dict, unflatten_state_dict
from ._traverse import get_element
from .metadata import Metadata, MetadataIndex, StorageMeta, TensorStorageMetadata
from .planner import LoadPlan, LoadPlanner, SavePlan, SavePlanner, WriteItemType
from .planner_helpers import create_default_global_load_plan, create_default_global_save_plan, create_default_local_load_plan, create_default_local_save_plan

__all__ = ["DefaultSavePlanner", "DefaultLoadPlanner", "create_default_local_load_plan", "create_default_global_load_plan", "create_default_local_save_plan", "create_default_global_save_plan"]


def _narrow(value: tp.Tensor, offsets: tuple[int, ...], lengths: tuple[int, ...]) -> tp.Tensor:
    result = value
    for dim, (offset, length) in enumerate(zip(offsets, lengths)):
        if length:
            result = result.narrow(dim, int(offset), int(length))
    return result


class DefaultSavePlanner(SavePlanner):
    def __init__(self, flatten_state_dict: bool = True, flatten_sharded_tensors: bool = True, dedup_replicated_tensors: bool | None = None, dedup_save_to_lowest_rank: bool = False, enable_plan_caching: bool = False) -> None:
        del dedup_replicated_tensors
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.dedup_save_to_lowest_rank = dedup_save_to_lowest_rank
        self._enable_plan_caching = enable_plan_caching
        self.mappings: dict[str, tuple[Any, ...]] = {}

    def set_up_planner(self, state_dict: dict[str, Any], storage_meta: StorageMeta | None = None, is_coordinator: bool = False) -> None:
        del storage_meta
        if self.flatten_state_dict:
            self.state_dict, self.mappings = flatten_state_dict(state_dict)
        else:
            self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        self.plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        if self.flatten_state_dict:
            self.plan = type(self.plan)(self.plan.items, self.plan.storage_data, self.mappings, self.plan.usable)
        return self.plan

    def create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
        self.global_plan, self.metadata = create_default_global_save_plan(all_plans)
        return self.global_plan, self.metadata

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self.plan = new_plan
        return new_plan

    def lookup_object(self, index: MetadataIndex) -> Any:
        return self.state_dict[index.fqn]

    def transform_object(self, write_item: Any, value: Any) -> Any:
        if write_item.type is WriteItemType.BYTE_IO:
            stream = io.BytesIO()
            pickle.dump(value, stream, pickle.HIGHEST_PROTOCOL)
            stream.seek(0)
            return stream
        return value

    def resolve_data(self, write_item: Any) -> Any:
        return self.transform_object(write_item, self.lookup_object(write_item.index))


class DefaultLoadPlanner(LoadPlanner):
    def __init__(self, flatten_state_dict: bool = True, flatten_sharded_tensors: bool = True, allow_partial_load: bool = False) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.allow_partial_load = allow_partial_load
        self.original_state_dict: dict[str, Any] = {}
        self.mappings: dict[str, tuple[Any, ...]] = {}

    def set_up_planner(self, state_dict: dict[str, Any], metadata: Metadata | None = None, is_coordinator: bool = False) -> None:
        self.original_state_dict = state_dict
        if self.flatten_state_dict:
            self.state_dict, self.mappings = flatten_state_dict(state_dict)
        else:
            self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        if self.metadata is None:
            raise ValueError("metadata is required")
        self.plan = create_default_local_load_plan(self.state_dict, self.metadata, not self.allow_partial_load)
        return self.plan

    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        return create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def load_bytes(self, read_item: Any, value: io.BytesIO) -> None:
        value.seek(0)
        loaded = pickle.load(value)
        key = read_item.dest_index.fqn
        if self.flatten_state_dict and key in self.mappings:
            current: Any = self.original_state_dict
            path = self.mappings[key]
            for part in path[:-1]:
                current = current[part]
            current[path[-1]] = loaded
        else:
            self.state_dict[key] = loaded

    def lookup_tensor(self, index: MetadataIndex) -> tp.Tensor:
        value = self.state_dict[index.fqn]
        if not isinstance(value, tp.Tensor):
            raise TypeError(f"{index.fqn} is not a tensor")
        return value

    def resolve_tensor(self, read_item: Any) -> tp.Tensor:
        tensor = _narrow(self.lookup_tensor(read_item.dest_index), read_item.dest_offsets, read_item.lengths)
        return self.transform_tensor(read_item, tensor)

    def transform_tensor(self, read_item: Any, tensor: tp.Tensor) -> tp.Tensor:
        return tensor

    def commit_tensor(self, read_item: Any, tensor: tp.Tensor) -> None:
        destination = self.lookup_tensor(read_item.dest_index)
        destination.copy_(tensor)


class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    def __init__(self, keys: set[str] | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.keys = keys

    def set_up_planner(self, state_dict: dict[str, Any], metadata: Metadata | None = None, is_coordinator: bool = False) -> None:
        if state_dict:
            raise ValueError("empty planner requires an empty state dictionary")
        if metadata is None:
            raise ValueError("metadata is required")
        for key, value in metadata.state_dict_metadata.items():
            if self.keys is not None and key not in self.keys:
                continue
            if isinstance(value, TensorStorageMetadata):
                state_dict[key] = tp.empty(value.size, dtype=value.properties.dtype)
            else:
                state_dict[key] = None
        super().set_up_planner(state_dict, metadata, is_coordinator)
