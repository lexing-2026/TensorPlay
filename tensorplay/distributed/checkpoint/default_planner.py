from __future__ import annotations

import io
import logging
import math
import pickle
import sys
from bisect import bisect_right, insort
from collections import ChainMap
from dataclasses import replace
from typing import Any

import tensorplay as tp

from ._nested_dict import flatten_state_dict, unflatten_state_dict
from ._sharded_tensor_utils import _flatten_sharded_tensors
from ._traverse import get_element, set_element
from ._dedup_save_plans import dedup_save_plans
from .metadata import BytesStorageMetadata, Metadata, MetadataIndex, StorageMeta, TensorStorageMetadata
from .planner import LoadPlan, LoadPlanner, SavePlan, SavePlanner, WriteItemType
from .planner_helpers import (
    _compare_save_plans,
    _contains_usable_plan,
    _merge_delta_local_plans,
    _create_default_local_metadata as _planner_create_default_local_metadata,
    create_default_global_load_plan as _planner_create_default_global_load_plan,
    create_default_global_save_plan as _planner_create_default_global_save_plan,
    create_default_local_load_plan as _planner_create_default_local_load_plan,
    create_default_local_save_plan as _planner_create_default_local_save_plan,
    _create_default_metadata_only_plan,
    _create_read_items,
    _create_write_items,
    _init_state_dict,
)

__all__ = ["DefaultSavePlanner", "DefaultLoadPlanner", "create_default_local_load_plan", "create_default_global_load_plan", "create_default_local_save_plan", "create_default_global_save_plan"]

logger = logging.getLogger(__name__)


def _narrow(value: tp.Tensor, offsets: tuple[int, ...], lengths: tuple[int, ...]) -> tp.Tensor:
    result = value
    for dim, (offset, length) in enumerate(zip(offsets, lengths)):
        if length:
            result = result.narrow(dim, int(offset), int(length))
    return result


class DefaultSavePlanner(SavePlanner):
    def __init__(self, flatten_state_dict: bool = True, flatten_sharded_tensors: bool = True, dedup_replicated_tensors: bool | None = None, dedup_save_to_lowest_rank: bool = False, enable_plan_caching: bool = False) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        del dedup_replicated_tensors
        self.dedup_save_to_lowest_rank = dedup_save_to_lowest_rank
        self._cached_plans_key = self.__class__.__name__
        self._enable_plan_caching = enable_plan_caching
        self.mappings: dict[str, tuple[Any, ...]] = {}

    def set_up_planner(self, state_dict: dict[str, Any], storage_meta: StorageMeta | None = None, is_coordinator: bool = False) -> None:
        del storage_meta
        if self.flatten_state_dict:
            self.state_dict, self.mappings = flatten_state_dict(state_dict)
        else:
            self.state_dict = state_dict
        if self.flatten_sharded_tensors:
            self.state_dict = _flatten_sharded_tensors(self.state_dict)
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        if self.flatten_state_dict:
            plan = replace(plan, planner_data=self.mappings)
        self.plan = plan
        if self._enable_plan_caching:
            if (
                self._cached_plans_key in SavePlanner._cached_save_plan
                and _compare_save_plans(
                    plan, SavePlanner._cached_save_plan[self._cached_plans_key]
                )
            ):
                return SavePlan([], usable=False)
            self._pending_local_plan = plan
        return self.plan

    def create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
        if self._enable_plan_caching:
            global_plan_delta, global_plan, metadata = self._create_global_plan_with_caching(all_plans)
        else:
            global_plan, metadata = self._create_global_plan(all_plans)
            global_plan_delta = global_plan
        self.global_plan = global_plan
        self.metadata = metadata
        return global_plan_delta, self.metadata

    def _dedup_save_plans(self, all_plans: list[SavePlan]) -> list[SavePlan]:
        return dedup_save_plans(all_plans, self.dedup_save_to_lowest_rank)

    def _create_global_plan(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], Metadata]:
        deduped_plans = self._dedup_save_plans(all_plans)
        global_plan, metadata = create_default_global_save_plan(deduped_plans)
        if self.flatten_state_dict:
            planner_data_dict = [plan.planner_data for plan in global_plan]
            metadata = replace(
                metadata,
                planner_data=dict(ChainMap(*planner_data_dict)),
            )
        validation_errors = _validate_global_plan(global_plan, metadata)
        if validation_errors:
            error_summary = "; ".join(validation_errors)
            if len(error_summary) > 500:
                error_summary = error_summary[:500] + "... (truncated)"
            raise ValueError(f"Failed to validate global plan: {error_summary}")
        return global_plan, metadata

    def _create_global_plan_with_caching(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], list[SavePlan], Metadata]:
        global_plan_delta: list[SavePlan] = []
        if self._cached_plans_key not in SavePlanner._cached_all_plans:
            global_plan, metadata = self._create_global_plan(all_plans)
            SavePlanner._cached_all_plans[self._cached_plans_key] = all_plans
            SavePlanner._cached_global_plan[self._cached_plans_key] = global_plan
            SavePlanner._cached_metadata[self._cached_plans_key] = metadata
            return global_plan, global_plan, metadata

        if not _contains_usable_plan(all_plans):
            global_plan_delta = [SavePlan([], usable=False)] * len(all_plans)
            global_plan = SavePlanner._cached_global_plan[self._cached_plans_key]
            metadata = SavePlanner._cached_metadata[self._cached_plans_key]
        else:
            merged_plans = _merge_delta_local_plans(
                SavePlanner._cached_all_plans[self._cached_plans_key], all_plans
            )
            SavePlanner._cached_all_plans[self._cached_plans_key] = merged_plans
            global_plan, metadata = self._create_global_plan(merged_plans)
            for cached_plan, new_plan in zip(
                SavePlanner._cached_global_plan[self._cached_plans_key], global_plan
            ):
                if _compare_save_plans(cached_plan, new_plan):
                    global_plan_delta.append(SavePlan([], usable=False))
                else:
                    global_plan_delta.append(new_plan)
            SavePlanner._cached_global_plan[self._cached_plans_key] = global_plan
            SavePlanner._cached_metadata[self._cached_plans_key] = metadata
        return global_plan_delta, global_plan, metadata

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        finished_plan = new_plan
        if self._enable_plan_caching:
            finished_plan = self._finish_plan_with_caching(new_plan)
            if hasattr(self, "_pending_local_plan"):
                SavePlanner._cached_save_plan[self._cached_plans_key] = self._pending_local_plan
                del self._pending_local_plan
        self.plan = finished_plan
        return self.plan

    def _finish_plan_with_caching(self, new_plan: SavePlan) -> SavePlan:
        if not new_plan.usable:
            return SavePlanner._cached_final_save_plan[self._cached_plans_key]
        SavePlanner._cached_final_save_plan[self._cached_plans_key] = new_plan
        return new_plan

    def lookup_object(self, index: MetadataIndex) -> Any:
        from .utils import find_state_dict_object

        return find_state_dict_object(self.state_dict, index)

    def transform_object(self, write_item: Any, object: Any) -> Any:
        if write_item.type is WriteItemType.BYTE_IO:
            stream = io.BytesIO()
            pickle.dump(object, stream, pickle.HIGHEST_PROTOCOL)
            stream.seek(0)
            return stream
        if hasattr(object, "to_local") and callable(object.to_local):
            return object.to_local()
        return object

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
        _init_state_dict(state_dict)
        self.original_state_dict = state_dict
        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)
        if self.flatten_state_dict:
            self.state_dict, self.mappings = flatten_state_dict(state_dict)
        else:
            self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        if self.metadata is None:
            raise ValueError("metadata is required")
        self.plan = _planner_create_default_local_load_plan(
            self.state_dict, self.metadata, not self.allow_partial_load
        )
        return self.plan

    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        return _planner_create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def load_bytes(self, read_item: Any, value: io.BytesIO) -> None:
        value.seek(0)
        loaded = pickle.load(value)
        key = read_item.dest_index.fqn
        if self.flatten_state_dict and key in self.mappings:
            set_element(self.original_state_dict, self.mappings[key], loaded)
        else:
            self.state_dict[key] = loaded

    def lookup_tensor(self, index: MetadataIndex) -> tp.Tensor:
        value = self.state_dict[index.fqn]
        if not hasattr(value, "__get_tensor_shard__"):
            from .utils import find_state_dict_object

            value = find_state_dict_object(self.state_dict, index)
        if hasattr(value, "__get_tensor_shard__"):
            value = value.__get_tensor_shard__(index)
        elif hasattr(value, "to_local") and callable(value.to_local):
            value = value.to_local()
        if not isinstance(value, tp.Tensor):
            raise TypeError(f"{index.fqn} is not a tensor")
        return value

    def resolve_tensor(self, read_item: Any) -> tp.Tensor:
        tensor = _narrow(self.lookup_tensor(read_item.dest_index), read_item.dest_offsets, read_item.lengths)
        return self.transform_tensor(read_item, tensor)

    def transform_tensor(self, read_item: Any, tensor: tp.Tensor) -> tp.Tensor:
        return tensor

    def commit_tensor(self, read_item: Any, tensor: tp.Tensor) -> None:
        del read_item, tensor


class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    def __init__(self, keys: set[str] | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.keys = keys

    def _should_include_key(self, key: str, metadata: Metadata) -> bool:
        if self.keys is None or key in self.keys:
            return True
        planner_data = metadata.planner_data or {}
        path = planner_data.get(key)
        if path is None:
            return False
        names: list[str] = []
        for part in path:
            names.append(str(part) if not names else f"{names[-1]}.{part}")
        return any(name in self.keys for name in names)

    def set_up_planner(self, state_dict: dict[str, Any], metadata: Metadata | None = None, is_coordinator: bool = False) -> None:
        if state_dict:
            raise ValueError("empty planner requires an empty state dictionary")
        if metadata is None:
            raise ValueError("metadata is required")
        for key, value in metadata.state_dict_metadata.items():
            if self.keys is not None and key not in self.keys:
                if not self._should_include_key(key, metadata):
                    continue
            if isinstance(value, TensorStorageMetadata):
                state_dict[key] = tp.empty(value.size, dtype=value.properties.dtype)
            else:
                state_dict[key] = None
            if metadata.planner_data is not None and key in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[key], state_dict.pop(key))
        super().set_up_planner(state_dict, metadata, is_coordinator)


def create_default_local_load_plan(state_dict: dict[str, Any], metadata: Metadata, strict: bool = True) -> LoadPlan:
    return _planner_create_default_local_load_plan(state_dict, metadata, strict)


def create_default_global_load_plan(all_plans: list[LoadPlan]) -> list[LoadPlan]:
    return _planner_create_default_global_load_plan(all_plans)


def create_default_local_save_plan(state_dict: dict[str, Any], is_coordinator: bool = True) -> SavePlan:
    return _planner_create_default_local_save_plan(state_dict, is_coordinator)


def create_default_global_save_plan(
    all_plans: list[SavePlan], rewrite_index_hints: bool = True
) -> tuple[list[SavePlan], Metadata]:
    return _planner_create_default_global_save_plan(all_plans, rewrite_index_hints)


def _create_default_local_metadata(state_dict: dict[str, Any]) -> Metadata:
    return _planner_create_default_local_metadata(state_dict)


def _check_box_overlap(
    box0: Any, box1: Any
) -> bool:
    for dimension in range(len(box0.offsets)):
        if box0.offsets[dimension] >= box1.offsets[dimension] + box1.sizes[dimension]:
            return False
        if box1.offsets[dimension] >= box0.offsets[dimension] + box0.sizes[dimension]:
            return False
    return True


def _check_box_bounds(outer_box_size: tuple[int, ...], inner_box: Any) -> bool:
    for dimension in range(len(outer_box_size)):
        if inner_box.offsets[dimension] < 0:
            return False
        if inner_box.sizes[dimension] < 0:
            return False
        if inner_box.offsets[dimension] + inner_box.sizes[dimension] > outer_box_size[dimension]:
            return False
    return True


def _validate_global_plan(global_plan: list[SavePlan], metadata: Metadata) -> list[str]:
    errors: list[str] = []
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        chunks = value.chunks
        chunks_volume = 0
        for chunk in chunks:
            if not _check_box_bounds(value.size, chunk):
                message = (
                    f"key:{key} has out of bounds chunk: "
                    f"tensor-size:{value.size} chunk: {chunk}"
                )
                logger.warning(message)
                errors.append(message)
            chunks_volume += math.prod(chunk.sizes)

        if len(chunks) > 1:
            dimensions = len(value.size)
            sweep_dimension = max(
                range(dimensions), default=0, key=lambda dimension: value.size[dimension]
            )
            sorted_indices = sorted(
                range(len(chunks)),
                key=lambda index: (
                    chunks[index].offsets[sweep_dimension],
                    *(chunks[index].offsets[dimension] for dimension in range(dimensions)),
                ),
            )
            active: list[tuple[int, int]] = []
            for index in sorted_indices:
                current = chunks[index]
                start = current.offsets[sweep_dimension]
                end = start + current.sizes[sweep_dimension]
                cutoff = bisect_right(active, (start, sys.maxsize))
                if cutoff:
                    del active[:cutoff]
                for _, other_index in active:
                    other = chunks[other_index]
                    if _check_box_overlap(current, other):
                        message = f"key:{key} has overlapping chunks: {current} {other}"
                        logger.warning(message)
                        errors.append(message)
                insort(active, (end, index))

        tensor_volume = math.prod(value.size)
        if len(global_plan) > 1 and chunks_volume != tensor_volume:
            message = (
                f"key:{key} invalid fill tensor-volume: "
                f"{tensor_volume} chunks-volume: {chunks_volume}"
            )
            logger.warning(message)
            errors.append(message)
    return errors
