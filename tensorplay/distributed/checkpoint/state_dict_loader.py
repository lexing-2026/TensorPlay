from __future__ import annotations

import copy
import inspect
from typing import Any

import tensorplay as tp

import tensorplay.distributed as dist

from ._storage_utils import _storage_setup
from .default_planner import DefaultLoadPlanner, _EmptyStateDictLoadPlanner
from .metadata import Metadata
from .planner import LoadPlan
from .state_dict_saver import _snapshot_state_dict
from .utils import _DistWrapper

__all__ = ["load_state_dict", "load"]


def _is_stateful(value: Any) -> bool:
    return callable(getattr(value, "state_dict", None)) and callable(
        getattr(value, "load_state_dict", None)
    )


def _restore_state_dict(
    state_dict: dict[str, Any], snapshot: dict[str, Any]
) -> None:
    for key in tuple(state_dict):
        if key not in snapshot:
            del state_dict[key]
    _fill_in_place(state_dict, snapshot)


def _load_state_dict(
    state_dict: dict[str, Any],
    storage_reader: Any,
    process_group: Any = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Any = None,
) -> None:
    dist_wrapper = _DistWrapper(process_group, not no_dist, coordinator_rank)
    planner = planner or DefaultLoadPlanner()
    rollback_state: dict[str, Any] | None = None
    metadata: Metadata | None = None
    use_collectives = True

    def local_step() -> LoadPlan:
        nonlocal rollback_state, metadata, use_collectives
        rollback_state = _snapshot_state_dict(state_dict)
        try:
            metadata = storage_reader.read_metadata()
        except BaseException as global_error:
            try:
                parameters = inspect.signature(storage_reader.read_metadata).parameters
                if not (
                    "rank" in parameters
                    or any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                ):
                    raise global_error
                metadata = storage_reader.read_metadata(rank=dist_wrapper.rank)
                use_collectives = False
            except BaseException:
                raise global_error
        if not isinstance(metadata, Metadata):
            raise TypeError("checkpoint metadata must be a Metadata object")
        planner.set_up_planner(
            state_dict,
            metadata,
            is_coordinator=dist_wrapper.is_coordinator,
        )
        storage_reader.set_up_storage_reader(
            metadata,
            dist_wrapper.is_coordinator,
            rank=dist_wrapper.rank,
            use_collectives=use_collectives,
        )
        local_plan = planner.create_local_plan()
        return storage_reader.prepare_local_plan(local_plan)

    def global_step(all_local_plans: list[LoadPlan]) -> list[LoadPlan]:
        all_local_plans = planner.create_global_plan(all_local_plans)
        return storage_reader.prepare_global_plan(all_local_plans)

    try:
        if use_collectives:
            central_plan = dist_wrapper.reduce_scatter(
                "checkpoint plan", local_step, global_step
            )
        else:
            central_plan = global_step([local_step()])[0]

        def read_data() -> None:
            final_local_plan = planner.finish_plan(central_plan)
            reads = storage_reader.read_data(final_local_plan, planner)
            reads.result()

        if use_collectives:
            dist_wrapper.all_gather("checkpoint read", read_data)
        else:
            read_data()
        dist_wrapper.barrier()
    except BaseException:
        if rollback_state is not None:
            try:
                _restore_state_dict(state_dict, rollback_state)
            except BaseException:
                pass
        raise


def load(
    state_dict,
    *,
    checkpoint_id=None,
    storage_reader=None,
    planner=None,
    process_group=None,
    no_dist=False,
) -> None:
    """Load checkpoint values into an existing state dictionary."""
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dictionary")
    keys = sorted(state_dict)
    stateful_state_dict: dict[str, Any] = {}
    for key in keys:
        value = state_dict[key]
        stateful_state_dict[key] = (
            value.state_dict() if _is_stateful(value) else value
        )
    working_state_dict = stateful_state_dict
    reader = _storage_setup(storage_reader, checkpoint_id, reader=True)
    _load_state_dict(
        working_state_dict,
        reader,
        process_group=process_group,
        no_dist=no_dist or not dist.is_initialized(),
        planner=planner,
    )
    for key in keys:
        value = state_dict[key]
        loaded = working_state_dict[key]
        if _is_stateful(value):
            value.load_state_dict(loaded)
        else:
            state_dict[key] = loaded


def load_state_dict(
    state_dict: dict[str, Any],
    storage_reader: Any,
    process_group: Any = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Any = None,
) -> None:
    storage_reader.reset()
    _load_state_dict(
        state_dict,
        storage_reader,
        process_group=process_group,
        coordinator_rank=coordinator_rank,
        no_dist=no_dist or not dist.is_initialized(),
        planner=planner,
    )


def _load_state_dict_from_keys(
    keys: set[str] | str | None = None,
    *,
    checkpoint_id: str | Any | None = None,
    storage_reader: Any = None,
    process_group: Any = None,
) -> dict[str, Any]:
    if isinstance(keys, str):
        keys = {keys}
    reader = _storage_setup(storage_reader, checkpoint_id, reader=True)
    planner = _EmptyStateDictLoadPlanner(keys=keys)
    result: dict[str, Any] = {}
    _load_state_dict(
        result,
        reader,
        process_group=process_group,
        no_dist=not dist.is_initialized(),
        planner=planner,
    )
    return result


def _fill_in_place(state_dict: dict[str, Any], saved: dict[str, Any]) -> None:
    if not isinstance(saved, dict):
        raise TypeError("saved state must be a dictionary")
    for key, loaded in saved.items():
        if key not in state_dict:
            state_dict[key] = copy.deepcopy(loaded)
            continue
        current = state_dict[key]
        current_to_local = getattr(current, "to_local", None)
        if callable(current_to_local) and hasattr(current, "device_mesh"):
            loaded_to_local = getattr(loaded, "to_local", None)
            loaded_value = loaded_to_local() if callable(loaded_to_local) else loaded
            if not isinstance(loaded_value, tp.Tensor):
                raise TypeError(f"loaded value for {key} is not a tensor")
            if tuple(current.shape) != tuple(loaded.shape):
                raise RuntimeError(
                    f"shape mismatch for {key}: expected {tuple(current.shape)}, "
                    f"got {tuple(loaded.shape)}"
                )
            current_to_local().copy_(loaded_value.to(current_to_local().device))
            continue
        if isinstance(current, tp.Tensor) and isinstance(loaded, tp.Tensor):
            if current.shape != loaded.shape:
                raise RuntimeError(
                    f"shape mismatch for {key}: expected {tuple(current.shape)}, "
                    f"got {tuple(loaded.shape)}"
                )
            current.copy_(loaded.to(current.device))
        elif isinstance(current, dict) and isinstance(loaded, dict):
            _fill_in_place(current, loaded)
        elif isinstance(current, list) and isinstance(loaded, list):
            if len(current) != len(loaded):
                state_dict[key] = copy.deepcopy(loaded)
            else:
                for index, value in enumerate(loaded):
                    if isinstance(current[index], dict) and isinstance(value, dict):
                        _fill_in_place(current[index], value)
                    elif isinstance(current[index], tp.Tensor) and isinstance(value, tp.Tensor):
                        if current[index].shape != value.shape:
                            raise RuntimeError(
                                f"shape mismatch for {key}[{index}]"
                            )
                        current[index].copy_(value.to(current[index].device))
                    else:
                        current[index] = copy.deepcopy(value)
        else:
            load_state_dict = getattr(current, "load_state_dict", None)
            if callable(load_state_dict) and isinstance(loaded, dict):
                load_state_dict(loaded)
            else:
                state_dict[key] = copy.deepcopy(loaded)
