from __future__ import annotations

import copy
import inspect
from typing import Any

import tensorplay as tp

import tensorplay.distributed as dist

from .api import CheckpointException
from .default_planner import DefaultLoadPlanner
from .metadata import Metadata

__all__ = ["load"]


def _default_reader(checkpoint_id):
    from .filesystem import FileSystemReader

    return FileSystemReader(checkpoint_id)


def _reader_for(checkpoint_id: Any, storage_reader: Any) -> Any:
    if storage_reader is None:
        if checkpoint_id is None:
            raise ValueError("must specify either checkpoint_id or storage_reader")
        return _default_reader(checkpoint_id)
    if checkpoint_id is not None:
        reset = getattr(storage_reader, "reset", None)
        if callable(reset):
            reset(checkpoint_id)
    return storage_reader


def _read_snapshot(reader: Any, state_dict: dict[str, Any], planner: Any) -> dict[str, Any]:
    metadata = reader.read_metadata()
    if not isinstance(metadata, Metadata):
        loaded = reader.read_data(metadata, state_dict)
        return loaded
    planner.set_up_planner(state_dict, metadata, is_coordinator=True)
    plan = planner.create_local_plan()
    prepare = getattr(reader, "prepare_local_plan", None)
    if callable(prepare):
        plan = prepare(plan)
    read_data = reader.read_data
    parameters = list(inspect.signature(read_data).parameters.values())
    if len(parameters) >= 3 or (
        len(parameters) >= 2 and parameters[1].name in {"state_dict", "destination"}
    ):
        result = read_data(plan, state_dict, planner)
    else:
        result = read_data(plan, planner)
    if hasattr(result, "result"):
        result = result.result()
    if isinstance(result, dict):
        return result
    return state_dict


def _raise_load_failures(message: str, statuses: list[Any]) -> None:
    failures = {
        rank: RuntimeError(str(status.get("error", "unknown checkpoint failure")))
        for rank, status in enumerate(statuses)
        if not isinstance(status, dict) or not status.get("ok", False)
    }
    if failures:
        raise CheckpointException(message, failures)


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
    if no_dist or not dist.is_initialized():
        reader = _reader_for(checkpoint_id, storage_reader)
        metadata = reader.read_metadata()
        if planner is None and isinstance(metadata, Metadata):
            planner = DefaultLoadPlanner()
        if planner is not None and isinstance(metadata, Metadata):
            saved = _read_snapshot(reader, state_dict, planner)
        else:
            saved = reader.read_data(metadata, state_dict)
        _fill_in_place(state_dict, saved)
        return

    pg = process_group or dist._get_default_group()
    local_error: str | None = None
    saved: dict[str, Any] | None = None
    try:
        reader = _reader_for(checkpoint_id, storage_reader)
        metadata = reader.read_metadata()
        local_planner = planner
        if local_planner is None and isinstance(metadata, Metadata):
            local_planner = DefaultLoadPlanner()
        if local_planner is not None and isinstance(metadata, Metadata):
            saved = _read_snapshot(reader, state_dict, local_planner)
        else:
            saved = reader.read_data(metadata, state_dict)
        if not isinstance(saved, dict):
            raise RuntimeError("checkpoint data must contain a dictionary")
    except BaseException as error:
        local_error = f"{type(error).__name__}: {error}"

    statuses = [None] * dist.get_world_size(pg)
    dist.all_gather_object(
        statuses,
        {"ok": local_error is None, "error": local_error},
        group=pg,
    )
    _raise_load_failures("checkpoint load failed", statuses)
    if saved is None:
        raise RuntimeError("checkpoint load produced no state")
    _fill_in_place(state_dict, saved)
    dist.barrier(pg)


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
