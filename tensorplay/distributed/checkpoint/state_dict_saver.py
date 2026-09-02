from __future__ import annotations

import copy
import inspect
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import tensorplay as tp

import tensorplay.distributed as dist

from .api import CheckpointException
from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from ._traverse import traverse_state_dict
from .default_planner import DefaultSavePlanner

__all__ = ["save", "async_save"]

_ASYNC_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tp-checkpoint")
_ASYNC_EXECUTOR_LOCK = threading.Lock()


def _default_writer(checkpoint_id):
    from .filesystem import FileSystemWriter

    return FileSystemWriter(checkpoint_id)


def _is_distributed_tensor(value: Any) -> bool:
    return (
        hasattr(value, "device_mesh")
        and hasattr(value, "placements")
        and hasattr(value, "to_local")
        and callable(value.to_local)
    )


def _copy_state_value(value: Any, memo: dict[int, Any]) -> Any:
    if _is_distributed_tensor(value):
        copied = value.detach().clone()
        memo[id(value)] = copied
        return copied
    if isinstance(value, tp.Tensor):
        copied = value.detach().clone()
        memo[id(value)] = copied
        return copied
    if isinstance(value, dict):
        copied = {key: _copy_state_value(child, memo) for key, child in value.items()}
        memo[id(value)] = copied
        return copied
    if isinstance(value, list):
        copied = [_copy_state_value(child, memo) for child in value]
        memo[id(value)] = copied
        return copied
    if isinstance(value, tuple):
        copied = tuple(_copy_state_value(child, memo) for child in value)
        memo[id(value)] = copied
        return copied
    state_dict = getattr(value, "state_dict", None)
    if callable(state_dict):
        try:
            return _copy_state_value(state_dict(), memo)
        except (AttributeError, TypeError):
            pass
    return copy.deepcopy(value, memo)


def _snapshot_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dictionary")
    return _copy_state_value(state_dict, {})


def _merge_values(values: list[Any]) -> Any:
    if not values:
        return None
    if all(_is_distributed_tensor(value) for value in values):
        first = values[0]
        shape = tuple(int(size) for size in first.shape)
        local = first.to_local()
        assembled = tp.zeros(shape, dtype=local.dtype, device=local.device)
        for value in values:
            if tuple(value.shape) != shape:
                raise RuntimeError("distributed checkpoint tensors have different shapes")
            current = value.to_local()
            for chunk in value.__create_chunk_list__():
                destination = assembled
                for dim, (offset, size) in enumerate(zip(chunk.offsets, chunk.sizes)):
                    destination = destination.narrow(dim, int(offset), int(size))
                if int(current.numel()) == int(destination.numel()):
                    destination.copy_(current.reshape(destination.shape))
        return assembled
    if all(isinstance(value, dict) for value in values):
        keys = []
        seen = set()
        for value in values:
            for key in value:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return {
            key: _merge_values([value[key] for value in values if key in value])
            for key in keys
        }
    if all(isinstance(value, tp.Tensor) for value in values):
        first = values[0]
        if all(
            value.shape == first.shape
            and value.dtype == first.dtype
            and value.tolist() == first.tolist()
            for value in values[1:]
        ):
            return first
        if all(value.dim() == first.dim() for value in values) and first.dim() > 0:
            compatible = all(
                value.shape[1:] == first.shape[1:] for value in values
            )
            if compatible:
                return tp.cat(tuple(values), dim=0)
        return first
    first = values[0]
    try:
        if all(value == first for value in values[1:]):
            return first
    except Exception:
        pass
    return first


def _merge_rank_states(states: list[dict[str, Any]]) -> dict[str, Any]:
    if not states:
        return {}
    keys = []
    seen = set()
    for state in states:
        for key in state:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return {
        key: _merge_values([state[key] for state in states if key in state])
        for key in keys
    }


def _metadata_from_state_dict(
    state_dict: dict[str, Any], checkpoint_id: Any = None
) -> Metadata:
    state_dict_metadata: dict[str, Any] = {}

    def visit(path: tuple[Any, ...], value: Any) -> None:
        name = ".".join(str(part) for part in path)
        distributed_tensor = _is_distributed_tensor(value)
        if isinstance(value, tp.Tensor) or distributed_tensor:
            shape = tuple(int(size) for size in value.shape)
            local = value.to_local() if distributed_tensor else value
            chunks = (
                list(value.__create_chunk_list__())
                if distributed_tensor
                else [ChunkStorageMetadata((0,) * len(shape), shape)]
            )
            state_dict_metadata[name] = TensorStorageMetadata(
                properties=TensorProperties.create_from_tensor(local),
                size=shape,
                chunks=chunks,
            )
        else:
            state_dict_metadata[name] = BytesStorageMetadata()

    traverse_state_dict(state_dict, visit)
    return Metadata(
        state_dict_metadata=state_dict_metadata,
        storage_meta=StorageMeta(checkpoint_id=checkpoint_id),
        version="tp-1",
    )


def _make_metadata(
    state_dict: dict[str, Any], checkpoint_id: Any, planner: Any
) -> Metadata:
    metadata = _metadata_from_state_dict(state_dict, checkpoint_id)
    if planner is None:
        return metadata
    planner.set_up_planner(
        state_dict,
        storage_meta=metadata.storage_meta,
        is_coordinator=True,
    )
    local_plan = planner.create_local_plan()
    global_result = planner.create_global_plan([local_plan])
    if not isinstance(global_result, tuple) or len(global_result) != 2:
        raise TypeError("save planner must return (plans, metadata)")
    planned_metadata = global_result[1]
    if not isinstance(planned_metadata, Metadata):
        raise TypeError("save planner must return Metadata")
    if not planned_metadata.state_dict_metadata:
        planned_metadata = replace(
            planned_metadata, state_dict_metadata=metadata.state_dict_metadata
        )
    if planned_metadata.storage_meta is None:
        planned_metadata = replace(
            planned_metadata, storage_meta=metadata.storage_meta
        )
    if planned_metadata.version is None:
        planned_metadata = replace(planned_metadata, version="tp-1")
    return planned_metadata


def _writer_for(checkpoint_id: Any, storage_writer: Any) -> Any:
    if storage_writer is None:
        if checkpoint_id is None:
            raise ValueError("must specify either checkpoint_id or storage_writer")
        storage_writer = _default_writer(checkpoint_id)
    elif checkpoint_id is not None:
        reset = getattr(storage_writer, "reset", None)
        if callable(reset):
            reset(checkpoint_id)
    return storage_writer


def _write_snapshot(writer: Any, snapshot: dict[str, Any], planner: Any) -> None:
    write_data = writer.write_data
    if planner is None:
        write_data(snapshot)
        return
    parameters = inspect.signature(write_data).parameters
    if "planner" in parameters:
        write_data(snapshot, planner=planner)
        return
    write_data(snapshot)


def _raise_failures(message: str, statuses: list[dict[str, Any]]) -> None:
    failures = {
        rank: RuntimeError(str(status.get("error", "unknown checkpoint failure")))
        for rank, status in enumerate(statuses)
        if not status.get("ok", False)
    }
    if failures:
        raise CheckpointException(message, failures)


def _save_without_distribution(
    state_dict: dict[str, Any],
    checkpoint_id: Any,
    storage_writer: Any,
    planner: Any,
) -> Metadata | Any:
    writer = _writer_for(checkpoint_id, storage_writer)
    snapshot = _snapshot_state_dict(state_dict)
    planner = planner or DefaultSavePlanner()
    writer.set_up_storage_writer(is_coordinator=True)
    metadata = _make_metadata(snapshot, checkpoint_id, planner)
    _write_snapshot(writer, snapshot, planner)
    return writer.finish(metadata)


def save(
    state_dict,
    *,
    checkpoint_id=None,
    storage_writer=None,
    planner=None,
    process_group=None,
    no_dist=False,
) -> Any:
    """Save a state dictionary with an atomic metadata commit."""
    if no_dist or not dist.is_initialized():
        return _save_without_distribution(
            state_dict, checkpoint_id, storage_writer, planner
        )

    pg = process_group or dist._get_default_group()
    rank = dist.get_rank(pg)
    is_coordinator = rank == 0
    writer = None
    local_snapshot = None
    local_error: str | None = None
    try:
        writer = _writer_for(checkpoint_id, storage_writer)
        writer.set_up_storage_writer(is_coordinator=is_coordinator)
        local_snapshot = _snapshot_state_dict(state_dict)
    except BaseException as error:
        local_error = f"{type(error).__name__}: {error}"

    statuses = [None] * dist.get_world_size(pg)
    dist.all_gather_object(
        statuses,
        {"ok": local_error is None, "error": local_error},
        group=pg,
    )
    _raise_failures("checkpoint preparation failed", statuses)

    states = [None] * dist.get_world_size(pg)
    dist.all_gather_object(states, local_snapshot, group=pg)
    merged = _merge_rank_states(states)

    commit = [{"ok": None, "error": None}]
    metadata = None
    if is_coordinator:
        try:
            planner = planner or DefaultSavePlanner()
            metadata = _make_metadata(merged, checkpoint_id, planner)
            _write_snapshot(writer, merged, planner)
            metadata = writer.finish(metadata)
            commit[0] = {"ok": True, "error": None}
        except BaseException as error:
            commit[0] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    dist.broadcast_object_list(commit, src=0, group=pg)
    if not commit[0].get("ok", False):
        failures = {0: RuntimeError(str(commit[0].get("error", "checkpoint commit failed")))}
        raise CheckpointException("checkpoint commit failed", failures)
    dist.barrier(pg)
    return metadata if is_coordinator else None


def async_save(
    state_dict,
    *,
    checkpoint_id=None,
    storage_writer=None,
    planner=None,
    process_group=None,
    no_dist=False,
) -> Future[Any]:
    """Stage the input immediately and perform the save in a worker thread."""
    try:
        staged = _snapshot_state_dict(state_dict)
    except BaseException as error:
        future: Future[Any] = Future()
        future.set_exception(error)
        return future
    with _ASYNC_EXECUTOR_LOCK:
        return _ASYNC_EXECUTOR.submit(
            save,
            staged,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            process_group=process_group,
            no_dist=no_dist,
        )
