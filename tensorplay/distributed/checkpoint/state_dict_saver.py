from __future__ import annotations

import copy
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import Any

import tensorplay as tp

import tensorplay.distributed as dist

from .metadata import Metadata
from .default_planner import DefaultSavePlanner
from .planner import SavePlan
from .staging import AsyncStager, DefaultStager, StagingOptions
from ._storage_utils import _storage_setup
from .utils import _DistWrapper

__all__ = ["save_state_dict", "save", "async_save", "AsyncCheckpointerType", "AsyncSaveResponse"]

class AsyncCheckpointerType(Enum):
    THREAD = "thread"
    PROCESS = "process"


@dataclass
class AsyncSaveResponse:
    staging_completion: Future[None]
    upload_completion: Future[Any]


def _is_distributed_tensor(value: Any) -> bool:
    return (
        hasattr(value, "device_mesh")
        and hasattr(value, "placements")
        and hasattr(value, "to_local")
        and callable(value.to_local)
    )


def _is_sharded_tensor(value: Any) -> bool:
    return (
        callable(getattr(value, "local_shards", None))
        and callable(getattr(value, "metadata", None))
        and callable(
            getattr(type(value), "_init_from_local_shards_and_global_metadata", None)
        )
    )


def _copy_sharded_tensor(value: Any, memo: dict[int, Any]) -> Any:
    metadata = copy.copy(value.metadata())
    shard_metadata = []
    for item in getattr(metadata, "shards_metadata", ()):
        copied_item = copy.copy(item)
        if hasattr(item, "shard_offsets"):
            object.__setattr__(copied_item, "shard_offsets", list(item.shard_offsets))
        if hasattr(item, "shard_sizes"):
            object.__setattr__(copied_item, "shard_sizes", list(item.shard_sizes))
        shard_metadata.append(copied_item)
    if hasattr(metadata, "shards_metadata"):
        metadata.shards_metadata = shard_metadata
    if hasattr(metadata, "tensor_properties"):
        metadata.tensor_properties = copy.copy(metadata.tensor_properties)
    shards = []
    for shard in value.local_shards():
        copied_metadata = copy.copy(shard.metadata)
        if hasattr(shard.metadata, "shard_offsets"):
            object.__setattr__(
                copied_metadata,
                "shard_offsets",
                list(shard.metadata.shard_offsets),
            )
        if hasattr(shard.metadata, "shard_sizes"):
            object.__setattr__(
                copied_metadata,
                "shard_sizes",
                list(shard.metadata.shard_sizes),
            )
        copied_tensor = _copy_state_value(shard.tensor, memo)
        shards.append(type(shard)(copied_tensor, copied_metadata))
    copied = type(value)._init_from_local_shards_and_global_metadata(
        shards,
        metadata,
        getattr(value, "_sharding_spec", None),
        getattr(value, "_process_group", None),
    )
    memo[id(value)] = copied
    return copied


def _copy_state_value(value: Any, memo: dict[int, Any]) -> Any:
    cached = memo.get(id(value))
    if cached is not None:
        return cached
    if _is_distributed_tensor(value):
        copied = value.detach().clone()
        memo[id(value)] = copied
        return copied
    if _is_sharded_tensor(value):
        return _copy_sharded_tensor(value, memo)
    if isinstance(value, tp.Tensor):
        copied = value.detach().clone()
        memo[id(value)] = copied
        for name, attribute in getattr(value, "__dict__", {}).items():
            try:
                setattr(copied, name, _copy_state_value(attribute, memo))
            except (AttributeError, TypeError):
                continue
        return copied
    if isinstance(value, dict):
        copied: dict[Any, Any] = {}
        memo[id(value)] = copied
        for key, child in value.items():
            copied[_copy_state_value(key, memo)] = _copy_state_value(child, memo)
        return copied
    if isinstance(value, list):
        copied: list[Any] = []
        memo[id(value)] = copied
        copied.extend(_copy_state_value(child, memo) for child in value)
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


def _stateful_to_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dictionary")
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        state_fn = getattr(value, "state_dict", None)
        if callable(state_fn) and not isinstance(value, (tp.Tensor, dict, list, tuple)):
            result[key] = state_fn()
        else:
            result[key] = value
    return result


def _abort_writer(writer: Any) -> None:
    abort = getattr(writer, "abort", None)
    if callable(abort):
        try:
            abort()
        except BaseException:
            return


def _mark_writer_committed(writer: Any) -> None:
    mark_committed = getattr(writer, "mark_committed", None)
    if callable(mark_committed):
        mark_committed()


def _save_state_dict(
    state_dict: dict[str, Any],
    storage_writer: Any,
    process_group: Any = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Any = None,
    use_collectives: bool = True,
) -> Metadata | Any:
    dist_wrapper = _DistWrapper(process_group, not no_dist, coordinator_rank)
    planner = planner or DefaultSavePlanner()
    global_metadata: Metadata | None = None

    def local_step() -> SavePlan:
        planner.set_up_planner(
            state_dict=state_dict,
            storage_meta=storage_writer.storage_meta(),
            is_coordinator=dist_wrapper.is_coordinator,
        )
        storage_writer.set_up_storage_writer(
            dist_wrapper.is_coordinator,
            rank=dist_wrapper.rank,
            use_collectives=use_collectives,
        )
        local_plan = planner.create_local_plan()
        return storage_writer.prepare_local_plan(local_plan)

    def global_step(all_local_plans: list[SavePlan]) -> list[SavePlan]:
        nonlocal global_metadata
        global_result = planner.create_global_plan(all_local_plans)
        if not isinstance(global_result, tuple) or len(global_result) != 2:
            raise TypeError("save planner must return (plans, metadata)")
        all_local_plans, global_metadata = global_result
        return storage_writer.prepare_global_plan(all_local_plans)

    try:
        if use_collectives:
            central_plan = dist_wrapper.reduce_scatter("checkpoint plan", local_step, global_step)
        else:
            central_plan = global_step([local_step()])[0]

        def write_data() -> list[Any]:
            final_local_plan = planner.finish_plan(central_plan)
            return storage_writer.write_data(final_local_plan, planner).result()

        def finish_checkpoint(all_results: list[list[Any]]) -> Metadata:
            if global_metadata is None:
                raise RuntimeError("checkpoint metadata was not created")
            storage_writer.finish(metadata=global_metadata, results=all_results)
            return global_metadata

        if use_collectives:
            metadata = dist_wrapper.all_reduce(
                "checkpoint write", write_data, finish_checkpoint
            )
        else:
            metadata = finish_checkpoint([write_data()])
            dist_wrapper.barrier()
        _mark_writer_committed(storage_writer)
        return metadata
    except BaseException:
        _abort_writer(storage_writer)
        raise


def save(
    state_dict,
    *,
    checkpoint_id=None,
    storage_writer=None,
    planner=None,
    process_group=None,
    no_dist=False,
    use_collectives=True,
) -> Any:
    """Save a state dictionary with coordinated metadata commit."""
    no_dist = no_dist or not dist.is_initialized()
    writer = _storage_setup(storage_writer, checkpoint_id, reader=False)
    return _save_state_dict(
        _stateful_to_state_dict(state_dict),
        writer,
        process_group=process_group,
        planner=planner,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )


def save_state_dict(
    state_dict: dict[str, Any],
    storage_writer: Any,
    process_group: Any = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Any = None,
) -> Metadata | Any:
    storage_writer.reset()
    return _save_state_dict(
        state_dict,
        storage_writer,
        process_group=process_group,
        coordinator_rank=coordinator_rank,
        no_dist=no_dist or not dist.is_initialized(),
        planner=planner,
    )


def async_save(
    state_dict,
    *,
    checkpoint_id=None,
    storage_writer=None,
    planner=None,
    process_group=None,
    async_checkpointer_type: AsyncCheckpointerType = AsyncCheckpointerType.THREAD,
    async_stager: AsyncStager | None = None,
    no_dist=False,
    use_collectives=True,
) -> Future[Any] | AsyncSaveResponse:
    """Stage the input and execute the checkpoint write asynchronously."""
    from ._async_process_executor import _ProcessBasedAsyncCheckpointExecutor
    from ._async_thread_executor import _ThreadBasedAsyncCheckpointExecutor

    state_dict = _stateful_to_state_dict(state_dict)
    owned_stager = False
    if async_stager is None:
        if storage_writer is not None and isinstance(storage_writer, AsyncStager):
            async_stager = storage_writer
        else:
            async_stager = DefaultStager(
                StagingOptions(
                    use_pinned_memory=False,
                    use_shared_memory=False,
                    use_async_staging=False,
                    use_non_blocking_copy=False,
                )
            )
            owned_stager = True
    try:
        staged = async_stager.stage(state_dict)
    except BaseException as error:
        failure: Future[Any] = Future()
        failure.set_exception(error)
        if owned_stager:
            async_stager.close()
        return failure
    executor = (
        _ProcessBasedAsyncCheckpointExecutor()
        if async_checkpointer_type is AsyncCheckpointerType.PROCESS
        else _ThreadBasedAsyncCheckpointExecutor()
    )
    upload = executor.execute_save(
        staged,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )
    if owned_stager:
        upload.add_done_callback(lambda _: async_stager.close())
    if isinstance(staged, Future):
        staging_completion: Future[None] = Future()

        def complete(future: Future[Any]) -> None:
            try:
                future.result()
                staging_completion.set_result(None)
            except BaseException as error:
                staging_completion.set_exception(error)

        staged.add_done_callback(complete)
        return AsyncSaveResponse(staging_completion, upload)
    if bool(getattr(async_stager, "should_synchronize_after_execute", True)):
        async_stager.synchronize_staging()
    return upload
