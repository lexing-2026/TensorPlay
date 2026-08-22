# Ported from torch/distributed/checkpoint/state_dict_saver.py.
#
# Adaptation: tp's DCP consolidates each rank's local state on the
# coordinator rank via object collectives, then writes through the storage
# writer. Torch's sharded-planner write path (parallel per-shard files) is
# tracked in docs/gap_analysis.md.
import warnings
from typing import Any

import tensorplay as tp

import tensorplay.distributed as dist


__all__ = ["save", "async_save"]


def _default_writer(checkpoint_id):
    from .filesystem import FileSystemWriter

    return FileSystemWriter(checkpoint_id)


def _default_reader(checkpoint_id):
    from .filesystem import FileSystemReader

    return FileSystemReader(checkpoint_id)


def _exchange_local_states(state_dict, pg) -> list[Any] | None:
    """Gather per-rank local states onto the coordinator via object broadcast."""
    from tensorplay.distributed.optim.zero_redundancy_optimizer import (
        _broadcast_object,
    )

    world_size = dist.get_world_size(pg)
    coordinator = 0
    my_rank = dist.get_rank(pg)
    is_coordinator = my_rank == coordinator
    empty = tp.tensor([0], dtype=tp.uint8, device="cpu")
    all_states = []
    for rank in range(world_size):
        global_rank = dist.get_global_rank(pg, rank)
        if is_coordinator:
            if rank == my_rank:
                all_states.append(state_dict)
            else:
                all_states.append(_broadcast_object(
                    empty, src_rank=global_rank, group=pg, device="cpu"))
        else:
            if rank == my_rank:
                _broadcast_object(state_dict, src_rank=my_rank, group=pg,
                                  device="cpu")
            elif rank != coordinator:
                _ = _broadcast_object(empty, src_rank=global_rank, group=pg,
                                      device="cpu")
    return all_states if is_coordinator else None


def save(state_dict, *, checkpoint_id=None, storage_writer=None,
         planner=None, process_group=None, no_dist=False) -> Any:
    """Save a distributed state dict (torch ``dist.checkpoint.save`` parity).

    Args:
        state_dict: the state dict to save.
        checkpoint_id: directory path for the checkpoint.
        storage_writer: optional StorageWriter; defaults to FileSystemWriter
            when ``checkpoint_id`` is provided.
        no_dist: save without any distributed coordination.

    Returns:
        Metadata produced by the writer, or None on non-coordinator ranks.
    """
    if no_dist or not dist.is_initialized():
        if checkpoint_id is None and storage_writer is None:
            raise ValueError("Must specify either checkpoint_id or storage_writer")
        writer = storage_writer or _default_writer(checkpoint_id)
        writer.set_up_storage_writer(is_coordinator=True)
        writer.write_data(state_dict)
        return writer.finish({"version": "tp-1"})

    pg = process_group or dist._get_default_group()
    is_coordinator = dist.get_rank(pg) == 0
    if storage_writer is None:
        if checkpoint_id is None:
            raise ValueError("Must specify either checkpoint_id or storage_writer")
        storage_writer = _default_writer(checkpoint_id)

    storage_writer.set_up_storage_writer(is_coordinator=is_coordinator)

    all_states = _exchange_local_states(state_dict, pg)

    metadata = {"version": "tp-1"}
    if is_coordinator:
        merged: dict[str, Any] = {}
        for rank_state in all_states:
            for key, value in rank_state.items():
                # SPMD replicas of the same key are redundant; keys that are
                # intentionally sharded accumulate across ranks.
                if key not in merged:
                    merged[key] = value
                elif isinstance(merged[key], list):
                    merged[key].append(value)
                elif key.endswith("_shard"):
                    merged[key] = [merged[key], value]
                else:
                    merged[key] = value  # keep coordinator's replica
        storage_writer.write_data(merged)
        metadata = storage_writer.finish(metadata)
    dist.barrier(pg)
    return metadata if is_coordinator else None


def async_save(state_dict, *, checkpoint_id=None, storage_writer=None,
               process_group=None) -> Any:
    """Async save (torch parity); tp currently performs it synchronously."""
    warnings.warn(
        "tp DCP async_save currently performs a synchronous save.",
        stacklevel=2,
    )
    return save(state_dict, checkpoint_id=checkpoint_id,
                storage_writer=storage_writer, process_group=process_group)
