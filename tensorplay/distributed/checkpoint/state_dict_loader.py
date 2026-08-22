# Ported from torch/distributed/checkpoint/state_dict_loader.py.
from typing import Any

import tensorplay as tp

import tensorplay.distributed as dist


__all__ = ["load"]


def _default_reader(checkpoint_id):
    from .filesystem import FileSystemReader

    return FileSystemReader(checkpoint_id)


def load(state_dict, *, checkpoint_id=None, storage_reader=None,
         planner=None, process_group=None, no_dist=False) -> None:
    """Load a distributed checkpoint in place (torch parity).

    Args:
        state_dict: the (possibly partially empty) state dict to load into;
            keys present here are filled from the checkpoint. Tensors are
            copied in place when shapes match.
        checkpoint_id: directory path of the checkpoint.
        storage_reader: optional StorageReader; defaults to FileSystemReader.
    """
    if no_dist or not dist.is_initialized():
        if checkpoint_id is None and storage_reader is None:
            raise ValueError("Must specify either checkpoint_id or storage_reader")
        reader = storage_reader or _default_reader(checkpoint_id)
        saved = reader.read_data(None, state_dict)
        _fill_in_place(state_dict, saved)
        return

    pg = process_group or dist._get_default_group()
    if storage_reader is None:
        if checkpoint_id is None:
            raise ValueError("Must specify either checkpoint_id or storage_reader")
        reader = _default_reader(checkpoint_id)
    else:
        reader = storage_reader
    reader.read_metadata()
    saved = reader.read_data(None, state_dict)

    # Broadcast the loaded values so every rank receives the full state.
    from tensorplay.distributed.optim.zero_redundancy_optimizer import (
        _broadcast_object,
    )

    coordinator_global = dist.get_global_rank(pg, 0)
    if dist.get_rank(pg) != 0:
        saved = {}
    saved = _broadcast_object(saved, src_rank=coordinator_global, group=pg,
                              device="cpu")
    _fill_in_place(state_dict, saved)


def _fill_in_place(state_dict, saved) -> None:
    """Copy loaded values into ``state_dict`` in place where possible."""
    for key, value in state_dict.items():
        if key not in saved:
            continue
        loaded = saved[key]
        if isinstance(value, tp.Tensor) and isinstance(loaded, tp.Tensor):
            if value.shape == loaded.shape:
                value.copy_(loaded.to(value.device))
            else:
                raise RuntimeError(
                    f"Shape mismatch for {key}: "
                    f"expected {tuple(value.shape)}, got {tuple(loaded.shape)}"
                )
        elif isinstance(value, dict) and isinstance(loaded, dict):
            _fill_in_place(value, loaded)
        else:
            state_dict[key] = loaded
