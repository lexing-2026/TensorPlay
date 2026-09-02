from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .barriers import create_barrier_from_config
from .checkpoint_process import CheckpointProcess
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter, CheckpointWriterConfig, WriterHook
from .checkpointer import AsyncCheckpointer, SyncCheckpointer
from .config import CheckpointerConfig
from .staging import DefaultStager
from .types import RankInfo


def _get_default_rank_info() -> RankInfo:
    try:
        from ... import distributed_core as dist
        if dist.is_initialized():
            return RankInfo(dist.get_rank(), dist.get_world_size())
    except Exception:
        pass
    return RankInfo(0, 1)


def default_subprocess_init_fn(*_: Any) -> None:
    return None


def default_writer_init_fn(
    rank_info: RankInfo,
    config: CheckpointWriterConfig | None = None,
    barrier_config: Any = None,
    commit_hook: WriterHook | None = None,
) -> CheckpointWriter:
    barrier = (
        create_barrier_from_config(barrier_config)
        if barrier_config is not None
        else None
    )
    return CheckpointWriter(config or CheckpointWriterConfig(), rank_info, barrier, commit_hook)


def make_sync_checkpointer(config: CheckpointerConfig = CheckpointerConfig(), rank_info: RankInfo | None = None, commit_hook: WriterHook | None = None) -> SyncCheckpointer:
    info = rank_info or _get_default_rank_info()
    barrier = create_barrier_from_config(config.barrier_config)
    writer = CheckpointWriter(config.writer_config, info, barrier, commit_hook)
    return SyncCheckpointer(writer, CheckpointReader(info))


def make_async_checkpointer(config: CheckpointerConfig = CheckpointerConfig(), rank_info: RankInfo | None = None, subprocess_init_fn: Callable[..., None] = default_subprocess_init_fn, subprocess_init_args: tuple[Any, ...] = (), checkpoint_writer_init_fn: Callable[..., CheckpointWriter] = default_writer_init_fn, checkpoint_writer_init_args: dict[str, Any] | None = None) -> AsyncCheckpointer:
    info = rank_info or _get_default_rank_info()
    writer_args = dict(checkpoint_writer_init_args or {})
    if checkpoint_writer_init_fn is default_writer_init_fn:
        writer_args.setdefault("config", config.writer_config)
        writer_args.setdefault("barrier_config", config.barrier_config)
    process = CheckpointProcess(info, config.process_config, subprocess_init_fn, subprocess_init_args, checkpoint_writer_init_fn, writer_args)
    return AsyncCheckpointer(DefaultStager(config.staging_config), process, CheckpointReader(info))
