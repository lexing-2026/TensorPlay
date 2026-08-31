from __future__ import annotations

import abc
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ... import distributed_core as dist

BARRIER_REGISTRY: dict[str, type["Barrier"]] = {}


def register_barrier(barrier_class: type["Barrier"]) -> type["Barrier"]:
    BARRIER_REGISTRY[barrier_class.barrier_type] = barrier_class
    return barrier_class


@dataclass
class BarrierConfig:
    barrier_type: str | None = None
    barrier_args: dict[str, Any] = field(default_factory=dict)


def create_barrier_from_config(barrier_config: BarrierConfig) -> "Barrier | None":
    if barrier_config.barrier_type is None:
        return None
    try:
        cls = BARRIER_REGISTRY[barrier_config.barrier_type]
    except KeyError as error:
        raise ValueError(f"unknown barrier type {barrier_config.barrier_type!r}") from error
    return cls(**barrier_config.barrier_args)


class Barrier(abc.ABC):
    @abc.abstractmethod
    def execute_barrier(self) -> None: ...


@register_barrier
class DistBarrier(Barrier):
    barrier_type = "dist_barrier"

    def __init__(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError("a process group is required for DistBarrier")

    def execute_barrier(self) -> None:
        dist.barrier()


@register_barrier
class TCPStoreBarrier(Barrier):
    barrier_type = "tcp_store"

    def __init__(self, global_rank: int, global_world_size: int, barrier_prefix: str = "checkpoint", timeout_barrier_init_secs: int = 30, use_checkpoint_barrier_tcpstore_libuv: bool = False, tcpstore_port: int = 0, master_address: str = "127.0.0.1", timeout_secs: int = 600) -> None:
        del timeout_barrier_init_secs, use_checkpoint_barrier_tcpstore_libuv, tcpstore_port, master_address
        self._rank = global_rank
        self._world_size = global_world_size
        self._prefix = barrier_prefix
        self._timeout_secs = timeout_secs
        self._sequence = Counter()

    def execute_barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()
        self._sequence[self._prefix] += 1
