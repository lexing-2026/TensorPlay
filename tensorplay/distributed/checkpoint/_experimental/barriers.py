from __future__ import annotations

import abc
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import tensorplay.distributed as dist
from ...elastic.utils import store as store_util

logger = logging.getLogger(__name__)

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
    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    @abc.abstractmethod
    def execute_barrier(self) -> None: ...


@register_barrier
class DistBarrier(Barrier):
    barrier_type = "dist_barrier"

    def __init__(self) -> None:
        if not dist.is_initialized():
            raise AssertionError("a process group is required for DistBarrier")

    def execute_barrier(self) -> None:
        dist.barrier()


@register_barrier
class TCPStoreBarrier(Barrier):
    barrier_type = "tcp_store"

    def __init__(
        self,
        global_rank: int,
        global_world_size: int,
        barrier_prefix: str,
        timeout_barrier_init_secs: int,
        use_checkpoint_barrier_tcpstore_libuv: bool,
        tcpstore_port: int,
        master_address: str,
        timeout_secs: int,
    ) -> None:
        del use_checkpoint_barrier_tcpstore_libuv
        self._tcp_store_barrier_seq = Counter()
        self._barrier_prefix = barrier_prefix
        self._global_rank = int(global_rank)
        self._global_world_size = int(global_world_size)
        self._timeout_secs = timeout_secs
        self._tcp_store = dist.TCPStore(
            master_address,
            int(tcpstore_port),
            world_size=self._global_world_size,
            timeout=float(timeout_barrier_init_secs),
            is_master=self._global_rank == 0,
        )

    def execute_barrier(self) -> None:
        prefix = self._barrier_prefix
        sequence = self._tcp_store_barrier_seq[prefix]
        self._tcp_store.set(f"rank{self._global_rank}", str(sequence))
        with store_util.barrier(
            store=self._tcp_store,
            world_size=self._global_world_size,
            key_prefix=f"{prefix}{sequence}",
        ):
            pass
        self._tcp_store_barrier_seq[prefix] += 1
