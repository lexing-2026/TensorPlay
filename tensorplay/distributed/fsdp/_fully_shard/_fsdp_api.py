"""Composable sharding policies and communication interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import tensorplay as tp

__all__ = [
    "MixedPrecisionPolicy",
    "Comm",
    "AllGather",
    "ReduceScatter",
    "DataParallelMeshDims",
    "OffloadPolicy",
    "CPUOffloadPolicy",
]


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: Any = None
    reduce_dtype: Any = None
    output_dtype: Any = None
    cast_forward_inputs: bool = True


class Comm(ABC):
    @abstractmethod
    def allocate(self, size: Sequence[int], *, dtype: Any, device: Any) -> Any:
        return tp.empty(tuple(size), dtype=dtype, device=device)


class AllGather(Comm):
    @abstractmethod
    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, async_op: bool = False) -> Any:
        raise NotImplementedError


class ReduceScatter(Comm):
    @abstractmethod
    def __call__(self, output_tensor: Any, input_tensor: Any, group: Any, op: Any, async_op: bool = False) -> Any:
        raise NotImplementedError


@dataclass
class DataParallelMeshDims:
    shard: str | tuple[str, ...] | None = None
    replicate: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.shard is None and self.replicate is None:
            raise ValueError("at least one data-parallel mesh dimension is required")

    @property
    def shard_names(self) -> tuple[str, ...]:
        if self.shard is None:
            return ()
        return (self.shard,) if isinstance(self.shard, str) else tuple(self.shard)

    @property
    def replicate_names(self) -> tuple[str, ...]:
        if self.replicate is None:
            return ()
        return (self.replicate,) if isinstance(self.replicate, str) else tuple(self.replicate)


@dataclass
class OffloadPolicy:
    pass


@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    pin_memory: bool = True
