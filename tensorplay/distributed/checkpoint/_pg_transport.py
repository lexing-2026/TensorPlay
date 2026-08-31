from __future__ import annotations

import contextlib
import io
import pickle
from dataclasses import dataclass
from typing import Any, Generator

import tensorplay as tp

from .. import distributed_core as dist


@dataclass
class _TensorMeta:
    shape: tuple[int, ...]
    dtype: Any
    device: Any


@dataclass
class _DTensorMeta:
    shape: tuple[int, ...]
    dtype: Any
    device: Any
    spec: Any = None


@dataclass
class _ShardedTensorMeta:
    shape: tuple[int, ...]
    dtype: Any
    device: Any
    metadata: Any = None


@dataclass
class _StateDictMeta:
    keys: list[str]
    values: list[Any]


@contextlib.contextmanager
def _timeit(name: str) -> Generator[None, None, None]:
    del name
    yield


def _prepare_tensor(tensor: tp.Tensor) -> tuple[tp.Tensor, _TensorMeta]:
    return tensor.detach().clone(), _TensorMeta(tuple(tensor.shape), tensor.dtype, tensor.device)


def _prepare_state_dict(state_dict: dict[str, Any]) -> tuple[dict[str, Any], _StateDictMeta]:
    values = {}
    for key, value in state_dict.items():
        values[key] = _prepare_tensor(value)[0] if isinstance(value, tp.Tensor) else value
    return values, _StateDictMeta(list(values), list(values.values()))


def _cast_tensor(tensor: tp.Tensor, dtype: Any) -> tp.Tensor:
    return tensor.to(dtype=dtype)


class PGTransport:
    def __init__(self, process_group: Any = None, device: Any = None, use_single_device: bool = False) -> None:
        self.process_group = process_group
        self.device = device
        self.use_single_device = use_single_device

    def send_checkpoint(self, dst_ranks: list[int], state_dict: object) -> None:
        if len(dst_ranks) != 1:
            raise ValueError("one destination is required for the native transport")
        payload, _ = _prepare_state_dict(state_dict) if isinstance(state_dict, dict) else (state_dict, None)
        dist.send_object_list([payload], dst=dst_ranks[0], group=self.process_group)

    def recv_checkpoint(self, src_rank: int) -> object:
        values = [None]
        dist.recv_object_list(values, src=src_rank, group=self.process_group)
        return values[0]
