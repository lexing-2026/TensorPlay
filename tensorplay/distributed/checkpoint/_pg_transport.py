from __future__ import annotations

import contextlib
import copy
import io
import logging
import pickle
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Generator, TypeVar

import tensorplay as tp

from .. import distributed_core as dist
from ...utils._pytree import TreeSpec, tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class _TensorMeta:
    shape: tuple[int, ...]
    dtype: Any
    storage_offset: int
    stride: tuple[int, ...]
    nbytes: int


@dataclass
class _DTensorMeta:
    local: _TensorMeta
    device_mesh: Any
    placements: Any
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass
class _ShardedTensorMeta:
    local_shards_meta: list[_TensorMeta]
    local_shards_shard_metadata: list[Any]
    sharded_tensor_metadata: Any


@dataclass
class _StateDictMeta:
    treespec: TreeSpec
    paths: list[tuple[Any, ...]]
    non_tensor_leaves: list[Any]


def _paths(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    if isinstance(value, dict):
        result: list[tuple[Any, ...]] = []
        for key, child in value.items():
            result.extend(_paths(child, path + (key,)))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, child in enumerate(value):
            result.extend(_paths(child, path + (index,)))
        return result
    return [path]


def _dtensor_type() -> Any:
    try:
        from ..tensor import DTensor

        return DTensor
    except ImportError:
        return ()


def _sharded_types() -> tuple[Any, Any, Any]:
    try:
        from .._shard.sharded_tensor import Shard, ShardedTensor
        from .._shard.metadata import ShardMetadata

        return ShardedTensor, Shard, ShardMetadata
    except ImportError:
        return (), (), ()


@contextlib.contextmanager
def _timeit(name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %ss", name, time.perf_counter() - start)


def _cast_tensor(tensor: tp.Tensor, dtype: Any) -> tp.Tensor:
    storage_fn = getattr(tensor, "untyped_storage", None)
    if not callable(storage_fn):
        raise TypeError("tensor storage is unavailable")
    storage = storage_fn()
    result = tp.empty((0,), dtype=dtype, device=tensor.device)
    result.set_(storage)
    return result


def _prepare_tensor(tensor: tp.Tensor) -> tuple[tp.Tensor, _TensorMeta]:
    storage = tensor.untyped_storage()
    raw = _cast_tensor(tensor, tp.uint8)
    return raw, _TensorMeta(
        shape=tuple(int(size) for size in tensor.shape),
        dtype=tensor.dtype,
        storage_offset=int(tensor.storage_offset()),
        stride=tuple(int(size) for size in tensor.stride()),
        nbytes=int(storage.nbytes()),
    )


def _prepare_state_dict(
    state_dict: object, device: Any = None
) -> tuple[_StateDictMeta, list[tp.Tensor]]:
    leaves, treespec = tree_flatten(state_dict)
    paths = _paths(state_dict)
    if len(leaves) != len(paths):
        raise ValueError("state dictionary path and leaf counts differ")
    dtensor = _dtensor_type()
    sharded_tensor, _, _ = _sharded_types()
    metadata: list[Any] = []
    tensors: list[tp.Tensor] = []
    for value in leaves:
        if dtensor and isinstance(value, dtensor):
            raw, tensor_meta = _prepare_tensor(value.to_local())
            tensors.append(raw.to(device=device or "cpu"))
            metadata.append(
                _DTensorMeta(
                    local=tensor_meta,
                    device_mesh=value.device_mesh,
                    placements=value.placements,
                    shape=tuple(int(size) for size in value.shape),
                    stride=tuple(int(size) for size in value.stride()),
                )
            )
            continue
        if sharded_tensor and isinstance(value, sharded_tensor):
            local_meta: list[_TensorMeta] = []
            shard_meta: list[Any] = []
            for shard in value.local_shards():
                raw, tensor_meta = _prepare_tensor(shard.tensor)
                tensors.append(raw.to(device=device or "cpu"))
                local_meta.append(tensor_meta)
                shard_meta.append(copy.deepcopy(shard.metadata))
            metadata.append(
                _ShardedTensorMeta(
                    local_shards_meta=local_meta,
                    local_shards_shard_metadata=shard_meta,
                    sharded_tensor_metadata=copy.deepcopy(value.metadata()),
                )
            )
            continue
        if isinstance(value, tp.Tensor):
            raw, tensor_meta = _prepare_tensor(value)
            tensors.append(raw.to(device=device or "cpu"))
            metadata.append(tensor_meta)
        else:
            metadata.append(value)
    return _StateDictMeta(treespec, paths, metadata), tensors


def _send_tensor(tensor: tp.Tensor, destination: int, tag: int, group: Any = None) -> None:
    work = dist.send(tensor, dst=destination, group=group, tag=tag)
    if work is not None and hasattr(work, "wait"):
        work.wait()


def _recv_tensor(tensor: tp.Tensor, source: int, tag: int, group: Any = None) -> None:
    work = dist.recv(tensor, src=source, group=group, tag=tag)
    if work is not None and hasattr(work, "wait"):
        work.wait()


class PGTransport:
    def __init__(
        self,
        pg: Any = None,
        timeout: timedelta | Any = timedelta(minutes=30),
        device: Any = "cpu",
        state_dict: Callable[[], object] | None = None,
        use_single_device: bool = False,
    ) -> None:
        if not isinstance(timeout, timedelta) and device == "cpu":
            device = timeout
            timeout = timedelta(minutes=30)
        self._work: list[Any] = []
        self._pg = pg
        self._timeout = timeout
        self._device = device or "cpu"
        self._state_dict = state_dict
        self.use_single_device = use_single_device

    def send_checkpoint(self, dst_ranks: list[int], state_dict: object) -> None:
        with _timeit("preparing state_dict"):
            metadata, tensors = _prepare_state_dict(state_dict, self._device)
        raw_metadata = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
        length = tp.tensor([len(raw_metadata)], dtype=tp.int64, device=self._device)
        data = tp.frombuffer(raw_metadata, dtype=tp.uint8).to(device=self._device)
        with _timeit("send metadata"):
            for destination in dst_ranks:
                _send_tensor(length, destination, 1, self._pg)
                _send_tensor(data, destination, 2, self._pg)
        with _timeit("send tensors"):
            for index, tensor in enumerate(tensors):
                if tensor.device != self._device:
                    tensor = tensor.to(device=self._device)
                for destination in dst_ranks:
                    _send_tensor(tensor, destination, 3 + index, self._pg)

    def recv_checkpoint(self, src_rank: int) -> object:
        existing = self._state_dict() if self._state_dict else {}
        existing_leaves, _ = tree_flatten(existing)
        metadata_length = tp.zeros((1,), dtype=tp.int64, device=self._device)
        _recv_tensor(metadata_length, src_rank, 1, self._pg)
        length = int(metadata_length.item())
        raw_metadata = tp.empty((length,), dtype=tp.uint8, device=self._device)
        _recv_tensor(raw_metadata, src_rank, 2, self._pg)
        metadata: _StateDictMeta = pickle.loads(raw_metadata.to(device="cpu").numpy().tobytes())
        existing_by_path = dict(zip(_paths(existing), existing_leaves))
        tensor_index = 0
        values: list[Any] = []
        dtensor = _dtensor_type()
        sharded_tensor, shard_type, shard_metadata_type = _sharded_types()

        def receive_tensor(tensor_meta: _TensorMeta, path: tuple[Any, ...]) -> tp.Tensor:
            nonlocal tensor_index
            destination = existing_by_path.get(path)
            if dtensor and isinstance(destination, dtensor):
                destination = destination.to_local()
            if isinstance(destination, tp.Tensor) and str(destination.device) == str(self._device):
                raw = _cast_tensor(destination, tp.uint8)
                if int(raw.numel()) != tensor_meta.nbytes:
                    raise ValueError("destination tensor storage size differs")
            else:
                raw = tp.empty((tensor_meta.nbytes,), dtype=tp.uint8, device=self._device)
            _recv_tensor(raw, src_rank, 3 + tensor_index, self._pg)
            tensor_index += 1
            if not isinstance(destination, tp.Tensor):
                raw = raw.to(device="cpu")
            return raw.view(tensor_meta.dtype).as_strided(
                tensor_meta.shape,
                tensor_meta.stride,
                tensor_meta.storage_offset,
            )

        for path, value in zip(metadata.paths, metadata.non_tensor_leaves):
            if isinstance(value, _TensorMeta):
                values.append(receive_tensor(value, path))
            elif isinstance(value, _DTensorMeta):
                local = receive_tensor(value.local, path)
                if not dtensor:
                    values.append(local)
                else:
                    values.append(
                        dtensor(
                            local,
                            value.device_mesh,
                            value.placements,
                            shape=value.shape,
                            stride=value.stride,
                        )
                    )
            elif isinstance(value, _ShardedTensorMeta):
                local_shards = []
                current_rank = dist.get_rank(self._pg)
                for shard_meta, original in zip(
                    value.local_shards_meta,
                    value.local_shards_shard_metadata,
                ):
                    tensor = receive_tensor(shard_meta, path)
                    if shard_type and shard_metadata_type:
                        updated = shard_metadata_type(
                            list(original.shard_offsets),
                            list(original.shard_sizes),
                            f"rank:{current_rank}/{tensor.device.type}",
                        )
                        local_shards.append(shard_type(tensor, updated))
                if sharded_tensor and local_shards:
                    values.append(
                        sharded_tensor._init_from_local_shards_and_global_metadata(
                            local_shards,
                            value.sharded_tensor_metadata,
                        )
                    )
                else:
                    values.append(local_shards)
            else:
                values.append(value)
        return tree_unflatten(values, metadata.treespec)
