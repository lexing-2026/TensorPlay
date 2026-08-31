from __future__ import annotations

import cProfile
import inspect
import io
import itertools
import os
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import wraps
from pstats import Stats
from typing import Any, TypeVar, cast

import tensorplay as tp
import tensorplay.distributed as dist

from .api import CheckpointException, _is_wrapped_exception, _wrap_exception
from .metadata import MetadataIndex
from .protocol import _get_checkpointable_tensor_shard, _is_checkpointable_tensor

try:
    from tensorplay.distributed._shard.sharded_tensor import ShardedTensor
    from tensorplay.distributed._shard.sharded_tensor.shard import Shard
except ImportError:
    ShardedTensor = ()
    Shard = Any

__all__ = ["find_tensor_shard", "find_state_dict_object"]

T = TypeVar("T")
R = TypeVar("R")


def _get_failure_dict(results: list[T | BaseException]) -> dict[int, BaseException]:
    return {
        index: value
        for index, value in enumerate(results)
        if _is_wrapped_exception(value)
    }


def _all_gather_keys(
    local_dict: dict[str, Any], group: Any = None
) -> set[str]:
    keys = list(local_dict)
    gathered: list[list[str] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, keys, group=group)
    return set(itertools.chain.from_iterable(item or [] for item in gathered))


def _assert_same_keys(
    state_dict: dict[str, Any], process_group: Any = None
) -> None:
    if dist.get_world_size(process_group) == 1:
        return
    all_keys = _all_gather_keys(state_dict, process_group)
    difference = all_keys.difference(state_dict)
    if difference:
        raise AssertionError(
            "Keys present on another worker but absent locally: "
            f"{difference}"
        )


class _DistWrapper:
    def __init__(self, group: Any, use_dist: bool, coordinator_rank: int):
        self.group = group
        self.use_dist = bool(use_dist and dist.is_initialized())
        self.coordinator_rank = int(coordinator_rank)
        if self.use_dist:
            self.global_coordinator_rank = (
                dist.get_global_rank(group, coordinator_rank)
                if group is not None
                else coordinator_rank
            )
            self.rank = dist.get_rank(group)
            self.is_coordinator = self.rank == coordinator_rank
        else:
            self.global_coordinator_rank = 0
            self.rank = 0
            self.is_coordinator = True

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        return dist.get_world_size(self.group) if self.use_dist else 1

    def broadcast_object(self, value: T | None) -> T:
        objects = [value]
        if self.use_dist:
            dist.broadcast_object_list(
                objects, src=self.global_coordinator_rank, group=self.group
            )
        return cast(T, objects[0])

    def gather_object(self, value: T) -> list[T] | None:
        if not self.use_dist:
            return [value]
        gathered = cast(
            list[T] | None,
            [None] * self.get_world_size() if self.is_coordinator else None,
        )
        dist.gather_object(
            value,
            object_gather_list=gathered if self.is_coordinator else None,
            dst=self.global_coordinator_rank,
            group=self.group,
        )
        return gathered

    def all_gather_object(self, value: T) -> list[T]:
        if not self.use_dist:
            return [value]
        gathered = cast(list[T], [None] * self.get_world_size())
        dist.all_gather_object(gathered, value, group=self.group)
        return gathered

    def scatter_object(self, values: list[T] | None) -> T:
        if not self.use_dist:
            if not values:
                raise AssertionError("values must contain an item")
            return values[0]
        result = [None]
        dist.scatter_object_list(
            result,
            scatter_object_input_list=values if self.is_coordinator else None,
            src=self.global_coordinator_rank,
            group=self.group,
        )
        return cast(T, result[0])

    def reduce_scatter(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[list[T]], list[R]],
    ) -> R:
        try:
            local: T | BaseException = map_fun()
        except BaseException as error:
            local = _wrap_exception(error)
        gathered = self.gather_object(local)
        results: list[R | CheckpointException] | None = None
        if self.is_coordinator:
            if gathered is None:
                raise AssertionError("coordinator did not receive gathered data")
            failures = _get_failure_dict(gathered)
            if not failures:
                try:
                    results = cast(list[R | CheckpointException], reduce_fun(cast(list[T], gathered)))
                except BaseException as error:
                    failures[self.rank] = _wrap_exception(error)
            if failures:
                results = [CheckpointException(step, failures)] * self.get_world_size()
        result = self.scatter_object(results)
        if isinstance(result, CheckpointException):
            raise result
        return cast(R, result)

    def all_reduce(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[list[T]], R],
    ) -> R:
        try:
            local: T | BaseException = map_fun()
        except BaseException as error:
            local = _wrap_exception(error)
        gathered = self.gather_object(local)
        result: R | CheckpointException | None = None
        if self.is_coordinator:
            if gathered is None:
                raise AssertionError("coordinator did not receive gathered data")
            failures = _get_failure_dict(gathered)
            if not failures:
                try:
                    result = reduce_fun(cast(list[T], gathered))
                except BaseException as error:
                    failures[self.rank] = _wrap_exception(error)
            if failures:
                result = CheckpointException(step, failures)
        result = self.broadcast_object(result)
        if isinstance(result, CheckpointException):
            raise result
        return cast(R, result)

    def all_gather(self, step: str, map_fun: Callable[[], T]) -> list[T]:
        try:
            result: T | BaseException = map_fun()
        except BaseException as error:
            result = _wrap_exception(error)
        gathered = self.all_gather_object(result)
        failures = _get_failure_dict(gathered)
        if failures:
            raise CheckpointException(step, failures)
        return cast(list[T], gathered)

    def broadcast(self, step: str, map_fun: Callable[[], T]) -> T:
        result: T | CheckpointException | None = None
        if self.is_coordinator:
            try:
                result = map_fun()
            except BaseException as error:
                result = CheckpointException(step, {self.rank: _wrap_exception(error)})
        result = self.broadcast_object(result)
        if isinstance(result, CheckpointException):
            raise result
        return cast(T, result)

    def barrier(self) -> None:
        if self.use_dist:
            dist.barrier(group=self.group)


def _find_shard(tensor: Any, index: MetadataIndex) -> Any:
    if index.offset is None:
        raise ValueError(
            f"Cannot locate {index.fqn!r} without an offset for a sharded value"
        )
    shards = tensor.local_shards()
    if index.index is not None and index.index < len(shards):
        candidate = shards[index.index]
        if tuple(candidate.metadata.shard_offsets) == tuple(index.offset):
            return candidate
    for shard in shards:
        if tuple(shard.metadata.shard_offsets) == tuple(index.offset):
            return shard
    raise ValueError(f"Could not find shard at {index.offset!r} for {index.fqn!r}")


def find_tensor_shard(tensor: Any, index: MetadataIndex) -> Any:
    getter = getattr(tensor, "__get_tensor_shard__", None)
    if getter is not None:
        return getter(index)
    if _is_checkpointable_tensor(tensor):
        return _get_checkpointable_tensor_shard(tensor, index)
    if ShardedTensor and isinstance(tensor, ShardedTensor):
        return _find_shard(tensor, index).tensor
    if index.offset is not None:
        shape = tuple(getattr(tensor, "shape", tensor.size()))
        if tuple(index.offset) == (0,) * len(shape):
            return tensor
        raise ValueError(
            f"{index.fqn!r} is not sharded and cannot be indexed by offset "
            f"{index.offset!r}"
        )
    return tensor


def find_state_dict_object(state_dict: dict[str, Any], index: MetadataIndex) -> Any:
    if index.fqn not in state_dict:
        raise ValueError(f"Could not find state-dict key {index.fqn!r}")
    value = state_dict[index.fqn]
    if isinstance(value, tp.Tensor) or (ShardedTensor and isinstance(value, ShardedTensor)):
        return find_tensor_shard(value, index)
    if index.offset is not None:
        raise ValueError(
            f"{index.fqn!r} is not sharded and cannot be indexed by offset "
            f"{index.offset!r}"
        )
    return value


def _element_wise_add(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return [left + right for left, right in zip(a, b)]


def _element_wise_sub(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return [left - right for left, right in zip(a, b)]


class _ReaderView(io.IOBase):
    def __init__(self, base_stream: io.IOBase, offset: int, length: int):
        super().__init__()
        self.offset = int(offset)
        self.len = int(length)
        self.base_stream = base_stream
        self.seek(0)

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int:
        if whence == os.SEEK_SET:
            offset = self.offset + offset
        elif whence == os.SEEK_END:
            whence = os.SEEK_SET
            offset = self.offset + self.len + offset
        return self.base_stream.seek(offset, whence)

    def tell(self) -> int:
        return self.base_stream.tell() - self.offset

    def readable(self) -> bool:
        return self.base_stream.readable()

    def seekable(self) -> bool:
        return self.base_stream.seekable()

    def readinto(self, buffer: Any) -> int:
        remaining = self.len - self.tell()
        if remaining <= 0:
            return 0
        view = buffer if len(buffer) <= remaining else memoryview(buffer)[:remaining]
        return self.base_stream.readinto(view)  # type: ignore[attr-defined]

    def read(self, size: int = -1) -> bytes:
        remaining = self.len - self.tell()
        if remaining <= 0:
            return b""
        if size < 0 or size > remaining:
            size = remaining
        return self.base_stream.read(size)


def _create_file_view(file: io.IOBase, offset: int, length: int) -> io.IOBase:
    return _ReaderView(file, offset, length)


def _normalize_device_info(device_type: str, device_id: int) -> str:
    return "cpu" if device_type == "cpu" else f"{device_type}:{device_id}"


ENABLE_PROFILE = False


@contextmanager
def _profile():
    if ENABLE_PROFILE and (not dist.is_initialized() or dist.get_rank() == 0):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            Stats(profiler).sort_stats("time").print_stats(10)
    else:
        yield


def _api_bc_check(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def inner_func(*args: Any, **kwargs: Any) -> Any:
        if len(args) != 2:
            return func(*args, **kwargs)
        warnings.warn(
            f"The argument order of {func.__name__} has changed.",
            stacklevel=2,
        )
        keyword_only = [
            parameter.name
            for parameter in inspect.signature(func).parameters.values()
            if parameter.kind is parameter.KEYWORD_ONLY
        ]
        if len(keyword_only) != 1:
            raise RuntimeError(f"Unexpected keyword-only parameters: {keyword_only}")
        key = keyword_only[0]
        if key in kwargs:
            raise AssertionError(f"{key} was supplied twice")
        kwargs[key] = args[1]
        return func(args[0], **kwargs)

    return inner_func
