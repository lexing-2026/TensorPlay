from __future__ import annotations

import os
import io
import json
import operator
import pickle
import queue
import threading
import uuid
from contextlib import contextmanager
from abc import ABC, abstractmethod
from concurrent.futures import Future
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import replace
from dataclasses import dataclass
from enum import Enum
from io import UnsupportedOperation
from pathlib import Path
from typing import Any, cast

import tensorplay as tp

from .metadata import Metadata, StorageMeta
from ._hf_utils import (
    CUSTOM_METADATA_KEY,
    DCP_VERSION_KEY,
    FORMAT_KEY,
    FORMAT_VALUE,
    HF_DCP_VERSION,
)
from .planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    SavePlan,
    SavePlanner,
    WriteItemType,
)
from .storage import StorageReader, StorageWriter, WriteResult
from ._extension import ExtensionRegistry, StreamTransformExtension
from .staging import BlockingAsyncStager
from .utils import _create_file_view

__all__ = [
    "FileSystemWriter",
    "FileSystemReader",
    "FileSystem",
    "FileSystemBase",
    "SerializationFormat",
    "StorageReader",
    "StorageWriter",
]

_METADATA_FILE = ".metadata"
_metadata_fn = _METADATA_FILE
CURRENT_DCP_VERSION = "1.0.0"
DEFAULT_SUFFIX = ".distcp"


@dataclass
class _StorageInfo:
    relative_path: str
    offset: int
    length: int
    transform_descriptors: Sequence[str] | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


@dataclass
class _StoragePrefix:
    prefix: str


class SerializationFormat(Enum):
    TORCH_SAVE = "torch_save"
    SAFETENSORS = "safetensors"


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class _TensorLoader(ABC):
    @abstractmethod
    def add(self, size: int, obj: object) -> None: ...

    @abstractmethod
    def start_loading(self) -> None: ...

    @abstractmethod
    def values(self) -> Iterator[tuple[tp.Tensor, object]]: ...


class _SerialCpuLoader(_TensorLoader):
    def __init__(self, resolve_fun: Callable[..., Any]) -> None:
        self.resolve_fun = resolve_fun
        self.items: list[tuple[int, object]] = []

    def add(self, size: int, obj: object) -> None:
        self.items.append((size, obj))

    def start_loading(self) -> None:
        return None

    def values(self) -> Iterator[tuple[tp.Tensor, object]]:
        for _, obj in self.items:
            value = self.resolve_fun(obj).detach()
            value = value.to(device="cpu")
            storage = getattr(value, "untyped_storage", None)
            if callable(storage):
                try:
                    if int(storage().size()) != int(value.numel()) * int(value.dtype.itemsize):
                        value = value.clone()
                except (AttributeError, RuntimeError, TypeError):
                    pass
            yield value, obj


class _OverlappingCpuLoader(_SerialCpuLoader):
    def __init__(
        self,
        resolve_fun: Callable[..., Any],
        stream: Any = None,
        inflight_threshhold: int = 1_000_000,
    ) -> None:
        super().__init__(resolve_fun)
        self.stream = stream
        self.inflight_threshhold = int(inflight_threshhold)
        self._in_flight_data = 0
        self._current_items: deque[tuple[tp.Tensor, object]] = deque()
        self._index = 0
        self._started = False

    @property
    def _done(self) -> bool:
        return self._index >= len(self.items)

    def __post_init__(self) -> None:
        return None

    def add(self, size: int, obj: object) -> None:
        if getattr(self, "_started", False):
            raise RuntimeError("cannot add items after loading started")
        self.items.append((size, obj))

    def start_loading(self) -> None:
        if self._started:
            return
        self._started = True
        self.items.sort(key=operator.itemgetter(0))
        self._refill()

    def _drain(self) -> list[tuple[tp.Tensor, object]]:
        if self._in_flight_data < self.inflight_threshhold:
            return []
        drained = list(self._current_items)
        self._current_items.clear()
        self._in_flight_data = 0
        return drained

    def _refill(self) -> None:
        limit = self.inflight_threshhold
        while not self._done and (limit <= 0 or self._in_flight_data < limit):
            size, obj = self.items[self._index]
            self._index += 1
            value = self.resolve_fun(obj).detach().to(device="cpu")
            self._current_items.append((value, obj))
            self._in_flight_data += max(int(size), int(value.numel()) * int(value.dtype.itemsize))

    def _finish(self) -> Iterable[tuple[tp.Tensor, object]]:
        if not self._done:
            raise AssertionError("all tensor items must be loaded before finishing")
        result = tuple(self._current_items)
        self._current_items.clear()
        self._in_flight_data = 0
        return result

    def values(self) -> Iterator[tuple[tp.Tensor, object]]:
        self.start_loading()
        while not self._done:
            drained = self._drain()
            self._refill()
            yield from drained
        yield from self._finish()


class _StorageWriterTransforms:
    def __init__(self, extensions: Sequence[StreamTransformExtension] | None = None) -> None:
        self.extensions = () if extensions is None else tuple(extensions)

    def transform_save_stream(
        self, write_item: Any, raw_stream: io.IOBase
    ) -> tuple[io.IOBase, list[str]]:
        del write_item
        class NoCloseWriter(io.IOBase):
            def __init__(self, raw: io.IOBase) -> None:
                self.raw = raw

            def writeable(self) -> bool:
                return True

            def writable(self) -> bool:
                return True

            def write(self, data: Any) -> int:
                return int(self.raw.write(data))

            def flush(self) -> None:
                self.raw.flush()

            def close(self) -> None:
                self.flush()

        stream: io.IOBase = NoCloseWriter(raw_stream)
        for extension in self.extensions:
            stream = extension.transform_to(stream)
        return stream, [extension.get_descriptor() for extension in reversed(self.extensions)]


class _StorageReaderTransforms:
    def __init__(self, extension_registry: ExtensionRegistry | None = None) -> None:
        self.extension_registry = extension_registry or ExtensionRegistry()

    def transform_load_stream(
        self,
        read_item: Any,
        transform_descriptors: Sequence[str],
        raw_stream: io.IOBase,
    ) -> io.IOBase:
        del read_item
        stream = raw_stream
        for extension in self.extension_registry.from_descriptor_list(transform_descriptors):
            stream = extension.transform_from(stream)
        return stream


def _item_size(item: Any) -> int:
    if item.tensor_data is None:
        return 1
    count = 1
    for size in item.tensor_data.size:
        count *= int(size)
    return count * int(getattr(item.tensor_data.properties.dtype, "itemsize", 1))


def _split_by_size_and_type(bins: int, items: list[Any]) -> list[list[Any]]:
    if bins <= 1:
        return [items]
    buckets: list[list[Any]] = [[] for _ in range(bins)]
    sizes = [0] * bins
    byte_items = [item for item in items if item.type is WriteItemType.BYTE_IO]
    tensor_items = sorted(
        (item for item in items if item.type is not WriteItemType.BYTE_IO),
        key=_item_size,
        reverse=True,
    )
    for index, item in enumerate(byte_items):
        buckets[index % bins].append(item)
    for item in tensor_items:
        bucket = min(range(bins), key=sizes.__getitem__)
        buckets[bucket].append(item)
        sizes[bucket] += _item_size(item)
    return buckets


def _write_item(
    transforms: _StorageWriterTransforms,
    stream: io.IOBase,
    data: Any,
    write_item: Any,
    storage_key: str,
    serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
) -> WriteResult:
    offset = stream.tell()
    output, descriptors = transforms.transform_save_stream(write_item, stream)
    if write_item.type is WriteItemType.BYTE_IO:
        if not hasattr(data, "getbuffer"):
            raise TypeError("byte write items require a byte stream")
        output.write(data.getbuffer())
    else:
        if serialization_format is SerializationFormat.TORCH_SAVE:
            tp.save(data, output)
    output.flush()
    output.close()
    if serialization_format is SerializationFormat.TORCH_SAVE or hasattr(
        data, "getbuffer"
    ):
        length = stream.tell() - offset
    else:
        length = int(data.numel()) * int(data.itemsize())
    return WriteResult(
        index=write_item.index,
        size_in_bytes=int(length),
        storage_data=_StorageInfo(
            storage_key,
            int(offset),
            int(length),
            None if not descriptors else descriptors,
        ),
    )


def _narrow_tensor(
    value: tp.Tensor, offsets: Sequence[int], lengths: Sequence[int]
) -> tp.Tensor:
    result = value
    for dimension, (offset, length) in enumerate(zip(offsets, lengths)):
        if int(length):
            result = result.narrow(dimension, int(offset), int(length))
    return result


def _write_files_from_queue(
    create_stream: Callable[..., Any],
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    transforms: _StorageWriterTransforms,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
    serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
) -> None:
    del inflight_threshhold, thread_count
    try:
        while True:
            file_name, storage_key, items = file_queue.get_nowait()
            results: list[WriteResult] = []
            tensor_loader = _SerialCpuLoader(planner.resolve_data)
            for item in items:
                if item.type is not WriteItemType.BYTE_IO:
                    tensor_loader.add(_item_size(item), item)
            tensor_loader.start_loading()
            with create_stream(file_name, "wb") as stream:
                for item in items:
                    if item.type is not WriteItemType.BYTE_IO:
                        continue
                    results.append(
                        _write_item(
                            transforms,
                            stream,
                            planner.resolve_data(item),
                            item,
                            storage_key,
                            serialization_format,
                        )
                    )
                tensor_dict: dict[str, tp.Tensor] = {}
                metadata_dict: dict[str, dict[str, Any]] = {}
                for tensor, item in tensor_loader.values():
                    results.append(
                        _write_item(
                            transforms,
                            stream,
                            tensor,
                            item,
                            storage_key,
                            serialization_format,
                        )
                    )
                    if serialization_format is SerializationFormat.SAFETENSORS:
                        tensor_dict[item.index.fqn] = tensor
                        metadata_dict[item.index.fqn] = {
                            "saved_offsets": item.tensor_data.chunk.offsets
                        }
                if serialization_format is SerializationFormat.SAFETENSORS:
                    from tensorplay.serialization.archive import write_safetensors_file

                    write_safetensors_file(
                        stream,
                        tensor_dict,
                        metadata={
                            CUSTOM_METADATA_KEY: json.dumps(metadata_dict),
                            DCP_VERSION_KEY: str(HF_DCP_VERSION),
                            FORMAT_KEY: FORMAT_VALUE,
                        },
                    )
                if use_fsync:
                    stream.flush()
                    try:
                        os.fsync(stream.fileno())
                    except (AttributeError, UnsupportedOperation):
                        pass
            result_queue.put(results)
    except queue.Empty:
        return None


class FileSystemBase(ABC):
    @contextmanager
    @abstractmethod
    def create_stream(self, path: str | os.PathLike[str], mode: str) -> Generator[io.IOBase, None, None]: ...

    @abstractmethod
    def concat_path(self, path: str | os.PathLike[str], suffix: str) -> str | os.PathLike[str]: ...

    @abstractmethod
    def rename(self, path: str | os.PathLike[str], new_path: str | os.PathLike[str]) -> None: ...

    @abstractmethod
    def init_path(self, path: str | os.PathLike[str]) -> str | os.PathLike[str]: ...

    @abstractmethod
    def mkdir(self, path: str | os.PathLike[str]) -> None: ...

    @classmethod
    @abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool: ...

    @abstractmethod
    def exists(self, path: str | os.PathLike[str]) -> bool: ...

    @abstractmethod
    def rm_file(self, path: str | os.PathLike[str]) -> None: ...

    def ls(self, path: str | os.PathLike[str]) -> list[str]:
        raise NotImplementedError


class FileSystem(FileSystemBase):
    @contextmanager
    def create_stream(self, path: str | os.PathLike[str], mode: str) -> Generator[io.IOBase, None, None]:
        with Path(path).open(mode) as stream:
            yield cast(io.IOBase, stream)

    def concat_path(self, path: str | os.PathLike[str], suffix: str) -> str | os.PathLike[str]:
        return Path(path) / suffix

    def init_path(self, path: str | os.PathLike[str]) -> str | os.PathLike[str]:
        return Path(path)

    def rename(self, path: str | os.PathLike[str], new_path: str | os.PathLike[str]) -> None:
        Path(path).rename(Path(new_path))

    def mkdir(self, path: str | os.PathLike[str]) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return isinstance(checkpoint_id, (str, os.PathLike)) and bool(str(checkpoint_id)) and "://" not in str(checkpoint_id)

    def exists(self, path: str | os.PathLike[str]) -> bool:
        return Path(path).exists()

    def rm_file(self, path: str | os.PathLike[str]) -> None:
        Path(path).unlink()

    def ls(self, path: str | os.PathLike[str]) -> list[str]:
        return [str(item) for item in Path(path).iterdir()]


class _FileSystemWriter(StorageWriter):
    """Write one transactionally committed checkpoint directory."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self.single_file_per_rank = bool(single_file_per_rank)
        self.sync_files = bool(sync_files)
        self.per_thread_copy_ahead = int(per_thread_copy_ahead)
        self.serialization_format = serialization_format
        self.transforms = _StorageWriterTransforms(_extensions)
        if int(thread_count) <= 0:
            raise ValueError("thread_count must be positive")
        self.fs: FileSystemBase = FileSystem()
        self.path = self.fs.init_path(path)
        self.overwrite = bool(overwrite)
        self.thread_count = int(thread_count)
        self._data_name: str | None = None
        self._metadata: Metadata | None = None
        self._rank = 0
        self._use_collectives = True
        self.save_id = _generate_uuid()
        self._written_files: set[str] = set()
        self._committed = False

    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None:
        if checkpoint_id is not None:
            self.path = self.fs.init_path(checkpoint_id)
        self._data_name = None
        self._metadata = None
        self._written_files.clear()
        self._committed = False
        self.save_id = _generate_uuid()

    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        del is_coordinator, args
        self._rank = int(kwargs.get("rank", 0))
        self._use_collectives = bool(kwargs.get("use_collectives", True))
        self._data_name = None
        self._metadata = None
        self._written_files.clear()
        self._committed = False
        self.fs.mkdir(self.path)
        if not self.overwrite and self._metadata_exists():
            raise FileExistsError(f"checkpoint already exists at {self.path}")

    def storage_meta(self) -> StorageMeta:
        return StorageMeta(checkpoint_id=self.path, save_id=getattr(self, "save_id", None))

    def _metadata_exists(self) -> bool:
        rank = None if self._use_collectives else 0
        return self.fs.exists(self._get_metadata_path(rank))

    def _path(self, name: str) -> str | os.PathLike[str]:
        return self.fs.concat_path(self.path, name)

    def _write_checkpoint_object(self, path: str | os.PathLike[str], value: Any) -> None:
        temporary = self.fs.concat_path(
            os.path.dirname(os.fspath(path)),
            f".{os.path.basename(os.fspath(path))}.{uuid.uuid4().hex}.tmp",
        )
        try:
            with self.fs.create_stream(temporary, "wb") as stream:
                pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
            if self.fs.exists(path):
                self.fs.rm_file(path)
            self.fs.rename(temporary, path)
        except BaseException:
            try:
                self.fs.rm_file(temporary)
            except BaseException:
                pass
            raise

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        if not isinstance(plan, SavePlan):
            raise TypeError("plan must be a SavePlan")
        self.fs.mkdir(self.path)
        if not self.overwrite and self._metadata_exists():
            raise FileExistsError(f"checkpoint already exists at {self.path}")
        if not self._use_collectives:
            plan = replace(plan, storage_data=_StoragePrefix(f"__{self._rank}_"))
        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        if not isinstance(plans, list) or not all(
            isinstance(plan, SavePlan) for plan in plans
        ):
            raise TypeError("plans must be a list of SavePlan objects")
        return [
            replace(
                plan,
                storage_data=(
                    plan.storage_data
                    if plan.storage_data is not None
                    else _StoragePrefix(f"__{index}_")
                ),
            )
            for index, plan in enumerate(plans)
        ]

    def _write_planned_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        storage_plan = plan.storage_data
        if not isinstance(storage_plan, _StoragePrefix):
            raise TypeError("plan.storage_data must be a _StoragePrefix")
        file_queue: queue.Queue = queue.Queue()
        file_count = 0

        def next_file() -> tuple[str | os.PathLike[str], str]:
            nonlocal file_count
            storage_key = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            self._data_name = storage_key
            self._written_files.add(storage_key)
            return self._path(storage_key), storage_key

        if self.single_file_per_rank:
            buckets = _split_by_size_and_type(self.thread_count, plan.items)
            for bucket in buckets:
                path, storage_key = next_file()
                file_queue.put((path, storage_key, bucket))
        else:
            for item in plan.items:
                path, storage_key = next_file()
                file_queue.put((path, storage_key, [item]))
        return self._write_data(planner, file_queue)

    def _write_data(
        self, planner: SavePlanner, file_queue: queue.Queue
    ) -> Future[list[WriteResult]]:
        result_queue: queue.Queue = queue.Queue()
        error_queue: queue.Queue = queue.Queue()

        def run() -> None:
            try:
                _write_files_from_queue(
                    self.fs.create_stream,
                    file_queue,
                    result_queue,
                    planner,
                    self.transforms,
                    self.per_thread_copy_ahead,
                    self.sync_files,
                    self.thread_count,
                    self.serialization_format,
                )
            except BaseException as error:
                error_queue.put(error)

        threads = [
            threading.Thread(target=run, daemon=True)
            for _ in range(1, self.thread_count)
        ]
        for thread in threads:
            thread.start()
        run()
        for thread in threads:
            thread.join()
        if not error_queue.empty():
            raise error_queue.get()

        results: list[WriteResult] = []
        try:
            while True:
                results.extend(result_queue.get_nowait())
        except queue.Empty:
            pass
        future: Future[list[WriteResult]] = Future()
        future.set_result(results)
        return future

    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        if not isinstance(plan, SavePlan):
            raise TypeError("plan must be a SavePlan")
        return self._write_planned_data(plan, planner)

    def finish(
        self, metadata: Metadata, results: list[list[WriteResult]]
    ) -> None:
        metadata.version = CURRENT_DCP_VERSION
        storage_data: dict[Any, Any] = {}
        for rank_results in results:
            storage_data.update(
                {result.index: result.storage_data for result in rank_results}
            )
        metadata.storage_data = storage_data
        metadata.storage_meta = self.storage_meta()
        metadata_name = (
            _METADATA_FILE
            if self._use_collectives
            else f"__{self._rank}{_METADATA_FILE}"
        )
        self._write_checkpoint_object(self._path(metadata_name), metadata)
        self._metadata = metadata
        self._committed = True

    def _get_metadata_path(self, rank: int | None = None) -> str | os.PathLike[str]:
        metadata_name = _METADATA_FILE if rank is None else f"__{int(rank)}{_METADATA_FILE}"
        return self._path(metadata_name)

    @property
    def checkpoint_id(self) -> str | os.PathLike[str]:
        return self.path

    def abort(self) -> None:
        if self._committed:
            return
        for file_name in tuple(self._written_files):
            path = self._path(file_name)
            try:
                self.fs.rm_file(path)
            except (FileNotFoundError, OSError):
                pass
        self._written_files.clear()
        self._data_name = None
        self._metadata = None

    def mark_committed(self) -> None:
        self._committed = True


class FileSystemWriter(_FileSystemWriter, BlockingAsyncStager):
    def __init__(
        self,
        path: str | os.PathLike[str],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        cache_staged_state_dict: bool = False,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
    ) -> None:
        _FileSystemWriter.__init__(
            self,
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        BlockingAsyncStager.__init__(
            self,
            cache_staged_state_dict=cache_staged_state_dict,
        )

    def stage(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.per_thread_copy_ahead = 0
        return BlockingAsyncStager.stage(self, state_dict, **kwargs)


class FileSystemReader(StorageReader):
    """Read checkpoint transactions written by :class:`FileSystemWriter`."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        _extension_registry: ExtensionRegistry | None = None,
    ) -> None:
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self._metadata: Any = None
        self.storage_data: dict[Any, Any] = {}
        self._rank = 0
        self._use_collectives = True
        self.load_id = _generate_uuid()
        self.transforms = _StorageReaderTransforms(_extension_registry)

    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None:
        if checkpoint_id is not None:
            self.path = self.fs.init_path(checkpoint_id)
        self._metadata = None
        self.storage_data = {}
        self._rank = 0
        self._use_collectives = True
        self.load_id = _generate_uuid()

    def _slice_file(self, file: io.IOBase, sinfo: _StorageInfo) -> io.IOBase:
        return cast(io.IOBase, _create_file_view(file, sinfo.offset, sinfo.length))

    def _get_metadata_path(self, rank: int | None = None) -> str | os.PathLike[str]:
        metadata_name = _METADATA_FILE if rank is None else f"__{int(rank)}{_METADATA_FILE}"
        return self.fs.concat_path(self.path, metadata_name)

    def read_metadata(self, *args: Any, **kwargs: Any) -> Any:
        del args
        rank = kwargs.get("rank")
        meta_path = self._get_metadata_path(rank)
        if not self.fs.exists(meta_path):
            raise FileNotFoundError(f"checkpoint metadata not found at {meta_path}")
        with self.fs.create_stream(meta_path, "rb") as stream:
            self._metadata = pickle.load(stream)
        if isinstance(self._metadata, Metadata):
            storage_meta = self._metadata.storage_meta
            if storage_meta is None:
                self._metadata.storage_meta = StorageMeta(load_id=self.load_id)
            else:
                self._metadata.storage_meta = replace(storage_meta, load_id=self.load_id)
        return self._metadata

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        del is_coordinator, args
        self._rank = int(kwargs.get("rank", 0))
        self._use_collectives = bool(kwargs.get("use_collectives", True))
        self._metadata = metadata
        if not isinstance(metadata.storage_data, dict):
            raise AssertionError("metadata.storage_data must be a dictionary")
        self.storage_data = metadata.storage_data

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        if not isinstance(plan, LoadPlan):
            raise TypeError("plan must be a LoadPlan")
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        if not isinstance(plans, list) or not all(
            isinstance(plan, LoadPlan) for plan in plans
        ):
            raise TypeError("plans must be a list of LoadPlan objects")
        return plans

    def _read_planned_data(
        self, plan: LoadPlan, planner: LoadPlanner, storage_data: dict[Any, Any]
    ) -> Future[None]:
        per_file: dict[str | os.PathLike[str], list[tuple[Any, _StorageInfo]]] = {}
        for read_item in plan.items:
            storage_info = storage_data[read_item.storage_index]
            if not isinstance(storage_info, _StorageInfo):
                raise TypeError(
                    f"checkpoint storage entry has invalid type for {read_item.storage_index}"
                )
            per_file.setdefault(storage_info.relative_path, []).append(
                (read_item, storage_info)
            )

        for relative_path, requests in per_file.items():
            path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(path, "rb") as stream:
                for read_item, storage_info in requests:
                    file_slice = self._slice_file(stream, storage_info)
                    transformed = self.transforms.transform_load_stream(
                        read_item,
                        storage_info.transform_descriptors or (),
                        file_slice,
                    )
                    try:
                        if read_item.type is LoadItemType.BYTE_IO:
                            value = io.BytesIO(transformed.read(-1))
                            value.seek(0)
                            planner.load_bytes(read_item, value)
                            continue
                        if getattr(transformed, "seekable", lambda: False)():
                            seekable = transformed
                        else:
                            seekable = io.BytesIO(transformed.read(-1))
                            seekable.seek(0)
                        tensor = tp.load(seekable, map_location="cpu")
                        if not isinstance(tensor, tp.Tensor):
                            raise TypeError(
                                f"checkpoint entry {read_item.storage_index.fqn} is not a tensor"
                            )
                        tensor = _narrow_tensor(
                            tensor,
                            read_item.storage_offsets,
                            read_item.lengths,
                        )
                        target_tensor = planner.resolve_tensor(read_item).detach()
                        if tuple(target_tensor.shape) != tuple(tensor.shape):
                            raise AssertionError(
                                f"request {read_item.storage_index} has shape "
                                f"{tuple(tensor.shape)}, expected {tuple(target_tensor.shape)}"
                            )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(read_item, target_tensor)
                    finally:
                        if transformed is not file_slice:
                            try:
                                transformed.close()
                            except (AttributeError, OSError, ValueError):
                                pass

        future: Future[None] = Future()
        future.set_result(None)
        return future

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        if not isinstance(plan, LoadPlan):
            raise TypeError("plan must be a LoadPlan")
        return self._read_planned_data(plan, planner, self.storage_data)

    @property
    def checkpoint_id(self) -> str | os.PathLike[str]:
        return self.path
