from __future__ import annotations

import os
import pickle
import tempfile
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any

from .metadata import Metadata, StorageMeta
from ._nested_dict import flatten_state_dict, unflatten_state_dict
from .planner import LoadItemType, LoadPlan, SavePlan, SavePlanner, WriteItemType
from .storage import WriteResult

__all__ = ["FileSystemWriter", "FileSystemReader", "StorageReader", "StorageWriter"]

_METADATA_FILE = ".metadata"
_DATA_FILE_FORMAT = "__{0}_0.distcp"
_BYTES_MARKER = "__tensorplay_checkpoint_bytes__"
_CHUNKS_MARKER = "__tensorplay_checkpoint_chunks__"


class StorageWriter(ABC):
    @abstractmethod
    def set_up_storage_writer(self, is_coordinator: bool) -> None: ...

    @abstractmethod
    def write_data(self, state_dict: dict[str, Any]) -> None: ...

    @abstractmethod
    def finish(self, metadata: Any) -> Any: ...

    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None:
        del checkpoint_id

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        return plans

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return isinstance(checkpoint_id, (str, os.PathLike)) and bool(str(checkpoint_id))

    def storage_meta(self) -> StorageMeta | None:
        return None


class StorageReader(ABC):
    @abstractmethod
    def read_metadata(self) -> Any: ...

    @abstractmethod
    def read_data(self, plan: Any, state_dict: dict[str, Any]) -> dict[str, Any]: ...

    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None:
        del checkpoint_id

    def set_up_storage_reader(self, metadata: Any, is_coordinator: bool, *args: Any, **kwargs: Any) -> None:
        del metadata, is_coordinator, args, kwargs

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        return plans

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return isinstance(checkpoint_id, (str, os.PathLike)) and bool(str(checkpoint_id))


def _atomic_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class FileSystemWriter(StorageWriter):
    """Write one transactionally committed checkpoint directory."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        single_file_per_rank: bool = True,
        thread_count: int = 1,
        overwrite: bool = True,
    ) -> None:
        del single_file_per_rank
        if int(thread_count) <= 0:
            raise ValueError("thread_count must be positive")
        self.path = Path(path)
        self.overwrite = bool(overwrite)
        self.thread_count = int(thread_count)
        self._data_name: str | None = None
        self._metadata: Metadata | None = None

    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None:
        if checkpoint_id is not None:
            self.path = Path(checkpoint_id)
        self._data_name = None
        self._metadata = None

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        del is_coordinator
        self.path.mkdir(parents=True, exist_ok=True)
        if not self.overwrite and (self.path / _METADATA_FILE).exists():
            raise FileExistsError(f"checkpoint already exists at {self.path}")

    def storage_meta(self) -> StorageMeta:
        return StorageMeta(checkpoint_id=self.path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return isinstance(checkpoint_id, (str, os.PathLike)) and bool(str(checkpoint_id))

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        if not isinstance(plan, SavePlan):
            raise TypeError("plan must be a SavePlan")
        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        if not isinstance(plans, list) or not all(
            isinstance(plan, SavePlan) for plan in plans
        ):
            raise TypeError("plans must be a list of SavePlan objects")
        return plans

    def _write_planned_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        plan = self.prepare_local_plan(plan)
        plan = planner.finish_plan(plan)
        flat_payload: dict[str, Any] = {}
        results: list[WriteResult] = []
        for item in plan.items:
            value = planner.resolve_data(item)
            key = item.index.fqn
            if item.type is WriteItemType.BYTE_IO:
                if not hasattr(value, "getvalue"):
                    raise TypeError("byte write items must resolve to a byte stream")
                raw = bytes(value.getvalue())
                flat_payload[key] = {_BYTES_MARKER: raw}
                size = len(raw)
            else:
                if not hasattr(value, "detach"):
                    raise TypeError("tensor write items must resolve to tensors")
                if hasattr(value, "detach"):
                    value = value.detach().clone()
                chunk = item.tensor_data.chunk if item.tensor_data is not None else None
                entry = {
                    "offsets": tuple(chunk.offsets) if chunk is not None else (),
                    "value": value,
                }
                existing = flat_payload.get(key)
                if isinstance(existing, dict) and _CHUNKS_MARKER in existing:
                    existing[_CHUNKS_MARKER].append(entry)
                elif existing is None:
                    flat_payload[key] = {_CHUNKS_MARKER: [entry]}
                else:
                    flat_payload[key] = {
                        _CHUNKS_MARKER: [
                            {"offsets": (), "value": existing},
                            entry,
                        ]
                    }
                size = int(getattr(value, "numel", lambda: 0)()) * int(
                    getattr(getattr(value, "dtype", None), "itemsize", 1)
                )
            results.append(
                WriteResult(
                    index=item.index,
                    size_in_bytes=size,
                    storage_data={
                        "file": self._data_name,
                        "offsets": tuple(item.tensor_data.chunk.offsets)
                        if item.tensor_data is not None
                        else (),
                    },
                )
            )
        mappings = getattr(planner, "mappings", None)
        payload = (
            unflatten_state_dict(flat_payload, mappings)
            if mappings
            else flat_payload
        )
        self.path.mkdir(parents=True, exist_ok=True)
        self._data_name = f"__0_{uuid.uuid4().hex}.distcp"
        for result in results:
            result.storage_data["file"] = self._data_name
        _atomic_dump(self.path / self._data_name, payload)
        future: Future[list[WriteResult]] = Future()
        future.set_result(results)
        return future

    def write_data(
        self, state_dict: dict[str, Any] | SavePlan, planner: SavePlanner | None = None
    ) -> Future[list[WriteResult]] | None:
        if isinstance(state_dict, SavePlan):
            if planner is None:
                raise TypeError("planner is required when writing a SavePlan")
            return self._write_planned_data(state_dict, planner)
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dictionary")
        self.path.mkdir(parents=True, exist_ok=True)
        if planner is not None:
            local_plan = planner.create_local_plan()
            local_plan = planner.finish_plan(local_plan)
            flat_payload: dict[str, Any] = {}
            for item in local_plan.items:
                value = planner.resolve_data(item)
                if item.type is WriteItemType.BYTE_IO:
                    if not hasattr(value, "getvalue"):
                        raise TypeError("byte write items must resolve to a byte stream")
                    flat_payload[item.index.fqn] = {
                        _BYTES_MARKER: value.getvalue(),
                    }
                else:
                    entry = {
                        "offsets": tuple(item.tensor_data.chunk.offsets)
                        if item.tensor_data is not None
                        else (),
                        "value": value.detach().clone()
                        if hasattr(value, "detach")
                        else value,
                    }
                    existing = flat_payload.get(item.index.fqn)
                    if isinstance(existing, dict) and _CHUNKS_MARKER in existing:
                        existing[_CHUNKS_MARKER].append(entry)
                    elif existing is None:
                        flat_payload[item.index.fqn] = {_CHUNKS_MARKER: [entry]}
                    else:
                        flat_payload[item.index.fqn] = {
                            _CHUNKS_MARKER: [
                                {"offsets": (), "value": existing},
                                entry,
                            ]
                        }
            mappings = getattr(planner, "mappings", None)
            state_dict = (
                unflatten_state_dict(flat_payload, mappings)
                if mappings
                else flat_payload
            )
        self._data_name = f"__0_{uuid.uuid4().hex}.distcp"
        _atomic_dump(self.path / self._data_name, state_dict)

    def finish(
        self, metadata: Metadata | Any, results: list[list[WriteResult]] | None = None
    ) -> Metadata | Any:
        del results
        if self._data_name is None:
            raise RuntimeError("write_data must complete before finish")
        if isinstance(metadata, Metadata):
            storage_data = dict(metadata.storage_data or {})
            storage_data["data_file"] = self._data_name
            if metadata.storage_meta is None:
                storage_meta = StorageMeta(checkpoint_id=self.path)
            elif metadata.storage_meta.checkpoint_id is None:
                storage_meta = replace(metadata.storage_meta, checkpoint_id=self.path)
            else:
                storage_meta = metadata.storage_meta
            committed = replace(
                metadata,
                storage_data=storage_data,
                storage_meta=storage_meta,
                version=metadata.version or "tp-1",
            )
        else:
            committed = metadata
        _atomic_dump(self.path / _METADATA_FILE, committed)
        self._metadata = committed if isinstance(committed, Metadata) else None
        return committed


class FileSystemReader(StorageReader):
    """Read checkpoint transactions written by :class:`FileSystemWriter`."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._metadata: Any = None

    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None:
        if checkpoint_id is not None:
            self.path = Path(checkpoint_id)
        self._metadata = None

    def read_metadata(self) -> Any:
        meta_path = self.path / _METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"checkpoint metadata not found at {meta_path}")
        self._metadata = _load_pickle(meta_path)
        return self._metadata

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        del is_coordinator, args, kwargs
        self._metadata = metadata

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return isinstance(checkpoint_id, (str, os.PathLike)) and bool(str(checkpoint_id))

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

    def read_data(
        self,
        plan: Any,
        state_dict: dict[str, Any] | Any,
        planner: Any = None,
    ) -> dict[str, Any] | Future[None]:
        canonical_call = (
            isinstance(plan, LoadPlan)
            and planner is None
            and not isinstance(state_dict, dict)
            and hasattr(state_dict, "resolve_tensor")
        )
        if canonical_call:
            planner = state_dict
            state_dict = getattr(planner, "original_state_dict", None)
            if state_dict is None:
                state_dict = getattr(planner, "state_dict", {})
        metadata = self._metadata if self._metadata is not None else self.read_metadata()
        data_name = None
        if isinstance(metadata, Metadata) and isinstance(metadata.storage_data, dict):
            data_name = metadata.storage_data.get("data_file")
        elif isinstance(metadata, dict):
            data_name = metadata.get("data_file")
        data_path = self.path / str(data_name or _DATA_FILE_FORMAT.format(0))
        if not data_path.exists():
            candidates = sorted(self.path.glob("__0_*.distcp"))
            if not candidates:
                raise FileNotFoundError(f"checkpoint data not found at {data_path}")
            data_path = candidates[-1]
        loaded = _load_pickle(data_path)
        if not isinstance(loaded, dict):
            raise RuntimeError("checkpoint data must contain a dictionary")
        if isinstance(plan, LoadPlan) and planner is not None:
            plan = self.prepare_local_plan(plan)
            flat_loaded = _flatten_storage_payload(loaded)
            for read_item in plan.items:
                value = flat_loaded.get(read_item.storage_index.fqn)
                if value is None and read_item.storage_index.fqn not in flat_loaded:
                    raise KeyError(
                        f"checkpoint is missing {read_item.storage_index.fqn}"
                    )
                if read_item.type is LoadItemType.BYTE_IO:
                    if not isinstance(value, dict) or _BYTES_MARKER not in value:
                        raise RuntimeError(
                            f"checkpoint entry {read_item.storage_index.fqn} is not byte data"
                        )
                    import io

                    stream = io.BytesIO(value[_BYTES_MARKER])
                    planner.load_bytes(read_item, stream)
                    continue
                source = _select_storage_chunk(
                    value, read_item.storage_index.offset
                )
                if not hasattr(source, "narrow") or not hasattr(source, "shape"):
                    raise TypeError(
                        f"checkpoint entry {read_item.storage_index.fqn} is not a tensor"
                    )
                if read_item.storage_offsets or read_item.lengths:
                    for dim, (offset, length) in enumerate(
                        zip(read_item.storage_offsets, read_item.lengths)
                    ):
                        source = source.narrow(dim, int(offset), int(length))
                target = planner.resolve_tensor(read_item)
                target.copy_(source)
                planner.commit_tensor(read_item, target)
            if canonical_call:
                future: Future[None] = Future()
                future.set_result(None)
                return future
            return state_dict
        return loaded


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _flatten_storage_payload(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if (
        isinstance(value, dict)
        and _BYTES_MARKER not in value
        and _CHUNKS_MARKER not in value
    ):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_storage_payload(child, child_prefix))
        return result
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            result.update(_flatten_storage_payload(child, child_prefix))
        return result
    if prefix:
        result[prefix] = value
    return result


def _select_storage_chunk(value: Any, offset: Any) -> Any:
    if not isinstance(value, dict) or _CHUNKS_MARKER not in value:
        return value
    entries = value[_CHUNKS_MARKER]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("checkpoint tensor chunk list is empty")
    requested = tuple(offset) if offset is not None else None
    if requested is not None:
        for entry in entries:
            if tuple(entry.get("offsets", ())) == requested:
                return entry["value"]
    if len(entries) == 1:
        return entries[0]["value"]
    raise KeyError(f"checkpoint tensor chunk {requested!r} is missing")
