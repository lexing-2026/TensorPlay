from __future__ import annotations

import io
import json
import os
import queue
import threading
from concurrent.futures import Future
from dataclasses import replace
from typing import Any

import tensorplay as tp

from ._consolidate_hf_safetensors import consolidate_safetensors_files
from ._hf_utils import (
    CUSTOM_METADATA_KEY,
    DCP_VERSION_KEY,
    SAVED_OFFSETS_KEY,
    SHARDED_DIR_NAME,
    _HFStorageInfo,
    _gen_file_name,
    _metadata_fn,
)
from .filesystem import FileSystemReader, FileSystemWriter, SerializationFormat
from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from .planner import LoadItemType, LoadPlan, LoadPlanner, SavePlan, SavePlanner, WriteItemType
from .storage import WriteResult

__all__ = ["HuggingFaceStorageWriter", "HuggingFaceStorageReader"]


def _tensor_num_bytes(value: Any) -> int:
    if not isinstance(value, tp.Tensor):
        return 0
    return int(value.numel()) * int(getattr(value.dtype, "itemsize", 1))


def _load_native(path: Any) -> dict[str, Any]:
    value = tp.load(path)
    if not isinstance(value, dict):
        raise TypeError("checkpoint contents must be a dictionary")
    return value


class HuggingFaceStorageWriter(FileSystemWriter):
    def __init__(
        self,
        path: str | os.PathLike[str],
        fqn_to_index_mapping: dict[str, int] | None = None,
        thread_count: int = 1,
        save_distributed: bool = False,
        enable_consolidation: bool = False,
        thread_count_consolidation: int = 1,
    ) -> None:
        super().__init__(
            path=path,
            thread_count=thread_count,
            serialization_format=SerializationFormat.SAFETENSORS,
        )
        self.fqn_to_index_mapping = fqn_to_index_mapping
        self.thread_count = int(thread_count)
        self.save_distributed = bool(save_distributed)
        self.enable_consolidation = bool(enable_consolidation)
        self.thread_count_consolidation = int(thread_count_consolidation)
        self.consolidated_output_path = self.path if enable_consolidation else None
        if self.enable_consolidation:
            self.path = self.fs.concat_path(self.path, SHARDED_DIR_NAME)
        self.weight_map: dict[str, str] = {}
        self.total_size = 0
        if self.thread_count <= 0 or self.thread_count_consolidation <= 0:
            raise ValueError("thread counts must be positive")

    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None:
        root = self.consolidated_output_path if checkpoint_id is None else checkpoint_id
        super().reset(root)
        if self.enable_consolidation:
            self.consolidated_output_path = self.fs.init_path(root)
            self.path = self.fs.concat_path(self.consolidated_output_path, SHARDED_DIR_NAME)
        self.weight_map.clear()
        self.total_size = 0

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        prepared: list[SavePlan] = []
        for index, plan in enumerate(plans, start=1):
            storage_data: dict[str, Any] = {}
            if self.fqn_to_index_mapping is not None:
                storage_data["fqn_to_index_mapping"] = self.fqn_to_index_mapping
            if self.save_distributed:
                storage_data["shard_index"] = index
            prepared.append(replace(plan, storage_data=storage_data))
        return prepared

    def _split_by_storage_plan(
        self, storage_plan: dict[str, int] | None, items: list[Any]
    ) -> dict[int, list[Any]]:
        if storage_plan is None:
            return {1: list(items)}
        buckets: dict[int, list[Any]] = {}
        for item in items:
            index = int(storage_plan.get(item.index.fqn, 1))
            buckets.setdefault(index, []).append(item)
        return buckets

    def _write_native_bucket(
        self, file_path: Any, file_name: str, items: list[Any], planner: SavePlanner
    ) -> list[WriteResult]:
        payload: dict[str, Any] = {}
        sharding: dict[str, dict[str, list[int]]] = {}
        results: list[WriteResult] = []
        for item in items:
            value = planner.resolve_data(item)
            if item.type is WriteItemType.BYTE_IO:
                raise TypeError("safetensors does not store byte streams")
            if not isinstance(value, tp.Tensor):
                raise TypeError("tensor write items require a tensor")
            if item.index.fqn in payload:
                raise ValueError(f"multiple chunks for {item.index.fqn!r} share one file")
            payload[item.index.fqn] = value.detach().to(device="cpu")
            size = _tensor_num_bytes(payload[item.index.fqn])
            chunk = item.tensor_data.chunk if item.tensor_data is not None else None
            sharding[item.index.fqn] = {
                SAVED_OFFSETS_KEY: list(chunk.offsets) if chunk is not None else [0] * value.dim()
            }
            results.append(
                WriteResult(
                    index=item.index,
                    size_in_bytes=size,
                    storage_data={"relative_path": file_name, "length": size},
                )
            )
            self.weight_map[item.index.fqn] = file_name
            self.total_size += size
        self.fs.mkdir(self.path)
        metadata = {
            DCP_VERSION_KEY: "1.0",
            CUSTOM_METADATA_KEY: json.dumps(sharding),
        }
        from tensorplay.serialization.archive import write_safetensors_file

        with self.fs.create_stream(file_path, "wb") as stream:
            write_safetensors_file(stream, payload, metadata=metadata)
        return results

    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        if not isinstance(plan, SavePlan):
            raise TypeError("plan must be a SavePlan")
        storage_plan_data: dict[str, Any] = plan.storage_data
        storage_plan = storage_plan_data.get("fqn_to_index_mapping")
        shard_index = storage_plan_data.get("shard_index")
        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest = max(storage_plan.values()) if storage_plan else 1
        results: list[WriteResult] = []
        for index, items in sorted(buckets.items()):
            name = _gen_file_name(index, highest, shard_index)
            results.extend(
                self._write_native_bucket(self.fs.concat_path(self.path, name), name, items, planner)
            )
        future: Future[list[WriteResult]] = Future()
        future.set_result(results)
        return future

    def finish(
        self,
        metadata: Metadata,
        results: list[list[WriteResult]],
    ) -> None:
        if self.save_distributed and not self.enable_consolidation:
            return
        output_path = self.consolidated_output_path or self.path
        self.fs.mkdir(output_path)
        if self.save_distributed:
            mapping = self.fqn_to_index_mapping or dict.fromkeys(
                metadata.state_dict_metadata, 1
            )
            consolidate_safetensors_files(
                self.path,
                output_path,
                mapping,
                self.thread_count_consolidation,
            )
        else:
            index_doc = {
                "metadata": {"total_size": self.total_size},
                "weight_map": self.weight_map,
            }
            with self.fs.create_stream(self.fs.concat_path(output_path, _metadata_fn), "w") as stream:
                json.dump(index_doc, stream, indent=2)
        del metadata
        del results

    @property
    def metadata_path(self) -> str:
        return _metadata_fn


class HuggingFaceStorageReader(FileSystemReader):
    def __init__(self, path: str | os.PathLike[str], thread_count: int = 1) -> None:
        super().__init__(path)
        self.thread_count = int(thread_count)
        self._weight_map: dict[str, str] = {}
        self._native_state: dict[str, Any] | None = None
        if self.thread_count <= 0:
            raise ValueError("thread_count must be positive")

    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None:
        super().reset(checkpoint_id)
        self._weight_map.clear()
        self._native_state = None

    def _read_index(self) -> dict[str, Any]:
        path = self.fs.concat_path(self.path, _metadata_fn)
        if not self.fs.exists(path):
            return {}
        with self.fs.create_stream(path, "r") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}

    def _list_safetensors(self) -> list[str]:
        files = sorted(
            path
            for path in self.fs.ls(self.path)
            if str(path).endswith(".safetensors")
        )
        if files:
            return files
        sharded_path = self.fs.concat_path(self.path, SHARDED_DIR_NAME)
        if not self.fs.exists(sharded_path):
            return []
        return sorted(
            path
            for path in self.fs.ls(sharded_path)
            if str(path).endswith(".safetensors")
        )

    def _resolve_data_file(self, name: str) -> str | os.PathLike[str]:
        path = self.fs.concat_path(self.path, name)
        if self.fs.exists(path):
            return path
        return self.fs.concat_path(
            self.fs.concat_path(self.path, SHARDED_DIR_NAME), name
        )

    def _load_state(self) -> dict[str, Any]:
        if self._native_state is not None:
            return self._native_state
        index = self._read_index()
        files = sorted({str(value) for value in index.get("weight_map", {}).values()})
        if not files:
            files = [os.path.basename(path) for path in self._list_safetensors()]
        loaded: dict[str, Any] = {}
        for name in files:
            loaded.update(_load_native(self._resolve_data_file(name)))
        self._native_state = loaded
        self._weight_map = {
            str(key): str(value)
            for key, value in index.get("weight_map", {}).items()
        }
        return loaded

    def _process_read_request(
        self, f: dict[str, Any], req: Any, planner: LoadPlanner
    ) -> None:
        key = req.storage_index.fqn
        if key not in f:
            raise KeyError(f"checkpoint is missing {key}")
        value = f[key]
        if req.type is LoadItemType.BYTE_IO:
            if isinstance(value, bytes):
                stream = io.BytesIO(value)
            else:
                stream = io.BytesIO()
                import pickle

                pickle.dump(value, stream, pickle.HIGHEST_PROTOCOL)
                stream.seek(0)
            planner.load_bytes(req, stream)
            return
        if not isinstance(value, tp.Tensor):
            raise TypeError(f"checkpoint value {key} is not a tensor")
        for dimension, (offset, length) in enumerate(
            zip(req.storage_offsets, req.lengths)
        ):
            value = value.narrow(dimension, int(offset), int(length))
        target = planner.resolve_tensor(req).detach()
        if tuple(target.shape) != tuple(value.shape):
            raise AssertionError(
                f"request {req.storage_index} mismatch sizes {target.shape} vs {value.shape}"
            )
        target.copy_(value)
        planner.commit_tensor(req, target)

    def _read_files_from_queue(
        self,
        file_queue: queue.Queue,
        result_queue: queue.Queue,
        planner: LoadPlanner,
    ) -> None:
        try:
            while True:
                file_name, requests = file_queue.get_nowait()
                values = _load_native(file_name)
                for request in requests:
                    self._process_read_request(values, request, planner)
                result_queue.put(True)
        except queue.Empty:
            return

    def read_data(
        self, plan: LoadPlan, planner: LoadPlanner
    ) -> Future[None]:
        if not isinstance(plan, LoadPlan):
            raise TypeError("plan must be a LoadPlan")
        per_file: dict[str, list[Any]] = {}
        for request in plan.items:
            info = self.storage_data[request.storage_index]
            if not isinstance(info, _HFStorageInfo):
                raise TypeError(
                    f"checkpoint storage entry has invalid type for {request.storage_index}"
                )
            per_file.setdefault(str(info.relative_path), []).append(request)
        if self.thread_count <= 1 or len(per_file) <= 1:
            for file_name, requests in per_file.items():
                values = _load_native(file_name)
                for request in requests:
                    self._process_read_request(values, request, planner)
        else:
            file_queue: queue.Queue = queue.Queue()
            result_queue: queue.Queue = queue.Queue()
            for file_name, requests in per_file.items():
                file_queue.put((file_name, requests))
            threads = [
                threading.Thread(
                    target=self._read_files_from_queue,
                    args=(file_queue, result_queue, planner),
                )
                for _ in range(min(self.thread_count, len(per_file)))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            processed = 0
            while True:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break
                processed += 1
            if processed != len(per_file):
                raise RuntimeError("not all safetensors files were processed")
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        del args, kwargs
        state_metadata: dict[str, Any] = {}
        storage_data: dict[MetadataIndex, Any] = {}
        index = self._read_index()
        files = self._list_safetensors()
        if not files:
            self._load_state()
            files = [
                self.fs.concat_path(self.path, name)
                for name in self._weight_map.values()
            ]
        for file_name in sorted(set(files)):
            values = _load_native(file_name)
            try:
                file_info = tp.inspect_checkpoint(file_name)
                extra = file_info.get("metadata", {})
                tensors = file_info.get("tensors", {})
            except (OSError, ValueError, TypeError):
                extra = {}
                tensors = {}
            raw_sharding = extra.get(CUSTOM_METADATA_KEY) if isinstance(extra, dict) else None
            try:
                sharding = json.loads(raw_sharding) if isinstance(raw_sharding, str) else {}
            except json.JSONDecodeError:
                sharding = {}
            for key, value in values.items():
                if not isinstance(value, tp.Tensor):
                    state_metadata[key] = BytesStorageMetadata()
                    continue
                info = tensors.get(key, {}) if isinstance(tensors, dict) else {}
                shape = tuple(int(size) for size in info.get("shape", value.shape))
                offsets = tuple(
                    int(item)
                    for item in sharding.get(key, {}).get(SAVED_OFFSETS_KEY, [0] * len(shape))
                )
                if key not in state_metadata:
                    state_metadata[key] = TensorStorageMetadata(
                        properties=TensorProperties.create_from_tensor(value),
                        size=tuple(size + offset for size, offset in zip(shape, offsets)),
                        chunks=[ChunkStorageMetadata(offsets, shape)],
                    )
                else:
                    tensor_metadata = state_metadata[key]
                    if not isinstance(tensor_metadata, TensorStorageMetadata):
                        raise TypeError(f"checkpoint entry {key!r} changes value type")
                    tensor_metadata.chunks.append(ChunkStorageMetadata(offsets, shape))
                    tensor_metadata.size = tuple(
                        max(old, current + offset)
                        for old, current, offset in zip(tensor_metadata.size, shape, offsets)
                    )
                storage_data[MetadataIndex(key, offsets)] = _HFStorageInfo(
                    relative_path=file_name,
                    shape=shape,
                    dtype=value.dtype,
                )
        return Metadata(
            state_dict_metadata=state_metadata,
            storage_data=storage_data,
            storage_meta=StorageMeta(
                checkpoint_id=self.path,
                load_id=self.load_id,
            ),
            version=str(index.get("metadata", {}).get("version", "1.0.0")),
        )
