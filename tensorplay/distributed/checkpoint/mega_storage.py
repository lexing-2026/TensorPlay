"""MEGA storage backends for tensorplay.distributed.checkpoint.

(``HuggingFaceStorageWriter/Reader`` — Xet-backed chunk storage), tensorplay
ships the MEGA equivalents: shards are written in tp's native ``.mega``
format plus a ``model.mega.index.json`` weight-map that interoperates with
``model.safetensors.index.json`` files, and paths of the form

    mega://<repo-id>[@<revision>]/<path/in/repo>
    mega://buckets/<bucket-id>/<path/in/bucket>

are transparently routed through ``megatensors``' ``MegaFileSystem``
(also Xet-backed), so the two ecosystems interoperate at the storage layer.

The writer consumes save plans and stores their resolved items in MEGA shards.
The reader consumes load plans and fills planner-owned destinations from those
shards.
"""
import json
import io
import os
import pickle
import tempfile
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Union

import tensorplay as tp

from .filesystem import FileSystemReader, FileSystemWriter
from .metadata import Metadata, MetadataIndex, StorageMeta
from .planner import LoadItemType, LoadPlan, LoadPlanner, SavePlan, SavePlanner, WriteItemType
from .storage import WriteResult

__all__ = ["MegaStorageWriter", "MegaStorageReader"]

_META_FN = "model.mega.index.json"
_SUFFIX = ".mega"
_CHECKPOINT_METADATA = ".metadata"


@dataclass(frozen=True)
class _MegaStorageInfo:
    relative_path: str


def _is_mega_uri(path: Union[str, os.PathLike]) -> bool:
    return str(path).startswith("mega://")


def _get_mega_fs(path: str):
    try:
        from megatensors._hub.mega_file_system import MegaFileSystem
    except ImportError as err:  # pragma: no cover - optional dependency
        raise RuntimeError(
            f"path '{path}' uses the mega:// protocol but megatensors is not "
            "installed (pip install megatensors)"
        ) from err
    return MegaFileSystem()


class _PathResolver:
    """Uniform local-dir / mega:// handling for the writer and reader."""

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self.raw = str(path)
        self.remote = _is_mega_uri(self.raw)
        self.fs = None
        if self.remote:
            self.fs = _get_mega_fs(self.raw)

    def join(self, name: str) -> str:
        return self.raw.rstrip("/") + "/" + name

    def exists(self, name: str) -> bool:
        target = self.join(name)
        if self.remote:
            return bool(self.fs.exists(target))
        return (Path(self.raw) / name).exists()

    def put_bytes(self, name: str, data: bytes) -> None:
        target = self.join(name)
        if self.remote:
            with self.fs.open(target, "wb") as f:
                f.write(data)
        else:
            (Path(self.raw) / name).write_bytes(data)

    def get_bytes(self, name: str) -> bytes:
        target = self.join(name)
        if self.remote:
            with self.fs.open(target, "rb") as f:
                return f.read()
        return (Path(self.raw) / name).read_bytes()

    def stage_and_load(self, name: str):
        """Return the object stored in shard ``name`` (remote-aware)."""
        target = self.join(name)
        if self.remote:
            fd, tmp = tempfile.mkstemp(suffix=_SUFFIX)
            os.close(fd)
            try:
                self.fs.get_file(target, tmp)
                return tp.load(tmp)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        return tp.load(target)

    def write_shard(self, name: str, payload: dict[str, Any]) -> int:
        """Persist ``payload`` as shard ``name``; returns its byte size."""
        target = self.join(name)
        if self.remote:
            fd, tmp = tempfile.mkstemp(suffix=_SUFFIX)
            os.close(fd)
            try:
                tp.save(payload, tmp, metadata={"dcp": True, "format": "mega"})
                self.fs.put_file(tmp, target)
                return os.path.getsize(tmp)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        final = Path(self.raw) / name
        tp.save(payload, str(final), metadata={"dcp": True, "format": "mega"})
        return os.path.getsize(final)


def _gen_file_name(index: int, highest: int) -> str:
    if highest > 1:
        return f"model-{index:05d}-of-{highest:05d}{_SUFFIX}"
    return f"model{_SUFFIX}"


class MegaStorageWriter(FileSystemWriter):
    """Writes MEGA-format shards (``model[-N-of-M].mega``) plus
    ``model.mega.index.json``; accepts plain directories and ``mega://`` URIs.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        fqn_to_index_mapping: dict[str, int] | None = None,
    ) -> None:
        super().__init__(str(path))
        # Bypass FileSystemWriter.__init__'s local-only assumptions.
        self.path = str(path)
        self.resolver = _PathResolver(path)
        self.fqn_to_index_mapping = fqn_to_index_mapping
        self.weight_map: Dict[str, str] = {}
        self.total_size = 0
        self._written_names: set[str] = set()

    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        del args
        self._rank = int(kwargs.get("rank", 0))
        self._use_collectives = bool(kwargs.get("use_collectives", True))
        self._committed = False
        if is_coordinator and not self.resolver.remote:
            Path(self.path).mkdir(parents=True, exist_ok=True)

    def reset(self, checkpoint_id=None) -> None:
        if checkpoint_id is not None:
            self.path = str(checkpoint_id)
            self.resolver = _PathResolver(checkpoint_id)
        self.weight_map = {}
        self.total_size = 0
        self._written_names.clear()
        self._committed = False

    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        if not isinstance(plan, SavePlan):
            raise TypeError("plan must be a SavePlan")
        mapping = self.fqn_to_index_mapping
        buckets: dict[int, dict[str, Any]] = {}
        items_by_bucket: dict[int, list[Any]] = {}
        for item in plan.items:
            fqn = item.index.fqn
            idx = mapping.get(fqn, 1) if mapping is not None else 1
            if fqn in buckets.setdefault(idx, {}):
                raise ValueError(f"multiple write items for {fqn!r} share one mega shard")
            value = planner.resolve_data(item)
            if item.type is WriteItemType.BYTE_IO:
                if not hasattr(value, "getbuffer"):
                    raise TypeError("byte write items require a byte stream")
                value = value.getvalue()
            buckets[idx][fqn] = value
            items_by_bucket.setdefault(idx, []).append(item)

        highest = max(buckets, default=1)
        results: list[WriteResult] = []
        for index in sorted(buckets):
            fname = _gen_file_name(index, highest)
            size = self.resolver.write_shard(fname, buckets[index])
            self._written_names.add(fname)
            for item in items_by_bucket[index]:
                fqn = item.index.fqn
                self.weight_map[fqn] = fname
                results.append(
                    WriteResult(
                        index=item.index,
                        size_in_bytes=size,
                        storage_data=_MegaStorageInfo(fname),
                    )
                )
            self.total_size += size
        future: Future[list[WriteResult]] = Future()
        future.set_result(results)
        return future

    def finish(
        self, metadata: Metadata, results: list[list[WriteResult]]
    ) -> None:
        index_doc = {
            "metadata": {"total_size": self.total_size},
            "weight_map": self.weight_map,
        }
        self.resolver.put_bytes(_META_FN, json.dumps(index_doc, indent=2).encode())
        storage_data: dict[MetadataIndex, _MegaStorageInfo] = {}
        for rank_results in results:
            storage_data.update(
                {result.index: result.storage_data for result in rank_results}
            )
        metadata.storage_data = storage_data
        metadata.storage_meta = metadata.storage_meta or StorageMeta()
        if metadata.storage_meta.checkpoint_id is None:
            metadata.storage_meta = replace(
                metadata.storage_meta, checkpoint_id=self.path
            )
        metadata.version = metadata.version or "1.0.0"
        self.resolver.put_bytes(
            _CHECKPOINT_METADATA,
            pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL),
        )
        self._committed = True

    def abort(self) -> None:
        if self._committed:
            return
        for name in tuple(self._written_names):
            try:
                if self.resolver.remote:
                    self.resolver.fs.rm(self.resolver.join(name))
                else:
                    (Path(self.path) / name).unlink()
            except (FileNotFoundError, OSError):
                pass
        self._written_names.clear()


class MegaStorageReader(FileSystemReader):
    """Reads MEGA checkpoints produced by :class:`MegaStorageWriter`
    (``.mega`` shards described by ``model.mega.index.json``)."""

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__(str(path))
        self.path = str(path)
        self.resolver = _PathResolver(path)

    def reset(self, checkpoint_id=None) -> None:  # noqa: D102 - see FileSystemReader
        if checkpoint_id is not None:
            self.path = str(checkpoint_id)
            self.resolver = _PathResolver(checkpoint_id)
        self._metadata = None
        self.storage_data = {}

    def read_metadata(self) -> Metadata:
        if not self.resolver.exists(_CHECKPOINT_METADATA):
            raise FileNotFoundError(
                self.resolver.join(_CHECKPOINT_METADATA)
            )
        metadata = pickle.loads(self.resolver.get_bytes(_CHECKPOINT_METADATA))
        if not isinstance(metadata, Metadata):
            raise TypeError("checkpoint metadata must be a Metadata object")
        storage_meta = metadata.storage_meta or StorageMeta()
        metadata.storage_meta = replace(storage_meta, load_id=self.load_id)
        self._metadata = metadata
        self.storage_data = metadata.storage_data
        if not isinstance(self.storage_data, dict):
            raise AssertionError("metadata.storage_data must be a dictionary")
        return metadata

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        if not isinstance(plan, LoadPlan):
            raise TypeError("plan must be a LoadPlan")
        if self._metadata is None:
            self.read_metadata()
        per_file: dict[str, list[Any]] = {}
        for request in plan.items:
            storage_info = self.storage_data[request.storage_index]
            if not isinstance(storage_info, _MegaStorageInfo):
                raise TypeError(
                    f"checkpoint storage entry has invalid type for {request.storage_index}"
                )
            per_file.setdefault(storage_info.relative_path, []).append(request)

        for relative_path, requests in per_file.items():
            loaded = self.resolver.stage_and_load(relative_path)
            if not isinstance(loaded, dict):
                raise TypeError("checkpoint shard must contain a dictionary")
            for request in requests:
                fqn = request.storage_index.fqn
                if fqn not in loaded:
                    raise KeyError(f"checkpoint is missing {fqn}")
                value = loaded[fqn]
                if request.type is LoadItemType.BYTE_IO:
                    if not isinstance(value, bytes):
                        raise TypeError(f"checkpoint entry {fqn} is not byte data")
                    stream = io.BytesIO(value)
                    planner.load_bytes(request, stream)
                    continue
                if not isinstance(value, tp.Tensor):
                    raise TypeError(f"checkpoint entry {fqn} is not a tensor")
                for dimension, (offset, length) in enumerate(
                    zip(request.storage_offsets, request.lengths)
                ):
                    if int(length):
                        value = value.narrow(dimension, int(offset), int(length))
                target = planner.resolve_tensor(request).detach()
                if tuple(target.shape) != tuple(value.shape):
                    raise AssertionError(
                        f"request {request.storage_index} has shape "
                        f"{tuple(value.shape)}, expected {tuple(target.shape)}"
                    )
                target.copy_(value)
                planner.commit_tensor(request, target)

        future: Future[None] = Future()
        future.set_result(None)
        return future
