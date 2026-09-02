"""MEGA storage backends for tensorplay.distributed.checkpoint.

(``HuggingFaceStorageWriter/Reader`` — Xet-backed chunk storage), tensorplay
ships the MEGA equivalents: shards are written in tp's native ``.mega``
format plus a ``model.mega.index.json`` weight-map that interoperates with
``model.safetensors.index.json`` files, and paths of the form

    mega://<repo-id>[@<revision>]/<path/in/repo>
    mega://buckets/<bucket-id>/<path/in/bucket>

are transparently routed through ``megatensors``' ``MegaFileSystem``
(also Xet-backed), so the two ecosystems interoperate at the storage layer.

Interface follows the package-consolidated contract: the coordinator hands
``write_data`` one merged state dict; ``read_data`` returns the loaded dict
and lets ``state_dict_loader._fill_in_place`` populate the caller's object.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorplay as tp

from .filesystem import FileSystemReader, FileSystemWriter, _atomic_dump
from .metadata import Metadata, StorageMeta

__all__ = ["MegaStorageWriter", "MegaStorageReader"]

_META_FN = "model.mega.index.json"
_SUFFIX = ".mega"


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

    def list_dir(self) -> List[str]:
        if self.remote:
            return [str(x) for x in self.fs.ls(self.raw)]
        base = Path(self.raw)
        return [str(x) for x in base.iterdir()] if base.exists() else []

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

    def write_shard(self, name: str, payload: Dict[str, Any]) -> int:
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


def _flatten(obj, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flat.update(_flatten(v, key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flat.update(_flatten(v, f"{prefix}.{i}" if prefix else str(i)))
    else:
        flat[prefix] = obj
    return flat


class MegaStorageWriter(FileSystemWriter):
    """Writes MEGA-format shards (``model[-N-of-M].mega``) plus
    ``model.mega.index.json``; accepts plain directories and ``mega://`` URIs.
    """

    def __init__(self, path: Union[str, os.PathLike],
                 fqn_to_index_mapping: Optional[Dict[str, int]] = None) -> None:
        super().__init__(str(path))
        # Bypass FileSystemWriter.__init__'s local-only assumptions.
        self.path = str(path)
        self.resolver = _PathResolver(path)
        self.fqn_to_index_mapping = fqn_to_index_mapping
        self.weight_map: Dict[str, str] = {}
        self.total_size = 0

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        if is_coordinator and not self.resolver.remote:
            Path(self.path).mkdir(parents=True, exist_ok=True)

    def reset(self, checkpoint_id=None) -> None:
        if checkpoint_id is not None:
            self.path = str(checkpoint_id)
            self.resolver = _PathResolver(checkpoint_id)
        self.weight_map = {}
        self.total_size = 0

    def write_data(self, state_dict) -> None:
        flat = _flatten(state_dict)
        mapping = self.fqn_to_index_mapping
        buckets: Dict[int, Dict[str, Any]] = {}
        for fqn, value in flat.items():
            idx = mapping.get(fqn, 1) if mapping is not None else 1
            buckets.setdefault(idx, {})[fqn] = value

        highest = max(buckets) if buckets else 1
        for index in sorted(buckets):
            fname = _gen_file_name(index, highest)
            size = self.resolver.write_shard(fname, buckets[index])
            for fqn in buckets[index]:
                self.weight_map[fqn] = fname
            self.total_size += size

    def finish(self, metadata) -> Any:
        index_doc = {
            "metadata": {"total_size": self.total_size},
            "weight_map": self.weight_map,
        }
        self.resolver.put_bytes(_META_FN, json.dumps(index_doc, indent=2).encode())
        if not self.resolver.remote:
            if isinstance(metadata, Metadata):
                storage_data = dict(metadata.storage_data or {})
                storage_data["format"] = "mega"
                if metadata.storage_meta is None:
                    storage_meta = StorageMeta(checkpoint_id=self.path)
                elif metadata.storage_meta.checkpoint_id is None:
                    from dataclasses import replace

                    storage_meta = replace(
                        metadata.storage_meta, checkpoint_id=self.path
                    )
                else:
                    storage_meta = metadata.storage_meta
                metadata = replace(
                    metadata,
                    storage_data=storage_data,
                    storage_meta=storage_meta,
                    version=metadata.version or "tp-1",
                )
            _atomic_dump(Path(self.path) / ".metadata", metadata)
        return metadata


class MegaStorageReader(FileSystemReader):
    """Reads MEGA checkpoints produced by :class:`MegaStorageWriter`
    (``.mega`` shards described by ``model.mega.index.json``)."""

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__(str(path))
        self.path = str(path)
        self.resolver = _PathResolver(path)
        self._index: Optional[Dict[str, Any]] = None

    def reset(self, checkpoint_id=None) -> None:  # noqa: D102 - see FileSystemReader
        if checkpoint_id is not None:
            self.path = str(checkpoint_id)
            self.resolver = _PathResolver(checkpoint_id)
        self._index = None
        self._metadata = None

    def read_metadata(self) -> Dict[str, Any]:
        if self._index is not None:
            return self._index
        if self.resolver.exists(_META_FN):
            doc = json.loads(self.resolver.get_bytes(_META_FN))
        else:
            weight_map: Dict[str, str] = {}
            for full in self.resolver.list_dir():
                name = os.path.basename(full)
                if not name.endswith(_SUFFIX):
                    continue
                for fqn in self.resolver.stage_and_load(name).keys():
                    weight_map[fqn] = name
            doc = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        self._index = doc
        return doc

    def read_data(self, plan, state_dict) -> Dict[str, Any]:
        doc = self.read_metadata()
        weight_map: Dict[str, str] = doc.get("weight_map", {})
        by_file: Dict[str, List[str]] = {}
        for fqn in _flatten_keys(state_dict):
            rel = weight_map.get(fqn)
            if rel is None and len(weight_map) == 1:
                rel = next(iter(weight_map.values()))
            if rel is None:
                continue
            by_file.setdefault(rel, []).append(fqn)

        saved: Dict[str, Any] = {}
        for rel, fqns in by_file.items():
            loaded = self.resolver.stage_and_load(rel)
            flat_loaded = _flatten(loaded)
            for fqn in fqns:
                if fqn in flat_loaded:
                    saved[fqn] = flat_loaded[fqn]
                elif fqn in loaded:
                    saved[fqn] = loaded[fqn]
        return saved


def _flatten_keys(state_dict) -> List[str]:
    return list(_flatten(state_dict).keys()) if isinstance(state_dict, dict) else []
