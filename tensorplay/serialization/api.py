"""Format dispatch and public checkpoint operations."""

from __future__ import annotations

import concurrent.futures
import json
import os
import pickle
import shutil
import tempfile
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import tensorplay as tp

from . import archive as _backend
from .common import (
    _file_position,
    _is_path_like,
    _sniff_format,
    resolve_map_location,
)
from .mega import (
    DEFAULT_ALIGNMENT,
    MEGA_EXTENSION,
    convert_model,
    convert_to_mega,
    load_mega,
    parse_mega_header,
    save_mega,
)
from .policy import (
    LoadEndianness,
    add_safe_globals,
    clear_safe_globals,
    get_crc32_options,
    get_default_load_endianness,
    get_default_mmap_options,
    get_safe_globals,
    get_unsafe_globals_in_checkpoint,
    safe_globals,
    set_crc32_options,
    set_default_load_endianness,
    set_default_mmap_options,
    skip_data,
)


DEFAULT_PROTOCOL = 2
MAGIC_NUMBER = _backend.TORCH_MAGIC_NUMBER
PROTOCOL_VERSION = _backend.TORCH_PROTOCOL_VERSION
MEGA_INDEX_SUFFIX = ".mega.index.json"
StorageType = _backend._StorageType

_PACKAGE_REGISTRY = []


def _probe_torch_stream(path: str | os.PathLike[str]) -> bool:
    try:
        with open(os.fspath(path), "rb") as handle:
            value = _backend._WeightsOnlyUnpickler(
                handle,
                persistent_load=lambda saved_id: None,
                resolve_global=lambda module, name: None,
            ).load()
        return value == MAGIC_NUMBER
    except (OSError, EOFError, pickle.PickleError, ValueError, TypeError):
        return False


def register_package(priority: int, tagger, deserializer):
    if not isinstance(priority, int):
        raise TypeError("priority must be an integer")
    if not callable(tagger) or not callable(deserializer):
        raise TypeError("tagger and deserializer must be callable")
    _PACKAGE_REGISTRY.append((priority, tagger, deserializer))
    _PACKAGE_REGISTRY.sort(key=lambda entry: entry[0])


def location_tag(storage):
    for _priority, tagger, _deserializer in _PACKAGE_REGISTRY:
        value = tagger(storage)
        if value is not None:
            return value
    if isinstance(storage, tp.Tensor):
        return _backend._device_string_of(storage)
    device = getattr(storage, "device", None)
    if device is not None:
        return str(device)
    raise RuntimeError(f"cannot determine checkpoint location for {type(storage).__name__}")


def default_restore_location(storage, location):
    for _priority, _tagger, deserializer in _PACKAGE_REGISTRY:
        value = deserializer(storage, location)
        if value is not None:
            return value
    if isinstance(storage, tp.Tensor):
        return _backend._move_to(storage, location)
    return storage


def normalize_storage_type(storage_type):
    name = getattr(storage_type, "__name__", None) or str(storage_type)
    return StorageType(name)


def storage_to_tensor_type(storage):
    return tp.Tensor


def _validate_pickle_protocol(protocol: int) -> int:
    if not isinstance(protocol, int) or isinstance(protocol, bool):
        raise TypeError("pickle_protocol must be an integer")
    if protocol < 0 or protocol > pickle.HIGHEST_PROTOCOL:
        raise ValueError(
            f"pickle_protocol must be between 0 and {pickle.HIGHEST_PROTOCOL}"
        )
    return protocol


def save(
    obj: Any,
    f,
    pickle_module=pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
    *,
    metadata: Mapping[str, Any] | None = None,
    alignment: int = DEFAULT_ALIGNMENT,
    checksum: str = "crc32",
) -> None:
    protocol = _validate_pickle_protocol(pickle_protocol)
    if pickle_module is None:
        pickle_module = pickle
    if not hasattr(pickle_module, "dump"):
        raise TypeError("pickle_module must provide dump")
    if not isinstance(_use_new_zipfile_serialization, bool):
        raise TypeError("_use_new_zipfile_serialization must be bool")
    if not isinstance(_disable_byteorder_record, bool):
        raise TypeError("_disable_byteorder_record must be bool")

    if not _is_path_like(f):
        if not callable(getattr(f, "write", None)):
            raise AttributeError("checkpoint output must provide write()")
        if not _use_new_zipfile_serialization:
            _backend.write_legacy_torch_file(
                f, obj, pickle_protocol=protocol, pickle_module=pickle_module
            )
        else:
            _backend.write_torch_file(
                f,
                obj,
                pickle_protocol=protocol,
                pickle_module=pickle_module,
                disable_byteorder_record=_disable_byteorder_record,
            )
        return

    filename = os.fspath(f)
    lower = filename.lower()
    if lower.endswith(MEGA_INDEX_SUFFIX):
        raise ValueError("cannot save directly to a shard index")
    if lower.endswith(MEGA_EXTENSION):
        if not _use_new_zipfile_serialization:
            raise ValueError("MEGA output does not support legacy serialization")
        save_mega(
            filename,
            obj,
            metadata=metadata,
            alignment=alignment,
            checksum=checksum,
        )
        return
    if lower.endswith(".safetensors"):
        if not _use_new_zipfile_serialization:
            raise ValueError("safetensors output does not support legacy serialization")
        with open(filename, "wb") as handle:
            _backend.write_safetensors_file(handle, obj, metadata=metadata)
        return
    if lower.endswith((".pt", ".pth")) or not lower.rsplit(".", 1)[-1]:
        with open(filename, "wb") as handle:
            if _use_new_zipfile_serialization:
                _backend.write_torch_file(
                    handle,
                    obj,
                    pickle_protocol=protocol,
                    pickle_module=pickle_module,
                    disable_byteorder_record=_disable_byteorder_record,
                )
            else:
                _backend.write_legacy_torch_file(
                    handle, obj, pickle_protocol=protocol, pickle_module=pickle_module
                )
        return
    raise ValueError(
        f"Supported checkpoint extensions are .mega, .safetensors, .pt, and .pth; "
        f"got {filename!r}"
    )


def _load_index(filename: str, map_location, mmap: bool, pickle_load_args: dict):
    with open(filename, "r", encoding="utf-8") as handle:
        try:
            index = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid shard index: {filename}") from error
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("shard index must contain a weight_map object")
    base = os.path.dirname(filename)
    groups = OrderedDict()
    for key, shard in index["weight_map"].items():
        shard_path = os.fspath(shard)
        if not os.path.isabs(shard_path):
            shard_path = os.path.join(base, shard_path)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"missing shard for {key!r}: {shard_path}")
        groups.setdefault(shard_path, []).append(str(key))

    def read_shard(shard_path):
        return load(
            shard_path,
            map_location=map_location,
            mmap=mmap,
            weights_only=True,
            **pickle_load_args,
        )

    shard_values = {}
    paths = list(groups)
    workers = min(4, len(paths), os.cpu_count() or 1)
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for shard_path, shard_value in zip(paths, pool.map(read_shard, paths)):
                shard_values[shard_path] = shard_value
    else:
        for shard_path in paths:
            shard_values[shard_path] = read_shard(shard_path)

    if mmap:
        from .mega import _ArtifactAware

        result = _ArtifactAware()
        result._mega_keepalive = tuple(shard_values.values())
    else:
        result = OrderedDict()
    for key, shard in index["weight_map"].items():
        shard_path = os.fspath(shard)
        if not os.path.isabs(shard_path):
            shard_path = os.path.join(base, shard_path)
        shard_value = shard_values[shard_path]
        if isinstance(shard_value, Mapping) and key in shard_value:
            result[key] = shard_value[key]
        elif isinstance(shard_value, Mapping):
            raise KeyError(f"shard {shard_path!r} does not contain {key!r}")
        else:
            raise TypeError(f"shard {shard_path!r} must contain a mapping")
    return result


def load(
    f,
    map_location=None,
    pickle_module=None,
    *,
    weights_only: bool = True,
    mmap: bool | None = None,
    **pickle_load_args,
) -> Any:
    if not isinstance(weights_only, bool):
        raise TypeError("weights_only must be bool")
    if not weights_only:
        warnings.warn(
            "TensorPlay checkpoint loading is restricted to registered data types; "
            "weights_only=False does not enable executable object loading.",
            stacklevel=2,
        )
    if mmap is None:
        mmap = False
    if not isinstance(mmap, bool):
        raise TypeError("mmap must be bool or None")
    if pickle_module is None:
        pickle_module = pickle
    if not hasattr(pickle_module, "Unpickler"):
        raise TypeError("pickle_module must provide Unpickler")

    if not _is_path_like(f):
        return _load_stream(f, map_location, mmap=mmap, pickle_load_args=pickle_load_args)

    filename = os.fspath(f)
    lower = filename.lower()
    if lower.endswith(MEGA_INDEX_SUFFIX):
        return _load_index(filename, map_location, mmap, pickle_load_args)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file or directory: {filename!r}")

    with open(filename, "rb") as handle:
        head = handle.read(512)
    kind = _sniff_format(head)
    if kind == "mega":
        return load_mega(filename, map_location, mmap=mmap)
    if kind == "torch_zip":
        with open(filename, "rb") as handle:
            return _backend.read_torch_file(
                handle,
                map_location=map_location,
                mmap=mmap,
                pickle_module=pickle_module,
                **pickle_load_args,
            )
    if kind == "safetensors":
        with open(filename, "rb") as handle:
            return _backend.read_safetensors_file(
                handle, map_location=map_location, mmap=mmap
            )
    if kind == "torch_tar" or _probe_torch_stream(filename):
        with open(filename, "rb") as handle:
            return _backend.read_torch_file(
                handle,
                map_location=map_location,
                mmap=False,
                pickle_module=pickle_module,
                **pickle_load_args,
            )

    if lower.endswith(MEGA_EXTENSION):
        raise ValueError(f"{filename}: not a valid MEGA file")
    if lower.endswith(".safetensors"):
        raise ValueError(f"{filename}: not a valid safetensors file")
    if lower.endswith((".pt", ".pth")):
        raise ValueError(f"{filename}: not a recognized checkpoint")
    raise ValueError(
        f"unrecognized checkpoint: {filename!r}; Supported formats are .mega, "
        ".safetensors, .pt, and .pth"
    )


def _load_stream(fileobj, map_location, *, mmap: bool, pickle_load_args: dict):
    position = _file_position(fileobj)
    head = fileobj.read(512)
    fileobj.seek(position)
    kind = _sniff_format(head)
    if kind == "mega":
        spool_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=MEGA_EXTENSION, delete=False) as spool:
                spool_path = spool.name
                shutil.copyfileobj(fileobj, spool)
            return load_mega(spool_path, map_location, mmap=mmap)
        finally:
            if spool_path is not None:
                try:
                    os.unlink(spool_path)
                except FileNotFoundError:
                    pass
    if kind == "safetensors":
        return _backend.read_safetensors_file(
            fileobj, map_location=map_location, mmap=mmap
        )
    return _backend.read_torch_file(
        fileobj,
        map_location=map_location,
        mmap=mmap,
        **pickle_load_args,
    )


def inspect_checkpoint(f, *, verify_checksums: bool = False) -> dict:
    if _is_path_like(f):
        filename = os.fspath(f)
        lower = filename.lower()
        if lower.endswith(MEGA_INDEX_SUFFIX):
            with open(filename, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            return {
                "format": "mega_shard_index",
                "metadata": index.get("metadata", {}),
                "weight_map": index.get("weight_map", {}),
                "shards": sorted(set(index.get("weight_map", {}).values())),
            }
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No such file or directory: {filename!r}")
        with open(filename, "rb") as handle:
            head = handle.read(512)
        kind = _sniff_format(head)
        if kind == "mega":
            return _inspect_mega(filename, verify_checksums=verify_checksums)
        with open(filename, "rb") as handle:
            if kind == "torch_zip" or kind == "safetensors":
                return (
                    _backend.describe_torch_file(handle)
                    if kind == "torch_zip"
                    else _backend.describe_safetensors_file(handle)
                )
        if kind == "torch_tar":
            return {"format": "torch_legacy_tar"}
        if _probe_torch_stream(filename):
            return {"format": "torch_stream"}
        raise ValueError(f"unrecognized checkpoint: {filename!r}")

    position = _file_position(f)
    try:
        head = f.read(512)
        f.seek(position)
        kind = _sniff_format(head)
        if kind == "torch_zip":
            return _backend.describe_torch_file(f)
        if kind == "safetensors":
            return _backend.describe_safetensors_file(f)
        return {"format": kind or "unknown"}
    finally:
        try:
            f.seek(position)
        except (AttributeError, OSError, ValueError):
            pass


def _inspect_mega(filename: str, *, verify_checksums: bool) -> dict:
    header = parse_mega_header(filename)
    tensors = {}
    for name, info in header["tensors"].items():
        tensors[name] = {
            "shape": info["shape"],
            "dtype": info["dtype"],
            "nbytes": info["logical_nbytes"],
            "payload_offset": info["payload_offset"],
            "checksum": {0: None, 1: "crc32", 2: "sha256"}.get(info["checksum_type"]),
        }
    result = {
        "format": "mega",
        "version": header["version"],
        "metadata": header["metadata"],
        "tensors": tensors,
        "header_length": header["header_length"],
        "file_size": header["file_size"],
    }
    if verify_checksums:
        value = load_mega(filename, "cpu", mmap=False)
        del value
        result["checksums_verified"] = True
    return result


__all__ = [
    "DEFAULT_ALIGNMENT",
    "DEFAULT_PROTOCOL",
    "LoadEndianness",
    "MAGIC_NUMBER",
    "MEGA_EXTENSION",
    "MEGA_INDEX_SUFFIX",
    "PROTOCOL_VERSION",
    "StorageType",
    "add_safe_globals",
    "clear_safe_globals",
    "convert_model",
    "convert_to_mega",
    "default_restore_location",
    "get_crc32_options",
    "get_default_load_endianness",
    "get_default_mmap_options",
    "get_safe_globals",
    "get_unsafe_globals_in_checkpoint",
    "inspect_checkpoint",
    "load",
    "location_tag",
    "normalize_storage_type",
    "register_package",
    "safe_globals",
    "save",
    "set_default_load_endianness",
    "set_default_mmap_options",
    "set_crc32_options",
    "skip_data",
    "storage_to_tensor_type",
]
