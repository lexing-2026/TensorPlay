"""Native MEGA artifact support for TensorPlay checkpoints."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import struct
import sys
import tempfile
import threading
import zlib
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import tensorplay as tp

from .common import (
    _apply_location,
    _contig_stride,
    resolve_restore_location,
)

MEGA_EXTENSION = ".mega"
DEFAULT_ALIGNMENT = 4096
_CHECKSUM_TYPES = {"none": 0, "crc32": 1, "sha256": 2}
_CHECKSUM_NAMES = {value: key for key, value in _CHECKSUM_TYPES.items()}

_MEGA_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
}
_MEGA_TO_TP = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
}
_MEGA_TO_NUMPY = {
    "BOOL": "?",
    "U8": "u1",
    "I8": "i1",
    "I16": "i2",
    "U16": "u2",
    "I32": "i4",
    "U32": "u4",
    "I64": "i8",
    "U64": "u8",
    "F16": "f2",
    "BF16": "u2",
    "F32": "f4",
    "F64": "f8",
}


def _require_megatensors():
    try:
        module = __import__("megatensors")
    except ImportError as error:
        raise ImportError(
            "TensorPlay MEGA serialization requires megatensors>=0.0.5"
        ) from error
    cpp = getattr(module, "cpp", None)
    if cpp is None or not callable(getattr(cpp, "write_file", None)):
        raise ImportError(
            "The installed megatensors package does not provide the MEGA file backend"
        )
    if not callable(getattr(cpp, "parse_metadata_fd", None)):
        raise ImportError(
            "The installed megatensors package does not provide the MEGA parser"
        )
    return module


def _native_metadata(filename: str) -> dict:
    module = _require_megatensors()
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    fd = os.open(filename, flags)
    try:
        size = os.fstat(fd).st_size
        parsed = module.cpp.parse_metadata_fd(fd, filename, size)
    finally:
        os.close(fd)
    if not isinstance(parsed, dict):
        raise ValueError(f"{filename}: invalid MEGA metadata result")
    return parsed


def _json_metadata_map(value, *, field: str) -> dict:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid {field} metadata") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid {field} metadata")
    return {str(key): str(item) for key, item in value.items()}


def _normalise_record(
    record,
    metadata: Mapping[str, Any],
    custom_checksums: Mapping[str, str] | None = None,
    checksum_mode: str | None = None,
) -> dict:
    if not isinstance(record, (tuple, list)) or len(record) < 14:
        raise ValueError("MEGA parser returned an invalid tensor record")
    (
        name,
        _tensor_id,
        dtype,
        shape,
        payload_offset,
        logical_nbytes,
        stored_nbytes,
        tensor_flags,
        compression_codec,
        shuffle_elem_size,
        checksum_type,
        checksum,
        storage_format,
        chunks,
    ) = record[:14]
    name = str(name)
    dtype = str(dtype)
    shape = [int(dim) for dim in shape]
    if any(dim < 0 for dim in shape):
        raise ValueError(f"MEGA tensor {name!r} has a negative dimension")
    if dtype not in _MEGA_DTYPE_SIZES:
        raise NotImplementedError(f"unsupported MEGA dtype {dtype!r}")
    payload_offset = int(payload_offset)
    logical_nbytes = int(logical_nbytes)
    stored_nbytes = int(stored_nbytes)
    if min(payload_offset, logical_nbytes, stored_nbytes) < 0:
        raise ValueError(f"MEGA tensor {name!r} has a negative payload range")
    expected_nbytes = _MEGA_DTYPE_SIZES[dtype]
    for dim in shape:
        expected_nbytes *= dim
    if logical_nbytes != expected_nbytes:
        raise ValueError(
            f"MEGA tensor {name!r} has {logical_nbytes} logical bytes, "
            f"expected {expected_nbytes}"
        )
    checksum_type = int(checksum_type)
    if checksum_type not in _CHECKSUM_NAMES:
        raise ValueError(f"unknown MEGA checksum type {checksum_type}")
    if isinstance(checksum, (bytes, bytearray)):
        checksum_bytes = bytes(checksum)
    else:
        checksum_bytes = b""
    if checksum_bytes and len(checksum_bytes) != 32:
        raise ValueError(f"invalid checksum field for MEGA tensor {name!r}")
    if custom_checksums is None:
        custom_checksums = _json_metadata_map(
            metadata.get("tensorplay.tensor_checksums"),
            field="tensorplay.tensor_checksums",
        )
    mode = checksum_mode or str(metadata.get("tensorplay.checksum", "none")).lower()
    if mode not in _CHECKSUM_TYPES:
        raise ValueError(f"unknown TensorPlay checksum mode {mode!r}")
    if checksum_type == 0 and mode != "none":
        checksum_type = _CHECKSUM_TYPES[mode]
        digest = custom_checksums.get(name)
        if digest:
            try:
                checksum_bytes = bytes.fromhex(digest)
            except ValueError as error:
                raise ValueError(f"invalid checksum for MEGA tensor {name!r}") from error
            if checksum_type == _CHECKSUM_TYPES["crc32"]:
                if len(checksum_bytes) != 4:
                    raise ValueError(f"invalid CRC32 checksum for MEGA tensor {name!r}")
                checksum_bytes += b"\x00" * 28
            elif len(checksum_bytes) != 32:
                raise ValueError(f"invalid SHA256 checksum for MEGA tensor {name!r}")
    return {
        "shape": shape,
        "dtype": dtype,
        "itemsize": _MEGA_DTYPE_SIZES[dtype],
        "payload_offset": payload_offset,
        "logical_nbytes": logical_nbytes,
        "stored_nbytes": stored_nbytes,
        "tensor_flags": int(tensor_flags),
        "compression_codec": int(compression_codec),
        "shuffle_elem_size": int(shuffle_elem_size),
        "checksum_type": checksum_type,
        "checksum": checksum_bytes,
        "storage_format": str(storage_format),
        "chunks": _normalise_chunks(chunks),
        "checksum_hex": custom_checksums.get(name),
    }


def _normalise_chunks(chunks) -> list[dict]:
    if not chunks:
        return []
    normalised = []
    for chunk in chunks:
        if isinstance(chunk, Mapping):
            values = chunk
        elif isinstance(chunk, (tuple, list)) and len(chunk) >= 10:
            values = dict(
                zip(
                    (
                        "tensor_id",
                        "chunk_id",
                        "logical_offset",
                        "logical_size",
                        "payload_offset",
                        "stored_size",
                        "codec",
                        "flags",
                        "checksum_type",
                        "checksum",
                    ),
                    chunk[:10],
                )
            )
        else:
            raise ValueError("MEGA parser returned an invalid chunk record")
        item = {
            key: int(values[key])
            for key in (
                "tensor_id",
                "chunk_id",
                "logical_offset",
                "logical_size",
                "payload_offset",
                "stored_size",
                "codec",
                "flags",
                "checksum_type",
            )
        }
        checksum = values.get("checksum", b"")
        if isinstance(checksum, bytearray):
            checksum = bytes(checksum)
        if not isinstance(checksum, bytes):
            raise ValueError("MEGA chunk checksum must be bytes")
        item["checksum"] = checksum
        normalised.append(item)
    return normalised


def parse_mega_header(path: str | os.PathLike[str]) -> dict:
    """Read MEGA metadata and tensor ranges without materializing tensors."""

    filename = os.fspath(path)
    file_size = os.path.getsize(filename)
    parsed = _native_metadata(filename)
    metadata = dict(parsed.get("metadata") or {})
    custom_checksums = _json_metadata_map(
        metadata.get("tensorplay.tensor_checksums"),
        field="tensorplay.tensor_checksums",
    )
    checksum_mode = str(metadata.get("tensorplay.checksum", "none")).lower()
    header_length = int(parsed.get("header_length", 0))
    if header_length <= 0 or header_length > file_size:
        raise ValueError(f"{filename}: invalid MEGA header length")
    tensors = OrderedDict()
    for record in parsed.get("tensor_records", []):
        name = str(record[0])
        if name in tensors:
            raise ValueError(f"{filename}: duplicate MEGA tensor name {name!r}")
        info = _normalise_record(record, metadata, custom_checksums, checksum_mode)
        end = header_length + info["payload_offset"] + info["stored_nbytes"]
        if end > file_size:
            raise ValueError(f"{filename}: tensor payload is outside the file")
        tensors[name] = info
    return {
        "version": int(parsed.get("version", 1)),
        "metadata": metadata,
        "tensors": tensors,
        "raw_header_size": header_length,
        "header_length": header_length,
        "file_size": file_size,
    }


_PRIMITIVES = (bool, int, float, str)


def _is_tensor(value: Any) -> bool:
    parameter_cls = getattr(tp.nn, "Parameter", None)
    return isinstance(value, tp.Tensor) or (
        parameter_cls is not None and isinstance(value, parameter_cls)
    )


def _flatten_tree(obj: Any) -> tuple:
    if _is_tensor(obj):
        return OrderedDict([("0", obj)]), "tensor", None

    flat: OrderedDict[str, Any] = OrderedDict()
    names_used: set[str] = set()
    active: set[int] = set()
    tensor_names: dict[int, str] = {}

    def assign(value, path):
        identity = id(value)
        existing = tensor_names.get(identity)
        if existing is not None:
            return {"__tensor__": existing}
        name = ".".join(str(part) for part in path) or "0"
        base = name
        counter = 1
        while name in names_used:
            name = f"{base}#{counter}"
            counter += 1
        names_used.add(name)
        tensor_names[identity] = name
        flat[name] = value
        return {"__tensor__": name}

    def encode(node, path):
        if _is_tensor(node):
            return assign(node, path)
        if isinstance(node, _PRIMITIVES) or node is None:
            return node
        if isinstance(node, (dict, list, tuple)):
            node_id = id(node)
            if node_id in active:
                raise ValueError("cyclic containers cannot be saved in MEGA")
            active.add(node_id)
            try:
                if isinstance(node, dict):
                    encoded = {}
                    for key, value in node.items():
                        if not isinstance(key, str):
                            location = ".".join(map(str, path)) or "<root>"
                            raise TypeError(
                                f"dict keys must be str for MEGA serialization "
                                f"(got {type(key).__name__} at {location})"
                            )
                        encoded[key] = encode(value, path + [key])
                    if len(encoded) == 1 and next(iter(encoded)) in {
                        "__tensor__", "__tuple__", "__dict__"
                    }:
                        return {"__dict__": [[key, value] for key, value in encoded.items()]}
                    return encoded
                if isinstance(node, list):
                    return [encode(value, path + [index]) for index, value in enumerate(node)]
                return {
                    "__tuple__": [
                        encode(value, path + [index]) for index, value in enumerate(node)
                    ]
                }
            finally:
                active.remove(node_id)
        location = ".".join(map(str, path)) or "<root>"
        raise TypeError(
            f"unsupported leaf of type {type(node).__name__} at {location}: "
            "MEGA stores tensors and JSON primitives"
        )

    layout = encode(obj, [])
    if isinstance(obj, Mapping) and all(_is_tensor(value) for value in obj.values()):
        if len(flat) == len(obj):
            return OrderedDict((key, obj[key]) for key in obj), "dict", None
    if isinstance(obj, tuple) and all(_is_tensor(value) for value in obj):
        if len(flat) == len(obj):
            return OrderedDict((str(index), value) for index, value in enumerate(obj)), "tuple", None
    if isinstance(obj, list) and all(_is_tensor(value) for value in obj):
        if len(flat) == len(obj):
            return OrderedDict((str(index), value) for index, value in enumerate(obj)), "list", None
    return flat, "tree", layout


def _rebuild_tree(layout, tensors):
    if isinstance(layout, dict):
        if set(layout) == {"__tensor__"}:
            name = layout["__tensor__"]
            if name not in tensors:
                raise ValueError(f"MEGA layout references unknown tensor {name!r}")
            return tensors[name]
        if set(layout) == {"__tuple__"}:
            return tuple(_rebuild_tree(item, tensors) for item in layout["__tuple__"])
        if set(layout) == {"__dict__"}:
            items = layout["__dict__"]
            if not isinstance(items, list):
                raise ValueError("invalid escaped MEGA dictionary layout")
            return {
                str(key): _rebuild_tree(value, tensors)
                for key, value in items
            }
        return {key: _rebuild_tree(value, tensors) for key, value in layout.items()}
    if isinstance(layout, list):
        return [_rebuild_tree(item, tensors) for item in layout]
    return layout


def _metadata_value(value):
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise TypeError(f"MEGA metadata value {value!r} is not serializable") from error


def _metadata_for_writer(metadata, *, container, layout, devices, checksum, alignment):
    result = {str(key): _metadata_value(value) for key, value in (metadata or {}).items()}
    result.setdefault("general.architecture", "tensorplay")
    result.setdefault("general.alignment", int(alignment))
    result.setdefault("mega.tensor_info.format", "self_describing")
    result["tensorplay.container"] = container
    result["tensorplay.byteorder"] = sys.byteorder
    result["tensorplay.tensor_devices"] = json.dumps(
        devices, ensure_ascii=False, separators=(",", ":")
    )
    result["tensorplay.checksum"] = checksum
    result["tensorplay.tensor_checksums"] = ""
    if layout is not None:
        result["tensorplay.layout"] = json.dumps(
            layout, ensure_ascii=False, separators=(",", ":")
        )
    return result


def _append_u8(buffer: bytearray, value: int) -> None:
    buffer.extend(struct.pack("<B", int(value)))


def _append_u32(buffer: bytearray, value: int) -> None:
    buffer.extend(struct.pack("<I", int(value)))


def _append_u64(buffer: bytearray, value: int) -> None:
    buffer.extend(struct.pack("<Q", int(value)))


def _append_string(buffer: bytearray, value: str) -> None:
    encoded = str(value).encode("utf-8")
    _append_u64(buffer, len(encoded))
    buffer.extend(encoded)


def _append_compact_string(buffer: bytearray, value: str) -> None:
    encoded = str(value).encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise ValueError("MEGA compact string is too large")
    _append_u32(buffer, len(encoded))
    buffer.extend(encoded)


def _append_metadata_value(buffer: bytearray, value) -> None:
    if isinstance(value, str):
        _append_u32(buffer, 11)
        _append_string(buffer, value)
        return
    if isinstance(value, bool):
        _append_u32(buffer, 10)
        _append_u8(buffer, int(value))
        return
    if isinstance(value, int):
        if value < 0:
            raise ValueError("MEGA metadata integers must be non-negative")
        if value <= 0xFFFFFFFF:
            _append_u32(buffer, 4)
            _append_u32(buffer, value)
        else:
            _append_u32(buffer, 6)
            _append_u64(buffer, value)
        return
    if isinstance(value, (list, tuple)):
        _append_u32(buffer, 12)
        _append_u64(buffer, len(value))
        if not value or isinstance(value[0], str):
            _append_u32(buffer, 11)
            for item in value:
                if not isinstance(item, str):
                    raise TypeError("MEGA metadata string arrays must contain strings")
                _append_string(buffer, item)
            return
        if isinstance(value[0], int) and not isinstance(value[0], bool):
            _append_u32(buffer, 4)
            for item in value:
                if not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 0xFFFFFFFF:
                    raise TypeError("MEGA metadata integer arrays must contain uint32 values")
                _append_u32(buffer, item)
            return
    raise TypeError(f"unsupported MEGA metadata value type: {type(value).__name__}")


def _build_raw_header(records, metadata: Mapping[str, Any], checksum: str) -> bytes:
    header = bytearray()
    _append_u32(header, 0x4147454D)
    _append_u32(header, 1)
    _append_u64(header, len(records))
    _append_u64(header, len(metadata))
    for key, value in metadata.items():
        _append_string(header, str(key))
        _append_metadata_value(header, value)

    checksum_type = _CHECKSUM_TYPES[checksum]
    for record in records:
        _append_compact_string(header, record["name"])
        flags = 0
        if record["stored_nbytes"] != record["logical_nbytes"]:
            flags |= 1 << 1
        if checksum_type:
            flags |= 1 << 5
        _append_u32(header, flags)
        shape = record["shape"]
        _append_u32(header, len(shape))
        for dimension in reversed(shape):
            _append_u64(header, dimension)
        _append_compact_string(header, record["logical_dtype"])
        _append_u64(header, record["payload_offset"])
        if flags & (1 << 1):
            _append_u64(header, record["stored_nbytes"])
        if checksum_type:
            _append_u32(header, checksum_type)
            digest = record["checksum_bytes"]
            if len(digest) != 32:
                raise ValueError("MEGA checksum must contain 32 bytes")
            header.extend(digest)
    return bytes(header)


def _raw_tensor_bytes_view(tensor):
    from . import archive

    if archive._device_string_of(tensor) != "cpu":
        return None
    import numpy as np

    if archive._dtype_name_of(tensor) == "bfloat16":
        array = tensor.view(getattr(tp, "uint16")).numpy()
    else:
        array = tensor.numpy()
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return memoryview(array).cast("B")


def _write_raw_mega_file(
    output: str,
    records: list[dict],
    metadata: Mapping[str, Any],
    alignment: int,
    checksum: str,
) -> bool:
    from .policy import _skip_payload_data

    saver = getattr(tp.Tensor, "_save_file_segments", None)
    if not callable(saver) or _skip_payload_data():
        return False

    writer_metadata = dict(metadata)
    checksums = {}
    for record in records:
        digest = b""
        if checksum != "none":
            view = _raw_tensor_bytes_view(record["tensor"])
            if view is None:
                return False
            if checksum == "crc32":
                value = zlib.crc32(view) & 0xFFFFFFFF
                digest = value.to_bytes(4, "little") + b"\x00" * 28
                checksums[record["name"]] = f"{value:08x}"
            else:
                digest = hashlib.sha256(view).digest()
                checksums[record["name"]] = digest.hex()
        record["checksum_bytes"] = digest
    writer_metadata["tensorplay.tensor_checksums"] = json.dumps(
        checksums, ensure_ascii=False, separators=(",", ":")
    )
    try:
        header = _build_raw_header(records, writer_metadata, checksum)
    except (TypeError, ValueError, struct.error):
        return False

    with open(output, "wb", buffering=0) as handle:
        handle.write(header)
        padding = (-len(header)) % int(alignment)
        if padding:
            handle.write(b"\x00" * padding)
    saver(output, [record["tensor"] for record in records])
    return True


def _worker_count(count: int) -> int:
    if count <= 1:
        return 1
    raw = os.environ.get("TENSORPLAY_SERIALIZATION_WORKERS")
    try:
        requested = int(raw) if raw else min(4, os.cpu_count() or 1)
    except ValueError:
        requested = min(4, os.cpu_count() or 1)
    return max(1, min(requested, count))


def _write_at(fd: int, data: bytes, offset: int, lock: threading.Lock):
    if hasattr(os, "pwrite"):
        view = memoryview(data)
        position = 0
        while position < len(view):
            written = os.pwrite(fd, view[position:], offset + position)
            if written <= 0:
                raise OSError("short write while staging MEGA payload")
            position += written
        return
    with lock:
        os.lseek(fd, offset, os.SEEK_SET)
        position = 0
        while position < len(data):
            written = os.write(fd, data[position:])
            if written <= 0:
                raise OSError("short write while staging MEGA payload")
            position += written


def _stage_payloads(records, payload_path: str, total_nbytes: int, checksum: str):
    from . import archive
    from .policy import _skip_payload_data

    fd = os.open(payload_path, os.O_RDWR)
    try:
        os.ftruncate(fd, total_nbytes)
        lock = threading.Lock()

        def stage(record):
            data = (
                b"\x00" * int(record["logical_nbytes"])
                if _skip_payload_data()
                else archive._tensor_bytes(record["tensor"])
            )
            expected = int(record["logical_nbytes"])
            if len(data) != expected:
                raise RuntimeError(
                    f"MEGA tensor {record['name']!r} produced {len(data)} bytes, "
                    f"expected {expected}"
                )
            _write_at(fd, data, int(record["payload_offset"]), lock)
            digest = None
            if checksum == "crc32":
                digest = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"
            elif checksum == "sha256":
                digest = hashlib.sha256(data).hexdigest()
            return record["name"], digest

        workers = _worker_count(len(records))
        if workers == 1:
            values = [stage(record) for record in records]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                values = list(pool.map(stage, records))
        return {name: digest for name, digest in values if digest is not None}
    finally:
        os.close(fd)


def save_mega(filename: str, obj: Any, *, metadata=None,
              alignment: int = DEFAULT_ALIGNMENT, checksum: str = "crc32"):
    from . import archive

    checksum = str(checksum).lower()
    if checksum not in _CHECKSUM_TYPES:
        raise ValueError(
            f"checksum must be one of {sorted(_CHECKSUM_TYPES)} (got {checksum!r})"
        )
    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError("MEGA alignment must be positive")
    flat, container, layout = _flatten_tree(obj)
    records = []
    devices = {}
    payload_offset = 0
    mega_dtypes = {
        "bool": "BOOL",
        "uint8": "U8",
        "int8": "I8",
        "int16": "I16",
        "uint16": "U16",
        "int32": "I32",
        "uint32": "U32",
        "int64": "I64",
        "uint64": "U64",
        "float16": "F16",
        "bfloat16": "BF16",
        "float32": "F32",
        "float64": "F64",
    }
    for name, tensor in flat.items():
        dtype_name = archive._dtype_name_of(tensor)
        mega_dtype = mega_dtypes.get(dtype_name)
        if mega_dtype is None:
            raise NotImplementedError(
                f"TensorPlay dtype {dtype_name!r} is not representable in MEGA"
            )
        shape = [int(dim) for dim in tensor.shape]
        logical_nbytes = int(tensor.numel()) * archive._ITEMSIZE[dtype_name]
        records.append(
            {
                "name": str(name),
                "shape": shape,
                "logical_dtype": mega_dtype,
                "storage_format": "raw_dense",
                "payload_offset": payload_offset,
                "logical_nbytes": logical_nbytes,
                "stored_nbytes": logical_nbytes,
                "tensor": tensor,
            }
        )
        devices[str(name)] = archive._device_string_of(tensor)
        payload_offset += logical_nbytes

    output = os.fspath(filename)
    writer_metadata = _metadata_for_writer(
        metadata,
        container=container,
        layout=layout,
        devices=devices,
        checksum=checksum,
        alignment=alignment,
    )
    if _write_raw_mega_file(
        output, records, writer_metadata, alignment, checksum
    ):
        return

    output_dir = os.path.dirname(os.path.abspath(output)) or "."
    temp_fd, payload_path = tempfile.mkstemp(
        prefix=".tensorplay-", suffix=".payload", dir=output_dir
    )
    os.close(temp_fd)
    try:
        checksums = _stage_payloads(records, payload_path, payload_offset, checksum)
        writer_metadata = _metadata_for_writer(
            metadata,
            container=container,
            layout=layout,
            devices=devices,
            checksum=checksum,
            alignment=alignment,
        )
        writer_metadata["tensorplay.tensor_checksums"] = json.dumps(
            checksums, ensure_ascii=False, separators=(",", ":")
        )
        descriptors = []
        for record in records:
            descriptor = {
                key: value
                for key, value in record.items()
                if key not in {"tensor", "name"}
            }
            descriptor["name"] = record["name"]
            descriptor["src_filename"] = payload_path
            descriptor["src_offset"] = record["payload_offset"]
            if checksum != "none":
                digest = bytes.fromhex(checksums[record["name"]])
                if checksum == "crc32":
                    digest += b"\x00" * 28
                descriptor["checksum_type"] = _CHECKSUM_TYPES[checksum]
                descriptor["checksum"] = digest
            descriptors.append(descriptor)
        module = _require_megatensors()
        module.cpp.write_file(output, descriptors, writer_metadata, alignment)
    finally:
        try:
            os.unlink(payload_path)
        except FileNotFoundError:
            pass


class _ArtifactAware(OrderedDict):
    pass


class _ArtifactList(list):
    pass


class _ArtifactTuple(tuple):
    pass


def _attach_keepalive(result, owner):
    if isinstance(result, OrderedDict):
        holder = _ArtifactAware(result)
        holder._mega_keepalive = owner
        return holder
    if isinstance(result, dict):
        holder = _ArtifactAware(result)
        holder._mega_keepalive = owner
        return holder
    if isinstance(result, list):
        holder = _ArtifactList(result)
        holder._mega_keepalive = owner
        return holder
    if isinstance(result, tuple):
        holder = _ArtifactTuple(result)
        holder._mega_keepalive = owner
        return holder
    return result


def _dtype_name(mega_dtype):
    name = _MEGA_TO_TP.get(str(mega_dtype))
    if name is None or not hasattr(tp, name):
        raise NotImplementedError(f"unsupported MEGA dtype {mega_dtype!r}")
    return name


def _read_at(fd: int, offset: int, size: int) -> bytes:
    if size == 0:
        return b""
    if hasattr(os, "pread"):
        data = os.pread(fd, size, offset)
    else:
        os.lseek(fd, offset, os.SEEK_SET)
        data = os.read(fd, size)
    if len(data) != size:
        raise ValueError("truncated MEGA tensor payload")
    return data


def _mmap_tensor(mapping, offset: int, info: dict):
    import numpy as np

    dtype_name = _dtype_name(info["dtype"])
    if info["stored_nbytes"] != info["logical_nbytes"]:
        return None
    length = int(info["logical_nbytes"])
    shape = list(info["shape"])
    if length == 0:
        return tp.empty(shape, dtype=getattr(tp, dtype_name))
    np_dtype = _MEGA_TO_NUMPY[info["dtype"]]
    array = np.frombuffer(
        memoryview(mapping)[offset:offset + length],
        dtype=np_dtype,
        count=length // info["itemsize"],
    )
    try:
        flat = tp.from_dlpack(array)
    except (AttributeError, TypeError):
        flat = tp.from_dlpack(array.__dlpack__())
    if dtype_name == "bfloat16":
        flat = flat.view(getattr(tp, "bfloat16"))
    try:
        flat._serialization_keepalive = mapping
    except (AttributeError, TypeError):
        pass
    result = flat.reshape(shape) if shape else flat.reshape([])
    try:
        result._serialization_keepalive = mapping
    except (AttributeError, TypeError):
        pass
    return result


def _verify_payload(fd, mapping, filename: str, header_length: int, name: str, info: dict):
    from .policy import _skip_payload_data

    if _skip_payload_data():
        return
    if info.get("chunks"):
        return
    checksum_type = int(info["checksum_type"])
    if checksum_type == 0:
        return
    offset = int(info["payload_offset"]) + int(header_length)
    size = int(info["stored_nbytes"])
    if mapping is not None:
        payload = memoryview(mapping)[offset:offset + size]
    else:
        payload = _read_at(fd, offset, size)
    expected_hex = info.get("checksum_hex")
    if expected_hex:
        if checksum_type == _CHECKSUM_TYPES["crc32"]:
            actual = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
        else:
            actual = hashlib.sha256(payload).hexdigest()
        if actual.lower() != expected_hex.lower():
            raise ValueError(f"MEGA tensor {name!r} checksum verification failed")
        return
    checksum = info.get("checksum", b"")
    if checksum_type == _CHECKSUM_TYPES["crc32"] and checksum and any(checksum):
        actual = (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "little")
        if checksum[:4] != actual:
            raise ValueError(f"MEGA tensor {name!r} checksum verification failed")
    elif checksum_type == _CHECKSUM_TYPES["sha256"] and checksum and any(checksum):
        if hashlib.sha256(payload).digest() != checksum[:32]:
            raise ValueError(f"MEGA tensor {name!r} checksum verification failed")


def load_mega(filename: str, map_location=None, *, mmap: bool = False):
    from . import archive
    from .policy import _skip_payload_data

    filename = os.fspath(filename)
    header = parse_mega_header(filename)
    metadata = header["metadata"]
    infos = header["tensors"]
    saved_byteorder = str(metadata.get("tensorplay.byteorder", "little"))
    if saved_byteorder not in {"little", "big"}:
        raise ValueError(f"unknown MEGA byteorder {saved_byteorder!r}")
    swap = saved_byteorder != sys.byteorder
    devices = _json_metadata_map(
        metadata.get("tensorplay.tensor_devices"), field="tensorplay.tensor_devices"
    )
    layout = metadata.get("tensorplay.layout")
    if isinstance(layout, str):
        try:
            layout = json.loads(layout)
        except json.JSONDecodeError as error:
            raise ValueError("invalid tensorplay.layout metadata") from error

    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    fd = os.open(filename, flags)
    mapping = None
    skip = _skip_payload_data()
    if mmap and not swap and not skip:
        try:
            from .policy import _mmap_file

            mapping = _mmap_file(fd, filename=filename)
        except (BufferError, OSError, TypeError, ValueError):
            mapping = None

    try:
        region_owner = {}
        loaded: dict[str, Any] = {}
        aliases: list[tuple[str, str, dict]] = []
        cpp = None
        for name, info in infos.items():
            _verify_payload(fd, mapping, filename, header["header_length"], name, info)
            raw_dense = (
                info["storage_format"] == "raw_dense"
                and int(info["tensor_flags"]) == 0
                and int(info["compression_codec"]) == 0
                and not info["chunks"]
            )
            region = (
                info["payload_offset"],
                info["stored_nbytes"],
                info["dtype"],
            )
            if raw_dense and region in region_owner:
                aliases.append((name, region_owner[region], info))
                continue
            if raw_dense:
                region_owner[region] = name
            dtype_name = _dtype_name(info["dtype"])
            saved_location = str(devices.get(name, "cpu"))
            target = resolve_restore_location(
                map_location, saved_location, info["logical_nbytes"]
            )
            tensor = None
            absolute_offset = header["header_length"] + int(info["payload_offset"])
            if (
                mapping is not None
                and isinstance(target, str)
                and target == "cpu"
                and raw_dense
            ):
                try:
                    tensor = _mmap_tensor(mapping, absolute_offset, info)
                except (BufferError, ValueError, RuntimeError):
                    tensor = None
            if tensor is None and raw_dense:
                data = (
                    b"\x00" * int(info["stored_nbytes"])
                    if skip
                    else _read_at(fd, absolute_offset, int(info["stored_nbytes"]))
                )
                if swap:
                    data = archive._byteswap_bytes(data, dtype_name)
                flat = archive._tensor_from_flat_bytes(data, dtype_name)
                if flat is None:
                    tensor = tp.empty(info["shape"], dtype=getattr(tp, dtype_name))
                else:
                    tensor = (
                        flat.reshape(info["shape"])
                        if info["shape"]
                        else flat.reshape([])
                    )
            if tensor is None and not skip:
                if cpp is None:
                    cpp = _require_megatensors().cpp
                decoder = getattr(cpp, "decode_payload_fd", None)
                if not callable(decoder):
                    raise ImportError("MEGA payload decoder is unavailable")
                target_device = target if isinstance(target, str) else "cpu"
                decode_device = "cpu" if swap else target_device
                if decode_device == "cpu":
                    tensor = tp.empty(info["shape"], dtype=getattr(tp, dtype_name))
                else:
                    tensor = tp.empty(
                        info["shape"],
                        dtype=getattr(tp, dtype_name),
                        device=archive._parse_device(decode_device),
                    )
                if info["chunks"]:
                    chunk_decoder = getattr(cpp, "decode_chunks_fd", None)
                    if not callable(chunk_decoder):
                        raise ImportError("MEGA chunk decoder is unavailable")
                    chunk_decoder(
                        fd,
                        filename,
                        str(name),
                        int(header["header_length"]),
                        info["chunks"],
                        int(info["logical_nbytes"]),
                        int(info["shuffle_elem_size"]),
                        int(tensor.data_ptr()),
                        decode_device != "cpu",
                    )
                else:
                    decoder(
                        fd,
                        filename,
                        str(name),
                        absolute_offset,
                        int(info["stored_nbytes"]),
                        int(info["logical_nbytes"]),
                        int(info["tensor_flags"]),
                        int(info["compression_codec"]),
                        int(info["shuffle_elem_size"]),
                        int(info["checksum_type"]),
                        bytes(info.get("checksum", b"")),
                        int(tensor.data_ptr()),
                        decode_device != "cpu",
                    )
                if swap:
                    tensor = archive._byteswap_tensor(tensor)
            if tensor is None:
                tensor = tp.zeros(info["shape"], dtype=getattr(tp, dtype_name))
            loaded[name] = _apply_location(tensor, target)
        for name, owner, info in aliases:
            base = loaded[owner]
            shape = list(info["shape"])
            loaded[name] = base.as_strided(shape, _contig_stride(shape), 0)

        ordered = OrderedDict((name, loaded[name]) for name in infos)
        container = str(metadata.get("tensorplay.container", "dict"))
        if container == "tree" and layout is not None:
            result = _rebuild_tree(layout, ordered)
        elif container == "tuple":
            result = tuple(ordered.values())
        elif container == "list":
            result = list(ordered.values())
        elif container == "tensor":
            if len(ordered) != 1:
                raise ValueError("MEGA tensor container must contain exactly one tensor")
            result = next(iter(ordered.values()))
        else:
            result = ordered
        if mapping is not None:
            result = _attach_keepalive(result, mapping)
            mapping = None
        return result
    finally:
        os.close(fd)
        if mapping is not None:
            mapping.close()


def _convert_safetensors_file(
    source: str,
    target: str,
    *,
    metadata=None,
    alignment: int = DEFAULT_ALIGNMENT,
    checksum: str = "none",
):
    from . import archive

    source = os.path.abspath(os.fspath(source))
    target = os.fspath(target)
    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError("MEGA alignment must be positive")
    with open(source, "rb") as handle:
        header, data_start = archive._read_safetensors_header(handle)
    file_size = os.path.getsize(source)
    source_metadata = header.get("__metadata__", {})
    if not isinstance(source_metadata, Mapping):
        raise ValueError("safetensors metadata must be a mapping")
    merged_metadata = dict(source_metadata)
    merged_metadata.update(metadata or {})

    records = []
    devices = {}
    cursor = 0
    payload_offset = 0
    checksum_mode = str(checksum).lower()
    if checksum_mode not in _CHECKSUM_TYPES:
        raise ValueError(
            f"checksum must be one of {sorted(_CHECKSUM_TYPES)} "
            f"(got {checksum!r})"
        )
    checksum_fd = None
    try:
        if checksum_mode != "none":
            checksum_fd = os.open(source, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(spec, Mapping):
                raise ValueError(f"safetensors tensor {name!r} spec must be an object")
            dtype = str(spec.get("dtype", ""))
            dtype_name = archive._SAFETENSORS_DTYPES.get(dtype)
            if dtype_name is None:
                raise NotImplementedError(
                    f"unsupported safetensors dtype: {dtype!r}"
                )
            raw_shape = spec.get("shape")
            if not isinstance(raw_shape, list):
                raise ValueError(f"invalid shape for safetensors tensor {name!r}")
            shape = [int(dim) for dim in raw_shape]
            if any(dim < 0 for dim in shape):
                raise ValueError(f"safetensors tensor {name!r} has a negative dimension")
            offsets = spec.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(item, int) and not isinstance(item, bool) for item in offsets)
            ):
                raise ValueError(f"invalid data_offsets for tensor {name!r}")
            start, end = (int(offsets[0]), int(offsets[1]))
            if not (cursor <= start <= end):
                raise ValueError(
                    f"safetensors tensor {name!r} has non-monotonic data_offsets"
                )
            nbytes = end - start
            expected = archive._ITEMSIZE[dtype_name]
            for dim in shape:
                expected *= dim
            if nbytes != expected:
                raise ValueError(
                    f"safetensors tensor {name!r} has {nbytes} payload bytes, "
                    f"expected {expected}"
                )
            if start < 0 or end > file_size - data_start:
                raise ValueError(f"safetensors tensor {name!r} payload is outside the file")

            descriptor = {
                "name": str(name),
                "shape": shape,
                "logical_dtype": dtype,
                "storage_format": "raw_dense",
                "payload_offset": payload_offset,
                "logical_nbytes": nbytes,
                "stored_nbytes": nbytes,
                "src_filename": source,
                "src_offset": data_start + start,
            }
            if checksum_fd is not None:
                if hasattr(os, "pread"):
                    data = os.pread(checksum_fd, nbytes, data_start + start)
                else:
                    os.lseek(checksum_fd, data_start + start, os.SEEK_SET)
                    data = os.read(checksum_fd, nbytes)
                if len(data) != nbytes:
                    raise ValueError(f"truncated safetensors tensor {name!r}")
                if checksum_mode == "crc32":
                    digest = (zlib.crc32(data) & 0xFFFFFFFF).to_bytes(4, "little")
                    descriptor["checksum_type"] = _CHECKSUM_TYPES[checksum_mode]
                    descriptor["checksum"] = digest + b"\x00" * 28
                else:
                    descriptor["checksum_type"] = _CHECKSUM_TYPES[checksum_mode]
                    descriptor["checksum"] = hashlib.sha256(data).digest()
            records.append(descriptor)
            devices[str(name)] = "cpu"
            cursor = end
            payload_offset += nbytes
    finally:
        if checksum_fd is not None:
            os.close(checksum_fd)

    writer_metadata = _metadata_for_writer(
        merged_metadata,
        container="dict",
        layout=None,
        devices=devices,
        checksum=checksum_mode,
        alignment=alignment,
    )
    writer_metadata["tensorplay.byteorder"] = "little"
    writer_metadata["tensorplay.tensor_checksums"] = ""
    os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
    _require_megatensors().cpp.write_file(target, records, writer_metadata, int(alignment))
    return target


def convert_to_mega(model_dir, output_dir=None, **kwargs):
    """Convert a model directory or a supported checkpoint into MEGA."""

    source = os.fspath(model_dir)
    if os.path.isfile(source):
        from pathlib import Path

        options = dict(kwargs)
        metadata = options.pop("metadata", None)
        alignment = options.pop("alignment", DEFAULT_ALIGNMENT)
        checksum = options.pop("checksum", "none")
        if options:
            names = ", ".join(sorted(options))
            raise TypeError(f"unsupported single-file conversion options: {names}")
        source_path = Path(source)
        if output_dir is None:
            target = source_path.with_suffix(MEGA_EXTENSION)
        else:
            target_path = Path(output_dir)
            target = (
                target_path / f"{source_path.stem}{MEGA_EXTENSION}"
                if target_path.is_dir() or target_path.suffix != MEGA_EXTENSION
                else target_path
            )
            target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix.lower() == ".safetensors":
            return _convert_safetensors_file(
                source_path,
                target,
                metadata=metadata,
                alignment=alignment,
                checksum=checksum,
            )
        value = tp.load(source_path, map_location="cpu", mmap=False)
        save_mega(target, value, metadata=metadata, alignment=alignment, checksum=checksum)
        return target
    return _require_megatensors().convert_model(
        model_dir, output_dir=output_dir, **kwargs
    )


convert_model = convert_to_mega

__all__ = [
    "DEFAULT_ALIGNMENT",
    "MEGA_EXTENSION",
    "_CHECKSUM_NAMES",
    "_CHECKSUM_TYPES",
    "_MEGA_DTYPE_SIZES",
    "_flatten_tree",
    "_rebuild_tree",
    "_require_megatensors",
    "convert_model",
    "convert_to_mega",
    "load_mega",
    "parse_mega_header",
    "save_mega",
]
