from __future__ import annotations

import concurrent.futures
import glob
import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import tensorplay as tp
import tensorplay.distributed as dist

from ._hf_utils import (
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    SAVED_OFFSETS_KEY,
    SHAPE_KEY,
    SUFFIX,
    _gen_file_name,
    _get_dcp_custom_metadata,
    _get_safetensors_file_metadata,
    _metadata_fn,
)

logger = logging.getLogger(__name__)


@dataclass
class _FqnData:
    offset_in_file: int = 0
    shape_in_file: list[int] = field(default_factory=list)
    dtype_size: int = 0
    dtype_str: str = ""


@dataclass
class _OutputFileData:
    metadata_size: int = 0
    fqn_data: dict[str, _FqnData] = field(default_factory=dict)


@dataclass
class _InputFileData:
    metadata_size: int = 0
    metadata: Any = None


def _dtype_size(dtype: Any) -> int:
    value = getattr(dtype, "itemsize", 1)
    return int(value() if callable(value) else value)


def _dtype_for_name(dtype_name: str) -> Any:
    names = {
        "BOOL": "bool",
        "U8": "uint8",
        "I8": "int8",
        "I16": "int16",
        "I32": "int32",
        "I64": "int64",
        "U32": "uint32",
        "U64": "uint64",
        "F16": "float16",
        "BF16": "bfloat16",
        "F32": "float32",
        "F64": "float64",
        "C64": "complex64",
        "C128": "complex128",
    }
    try:
        return getattr(tp, names[dtype_name])
    except KeyError as error:
        raise ValueError(f"unsupported safetensors dtype {dtype_name!r}") from error


def _parse_input_metadata(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
) -> None:
    sizes: dict[str, tuple[list[int], str]] = {}
    for file_data in input_files_data.values():
        metadata = file_data.metadata
        if not isinstance(metadata, dict):
            raise ValueError("safetensors metadata must be a dictionary")
        dcp_metadata = _get_dcp_custom_metadata(metadata) or {}
        for fqn, value in metadata.items():
            if fqn == DEFAULT_EXTRA_METADATA_KEY:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"invalid tensor metadata for {fqn!r}")
            shape = [int(item) for item in value[SHAPE_KEY]]
            shard = dcp_metadata.get(fqn, {}) if isinstance(dcp_metadata, dict) else {}
            offsets = [
                int(item)
                for item in shard.get(SAVED_OFFSETS_KEY, [0] * len(shape))
            ]
            if len(offsets) != len(shape):
                raise ValueError(f"invalid shard offsets for {fqn!r}")
            dtype = str(value[DTYPE_KEY])
            if fqn not in sizes:
                sizes[fqn] = (
                    [size + offset for size, offset in zip(shape, offsets)],
                    dtype,
                )
            else:
                current, current_dtype = sizes[fqn]
                if current_dtype != dtype:
                    raise ValueError(f"tensor {fqn!r} has inconsistent dtypes")
                for dimension, (size, offset) in enumerate(zip(shape, offsets)):
                    current[dimension] = max(current[dimension], size + offset)

    for fqn, (shape, dtype_name) in sizes.items():
        dtype_size = _dtype_size(_dtype_for_name(dtype_name))
        for output_data in output_files_data.values():
            if fqn in output_data.fqn_data:
                output_data.fqn_data[fqn] = _FqnData(
                    shape_in_file=shape.copy(),
                    dtype_size=dtype_size,
                    dtype_str=dtype_name,
                )


def _write_metadata(output_files_data: dict[str, _OutputFileData]) -> None:
    for file_path, output_data in output_files_data.items():
        with open(file_path, "wb") as stream:
            metadata: dict[str, Any] = {}
            offset = 0
            for fqn, fqn_data in output_data.fqn_data.items():
                end = offset + math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
                metadata[fqn] = {
                    SHAPE_KEY: fqn_data.shape_in_file,
                    DTYPE_KEY: fqn_data.dtype_str,
                    DATA_OFFSETS_KEY: [offset, end],
                }
                fqn_data.offset_in_file = offset
                offset = end
            raw = json.dumps(metadata, separators=(",", ":")).encode()
            raw += b" " * ((8 - len(raw) % 8) % 8)
            stream.write(struct.pack("<Q", len(raw)))
            stream.write(raw)
            output_data.metadata_size = stream.tell()


def _read_tensor_data(
    f: Any, start_offset: int, end_offset: int, metadata_size: int
) -> bytes:
    f.seek(metadata_size + int(start_offset))
    return f.read(int(end_offset) - int(start_offset))


def _process_output_file(
    output_file: str,
    output_data: _OutputFileData,
    input_files_data: dict[str, _InputFileData],
) -> None:
    custom_metadata = {
        path: _get_dcp_custom_metadata(data.metadata) or {}
        for path, data in input_files_data.items()
    }
    handles: dict[str, Any] = {}
    try:
        handles = {path: open(path, "rb") for path in input_files_data}
        with open(output_file, "r+b") as output_stream:
            output_stream.seek(0, os.SEEK_END)
            for fqn, fqn_data in sorted(
                output_data.fqn_data.items(), key=lambda item: item[1].offset_in_file
            ):
                byte_count = math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
                full_tensor = memoryview(bytearray(byte_count))
                for input_file, input_data in input_files_data.items():
                    metadata = input_data.metadata
                    if fqn not in metadata:
                        continue
                    tensor_metadata = metadata[fqn]
                    offsets = custom_metadata[input_file].get(fqn, {}).get(
                        SAVED_OFFSETS_KEY, [0] * len(tensor_metadata[SHAPE_KEY])
                    )
                    source = _read_tensor_data(
                        handles[input_file],
                        tensor_metadata[DATA_OFFSETS_KEY][0],
                        tensor_metadata[DATA_OFFSETS_KEY][1],
                        input_data.metadata_size,
                    )
                    _write_sub_tensor_to_file_optimized(
                        full_tensor,
                        source,
                        fqn_data.dtype_size,
                        fqn_data.shape_in_file,
                        [int(item) for item in offsets],
                        [int(item) for item in tensor_metadata[SHAPE_KEY]],
                    )
                output_stream.write(full_tensor)
    finally:
        for stream in handles.values():
            stream.close()


def _write_data(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
    num_threads: int = 1,
) -> None:
    if num_threads <= 1 or len(output_files_data) <= 1:
        for output_file, output_data in output_files_data.items():
            _process_output_file(output_file, output_data, input_files_data)
        return
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(int(num_threads), len(output_files_data))
    ) as executor:
        futures = [
            executor.submit(
                _process_output_file, output_file, output_data, input_files_data
            )
            for output_file, output_data in output_files_data.items()
        ]
        for future in futures:
            future.result()


def _write_sub_tensor_to_file_optimized(
    full_tensor_mv: memoryview,
    sub_tensor_bytes: bytes,
    element_size: int,
    tensor_shape: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
) -> None:
    if not sub_tensor_shape:
        full_tensor_mv[: len(sub_tensor_bytes)] = sub_tensor_bytes
        return
    if any(int(size) == 0 for size in sub_tensor_shape):
        return
    tensor_strides = [1]
    for size in reversed(tensor_shape[1:]):
        tensor_strides.insert(0, tensor_strides[0] * int(size))
    sub_tensor_strides = [1]
    for size in reversed(sub_tensor_shape[1:]):
        sub_tensor_strides.insert(0, sub_tensor_strides[0] * int(size))
    total_elements = math.prod(sub_tensor_shape)
    elements_written = 0
    while elements_written < total_elements:
        remaining = elements_written
        indices: list[int] = []
        for size in reversed(sub_tensor_shape):
            indices.append(remaining % int(size))
            remaining //= int(size)
        indices.reverse()
        contiguous = _calculate_max_contiguous_elements(
            indices, sub_tensor_shape, tensor_shape
        )
        source_position = sum(
            index * stride for index, stride in zip(indices, sub_tensor_strides)
        )
        destination_indices = [
            index + offset for index, offset in zip(indices, sub_tensor_offsets)
        ]
        destination_position = sum(
            index * stride for index, stride in zip(destination_indices, tensor_strides)
        )
        byte_count = contiguous * int(element_size)
        source_start = source_position * int(element_size)
        destination_start = destination_position * int(element_size)
        full_tensor_mv[destination_start : destination_start + byte_count] = (
            sub_tensor_bytes[source_start : source_start + byte_count]
        )
        elements_written += contiguous


def _calculate_max_contiguous_elements(
    indices: list[int], sub_tensor_shape: list[int], tensor_shape: list[int]
) -> int:
    if not indices or not sub_tensor_shape or not tensor_shape:
        raise ValueError("Input lists cannot be empty")
    if not len(indices) == len(sub_tensor_shape) == len(tensor_shape):
        raise ValueError(
            "All input lists must have the same length. "
            f"Got indices: {len(indices)}, sub_tensor_shape: {len(sub_tensor_shape)}, "
            f"tensor_shape: {len(tensor_shape)}"
        )
    for dimension, (index, size) in enumerate(zip(indices, sub_tensor_shape)):
        if index < 0 or index >= int(size):
            raise ValueError(
                f"Index {index} at dimension {dimension} is out of bounds "
                f"for sub-tensor shape {sub_tensor_shape}"
            )
    for dimension, (source_size, target_size) in enumerate(
        zip(sub_tensor_shape, tensor_shape)
    ):
        if int(source_size) > int(target_size):
            raise ValueError(
                f"Sub-tensor dimension {source_size} at position {dimension} "
                f"exceeds tensor dimension {target_size}"
            )
    contiguous = int(sub_tensor_shape[-1]) - int(indices[-1])
    if indices[-1] == 0 and sub_tensor_shape[-1] == tensor_shape[-1]:
        rows = (
            int(sub_tensor_shape[-2]) - int(indices[-2])
            if len(sub_tensor_shape) >= 2
            else 1
        )
        contiguous = rows * int(sub_tensor_shape[-1])
        if (
            len(sub_tensor_shape) >= 3
            and indices[-2] == 0
            and sub_tensor_shape[-2] == tensor_shape[-2]
            and sub_tensor_shape[-1] == tensor_shape[-1]
        ):
            contiguous = (
                int(sub_tensor_shape[-3]) - int(indices[-3])
            ) * int(sub_tensor_shape[-2]) * int(sub_tensor_shape[-1])
    return max(1, contiguous)


def _write_overall_metadata_file(
    output_dir: str | os.PathLike[str],
    output_files_data: dict[str, _OutputFileData],
) -> None:
    total_size = 0
    weight_map: dict[str, str] = {}
    for output_file, output_data in output_files_data.items():
        for fqn, fqn_data in output_data.fqn_data.items():
            total_size += math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
            weight_map[fqn] = os.path.basename(output_file)
    path = os.path.join(os.fspath(output_dir), _metadata_fn)
    with open(path, "w") as stream:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            stream,
            indent=2,
        )


def _consolidate_safetensors_files(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    fqn_to_file_mapping: dict[str, str],
    num_threads: int,
) -> dict[str, _OutputFileData]:
    os.makedirs(output_dir, exist_ok=True)
    output_files_data: dict[str, _OutputFileData] = {}
    for fqn, file_name in fqn_to_file_mapping.items():
        output_file = os.path.join(os.fspath(output_dir), file_name)
        output_files_data.setdefault(output_file, _OutputFileData()).fqn_data[fqn] = _FqnData()

    input_files_data: dict[str, _InputFileData] = {}
    for input_file in glob.glob(os.path.join(os.fspath(input_dir), f"*{SUFFIX}")):
        with open(input_file, "rb") as stream:
            metadata, metadata_size = _get_safetensors_file_metadata(stream)
        input_files_data[input_file] = _InputFileData(
            metadata_size=metadata_size, metadata=metadata
        )
    _parse_input_metadata(input_files_data, output_files_data)
    _write_metadata(output_files_data)
    _write_data(input_files_data, output_files_data, num_threads)
    return output_files_data


def consolidate_safetensors_files(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
) -> None:
    start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    if not fqn_to_index_mapping:
        _write_overall_metadata_file(output_dir, {})
        return
    largest_index = max(fqn_to_index_mapping.values())
    mapping = {
        fqn: _gen_file_name(index, largest_index)
        for fqn, index in fqn_to_index_mapping.items()
    }
    output_files_data = _consolidate_safetensors_files(
        input_dir, output_dir, mapping, num_threads
    )
    _write_overall_metadata_file(output_dir, output_files_data)
    logger.info("consolidated checkpoint files in %.2f seconds", time.time() - start)


def consolidate_safetensors_files_on_every_rank(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
    process_group: Any = None,
) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return consolidate_safetensors_files(
            input_dir, output_dir, fqn_to_index_mapping, num_threads
        )
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    indices = {
        index
        for index in set(fqn_to_index_mapping.values())
        if index % world_size == rank
    }
    filtered = {
        fqn: index
        for fqn, index in fqn_to_index_mapping.items()
        if index in indices
    }
    output_data: dict[str, _OutputFileData] = {}
    if filtered:
        largest_index = max(fqn_to_index_mapping.values())
        mapping = {
            fqn: _gen_file_name(index, largest_index)
            for fqn, index in filtered.items()
        }
        output_data = _consolidate_safetensors_files(
            input_dir, output_dir, mapping, num_threads
        )
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(output_data, gathered, dst=0, group=process_group)
    if rank == 0:
        merged: dict[str, _OutputFileData] = {}
        for item in gathered or []:
            merged.update(item)
        _write_overall_metadata_file(output_dir, merged)
    dist.barrier(group=process_group)
