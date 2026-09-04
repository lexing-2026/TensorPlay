from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from typing import Any

import tensorplay as tp

_metadata_fn = "model.safetensors.index.json"
FILE_NAME = "model-{cpt_idx}-of-{num_files}"
SHARDED_FILE_NAME = "shard-{shard_idx}-model-{cpt_idx}-of-{num_files}"
SUFFIX = ".safetensors"
CUSTOM_METADATA_KEY = "DCP_SHARDING_INFO"
DEFAULT_EXTRA_METADATA_KEY = "__metadata__"
SAVED_OFFSETS_KEY = "saved_offsets"
SHAPE_KEY = "shape"
DATA_KEY = "data"
DTYPE_KEY = "dtype"
DATA_OFFSETS_KEY = "data_offsets"
DTYPE_MAP = {
    "BOOL": tp.bool,
    "U8": tp.uint8,
    "I8": tp.int8,
    "I16": tp.int16,
    "I32": tp.int32,
    "I64": tp.int64,
    "U32": tp.uint32,
    "U64": tp.uint64,
    "F16": tp.float16,
    "BF16": tp.bfloat16,
    "F32": tp.float32,
    "F64": tp.float64,
}
HF_DCP_VERSION = 1.0
DCP_VERSION_KEY = "DCP_VERSION"
DCP_SHARDING_INFO_KEY = "DCP_SHARDING_INFO"
FORMAT_KEY = "format"
FORMAT_VALUE = "pt"
NUM_BYTES_FOR_HEADER_LEN = 8
SHARDED_DIR_NAME = "sharded"


@dataclass
class _HFStorageInfo:
    relative_path: str
    shape: tuple[int, ...]
    dtype: Any


def _gen_file_name(index: int, largest_index: int, shard_index: int | None = None) -> str:
    base = SHARDED_FILE_NAME.format(shard_idx=f"{shard_index:05d}", cpt_idx=f"{index:05d}", num_files=f"{largest_index:05d}") if shard_index is not None else FILE_NAME.format(cpt_idx=f"{index:05d}", num_files=f"{largest_index:05d}")
    return base + SUFFIX


def _get_safetensors_file_metadata(file_bytes: io.IOBase) -> tuple[Any, int]:
    header_length = struct.unpack("<Q", file_bytes.read(NUM_BYTES_FOR_HEADER_LEN))[0]
    header = json.loads(file_bytes.read(header_length))
    return header, header_length + NUM_BYTES_FOR_HEADER_LEN


def _get_dtype(dtype_str: str) -> Any:
    return getattr(tp, {"F16": "float16", "F32": "float32", "F64": "float64", "I8": "int8", "U8": "uint8", "I16": "int16", "I32": "int32", "I64": "int64", "BF16": "bfloat16"}.get(dtype_str, "float32"), tp.get_default_dtype())


def _get_dcp_custom_metadata(metadata: Any) -> Any | None:
    extra = metadata.get(DEFAULT_EXTRA_METADATA_KEY, {}) if isinstance(metadata, dict) else {}
    value = extra.get(CUSTOM_METADATA_KEY) if isinstance(extra, dict) else None
    return json.loads(value) if isinstance(value, str) else value
