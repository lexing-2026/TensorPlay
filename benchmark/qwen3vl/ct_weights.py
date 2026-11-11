"""Reader for pack-quantized 4-bit checkpoints stored as safetensors shards.

The shards are read straight from their byte buffers, so nothing beyond numpy
is needed to open them: the header names every tensor, its element type, its
shape and the byte range it occupies in the shard.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import numpy as np

# Element types the format spells out, mapped onto the numpy view that reads
# the same bytes back.
_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Widen truncated-precision halves to single precision."""
    u = raw.astype(np.uint32) << 16
    return u.view(np.float32)


class ShardReader:
    """A single safetensors shard opened for random reads."""

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "rb")
        header_len = int.from_bytes(self._file.read(8), "little")
        self.header = json.loads(self._file.read(header_len).decode("utf-8"))
        self.data_start = 8 + header_len

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def raw(self, name: str) -> np.ndarray:
        """Return the tensor as it sits on disk, without converting halves."""
        info = self.header[name]
        dtype = _DTYPES[info["dtype"]]
        nbytes = int(np.prod(info["shape"])) * dtype.itemsize if info["shape"] else 0
        start, end = info["data_offsets"]
        assert end - start == nbytes, (name, end - start, nbytes)
        self._file.seek(self.data_start + start)
        buf = self._file.read(nbytes)
        return np.frombuffer(buf, dtype=dtype).reshape(info["shape"])

    def float32(self, name: str) -> np.ndarray:
        arr = self.raw(name)
        if self.header[name]["dtype"] == "BF16":
            return bf16_to_f32(arr)
        return arr.astype(np.float32)

    def close(self):
        self._file.close()


class Checkpoint:
    """All shards of one checkpoint directory, addressed by tensor name."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            weight_map = json.load(open(index_path))["weight_map"]
            shards = sorted(set(weight_map.values()))
            self.by_shard = {s: [k for k, v in weight_map.items() if v == s] for s in shards}
        else:
            single = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")][0]
            shards = [single]
            self.by_shard = {single: None}
        self.readers: dict[str, ShardReader] = {}
        self._names = set()
        for s in shards:
            rd = ShardReader(os.path.join(model_dir, s))
            self.readers[s] = rd
            names = rd.keys() if self.by_shard[s] is None else self.by_shard[s]
            self.by_shard[s] = names
            self._names.update(names)
        self._owner = {n: s for s, names in self.by_shard.items() for n in names}

    def names(self) -> set[str]:
        return self._names

    def has(self, name: str) -> bool:
        return name in self._names

    def float32(self, name: str) -> np.ndarray:
        return self.readers[self._owner[name]].float32(name)

    def raw(self, name: str) -> np.ndarray:
        return self.readers[self._owner[name]].raw(name)

    def close(self):
        for rd in self.readers.values():
            rd.close()


def unpack_int4(packed: np.ndarray, columns: int, bits: int = 4) -> np.ndarray:
    """Expand little-endian ``bits``-wide fields into signed codes."""
    per_word = 32 // bits
    out_dim = packed.shape[0]
    unpacked = np.empty((out_dim, packed.shape[1] * per_word), dtype=np.int32)
    words = packed.astype(np.int32)
    mask = (1 << bits) - 1
    for i in range(per_word):
        unpacked[:, i::per_word] = (words >> (bits * i)) & mask
    unpacked = unpacked[:, :columns]
    return unpacked - (1 << (bits - 1))


def dequantize_groupwise(packed: np.ndarray, scale: np.ndarray, shape, bits: int = 4) -> np.ndarray:
    """Recover a dense matrix from packed codes and per-group scales.

    ``scale`` holds one factor per group of ``columns / scale.shape[1]``
    inputs; symmetric codes carry no zero point, so the value is the code
    times its group factor.
    """
    out_dim, columns = int(shape[0]), int(shape[1])
    codes = unpack_int4(packed, columns, bits)
    groups = scale.shape[1]
    if columns % groups:
        raise ValueError(f"{columns} columns do not split into {groups} groups")
    group_size = columns // groups
    wide = np.repeat(scale.astype(np.float32), group_size, axis=1)
    return codes.astype(np.float32) * wide


def pack_codes_uint8(codes: np.ndarray, bits: int = 4) -> np.ndarray:
    """Pack codes into bytes, the first code of each group in the high field."""
    per_byte = 8 // bits
    out_dim, columns = codes.shape
    padded = columns
    if columns % per_byte:
        padded = columns + per_byte - columns % per_byte
        codes = np.concatenate(
            [codes, np.zeros((out_dim, padded - columns), dtype=np.uint8)], axis=1
        )
    packed = np.zeros((out_dim, padded // per_byte), dtype=np.uint8)
    for i in range(per_byte):
        shift = 8 - (i + 1) * bits
        packed |= codes[:, i::per_byte] << shift
    return packed


def weight_only_group(ckpt: Checkpoint, base: str, bits: int = 4):
    """Turn a packed checkpoint weight into codes, scales and zero points.

    The stored codes are signed and carry no zero point (symmetric groups);
    the weight-only linear reads unsigned codes, so a constant half-range
    offset is added back and recorded as the group's zero point.
    """
    packed = ckpt.raw(base + "_packed")
    scale = ckpt.float32(base + "_scale")
    shape = ckpt.raw(base + "_shape")
    out_dim, columns = int(shape[0]), int(shape[1])
    codes = unpack_int4(packed, columns, bits) + (1 << (bits - 1))
    if (codes < 0).any() or (codes >= (1 << bits)).any():
        raise RuntimeError(f"{base}: codes outside the {bits}-bit unsigned range")
    groups = scale.shape[1]
    if columns % groups:
        raise ValueError(f"{base}: {columns} inputs do not split into {groups} groups")
    zero = np.full((out_dim, groups), 1 << (bits - 1), dtype=np.float32)
    return pack_codes_uint8(codes.astype(np.uint8), bits), scale, zero


def dequantize_linear(ckpt: Checkpoint, weight_name: str) -> np.ndarray:
    """Dense fp32 weights for one linear layer, quantized or stored outright.

    ``weight_name`` is the dense spelling (``...q_proj.weight``); a checkpoint
    that stores the layer packed answers to the ``.weight_packed``,
    ``.weight_scale`` and ``.weight_shape`` spellings instead.
    """
    if ckpt.has(weight_name):
        return ckpt.float32(weight_name)
    packed = weight_name + "_packed"
    if not ckpt.has(packed):
        raise KeyError(f"{weight_name} is absent from the checkpoint")
    scale = ckpt.float32(weight_name + "_scale")
    shape = ckpt.raw(weight_name + "_shape")
    return dequantize_groupwise(ckpt.raw(packed), scale, tuple(int(x) for x in shape))


def iter_dense(ckpt: Checkpoint, prefix: str) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(name, fp32)`` for every dense tensor under ``prefix``."""
    for name in sorted(n for n in ckpt.names() if n.startswith(prefix)):
        if name.endswith(".weight_scale") or name.endswith(".weight_shape"):
            continue
        yield name, ckpt.float32(name)
