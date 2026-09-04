"""Shared validation and location utilities for checkpoint readers."""

from __future__ import annotations

import os
import re
import struct
import sys
from collections.abc import Mapping
from typing import Any

import tensorplay as tp


_CUDA_SPEC = re.compile(r"^cuda(?::([0-9]+))?$")
_MEGA_MAGIC = b"MEGA"
_MAX_SAFE_HEADER = 256 * 1024 * 1024


class _StorageStub:
    __slots__ = ("device", "nbytes")

    def __init__(self, location: str, nbytes: int = 0):
        self.device = location
        self.nbytes = int(nbytes)


def _coerce_location(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("device locations must contain ASCII text") from error
    if not isinstance(value, str):
        raise TypeError(f"device location must be a string, got {type(value).__name__}")
    return value.strip()


def _validate_device_spec(spec: Any) -> str:
    spec = _coerce_location(spec)
    if spec == "cpu":
        return spec

    match = _CUDA_SPEC.fullmatch(spec)
    if match is not None:
        if not hasattr(tp, "cuda") or not tp.cuda.is_available():
            raise RuntimeError(
                f"Attempting to deserialize onto {spec} but TensorPlay CUDA "
                "support is not available; use map_location='cpu'."
            )
        index = int(match.group(1) or 0)
        device_count = getattr(tp.cuda, "device_count", None)
        if callable(device_count):
            count = int(device_count())
            if index >= count:
                raise RuntimeError(
                    f"Attempting to deserialize onto cuda:{index}, but only "
                    f"{count} CUDA device(s) are available."
                )
        return f"cuda:{index}"

    raise RuntimeError(
        f"TensorPlay serialization does not support device {spec!r}; "
        "supported targets are 'cpu' and 'cuda[:index]'."
    )


def resolve_map_location(
    map_location: Any,
    location: str,
    nbytes: int = 0,
):
    """Resolve one saved storage location.

    ``None`` means that the saved location should be used by the caller.
    A callable may return a device, a destination tensor, or ``None``.
    """

    location = _coerce_location(location)
    if map_location is None:
        return None
    if isinstance(map_location, (str, bytes)):
        return _validate_device_spec(map_location)
    if isinstance(map_location, tp.Device):
        return _validate_device_spec(str(map_location))
    if isinstance(map_location, Mapping):
        target = map_location.get(location, location)
        if target is None:
            return None
        return _validate_device_spec(target)
    if callable(map_location):
        result = map_location(_StorageStub(location, nbytes), location)
        if result is None:
            return None
        if isinstance(result, tp.Tensor):
            return result
        if isinstance(result, tp.Device):
            result = str(result)
        return _validate_device_spec(result)
    raise TypeError(
        "map_location must be None, a device string, a TensorPlay Device, a "
        "mapping, or a callable"
    )


def resolve_restore_location(map_location: Any, location: str, nbytes: int = 0):
    resolved = resolve_map_location(map_location, location, nbytes)
    return _validate_device_spec(location) if resolved is None else resolved


def _parse_device(spec: Any):
    spec = _validate_device_spec(spec)
    if spec == "cpu":
        return tp.Device(tp.DeviceType.CPU)
    _, index = spec.split(":", 1)
    return tp.Device(tp.DeviceType.CUDA, int(index))


def _move_to(tensor, target):
    if target is None:
        return tensor
    if isinstance(target, tp.Tensor):
        return _apply_location(tensor, target)
    target = _validate_device_spec(target)
    if target == "cpu":
        return tensor
    return tensor.to(_parse_device(target))


def _apply_location(flat, resolved):
    if resolved is None:
        return flat
    if isinstance(resolved, tp.Tensor):
        if int(resolved.numel()) != int(flat.numel()):
            raise ValueError(
                f"map_location returned a tensor with {resolved.numel()} elements "
                f"but the checkpoint storage holds {flat.numel()}"
            )
        resolved.copy_(flat.reshape(resolved.shape))
        return resolved
    return _move_to(flat, resolved)


def _contig_stride(size):
    stride = [0] * len(size)
    running = 1
    for index in range(len(size) - 1, -1, -1):
        stride[index] = running
        running *= int(size[index])
    return stride


def _byteswap_tensor(tensor):
    from .archive import (
        _contiguous_stride,
        _dtype_name_of,
        _tensor_from_flat_bytes,
    )

    if not tensor.is_contiguous():
        raise ValueError("byte-swapping requires a contiguous tensor")
    if tensor.numel() == 0:
        return tensor
    import numpy as np

    dtype_name = _dtype_name_of(tensor)
    if dtype_name in {"bfloat16", "complex32", "bcomplex32"}:
        array = tensor.view(getattr(tp, "uint16")).numpy()
    else:
        array = tensor.numpy()
    flat = _tensor_from_flat_bytes(array.byteswap().tobytes(), dtype_name)
    shape = [int(dim) for dim in tensor.shape]
    return flat.reshape(shape) if shape else flat.reshape([])


def _is_path_like(value: Any) -> bool:
    return isinstance(value, (str, os.PathLike))


def _sniff_format(head: bytes) -> str | None:
    if head[:4] == b"PK\x03\x04":
        return "torch_zip"
    if head[:4] == _MEGA_MAGIC:
        return "mega"
    if len(head) >= 8:
        (header_length,) = struct.unpack_from("<Q", head, 0)
        if 2 <= header_length <= _MAX_SAFE_HEADER:
            first = head[8:].lstrip()[:1]
            if first == b"{":
                return "safetensors"
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "torch_tar"
    return None


def _file_position(fileobj) -> int:
    try:
        return int(fileobj.tell())
    except (AttributeError, OSError, ValueError) as error:
        raise ValueError(
            "checkpoint streams must provide tell(), seek(), and read()"
        ) from error


__all__ = [
    "_StorageStub",
    "_apply_location",
    "_byteswap_tensor",
    "_contig_stride",
    "_file_position",
    "_is_path_like",
    "_MAX_SAFE_HEADER",
    "_move_to",
    "_parse_device",
    "_sniff_format",
    "_validate_device_spec",
    "resolve_map_location",
    "resolve_restore_location",
]
