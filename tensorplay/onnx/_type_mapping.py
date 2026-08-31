"""TensorPlay -> ONNX type mapping.

Maps TensorPlay :class:`tensorplay.DType` values and tensor shape metadata to
``onnx.TensorProto`` data types used when building ``ValueInfoProto`` entries,
initializers and constants.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import TensorProto

try:
    import tensorplay as _tp
except ImportError:  # pragma: no cover - the package is the host project
    _tp = None


__all__ = [
    "_dtype_to_onnx",
    "_np_dtype_to_onnx",
    "_onnx_to_np_dtype",
    "_size_to_tuple",
    "_to_numpy",
]


# ``str(dtype)`` renders as ``"tensorplay.<name>"`` (see src/bindings/python/DType.cpp).
_NAME_TO_ONNX: dict[str, int] = {
    "bool": TensorProto.BOOL,
    "uint8": TensorProto.UINT8,
    "int8": TensorProto.INT8,
    "uint16": TensorProto.UINT16,
    "int16": TensorProto.INT16,
    "uint32": TensorProto.UINT32,
    "int32": TensorProto.INT32,
    "uint64": TensorProto.UINT64,
    "int64": TensorProto.INT64,
    "float16": TensorProto.FLOAT16,
    "bfloat16": TensorProto.BFLOAT16,
    "float32": TensorProto.FLOAT,
    "float64": TensorProto.DOUBLE,
    "complex64": TensorProto.COMPLEX64,
    "complex128": TensorProto.COMPLEX128,
}

_NAME_TO_NUMPY: dict[str, np.dtype] = {
    "bool": np.dtype("bool"),
    "uint8": np.dtype("uint8"),
    "int8": np.dtype("int8"),
    "uint16": np.dtype("uint16"),
    "int16": np.dtype("int16"),
    "uint32": np.dtype("uint32"),
    "int32": np.dtype("int32"),
    "uint64": np.dtype("uint64"),
    "int64": np.dtype("int64"),
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "complex64": np.dtype("complex64"),
    "complex128": np.dtype("complex128"),
}

_NUMPY_TO_ONNX: dict[np.dtype, int] = {
    np.dtype("bool"): TensorProto.BOOL,
    np.dtype("uint8"): TensorProto.UINT8,
    np.dtype("int8"): TensorProto.INT8,
    np.dtype("uint16"): TensorProto.UINT16,
    np.dtype("int16"): TensorProto.INT16,
    np.dtype("uint32"): TensorProto.UINT32,
    np.dtype("int32"): TensorProto.INT32,
    np.dtype("uint64"): TensorProto.UINT64,
    np.dtype("int64"): TensorProto.INT64,
    np.dtype("float16"): TensorProto.FLOAT16,
    np.dtype("float32"): TensorProto.FLOAT,
    np.dtype("float64"): TensorProto.DOUBLE,
    np.dtype("complex64"): TensorProto.COMPLEX64,
    np.dtype("complex128"): TensorProto.COMPLEX128,
}

_ONNX_TO_NUMPY: dict[int, np.dtype] = {
    onnx_type: np_type for np_type, onnx_type in _NUMPY_TO_ONNX.items()
}


def _dtype_name(dtype: Any) -> str | None:
    """Bare TensorPlay dtype name (``"float32"``) for a DType-like object."""

    if _tp is not None and isinstance(dtype, _tp.DType):
        return str(dtype).rsplit(".", 1)[-1]
    text = str(dtype)
    if text.startswith("tensorplay."):
        return text.rsplit(".", 1)[-1]
    return None


def _dtype_to_onnx(dtype: Any) -> int:
    """Convert a TensorPlay ``DType`` (or numpy dtype) to a TensorProto type."""

    name = _dtype_name(dtype)
    if name is not None:
        onnx_type = _NAME_TO_ONNX.get(name)
        if onnx_type is not None:
            return onnx_type
        raise TypeError(f"unsupported dtype for ONNX export: {dtype}")
    if isinstance(dtype, int):  # already a TensorProto value
        return dtype
    numpy_dtype = getattr(dtype, "numpy_dtype", None)
    if numpy_dtype is None:
        numpy_dtype = np.dtype(dtype)
    return _np_dtype_to_onnx(numpy_dtype)


def _dtype_to_numpy(dtype: Any) -> np.dtype:
    """Numpy dtype behind a TensorPlay ``DType``."""

    name = _dtype_name(dtype)
    if name is not None:
        numpy_dtype = _NAME_TO_NUMPY.get(name)
        if numpy_dtype is None:
            raise TypeError(f"unsupported dtype for ONNX export: {dtype}")
        return numpy_dtype
    return np.dtype(dtype)


def _np_dtype_to_onnx(np_dtype: Any) -> int:
    """Map a numpy dtype to the corresponding ``TensorProto`` data type."""

    np_dtype = np.dtype(np_dtype)
    onnx_type = _NUMPY_TO_ONNX.get(np_dtype)
    if onnx_type is None:
        raise TypeError(f"unsupported dtype for ONNX export: {np_dtype}")
    return onnx_type


def _onnx_to_np_dtype(onnx_type: int) -> np.dtype:
    """Inverse of :func:`_np_dtype_to_onnx`."""

    np_dtype = _ONNX_TO_NUMPY.get(int(onnx_type))
    if np_dtype is None:
        raise TypeError(f"no numpy equivalent for ONNX data type {onnx_type}")
    return np_dtype


def _size_to_tuple(size: Any) -> tuple:
    """Convert a TensorPlay ``Size``-like object to a plain tuple of ints."""

    if hasattr(size, "__iter__") and not isinstance(size, (str, bytes)):
        return tuple(int(dim) for dim in size)
    return (int(size),)


def _to_numpy(value: Any) -> np.ndarray:
    """Materialize a TensorPlay tensor / python value as a numpy array."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        tensor = value.detach() if hasattr(value, "detach") else value
        # bfloat16 has no numpy equivalent; round-trip through float32.
        if _dtype_name(getattr(tensor, "dtype", None)) == "bfloat16":
            tensor = tensor.to(_tp.float32)
        array = tensor.numpy()
        # ascontiguousarray would promote a 0-d tensor to shape (1,).
        if array.ndim and not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        return array
    if isinstance(value, bool):
        return np.asarray(value, dtype=np.bool_)
    if isinstance(value, int):
        return np.asarray(value, dtype=np.int64)
    if isinstance(value, float):
        return np.asarray(value, dtype=np.float32)
    array = np.asarray(value)
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    elif array.dtype == np.int32:
        array = array.astype(np.int64)
    return array
