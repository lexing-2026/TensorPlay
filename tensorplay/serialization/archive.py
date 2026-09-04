"""Checkpoint archives and tensor payload readers."""

from __future__ import annotations

import copyreg
import io
import json
import os
import pickle
import struct
import sys
import tarfile
import zipfile
import zlib
from collections import OrderedDict
from types import FunctionType
from typing import Any, BinaryIO, Mapping

import tensorplay as tp

from .weights_only import WeightsOnlyUnpickler

__all__ = [
    "read_torch_file",
    "write_torch_file",
    "write_legacy_torch_file",
    "describe_torch_file",
    "read_safetensors_file",
    "write_safetensors_file",
    "describe_safetensors_file",
    "_ITEMSIZE",
    "_NUMPY_DTYPES",
    "_STORAGE_NAMES_BY_DTYPE",
    "_TORCH_STORAGE_DTYPES",
    "_contiguous_stride",
    "_dtype_name_of",
    "_tensor_bytes",
    "_tensor_from_flat_bytes",
    "_rebuild_tensor",
    "_rebuild_tensor_v2",
    "_rebuild_tensor_v3",
    "_rebuild_parameter_with_state",
    "_WeightsOnlyUnpickler",
]


TORCH_MAGIC_NUMBER = 0x1950A86A20F9469CFC6C
TORCH_PROTOCOL_VERSION = 1001


# ---------------------------------------------------------------------------
# dtype plumbing
# ---------------------------------------------------------------------------

_TORCH_STORAGE_DTYPES = {
    "DoubleStorage": "float64",
    "FloatStorage": "float32",
    "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "ShortStorage": "int16",
    "CharStorage": "int8",
    "ByteStorage": "uint8",
    "BoolStorage": "bool",
    "UInt16Storage": "uint16",
    "UInt32Storage": "uint32",
    "UInt64Storage": "uint64",
    "ComplexDoubleStorage": "complex128",
    "ComplexFloatStorage": "complex64",
    "UntypedStorage": "uint8",
}

_STORAGE_NAMES_BY_DTYPE = {
    "float64": "DoubleStorage",
    "float32": "FloatStorage",
    "float16": "HalfStorage",
    "bfloat16": "BFloat16Storage",
    "int64": "LongStorage",
    "int32": "IntStorage",
    "int16": "ShortStorage",
    "int8": "CharStorage",
    "uint8": "ByteStorage",
    "uint16": "UInt16Storage",
    "uint32": "UInt32Storage",
    "uint64": "UInt64Storage",
    "bool": "BoolStorage",
    "complex128": "ComplexDoubleStorage",
    "complex64": "ComplexFloatStorage",
}

_ITEMSIZE = {
    "bool": 1, "uint8": 1, "int8": 1, "int16": 2, "uint16": 2, "int32": 4,
    "uint32": 4, "int64": 8, "uint64": 8, "float16": 2, "bfloat16": 2,
    "float32": 4, "float64": 8, "complex64": 8, "complex128": 16,
}

_NUMPY_DTYPES = {
    "bool": "?", "uint8": "u1", "int8": "i1", "int16": "i2", "int32": "i4",
    "int64": "i8", "uint16": "u2", "uint32": "u4", "uint64": "u8", "float16": "f2",
    "float32": "f4", "float64": "f8", "complex64": "c8", "complex128": "c16",
}


def _tp_dtype(name: str):
    dtype = getattr(tp, name, None)
    if dtype is None:
        raise NotImplementedError(
            f"TensorPlay does not expose the {name!r} dtype required by this checkpoint"
        )
    return dtype


def _dtype_name_of(tensor) -> str:
    for name in _NUMPY_DTYPES:
        if tensor.dtype == _tp_dtype(name):
            return name
    if tensor.dtype == _tp_dtype("bfloat16"):
        return "bfloat16"
    raise NotImplementedError(f"unsupported TensorPlay dtype: {tensor.dtype}")


def _contiguous_stride(shape: list) -> list:
    stride = [0] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= shape[index]
    return stride


def _tensor_bytes(tensor) -> bytes:
    """Return the raw contiguous CPU bytes of ``tensor``."""

    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if _device_string_of(tensor) != "cpu":
        tensor = tensor.to(tp.Device(tp.DeviceType.CPU))
    dtype_name = _dtype_name_of(tensor)
    if dtype_name == "bfloat16":
        return tensor.view(_tp_dtype("uint16")).numpy().tobytes()
    return tensor.numpy().tobytes()


def _tensor_from_flat_bytes(data: bytes, dtype_name: str):
    if dtype_name not in _ITEMSIZE:
        raise NotImplementedError(f"unsupported storage dtype: {dtype_name!r}")
    if len(data) % _ITEMSIZE[dtype_name]:
        raise ValueError(f"payload size is not divisible by dtype size for {dtype_name!r}")
    numel = len(data) // _ITEMSIZE[dtype_name]
    if numel == 0:
        return None
    if dtype_name == "bfloat16":
        flat = tp.Tensor._from_bytes(data, [numel], _tp_dtype("uint16"))
        return flat.view(_tp_dtype("bfloat16"))
    return tp.Tensor._from_bytes(data, [numel], _tp_dtype(dtype_name))


def _reshape_or_view(flat, size: list, stride: list, storage_offset: int):
    size = [int(dim) for dim in size]
    stride = [int(dim) for dim in stride]
    storage_offset = int(storage_offset)
    if any(dim < 0 for dim in size) or any(step < 0 for step in stride):
        raise ValueError("negative tensor dimensions and strides are not supported")
    if len(size) != len(stride) or storage_offset < 0:
        raise ValueError("invalid tensor view metadata")
    numel = 1
    for dim in size:
        numel *= dim
    if numel == 0 and any(dim == 0 for dim in size):
        return tp.empty(size, dtype=flat.dtype, device=flat.device)
    if flat is None:
        raise ValueError("non-empty tensor is missing its storage payload")
    max_index = storage_offset
    for dim, step in zip(size, stride):
        if dim:
            max_index += (dim - 1) * step
    if max_index >= int(flat.numel()):
        raise ValueError("tensor view exceeds its backing storage")
    if storage_offset == 0 and stride == _contiguous_stride(size):
        return flat.reshape(size) if size else flat.reshape([])
    return flat.as_strided(size, stride, storage_offset)


def _byteswap_bytes(data: bytes, dtype_name: str) -> bytes:
    import numpy as np

    if dtype_name == "bfloat16":
        dtype_name = "uint16"
    itemsize = np.dtype(_NUMPY_DTYPES[dtype_name]).itemsize
    if itemsize == 1 or not data:
        return data
    array = np.frombuffer(data, dtype=_NUMPY_DTYPES[dtype_name])
    return array.byteswap().tobytes()


def _parse_device(spec: str):
    if spec == "cpu":
        return tp.Device(tp.DeviceType.CPU)
    if ":" in spec:
        kind, index = spec.split(":", 1)
        return tp.Device(getattr(tp.DeviceType, kind.upper()), int(index))
    return tp.Device(getattr(tp.DeviceType, spec.upper()))


def _move_to(tensor, target):
    if target is None or target == "cpu":
        return tensor
    return tensor.to(_parse_device(target))


def _device_string_of(tensor) -> str:
    raw = tensor.device
    if raw.type == tp.DeviceType.CPU:
        return "cpu"
    if raw.type == tp.DeviceType.CUDA:
        return f"cuda:{raw.index or 0}"
    return str(raw)


# ---------------------------------------------------------------------------
# allowlisted unpickler
# ---------------------------------------------------------------------------


class _StorageType:

    def __init__(self, storage_name: str):
        try:
            self.dtype_name = _TORCH_STORAGE_DTYPES[storage_name]
        except KeyError as error:
            raise NotImplementedError(
                f"unsupported serialized storage type: {storage_name!r}"
            ) from error

    def __repr__(self):
        return f"_StorageType({self.dtype_name})"


def _rebuild_tensor(storage, storage_offset, size, stride):
    return _rebuild_tensor_v2(storage, storage_offset, size, stride, False, {})


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad,
                       backward_hooks, metadata=None):
    flat = storage.materialize()
    size = [int(dim) for dim in size]
    stride = [int(dim) for dim in stride]
    tensor = _reshape_or_view(flat, size, stride, int(storage_offset))
    keepalive = getattr(flat, "_serialization_keepalive", None)
    if keepalive is not None:
        try:
            tensor._serialization_keepalive = keepalive
        except (AttributeError, TypeError):
            pass
    if requires_grad and hasattr(tensor, "requires_grad_"):
        try:
            tensor.requires_grad_(True)
        except (RuntimeError, TypeError):
            pass
    return tensor


def _dtype_name_from_dtype(dtype):
    for name in _ITEMSIZE:
        try:
            if dtype == _tp_dtype(name):
                return name
        except NotImplementedError:
            continue
    raise NotImplementedError(f"unsupported serialized dtype: {dtype!r}")


class _PendingStorageValue:
    def __init__(self, tensor):
        self.tensor = tensor

    def materialize(self):
        return self.tensor


def _rebuild_tensor_v3(storage, storage_offset, size, stride, requires_grad,
                       backward_hooks, dtype, metadata=None):
    flat = storage.materialize()
    dtype_name = _dtype_name_from_dtype(dtype)
    if getattr(storage, "dtype_name", dtype_name) != dtype_name:
        flat = flat.view(_tp_dtype(dtype_name))
    return _rebuild_tensor_v2(
        _PendingStorageValue(flat),
        storage_offset,
        size,
        stride,
        requires_grad,
        backward_hooks,
        metadata,
    )


_rebuild_tensor_v2.__module__ = "torch._utils"
_rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
_rebuild_tensor_v2._tp_torch_ref = ("torch._utils", "_rebuild_tensor_v2")
_rebuild_tensor.__module__ = "torch._utils"
_rebuild_tensor.__name__ = "_rebuild_tensor"
_rebuild_tensor.__qualname__ = "_rebuild_tensor"
_rebuild_tensor._tp_torch_ref = ("torch._utils", "_rebuild_tensor")
_rebuild_tensor_v3.__module__ = "torch._utils"
_rebuild_tensor_v3.__name__ = "_rebuild_tensor_v3"
_rebuild_tensor_v3.__qualname__ = "_rebuild_tensor_v3"
_rebuild_tensor_v3._tp_torch_ref = ("torch._utils", "_rebuild_tensor_v3")


def _rebuild_parameter(data, requires_grad, backward_hooks, process_dict=None):
    parameter_cls = getattr(tp.nn, "Parameter", None)
    if parameter_cls is not None:
        try:
            return parameter_cls(data, requires_grad=bool(requires_grad))
        except (TypeError, RuntimeError):
            pass
    return data


_rebuild_parameter.__module__ = "torch.nn.parameter"
_rebuild_parameter.__name__ = "_rebuild_parameter"
_rebuild_parameter.__qualname__ = "_rebuild_parameter"
_rebuild_parameter._tp_torch_ref = ("torch.nn.parameter", "_rebuild_parameter")


def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
    parameter = _rebuild_parameter(data, requires_grad, backward_hooks)
    if state and hasattr(parameter, "__dict__"):
        if isinstance(state, tuple) and len(state) == 2:
            dictionary, slots = state
            if isinstance(dictionary, dict):
                parameter.__dict__.update(dictionary)
            if isinstance(slots, dict):
                parameter.__dict__.update(slots)
        elif isinstance(state, dict):
            parameter.__dict__.update(state)
    return parameter


_rebuild_parameter_with_state.__module__ = "torch._utils"
_rebuild_parameter_with_state.__name__ = "_rebuild_parameter_with_state"
_rebuild_parameter_with_state.__qualname__ = "_rebuild_parameter_with_state"
_rebuild_parameter_with_state._tp_torch_ref = (
    "torch._utils", "_rebuild_parameter_with_state"
)


_WeightsOnlyUnpickler = WeightsOnlyUnpickler


def _make_global_resolver():
    import numpy as np

    def make_device(value="cpu", index=None):
        if isinstance(value, bytes):
            value = value.decode("ascii")
        text = str(value)
        if index is not None and ":" not in text:
            text = f"{text}:{int(index)}"
        return _parse_device(text)

    def resolve(module: str, name: str):
        if module == __name__ and name == "_StorageType":
            return _StorageType
        if module == "torch":
            if name.endswith("Storage"):
                return _StorageType(name)
            if name in {"strided", "sparse_coo", "sparse_csr", "sparse_csc",
                        "sparse_bsr", "sparse_bsc"}:
                return name
            if name == "_get_layout":
                return lambda layout: layout
            if name in _NUMPY_DTYPES or name == "bfloat16":
                return _tp_dtype(name)
            if name in {"Tensor"}:
                return tp.Tensor
            if name == "Size":
                return lambda value=(): tuple(int(item) for item in value)
            if name == "device":
                return make_device
            return None
        if module == "torch.serialization" and name == "_get_layout":
            return lambda layout: layout
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_tensor_v3":
                return _rebuild_tensor_v3
            if name == "_rebuild_parameter_with_state":
                return _rebuild_parameter_with_state
            if name == "_rebuild_tensor":
                return _rebuild_tensor
            if name == "_rebuild_parameter":
                return _rebuild_parameter
            if name == "_rebuild_sparse_tensor":
                raise NotImplementedError(
                )
            return None
        if module == "torch.storage" and name in {
            "TypedStorage", "UntypedStorage"
        }:
            return _StorageType("UntypedStorage")
        if module == "torch.nn.parameter" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch.nn.parameter" and name == "Parameter":
            return getattr(tp.nn, "Parameter", tp.Tensor)
        if module == "_codecs" and name == "encode":
            from codecs import encode

            return encode
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "collections" and name == "defaultdict":
            from collections import defaultdict

            return defaultdict
        if module == "collections" and name == "Counter":
            from collections import Counter

            return Counter
        if module == "builtins" and name in {
            "bytearray", "complex", "frozenset", "set", "slice"
        }:
            return getattr(__import__("builtins"), name)
        if module == "copyreg" and name in {"_reconstructor", "__newobj__"}:
            return getattr(copyreg, name)
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "_reconstruct":
            return np.core.multiarray._reconstruct
        if module in {"numpy", "numpy.dtypes"} and name.startswith("unsignedinteger"):
            return None
        return None

    return resolve


def _convert_numpy_leaves(obj):
    """Replace numpy arrays left in an unpickled structure by tp tensors."""

    import numpy as np

    if isinstance(obj, np.ndarray):
        dtype_name = str(obj.dtype.name)
        if dtype_name not in _NUMPY_DTYPES:
            raise NotImplementedError(f"numpy dtype {dtype_name} is not supported")
        data = np.ascontiguousarray(obj).tobytes()
        flat = _tensor_from_flat_bytes(data, dtype_name)
        if flat is None:
            return tp.empty(list(obj.shape), dtype=_tp_dtype(dtype_name))
        return _reshape_or_view(flat, list(obj.shape), _contiguous_stride(list(obj.shape)), 0)
    if isinstance(obj, dict):
        return {key: _convert_numpy_leaves(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_leaves(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_numpy_leaves(value) for value in obj)
    return obj


# ---------------------------------------------------------------------------
# zip payload access
# ---------------------------------------------------------------------------


class _KeepAliveDict(OrderedDict):
    pass


class _KeepAliveList(list):
    pass


class _KeepAliveTuple(tuple):
    pass


def _attach_keepalive(value, owner):
    if isinstance(value, OrderedDict):
        result = _KeepAliveDict(value)
        result._serialization_keepalive = owner
        return result
    if isinstance(value, dict):
        result = _KeepAliveDict(value)
        result._serialization_keepalive = owner
        return result
    if isinstance(value, list):
        result = _KeepAliveList(value)
        result._serialization_keepalive = owner
        return result
    if isinstance(value, tuple):
        result = _KeepAliveTuple(value)
        result._serialization_keepalive = owner
        return result
    return value


def _zip_data_offset(archive, name: str):
    """Return the local payload offset for an uncompressed archive member."""

    try:
        info = archive.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
            return None
        fp = archive.fp
        fileno = fp.fileno()
        header = os.pread(fileno, 30, int(info.header_offset))
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            return None
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        return int(info.header_offset) + 30 + name_length + extra_length
    except (AttributeError, OSError, KeyError, TypeError, ValueError):
        return None


def _open_zip_mapping(archive):
    try:
        from .policy import _mmap_file

        fileno = archive.fp.fileno()
        size = os.fstat(fileno).st_size
        if size == 0:
            return None
        filename = getattr(archive.fp, "name", None)
        return _mmap_file(fileno, filename=filename)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _mmap_flat(mapping, offset: int, numel: int, dtype_name: str):
    import numpy as np

    if numel == 0:
        return tp.empty((0,), dtype=_tp_dtype(dtype_name))
    np_name = "u2" if dtype_name == "bfloat16" else _NUMPY_DTYPES[dtype_name]
    array = np.frombuffer(mapping, dtype=np_name, count=numel, offset=offset)
    if dtype_name == "bfloat16":
        try:
            flat = tp.from_dlpack(array)
        except (AttributeError, TypeError):
            flat = tp.from_dlpack(array.__dlpack__())
        flat = flat.view(_tp_dtype("bfloat16"))
    else:
        try:
            flat = tp.from_dlpack(array)
        except (AttributeError, TypeError):
            flat = tp.from_dlpack(array.__dlpack__())
    try:
        flat._serialization_keepalive = mapping
    except (AttributeError, TypeError):
        pass
    return flat


class _LazyZipStorage:
    def __init__(self, archive: zipfile.ZipFile, reader: "_ReaderState",
                 key: str, dtype_name: str, location: str, numel: int,
                 mapping=None):
        self._archive = archive
        self._reader = reader
        self.key = key
        self.dtype_name = dtype_name
        self.location = location
        self.numel = max(numel, 0)
        self.mapping = mapping
        self._flat = None

    def materialize(self):
        if self._flat is None:
            record = f"data/{self.key}"
            expected = self.numel * _ITEMSIZE[self.dtype_name]
            flat = None
            from .policy import _skip_payload_data

            skip = _skip_payload_data()
            if self.mapping is not None and not self._reader.swap_bytes and not skip:
                offset = _zip_data_offset(self._archive, record)
                if offset is not None:
                    try:
                        flat = _mmap_flat(self.mapping, offset, self.numel, self.dtype_name)
                    except (BufferError, ValueError, RuntimeError):
                        flat = None
            if flat is None:
                data = b"\x00" * expected if skip else self._archive.read(record)
                if len(data) != expected:
                    raise ValueError(
                        f"archive storage {self.key!r} has {len(data)} bytes, "
                        f"expected {expected}"
                    )
                if self._reader.swap_bytes and not skip:
                    data = _byteswap_bytes(data, self.dtype_name)
                flat = _tensor_from_flat_bytes(data, self.dtype_name)
            if flat is None:
                flat = tp.empty((0,), dtype=_tp_dtype(self.dtype_name))
            self._flat = _move_to(flat, self._reader.resolve_location(self.location))
        return self._flat


class _ReaderState:
    def __init__(self, map_location):
        self.map_location = map_location
        self.swap_bytes = False

    def resolve_location(self, location: str):
        from .common import resolve_map_location

        return resolve_map_location(self.map_location, location)


class _RootedZipReader:

    def __init__(self, archive: zipfile.ZipFile, prefix: str):
        self._archive = archive
        self._prefix = prefix

    def namelist(self) -> list:
        cut = len(self._prefix)
        return [name[cut:] if name.startswith(self._prefix) else name
                for name in self._archive.namelist()]

    def read(self, name: str) -> bytes:
        return self._archive.read(self._prefix + name)

    def getinfo(self, name: str):
        return self._archive.getinfo(self._prefix + name)

    @property
    def fp(self):
        return self._archive.fp


def _normalize_torch_zip_root(archive: zipfile.ZipFile) -> zipfile.ZipFile:
    names = set(archive.namelist())
    if "data.pkl" in names:
        return archive
    roots = {name[:-len("data.pkl")] for name in names if name.endswith("/data.pkl")}
    if len(roots) == 1:
        return _RootedZipReader(archive, next(iter(roots)))
    return archive


def _read_zip_archive(
    archive: zipfile.ZipFile,
    *,
    map_location,
    mmap: bool = False,
    pickle_load_args: dict | None = None,
) -> Any:
    archive = _normalize_torch_zip_root(archive)
    names = set(archive.namelist())
    if "constants.pkl" in names:
        raise RuntimeError(
            "modules. Export the weights as a state_dict instead."
        )
    if "data.pkl" not in names:
        raise ValueError("not a legacy checkpoint: missing data.pkl record")

    state = _ReaderState(map_location)
    from .policy import _skip_payload_data

    mapping = _open_zip_mapping(archive) if mmap and not _skip_payload_data() else None
    storage_cache = {}
    if "byteorder" in names:
        byteorder = archive.read("byteorder").decode("ascii")
        if byteorder not in {"little", "big"}:
            raise ValueError(f"unknown byteorder record: {byteorder!r}")
        state.swap_bytes = byteorder != sys.byteorder
    else:
        from .policy import LoadEndianness, get_default_load_endianness

        fallback = get_default_load_endianness()
        if fallback == LoadEndianness.BIG:
            state.swap_bytes = sys.byteorder != "big"
        elif fallback == LoadEndianness.NATIVE:
            state.swap_bytes = False
        else:
            state.swap_bytes = sys.byteorder != "little"

    def persistent_load(saved_id):
        if not isinstance(saved_id, tuple) or not saved_id or saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id!r}")
        if len(saved_id) < 5:
            raise pickle.UnpicklingError(f"invalid storage id: {saved_id!r}")
        storage_type, key, location, numel = saved_id[1:5]
        if isinstance(location, bytes):
            location = location.decode("ascii")
        cache_key = (str(key), str(getattr(storage_type, "dtype_name", storage_type)))
        storage = storage_cache.get(cache_key)
        if storage is None:
            dtype_name = getattr(storage_type, "dtype_name", None)
            if dtype_name is None:
                raise pickle.UnpicklingError(f"invalid storage type: {storage_type!r}")
            storage = _LazyZipStorage(
                archive, state, str(key), dtype_name, str(location), int(numel), mapping
            )
            storage_cache[cache_key] = storage
        return storage

    unpickler = _WeightsOnlyUnpickler(
        io.BytesIO(archive.read("data.pkl")),
        persistent_load=persistent_load,
        resolve_global=_make_global_resolver(),
        **(pickle_load_args or {}),
    )
    try:
        result = _convert_numpy_leaves(unpickler.load())
    except BaseException:
        if mapping is not None:
            mapping.close()
        raise
    if mapping is not None:
        return _attach_keepalive(result, mapping)
    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _read_magic_number_stream(
    fileobj: BinaryIO,
    *,
    map_location,
    pickle_load_args: dict | None = None,
) -> Any:
    state = _ReaderState(map_location)
    load_args = pickle_load_args or {}

    trivial_resolver = lambda module, name: None
    magic = _WeightsOnlyUnpickler(
        fileobj, persistent_load=lambda saved_id: None,
        resolve_global=trivial_resolver,
        **load_args,
    ).load()
    if magic != TORCH_MAGIC_NUMBER:
        raise ValueError(f"bad magic number {magic!r}; corrupt legacy checkpoint?")
    protocol_version = _WeightsOnlyUnpickler(
        fileobj, persistent_load=lambda saved_id: None,
        resolve_global=trivial_resolver,
        **load_args,
    ).load()
    if protocol_version != TORCH_PROTOCOL_VERSION:
        raise ValueError(f"unsupported legacy protocol version {protocol_version!r}")
    sys_info = _WeightsOnlyUnpickler(
        fileobj, persistent_load=lambda saved_id: None,
        resolve_global=trivial_resolver,
        **load_args,
    ).load()
    if isinstance(sys_info, dict):
        little_endian = bool(sys_info.get("little_endian", True))
        state.swap_bytes = little_endian != (sys.byteorder == "little")

    pending: "OrderedDict[str, dict]" = OrderedDict()
    deferred_views: list = []

    def persistent_load(saved_id):
        if not isinstance(saved_id, tuple) or not saved_id or saved_id[0] != "storage":
            if saved_id and saved_id[0] == "module":
                return saved_id[1]
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id!r}")
        data = saved_id[1:]
        storage_type, key, location, numel = data[0], str(data[1]), data[2], data[3]
        view_metadata = data[4] if len(data) > 4 else None
        if isinstance(location, bytes):
            location = location.decode("ascii")
        slot = pending.get(key)
        if slot is None:
            # Payload bytes trail the pickle stream, so allocate the backing
            # fill it in place afterwards -- rebuilt views reference it.
            dtype_name = getattr(storage_type, "dtype_name", None)
            if dtype_name is None:
                raise pickle.UnpicklingError(
                    f"invalid storage type: {storage_type!r}"
                )
            target = state.resolve_location(str(location))
            numel = max(int(numel), 0)
            if isinstance(target, tp.Tensor):
                if int(target.numel()) != numel:
                    raise ValueError(
                        "map_location returned a tensor with an incompatible size"
                    )
                tensor = target.reshape((numel,))
            elif target is not None and target != "cpu":
                tensor = tp.empty((numel,), dtype=_tp_dtype(dtype_name),
                                  device=_parse_device(target))
            else:
                tensor = tp.empty((numel,), dtype=_tp_dtype(dtype_name))
            slot = {"dtype": dtype_name, "location": str(location),
                    "numel": numel, "tensor": tensor}
            pending[key] = slot
        if view_metadata is not None:
            view_key, offset, view_numel = view_metadata
            deferred_views.append((str(view_key), key, int(offset), int(view_numel)))
        return _PendingStorage(slot, state)

    unpickler = _WeightsOnlyUnpickler(
        fileobj,
        persistent_load=persistent_load,
        resolve_global=_make_global_resolver(),
        **load_args,
    )
    result = unpickler.load()

    # Storage payloads follow the pickle stream, consumed in the order of the
    stored_keys = _WeightsOnlyUnpickler(
        fileobj, persistent_load=lambda saved_id: None,
        resolve_global=trivial_resolver,
        **load_args,
    ).load()

    def fill(slot):
        raw_numel = fileobj.read(8)
        if len(raw_numel) != 8:
            raise ValueError("truncated legacy checkpoint storage header")
        (stored_numel,) = struct.unpack("<q", raw_numel)
        if stored_numel < 0:
            raise ValueError("legacy checkpoint storage has a negative size")
        if stored_numel != slot["numel"]:
            raise ValueError(
                f"legacy checkpoint storage size changed from {slot['numel']} "
                f"to {stored_numel}"
            )
        itemsize = _ITEMSIZE[slot["dtype"]]
        nbytes = stored_numel * itemsize
        data = fileobj.read(nbytes)
        if len(data) != nbytes:
            raise ValueError("truncated legacy checkpoint storage")
        if state.swap_bytes:
            data = _byteswap_bytes(data, slot["dtype"])
        incoming = _tensor_from_flat_bytes(data, slot["dtype"])
        if incoming is None:
            return
        slot["tensor"].copy_(incoming.reshape(slot["tensor"].shape))

    for key in stored_keys:
        key = str(key)
        if key in pending:
            fill(pending[key])
    for view_key, root_key, offset, view_numel in deferred_views:
        base = pending[root_key]["tensor"]
        pending[view_key] = {
            "dtype": pending[root_key]["dtype"],
            "location": pending[root_key]["location"],
            "numel": view_numel,
            "tensor": base.as_strided([view_numel], [1], offset),
        }
    return _convert_numpy_leaves(result)


class _PendingStorage:
    def __init__(self, slot: dict, state: _ReaderState):
        self.slot = slot
        self.state = state

    def materialize(self):
        return self.slot["tensor"]


# ---------------------------------------------------------------------------
# ancient tar format (pre-0.1, reading only)
# ---------------------------------------------------------------------------


def _read_tar_archive(archive: tarfile.TarFile, *, map_location) -> Any:
    state = _ReaderState(map_location)
    slots: OrderedDict[str, dict] = OrderedDict()

    def safe_load(stream):
        return _WeightsOnlyUnpickler(
            stream,
            persistent_load=lambda saved_id: None,
            resolve_global=_make_global_resolver(),
        ).load()

    with archive.extractfile("storages") as handle:
        stream = io.BytesIO(handle.read())
        num_storages = safe_load(stream)
        for _ in range(num_storages):
            key, location, storage_type = safe_load(stream)
            (numel,) = struct.unpack("<q", stream.read(8))
            dtype_name = storage_type.dtype_name
            nbytes = int(numel) * _ITEMSIZE[dtype_name]
            data = stream.read(nbytes)
            if len(data) != nbytes:
                raise ValueError(f"truncated legacy tar storage {key!r}")
            if state.swap_bytes:
                data = _byteswap_bytes(data, dtype_name)
            flat = _tensor_from_flat_bytes(data, dtype_name)
            if flat is None:
                flat = tp.empty((max(int(numel), 0),), dtype=_tp_dtype(dtype_name))
            slots[str(key)] = {
                "tensor": _move_to(flat, state.resolve_location(str(location))),
            }

    specs = []
    with archive.extractfile("tensors") as handle:
        stream = io.BytesIO(handle.read())
        num_tensors = safe_load(stream)
        for _ in range(num_tensors):
            key, storage_id, _original_type = safe_load(stream)
            (ndim,) = struct.unpack("<i", stream.read(4))
            stream.read(4)  # legacy treated ndim as 8 bytes
            size = list(struct.unpack(f"<{ndim}q", stream.read(8 * ndim))) if ndim else []
            stride = list(struct.unpack(f"<{ndim}q", stream.read(8 * ndim))) if ndim else []
            (storage_offset,) = struct.unpack("<q", stream.read(8))
            specs.append((str(key), str(storage_id), size, stride, int(storage_offset)))

    rebuilt = {}
    for key, storage_id, size, stride, storage_offset in specs:
        flat = slots[storage_id]["tensor"]
        rebuilt[key] = _reshape_or_view(flat, size, stride, storage_offset)

    def persistent_load(saved_id):
        if isinstance(saved_id, tuple):
            if saved_id and saved_id[0] == "module":
                return saved_id[1]
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id!r}")
        return rebuilt[str(saved_id)]

    with archive.extractfile("pickle") as handle:
        unpickler = _WeightsOnlyUnpickler(
            handle,
            persistent_load=persistent_load,
            resolve_global=_make_global_resolver(),
        )
        result = unpickler.load()
    return _convert_numpy_leaves(result)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def read_torch_file(
    fileobj: BinaryIO,
    *,
    map_location=None,
    mmap: bool = False,
    pickle_module=None,
    **pickle_load_args,
) -> Any:
    pickle_load_args.setdefault("encoding", "utf-8")
    position = int(fileobj.tell())
    head = fileobj.read(4)
    fileobj.seek(position)
    if head == b"PK\x03\x04":
        with zipfile.ZipFile(fileobj) as archive:
            return _read_zip_archive(
                archive,
                map_location=map_location,
                mmap=bool(mmap),
                pickle_load_args=pickle_load_args,
            )

    try:
        with tarfile.open(fileobj=fileobj, mode="r:") as archive:
            member_names = {member.name for member in archive.getmembers()}
            if {"storages", "tensors", "pickle"} <= member_names:
                return _read_tar_archive(archive, map_location=map_location)
    except (tarfile.TarError, EOFError, KeyError):
        pass

    fileobj.seek(position)
    try:
        probe = _WeightsOnlyUnpickler(
            fileobj,
            persistent_load=lambda saved_id: None,
            resolve_global=lambda module, name: None,
        ).load()
    except Exception:
        probe = None
    finally:
        fileobj.seek(position)
    if probe == TORCH_MAGIC_NUMBER:
        return _read_magic_number_stream(
            fileobj, map_location=map_location, pickle_load_args=pickle_load_args
        )

    raise ValueError("unrecognized archive stream")


class _PTStorageRef:

    __slots__ = ("key",)

    def __init__(self, key: str):
        self.key = key


def _make_pt_storage_class(dtype_name: str):
    storage_name = _STORAGE_NAMES_BY_DTYPE[dtype_name]

    cls = type(
        storage_name,
        (),
        {"__module__": "torch", "__qualname__": storage_name, "__name__": storage_name},
    )
    cls._tp_torch_ref = ("torch", storage_name)
    return cls


_PT_STORAGE_CLASSES = {
    dtype_name: _make_pt_storage_class(dtype_name) for dtype_name in _STORAGE_NAMES_BY_DTYPE
}


class _TorchCompatPickler(pickle._Pickler):
    """Pickler that emits the external storage class names required by the format."""

    def save_global(self, obj, name=None):
        forced = getattr(obj, "_tp_torch_ref", None)
        if forced is not None:
            module, forced_name = forced
            self.write(pickle.GLOBAL
                       + module.encode("ascii") + b"\n"
                       + forced_name.encode("ascii") + b"\n")
            self.memoize(obj)
            return
        super().save_global(obj, name)

    dispatch = dict(pickle._Pickler.dispatch)
    dispatch[FunctionType] = save_global


def _storage_nbytes(storage) -> int:
    value = getattr(storage, "nbytes", 0)
    value = value() if callable(value) else value
    return max(int(value), 0)


def _storage_bytes(tensor, storage_nbytes: int) -> bytes:
    """Read one complete backing storage into a contiguous byte string."""

    from .policy import _skip_payload_data

    if _skip_payload_data():
        return b"\x00" * storage_nbytes
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    if tensor.device.type != tp.DeviceType.CPU:
        tensor = tensor.to(tp.Device(tp.DeviceType.CPU))
    storage = tensor.untyped_storage()
    actual_nbytes = _storage_nbytes(storage)
    if actual_nbytes != storage_nbytes:
        raise RuntimeError(
            f"backing storage size changed from {storage_nbytes} to {actual_nbytes}"
        )
    if storage_nbytes == 0:
        return b""
    raw = tp.empty((0,), dtype=_tp_dtype("uint8"))
    raw.set_(storage, 0, [storage_nbytes], [1])
    return raw.numpy().tobytes()


def _serialize_object(obj: Any, pickle_protocol: int):
    storages: dict[tuple, dict] = {}
    order: list[dict] = []
    record_by_key: dict[str, dict] = {}

    def storage_record(tensor) -> _PTStorageRef:
        dtype_name = _dtype_name_of(tensor)
        storage = tensor.untyped_storage()
        storage_nbytes = _storage_nbytes(storage)
        itemsize = _ITEMSIZE[dtype_name]
        if storage_nbytes % itemsize:
            raise ValueError(
                f"storage size {storage_nbytes} is not divisible by {dtype_name} size"
            )
        identity = (int(getattr(storage, "_cdata", 0)), dtype_name)
        if identity[0] == 0:
            identity = (int(tensor.data_ptr()), storage_nbytes, dtype_name)
        record = storages.get(identity)
        if record is not None:
            return _PTStorageRef(record["key"])
        record = {
            "key": str(len(order)),
            "tensor": tensor,
            "dtype": dtype_name,
            "location": _device_string_of(tensor),
            "numel": storage_nbytes // itemsize,
            "storage_nbytes": storage_nbytes,
        }
        order.append(record)
        record_by_key[record["key"]] = record
        storages[identity] = record
        return _PTStorageRef(record["key"])

    def reduce_tensor(tensor):
        ref = storage_record(tensor)
        shape = [int(dim) for dim in tensor.shape]
        stride = [int(step) for step in tensor.stride()]
        return (
            _rebuild_tensor_v2,
            (
                ref,
                int(tensor.storage_offset()),
                shape,
                stride,
                bool(getattr(tensor, "requires_grad", False)),
                {},
            ),
        )

    def reduce_parameter(tensor):
        class _TensorReduction:
            def __reduce__(self):
                return reduce_tensor(tensor)

        return (
            _rebuild_parameter,
            (
                _TensorReduction(),
                bool(getattr(tensor, "requires_grad", True)),
                {},
                None,
            ),
        )

    def persistent_id(value):
        if isinstance(value, _PTStorageRef):
            record = record_by_key[value.key]
            storage_cls = _PT_STORAGE_CLASSES[record["dtype"]]
            return (
                "storage",
                storage_cls,
                record["key"],
                record["location"],
                record["numel"],
            )
        return None

    data_buf = io.BytesIO()
    pickler = _TorchCompatPickler(data_buf, pickle_protocol)
    dispatch = copyreg.dispatch_table.copy()
    dispatch[tp.Tensor] = reduce_tensor
    parameter_cls = getattr(tp.nn, "Parameter", None)
    if parameter_cls is not None and parameter_cls is not tp.Tensor:
        dispatch[parameter_cls] = reduce_parameter
    pickler.dispatch_table = dispatch
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    return data_buf.getvalue(), order


def _write_zip_member(archive, name: str, data: bytes, *, align: int | None = None):
    if align is None or align <= 1 or not name.startswith("archive/data/"):
        archive.writestr(name, data)
        return
    position = int(archive.fp.tell())
    name_bytes = name.encode("utf-8")
    padding = (-position - 30 - len(name_bytes)) % align
    if padding and padding < 4:
        padding += align
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    if padding:
        extra_payload = padding - 4
        if extra_payload > 0xFFFF:
            raise ValueError("archive alignment padding is too large")
        info.extra = b"\x00\x00" + struct.pack("<H", extra_payload) + b"\x00" * extra_payload
    archive.writestr(info, data)


def _raw_storage_tensor(tensor, storage_nbytes: int):
    if _device_string_of(tensor) != "cpu":
        return None
    storage = tensor.untyped_storage()
    raw = tp.empty((0,), dtype=_tp_dtype("uint8"))
    raw.set_(storage, 0, [int(storage_nbytes)], [1])
    return raw


def _raw_storage_view(tensor, storage_nbytes: int):
    raw = _raw_storage_tensor(tensor, storage_nbytes)
    if raw is None:
        return None, None
    array = raw.numpy()
    return raw, memoryview(array).cast("B")


def _direct_zip_member(archive, name: str, size: int, checksum: int,
                       *, align: int | None = None):
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    info.file_size = int(size)
    info.compress_size = int(size)
    info.CRC = int(checksum)
    position = int(archive.fp.tell())
    if align is not None and align > 1 and name.startswith("archive/data/"):
        name_bytes = name.encode("utf-8")
        padding = (-position - 30 - len(name_bytes)) % int(align)
        if padding and padding < 4:
            padding += int(align)
        if padding:
            extra_payload = padding - 4
            if extra_payload > 0xFFFF:
                raise ValueError("archive alignment padding is too large")
            info.extra = (
                b"\x00\x00" + struct.pack("<H", extra_payload)
                + b"\x00" * extra_payload
            )
    info.header_offset = position
    archive._writecheck(info)
    zip64 = info.file_size > zipfile.ZIP64_LIMIT
    archive.fp.write(info.FileHeader(zip64))
    archive.filelist.append(info)
    archive.NameToInfo[info.filename] = info
    archive.start_dir = int(archive.fp.tell())
    archive.fp.flush()
    return info


def _write_torch_file_direct(fileobj, data_value: bytes, order: list[dict],
                             *, disable_byteorder_record: bool,
                             storage_alignment: int) -> bool:
    from .policy import _skip_payload_data

    if _skip_payload_data():
        return False
    saver = getattr(tp.Tensor, "_save_file_segments", None)
    if not callable(saver):
        return False
    filename = getattr(fileobj, "name", None)
    if not isinstance(filename, (str, bytes, os.PathLike)):
        return False
    try:
        if int(fileobj.tell()) != 0:
            return False
    except (AttributeError, OSError, ValueError):
        return False

    raw_storages = []
    for record in order:
        raw, view = _raw_storage_view(record["tensor"], record["storage_nbytes"])
        if raw is None:
            return False
        raw_storages.append((record, raw, zlib.crc32(view) & 0xFFFFFFFF))

    with zipfile.ZipFile(
        fileobj, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        archive.writestr("archive/data.pkl", data_value)
        archive.writestr("archive/version", "3")
        archive.writestr("archive/.format_version", "1")
        archive.writestr("archive/.storage_alignment", str(int(storage_alignment)))
        if not disable_byteorder_record:
            archive.writestr("archive/byteorder", sys.byteorder)
        for record, raw, checksum in raw_storages:
            name = f"archive/data/{record['key']}"
            _direct_zip_member(
                archive,
                name,
                record["storage_nbytes"],
                checksum,
                align=storage_alignment,
            )
            saver(os.fspath(fileobj.name), [raw])
            archive.fp.seek(0, os.SEEK_END)
            archive.start_dir = int(archive.fp.tell())
    return True


def write_torch_file(
    fileobj: BinaryIO,
    obj: Any,
    *,
    pickle_protocol: int = 2,
    pickle_module=None,
    disable_byteorder_record: bool = False,
    storage_alignment: int = 64,
) -> None:
    storage_alignment = int(storage_alignment)
    if storage_alignment <= 0:
        raise ValueError("storage_alignment must be positive")
    data_value, order = _serialize_object(obj, pickle_protocol)
    if _write_torch_file_direct(
        fileobj,
        data_value,
        order,
        disable_byteorder_record=disable_byteorder_record,
        storage_alignment=storage_alignment,
    ):
        return
    with zipfile.ZipFile(
        fileobj, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        archive.writestr("archive/data.pkl", data_value)
        archive.writestr("archive/version", "3")
        archive.writestr("archive/.format_version", "1")
        archive.writestr("archive/.storage_alignment", str(int(storage_alignment)))
        if not disable_byteorder_record:
            archive.writestr("archive/byteorder", sys.byteorder)
        for record in order:
            payload = _storage_bytes(record["tensor"], record["storage_nbytes"])
            _write_zip_member(
                archive,
                f"archive/data/{record['key']}",
                payload,
                align=storage_alignment,
            )
    flush = getattr(fileobj, "flush", None)
    if callable(flush):
        flush()


def write_legacy_torch_file(
    fileobj: BinaryIO,
    obj: Any,
    *,
    pickle_protocol: int = 2,
    pickle_module=None,
) -> None:
    data_value, order = _serialize_object(obj, pickle_protocol)
    pickle.dump(TORCH_MAGIC_NUMBER, fileobj, protocol=pickle_protocol)
    pickle.dump(TORCH_PROTOCOL_VERSION, fileobj, protocol=pickle_protocol)
    pickle.dump(
        {
            "protocol_version": TORCH_PROTOCOL_VERSION,
            "little_endian": sys.byteorder == "little",
            "type_sizes": {"short": 2, "int": 4, "long": 8},
        },
        fileobj,
        protocol=pickle_protocol,
    )
    fileobj.write(data_value)
    pickle.dump([record["key"] for record in order], fileobj, protocol=pickle_protocol)
    for record in order:
        payload = _storage_bytes(record["tensor"], record["storage_nbytes"])
        fileobj.write(struct.pack("<q", record["numel"]))
        fileobj.write(payload)
    flush = getattr(fileobj, "flush", None)
    if callable(flush):
        flush()


def describe_torch_file(fileobj: BinaryIO) -> dict:

    head = fileobj.read(4)
    fileobj.seek(0)
    if head != b"PK\x03\x04":
        return {"format": "torch_stream_or_tar"}

    with zipfile.ZipFile(fileobj) as archive:
        archive = _normalize_torch_zip_root(archive)
        names = set(archive.namelist())
        if "constants.pkl" in names:
            return {"format": "torchscript_zip"}
        if "data.pkl" not in names:
            return {"format": "unknown_zip"}
        storages: dict[str, dict] = {}

        def persistent_load(saved_id):
            if not isinstance(saved_id, tuple) or len(saved_id) < 5:
                raise pickle.UnpicklingError(f"invalid storage id: {saved_id!r}")
            _, storage_type, key, location, numel = saved_id[:5]
            dtype_name = getattr(storage_type, "dtype_name", None)
            if dtype_name is None:
                raise pickle.UnpicklingError(f"invalid storage type: {storage_type!r}")
            storages[str(key)] = {
                "dtype": dtype_name,
                "numel": int(numel),
                "location": location.decode() if isinstance(location, bytes) else str(location),
            }
            return None

        def rebuild_stub(storage, storage_offset, size, stride, *args):
            return {"shape": [int(dim) for dim in size],
                    "dtype": getattr(storage, "dtype_name", None)}

        resolver = _make_inspect_resolver(rebuild_stub)
        unpickler = _WeightsOnlyUnpickler(
            io.BytesIO(archive.read("data.pkl")),
            persistent_load=persistent_load,
            resolve_global=resolver,
        )
        structure = unpickler.load()
        info = {
            "format": "torch_zip",
            "byteorder": archive.read("byteorder").decode("ascii")
            if "byteorder" in names else None,
            "storages": storages,
            "structure": _summarize(structure),
        }
        return info


def _make_inspect_resolver(rebuild_stub):
    def resolve(module: str, name: str):
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        if module == "torch._utils" and name in {
            "_rebuild_tensor_v2", "_rebuild_tensor_v3", "_rebuild_tensor"
        }:
            return rebuild_stub
        if module == "torch._utils" and name == "_rebuild_parameter_with_state":
            return lambda data, *args, **kwargs: data
        if module == "torch.nn.parameter" and name == "_rebuild_parameter":
            return lambda data, *args, **kwargs: data
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "numpy" and name == "dtype":
            import numpy as np

            return np.dtype
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "_reconstruct":
            import numpy as np

            return lambda *args, **kwargs: None
        return None

    return resolve


def _summarize(obj, depth: int = 0):
    if depth > 6:
        return "..."
    if isinstance(obj, _StorageType):
        return repr(obj)
    if isinstance(obj, dict):
        return {str(key): _summarize(value, depth + 1) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_summarize(value, depth + 1) for value in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return type(obj).__name__


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------

_SAFETENSORS_DTYPES = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "I32": "int32",
    "I64": "int64", "U32": "uint32", "U64": "uint64", "F16": "float16",
    "BF16": "bfloat16", "F32": "float32", "F64": "float64",
}

_SAFETENSORS_DTYPE_NAMES = {v: k for k, v in _SAFETENSORS_DTYPES.items()}


def _read_safetensors_header(fileobj: BinaryIO) -> tuple[dict, int]:
    raw_length = fileobj.read(8)
    if len(raw_length) != 8:
        raise ValueError("truncated safetensors file: missing header length")
    header_length = struct.unpack("<Q", raw_length)[0]
    if header_length > 256 * 1024 * 1024:
        raise ValueError("safetensors header is too large")
    header_bytes = fileobj.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("truncated safetensors file: truncated header")
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid safetensors header JSON: {error}") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("safetensors __metadata__ must be a string mapping")
    return header, 8 + header_length


def read_safetensors_file(fileobj: BinaryIO, *, map_location=None, mmap: bool = False) -> "OrderedDict[str, Any]":
    """Load a safetensors file as an ``OrderedDict`` of TensorPlay tensors.

    With ``mmap=True`` tensors become zero-copy views over a private mapping
    are paged in on first touch and never duplicated into anonymous memory.
    The returned dict keeps the mapping alive.  Falls back to an eager read
    when the file object has no real file descriptor or the platform needs a
    byte swap.  File-level ``__metadata__`` is surfaced by
    :func:`tensorplay.serialization.inspect_checkpoint`, not injected into the
    """

    header, data_start = _read_safetensors_header(fileobj)
    metadata = header.pop("__metadata__", {})

    target = None
    if map_location is not None:
        from .common import resolve_map_location

        target = resolve_map_location(map_location, "cpu")

    mapping = None
    from .policy import _skip_payload_data

    if mmap and sys.byteorder == "little" and not _skip_payload_data():
        import mmap as _mmap

        try:
            fileno = fileobj.fileno()
        except (AttributeError, OSError, io.UnsupportedOperation):
            fileno = None
        if fileno is not None and fileno >= 0:
            from .policy import _mmap_file

            filename = getattr(fileobj, "name", None)
            try:
                mapping = _mmap_file(fileno, filename=filename)
            except (OSError, TypeError, ValueError):
                mapping = None

    import numpy as np

    result: "OrderedDict[str, Any]" = OrderedDict()
    cursor = 0
    try:
        file_size = os.fstat(fileobj.fileno()).st_size
    except (AttributeError, OSError, io.UnsupportedOperation):
        position = fileobj.tell()
        fileobj.seek(0, os.SEEK_END)
        file_size = fileobj.tell()
        fileobj.seek(position)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError(f"safetensors tensor {name!r} spec must be an object")
        offsets = info.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool)
                       for value in offsets)
        ):
            raise ValueError(f"invalid data_offsets for tensor {name!r}")
        start, end = offsets
        if not (cursor <= start <= end):
            raise ValueError(
                f"safetensors tensor {name!r} has non-monotonic data_offsets "
                f"[{start}, {end})"
            )
        cursor = end
        dtype_name = _SAFETENSORS_DTYPES.get(info["dtype"])
        if dtype_name is None:
            raise NotImplementedError(
                f"unsupported safetensors dtype: {info['dtype']!r}"
            )
        shape = [int(dim) for dim in info["shape"]]
        if any(dim < 0 for dim in shape):
            raise ValueError(f"safetensors tensor {name!r} has a negative dimension")
        numel = 1
        for dim in shape:
            numel *= dim
        expected_nbytes = numel * _ITEMSIZE[dtype_name]
        if end - start != expected_nbytes:
            raise ValueError(
                f"safetensors tensor {name!r} has {end - start} payload bytes, "
                f"expected {expected_nbytes}"
            )
        if start < 0 or end > file_size - data_start:
            raise ValueError(f"safetensors tensor {name!r} payload is outside the file")

        skip = _skip_payload_data()
        tensor = None
        if mapping is not None and numel > 0 and not skip:
            tensor = _safetensors_mmap_tensor(mapping, data_start + start, end - start,
                                              shape, dtype_name)
        if tensor is None:
            data = (
                b"\x00" * (end - start)
                if skip
                else _read_exact(fileobj, data_start + start, end - start)
            )
            if sys.byteorder == "big":
                data = _byteswap_bytes(data, dtype_name)
            flat = _tensor_from_flat_bytes(data, dtype_name)
            if flat is None:
                tensor = tp.empty(shape, dtype=_tp_dtype(dtype_name))
            else:
                tensor = _reshape_or_view(flat, shape, _contiguous_stride(shape), 0)
        result[name] = _move_to(tensor, target)

    if mapping is not None:
        return _attach_keepalive(result, mapping)
    return result


def _read_exact(fileobj: BinaryIO, offset: int, length: int) -> bytes:
    fileobj.seek(offset)
    data = fileobj.read(length)
    if len(data) != length:
        raise ValueError("truncated safetensors payload")
    return data


def _safetensors_mmap_tensor(mapping, offset: int, length: int, shape: list,
                             dtype_name: str):
    """Zero-copy tensor over ``mapping[offset:offset+length]`` via DLPack.

    Chain: mmap -> numpy view (no copy) -> ``tp.from_dlpack`` (no copy).  The
    DLPack capsule keeps the numpy array (and through it the mapping) alive.
    Returns None when the zero-copy path is unavailable so callers can fall
    back to an eager read.
    """

    if dtype_name == "bfloat16":
        np_dtype = "u2"
    elif dtype_name in _NUMPY_DTYPES:
        np_dtype = _NUMPY_DTYPES[dtype_name]
    else:
        return None
    import numpy as np

    window = memoryview(mapping)[offset:offset + length]
    array = np.frombuffer(window, dtype=np_dtype)
    try:
        try:
            tensor = tp.from_dlpack(array)
        except (TypeError, AttributeError):
            tensor = tp.from_dlpack(array.__dlpack__())
    except Exception:
        return None
    if dtype_name == "bfloat16":
        tensor = tensor.view(_tp_dtype("bfloat16"))
    try:
        tensor._serialization_keepalive = mapping
    except (AttributeError, TypeError):
        pass
    result = _reshape_or_view(tensor, shape, _contiguous_stride(shape), 0)
    try:
        result._serialization_keepalive = mapping
    except (AttributeError, TypeError):
        pass
    return result


def write_safetensors_file(fileobj: BinaryIO, obj: Mapping[str, Any], *,
                           metadata: Mapping[str, str] | None = None) -> None:
    """Write a flat mapping of name -> tensor as a safetensors file."""

    if not isinstance(obj, Mapping):
        raise TypeError(
            "safetensors stores a flat name->tensor mapping only; use '.mega' "
            "or '.pt' for nested containers"
        )

    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    prepared = []
    from .policy import _skip_payload_data

    skip = _skip_payload_data()
    offset = 0
    for name, tensor in obj.items():
        if not isinstance(tensor, tp.Tensor):
            raise TypeError(
                f"safetensors requires a flat mapping of name to Tensor; "
                f"got non-tensor value at {name!r}"
            )
        name = str(name)
        if name == "__metadata__":
            raise ValueError("__metadata__ is reserved for file metadata")
        dtype_name = _dtype_name_of(tensor)
        st_name = _SAFETENSORS_DTYPE_NAMES.get(dtype_name)
        if st_name is None:
            raise NotImplementedError(
                f"dtype {dtype_name!r} is not representable in safetensors"
            )
        nbytes = int(tensor.numel()) * _ITEMSIZE[dtype_name]
        header[name] = {
            "dtype": st_name,
            "shape": [int(dim) for dim in tensor.shape],
            "data_offsets": [offset, offset + nbytes],
        }
        prepared.append((tensor, nbytes))
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * padding

    fileobj.write(struct.pack("<Q", len(header_bytes)))
    fileobj.write(header_bytes)
    for tensor, nbytes in prepared:
        data = b"\x00" * nbytes if skip else _tensor_bytes(tensor)
        if len(data) != nbytes:
            raise RuntimeError(
                f"internal error: expected {nbytes} payload bytes, got {len(data)}"
            )
        fileobj.write(data)
    flush = getattr(fileobj, "flush", None)
    if callable(flush):
        flush()


def describe_safetensors_file(fileobj: BinaryIO) -> dict:
    header, _data_start = _read_safetensors_header(fileobj)
    metadata = header.get("__metadata__", {})
    tensors = {}
    total = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        total = max(total, end)
        tensors[name] = {
            "shape": [int(dim) for dim in info["shape"]],
            "dtype": info["dtype"],
            "nbytes": end - start,
        }
    return {"format": "safetensors", "metadata": metadata, "tensors": tensors,
            "payload_nbytes": total}
