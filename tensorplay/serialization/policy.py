"""Runtime policy and safety controls for checkpoint loading."""

from __future__ import annotations

import io
import mmap as _mmap
import os
import pickletools
import threading
import zipfile
from contextlib import contextmanager
from enum import Enum
from typing import Any


class LoadEndianness(Enum):
    NATIVE = 1
    LITTLE = 2
    BIG = 3


class _SerializationState(threading.local):
    def __init__(self):
        super().__init__()
        self.skip_data = False
        self.materialize_fake_tensors = False


_serialization_state = _SerializationState()
_default_load_endianness: LoadEndianness | None = None
_compute_crc32 = True
_default_mmap_options = getattr(_mmap, "MAP_PRIVATE", None)
_user_safe_globals: dict[str, Any] = {}


def _get_safe_global(module: str, name: str):
    return _user_safe_globals.get(f"{module}.{name}")


def _skip_payload_data() -> bool:
    return bool(_serialization_state.skip_data)


def get_default_load_endianness() -> LoadEndianness | None:
    return _default_load_endianness


def set_default_load_endianness(endianness):
    global _default_load_endianness
    if endianness is not None and not isinstance(endianness, LoadEndianness):
        raise TypeError("endianness must be LoadEndianness or None")
    _default_load_endianness = endianness


def get_crc32_options() -> bool:
    return _compute_crc32


def set_crc32_options(compute_crc32: bool):
    global _compute_crc32
    if not isinstance(compute_crc32, bool):
        raise TypeError("compute_crc32 must be bool")
    _compute_crc32 = compute_crc32


def get_default_mmap_options() -> int | None:
    return _default_mmap_options


def _mmap_access():
    """Translate the public mapping flags into the Python mmap access mode."""

    if _default_mmap_options == getattr(_mmap, "MAP_SHARED", None):
        return _mmap.ACCESS_WRITE
    return _mmap.ACCESS_COPY


def _mmap_file(fd: int, *, filename: str | os.PathLike[str] | None = None):
    access = _mmap_access()
    try:
        return _mmap.mmap(fd, 0, access=access)
    except (OSError, ValueError):
        if access != _mmap.ACCESS_WRITE or filename is None:
            raise
        flags = os.O_RDWR | int(getattr(os, "O_BINARY", 0))
        writable_fd = os.open(os.fspath(filename), flags)
        try:
            return _mmap.mmap(writable_fd, 0, access=access)
        finally:
            os.close(writable_fd)


class set_default_mmap_options:
    def __init__(self, flags: int):
        global _default_mmap_options
        private = getattr(_mmap, "MAP_PRIVATE", None)
        shared = getattr(_mmap, "MAP_SHARED", None)
        if flags not in {private, shared}:
            raise ValueError("flags must be mmap.MAP_PRIVATE or mmap.MAP_SHARED")
        self.previous = _default_mmap_options
        _default_mmap_options = flags

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _default_mmap_options
        _default_mmap_options = self.previous


def _global_name(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not module or not qualname:
        raise TypeError("safe globals must expose __module__ and __qualname__")
    return f"{module}.{qualname}"


def _normalize_safe_globals(values) -> dict[str, Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError("safe_globals must be an iterable of objects")
    normalized = {}
    for value in values:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("safe global tuples must contain two items")
            obj, name = value
            if not isinstance(name, str) or not name:
                raise TypeError("safe global names must be non-empty strings")
        else:
            obj, name = value, _global_name(value)
        normalized[name] = obj
    return normalized


def clear_safe_globals() -> None:
    _user_safe_globals.clear()


def get_safe_globals() -> list[Any]:
    return list(_user_safe_globals.values())


def add_safe_globals(values) -> None:
    _user_safe_globals.update(_normalize_safe_globals(values))


class safe_globals:
    def __init__(self, values):
        self.values = _normalize_safe_globals(values)
        self.previous = None

    def __enter__(self):
        self.previous = dict(_user_safe_globals)
        _user_safe_globals.update(self.values)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _user_safe_globals.clear()
        _user_safe_globals.update(self.previous or {})


_DEFAULT_SAFE_GLOBAL_NAMES = {
    "torch.Tensor",
    "torch.Size",
    "torch.device",
    "torch.strided",
    "torch.sparse_coo",
    "torch.sparse_csr",
    "torch.sparse_csc",
    "torch.sparse_bsr",
    "torch.sparse_bsc",
    "torch._utils._rebuild_tensor",
    "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_tensor_v3",
    "torch._utils._rebuild_parameter",
    "torch._utils._rebuild_parameter_with_state",
    "torch.nn.parameter._rebuild_parameter",
    "torch.nn.parameter.Parameter",
    "torch.serialization._get_layout",
    "torch.storage.TypedStorage",
    "torch.storage.UntypedStorage",
    "collections.OrderedDict",
    "collections.defaultdict",
    "collections.Counter",
    "_codecs.encode",
    "builtins.bytearray",
    "builtins.complex",
    "builtins.frozenset",
    "builtins.set",
    "builtins.slice",
    "copyreg._reconstructor",
    "copyreg.__newobj__",
    "numpy.dtype",
    "numpy.core.multiarray._reconstruct",
    "numpy._core.multiarray._reconstruct",
}


def _checkpoint_data_pickle(fileobj) -> bytes:
    start = fileobj.tell()
    try:
        fileobj.seek(0)
        with zipfile.ZipFile(fileobj) as archive:
            names = set(archive.namelist())
            if "data.pkl" in names:
                return archive.read("data.pkl")
            suffix = "/data.pkl"
            roots = {name[:-len("data.pkl")] for name in names if name.endswith(suffix)}
            if len(roots) == 1:
                return archive.read(next(iter(roots)) + "data.pkl")
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("expected a zip checkpoint with data.pkl") from error
    finally:
        fileobj.seek(start)
    raise ValueError("checkpoint does not contain data.pkl")


def get_unsafe_globals_in_checkpoint(f) -> list[str]:
    should_close = False
    if isinstance(f, (str, os.PathLike)):
        f = open(os.fspath(f), "rb")
        should_close = True
    try:
        data = _checkpoint_data_pickle(f)
        names = set()
        string_stack = []
        for opcode, arg, _position in pickletools.genops(data):
            if opcode.name == "GLOBAL" and isinstance(arg, str):
                module, name = arg.split(" ", 1)
                names.add(f"{module}.{name}")
            elif opcode.name in {
                "BINSTRING",
                "SHORT_BINSTRING",
                "BINUNICODE",
                "SHORT_BINUNICODE",
                "UNICODE",
            }:
                if isinstance(arg, bytes):
                    try:
                        arg = arg.decode("utf-8")
                    except UnicodeDecodeError:
                        arg = None
                string_stack.append(arg if isinstance(arg, str) else None)
            elif opcode.name == "STACK_GLOBAL" and len(string_stack) >= 2:
                module, name = string_stack[-2:]
                if isinstance(module, str) and isinstance(name, str):
                    names.add(f"{module}.{name}")
                del string_stack[-2:]
        return sorted(names - _DEFAULT_SAFE_GLOBAL_NAMES - set(_user_safe_globals))
    finally:
        if should_close:
            f.close()


class skip_data:
    def __init__(self, materialize_fake_tensors: bool = False):
        if not isinstance(materialize_fake_tensors, bool):
            raise TypeError("materialize_fake_tensors must be bool")
        self.materialize_fake_tensors = materialize_fake_tensors
        self.previous = None

    def __enter__(self):
        self.previous = (
            _serialization_state.skip_data,
            _serialization_state.materialize_fake_tensors,
        )
        _serialization_state.skip_data = True
        _serialization_state.materialize_fake_tensors = self.materialize_fake_tensors
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        old_skip, old_materialize = self.previous
        _serialization_state.skip_data = old_skip
        _serialization_state.materialize_fake_tensors = old_materialize


@contextmanager
def serialization_state():
    yield _serialization_state


__all__ = [
    "LoadEndianness",
    "add_safe_globals",
    "clear_safe_globals",
    "get_crc32_options",
    "get_default_load_endianness",
    "get_default_mmap_options",
    "get_safe_globals",
    "get_unsafe_globals_in_checkpoint",
    "safe_globals",
    "serialization_state",
    "set_crc32_options",
    "set_default_load_endianness",
    "set_default_mmap_options",
    "skip_data",
]
