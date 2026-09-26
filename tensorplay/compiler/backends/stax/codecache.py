"""Where compiled artifacts are kept, and how a name is settled for one.

A compiled artifact is named after what it was built from rather than after
where it was written, so that the same source compiles once no matter how many
times or from where it is asked for.  A name that is settled on content has to
be short enough to be a file name and cannot collide, so it is a digest; a
human-readable stem is kept alongside it for the sake of anyone looking at the
directory.

Writing one is not atomic by itself, so it is written to a name beside the
target and moved into place: a reader either sees the previous artifact or the
new one, never half of either.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Generic, TypeVar

import tensorplay as tp

from .cache_key import CODE_CACHE_KEY_STRATEGY, SYSTEM_CACHE_KEY_STRATEGY
from .compile_log import dynamo_timed
from .runtime.cache_dir_utils import cache_dir
from .runtime.device_compiler import compiler_module
from .utils import clear_on_fresh_cache

#: Whether a path built here is being written on a platform where a rename
#: onto an existing file fails.  There the write is a copy and a remove, which
#: is not atomic and so is done only where there is no alternative.
_IS_WINDOWS = sys.platform == "win32"


def triton_key() -> str | None:
    """Which device compiler build is loaded, as a cache-key component.

    The compiler's own version string is not enough: it is not changed when the
    compiler's source is, so two builds of the same source under different
    code would share a key.  Its build identity covers that, and there is none
    when the compiler is not here -- which is not a failure, since a run without
    it produces no device code to key.
    """

    module = compiler_module()
    if module is None:
        return None
    for attribute in ("__version__", "TRITON_VERSION"):
        version = getattr(module, attribute, None)
        if isinstance(version, str):
            return version
    return None

T = TypeVar("T")

class CacheBase:
    @staticmethod
    @functools.cache
    def get_system() -> SystemInfo:
        with dynamo_timed("CacheBase.get_system.triton_key"):
            triton_version = triton_key()

        try:
            device_info: SystemDeviceInfo = {"name": None}
            version_info: SystemVersionInfo = {"triton": triton_version}
            device_properties = tp.cuda.get_device_properties(
                tp.cuda.current_device()
            )
            if tp.version.cuda is not None:
                device_info["name"] = device_properties.name
                version_info["cuda"] = tp.version.cuda
            else:
                device_info["name"] = device_properties.gcnArchName
                version_info["hip"] = tp.version.hip
            hash_input: dict[str, Any] = {
                "device": device_info,
                "version": version_info,
            }
            return {
                "device": device_info,
                "version": version_info,
                "hash": SYSTEM_CACHE_KEY_STRATEGY.key_from_json(hash_input),
            }
        except (AssertionError, RuntimeError):
            # If cuda is not installed, none of the above config is relevant.
            return {"hash": SYSTEM_CACHE_KEY_STRATEGY.key_from_json({})}

    @staticmethod
    @clear_on_fresh_cache
    @functools.cache
    def get_local_cache_path() -> Path:
        return Path(os.path.join(cache_dir(), "cache", CacheBase.get_system()["hash"]))

    def __init__(self) -> None:
        self.system = CacheBase.get_system()

    def get_local_cache(self) -> dict[str, Any]:
        local_cache_path = self.get_local_cache_path()
        if not local_cache_path.is_file():
            return {}
        with open(local_cache_path) as local_cache_fp:
            local_cache = json.load(local_cache_fp)
        return local_cache["cache"]

    def update_local_cache(self, local_cache: dict[str, Any]) -> None:
        local_cache_path = self.get_local_cache_path()
        write_atomic(
            str(local_cache_path),
            json.dumps({"system": self.system, "cache": local_cache}, indent=4),
            make_dirs=True,
        )


class LocalCache(CacheBase):
    """A small key-value store that outlives the process that wrote it.

    Used for what is worth keeping between compilations but is not an artifact:
    a measured runtime, a choice that was picked, a decision that took a while.
    Those are answers rather than build products, so they are kept in one file
    keyed by what they are about, rather than in the artifact store keyed by
    what built them.

    The file is read and written whole, so this is for a handful of entries and
    not for a cache of anything large.
    """

    def lookup(self, *keys: str) -> dict[str, Any] | None:
        """The value filed under these keys, or nothing if any step is missing.

        A miss part way down is a miss: a value filed under a key that is no
        longer there was filed under a different question, and answering with it
        would be answering a question that was not asked.
        """

        cache = self.get_local_cache()
        found: Any = cache
        for key in keys:
            if not isinstance(found, dict) or key not in found:
                return None
            found = found[key]
        return found

    def set_value(self, *keys: str, value: Any) -> None:
        """File a value under these keys, making the levels above it as needed."""

        if not keys:
            raise ValueError("a value has to be filed under at least one key")
        cache = self.get_local_cache()
        found = cache
        for key in keys[:-1]:
            nested = found.get(key)
            if not isinstance(nested, dict):
                nested = {}
                found[key] = nested
            found = nested
        found[keys[-1]] = value
        self.update_local_cache(cache)


def code_hash(code: str | bytes, extra: str | bytes = "") -> str:
    if extra:
        return CODE_CACHE_KEY_STRATEGY.key(code, extra)
    return CODE_CACHE_KEY_STRATEGY.key(code)


def _load_python_module(key: str, path: str, set_sys_modules: bool = True):
    """Build a module from a file of generated code.

    The module is named after the key its code hashes to, so that the same code
    always produces the same name, and the file it came from is recorded on it
    so that clearing what was built can find it again.
    """

    with open(path) as f:
        try:
            code = compile(f.read(), path, "exec", dont_inherit=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        mod = ModuleType(f"{__name__}.{key}")
        mod.__file__ = path
        mod.key = key  # type: ignore[attr-defined]
        exec(code, mod.__dict__, mod.__dict__)
        if set_sys_modules:
            sys.modules[mod.__name__] = mod
        return mod


class PyCodeCache:
    """Generated Python modules, and where each came from.

    A generated module is remembered by the code it was built from, so that
    asking again for the same code gives back the module already built from it
    rather than a second one.  Which modules are remembered is tracked here, so
    that clearing what is remembered can also remove what was written to disk.
    """

    #: Every module handed out, in the order they were first asked for.
    modules: list = []

    #: Those asked for without anything attached to them; asking again for one
    #: of these gives the same module back rather than building it twice.
    modules_no_attr: dict = {}

    #: Where each line of a module came from, so that a report about a line can
    #: name what generated it.
    linemaps: dict = {}

    @classmethod
    def write(cls, source_code: str, extra: str = "") -> tuple[str, str]:
        return write(source_code, "py", extra=extra)

    @classmethod
    def load(
        cls,
        source_code: str,
        extra: str = "",
        *,
        set_sys_modules: bool | None = None,
    ):
        """The module for this code, building it only if it is not already here.

        What is remembered is the module, keyed by where it was written, so
        that a second ask for the same code costs a dictionary lookup.
        """

        key, path = write(source_code, "py", extra=extra)
        if path in cls.modules_no_attr:
            mod = cls.modules_no_attr[path]
            if set_sys_modules:
                sys.modules.setdefault(mod.__name__, mod)
            return mod
        mod = _load_python_module(key, path)
        if set_sys_modules:
            sys.modules.setdefault(mod.__name__, mod)
        cls.modules_no_attr[path] = mod
        cls.modules.append(mod)
        return mod

    @classmethod
    def cache_clear(cls, purge: bool = False) -> None:
        """Forget the modules remembered here, and optionally what was written.

        Purging removes what was written to disk, which is for the case where
        the cache itself is being discarded rather than merely not trusted.
        """

        if purge:
            for mod in cls.modules:
                try:
                    if not mod.__file__:
                        raise AssertionError(f"Module {mod} has no __file__ attribute")
                    os.remove(mod.__file__)
                except FileNotFoundError:
                    pass
        cls.modules.clear()
        cls.modules_no_attr.clear()
        cls.linemaps.clear()


def get_hash(content, extra: str = "", hash_type: str = "code") -> str:
    """The name a piece of generated code is remembered under.

    The content is stripped first, so that two pieces of code differing only in
    surrounding whitespace are one piece of code and not two.
    """

    if hash_type == "code":
        return code_hash(content, extra)
    raise NotImplementedError(f"unknown hash type {hash_type}")


def write(
    content,
    extension: str,
    extra: str = "",
    hash_type: str = "code",
    specified_dir: str = "",
    key: str | None = None,
) -> tuple[str, str]:
    """Put generated code where it can be found again, and say where.

    What is written is only written when nothing is there already, so that a
    second process asking for the same code finds the first one's copy rather
    than replacing it.
    """

    if key is None:
        key = get_hash(content.strip(), extra, hash_type)
    basename, _subdir, path = get_path(key, extension, specified_dir)
    if not os.path.exists(path):
        write_atomic(path, content, make_dirs=True)
    return basename, path


def get_path(
    basename: str, extension: str, specified_dir: str = ""
) -> tuple[str, str, str]:
    if specified_dir:
        if os.path.isabs(specified_dir):
            subdir = specified_dir
        else:
            subdir = os.path.join(cache_dir(), specified_dir)
    else:
        subdir = os.path.join(cache_dir(), basename[1:3])
    path = os.path.join(subdir, f"{basename}.{extension}")
    return basename, subdir, path


def write_atomic(
    path_: str,
    content: str | bytes,
    make_dirs: bool = False,
    encode_utf_8: bool = False,
) -> None:
    # Write into temporary file first to avoid conflicts between threads
    # Avoid using a named temporary file, as those have restricted permissions
    if not isinstance(content, (str, bytes)):
        raise AssertionError("Only strings and byte arrays can be saved in the cache")
    path = pathlib.Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    write_mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(write_mode, encoding="utf-8" if encode_utf_8 else None) as f:
        f.write(content)
    try:
        tmp_path.rename(target=path)
    except FileExistsError:
        if not _IS_WINDOWS:
            raise
        # On Windows file exist is expected: https://docs.python.org/3/library/pathlib.html#pathlib.Path.rename
        # Below two lines code is equal to `tmp_path.rename(path)` on non-Windows OS.
        # 1. Copy tmp_file to Target(Dst) file.
        shutil.copy2(src=tmp_path, dst=path)
        # 2. Delete tmp_file.
        os.remove(tmp_path)


class CodeCacheFuture:
    def result(self, timeout: float | None = None) -> Callable[..., Any]:
        raise NotImplementedError


class LambdaFuture(CodeCacheFuture):
    def __init__(
        self, result_fn: Callable[..., Any], future: Future[Any] | None = None
    ) -> None:
        self.result_fn = result_fn
        self.future = future

    def result(self, timeout: float | None = None) -> Callable[..., Any]:
        if timeout is not None and self.future is not None:
            # Wait on the underlying cross-process future with the caller's
            # timeout; raises concurrent.futures.TimeoutError if it does not
            # resolve in time. result_fn will then consume the completed
            # future without blocking further.
            self.future.result(timeout=timeout)
        return self.result_fn()


#: The name a built kernel is entered through.  One name for every kernel, so
#: that a loader needs to know the calling convention but not which kernel it is
#: loading, and so that a kernel built by one thing is entered the same way as
#: one built by another.
NATIVE_ENTRY_NAME = "tp_cpu_native_entry"


class LoadedKernel:
    """A built kernel, loaded and ready to be called.

    Loaded by what it is rather than by what it was called: the content key says
    what it was built from, and the path says where that content was written.
    Both are needed, because the key decides whether this is the kernel wanted
    and the path is the only way to reach it.
    """

    def __init__(self, path: str, key: str, entry_name: str = NATIVE_ENTRY_NAME) -> None:
        import ctypes

        self.__file__ = os.path.abspath(path)
        self.key = key
        self.entry_name = entry_name
        self._ctypes = ctypes
        # Loading the library resolves every symbol in it, so a second load of
        # the same file is a second mapping of the same code rather than a
        # second copy of it, and asking for the same kernel twice costs one
        # mapping rather than two.
        self._library = ctypes.CDLL(self.__file__)
        self._entry = getattr(self._library, entry_name)
        self._entry.restype = None
        self._entry.argtypes = [
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]

    def call(self, args: list) -> Any:
        """Run the kernel over a flat argument list, and hand back its result.

        The list is whatever the call site had: sizes that are only known when
        the kernel runs come first, then the values.  A value is told from a size
        by having somewhere to point -- which is the only difference between
        them that matters here, since a size is a number and a value is memory.
        """

        import tensorplay as tp

        ctypes = self._ctypes
        values = [a for a in args if hasattr(a, "data_ptr")]
        if not values:
            raise ValueError(
                "a kernel was called with no values to read or write"
            )
        handles = (ctypes.c_void_p * len(values))()
        for position, value in enumerate(values):
            handles[position] = ctypes.c_void_p(value.data_ptr())
        shape = tuple(int(s) for s in values[0].shape)
        count = 1
        for extent in shape:
            count *= extent
        result = tp.empty(shape, dtype=values[0].dtype, device=values[0].device)
        self._entry(
            ctypes.c_long(count),
            ctypes.cast(handles, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.c_void_p(result.data_ptr()),
        )
        return result

    def __call__(self, args: list) -> Any:
        return self.call(args)

    def __repr__(self) -> str:
        return f"<LoadedKernel {self.key[:12]} {os.path.basename(self.__file__)}>"


#: What has already been loaded, by path.  A load is a mapping of a file into this
#: process, and doing it twice for one file maps it twice, so what has been
#: loaded is remembered rather than repeated.
_LOADED: dict[str, LoadedKernel] = {}
_LOADED_LOCK = threading.Lock()




def load_by_key_path(
    key: str,
    path: str,
    *,
    entry_name: str = NATIVE_ENTRY_NAME,
    set_sys_modules: bool | None = None,
) -> LoadedKernel:
    """The kernel built from ``key`` and written to ``path``, loaded to be called.

    Returns the already-loaded kernel when this path has been loaded before, so
    that a caller may ask as often as it likes and pay for the mapping once.  The
    key is not consulted to decide whether the file is the right one -- the caller
    that computed it is the one that knows -- but it travels with the result so
    that a caller holding a kernel can say what it was built from.

    ``set_sys_modules`` registers the result under its own name in the module
    table, which is what makes a kernel reachable by name from another process
    that loads this one.  Left unset, registration follows whether this is the
    process's own top level, since that is the case in which nothing else will
    register it.
    """

    resolved = os.path.abspath(path)
    kernel = _LOADED.get(resolved)
    if kernel is None:
        with _LOADED_LOCK:
            kernel = _LOADED.get(resolved)
            if kernel is None:
                if not os.path.exists(resolved):
                    raise FileNotFoundError(
                        f"no artifact was built for {key[:12]} at {resolved}"
                    )
                kernel = LoadedKernel(resolved, key, entry_name)
                _LOADED[resolved] = kernel

    if set_sys_modules is None:
        set_sys_modules = threading.current_thread() is threading.main_thread()
    if set_sys_modules:
        import sys

        sys.modules.setdefault(kernel.entry_name, kernel)  # type: ignore[arg-type]
    return kernel




