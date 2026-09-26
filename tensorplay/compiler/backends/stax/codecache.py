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
    path = Path(path_)
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
