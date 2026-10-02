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

import base64
import dataclasses
import hashlib
import importlib.util
import logging
import os
import pickle
import sys
import sysconfig
import textwrap
from functools import lru_cache

from .compile_worker.utils import in_toplevel_process

import functools
import hashlib
import json
import os
import pathlib
import pkgutil
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import tensorplay as tp

if TYPE_CHECKING:
    from .remote_cache import JsonDataTy, RemoteCache

from tensorplay.utils._functools import prefetchable_cache as code_key_cache

from .cache_key import CODE_CACHE_KEY_STRATEGY, SYSTEM_CACHE_KEY_STRATEGY
from .compile_log import timed_block
from .cpp_builder import _REPO_ROOT
from tensorplay._subclasses.fake_tensor import (
    extract_tensor_metadata,
    TensorMetadata,
)

from .runtime.cache_artifacts import CacheArtifact, CacheArtifactFactory
from .runtime.cache_dir_utils import cache_dir
from .runtime.device_compiler import compiler_module
from .utils import clear_on_fresh_cache
from . import config
from tensorplay.graph.experimental.symbolic_shapes import has_guarding_hint

#: What a guarded cache holds: an entry of its own kind, opaque here.
_T = TypeVar("_T")
T = _T

log = logging.getLogger(__name__)

#: Whether a path built here is being written on a platform where a rename
#: onto an existing file fails.  There the write is a copy and a remove, which
#: is not atomic and so is done only where there is no alternative.
_IS_WINDOWS = sys.platform == "win32"


@dataclasses.dataclass
class TensorMetadataAndValues:
    """A tensor's metadata beside the values themselves.

    Used where a constant is inlined into a graph: the geometry says what the
    computation is, and the values say what it computes, and a name that
    covers only the first would collide two graphs that differ in the second.
    """

    tensor_metadata: TensorMetadata
    values: list


def _ident(x):
    return x


def extract_tensor_metadata_for_cache_key(t) -> TensorMetadata:
    """A tensor's metadata, with what does not belong in a name removed.

    Where the storage happens to begin is a fact about this run rather than
    about the computation, so it is dropped -- unless the tensor is one the
    compiler produced and therefore placed itself, in which case the offset is
    part of what was built and has to stay.
    """

    meta = extract_tensor_metadata(t)
    if not getattr(t, "_is_tp_static", False):
        meta = dataclasses.replace(meta, storage_offset=0, storage_bytes=None)
    return meta


class GuardedCache(Generic[T]):
    """A cache whose entries each carry the conditions they hold under.

    An entry here is only right for the shapes it was recorded for, so looking
    one up is two questions: which entries are there, and which of them holds
    here.  The first is answered by listing a directory; the second by asking
    the entry's own guard, against the values this call actually has rather
    than against the symbols it came in as -- asking with symbols would let a
    lookup that misses leave new conditions behind, and a miss that changed what
    the next compile believes costs more than the one it saved.
    """

    @classmethod
    def _get_tmp_dir_for_key(cls, _key: str) -> str:
        raise NotImplementedError(
            "the cache this is mixed into has to say where its entries live"
        )

    @classmethod
    def _record_result(
        cls,
        key: str,
        local_hit: bool,
        local_miss: bool,
        remote_hit: bool,
        remote_miss: bool,
    ) -> None:
        raise NotImplementedError(
            "the cache this is mixed into has to say what to record about a "
            "lookup"
        )

    @classmethod
    def iterate_over_candidates(
        cls,
        local: bool,
        remote_cache: "RemoteCache[JsonDataTy] | None",
        key: str,
    ) -> "Generator[tuple[T, bytes, bool], None, None]":
        """Every entry stored under this key, and where each one was found.

        An entry that cannot be read is passed over rather than raised from:
        one unreadable file in a directory is a rebuild, while refusing the
        whole directory is a failure.  An entry whose name begins with a dot is
        the temporary file of a write that has not finished, and is not an
        entry at all.
        """

        if local:
            subdir = cls._get_tmp_dir_for_key(key)
            if os.path.exists(subdir):
                for path in sorted(os.listdir(subdir)):
                    if path.startswith("."):
                        continue
                    try:
                        with open(os.path.join(subdir, path), "rb") as f:
                            content = f.read()
                            yield pickle.loads(content), content, True
                    except Exception:
                        log.warning("cache unable to load an entry", exc_info=True)

        if remote_cache:
            try:
                if (cache_data := remote_cache.get(key)) is not None:
                    if not isinstance(cache_data, dict):
                        raise AssertionError(
                            f"expected a mapping from the remote cache, got "
                            f"{type(cache_data)}"
                        )
                    data = cache_data["data"]
                    if not isinstance(data, (str, bytes)):
                        raise AssertionError(
                            f"expected the cache data as text or bytes, got "
                            f"{type(data)}"
                        )
                    content = base64.b64decode(data)
                    yield pickle.loads(content), content, False
            except Exception:
                log.warning(
                    "%s unable to load an entry", cls.__name__, exc_info=True
                )

    @classmethod
    def find_guarded_entry(
        cls,
        key: str,
        local: bool,
        remote_cache: "RemoteCache[JsonDataTy] | None",
        evaluate_guards: Callable[[str, list], bool],
        hints: list,
    ) -> tuple[T | None, bytes | None, dict]:
        """The first entry under this key whose guard holds here.

        An entry with no guard holds anywhere, so it is taken as soon as it is
        read.  Otherwise each entry in turn is asked, and the first that holds
        is the answer: an entry that holds is as good as another that holds,
        and there is no reason to prefer one over the other.

        What happened is reported alongside, because a miss that names the
        guard that missed is worth having and a bare miss is not.
        """

        graph = None
        pickled_content = None
        result_status = "full_miss"
        sample_guards_expr = None
        in_local = False

        for candidate, content, in_local in cls.iterate_over_candidates(
            local, remote_cache, key
        ):
            if not hasattr(candidate, "guards_expr"):
                raise AssertionError(
                    f"a cache entry of type {type(candidate)} has no guard to "
                    f"check"
                )
            if not candidate.guards_expr:
                graph = candidate
                pickled_content = content
                result_status = "hit"
                break

            hit = bool(evaluate_guards(candidate.guards_expr, hints))
            if hit:
                graph = candidate
                pickled_content = content
                result_status = "hit"
                sample_guards_expr = candidate.guards_expr
                break
            result_status = "guard_miss"
            sample_guards_expr = candidate.guards_expr

        info: dict = {"cache_status_detailed": result_status}
        if sample_guards_expr is not None:
            info["cache_status_guard_expr"] = sample_guards_expr

        # A hit on the far side implies a miss on this one when the local cache
        # is in use, so the two are not counted independently.
        local_hit = graph is not None and in_local
        remote_hit = graph is not None and not in_local
        local_miss = (graph is None or remote_hit) and local
        remote_miss = graph is None and remote_cache is not None
        cls._record_result(
            key,
            local_hit=local_hit,
            local_miss=local_miss,
            remote_hit=remote_hit,
            remote_miss=remote_miss,
        )

        return graph, pickled_content, info

    @classmethod
    def _filter_backed_symints(cls, inputs: Sequence) -> list:
        """The inputs whose values came from somewhere, rather than being free.

        A guard can only be about a value that has one: an input still free at
        trace time has no value yet, so nothing about it can be recorded, and
        recording about it would be recording about a number chosen later.
        """

        return [
            s
            for s in inputs
            if isinstance(s, tp.SymInt) and has_guarding_hint(s)
        ]

    @classmethod
    def _get_shape_env(cls):
        """The environment the shapes are being worked out in, if there is one."""

        tracing = getattr(tp._guards, "TracingContext", None)
        ctx = tracing.try_get() if tracing is not None else None
        if not ctx or not ctx.fake_mode:
            return None
        return ctx.fake_mode.shape_env


@CacheArtifactFactory.register
class TpCacheArtifact(CacheArtifact):
    """One compiled graph, to be put back in the local cache when read."""

    def populate_cache(self) -> None:
        FxGraphCache._write_to_local_cache(self.key, self.content)

    @staticmethod
    def type() -> str:
        return "tp"


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
        with timed_block("CacheBase.get_system.triton_key"):
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


class PersistentCache(CacheBase):
    """A cache of measurements that outlives the process that took them.

    A measurement is expensive enough that taking it twice is a waste, and it
    is a measurement of the machine rather than of the build -- so it is worth
    keeping, and worth keeping next to what it was measured on.
    """

    def lookup(
        self,
        choices: list,
        op: str,
        inputs: str,
        benchmark,
        hint_override: int | None = None,
    ) -> dict:
        """The times already measured for these choices on these inputs, if any.

        A stored time is only usable if it was measured on this machine under
        this precision: a time from a machine that reads the same numbers
        differently is not a slower answer but a different question.  So the
        precision is part of the key rather than assumed.

        When something is missing, whether to measure it is a question about
        what was asked for.  Asked to measure, everything is measured again
        rather than only what was missing -- mixing times from different runs
        would make the set incomparable, and the whole point of comparing them
        is what was being asked.  Asked only to look, a partial answer is
        returned as far as it goes, and the rest is left unsaid rather than
        filled with a number nobody measured.
        """

        precision = tp.get_float32_matmul_precision()
        cache_key = f"{inputs}_{hint_override}" if hint_override is not None else inputs

        timings = {}

        def check_cache(cache: dict) -> bool:
            """Whether `cache` holds a time for every one of the choices."""
            hit = True
            for choice in choices:
                choice_hash = choice.hash_key()
                if choice_hash in cache.get(op, {}).get(cache_key, {}).get(
                    precision, {}
                ):
                    # cache hit
                    timings[choice] = cache[op][cache_key][precision][choice_hash]
                else:
                    # cache miss
                    hit = False
                    break
            return hit

        local_cache = self.get_local_cache() if config.autotune_local_cache else {}
        if (not check_cache(local_cache)) and (benchmark is not None):
            # re-benchmark everything to try to get consistent numbers from the same machine
            timings = benchmark(choices)
            if not all(choice in timings for choice in choices):
                missing = [c for c in choices if c not in timings]
                raise AssertionError(
                    f"Benchmark results missing for choices: {missing}"
                )
            local_cache.setdefault(op, {})
            local_cache[op].setdefault(cache_key, {}).setdefault(precision, {})
            for choice, timing in timings.items():
                local_cache[op][cache_key][precision][choice.hash_key()] = timing

            self.update_local_cache(local_cache)

        return timings


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


def build_code_hash(
    roots: list[str] | None, prefix: str, hasher
) -> None:
    """Fold the source of a package, and everything under it, into a hash.

    Sorted by name so that the same source produces the same hash whatever
    order the modules happen to be listed in, and recursing into packages so
    that a change in a submodule shows up.
    """

    for lib in sorted(pkgutil.iter_modules(roots, prefix), key=lambda x: x.name):
        spec = lib.module_finder.find_spec(lib.name, None)
        if spec is None:
            raise AssertionError(f"Failed to find spec for module {lib.name}")
        module = spec.origin
        if module is None:
            raise AssertionError(f"Module spec for {lib.name} has no origin")
        with open(module, "rb") as f:
            hasher.update(spec.name.encode("utf-8"))
            hasher.update(f.read())
        if lib.ispkg:
            # need to also hash submodules
            build_code_hash(spec.submodule_search_locations, f"{spec.name}.", hasher)


@code_key_cache
def code_key() -> bytes:
    """A hash of the source this build was made from.

    Written into every cache entry, because an answer found by an earlier build
    is only an answer about the source that build was made from.  So a change
    anywhere in the sources has to change this, and a change anywhere else must
    not -- otherwise every entry is invalidated by an edit that could not have
    affected any of them.
    """

    def get_code_hash(root: str) -> bytes:
        # A helper rather than inlining this, so that the one thing a caller
        # should reach for is the function above and not the walk underneath.
        extra_files = (
            "script.ld",
        )
        tp_root = os.path.dirname(__file__)
        extra_files = [os.path.join(tp_root, x) for x in extra_files]
        hasher = hashlib.sha256()
        hasher.update(tp.__version__.encode("utf-8"))
        build_code_hash([root], "", hasher)
        for path in extra_files:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    hasher.update(f.read())
        return hasher.digest()

    with timed_block("code_key"):
        return get_code_hash(_REPO_ROOT)


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
    def load_by_key_path(
        cls,
        key: str,
        path: str,
        linemap: list[tuple[int, str]] | None = None,
        attrs: dict[str, Any] | None = None,
        *,
        set_sys_modules: bool | None = None,
    ):
        """The module written to ``path`` under ``key``, built if not already here.

        Asking by key and path rather than by source is how a module written in
        one process is reached from another: what travels is where it was
        written and what it was written from, and the source itself need not
        travel at all -- it is already on disk, which is the same place for both
        processes.

        ``set_sys_modules`` registers the module under its own name, which is
        what makes it reachable by name from a module that loads this one.  Left
        unset, registration follows whether this is the process's own top
        level, since that is the case in which nothing else will register it.

        ``attrs`` are set on the module once it is built, and a module carrying
        them is one this cache does not keep: a module with something bound into
        it belongs to whoever bound it.
        """

        if linemap is None:
            linemap = []

        in_toplevel = in_toplevel_process()
        set_sys_modules = in_toplevel if set_sys_modules is None else set_sys_modules

        # Only a module with nothing bound into it is kept.
        if attrs is None and path in cls.modules_no_attr:
            mod = cls.modules_no_attr[path]
            if set_sys_modules:
                sys.modules.setdefault(mod.__name__, mod)
            return mod

        mod = _load_python_module(key, path)

        if set_sys_modules:
            cls.linemaps[path] = list(zip(*linemap))

        if attrs is not None:
            for k, v in attrs.items():
                setattr(mod, k, v)

        if in_toplevel:
            if attrs is None:
                cls.modules_no_attr[path] = mod

            cls.modules.append(mod)
        return mod

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


@clear_on_fresh_cache
class CppPythonBindingsCodeCache:
    """Build and load the Python entry point for a generated host kernel."""

    cache: dict[str, Any] = {}
    _loaded_module_names: set[str] = set()
    entry_function = "kernel"

    @staticmethod
    def cache_clear() -> None:
        CppPythonBindingsCodeCache.cache.clear()
        for name in CppPythonBindingsCodeCache._loaded_module_names:
            sys.modules.pop(name, None)
        CppPythonBindingsCodeCache._loaded_module_names.clear()

    @classmethod
    def _load_library_inner(cls, path: str, key: str) -> ModuleType:
        module_name = f"{key}.{cls.entry_function}"
        try:
            return sys.modules[module_name]
        except KeyError:
            pass
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise AssertionError(f"failed to create module loader for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        cls._loaded_module_names.add(module_name)
        spec.loader.exec_module(module)
        return module

    @classmethod
    def load_pybinding_async(
        cls,
        argtypes,
        main_code: str,
        device_type: str = "cpu",
        submit_fn=None,
        **kwargs,
    ):
        del kwargs
        parseargs = ", ".join(
            f"parse_arg<{argtype.replace('const ', '')}>(args, {index})"
            for index, argtype in enumerate(argtypes)
        )
        suffix = textwrap.dedent(
            f"""
            #define PY_SSIZE_T_CLEAN
            #include <Python.h>
            #include <cstdint>
            #include <stdexcept>
            #include <type_traits>

            template <typename T>
            static inline T parse_arg(PyObject* const* args, size_t index) {{
                static_assert(std::is_pointer_v<T>);
                PyObject* method = PyObject_GetAttrString(args[index], "data_ptr");
                if (method == nullptr) {{
                    throw std::runtime_error("expected a tensor pointer argument");
                }}
                PyObject* raw = PyObject_CallNoArgs(method);
                Py_DECREF(method);
                if (raw == nullptr) {{
                    throw std::runtime_error("data_ptr() failed");
                }}
                const unsigned long long address = PyLong_AsUnsignedLongLong(raw);
                Py_DECREF(raw);
                if (address == static_cast<unsigned long long>(-1) &&
                    PyErr_Occurred()) {{
                    throw std::runtime_error("data_ptr() did not return an address");
                }}
                return reinterpret_cast<T>(static_cast<uintptr_t>(address));
            }}

            template <>
            inline int64_t parse_arg<int64_t>(PyObject* const* args, size_t index) {{
                const auto value = PyLong_AsLongLong(args[index]);
                if (value == -1 && PyErr_Occurred()) {{
                    throw std::runtime_error("expected an integer argument");
                }}
                return static_cast<int64_t>(value);
            }}

            template <>
            inline float parse_arg<float>(PyObject* const* args, size_t index) {{
                const double value = PyFloat_AsDouble(args[index]);
                if (value == -1.0 && PyErr_Occurred()) {{
                    throw std::runtime_error("expected a floating-point argument");
                }}
                return static_cast<float>(value);
            }}

            static PyObject* kernel_py(
                PyObject*, PyObject* const* args, Py_ssize_t nargs) {{
                try {{
                    if (nargs != {len(argtypes)}) {{
                        throw std::runtime_error("wrong number of kernel arguments");
                    }}
                    kernel({parseargs});
                    Py_RETURN_NONE;
                }} catch (const std::exception& error) {{
                    PyErr_SetString(PyExc_RuntimeError, error.what());
                    return nullptr;
                }} catch (...) {{
                    PyErr_SetString(PyExc_RuntimeError, "host kernel failed");
                    return nullptr;
                }}
            }}

            static PyMethodDef kernel_methods[] = {{
                {{"kernel", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(kernel_py)),
                  METH_FASTCALL, nullptr}},
                {{nullptr, nullptr, 0, nullptr}}
            }};

            static PyModuleDef kernel_module = {{
                PyModuleDef_HEAD_INIT, "kernel", nullptr, -1, kernel_methods
            }};

            PyMODINIT_FUNC PyInit_kernel(void) {{
                return PyModule_Create(&kernel_module);
            }}
            """
        )
        source_code = main_code + suffix

        from .cpp_builder import CppBuilder, CppOptions, get_cpp_compiler, package_paths
        from .kernel_cache import file_lock
        from .cpu_vec_isa import InvalidVecISA, pick_vec_isa

        paths = package_paths()
        compiler = get_cpp_compiler()
        if paths is None or not compiler:
            raise RuntimeError("host C++ runtime is unavailable")
        include_dir, generated_include_dir, lib_dir = paths
        python_include = sysconfig.get_paths().get("include")
        if not python_include:
            raise RuntimeError("Python headers are unavailable")
        # The generated unit contains vectorized kernels that call into the
        # AVX2/AVX-512 runtime layers, so the unit must be built with the
        # same ISA flags and capability macros as the native kernel path.
        isa = (
            pick_vec_isa()
            if device_type.split(":", maxsplit=1)[0] == "cpu"
            else InvalidVecISA()
        )
        options = CppOptions(
            compiler=compiler,
            include_dirs=[include_dir, generated_include_dir, python_include],
            cflags=[
                "-std=c++20",
                "-O3",
                "-fPIC",
                "-shared",
                "-pthread",
                "-fopenmp",
                *isa.build_arch_flags(),
            ],
            definitions=isa.definitions(),
            library_dirs=[lib_dir],
            libraries=["p10", "gomp"],
            ldflags=["-pthread", f"-Wl,-rpath,{lib_dir}"],
        )
        key, source_path = write(
            source_code,
            "cpp",
            extra=options.command(["<sources>"], "<output>").__repr__(),
        )
        extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        output_path = os.path.join(
            os.path.dirname(source_path), f"{key}{extension_suffix}"
        )

        if key not in cls.cache:
            def build() -> None:
                if os.path.exists(output_path):
                    return
                with file_lock(output_path + ".lock"):
                    if os.path.exists(output_path):
                        return
                    builder = CppBuilder(
                        name=os.path.basename(output_path),
                        sources=[source_path],
                        options=options,
                        output_dir=os.path.dirname(output_path),
                    )
                    builder.build()

            pending = submit_fn(build) if submit_fn is not None else None
            loaded = None

            def load() -> ModuleType:
                nonlocal loaded
                if loaded is None:
                    if pending is not None:
                        pending.result()
                    else:
                        build()
                    loaded = cls._load_library_inner(output_path, key)
                return loaded

            cls.cache[key] = load

        get_result = cls.cache[key]
        result = None

        def future():
            nonlocal result
            if result is None:
                module = get_result()
                if not isinstance(module, ModuleType):
                    raise AssertionError(f"expected a Python module, got {type(module)}")
                result = getattr(module, cls.entry_function)
            return result

        return future

    @classmethod
    def load_pybinding(cls, *args, **kwargs):
        return cls.load_pybinding_async(*args, **kwargs)()


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






@lru_cache(maxsize=1)
def package_key() -> bytes:
    """A key that changes whenever anything that generates code changes.

    Returned as bytes rather than as text because it travels to a worker
    process as a command-line argument, where bytes have to be encoded and
    text does not.

    A compiled kernel is cached against the source it was generated from.  If
    the generator is edited and an old entry is reused, the entry is a kernel
    this compiler would no longer produce, and the mismatch shows up as a
    wrong answer rather than as a stale cache, so the key has to cover the
    generator and not only the code it was handed.

    So the key covers two things: the version this package reports, which
    changes when the source is committed or the build differs, and the
    contents of the compiler's own source tree, which also catches an edit
    that has not been committed.  The tree is walked once per process and the
    answer kept, because walking it for every cache lookup would cost more
    than the lookup.
    """
    hasher = hashlib.sha256()
    hasher.update(str(tp.__version__).encode("utf-8"))
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        hasher.update(str(path.relative_to(root)).encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.digest()


def find_compile_subproc_binary() -> str | None:
    """Which binary a worker process is started as, if not the running one.

    Answered with nothing, so a worker is started as the interpreter that
    started it.  A build that wants a different interpreter names it in the
    environment, and that is read where the environment is built.
    """
    return os.environ.get("TP_COMPILE_SUBPROC_BINARY") or None
