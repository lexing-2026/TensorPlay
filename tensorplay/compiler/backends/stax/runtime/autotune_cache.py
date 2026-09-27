"""Remembering which configuration of a kernel measured best.

Measuring a kernel means compiling it many ways and timing each, and that is
the most expensive thing a build does.  The answer does not change while the
sources do, so it is written down and read back rather than measured again --
which is only safe if everything that could change the answer is part of what
the entry is keyed on.

So three things go into the key: a name for the set of configurations
measured, a name for the source that was measured, and a salt saying which
shape of entry this is.  An entry that does not match all three is not an
answer to this question.

Where the entry lives is a separate question, and there can be more than one
place: the local filesystem, and somewhere shared.  The most local one that
has an answer wins, because the local one is the one this build is writing to
and the one that will be there next time.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import os.path
import re
import threading
import weakref
from typing import Any
from typing_extensions import override

import tensorplay as tp
from tensorplay._guards import CompileContext
from tensorplay.utils._triton import triton_hash_with_backend
from tensorplay.compiler import config as cconfig

from .. import config

from .cache_artifacts import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactRecorder,
)
from .device_compiler import has_triton_package
from .hints import InductorMeta
from .runtime_utils import cache_dir
from .triton_compat import Config, HAS_WARP_SPEC
from ..cache_key import AUTOTUNE_CACHE_KEY_STRATEGY
from ..remote_cache import (
    create_cache,
    JsonDataTy,
    LocalAutotuneCache,
    LocalCacheBackend,
    RemoteCache,
    RemoteCacheJsonSerde,
)


log = logging.getLogger(__name__)


_InductorMetaTy = InductorMeta


def inductor_meta_from_config() -> _InductorMetaTy:
    """The facts about this build that a measured answer depends on.

    Anything that would change which configuration wins goes in here, because
    this is what the entry is keyed on alongside the configurations
    themselves.  The state of the compiler backend is included for the same
    reason: the same configuration compiled by a different backend is a
    different measurement.
    """

    backend_hash = None
    if has_triton_package():
        try:
            backend_hash = triton_hash_with_backend()
        except RuntimeError:
            # This can get the error:
            #   RuntimeError: 0 active drivers ([]). There should only be one.
            pass

    is_hip = None
    if tp.version.hip is not None:
        is_hip = True

    return {
        "autotune_local_cache": config.autotune_local_cache,
        "autotune_remote_cache": config.autotune_remote_cache,
        "backend_hash": backend_hash,
        "bundled_autotune_remote_cache": config.bundled_autotune_remote_cache,
        "coordinate_descent_tuning": config.coordinate_descent_tuning,
        "is_hip": is_hip,
    }


@CacheArtifactFactory.register
class AutotuneCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None:
        autotune_cache = LocalCacheBackend()
        key = os.path.join(cache_dir(), self.key)
        autotune_cache._put(key, self.content)

    @override
    @staticmethod
    def type() -> str:
        return "autotune"

    @override
    @staticmethod
    def encode(content: JsonDataTy) -> bytes:
        if isinstance(content, bytes):
            raise AssertionError("content must not be bytes before encoding")
        serde = RemoteCacheJsonSerde()
        content_bytes = serde.encode(content)
        if not isinstance(content_bytes, bytes):
            raise AssertionError(
                f"Expected bytes after encoding, got {type(content_bytes)}"
            )
        return content_bytes


@dataclasses.dataclass
class AutotuneCache:
    """Looks in every place an answer might be, for one set of configurations.

    More than one of these may answer, and the most local one wins: the local
    filesystem is both the one this build writes and the one most likely to be
    there next time.
    """

    configs_hash: str
    local_cache: tuple[RemoteCache[JsonDataTy], str] | None = None
    remote_cache: tuple[RemoteCache[JsonDataTy], str] | None = None
    artifact_recorder: CacheArtifactRecorder | None = None

    # Create an AutotuneCache. Returns None if none of the caches can be used.
    @staticmethod
    def create(
        inductor_meta: _InductorMetaTy, filename: str, configs_hash: str
    ) -> AutotuneCache | None:
        cache = AutotuneCache(configs_hash)
        key = AutotuneCache._prepare_key(filename)
        local_cache_key = AutotuneCache._make_local_cache_key(
            os.path.dirname(filename), key
        )
        cache.artifact_recorder = CacheArtifactRecorder(
            AutotuneCacheArtifact.type(),
            AutotuneCache._artifact_key_from_local_cache_key(local_cache_key),
        )

        cache._setup_local_cache(inductor_meta, local_cache_key)
        cache._setup_remote_autotune_cache(inductor_meta, key)
        if cache.local_cache or cache.remote_cache:
            return cache
        else:
            return None

    @staticmethod
    def _prepare_key(filename: str) -> str:
        # base of filename is already sha256 hash the source contents
        key = f"{os.path.basename(filename)}:{cconfig.cache_key_tag}"
        return AUTOTUNE_CACHE_KEY_STRATEGY.key(key)

    @staticmethod
    def _make_local_cache_key(dirname: str, cache_key: str) -> str:
        """The local entry's name, including what the source hashed to.

        Including the source in the key is what makes a change to the compiler
        itself invalidate the answer: the entry records a configuration, and a
        change to how configurations are compared or how the best one is
        recorded is not something the entry can be made compatible with.
        """

        from ..codecache import torch_key

        updated_cache_key = AUTOTUNE_CACHE_KEY_STRATEGY.key(cache_key, torch_key())
        return os.path.join(dirname, f"{updated_cache_key}.best_config")

    @staticmethod
    def _artifact_key_from_local_cache_key(local_cache_key: str) -> str:
        return os.path.join(*local_cache_key.split(os.sep)[-2:])

    def _record_artifact(self, data: JsonDataTy) -> None:
        # Older pickled AutotuneCache instances may not have this field.
        if recorder := getattr(self, "artifact_recorder", None):
            recorder.record(data)

    # Read the best config options from the most local cache and return it.
    def _read(self) -> dict[str, JsonDataTy] | None:
        if local_cache := self.local_cache:
            cache, key = local_cache
            AutotuneCacheBundler.sync()
            best_config = cache.get(key)
            if best_config is not None:
                if not isinstance(best_config, dict):
                    raise AssertionError(
                        f"Expected dict for best_config, got {type(best_config)}"
                    )
                # A new model may reuse kernels an earlier one already
                # compiled.  Recording the answer again on a hit is what makes
                # those kernels part of what this model carries, rather than
                # only the ones it compiled for itself.
                AutotuneCacheBundler.put(key, best_config)
                self._record_artifact(best_config)
                if best_config:
                    return best_config

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            if best_config := cache.get(key):
                if isinstance(best_config, dict):
                    self._record_artifact(best_config)
                    return best_config

        return None

    # Read the best config options from the most local cache and figure out
    # which `configs` represents that option.
    def read_best(
        self, inductor_meta: _InductorMetaTy, configs: list[Config]
    ) -> Config | None:
        if best := self._read():
            return _load_cached_autotuning(
                best, self.configs_hash, configs, inductor_meta
            )
        return None

    # Set up local filesystem caching information
    def _setup_local_cache(
        self, inductor_meta: _InductorMetaTy, cache_key: str
    ) -> None:
        if not inductor_meta.get("autotune_local_cache", True):
            return

        local_cache = create_cache(
            "local-autotune",
            local_cache_cls=LocalAutotuneCache.__name__,
        )
        if local_cache is None:
            return
        self.local_cache = (local_cache, cache_key)

    # Set up remote caching information
    def _setup_remote_autotune_cache(
        self, inductor_meta: _InductorMetaTy, cache_key: str
    ) -> None:
        if not _should_use_remote_autotune_cache(inductor_meta):
            return

        if (backend_hash := inductor_meta.get("backend_hash", None)) is None:
            log.debug(
                "backend_hash is not passed on the inductor_meta, unable to use autotune remote cache"
            )
            return
        if not isinstance(backend_hash, str):
            raise AssertionError(
                f"Expected str for backend_hash, got {type(backend_hash)}"
            )

        from ..codecache import torch_key

        salt = "autotune-best-config-v2"
        # re: torch_key - see the note on what goes into the local key
        key = AUTOTUNE_CACHE_KEY_STRATEGY.key(
            torch_key().hex(), backend_hash, self.configs_hash, salt
        )

        remote_cache = create_cache(
            key,
            "RemoteAutotuneCache",
        )
        if not remote_cache:
            return

        # Save the args passed to create_cache
        # in case AutotuneCache needs to be pickled
        self.remote_cache_full_key = key
        self.remote_cache = (remote_cache, cache_key)

    # The AutotuneCache may be serialized/deserialized if we're using
    # AsyncCompile worker processes to run triton compilation.
    # This is because AutotuneCache instances are created on the worker
    # process, but we need to run AutotuneCache.save on the parent process
    # when actually doing autotuning.
    def __getstate__(self) -> dict[str, Any]:
        # The remote cache handles themselves may not be serializable
        # So clear it and reconstruct it on setstate
        remote_cache = getattr(self, "remote_cache", None)
        return {
            **self.__dict__,
            # Save the cache_key portion
            "remote_cache": remote_cache and remote_cache[1],
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Reconstruct the remote cache on the parent class
        self.__dict__.update(state)
        if self.remote_cache is not None:
            if not isinstance(self.remote_cache, str):
                raise AssertionError(
                    f"Expected str for remote_cache after deserialization, got {type(self.remote_cache)}"
                )
            if not hasattr(self, "remote_cache_full_key"):
                raise AssertionError(
                    "Missing remote_cache_full_key attribute after deserialization"
                )
            cache_key = self.remote_cache
            remote_cache = create_cache(
                self.remote_cache_full_key,
                "RemoteAutotuneCache",
            )
            if remote_cache is not None:
                self.remote_cache = (remote_cache, cache_key)
            else:
                log.warning("Warning, failed to recreate remote cache after pickling")
                self.remote_cache = None

    # Save the config in the caches
    def save(
        self,
        config: Config,
        time_taken_ns: int,
        found_by_coordesc: bool = False,
        triton_cache_hash: str | None = None,
    ) -> None:
        """Write down which configuration won, and how long the others took.

        The settings are stored flat alongside the name of the set they came
        from, rather than as a configuration object, because the entry has to
        be readable by a build whose idea of what a configuration is may have
        moved on -- and can then be matched against that build's own list.
        """

        data: dict[str, JsonDataTy] = {
            **config.kwargs,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "configs_hash": self.configs_hash,
            "found_by_coordesc": found_by_coordesc,
            "time_taken_ms": time_taken_ns // 1000000,  # Convert from NS to MS
            "triton_cache_hash": triton_cache_hash,
        }
        # Save extra_options if present on the config. This allows third-party
        # backends to store custom tuned options alongside the standard config.
        if extra_options := getattr(config, "extra_options", None):
            data["extra_options"] = extra_options
        if HAS_WARP_SPEC:
            data.update(
                {
                    "num_consumer_groups": getattr(config, "num_consumer_groups", 0),
                    "num_buffers_warp_spec": getattr(
                        config, "num_buffers_warp_spec", 0
                    ),
                }
            )

        self._record_artifact(data)

        if local_cache := self.local_cache:
            cache, key = local_cache
            cache.put(key, data)
            AutotuneCacheBundler.put(key, data)

            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, key)

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            cache.put(key, data)


class _AutotuneCacheBundlerImpl:
    """A set of local autotune entries kept together under one key.

    One write per kernel would be one file per kernel, and on a machine
    building a model of any size that is a great many small files. So the
    entries are gathered while a build runs and written once at the end.
    """

    _key: str
    _cache: RemoteCache[JsonDataTy]

    # All known entries from local autotune cache writes.
    _entries: dict[str, JsonDataTy]

    def end_compile(self) -> None:
        # TODO: Do we need to compute time_taken_ms and encode that somehow?
        if self._entries:
            self._cache.put(self._key, self._entries)

    def put(self, basename: str, data: JsonDataTy) -> None:
        # Do we need to worry about duplicates? We only have a single local fs
        # entry - so probably not.
        self._entries[basename] = data

    def __init__(self, key: str, cache: RemoteCache[JsonDataTy]) -> None:
        self._key = key
        self._cache = cache
        self._entries = {}

    def sync(self) -> None:
        # We don't currently use this - but we could async load starting at
        # `begin_compile` and wait for the load to be finished here.
        pass

    @classmethod
    def _should_use_bundled_autotune_remote_cache(
        cls, inductor_meta: _InductorMetaTy
    ) -> bool:
        # The bundled autotune cache is only available if you've also got local
        # caching enabled (because we feed the bundled data to the local cache).
        if not inductor_meta.get("autotune_local_cache", True):
            return False

        # Check if we're enabled via config
        if (
            bundled_autotune_remote_cache := inductor_meta.get(
                "bundled_autotune_remote_cache"
            )
        ) is not None:
            return bool(bundled_autotune_remote_cache)

        return False

    def _load_cache(self) -> bool:
        from ..codecache import get_path

        # The single key is defined on construction of the cache.
        entries = self._cache.get(self._key)
        if entries is None or not isinstance(entries, dict):
            # We couldn't load the cache - so mark _entries as non-None so we
            # store local cache values.
            return False

        # Go through the entries we got from the cache and save them locally.
        local_cache = create_cache(
            "local-autotune",
            local_cache_cls=LocalAutotuneCache.__name__,
        )
        for basename, data in entries.items():
            # Reconstruct the final filename (see put())
            root, ext = _splitext_nodot(basename)
            _, _, filename = get_path(root, ext)
            if local_cache is not None:
                local_cache.put(filename, data)

        return True

    @staticmethod
    def _get_backend_hash(inductor_meta: _InductorMetaTy) -> str:
        backend_hash = inductor_meta["backend_hash"]
        if not isinstance(backend_hash, str):
            raise AssertionError(
                f"Expected str for backend_hash, got {type(backend_hash)}"
            )
        return backend_hash


class AutotuneCacheBundler:
    """One gathered set of answers per build, kept against that build.

    Builds run in parallel, so "the entries gathered so far" is a per-build
    thing rather than a per-process one.  It is held against the build rather
    than the thread so that a build which moves between threads still finds
    its own entries, and so that finishing a build releases them.
    """

    _context_bundlers: weakref.WeakKeyDictionary[
        CompileContext, AutotuneCacheBundler
    ] = weakref.WeakKeyDictionary()
    _context_bundlers_lock = threading.Lock()

    def __init__(self) -> None:
        self._bundler: _AutotuneCacheBundlerImpl | None = None

    @classmethod
    def _get_context_bundler(
        cls,
        ctx: CompileContext | None = None,
        *,
        create: bool,
    ) -> AutotuneCacheBundler | None:
        if ctx is None:
            return None

        with cls._context_bundlers_lock:
            bundler = cls._context_bundlers.get(ctx)
            if bundler is None and create:
                bundler = cls()
                cls._context_bundlers[ctx] = bundler
            if bundler is not None and not isinstance(bundler, cls):
                raise AssertionError(
                    f"Expected AutotuneCacheBundler or None, got {type(bundler)}"
                )
            return bundler

    @classmethod
    def has_active_compile(cls, compile_context: CompileContext | None) -> bool:
        context_bundler = cls._get_context_bundler(compile_context, create=False)
        return context_bundler is not None and context_bundler._bundler is not None

    # Call this before we start any autotune computation for an inductor python
    # file. On a cache hit it copies the individual results into the local
    # autotune caches.
    @classmethod
    def begin_compile(
        cls,
        inductor_meta: _InductorMetaTy,
        *,
        code: str | None = None,
        code_hash: str | None = None,
    ) -> None:
        if code is not None:
            if code_hash is not None:
                raise AssertionError("Cannot specify both code and code_hash")
            from ..codecache import code_hash as code_hash_fn

            code_hash = _comment_stripped_hash(code, code_hash_fn)
        if code_hash is None:
            raise AssertionError("Either code or code_hash must be provided")

        if not _AutotuneCacheBundlerImpl._should_use_bundled_autotune_remote_cache(
            inductor_meta
        ):
            return

        context_bundler = cls._get_context_bundler(
            CompileContext.try_get(), create=True
        )
        if context_bundler is None:
            log.debug(
                "Skipping bundled autotune cache because compile_context is not set"
            )
            return
        if context_bundler._bundler is not None:
            raise AssertionError(
                "begin_compile called while a bundler is already active"
            )

        cache = create_cache(
            "bundled-autotune-v1",
            "RemoteBundledAutotuneCache",
        )
        if not cache:
            return

        # We're starting a compilation phase. We have a cache key for the code
        # we're compiling. We'll get the individual autotune bundles later (via
        # self.put()). For now create the AutotuneCacheBundler and try to load
        # from the cache.

        salt = "bundled-autotune-best-configs-v1"
        backend_hash = _AutotuneCacheBundlerImpl._get_backend_hash(inductor_meta)
        key = AUTOTUNE_CACHE_KEY_STRATEGY.key(code_hash, backend_hash, salt)

        bundler = _AutotuneCacheBundlerImpl(key, cache)
        if not bundler._load_cache():
            # We couldn't load from the cache - so save the data so we can store
            # the saved autotunes.
            context_bundler._bundler = bundler

        # If we get a cache hit don't bother saving any of the individual
        # autotune results.

    # Call this after all individual autotune results are finished for a
    # inductor python file. If we gathered any individual results then we bundle
    # those and put it into the cache.
    @classmethod
    def end_compile(cls) -> None:
        if not (
            context_bundler := cls._get_context_bundler(
                CompileContext.try_get(), create=False
            )
        ):
            return
        if bundler := context_bundler._bundler:
            context_bundler._bundler = None
            bundler.end_compile()

    @classmethod
    def sync(cls) -> None:
        if not (
            context_bundler := cls._get_context_bundler(
                CompileContext.try_get(), create=False
            )
        ):
            return
        if bundler := context_bundler._bundler:
            bundler.sync()

    @classmethod
    def put(cls, filename: str, data: JsonDataTy) -> None:
        if not (
            context_bundler := cls._get_context_bundler(
                CompileContext.try_get(), create=False
            )
        ):
            return
        if bundler := context_bundler._bundler:
            # The filename comes in as something like
            # "/tmp/tmp{random}/{aa}/{basename}.py" (where aa is
            # basename[1:3]). Strip it down and make sure that it looks like a path
            # we could reconstruct (because it's possible for the caller to
            # customize the path).
            basename = os.path.basename(filename)

            bundler.put(basename, data)


# Remove the comments from the code (which include things like run ids and file
# paths) and then hash the result, so that two builds of the same source land
# on the same key.
def _comment_stripped_hash(code: str, code_hash_fn) -> str:
    from ..codecache import code_hash as code_hash_fn

    code = re.sub(r"#.*$", "", code, count=0, flags=re.MULTILINE)
    return code_hash_fn(code)


def _should_use_remote_autotune_cache(inductor_meta: _InductorMetaTy) -> bool:
    if (config := inductor_meta.get("autotune_remote_cache")) is not None:
        return bool(config)
    return False


def _reconstruct_triton_config(
    best_config: dict[str, Any],
    extra_options: JsonDataTy | None,
) -> Config:
    num_warps = best_config.pop("num_warps")
    num_stages = best_config.pop("num_stages")
    config_args: dict[str, Any] = {
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    if HAS_WARP_SPEC:
        config_args.update(
            {
                "num_consumer_groups": best_config.pop("num_consumer_groups", 0),
                "num_buffers_warp_spec": best_config.pop("num_buffers_warp_spec", 0),
            }
        )
    triton_config = Config(best_config, **config_args)
    triton_config.extra_options = extra_options
    return triton_config


def _load_cached_autotuning(
    best_config: dict[str, JsonDataTy],
    configs_hash: str,
    configs: list[Config],
    inductor_meta: _InductorMetaTy,
) -> Config | None:
    """The configuration a stored answer names, as one of the ones asked about.

    The entry names settings rather than a configuration, so it has to be
    matched back against the list this build is considering.  Matching exactly
    one is the common case and the good one: it means the answer is one of the
    candidates rather than something a previous build invented.

    Where nothing matches exactly, the settings are used to build a
    configuration directly.  That happens when the answer came from tuning
    that moved away from the starting list, in which case there is nothing in
    the list to match.
    """

    if best_config is None:
        return None
    if best_config.pop("configs_hash", None) != configs_hash:
        return None

    # Remove time taken for comparison
    best_config.pop("time_taken_ms", None)

    best_config.pop("triton_cache_hash", None)

    # Extract extra_options if present. This allows third-party backends
    # to restore custom tuned options from the cache.
    extra_options = best_config.pop("extra_options", None)

    found_by_coordesc = inductor_meta.get(
        "coordinate_descent_tuning"
    ) and best_config.pop("found_by_coordesc", False)

    if not found_by_coordesc:
        matching_configs = [
            cfg
            for cfg in configs
            if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
            and cfg.num_warps == best_config.get("num_warps")
            and cfg.num_stages == best_config.get("num_stages")
        ]
        if len(matching_configs) == 1:
            matched_config = matching_configs[0]
            matched_config.extra_options = extra_options
            return matched_config

    # Reconstruct Config from cached data. This handles both coordesc
    # configs and dynamically added configs that aren't in the original list.
    best_config.pop("found_by_coordesc", None)
    triton_config = _reconstruct_triton_config(best_config, extra_options)
    if found_by_coordesc:
        triton_config.found_by_coordesc = True
    return triton_config


def _splitext_nodot(basename: str) -> tuple[str, str]:
    root, ext = os.path.splitext(basename)
    if ext:
        ext = ext[1:]
    return root, ext
