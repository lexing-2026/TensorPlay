"""Kernel code-generation cache (L5-M1).

The lowering backend (stax/Triton/AVX) supplies the compiled artifact
generated ``source`` plus its compile callable; this module owns key
canonicalization, lookup, atomic publication and process-level memoization,
so backends stay free of cache plumbing.

Layout: ``$TP_CACHE_DIR`` or ``<cwd>/.tp_cache/kernels/<backend>/ab/<key>.<ext>``
local cache). Publication is temp-file + rename, so concurrent processes
never observe partial artifacts.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from typing import Any, Callable, Dict, Iterator, Optional, Tuple


@contextlib.contextmanager
def file_lock(path: str, *, shared: bool = False) -> Iterator[None]:
    """Advisory exclusive/shared lock guarding out-of-process builds.

    External toolchain invocations write their output in place, so two
    processes racing on the same cache key could interleave writes into one
    corrupt artifact.  The lock file itself carries no data.

    Some cache subfolders may be unwritable for this user (entries written
    by another uid earlier).  The lock then moves to a mirror path under the
    temp directory: mutual exclusion is preserved per path, and the build
    proceeds against the cache folder.
    """

    def acquire(target: str) -> int:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        return os.open(target, os.O_CREAT | os.O_RDWR, 0o644)

    try:
        fd = acquire(path)
    except OSError:
        mirror = os.path.join(
            tempfile.gettempdir(),
            "tp-locks",
            hashlib.sha256(path.encode()).hexdigest()[:32] + ".lock",
        )
        fd = acquire(mirror)
    try:
        import fcntl

        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(fd, mode)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class CodeCache:
    """Content-addressed store mapping (source, entry, options) -> artifact."""

    def __init__(self, backend: str, root: Optional[str] = None) -> None:
        self.backend = backend
        env_root = os.environ.get("TP_CACHE_DIR")
        base = root or env_root or os.path.join(os.getcwd(), ".tp_cache")
        self.root = os.path.join(base, "kernels", backend)
        # Cache entries may have been written by another uid (a root-run
        # process earlier): key-prefix subfolders then reject every write
        # for this user and silently disable compilation.  Move to a
        # user-writable cache root once per instance instead.
        if not self._writable(base) or self._poisoned_prefixes():
            fallback = os.path.join(tempfile.gettempdir(), f"tp-cache-{os.getuid()}")
            self.root = os.path.join(fallback, "kernels", backend)

    @staticmethod
    def _writable(base: str) -> bool:
        probe = os.path.join(base, ".writable-probe")
        try:
            os.makedirs(probe, exist_ok=True)
            return os.access(probe, os.W_OK)
        except OSError:
            return False

    def _poisoned_prefixes(self) -> bool:
        try:
            with os.scandir(self.root) as entries:
                return any(
                    entry.is_dir() and not os.access(entry.path, os.W_OK)
                    for entry in entries
                )
        except OSError:
            return False

    # -- keying -------------------------------------------------------------

    @staticmethod
    def _canonical(options: Optional[Dict[str, Any]]) -> str:
        return json.dumps(options or {}, sort_keys=True, default=repr)

    def cache_key(
        self,
        source: str,
        entry: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        h = hashlib.sha256()
        h.update(self.backend.encode())
        h.update(b"\x00")
        h.update(entry.encode())
        h.update(b"\x00")
        h.update(self._canonical(options).encode())
        h.update(b"\x00")
        h.update(source.encode())
        return h.hexdigest()

    @staticmethod
    def _caches_disabled() -> bool:
        from tensorplay.compiler import config

        value = config.force_disable_caches
        if not isinstance(value, bool):
            raise TypeError("config.force_disable_caches must be a bool")
        return value

    def _path_for(self, key: str, ext: str, *, create: bool) -> str:
        folder = os.path.join(self.root, key[:2])
        if create:
            os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{key}.{ext}")

    def path_for(self, key: str, ext: str = "bin") -> str:
        return self._path_for(key, ext, create=not self._caches_disabled())

    # -- storage ------------------------------------------------------------

    def load(self, key: str, ext: str = "bin") -> Optional[bytes]:
        if self._caches_disabled():
            return None
        memo = self._memo().get((key, ext))
        if memo is not None:
            return memo
        path = self.path_for(key, ext)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            return None
        self._memo()[(key, ext)] = data
        return data

    def store(self, key: str, payload: bytes, ext: str = "bin") -> str:
        if self._caches_disabled():
            return self._path_for(key, ext, create=False)
        path = self._path_for(key, ext, create=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            os.replace(tmp, path)  # atomic publication
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        self._memo()[(key, ext)] = payload
        return path

    def compile_or_load(
        self,
        compile_fn: Callable[[str], bytes],
        source: str,
        *,
        entry: str = "",
        options: Optional[Dict[str, Any]] = None,
        ext: str = "bin",
    ) -> Tuple[bytes, str]:
        """Return ``(artifact, path)``, compiling through ``compile_fn`` on miss."""

        key = self.cache_key(source, entry, options)
        if self._caches_disabled():
            return compile_fn(source), self._path_for(key, ext, create=False)
        cached = self.load(key, ext)
        if cached is not None:
            return cached, self.path_for(key, ext)
        artifact = compile_fn(source)
        self.store(key, artifact, ext)
        return artifact, self.path_for(key, ext)

    # -- process-level memo ---------------------------------------------------

    def _memo(self) -> Dict[Tuple[str, str], bytes]:
        memo = getattr(self, "_memo_dict", None)
        if memo is None:
            memo = {}
            self._memo_dict = memo
        return memo


_default_caches: Dict[str, CodeCache] = {}


def default_cache(backend: str) -> CodeCache:
    """Process-wide cache instance per backend."""

    cache = _default_caches.get(backend)
    if cache is None:
        cache = CodeCache(backend)
        _default_caches[backend] = cache
    return cache
