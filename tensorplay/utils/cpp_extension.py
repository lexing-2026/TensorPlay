"""Build and load C++ extensions that link against the TensorPlay runtime.

Unlike :mod:`tensorplay.utils.cpp_jit`, whose kernels see opaque DLPack
views, an extension built here compiles against the ``p10`` headers and
links ``libp10``.  It sees the real object model (``Tensor``, ``Device``,
``Allocator``, the dispatcher), so it can register operator kernels, an
allocator and device hooks at load time.  That is how an out-of-tree device
backend is written (see :mod:`tensorplay.utils.backend_registration`).

An extension is a plain shared library whose static initializers do the
registration (``TENSORPLAY_LIBRARY_IMPL``, ``REGISTER_ALLOCATOR``,
``REGISTER_PRIVATEUSE1_HOOKS``).  :func:`load` compiles it once per distinct
source text and flag set, caches the result, and loads it into the process
with its symbols visible to later extensions.

Example::

    from tensorplay.utils import cpp_extension

    lib = cpp_extension.load("my_backend", ["my_backend.cpp"])
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from typing import Optional

__all__ = ["include_paths", "library_paths", "load", "load_inline"]

_LOADED: dict[str, ctypes.CDLL] = {}
_LOCK = threading.Lock()


def _package_dir() -> str:
    import tensorplay

    return os.path.dirname(os.path.abspath(tensorplay.__file__))


def include_paths() -> list[str]:
    """Header directories an extension compiles against.

    A development checkout serves ``p10/include`` and the generated headers
    under ``build/``; an installed wheel serves the copies under
    ``tensorplay/include``.
    """
    pkg = _package_dir()
    root = os.path.dirname(pkg)
    tree = [
        os.path.join(root, "p10", "include"),
        os.path.join(root, "build", "include"),
        os.path.join(root, "build", "generated"),
    ]
    if os.path.isdir(tree[0]) and os.path.isdir(tree[2]):
        return [path for path in tree if os.path.isdir(path)]
    installed = [
        os.path.join(pkg, "include", "p10"),
        os.path.join(pkg, "include", "generated"),
        os.path.join(pkg, "include", "generated_ops"),
    ]
    found = [path for path in installed if os.path.isdir(path)]
    if not found:
        raise RuntimeError(
            "the TensorPlay runtime headers were not found in a source tree "
            f"next to {pkg} nor under {os.path.join(pkg, 'include')}"
        )
    return found


def library_paths() -> list[str]:
    """Directories holding the runtime libraries (``libp10``)."""
    return [os.path.join(_package_dir(), "lib")]


def _compiler() -> str:
    from tensorplay.compiler.backends.stax.cpp_builder import get_cpp_compiler

    compiler = os.environ.get("CXX") or get_cpp_compiler()
    if not compiler:
        raise RuntimeError("no C++ compiler found (set CXX or install g++/clang++)")
    return compiler


def _default_build_root() -> str:
    base = os.environ.get("TENSORPLAY_EXTENSIONS_DIR")
    if base:
        return base
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache, "tensorplay_extensions")


def _compile_command(compiler: str, source: str, output: str,
                     cflags: Sequence[str], includes: Sequence[str]) -> list[str]:
    return [
        compiler, "-std=c++20", "-O2", "-fPIC", "-pthread",
        *[f"-I{path}" for path in includes],
        *cflags,
        "-c", source, "-o", output,
    ]


def _link_command(compiler: str, objects: Sequence[str], output: str,
                  ldflags: Sequence[str]) -> list[str]:
    libs = library_paths()
    return [
        compiler, "-shared", "-pthread", *objects, "-o", output,
        *[f"-L{path}" for path in libs],
        "-lp10",
        *[f"-Wl,-rpath,{path}" for path in libs],
        *ldflags,
    ]


def _run(command: list[str], name: str, verbose: bool) -> None:
    if verbose:
        print(" ".join(command), file=sys.stderr)
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"building extension '{name}' failed:\n{' '.join(command)}\n"
            f"{proc.stdout}{proc.stderr}"
        )


def _local_headers(sources: Sequence[str]) -> list[str]:
    """Headers next to the sources: an edit to one of them must rebuild."""
    found = set()
    for directory in {os.path.dirname(path) for path in sources}:
        for entry in os.listdir(directory):
            if entry.endswith((".h", ".hpp", ".cuh", ".inc")):
                found.add(os.path.join(directory, entry))
    return sorted(found)


def _jobs() -> int:
    value = os.environ.get("MAX_JOBS")
    if value and value.isdigit() and int(value) > 0:
        return int(value)
    return max(1, min(os.cpu_count() or 1, 16))


def load(
    name: str,
    sources: Sequence[str],
    extra_cflags: Optional[Sequence[str]] = None,
    extra_ldflags: Optional[Sequence[str]] = None,
    extra_include_paths: Optional[Sequence[str]] = None,
    build_directory: Optional[str] = None,
    verbose: bool = False,
) -> ctypes.CDLL:
    """Compile ``sources`` into a shared library and load it.

    Each source compiles to its own cached object, in parallel (up to
    ``$MAX_JOBS`` at a time); an object is rebuilt only when its source, a
    header next to the sources, the flags or the compiler change.  Loading runs its static initializers, so kernels,
    allocators and hooks it registers are live when this returns.  Loading
    the same build twice returns the already-loaded library.

    Args:
        name: Base name of the library file.
        sources: C++ source paths.
        extra_cflags / extra_ldflags: Additional compiler and linker flags.
        extra_include_paths: Header directories searched before the runtime's.
        build_directory: Where the library is written; defaults to
            ``$TENSORPLAY_EXTENSIONS_DIR`` or ``~/.cache/tensorplay_extensions``.
        verbose: Echo the compile command.
    """
    if not name.isidentifier():
        raise ValueError(f"extension name must be an identifier, got {name!r}")
    if not sources:
        raise ValueError("load() needs at least one source file")
    sources = [os.path.abspath(path) for path in sources]
    cflags = list(extra_cflags or ())
    ldflags = list(extra_ldflags or ())
    includes = [os.path.abspath(p) for p in (extra_include_paths or ())] + include_paths()
    compiler = _compiler()

    # A rebuilt runtime may change the layouts the extension was compiled
    # against, so the runtime library's identity is part of every key.
    runtime = []
    for path in library_paths():
        candidate = os.path.join(path, "libp10.so")
        if os.path.exists(candidate):
            stat = os.stat(candidate)
            runtime.append((candidate, stat.st_size, stat.st_mtime_ns))
    common = hashlib.sha256()
    common.update(repr((compiler, cflags, includes, sys.version, runtime)).encode())
    for header in _local_headers(sources):
        with open(header, "rb") as handle:
            common.update(header.encode())
            common.update(handle.read())

    def object_key(path: str) -> str:
        digest = common.copy()
        with open(path, "rb") as handle:
            digest.update(path.encode())
            digest.update(handle.read())
        return digest.hexdigest()[:16]

    object_keys = [object_key(path) for path in sources]
    final = hashlib.sha256()
    final.update(repr((object_keys, ldflags)).encode())
    key = final.hexdigest()[:16]

    directory = build_directory or os.path.join(_default_build_root(), name)
    os.makedirs(directory, exist_ok=True)
    output = os.path.join(directory, f"lib{name}_{key}.so")

    with _LOCK:
        if output in _LOADED:
            return _LOADED[output]
        if not os.path.exists(output):
            objects = []
            pending = []
            for path, okey in zip(sources, object_keys):
                stem = os.path.splitext(os.path.basename(path))[0]
                obj = os.path.join(directory, f"{stem}_{okey}.o")
                objects.append(obj)
                if not os.path.exists(obj):
                    pending.append((path, obj))

            # Each object is written under a temporary name and renamed, so
            # a concurrent loader never sees a half-written file.
            def compile_one(item):
                path, obj = item
                fd, partial = tempfile.mkstemp(prefix=".obj_", suffix=".o", dir=directory)
                os.close(fd)
                try:
                    _run(_compile_command(compiler, path, partial, cflags, includes), name,
                         verbose)
                    os.replace(partial, obj)
                finally:
                    if os.path.exists(partial):
                        os.unlink(partial)

            if pending:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=min(_jobs(), len(pending))) as pool:
                    for future in [pool.submit(compile_one, item) for item in pending]:
                        future.result()
            fd, partial = tempfile.mkstemp(prefix=f".{name}_", suffix=".so", dir=directory)
            os.close(fd)
            try:
                _run(_link_command(compiler, objects, partial, ldflags), name, verbose)
                os.replace(partial, output)
            finally:
                if os.path.exists(partial):
                    os.unlink(partial)
        library = ctypes.CDLL(output, mode=ctypes.RTLD_GLOBAL)
        _LOADED[output] = library
        return library


def load_inline(
    name: str,
    cpp_sources: Sequence[str] | str,
    **kwargs,
) -> ctypes.CDLL:
    """:func:`load` for sources given as strings.

    The text is written into the build directory under a name derived from
    its content, so unchanged text reuses the cached build.
    """
    texts = [cpp_sources] if isinstance(cpp_sources, str) else list(cpp_sources)
    directory = kwargs.get("build_directory") or os.path.join(_default_build_root(), name)
    os.makedirs(directory, exist_ok=True)
    paths = []
    for index, text in enumerate(texts):
        stem = hashlib.sha256(text.encode()).hexdigest()[:16]
        path = os.path.join(directory, f"{name}_{index}_{stem}.cpp")
        if not os.path.exists(path):
            with open(path, "w") as handle:
                handle.write(text)
        paths.append(path)
    kwargs.setdefault("build_directory", directory)
    return load(name, paths, **kwargs)
