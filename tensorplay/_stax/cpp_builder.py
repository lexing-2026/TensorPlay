"""C++ build plumbing for runtime-generated kernels.

Owns the three pieces the generated kernels need before source becomes a
loadable shared object:

1. compiler discovery (``g++``/``c++``/``clang++`` search with caching);
2. a version fingerprint that feeds every cache key, so artifacts are never
   reused across toolchain upgrades;
3. :class:`CppOptions` / :class:`CppBuilder` — flag assembly and the actual
   compile invocation.

The builder is deliberately small: it renders one command line, runs it in
a temporary directory, and returns the output path.  Content-addressed
storage and process-level memoization live in :mod:`.codecache`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Sequence

_COMPILER_CANDIDATES = ("g++", "c++", "clang++")

_PROCESS_STATE: dict[str, object] = {}

_PATHS_STATE: dict[str, object] = {}


def package_paths() -> Optional[tuple[str, str, str]]:
    """Return ``(include_dir, generated_include_dir, lib_dir)`` or None.

    Locates the runtime headers a generated kernel compiles against and the
    runtime library it links to.  Development-tree layout only; installed
    wheels fall back to ``None`` and the native path stays disabled.
    """

    if "paths" in _PATHS_STATE:
        return _PATHS_STATE["paths"]
    try:
        import tensorplay

        pkg = os.path.dirname(os.path.abspath(tensorplay.__file__))
        root = os.path.dirname(pkg)
        paths = (
            os.path.join(root, "p10", "include"),
            os.path.join(root, "build", "include"),
            os.path.join(pkg, "lib"),
        )
    except Exception:
        paths = None
    if paths and not os.path.isdir(paths[0]):
        paths = None
    _PATHS_STATE["paths"] = paths
    return paths


def get_cpp_compiler() -> str:
    """Return the discovered system C++ compiler, or ``""``."""

    cached = _PROCESS_STATE.get("compiler")
    if cached is None:
        compiler = ""
        for candidate in _COMPILER_CANDIDATES:
            found = shutil.which(candidate)
            if found is not None:
                compiler = found
                break
        _PROCESS_STATE["compiler"] = compiler
        return compiler
    return str(cached)


def _compiler_version_first_line(compiler: str) -> str:
    try:
        proc = subprocess.run(
            [compiler, "--version"],
            capture_output=True,
            timeout=30,
        )
        text = proc.stdout.decode("utf-8", "replace") or proc.stderr.decode(
            "utf-8", "replace"
        )
        return text.strip().splitlines()[0] if text.strip() else ""
    except Exception:
        return ""


def get_compiler_version_info(compiler: str) -> str:
    """Cached one-line compiler identity used in cache fingerprints."""

    cached = _PROCESS_STATE.get("compiler_version")
    if isinstance(cached, dict):
        if compiler in cached:
            return str(cached[compiler])
    else:
        cached = {}
        _PROCESS_STATE["compiler_version"] = cached
    info = _compiler_version_first_line(compiler)
    cached[compiler] = info
    return info


class CppOptions:
    """Assembled compiler invocation pieces for one build."""

    def __init__(
        self,
        compiler: str = "",
        definitions: Sequence[str] = (),
        include_dirs: Sequence[str] = (),
        cflags: Sequence[str] = (),
        ldflags: Sequence[str] = (),
        library_dirs: Sequence[str] = (),
        libraries: Sequence[str] = (),
    ) -> None:
        self.compiler = compiler or get_cpp_compiler()
        self.definitions = list(definitions)
        self.include_dirs = list(include_dirs)
        self.cflags = list(cflags)
        self.ldflags = list(ldflags)
        self.library_dirs = list(library_dirs)
        self.libraries = list(libraries)

    def flags_hash(self) -> str:
        """Stable string of every flag that affects the output artifact."""

        import hashlib
        import json

        payload = json.dumps(
            [
                os.path.basename(self.compiler),
                self.definitions,
                self.include_dirs,
                self.cflags,
                self.ldflags,
                self.library_dirs,
                self.libraries,
            ],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def command(self, sources: Sequence[str], output_path: str) -> list[str]:
        # Libraries follow the sources: a single-pass linker only resolves
        # symbols for objects it has already seen.
        cmd = [self.compiler]
        cmd.extend(self.definitions)
        cmd.extend(f"-I{d}" for d in self.include_dirs)
        cmd.extend(self.cflags)
        cmd.extend(sources)
        cmd.extend(f"-L{d}" for d in self.library_dirs)
        cmd.extend(f"-l{lib}" for lib in self.libraries)
        cmd.extend(self.ldflags)
        cmd.extend(["-o", output_path])
        return cmd


class CppBuilder:
    """Compile ``sources`` into one artifact under ``output_dir``."""

    def __init__(
        self,
        name: str,
        sources: Sequence[str],
        options: CppOptions,
        output_dir: str,
    ) -> None:
        self.name = name
        self.sources = [os.path.abspath(s) for s in sources]
        self.options = options
        self.output_dir = os.path.abspath(output_dir)

    def get_target_file_path(self) -> str:
        return os.path.join(self.output_dir, self.name)

    def get_command_line(self) -> str:
        """Rendered command for cache-key mixing (flags, not paths)."""

        cmd = self.options.command(["<sources>"], "<output>")
        return " ".join(cmd)

    def build(self, timeout: int = 180) -> str:
        output_path = self.get_target_file_path()
        os.makedirs(self.output_dir, exist_ok=True)
        cmd = self.options.command(self.sources, output_path)
        with tempfile.TemporaryDirectory(prefix="tp_cpp_build_") as workdir:
            log_path = os.path.join(workdir, "build.log")
            with open(log_path, "wb") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=log, timeout=timeout)
            if proc.returncode != 0 or not os.path.exists(output_path):
                with open(log_path, "rb") as fh:
                    detail = fh.read().decode("utf-8", "replace")
                raise RuntimeError(
                    f"cpp build failed (exit {proc.returncode}): {detail[-1200:]}"
                )
        return output_path
