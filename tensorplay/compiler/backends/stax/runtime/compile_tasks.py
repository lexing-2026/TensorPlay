from __future__ import annotations

import functools
import linecache
import os
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, TYPE_CHECKING

import tensorplay as tp

from ..codecache import package_key
from ..utils import GPU_TYPES, apply_subprocess_env, clear_caches


if TYPE_CHECKING:
    from collections.abc import Callable

    from .triton_heuristics import CachingAutotuner


def _reload_python_module(
    key: str, path: str, set_sys_modules: bool = True
) -> ModuleType:
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


@functools.cache
def _set_triton_ptxas_path() -> None:
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas = Path(__file__).absolute().parents[2] / "bin" / "ptxas"
    if not ptxas.exists():
        return
    if ptxas.is_file() and os.access(ptxas, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    else:
        warnings.warn(f"{ptxas} exists but is not an executable")


def _set_triton_libdevice_path() -> None:
    """
    Use the CUDA toolkit's libdevice instead of Triton's bundled version.
    This ensures Triton's libdevice calls match CUDA eager numerics for bitwise
    precision.  Gated by config.eager_numerics.use_project_libdevice and by
    config.emulate_precision_casts, which also requests eager-like numerics.
    """
    from .. import config

    if not (
        config.eager_numerics.use_project_libdevice or config.emulate_precision_casts
    ):
        return

    _set_triton_libdevice_path_impl()


def _set_triton_libdevice_path_impl() -> None:
    import tensorplay as tp

    if tp.version.cuda is None:
        return

    try:
        from triton import knobs
    except ImportError:
        return

    env_path = os.environ.get("TP_TRITON_LIBDEVICE_PATH")
    if env_path is not None:
        knobs.nvidia.libdevice_path = env_path
        return

    if knobs.nvidia.libdevice_path is not None:
        return

    try:
        from tensorplay.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            warnings.warn(
                "CUDA_HOME not set; using Triton's bundled libdevice which may "
                "cause minor precision differences in pow operations. "
                "To fix: set TP_TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TP_TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
            return
        libdevice = Path(CUDA_HOME) / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.is_file():
            knobs.nvidia.libdevice_path = str(libdevice)
            # Also set env var so subprocess compile workers inherit it
            os.environ["TP_TRITON_LIBDEVICE_PATH"] = str(libdevice)
        else:
            warnings.warn(
                f"CUDA libdevice not found at {libdevice}; using Triton's bundled "
                "libdevice which may cause minor precision differences in pow operations. "
                "To fix: set TP_TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TP_TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "the compiler extension is not available; using Triton's bundled "
            "libdevice which may cause minor precision differences in pow operations. "
            "To fix: set TP_TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
            "e.g., export TP_TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
            stacklevel=3,
        )


_WORKER_CACHE_ENV_VARS = (
    "TP_CACHE_DIR",
    "TP_TRITON_CACHE_DIR",
)
_last_applied_cache_env: dict[str, str | None] | None = None


def _apply_subprocess_env_and_clear_caches(extra_env: dict[str, str | None]) -> None:
    global _last_applied_cache_env

    cache_env = {
        key: extra_env.get(key) for key in _WORKER_CACHE_ENV_VARS if key in extra_env
    }
    if cache_env and cache_env != _last_applied_cache_env:
        clear_caches()
        _last_applied_cache_env = cache_env.copy()
    apply_subprocess_env(extra_env)


def _worker_compile_pycodecache_kernel(
    kernel_name: str,
    source_code: str,
    main_suffix: str,
    extra_env: dict[str, str | None],
    precompile_metadata: dict[str, Any] | None = None,
) -> tuple[str, str, int]:
    """
    Subprocess worker for PyCodeCache-based kernel compilation.

    Writes source to PyCodeCache, loads the module, validates the entry point,
    and optionally triggers real GPU compilation (MLIR -> PTX -> CUBIN) via a
    _precompile entry point. Compiled artifacts are persisted to disk cache so
    the parent process can load them without recompilation.

    Used by CuteDSL, FlyDSL, and NV Universal GEMM backends.
    """
    _apply_subprocess_env_and_clear_caches(extra_env)

    start_ns = time.time_ns()

    from .. import codecache

    key, path = codecache.PyCodeCache.write(source_code)
    mod = codecache.PyCodeCache.load_by_key_path(key, path)

    main_func_name = f"{kernel_name}_{main_suffix}"
    if not hasattr(mod, main_func_name):
        available = [name for name in dir(mod) if callable(getattr(mod, name))]
        raise RuntimeError(
            f"Could not find kernel function '{main_func_name}'. "
            f"Available callables: {available}"
        )

    if precompile_metadata is not None:
        precompile_fn_name = f"{kernel_name}_precompile"
        precompile_fn = getattr(mod, precompile_fn_name, None)
        if precompile_fn is not None:
            precompile_fn(**precompile_metadata)
        else:
            import logging

            logging.getLogger(__name__).warning(
                "Precompile metadata was provided but module has no %s "
                "— the scheduling layer expected this template to support "
                "subprocess precompilation. Kernel will compile lazily on "
                "first call instead.",
                precompile_fn_name,
            )

    elapsed_ns = time.time_ns() - start_ns
    linecache.clearcache()
    return key, path, elapsed_ns // 1000


def _worker_compile_triton(
    load_kernel: Callable[[], CachingAutotuner],
    extra_env: dict[str, str | None],
    extra_config: dict[str, Any],
) -> tuple[CachingAutotuner, int]:
    _set_triton_ptxas_path()
    _apply_subprocess_env_and_clear_caches(extra_env)
    # Keep Triton's in-process knob in sync with the parent environment, including
    # clearing stale worker state when the parent no longer has this variable.
    if "TP_TRITON_LIBDEVICE_PATH" in extra_env:
        try:
            from triton import knobs

            knobs.nvidia.libdevice_path = extra_env["TP_TRITON_LIBDEVICE_PATH"]
        except ImportError:
            pass
    from .. import config
    from ..compile_worker import watchdog
    from . import triton_helpers

    with config.patch(extra_config):
        fail = None
        start_ns = time.time_ns()
        # Generated Triton modules set up the GPU driver at import time,
        # but compile workers only need to warm the compile cache.
        with triton_helpers.skip_gpu_driver_setup():
            kernel = load_kernel()
            watchdog.report_phase(watchdog.Phase.COMPILING)
            kernel.precompile(warm_cache_only=True)
        elapsed_ns = time.time_ns() - start_ns
        kernel.prepare_for_pickle()
        # We can release this memory in the compile subprocesses:
        linecache.clearcache()
        return kernel, elapsed_ns // 1000


def pre_fork_setup() -> None:
    """Warm what a worker would otherwise have to compute for itself.

    A worker that inherits an already-warm parent does not repeat work that
    is the same for every worker: what a device is, and what the code
    generator's own identity is.  Both are asked once here, in the parent,
    and a worker that did not inherit them would otherwise ask again per
    worker -- and the second one is a walk over the generator's source.

    Asked before the workers exist rather than in them, which is the whole
    point: a worker cannot answer a question that needs something only the
    parent has.
    """
    for kind in GPU_TYPES:
        device = getattr(tp, kind, None)
        available = getattr(device, "is_available", None)
        if available is None or not available():
            continue
        count = getattr(device, "device_count", None)
        index = 0 if count is None or count() > 0 else None
        if index is not None:
            device.get_device_properties(index)

    # The code generator's key is a walk over its source tree, which is the
    # same answer for every worker and is not cheap.
    package_key()
