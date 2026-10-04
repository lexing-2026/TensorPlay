"""Generating things that take a long time, in more than one process.

A generated kernel takes long enough that doing several one at a time is
felt.  The work is therefore handed to a pool: a pool of threads when the
work is generating code inside this process anyway, and a pool of processes
when it is not.  Which one is decided per call, because the two answer
different questions -- a thread pool is cheaper and shares what has already
been imported, a process pool survives a generator that takes the
interpreter down with it.
"""

from __future__ import annotations

import functools
import json
import logging
import multiprocessing
import os
import sys
from collections.abc import Callable
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from time import time, time_ns
from typing import Any

from .codecache import (
    CodeCacheFuture,
    LambdaFuture,
    PyCodeCache,
    code_hash,
    code_key,
)

import tensorplay as tp

from . import config
from .compile_worker.subproc_pool import AnyPool, SubprocException, SubprocPool
from .compile_worker.tracked_process_pool import TrackedProcessPoolExecutor
from .compile_worker.utils import _async_compile_initializer
from .runtime.compile_tasks import _worker_compile_triton, pre_fork_setup
# Whether the code generator this project falls back on is present.  The
# module answers that by having its optional import be None when it is not.
from .runtime.triton_compat import triton as _triton

HAS_TRITON = _triton is not None
from .utils import clear_on_fresh_cache, counters, has_triton_package
from tensorplay.graph.experimental.sympy_functions import OrderedSet

log = logging.getLogger(__name__)

_cumulative_compile_time = 0.0
_t0: float | None = None
_triton_kernel_metrics: dict[str, dict[str, Any]] | None = None

#: The pools that exist, so that shutting down means all of them.  Held at
#: module level rather than on the class because a module-level shutdown has
#: no business knowing which class they belong to.
_pool_set: OrderedSet[Any] = OrderedSet()


def _pycodecache_kernel_compile_env() -> dict[str, str | None]:
    env_vars = [
        "TP_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TP_CUTLASS_DIR",
    ]
    return {v: os.environ.get(v) for v in env_vars}


def caching_device_properties():
    for _, device_interface in get_registered_device_interfaces():
        if device_interface.is_available():
            device_interface.Worker.get_device_properties()


def _compile_start() -> None:
    global _t0, _triton_kernel_metrics
    if _t0 is None:
        _t0 = time()
    if _triton_kernel_metrics is None:
        _triton_kernel_metrics = {}


def _compile_end() -> None:
    global _cumulative_compile_time, _t0, _triton_kernel_metrics
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)
    if _triton_kernel_metrics:
        # Log triton kernel info
        sorted_info = dict(sorted(_triton_kernel_metrics.items()))
        tp._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "triton_kernel_info",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(sorted_info),
        )
        _triton_kernel_metrics = None


def _load_triton_kernel_from_source(
    kernel_name: str, source_code: str
) -> Any:
    return getattr(PyCodeCache.load(source_code), kernel_name)


@clear_on_fresh_cache
class CompiledTritonKernels:
    _cache: dict[str, CodeCacheFuture] = {}

    @staticmethod
    def key(kernel_src: str) -> str:
        return code_hash(kernel_src, extra=code_key())

    @staticmethod
    def save(kernel_src: str, future: CodeCacheFuture) -> None:
        CompiledTritonKernels._cache[CompiledTritonKernels.key(kernel_src)] = future

    @staticmethod
    def get(kernel_src: str) -> CodeCacheFuture | None:
        return CompiledTritonKernels._cache.get(CompiledTritonKernels.key(kernel_src))

    @staticmethod
    def cache_clear() -> None:
        CompiledTritonKernels._cache = {}

    @staticmethod
    def remove_future(kernel_src: str) -> None:
        CompiledTritonKernels._cache.pop(CompiledTritonKernels.key(kernel_src), None)


def shutdown_compile_workers() -> None:
    """Shut down all outstanding compile-worker pools."""
    for pool in _pool_set:
        pool.shutdown()
    AsyncCompile._ready_future = None
    after_fork()


def after_fork():
    """Reset pools to initial state without shutting them down"""
    _pool_set.clear()
    AsyncCompile._ready_future = None
    AsyncCompile.process_pool.cache_clear()


def get_compile_threads() -> int:
    """
    Temporary for internal rollout. Assign config.compile_threads lazily and return it.
    TODO: remove after rollout.
    """
    if config.compile_threads is None:
        config.compile_threads = config.decide_compile_threads()
    return config.compile_threads


def _process_pool_allowed() -> bool:
    # Multiprocessing daemons are not allowed to create child processes. This
    # only applies to direct multiprocessing modes: SubprocPool starts its
    # sidecar with subprocess.Popen, so the sidecar does not inherit the
    # multiprocessing daemon flag and can own its own ProcessPoolExecutor.
    return (
        config.worker_start_method == "subprocess"
        or not multiprocessing.current_process().daemon
    )


class AsyncCompile:
    """Where long work is sent, and whether the workers are ready.

    The pools are held on the class rather than on an instance because there
    is one of each: a second pool of the same kind would divide the same
    cores further and answer the same questions twice.
    """

    # Set when a pool has been asked for and is still filling.  Waiting on
    # this is how a caller waits for the pool without waiting for a job: the
    # pool is usable the moment it exists, and ready is a stronger claim
    # about what is already in it.
    _ready_future: Future[Any] | None = None
    _metal_sources: list[tuple[str, str, list[str]]] | None = None

    def __init__(self) -> None:
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        if get_compile_threads() <= 1:
            raise AssertionError(
                f"expected get_compile_threads() > 1, got {get_compile_threads()}"
            )
        return ThreadPoolExecutor(get_compile_threads())

    @staticmethod
    def _get_ready():
        """No-op function to help mark when the subprocess pool is ready."""
        return "ready"

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> AnyPool:
        if get_compile_threads() <= 1:
            raise AssertionError(
                f"expected get_compile_threads() > 1, got {get_compile_threads()}"
            )
        if not _process_pool_allowed():
            raise RuntimeError(
                "async compile process pools are disabled in daemonic "
                "multiprocessing processes. Set "
                "the configuration that says how a worker process is started"
                "(or TP_WORKER_START=subprocess) to use the "
                "SubprocPool path, which is not affected by the daemon restriction."
            )
        AsyncCompile._ready_future = None
        log.info(
            "Creating '%s' pool with %d workers",
            config.worker_start_method,
            get_compile_threads(),
        )

        pool: AnyPool
        if config.worker_start_method == "subprocess":
            # Wrapper around ProcessPoolExecutor forks in a new process we control
            pool = SubprocPool(
                get_compile_threads(), quiesce=config.quiesce_async_compile_pool
            )
        else:
            if config.worker_start_method == "spawn":
                # Avoid creating pools in the spawned subprocs themselves:
                os.environ["TP_WARM_POOL"] = "0"
            pre_fork_setup()
            ctx = multiprocessing.get_context(config.worker_start_method)
            pool = TrackedProcessPoolExecutor(
                get_compile_threads(),
                mp_context=ctx,
                initializer=partial(_async_compile_initializer, os.getpid()),
            )

        # When this pool is created in a multiprocessing subprocess, the normal
        # atexit handler may not run, and we need to register our own handler.
        # exitpriority has to be high, because another one of the finalizers will
        # kill the worker thread that sends the shutdown message to the workers.
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)

        _pool_set.add(pool)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        if get_compile_threads() <= 1 or not _process_pool_allowed():
            return
        _compile_start()
        # Pool is created on first access. Note for a SubprocPool, the sidecar process starts,
        # but its ProcessPoolExecutor does not initialize until a wakeup() call or the first
        # job is submitted.
        cls.process_pool()
        _compile_end()

    @classmethod
    def wait_pool_ready(cls, timeout=120) -> None:
        cls.use_process_pool()
        if cls._ready_future is not None:
            cls._ready_future.result(timeout=timeout)

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        if get_compile_threads() <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def use_process_pool(cls):
        if get_compile_threads() <= 1 or not _process_pool_allowed():
            return False

        # Proton instrumentation backend requires compilation to happen in the main
        # process so it can instrument the Triton IR during JIT compilation.
        # Force synchronous compilation when proton profiling is enabled.
        if config.triton.proton_profiling:
            return False

        # Create a dummy job to check if the pool is ready. Submit it here instead of at
        # pool creation so we don't launch the full pool of worker subprocesses until
        # we're sure they're needed.
        if not cls._ready_future:
            cls._ready_future = cls.process_pool().submit(cls._get_ready)
        return cls._ready_future.done()

    @classmethod
    def wait_process_pool_ready(cls, timeout: float = 120) -> bool:
        """Block (up to ``timeout`` s) until the process pool is ready, returning
        whether it's usable.

        Like use_process_pool() but blocking. Use when a backend's serial
        fallback is far costlier than the warmup wait -- e.g. NVGEMM subprocess
        precompile, where skipping the pool forces ~15x-slower lazy compilation
        at benchmark time. (use_process_pool()'s non-blocking readiness check can
        race pool warmup when little other compilation precedes the decision.)
        On timeout, degrade gracefully (return False -> serial) rather than hang
        on a stuck worker.
        """
        if get_compile_threads() <= 1 or not _process_pool_allowed():
            return False
        if config.triton.proton_profiling:
            return False
        if not cls._ready_future:
            cls._ready_future = cls.process_pool().submit(cls._get_ready)
        try:
            cls._ready_future.result(timeout=timeout)
        except FuturesTimeoutError:
            log.warning(
                "Process pool not ready after %ss; falling back to serial", timeout
            )
            return False
        except (BrokenProcessPool, RuntimeError) as e:
            # A warmup worker died or the pool was closed. The readiness probe
            # failing must degrade to serial (the documented contract), not
            # propagate and abort the caller's algorithm selection.
            log.warning("Process pool unusable (%s); falling back to serial", e)
            return False
        return True

    @classmethod
    def wakeup(cls) -> None:
        """
        If using a SubprocPool, signal the sidecar process to start up its
        ProcessPoolExecutor.
        """
        if not cls.use_process_pool():
            return
        pool = cls.process_pool()
        if isinstance(pool, SubprocPool):
            pool.wakeup()

    def triton(self, kernel_name: str, source_code: str, device_str: str = "cuda"):
        load_kernel = functools.partial(
            _load_triton_kernel_from_source, kernel_name, source_code
        )

        def reload_kernel_in_parent():
            return load_kernel()

        counters["tp"]["async_compile_cache_miss"] += 1
        _compile_start()

        if os.environ.get("TRITON_INTERPRET", "0") == "1":
            return load_kernel()

        is_parallel = self.use_process_pool()
        cached = CompiledTritonKernels.get(source_code)
        if cached is not None:
            counters["tp"]["async_compile_cache_hit"] += 1
            return cached if is_parallel else cached.result()

        if is_parallel:
            from .runtime.compile_tasks import _set_triton_libdevice_path

            _set_triton_libdevice_path()
            env_vars = (
                "TP_CACHE_DIR",
                "TP_TRITON_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TP_TRITON_LIBDEVICE_PATH",
            )
            extra_env = {name: os.environ.get(name) for name in env_vars}
            extra_config = {
                "use_static_triton_launcher": config.use_static_triton_launcher,
            }
            task = self.process_pool().submit(
                _worker_compile_triton,
                load_kernel,
                extra_env,
                extra_config,
            )

            def get_result() -> Any:
                try:
                    kernel, _elapsed_us = task.result()
                except SubprocException as e:
                    raise e.with_name(kernel_name) from e
                CompiledTritonKernels.remove_future(source_code)
                kernel.set_compile_info(None, False)
                kernel.restore_after_unpickle(old_values=None)
                kernel.precompile(
                    warm_cache_only=False,
                    reload_kernel=reload_kernel_in_parent,
                    static_triton_bundle_key=CompiledTritonKernels.key(source_code),
                )
                return kernel

            future = LambdaFuture(get_result, future=task)
            CompiledTritonKernels.save(source_code, future)
            return future

        from .runtime.compile_tasks import (
            _set_triton_libdevice_path,
            _set_triton_ptxas_path,
        )

        _set_triton_ptxas_path()
        _set_triton_libdevice_path()
        kernel = load_kernel()
        kernel.set_compile_info(None, False)
        kernel.precompile(
            warm_cache_only=False,
            static_triton_bundle_key=CompiledTritonKernels.key(source_code),
        )
        return kernel

    def cpp_pybinding(self, argtypes, source_code: str):
        from .codecache import CppPythonBindingsCodeCache

        if get_compile_threads() <= 1:
            return CppPythonBindingsCodeCache.load_pybinding(argtypes, source_code)
        get_result = CppPythonBindingsCodeCache.load_pybinding_async(
            argtypes, source_code, submit_fn=self.submit
        )
        return LambdaFuture(get_result)

    def _wait_futures(self, scope: dict) -> None:
        """Replace each thing that was only started with the thing itself.

        A kernel handed to a worker comes back as something that will produce
        the kernel once it is asked; a caller who wants to call the kernel
        cannot wait on that at the call site, so it is waited for here and the
        result put back under the same name.  A name that is not a started
        kernel is left alone, since it is already what it is.
        """

        kernels = {
            key: value
            for key, value in scope.items()
            if isinstance(value, (Future, CodeCacheFuture))
        }
        for key, result in kernels.items():
            scope[key] = result.result()

    def wait(self, scope: dict) -> None:
        """Wait for every kernel in this scope that was only started.

        Nothing to wait for when only one thread is compiling, since then each
        kernel was compiled as it was reached and there is nothing outstanding.
        """

        if config.compile_threads <= 1:
            return
        self._wait_futures(scope)


def maybe_warm_pool() -> None:
    if (
        os.environ.get("TP_TNT_IN_USE", "0") == "1"
        or os.environ.get("TP_WARM_POOL", "1") != "1"
        # The subprocess pool is only used for the Triton backend
        or not has_triton_package()
        # Skip for fbcode. We have internal reports of usages inside multiprocessing
        # pools that lead a multiplicative number of compile subprocesses.
        or config.is_fbcode()
    ):
        return

    AsyncCompile.warm_pool()
    # TODO: This starts the SubprocPool's internal process pool as early as possible at
    # the expense of creating a bunch of worker processes that might not be needed. We
    # could start them lazily if we're willing to lose a small amount of compile time.
    from .utils import has_triton_package

    AsyncCompile.wakeup()
