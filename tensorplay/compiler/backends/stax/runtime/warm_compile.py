"""Process-pool warm compilation for Stax Triton autotuning (L5-M6).

Benchmarking a candidate table pays one Triton compile per candidate --
MLIR -> PTX -> cubin, a few hundred milliseconds each -- serially, before
the first timed round can start.  The generated kernel source is a plain
string artifact, and Triton persists every binary it builds into a shared
on-disk cache: a helper process handed the same source text plus
placeholder inputs shaped like the caller's examples drives the identical
JIT call the parent would, so the parent's own first launches afterwards
load warm binaries instead of compiling.

This module fans the candidate set out to a persistent helper pool (fresh
interpreters, since a forked child must not touch the parent's CUDA
context), waits for the compiles to land, and returns.  The pool is built
once per process and reused by later autotune rounds.

Warm compilation is strictly an optimization, never a correctness
dependency: every failure mode -- the environment switch, a daemonic
caller that may not spawn children, a helper that crashes or stalls --
degrades silently to the serial route, where candidates compile lazily
during their untimed first launch exactly as before.  Benchmark timing
therefore stays in the parent process and keeps its determinism.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from typing import Any

_DISABLE_ENV = "TP_STAX_PARALLEL_COMPILE"

# A helper pays interpreter start plus the tensorplay/triton imports
# before its first compile; beyond a handful of workers the startup cost
# outweighs the overlap for the candidate-table sizes in use.
_MAX_WORKERS = 4

# Ceiling for the whole warm phase.  Compilation of one candidate is
# seconds at worst; a helper stuck past this budget means the pool is
# unhealthy and the serial route is the better place to wait.
_POOL_TIMEOUT = 120.0

# (source text, source filename, input metadata) per candidate.  Input
# metadata is a tuple of (shape, dtype repr, device repr) triples
# describing placeholders shaped like the caller's example inputs.
WarmTask = tuple[str, str, tuple]

_executor: "concurrent.futures.ProcessPoolExecutor | None" = None


def enabled() -> bool:
    """True when warm compilation may run from the current context.

    False when switched off by environment, or when running inside a
    daemonic child process, whose children may not spawn helpers.
    """

    if os.environ.get(_DISABLE_ENV, "").lower() in ("0", "false"):
        return False
    if multiprocessing.current_process().daemon:
        return False
    return True


def warm_sources(tasks: "list[WarmTask]", *, timeout: float = _POOL_TIMEOUT) -> int:
    """Compile every candidate source in helper processes.

    Returns the number of tasks dispatched; zero when the warm is gated
    off (disabled, daemonic caller, no usable CUDA device, fewer than two
    candidates) or when the pool cannot be created.  Blocks until every
    helper finishes or ``timeout`` elapses; helper failures are ignored --
    an unwarmed candidate simply compiles lazily in the parent later.
    """

    if not enabled() or len(tasks) < 2:
        return 0
    try:
        import tensorplay as tp

        if not tp.cuda.is_available():
            return 0
    except Exception:  # noqa: BLE001 - warm is best-effort
        return 0
    try:
        executor = _get_executor(len(tasks))
        futures = [executor.submit(_warm_one, task) for task in tasks]
    except Exception:  # noqa: BLE001 - warm is best-effort
        return 0
    _done, pending = concurrent.futures.wait(futures, timeout=timeout)
    if pending:
        # A stalled helper would poison every later warm on this pool;
        # drop it and let the next warm start from a fresh one.
        _shutdown()
    return len(tasks)


def _get_executor(count: int) -> "concurrent.futures.ProcessPoolExecutor":
    global _executor
    if _executor is None:
        workers = max(1, min(count, _MAX_WORKERS, os.cpu_count() or 1))
        # Spawn, not fork: the caller holds a CUDA context, and a forked
        # child re-entering the driver through it is undefined.  A fresh
        # interpreter initializes its own device.
        context = multiprocessing.get_context("spawn")
        _executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        )
    return _executor


def _shutdown() -> None:
    global _executor
    if _executor is not None:
        executor, _executor = _executor, None
        executor.shutdown(wait=False, cancel_futures=True)


def _warm_one(task: WarmTask) -> bool:
    """Helper-process entry: exec one candidate source and drive its launch.

    The generated source imports everything it needs (triton, the runtime
    bridge), so an empty namespace suffices.  Placeholder inputs match the
    caller's example shapes, dtypes and device, which reproduces the exact
    JIT specialization -- argument dtypes, pointer alignment, the xnumel
    literal baked into the source -- that the parent's first launch will
    request, making the disk-cache entry a hit rather than a near miss.
    """

    source, source_file, input_meta = task
    # The JIT reads kernel bodies back through linecache, so the source
    # must be registered under its synthetic filename before the exec --
    # exactly what the parent's exec path does.
    import linecache

    linecache.cache[source_file] = (
        len(source),
        None,
        source.splitlines(True),
        source_file,
    )
    namespace: dict[str, Any] = {}
    exec(compile(source, source_file, "exec"), namespace, namespace)
    launch = namespace["kernel_launch"]

    import tensorplay as tp

    inputs = [
        tp.empty(
            tuple(shape),
            dtype=_dtype_by_repr(tp, dtype_repr),
            device=_device_by_repr(tp, device_repr),
        )
        for shape, dtype_repr, device_repr in input_meta
    ]
    launch(inputs)
    return True


def _dtype_by_repr(tp_module: Any, repr_text: str) -> Any:
    for name in dir(tp_module):
        value = getattr(tp_module, name)
        if isinstance(value, tp_module.dtype) and repr(value) == repr_text:
            return value
    return tp_module.float32


def _device_by_repr(tp_module: Any, repr_text: str) -> Any:
    text = str(repr_text).strip("'\"")
    try:
        return tp_module.device(text)
    except Exception:  # noqa: BLE001 - fall back to the plain device string
        return text
