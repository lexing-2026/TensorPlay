"""Overlap generated-kernel builds across a small thread pool.

One generated kernel is one toolchain subprocess, so a build releases the
interpreter lock while its compiler runs; a pool of a few threads is enough
to bring the first-compile latency of a region that emits several kernels
from the sum of its builds down toward the slowest single build.

The worker count resolves in this order: an explicit
:data:`tensorplay.compiler.config.compile_threads` value, the
``TP_COMPILE_THREADS`` environment variable, then the machine's CPU count
capped at :data:`MAX_BUILD_WORKERS`.  ``1`` keeps every build on the
calling thread, which keeps debugger stepping intact.

Results keep job order.  A job's failure travels the same way it would on
a serial run: builders signal failure by returning ``None``, and a job
that raises propagates the first raise to the caller.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

MAX_BUILD_WORKERS = 8
"""Upper bound on concurrent toolchain invocations.

Each generated unit is small, but the bound keeps a many-kernel region from
spawning one compiler process per kernel on a large machine, so memory
stays proportional to a fixed pool rather than to the kernel count.
"""

_T = TypeVar("_T")

_ENV_OVERRIDE = "TP_COMPILE_THREADS"


def resolve_compile_threads() -> int:
    """Return the worker count for kernel builds (at least one)."""

    from tensorplay.compiler import config

    value = getattr(config, "compile_threads", None)
    if value is not None:
        count = value
    else:
        raw = os.environ.get(_ENV_OVERRIDE)
        if raw is None or not raw.strip():
            count = min(MAX_BUILD_WORKERS, os.cpu_count() or 1)
        else:
            try:
                count = int(raw)
            except ValueError:
                count = min(MAX_BUILD_WORKERS, os.cpu_count() or 1)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise TypeError(
            f"compile_threads must be an integer >= 1, got {count!r}"
        )
    return count


def run_builds(jobs: Sequence[Callable[[], _T]]) -> list[_T]:
    """Run zero-argument build jobs, overlapping them when it pays.

    With fewer than two jobs, or when the resolved worker count is one,
    the jobs run inline on the calling thread in order.  Otherwise they go
    to a thread pool sized to the smaller of the worker count and the job
    count; results come back in job order regardless of completion order.
    A pool that cannot be created (for instance, thread exhaustion in an
    embedded interpreter) falls back to the inline path rather than
    failing the compile.
    """

    if len(jobs) < 2:
        return [job() for job in jobs]
    threads = resolve_compile_threads()
    if threads <= 1:
        return [job() for job in jobs]

    workers = min(threads, len(jobs))
    try:
        pool = ThreadPoolExecutor(max_workers=workers)
    except RuntimeError:
        return [job() for job in jobs]

    with pool:
        futures: list[Future[_T]] = [pool.submit(job) for job in jobs]
        try:
            return [future.result() for future in futures]
        except BaseException:
            # Surface the first failure to the caller the way an inline
            # run would; queued-but-unstarted builds are dropped.
            for future in futures:
                future.cancel()
            raise


__all__ = ["MAX_BUILD_WORKERS", "resolve_compile_threads", "run_builds"]
