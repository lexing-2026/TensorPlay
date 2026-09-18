"""Static fast-launch for generated Triton kernels.

After the first full ``JITFunction`` dispatch compiles a kernel for a pinned
configuration, every later launch can call the compiled kernel directly::

    kernel.run(grid0, grid1, grid2, stream, kernel.function,
               kernel.packed_metadata, launch_metadata, enter_hook,
               exit_hook, *bound_args)

skipping the per-call binder, specialization-key build, cache lookup and
used-globals revalidation inside ``JITFunction.run``.  This is the Python
equivalent of the compiled launcher generated per kernel
``binary.run`` call site per autotuned config); the native
``static_triton_launcher`` is the same idea one level lower and is NOT
adoptable here without a ``_C`` rebuild.

The generated launchers only take the fast path when every guard the
recorded binary was specialized under still holds (see
``triton/backends/compiler.py::get_arg_specialization``):

* every tensor argument keeps divisibility-16 pointer alignment,
* the integer scalars equal the recorded values (ints specialize on
  ``== 1`` and ``% 16 == 0``),
* no profiling hooks are installed (otherwise ``launch_metadata`` must be
  built per launch).

Any miss falls back to the normal dispatch, which re-specializes and may
record a new binary.  ``FAST_CALLS``/``SLOW_CALLS`` expose the split for
tests and diagnostics.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

try:  # pragma: no cover - trivial import
    from triton import knobs as _knobs
except Exception:  # pragma: no cover - older/newer triton layouts
    _knobs = None

try:
    from triton.runtime import driver as _driver
except Exception:  # pragma: no cover - triton without a runtime driver
    _driver = None

FAST_CALLS = 0
SLOW_CALLS = 0


def bump(count: int = 1, fast: bool = True) -> None:
    """Launch-path counters used by tests and the generated fast path."""

    global FAST_CALLS, SLOW_CALLS
    if fast:
        FAST_CALLS += count
    else:
        SLOW_CALLS += count


def hooks_clear() -> bool:
    """True when no launch hooks demand per-launch metadata."""

    if _knobs is None:
        return False
    try:
        runtime = _knobs.runtime
        for hook in (runtime.launch_enter_hook, runtime.launch_exit_hook):
            if hook is None:
                continue
            # A hook is either a plain callable or a chain container; only a
            # chain holding at least one callback builds per-launch metadata.
            calls = getattr(hook, "calls", None)
            if calls is None:
                if hook:
                    return False
            elif calls:
                return False
        return True
    except Exception:  # noqa: BLE001 - unknown knobs layout: stay slow
        return False


_CAPTURE_STREAM: Optional[int] = None


def set_capture_stream(handle: Optional[int]) -> Optional[int]:
    """Pin generated-kernel launches to ``handle`` for a capture window.

    Generated launchers resolve the raw launch stream through this module on
    every call (fast and slow paths alike).  During a CUDA capture window the
    native runtime runs on its capture side stream while the driver-level
    query below still reports the legacy default stream, so kernels would be
    launched outside the window and the recorded graph would stay empty.
    Returns the previous override for restoration.
    """
    global _CAPTURE_STREAM
    previous = _CAPTURE_STREAM
    _CAPTURE_STREAM = handle
    return previous


def current_stream() -> int:
    """Raw current-device CUDA stream, exactly what ``JITFunction.run`` uses."""

    if _CAPTURE_STREAM is not None:
        return _CAPTURE_STREAM
    drv = _driver.active
    device = drv.get_current_device()
    return drv.get_current_stream(device)


def cache_size(jitfn: Any) -> int:
    """Number of compiled binaries cached for the current device, or -1."""

    if _driver is None:
        return -1
    try:
        device = _driver.active.get_current_device()
        return len(jitfn.device_caches[device][0])
    except Exception:  # noqa: BLE001 - keep the caller on the slow path
        return -1


def take_kernel(
    jitfn: Any, before: int
) -> Optional[Tuple[Any, Any, Any]]:
    """Extract ``(run, function, packed_metadata)`` of the kernel the dispatch
    just used/compiled.

    ``before`` is the :func:`cache_size` snapshot taken immediately before
    the dispatch.  A clean diff (one new entry) or a single-entry cache both
    identify the binary unambiguously; anything else returns ``None`` and the
    caller simply stays on the dispatch path.
    """

    if _driver is None or before < 0:
        return None
    try:
        device = _driver.active.get_current_device()
        cache = jitfn.device_caches[device][0]
        if len(cache) > before:
            kernel = cache[next(reversed(cache))]
        elif len(cache) == 1 and before <= 1:
            kernel = next(iter(cache.values()))
        else:
            return None
        kernel._init_handles()
        return (kernel.run, kernel.function, kernel.packed_metadata)
    except Exception:  # noqa: BLE001 - recording is best-effort
        return None
