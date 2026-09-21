"""Static fast-launch for generated Triton kernels.

After the first full ``JITFunction`` dispatch compiles a kernel for a pinned
configuration, every later launch can call the compiled kernel directly::

    kernel.run(grid0, grid1, grid2, stream, kernel.function,
               kernel.packed_metadata, launch_metadata, enter_hook,
               exit_hook, *bound_args)

skipping the per-call binder, specialization-key build, cache lookup and
used-globals revalidation inside ``JITFunction.run``.  This is the Python
equivalent of the compiled launcher generated per kernel
``binary.run`` call site per autotuned config).  ``native_wrap`` goes one
level lower: it swaps the recorded callable for ``_C``'s pre-bound
``_StaxFastLauncher`` (a vectorcall object holding the CUfunction handle,
block geometry, shared-memory size and live-parameter type string) when
the binary carries nothing beyond a direct ``cuLaunchKernel`` -- no
scratch storage, extra CTAs, PDL or cooperative launch -- so one launch
is a single C entry that packs stack slots and calls the driver.
The launchers only take the fast path when every guard the
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


def _locate_kernel(jitfn: Any, before: int) -> Optional[Any]:
    """The compiled binary the dispatch just used/compiled, or ``None``.

    ``before`` is the :func:`cache_size` snapshot taken immediately before
    the dispatch.  A clean diff (one new entry) or a single-entry cache both
    identify the binary unambiguously.
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
        return kernel
    except Exception:  # noqa: BLE001 - recording is best-effort
        return None


def take_kernel(
    jitfn: Any, before: int
) -> Optional[Tuple[Any, Any, Any]]:
    """Extract ``(run, function, packed_metadata)`` of the kernel the dispatch
    just used/compiled.

    See :func:`_locate_kernel` for the identification rule.  ``None`` makes
    the caller simply stay on the dispatch path.
    """

    kernel = _locate_kernel(jitfn, before)
    if kernel is None:
        return None
    try:
        return (kernel.run, kernel.function, kernel.packed_metadata)
    except Exception:  # noqa: BLE001 - recording is best-effort
        return None


# Triton scalar type string -> C packing char consumed by
# ``_C._StaxFastLauncher``.  Pointers (``*...``) are handled separately;
# half-precision scalars travel as float (the kernel ABI widens them).
_TYPE_CHARS = {
    "i1": "b",
    "i8": "b",
    "i16": "h",
    "i32": "i",
    "i64": "l",
    "u1": "B",
    "u8": "B",
    "u16": "H",
    "u32": "I",
    "u64": "K",
    "fp16": "f",
    "bf16": "f",
    "fp32": "f",
    "f32": "f",
    "fp64": "d",
    "f64": "d",
}

_MAX_NATIVE_ARGS = 128


def _static_launcher_type() -> Optional[type]:
    """The ``_C`` pre-bound launcher type, or ``None`` when unavailable."""

    try:
        from tensorplay import _C as _tp_C

        return getattr(_tp_C, "_StaxFastLauncher", None)
    except Exception:  # noqa: BLE001 - import order: stay on the replay path
        return None


def _native_for_kernel(kernel: Any, full_arg_count: int) -> Optional[tuple]:
    """Build the pre-bound native record triple, or ``None``.

    The native surface covers exactly the binaries a direct
    ``cuLaunchKernel`` can serve: single CTA cluster, no PDL / cooperative
    launch, no scratch storage (fixed trailing scratch slots are satisfied
    with NULLs), and only the scalar/pointer parameter types in the packing
    table.  The returned triple keeps the ``(run, function,
    packed_metadata)`` shape so the call site is unchanged; the launcher
    object holds the owning kernel so its module stays loaded.
    """

    launcher_type = _static_launcher_type()
    if launcher_type is None or not hooks_clear():
        return None
    try:
        metadata = kernel.metadata
        signature = kernel.src.signature
    except Exception:  # noqa: BLE001 - unsupported kernel layout
        return None
    if getattr(metadata, "num_ctas", 1) != 1:
        return None
    if getattr(metadata, "launch_pdl", False):
        return None
    if getattr(metadata, "launch_cooperative_grid", False):
        return None
    global_scratch = getattr(metadata, "global_scratch_size", None)
    profile_scratch = getattr(metadata, "profile_scratch_size", None)
    if (global_scratch or 0) > 0 or (profile_scratch or 0) > 0:
        return None
    num_scratch = (
        (1 if global_scratch is not None else 0)
        + (1 if profile_scratch is not None else 0)
    )
    try:
        types = list(signature.values())
    except Exception:  # noqa: BLE001 - unexpected signature layout
        return None
    if len(types) != full_arg_count:
        return None
    live: list = []
    chars: list = []
    for position, ty in enumerate(types):
        if not isinstance(ty, str):
            return None
        if ty == "constexpr":
            continue
        char = "O" if ty.startswith("*") else _TYPE_CHARS.get(ty)
        if char is None:
            return None
        live.append(position)
        chars.append(char)
    if not chars:
        return None
    if len(chars) + num_scratch > _MAX_NATIVE_ARGS:
        return None
    try:
        num_warps = metadata.num_warps
        shared = (
            kernel.shared if hasattr(kernel, "shared") else metadata.shared
        )
        launcher = launcher_type(
            kernel.function,
            num_warps,
            shared,
            "".join(chars),
            tuple(live),
            num_scratch,
            full_arg_count,
            kernel,
        )
    except Exception:  # noqa: BLE001 - reject, never break the record path
        return None
    return (launcher, kernel.function, None)


def native_wrap(
    jitfn: Any, before: int, full_arg_count: int
) -> Optional[tuple]:
    """Swap the recorded ``run`` callable for the pre-bound native launcher.

    Same identification rule as :func:`take_kernel`; the returned triple is
    record-compatible with it, so callers splice it in front of the replay
    record and fall back to the replay triple when this returns ``None``.
    """

    kernel = _locate_kernel(jitfn, before)
    if kernel is None:
        return None
    return _native_for_kernel(kernel, full_arg_count)
