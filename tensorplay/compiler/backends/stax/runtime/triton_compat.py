"""Re-exports of the pieces of the kernel-writing runtime that moved around.

The pieces below live in different places depending on the version, and which
place moves is not something a caller can be asked to know.  So each is looked
for where it might be and re-exported under one name, or the re-export is
``None`` and the caller decides what a missing one means for it.
"""

from __future__ import annotations

import tensorplay._C as _C

try:
    import triton
except ImportError:
    triton = None

if triton is not None:
    import triton.language as tl
    from triton import Config
    from triton.compiler import CompiledKernel
    from triton.runtime.jit import JITFunction, KernelInterface

    def statically_launched_kernel_by_device(kernel, device_type: str = "cuda"):
        """The form of a compiled kernel that launches from its binary alone.

        A kernel compiled to a binary can be loaded onto the device once and
        launched by calling it directly, which is a much smaller thing to carry
        than everything the runtime keeps alongside a kernel it launches itself.
        Whether that is available is a property of the runtime: where the
        loader for it is not built in, there is nothing to hand back, and the
        caller is told so rather than being given something that would fail
        later.
        """

        raise NotImplementedError(
            f"this runtime has no loader for a {device_type} binary, so a "
            f"compiled kernel cannot be launched from one"
        )

    #: Whether a launch can be split across producer and consumer warp groups.
    #: The scheduling knobs for it are read from the runtime's own knob table,
    #: and where that table lives has moved, so it is looked for in both places
    #: and the capability is reported off when neither has it.
    knobs = None
    for _knobs_module in ("triton.runtime.knobs", "triton.knobs"):
        try:
            knobs = __import__(_knobs_module, fromlist=["knobs"])
            break
        except ImportError:
            continue
    HAS_WARP_SPEC = knobs is not None and hasattr(knobs, "warp_specialize")

    #: Holds the language's dispatch key in the thread's key set while entered.
    #: An operator then reaches the Python layer rather than stopping at a
    #: compiled path, which is what lets a mode that intercepts operators see
    #: every one of them -- including the ones a hand-written shortcut would
    #: otherwise answer before anything asked.  Leaving the block puts back
    #: what was there before it.
    enable_python_dispatcher = _C._IncludePythonDispatcher

    def has_triton_block_ptr() -> bool:
        """Whether the runtime can address a tile by a pointer into it.

        A tile can be named either by computing each element's place or by
        being pointed at, and the second is both faster and a different shape of
        kernel. Which one is available is a property of the runtime that has
        moved between versions, so it is asked for rather than assumed, and a
        runtime that cannot say is taken not to have it.
        """

        try:
            from triton.language import block_ptr
        except ImportError:
            return False
        return block_ptr is not None

    try:
        from triton.runtime.cache import triton_key
    except ImportError:
        try:
            from triton.compiler.compiler import triton_key
        except ImportError:

            def triton_key(*args, **kwargs):
                raise RuntimeError("the kernel-writing runtime has no cache key")

    try:
        from triton.runtime.errors import IntelGPUError
    except ImportError:

        class IntelGPUError(Exception):
            pass

    try:
        from triton.language.extra import libdevice

        libdevice = tl.extra.libdevice  # noqa: F811
        math = tl.math
    except ImportError:
        if hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
            libdevice = tl.extra.cuda.libdevice
            math = tl.math
        elif hasattr(tl.extra, "intel") and hasattr(tl.extra.intel, "libdevice"):
            libdevice = tl.extra.intel.libdevice
            math = tl.math
        else:
            libdevice = tl.math
            math = tl

    try:
        from triton.backends.compiler import GPUTarget
    except ImportError:
        GPUTarget = None

    try:
        from triton.runtime.autotuner import OutOfResources
    except ImportError:
        OutOfResources = None

    try:
        from triton.runtime.autotuner import PTXASError
    except ImportError:

        class PTXASError(Exception):
            pass

    try:
        from triton.compiler.compiler import ASTSource
    except ImportError:
        ASTSource = None

    try:
        from triton.language.standard import _log2
    except ImportError:

        def _log2(x):
            raise NotImplementedError("this runtime has no compile-time log2")

    import inspect as _inspect

    #: Whether a compile-time builtin is handed the semantic object under
    #: ``_semantic`` rather than the builder under ``_builder``; the spelling
    #: changed between runtime versions.
    builtins_use_semantic_kwarg = (
        "_semantic" in _inspect.signature(triton.language.core.view).parameters
    )

else:  # pragma: no cover - the kernel-writing runtime is absent
    tl = None
    CompiledKernel = None
    Config = object
    GPUTarget = None
    OutOfResources = None
    PTXASError = None
    ASTSource = None
    JITFunction = None
    KernelInterface = None
    knobs = None
    HAS_WARP_SPEC = False

    #: No runtime, so no dispatch key to hold open: a mode that intercepts
    #: operators has nothing to intercept with.
    enable_python_dispatcher = None

    def has_triton_block_ptr() -> bool:
        """No kernel-writing runtime, so no way to point at a tile."""

        return False
    IntelGPUError = None
    libdevice = None
    math = None
    builtins_use_semantic_kwarg = False

    def _log2(x):
        raise RuntimeError("the kernel-writing runtime is not installed")

    def triton_key(*args, **kwargs):
        raise RuntimeError("the kernel-writing runtime is not installed")

