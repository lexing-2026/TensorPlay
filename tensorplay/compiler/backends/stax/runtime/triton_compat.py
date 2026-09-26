"""Re-exports of the pieces of the kernel-writing runtime that moved around.

The pieces below live in different places depending on the version, and which
place moves is not something a caller can be asked to know.  So each is looked
for where it might be and re-exported under one name, or the re-export is
``None`` and the caller decides what a missing one means for it.
"""

from __future__ import annotations

try:
    import triton
except ImportError:
    triton = None

if triton is not None:
    import triton.language as tl
    from triton import Config
    from triton.compiler import CompiledKernel
    from triton.runtime.jit import JITFunction, KernelInterface

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
    IntelGPUError = None

    def triton_key(*args, **kwargs):
        raise RuntimeError("the kernel-writing runtime is not installed")

