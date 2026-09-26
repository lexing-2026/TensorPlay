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
    from triton.compiler import CompiledKernel
    from triton.runtime.jit import JITFunction, KernelInterface

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
    GPUTarget = None
    OutOfResources = None
    PTXASError = None
    ASTSource = None
    JITFunction = None
    KernelInterface = None
