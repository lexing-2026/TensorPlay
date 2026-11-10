"""Huawei Ascend NPU device backend for TensorPlay.

The backend fills TensorPlay's out-of-tree device slot: memory comes from
the Ascend runtime through a caching allocator, kernels run as Ascend
operator calls, and every operator without an NPU kernel runs on the host
through the generic fallback.  The device is named ``"npu"``::

    import tensorplay as tp
    import tensorplay_npu  # or let the tensorplay.backends entry point load it

    x = tp.randn(64, 64, device="npu")
    y = (x @ x).relu().sum()

The native part is compiled against the installed CANN toolkit the first
time it is loaded (``$ASCEND_TOOLKIT_HOME``, default
``/usr/local/Ascend/cann-8.5.0``) and cached under
``~/.cache/tensorplay_extensions``.
"""

from __future__ import annotations

import ctypes
import glob
import os
import threading

__all__ = ["ascend_home", "is_loaded", "load"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")
_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None


def ascend_home() -> str:
    """Root of the CANN toolkit the backend builds against."""
    return os.environ.get("ASCEND_TOOLKIT_HOME") or "/usr/local/Ascend/cann-8.5.0"


def is_loaded() -> bool:
    return _LIB is not None


def _library() -> ctypes.CDLL:
    if _LIB is None:
        raise RuntimeError("the NPU backend is not loaded; call tensorplay_npu.load()")
    return _LIB


def _build(verbose: bool) -> ctypes.CDLL:
    from tensorplay.utils import cpp_extension

    home = ascend_home()
    if not os.path.isfile(os.path.join(home, "include", "acl", "acl.h")):
        raise RuntimeError(f"Ascend toolkit not found under {home} (set ASCEND_TOOLKIT_HOME)")
    libdir = os.path.join(home, "lib64")
    return cpp_extension.load(
        "tensorplay_npu",
        sorted(glob.glob(os.path.join(_CSRC, "*.cpp"))),
        extra_cflags=["-O3", "-DNDEBUG", "-Wno-deprecated-declarations"],
        # aclCreateTensor and friends are the operator-side tensor API of
        # libnnopbase, not the runtime API of libascendcl.
        extra_ldflags=[f"-L{libdir}", "-lascendcl", "-lnnopbase", "-lopapi",
                       f"-Wl,-rpath,{libdir}"],
        extra_include_paths=[_CSRC, os.path.join(home, "include")],
        verbose=verbose,
    )


def load(verbose: bool = False) -> None:
    """Build (once) and load the native backend, then name the device."""
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return
        import tensorplay as tp

        lib = _build(verbose)
        lib.tp_npu_init.restype = ctypes.c_int
        if lib.tp_npu_init() != 0:
            raise RuntimeError("registering the NPU host fallbacks failed")
        lib.tp_npu_device_count.restype = ctypes.c_int
        lib.tp_npu_memory_stats.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
        lib.tp_npu_reset_peak_memory_stats.argtypes = [ctypes.c_int]
        _LIB = lib

        tp.utils.rename_privateuse1_backend("npu")
        tp.utils.generate_methods_for_privateuse1_backend()
        from . import runtime

        tp._register_device_module("npu", runtime)


def _autoload() -> None:
    """``tensorplay.backends`` entry point: load when the toolkit is present."""
    if os.path.isfile(os.path.join(ascend_home(), "include", "acl", "acl.h")):
        load()
