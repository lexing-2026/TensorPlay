"""Just-in-time loading of C++ / CUDA kernels.

Thin frontend over the ``tvm-ffi`` compile engine.  Sources are compiled
into a shared library whose exported functions are registered through a
stable C FFI; arguments crossing the boundary use the DLPack tensor
structure, so :class:`tensorplay.Tensor` passes through zero-copy (the
kernel sees a tensor view: data pointer, shape, strides, dtype, device)
and no adapter layer is involved.

Two usage modes:

- JIT: :func:`load_inline` compiles string sources, caches the build and
  returns a callable module in one step; :func:`load` does the same from
  real source files.
- AOT: :func:`build_inline` / :func:`build` produce a ``.so`` artifact
  suitable for shipping; :func:`load_module` loads it back at runtime.

The string entry points generate the export registration for the named
``functions``; the file entry points compile the sources as-is, so the
files must carry their own export macros
(``TVM_FFI_DLL_EXPORT_TYPED_FUNC``).  For kernels linked straight into
the process, :func:`system_lib` exposes the statically registered
functions without dynamic loading.

The compiled functions are opaque foreign kernels: they see DLPack views,
not the TensorPlay object model, so they cannot call back into the
dispatcher or register operators.  Wrap a call in
:func:`tensorplay.library.custom_op` (with an explicit schema, fake
kernel and autograd formula) to make the kernel a first-class operator.

The frontend also installs a process-wide *environment allocator* on
first engine use (only when no host allocator is already present), so
kernels can allocate TensorPlay tensors directly through
``TVMFFIEnvTensorAlloc`` / ``Tensor::FromEnvAlloc``.  The allocated
tensor returns to Python as the engine's tensor wrapper; convert it
with :func:`tensorplay.from_dlpack`.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "is_available",
    "load_inline",
    "build_inline",
    "load_module",
    "load",
    "build",
    "system_lib",
]

_INSTALL_HINT = (
    "the cpp JIT engine requires the 'tvm-ffi' package; "
    "install it (e.g. `pip install tvm-ffi`) and retry"
)


def is_available() -> bool:
    """Whether the compile engine is importable in this environment."""
    try:
        import tvm_ffi  # noqa: F401
    except ImportError:
        return False
    return True


def _engine() -> Any:
    try:
        import tvm_ffi
        import tvm_ffi.cpp
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    _ensure_env_allocator(tvm_ffi)
    return tvm_ffi


# ---------------------------------------------------------------------------
# Environment allocator
#
# FFI kernels allocate host tensors through TVMFFIEnvTensorAlloc (see
# Tensor::FromEnvAlloc in kernel code).  The trampoline below fulfills
# those requests with ordinary TensorPlay allocations, crossing the
# boundary as DLPack:
#
#   tp.empty -> __dlpack__ capsule -> engine Tensor object
#     -> versioned DLPack wrapper handed to the caller
#
# Ownership is a chain of DLPack deleters: when the engine-side tensor
# dies it releases the versioned wrapper, which releases the engine
# object, which releases the capsule's managed tensor, whose deleter
# drops the Python-side reference and recycles the pooled wrapper.  The
# allocator is installed once, on first engine use, and only when no
# host allocator is present (an already-installed allocator wins).

_ALLOCATOR_LOCK = threading.Lock()
_ALLOCATOR_STATE = 0  # 0 = not tried, 1 = installed, 2 = foreign or absent
_ALLOCATOR_CB: Any = None  # trampoline keep-alive
_ENGINE_LIB: Any = None
_DTYPE_TABLE: dict | None = None


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice),
                ("ndim", ctypes.c_int32), ("dtype", _DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


_SET_ERROR_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
_ALLOCATOR_FUNC = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(_DLTensor),
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, _SET_ERROR_FUNC)

# capsule accessors on a private pythonapi handle: restype must be a
# real pointer width, and the prototype must not leak onto the shared
# ctypes.pythonapi function objects
_PYAPI = ctypes.PyDLL(None)
_PYAPI.PyCapsule_GetPointer.restype = ctypes.c_void_p
_PYAPI.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PYAPI.PyCapsule_SetName.restype = ctypes.c_int
_PYAPI.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]

_KDL_INT, _KDL_UINT, _KDL_FLOAT, _KDL_BFLOAT, _KDL_COMPLEX, _KDL_BOOL = 0, 1, 2, 4, 5, 6
_DEVICE_NAMES = {1: "cpu", 2: "cuda"}


def _dl_dtype_table() -> dict:
    global _DTYPE_TABLE
    if _DTYPE_TABLE is None:
        import tensorplay as tp

        _DTYPE_TABLE = {
            (_KDL_FLOAT, 16): tp.float16, (_KDL_FLOAT, 32): tp.float32,
            (_KDL_FLOAT, 64): tp.float64, (_KDL_BFLOAT, 16): tp.bfloat16,
            (_KDL_INT, 8): tp.int8, (_KDL_INT, 16): tp.int16,
            (_KDL_INT, 32): tp.int32, (_KDL_INT, 64): tp.int64,
            (_KDL_UINT, 8): tp.uint8, (_KDL_UINT, 16): tp.uint16,
            (_KDL_UINT, 32): tp.uint32, (_KDL_UINT, 64): tp.uint64,
            (_KDL_COMPLEX, 64): tp.complex64,
            (_KDL_COMPLEX, 128): tp.complex128,
            (_KDL_BOOL, 8): tp.bool,
        }
    return _DTYPE_TABLE


def _tp_env_alloc(prototype, out, error_ctx, set_error):
    """Fulfill a kernel-side allocation request with a TensorPlay tensor."""
    try:
        import tensorplay as tp

        request = prototype.contents
        if request.dtype.lanes != 1:
            raise ValueError("vector dtypes are not supported")
        dtype = _dl_dtype_table().get((request.dtype.code, request.dtype.bits))
        if dtype is None:
            raise ValueError(
                f"unsupported dtype code={request.dtype.code} "
                f"bits={request.dtype.bits}")
        device = _DEVICE_NAMES.get(request.device.device_type)
        if device is None:
            raise ValueError(
                f"unsupported device type {request.device.device_type}")
        shape = [request.shape[i] for i in range(request.ndim)]
        tensor = tp.empty(
            shape, dtype=dtype,
            device=tp.device(device, request.device.device_id))

        capsule = tensor.__dlpack__()
        managed = _PYAPI.PyCapsule_GetPointer(capsule, b"dltensor")
        obj = ctypes.c_void_p()
        if _ENGINE_LIB.TVMFFITensorFromDLPack(
                managed, 0, 0, ctypes.byref(obj)) != 0:
            raise RuntimeError("wrapping the allocated tensor failed")
        # the engine object owns the unversioned wrapper from here on;
        # mark the capsule consumed so its destructor stays inert
        _PYAPI.PyCapsule_SetName(capsule, b"used_dltensor")

        versioned = ctypes.c_void_p()
        if _ENGINE_LIB.TVMFFITensorToDLPackVersioned(
                obj, ctypes.byref(versioned)) != 0:
            raise RuntimeError("producing the versioned wrapper failed")
        # the versioned wrapper holds its own reference to the object
        _ENGINE_LIB.TVMFFIObjectDecRef(obj)
        out[0] = versioned
        return 0
    except Exception as exc:  # noqa: BLE001 - reported through SetError
        kind = type(exc).__name__ if isinstance(exc, ValueError) else "RuntimeError"
        set_error(error_ctx, kind.encode(), str(exc).encode())
        return -1


def _ensure_env_allocator(engine: Any) -> None:
    """Install the TensorPlay env allocator, once, if none is present."""
    global _ALLOCATOR_STATE, _ALLOCATOR_CB, _ENGINE_LIB
    if _ALLOCATOR_STATE:
        return
    with _ALLOCATOR_LOCK:
        if _ALLOCATOR_STATE:
            return
        try:
            lib = ctypes.CDLL(engine.LIB._name)
        except AttributeError:
            lib = engine.LIB
        lib.TVMFFIEnvGetDLPackManagedTensorAllocator.restype = ctypes.c_void_p
        lib.TVMFFIEnvGetDLPackManagedTensorAllocator.argtypes = []
        if lib.TVMFFIEnvGetDLPackManagedTensorAllocator():
            _ALLOCATOR_STATE = 2
            return
        lib.TVMFFITensorFromDLPack.restype = ctypes.c_int
        lib.TVMFFITensorFromDLPack.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p)]
        lib.TVMFFITensorToDLPackVersioned.restype = ctypes.c_int
        lib.TVMFFITensorToDLPackVersioned.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.TVMFFIObjectDecRef.restype = ctypes.c_int
        lib.TVMFFIObjectDecRef.argtypes = [ctypes.c_void_p]
        lib.TVMFFIEnvSetDLPackManagedTensorAllocator.restype = ctypes.c_int
        lib.TVMFFIEnvSetDLPackManagedTensorAllocator.argtypes = [
            _ALLOCATOR_FUNC, ctypes.c_int, ctypes.POINTER(_ALLOCATOR_FUNC)]

        _ENGINE_LIB = lib
        _ALLOCATOR_CB = _ALLOCATOR_FUNC(_tp_env_alloc)
        if lib.TVMFFIEnvSetDLPackManagedTensorAllocator(
                _ALLOCATOR_CB, 1, None) != 0:
            raise RuntimeError("installing the env tensor allocator failed")
        _ALLOCATOR_STATE = 1


def load_inline(
    name: str,
    *,
    cpp_sources: str | Sequence[str] | None = None,
    cuda_sources: str | Sequence[str] | None = None,
    functions: str | Sequence[str] | Mapping[str, str] | None = None,
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    extra_ldflags: Sequence[str] | None = None,
    extra_include_paths: Sequence[str] | None = None,
    build_directory: str | None = None,
    **kwargs: Any,
) -> Any:
    """Compile string sources and return the loaded module.

    ``functions`` names the exported kernels (a sequence of names, or a
    mapping from name to docstring).  Each becomes an attribute of the
    returned module and accepts DLPack-compatible tensors directly, so
    TensorPlay tensors cross the boundary without copies.  Repeated calls
    with the same ``name`` reuse the cached build.  Remaining keyword
    arguments are forwarded to the engine.
    """
    engine = _engine()
    return engine.cpp.load_inline(
        name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        functions=functions,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        **kwargs,
    )


def build_inline(
    name: str,
    *,
    cpp_sources: str | Sequence[str] | None = None,
    cuda_sources: str | Sequence[str] | None = None,
    functions: str | Sequence[str] | Mapping[str, str] | None = None,
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    extra_ldflags: Sequence[str] | None = None,
    extra_include_paths: Sequence[str] | None = None,
    build_directory: str | None = None,
    **kwargs: Any,
) -> str:
    """Compile string sources ahead of time and return the ``.so`` path.

    Takes the same arguments as :func:`load_inline` but stops after the
    build: the artifact can be shipped with an application and brought
    back through :func:`load_module`.
    """
    engine = _engine()
    return engine.cpp.build_inline(
        name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        functions=functions,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        **kwargs,
    )


def load_module(path: str, **kwargs: Any) -> Any:
    """Load a previously built ``.so`` artifact by path."""
    engine = _engine()
    return engine.load_module(path, **kwargs)


def load(
    name: str,
    *,
    sources: str | Sequence[str] | None = None,
    cpp_files: str | Sequence[str] | None = None,
    cuda_files: str | Sequence[str] | None = None,
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    extra_ldflags: Sequence[str] | None = None,
    extra_include_paths: Sequence[str] | None = None,
    build_directory: str | None = None,
    **kwargs: Any,
) -> Any:
    """Compile real source files and return the loaded module.

    The file-based counterpart of :func:`load_inline`.  Sources are
    compiled as-is: unlike the inline entry points nothing is generated,
    so the files must contain the export macros for every kernel that
    should be reachable on the returned module.
    """
    engine = _engine()
    return engine.cpp.load(
        name,
        sources=sources,
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        **kwargs,
    )


def build(
    name: str,
    *,
    sources: str | Sequence[str] | None = None,
    cpp_files: str | Sequence[str] | None = None,
    cuda_files: str | Sequence[str] | None = None,
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    extra_ldflags: Sequence[str] | None = None,
    extra_include_paths: Sequence[str] | None = None,
    build_directory: str | None = None,
    **kwargs: Any,
) -> str:
    """Compile real source files ahead of time; return the ``.so`` path.

    Takes the same arguments as :func:`load` but stops after the build.
    Combine with :func:`load_module` on the target machine.
    """
    engine = _engine()
    return engine.cpp.build(
        name,
        sources=sources,
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        **kwargs,
    )


def system_lib(symbol_prefix: str = "") -> Any:
    """Return the process-wide library module for linked-in kernels.

    Kernels registered at static-link time (compiled into the executable
    or an imported library rather than loaded from an artifact) are
    reached through this module without any dynamic loading.
    ``symbol_prefix`` selects a registration namespace.
    """
    engine = _engine()
    return engine.system_lib(symbol_prefix)
