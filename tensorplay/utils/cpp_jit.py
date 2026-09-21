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
"""

from __future__ import annotations

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
    return tvm_ffi


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
