"""Asking the kernel-writing runtime what machine it is compiling for.

A compiled kernel is cached under a key that says which runtime built it and
for which machine.  Both of those can be asked of the runtime directly, but only
once it has a machine to ask about, and asking on a machine with no device is
an error rather than an answer.  So the asking is kept behind one function
that a caller reaches through, and the one failure it is worth explaining --
a runtime that cannot read a cache entry it is being handed -- is explained
here rather than left to look like a corrupt file.
"""

from __future__ import annotations

import hashlib
import os
import functools
from typing import Any

#: The message a runtime's loader gives when it cannot read a shared object it
#: was handed.  It usually means the object was built by a different runtime
#: than the one now reading it, which is worth saying plainly: the entry is not
#: damaged, it is not this runtime's.
_FAILED_TO_MAP_SEGMENT_FROM_SHARED_OBJECT = "failed to map segment from shared object"


def _triton_cache_dir_for_error_message() -> str | None:
    from tensorplay.compiler.backends.stax.runtime.cache_dir_utils import cache_dir

    try:
        return cache_dir()
    except Exception:
        return None


def _raise_triton_cache_load_error(original: Exception) -> None:
    cache_dir = _triton_cache_dir_for_error_message()
    if cache_dir is None:
        raise RuntimeError(
            "a compiled kernel could not be read by this runtime; the cache "
            "directory could not be determined either"
        ) from original
    raise RuntimeError(
        f"a compiled kernel in {cache_dir} could not be read by this runtime, "
        "which usually means it was built by a different version; deleting the "
        "cache directory and compiling again will rebuild it"
    ) from original


def triton_backend() -> Any:
    """The backend for the machine this process will run kernels on.

    Both halves come from the runtime: the driver says which machine is
    current, and the backend says how to compile for it.  Asking can fail for
    want of a machine, and a cache entry that cannot be read is a different
    failure with a different fix, so that one is singled out.
    """

    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    try:
        target = driver.active.get_current_target()
        return make_backend(target)
    except (ImportError, OSError) as e:
        if _FAILED_TO_MAP_SEGMENT_FROM_SHARED_OBJECT in str(e):
            _raise_triton_cache_load_error(e)
        raise


def _extern_libs_key(backend: Any) -> str:
    """A cache key fragment for the external libraries a backend links.

    These files change what is generated but are not covered by either half of
    the key -- the sources hash does not see them and the backend's own hash
    covers the assembler and the machine, not the libraries.  Their contents
    are hashed so that a different library produces a different key.
    """

    opts = backend.parse_options({})
    extern_libs = getattr(opts, "extern_libs", None)
    if not extern_libs:
        return ""
    parts = []
    for name, path in sorted(extern_libs):
        if os.path.isfile(path):
            with open(path, "rb") as f:
                parts.append(f"{name}-{hashlib.sha256(f.read()).hexdigest()}")
    return "-".join(parts)


def triton_hash_with_backend() -> str:
    """A cache key fragment for the runtime and the machine together.

    The runtime's own key covers the sources; the backend's covers the
    assembler and the machine.  A kernel is only interchangeable with another
    when both match, so both go in.  The result is written in upper case so
    that it cannot spell a Python keyword.
    """

    from ..compiler.backends.stax.runtime.triton_compat import triton_key

    backend = triton_backend()
    key = f"{triton_key()}-{backend.hash()}-{_extern_libs_key(backend)}"

    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()


def has_triton_package() -> bool:
    """Whether the kernel-writing runtime is installed at all.

    Asked before anything that reaches into it, because a program that has no
    runtime installed is not a program with a broken one -- it is a program
    that never needed it, and the difference decides what to say.
    """

    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


@functools.cache
def has_triton_cpu_backend() -> bool:
    """Whether the runtime can write a kernel for the host rather than a device.

    Asked of the runtime rather than of this package because whether it can is
    a property of the version installed, and a version that cannot is not an
    error -- it is a runtime that writes kernels somewhere else.
    """

    if has_triton_package():
        import triton

        return "cpu" in triton.backends.backends

    return False


@functools.lru_cache(None)
def has_triton_stable_tma_api() -> bool:
    """Whether the runtime can build a device-side descriptor.

    A descriptor is how a kernel is told about a tensor's shape and strides
    without being handed the tensor, and whether the runtime can be asked for
    one is a property of its version and of what the device can do.  Both are
    asked, and the ability is only claimed when the runtime actually offers it.
    """

    if has_triton_package():
        import tensorplay as tp

        if (
            (
                tp.cuda.is_available()
                and tp.cuda.get_device_capability() >= (9, 0)
                and not tp.version.hip
            )
            # Not every build has every device; a device this build does not
            # offer is not one a descriptor could be used on.
            or (getattr(tp, "xpu", None) is not None and tp.xpu.is_available())
            or has_triton_cpu_backend()
        ):
            try:
                from triton.language import make_tensor_descriptor  # noqa: F401

                return True
            except ImportError:
                pass
    return False
