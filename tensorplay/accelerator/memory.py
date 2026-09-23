# mypy: allow-untyped-defs
"""Device-agnostic memory queries for the current accelerator.

Every function below targets the active accelerator backend (the GPU
backend in this build) and accepts a device object, a spelling such
as ``"cuda:1"``, an integer index, or ``None`` for the current device.
"""

from typing import Any

from ._utils import _require_accelerator

__all__ = [
    "empty_cache",
    "empty_host_cache",
    "get_memory_info",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]


def _has_accelerator() -> bool:
    try:
        from . import current_accelerator

        return current_accelerator() is not None
    except Exception:
        return False


def _backend():
    _require_accelerator()
    import tensorplay.cuda.memory as _mem

    return _mem


def empty_cache() -> None:
    """Release unoccupied cached device memory held by the allocator."""
    if not _has_accelerator():
        return
    _backend().empty_cache()


def empty_host_cache() -> None:
    """Release unoccupied cached host memory.

    The host side keeps no releasable block cache, so this only probes
    accelerator presence and returns; it exists so generic code can
    spell the same cleanup sequence on any device.
    """
    if not _has_accelerator():
        return
    return None


def get_memory_info(device: Any = None) -> tuple[int, int]:
    """Free and total device memory in bytes for the given device."""
    from ._utils import _native, _resolve_device

    mod = _native()
    probe = (
        getattr(mod, "_accelerator_getMemoryInfo", None)
        if mod is not None
        else None
    )
    if callable(probe):
        free, total = probe(_resolve_device(device, optional=True))
        return (int(free), int(total))
    return _backend().mem_get_info(device)


def memory_stats(device: Any = None) -> dict[str, Any]:
    """Allocator statistics for the given device."""
    return _backend().memory_stats(device)


def memory_allocated(device: Any = None) -> int:
    """Bytes currently occupied by live tensors on the given device."""
    return _backend().memory_allocated(device)


def max_memory_allocated(device: Any = None) -> int:
    """Peak bytes occupied by live tensors on the given device."""
    return _backend().max_memory_allocated(device)


def memory_reserved(device: Any = None) -> int:
    """Bytes currently managed by the allocator on the given device."""
    return _backend().memory_reserved(device)


def max_memory_reserved(device: Any = None) -> int:
    """Peak bytes managed by the allocator on the given device."""
    return _backend().max_memory_reserved(device)


def reset_accumulated_memory_stats(device: Any = None) -> None:
    """Reset historical accumulation counters on the given device."""
    if not _has_accelerator():
        return
    _backend().reset_accumulated_memory_stats(device)


def reset_peak_memory_stats(device: Any = None) -> None:
    """Reset peak counters on the given device."""
    if not _has_accelerator():
        return
    _backend().reset_peak_memory_stats(device)
