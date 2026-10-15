# mypy: allow-untyped-defs
r"""MPS (Metal Performance Shaders) namespace and runtime helpers.

This package adds support for MPS tensor types on Apple silicon.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports MPS.
"""

import threading
from typing import Any, Optional

import tensorplay
import tensorplay._C
from tensorplay._C import Device
from tensorplay._C import _mps as _lmps

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()


def _lazy_init() -> None:
    global _initialized
    if _initialized:
        return
    with _initialization_lock:
        if _initialized:
            return
        _initialized = True


def is_available() -> bool:
    r"""Return a bool indicating if MPS is currently available."""
    return _lmps.is_available()


def device_count() -> int:
    r"""Return the number of MPS devices available."""
    return _lmps.device_count()


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    _lazy_init()
    return _lmps.current_device()


def synchronize(device: Any = None) -> None:
    r"""Wait for all kernels in all streams on the MPS device to complete.

    Args:
        device (tensorplay.Device or int, optional): device for which to synchronize.
            MPS exposes a single device, so this argument is accepted for API
            compatibility and defaults to device 0.
    """
    _lazy_init()
    if isinstance(device, int):
        idx = device
    elif isinstance(device, Device):
        idx = device.index
    elif device is None:
        idx = -1
    else:
        raise TypeError(f"Invalid device: {device!r}")
    return _lmps.synchronize(idx)


def memory_allocated(device: Any = None) -> int:
    r"""Return the current memory allocated by tensors on the MPS device.

    Args:
        device (tensorplay.Device or int, optional): accepted for API compatibility.
    """
    _lazy_init()
    return _lmps.memory_allocated(_get_device_index(device, optional=True))


def memory_reserved(device: Any = None) -> int:
    r"""Return the current reserved memory on the MPS device.

    Args:
        device (tensorplay.Device or int, optional): accepted for API compatibility.
    """
    _lazy_init()
    return _lmps.memory_reserved(_get_device_index(device, optional=True))


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used by other GPU applications.
    """
    _lazy_init()
    return _lmps.empty_cache()


def _get_device_index(device: Any, optional: bool = False) -> int:
    """Resolve a device argument to an integer index."""
    if isinstance(device, int):
        return device
    if isinstance(device, Device):
        return device.index if device.index >= 0 else 0
    if device is None:
        return 0
    if optional:
        return 0
    raise ValueError(f"Expected an MPS device, but got: {device!r}")


__all__ = [
    "is_available",
    "device_count",
    "current_device",
    "synchronize",
    "memory_allocated",
    "memory_reserved",
    "empty_cache",
]