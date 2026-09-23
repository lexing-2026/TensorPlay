# mypy: allow-untyped-defs
"""Device-agnostic accelerator helpers.

The accelerator is the non-host device the runtime selects for tensor
work (a GPU in this build, ``None`` on a host-only build). All queries
below delegate to the native layer when it is present and fall back to
the per-backend module otherwise.
"""

from functools import cache
from typing import Any

import tensorplay

from ._utils import (
    _device_module_for_accelerator,
    _native,
    _native_accelerator,
    _require_accelerator,
    _resolve_device,
)
from .graphs import Graph
from .memory import (
    empty_cache,
    empty_host_cache,
    get_memory_info,
    max_memory_allocated,
    max_memory_reserved,
    memory_allocated,
    memory_reserved,
    memory_stats,
    reset_accumulated_memory_stats,
    reset_peak_memory_stats,
)
from . import graphs as graphs
from . import memory as memory
from . import random as random

__all__ = [
    "Graph",
    "current_accelerator",
    "current_device_index",
    "get_device_capability",
    "device_count",
    "device_index",
    "empty_cache",
    "empty_host_cache",
    "get_memory_info",
    "graphs",
    "is_available",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "random",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "set_device_index",
    "set_stream",
    "current_stream",
    "synchronize",
]


def current_accelerator(check_available: bool = False):
    """The accelerator device selected at build time, if any.

    Returns ``None`` on a host-only build. With ``check_available``
    set, a runtime availability probe is also required before the
    device is reported.
    """
    acc = _native_accelerator()
    if acc is None:
        try:
            if tensorplay.cuda.is_available():
                acc = tensorplay.device("cuda")
            else:
                return None
        except Exception:
            return None
    if check_available and not is_available():
        return None
    return acc


def device_count() -> int:
    """Number of devices for the current accelerator, or zero without one."""
    if current_accelerator() is None:
        return 0
    return _device_module_for_accelerator().device_count()


def is_available() -> bool:
    """Whether an accelerator was built and at least one device is visible."""
    if current_accelerator() is None:
        return False
    return _device_module_for_accelerator().is_available()


def current_device_index() -> int:
    """Index of the currently selected accelerator device."""
    mod = _native()
    probe = getattr(mod, "_accelerator_getDeviceIndex", None) if mod is not None else None
    if callable(probe):
        return int(probe())
    _require_accelerator()
    return int(tensorplay.cuda.current_device())


def set_device_index(device) -> None:
    """Select the accelerator device by index; negative indices are no-ops."""
    index = device if isinstance(device, int) else _resolve_device(
        device, optional=False
    )
    if index < 0:
        return
    mod = _native()
    probe = getattr(mod, "_accelerator_setDeviceIndex", None) if mod is not None else None
    if callable(probe):
        probe(index)
        return
    _require_accelerator()
    tensorplay.cuda.set_device(index)


@cache
def get_device_capability(device=None) -> dict[str, Any]:
    """Capability map for an accelerator device.

    The map carries a ``supported_dtypes`` set listing the data types
    that can be allocated on the device.
    """
    index = _resolve_device(device, optional=True)
    mod = _native()
    probe = (
        getattr(mod, "_accelerator_getDeviceCapability", None)
        if mod is not None
        else None
    )
    if callable(probe):
        return dict(probe(index))
    _require_accelerator()
    return {
        "supported_dtypes": {
            tensorplay.uint8,
            tensorplay.int8,
            tensorplay.int16,
            tensorplay.int32,
            tensorplay.int64,
            tensorplay.uint16,
            tensorplay.bool,
            tensorplay.float16,
            tensorplay.bfloat16,
            tensorplay.float32,
            tensorplay.float64,
            tensorplay.complex64,
            tensorplay.complex128,
        }
    }


def current_stream(device=None):
    """The currently selected stream for an accelerator device."""
    index = _resolve_device(device, optional=True)
    _require_accelerator()
    return tensorplay.cuda.current_stream(index)


def set_stream(stream) -> None:
    """Select the current stream for the accelerator device."""
    _require_accelerator()
    tensorplay.cuda.set_stream(stream)


def synchronize(device=None) -> None:
    """Wait for all work on an accelerator device to complete."""
    index = _resolve_device(device, optional=True)
    mod = _native()
    probe = (
        getattr(mod, "_accelerator_synchronizeDevice", None)
        if mod is not None
        else None
    )
    if callable(probe):
        probe(index)
        return
    if current_accelerator() is None:
        return
    tensorplay.cuda.synchronize(index)


class device_index:
    """Temporarily select an accelerator device index."""

    def __init__(self, device) -> None:
        self.idx = None if device is None else _resolve_device(device, optional=True)
        self.prev_idx = -1

    def __enter__(self) -> None:
        if self.idx is not None:
            mod = _native()
            probe = (
                getattr(mod, "_accelerator_exchangeDevice", None)
                if mod is not None
                else None
            )
            if callable(probe):
                self.prev_idx = int(probe(self.idx))
            else:
                self.prev_idx = int(current_device_index())
                tensorplay.cuda.set_device(self.idx)

    def __exit__(self, *exc_info: object) -> None:
        if self.idx is not None:
            mod = _native()
            probe = (
                getattr(mod, "_accelerator_maybeExchangeDevice", None)
                if mod is not None
                else None
            )
            if callable(probe):
                probe(int(self.prev_idx))
            else:
                tensorplay.cuda.set_device(int(self.prev_idx))
