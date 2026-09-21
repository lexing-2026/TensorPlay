"""Host-device module: the always-present backend.

The CPU executes eagerly and synchronously, so most of the usual device
queries collapse to constants here. Ordering primitives that other backends
provide through streams are not available for host memory; the corresponding
accessors report that honestly instead of silently pretending a stream exists.
"""
from typing import Any

__all__ = [
    "current_device",
    "device_count",
    "get_capabilities",
    "is_available",
    "is_initialized",
    "set_device",
    "synchronize",
]


def is_available() -> bool:
    """The host backend is always compiled in."""
    return True


def is_initialized() -> bool:
    """The host backend needs no lazy initialization."""
    return True


def device_count() -> int:
    """There is exactly one host."""
    return 1


def current_device() -> int:
    """The host device index, which is always zero."""
    return 0


def set_device(device: Any) -> None:
    """The host device is always selected; nothing to change."""
    pass


def synchronize(device: Any = None) -> None:
    """Host execution is synchronous, so there is nothing to wait for."""
    pass


def get_capabilities() -> dict:
    """ISA capability flags for the host, as advertised by the kernel."""
    flags = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("flags") or line.startswith("Features"):
                    flags.update(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    known = [flag for flag in ("avx2", "avx512f", "avx512_bf16", "amx_tile") if flag in flags]
    return {"isa": known}
