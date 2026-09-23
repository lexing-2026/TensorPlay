# mypy: allow-untyped-defs
"""Host-device module: the always-present backend.

The host executes eagerly and synchronously, so most device queries
collapse to constants. Ordering primitives that other backends provide
through streams are emulated with lightweight placeholders so generic
code can spell the same stream/event sequence on any device.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from tensorplay._streambase import EventBase, StreamBase

__all__ = [
    "is_available",
    "is_initialized",
    "synchronize",
    "current_device",
    "current_stream",
    "stream",
    "set_device",
    "device_count",
    "Stream",
    "StreamContext",
    "Event",
    "get_capabilities",
]


def _native_capabilities() -> dict:
    try:
        from . import _C as _native_mod

        probe = getattr(getattr(_native_mod, "_cpu", None), "_get_cpu_capability", None)
        if callable(probe):
            return dict(probe())
    except ImportError:
        pass
    return {"architecture": "unknown"}


@lru_cache(None)
def get_capabilities() -> Mapping[str, Any]:
    """Runtime capability flags for the host.

    Keys are feature names with boolean support values, plus an
    ``architecture`` string entry. The result is cached after the first
    call.
    """
    return MappingProxyType(_native_capabilities())


def _is_feature_supported(name: str) -> bool:
    return bool(get_capabilities().get(name, False))


def is_available() -> bool:
    """The host backend is always compiled in."""
    return True


def synchronize(device=None) -> None:
    """Wait for host work to complete; host execution is synchronous."""


class Stream(StreamBase):
    """Placeholder stream; host memory has no asynchronous ordering."""

    def __init__(self, priority: int = -1) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    def wait_stream(self, stream) -> None:
        pass

    def record_event(self) -> None:
        pass

    def wait_event(self, event) -> None:
        pass


class Event(EventBase):
    def query(self) -> bool:
        return True

    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(self, stream=None) -> None:
        pass


_default_stream = Stream()
_current_stream = _default_stream


def current_stream(device=None) -> Stream:
    """The currently selected placeholder stream for the host."""
    return _current_stream


class StreamContext(AbstractContextManager):
    """Context manager selecting a placeholder host stream."""

    cur_stream: Stream | None

    def __init__(self, stream):
        self.stream = stream
        self.prev_stream = _default_stream

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        self.prev_stream = _current_stream
        _current_stream = cur_stream

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        _current_stream = self.prev_stream


def stream(stream: Stream) -> AbstractContextManager:
    """Select a placeholder host stream within a ``with`` block."""
    return StreamContext(stream)


def device_count() -> int:
    """Number of host devices; always one."""
    return 1


def set_device(device) -> None:
    """Select the host device; there is only one, so this is a no-op."""


def current_device() -> str:
    """The host device spelling."""
    return "cpu"


def is_initialized() -> bool:
    """The host backend needs no lazy initialization."""
    return True
