"""Readable representations of pipeline metadata."""

from typing import Any

__all__ = ["friendly_debug_info", "map_debug_info"]


def friendly_debug_info(value: Any) -> str:
    if hasattr(value, "shape"):
        return f"{type(value).__name__}(shape={tuple(value.shape)}, dtype={getattr(value, 'dtype', None)})"
    return repr(value)


def map_debug_info(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: map_debug_info(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(map_debug_info(item) for item in value)
    if isinstance(value, list):
        return [map_debug_info(item) for item in value]
    return friendly_debug_info(value)
