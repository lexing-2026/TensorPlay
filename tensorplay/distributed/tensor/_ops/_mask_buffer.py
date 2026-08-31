"""Reusable masks for partial and uneven shard operations."""

from __future__ import annotations

from typing import Any

__all__ = ["MaskBuffer"]


class MaskBuffer:
    def __init__(self) -> None:
        self._values: dict[tuple[Any, ...], Any] = {}

    def get(self, key: tuple[Any, ...]) -> Any:
        return self._values.get(tuple(key))

    def set(self, key: tuple[Any, ...], value: Any) -> Any:
        self._values[tuple(key)] = value
        return value

    def clear(self) -> None:
        self._values.clear()
