"""Callable module wrappers used by export helpers."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["_WrapperModule"]


class _WrapperModule:
    def __init__(self, function: Callable[..., Any]) -> None:
        if not callable(function):
            raise TypeError("wrapped value must be callable")
        self.f = function

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.f(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
