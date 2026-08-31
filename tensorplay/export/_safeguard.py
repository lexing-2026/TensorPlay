"""Safeguards for global execution-state changes during capture."""

from __future__ import annotations

from typing import Any

__all__ = ["AutogradStateOpsFailSafeguard"]


class AutogradStateOpsFailSafeguard:
    """Reject grad-state mutations that cannot be represented in a graph."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._active = False

    def __enter__(self) -> "AutogradStateOpsFailSafeguard":
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self._active = False

    def check(self, function: Any, args: tuple[Any, ...] = ()) -> None:
        if not self.enabled or not self._active:
            return
        name = getattr(function, "__name__", str(function))
        if name in {"set_grad_enabled", "_set_grad_enabled", "enable_grad", "no_grad"}:
            raise RuntimeError(f"execution-state operation {name!r} is not graph-representable")

    def __call__(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        self.check(function, args)
        return function(*args, **kwargs)
