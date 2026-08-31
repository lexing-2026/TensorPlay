from __future__ import annotations

from typing import Any, Optional

from .tracer import Tracer


def symbolic_trace(root: Any, concrete_args: Optional[dict[str, Any]] = None):
    return Tracer(concrete_args=concrete_args).trace(root)


def wrap(fn_or_name=None):
    if fn_or_name is None:
        return lambda fn: fn
    if callable(fn_or_name):
        return fn_or_name
    return lambda fn: fn


__all__ = ["symbolic_trace", "wrap"]
