from __future__ import annotations

import functools
from typing import Any, Callable

__all__ = ["async_execution"]


def async_execution(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    wrapper._wrapped_async_rpc_function = fn  # type: ignore[attr-defined]
    return wrapper
