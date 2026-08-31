from __future__ import annotations

import warnings
from typing import Any

import tensorplay as tp

__all__ = ["get_untyped_storages"]


def _is_wrapper(value: Any) -> bool:
    return (
        isinstance(value, tp.Tensor)
        and type(value) is not tp.Tensor
        and callable(getattr(value, "__tensor_flatten__", None))
    )


def get_untyped_storages(t: tp.Tensor) -> set[Any]:
    """Collect the underlying storage objects reachable from a tensor."""
    pending = [t]
    result: set[Any] = set()
    while pending:
        value = pending.pop()
        if _is_wrapper(value):
            attrs, _ = value.__tensor_flatten__()
            for name in attrs:
                child = getattr(value, name)
                if isinstance(child, tp.Tensor):
                    pending.append(child)
                elif child is not None:
                    raise AssertionError(f"wrapper attribute {name!r} is not a tensor")
            continue
        storage_fn = getattr(value, "untyped_storage", None)
        if not callable(storage_fn):
            warnings.warn(
                f"expected a tensor-like object, got {type(value)!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        result.add(storage_fn())
    return result
