"""Comparison operations over local shards."""

from typing import Any

__all__ = ["binary_cmp", "equal", "allclose"]


def _communicate_result(value: Any, process_group: Any = None) -> Any:
    del process_group
    return value


def binary_cmp(op: Any, left: Any, right: Any) -> Any:
    left = left.to_local() if hasattr(left, "to_local") else left
    right = right.to_local() if hasattr(right, "to_local") else right
    return _communicate_result(op(left, right))


def equal(left: Any, right: Any) -> bool:
    return bool(binary_cmp(lambda a, b: (a == b).all(), left, right).item())


def allclose(left: Any, right: Any, **kwargs: Any) -> bool:
    import tensorplay
    a = left.to_local() if hasattr(left, "to_local") else left
    b = right.to_local() if hasattr(right, "to_local") else right
    return bool(tensorplay.allclose(a, b, **kwargs))
