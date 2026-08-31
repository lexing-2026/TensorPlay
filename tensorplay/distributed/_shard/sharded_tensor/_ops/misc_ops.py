"""Miscellaneous sharded tensor predicates."""

from typing import Any

__all__ = ["tensor_has_compatible_shallow_copy_type"]


def tensor_has_compatible_shallow_copy_type(value: Any) -> bool:
    return hasattr(value, "local_shards") and hasattr(value, "metadata")
