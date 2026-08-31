"""Utilities for checking and reordering structured argument trees."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..graph._pytree import TreeSpec, tree_flatten, tree_unflatten

__all__ = ["is_equivalent", "reorder_kwargs"]


def is_equivalent(
    first: TreeSpec,
    second: TreeSpec,
    *,
    equivalence_fn: Any | None = None,
) -> bool:
    """Compare two tree descriptions, including their child ordering."""

    if equivalence_fn is not None:
        if not equivalence_fn(first.type, first.context, second.type, second.context):
            return False
    elif first.type is not second.type or first.context != second.context:
        return False
    return len(first.children_specs) == len(second.children_specs) and all(
        is_equivalent(left, right, equivalence_fn=equivalence_fn)
        for left, right in zip(first.children_specs, second.children_specs)
    )


def reorder_kwargs(user_kwargs: dict[str, Any], spec: TreeSpec) -> dict[str, Any]:
    """Return keyword values in the order encoded by a mapping tree."""

    if not isinstance(user_kwargs, dict):
        raise TypeError("user_kwargs must be a dictionary")
    if spec.type is dict:
        keys = tuple(spec.context or ())
        missing = [key for key in keys if key not in user_kwargs]
        extra = [key for key in user_kwargs if key not in keys]
        if missing or extra:
            raise TypeError(f"keyword tree mismatch: missing={missing}, extra={extra}")
        return {key: user_kwargs[key] for key in keys}
    flattened, actual = tree_flatten(user_kwargs)
    if not is_equivalent(actual, spec):
        raise TypeError("keyword tree does not match the expected specification")
    rebuilt = tree_unflatten(flattened, spec)
    return dict(rebuilt) if isinstance(rebuilt, Mapping) else user_kwargs
