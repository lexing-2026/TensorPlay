"""Attach optional type information to symbolic graph values."""

from __future__ import annotations

from typing import Any

from ._compatibility import compatibility
from .proxy import Proxy

__all__ = ["annotate"]


@compatibility(is_backward_compatible=False)
def annotate(val: Any, type: type) -> Any:
    """Record ``type`` on a symbolic value and return that value unchanged."""

    if not isinstance(val, Proxy):
        return val
    if val.node.type is not None:
        raise RuntimeError(
            "Tried to annotate a value that already has type information: "
            f"existing={val.node.type!r}, new={type!r}"
        )
    val.node.type = type
    return val
