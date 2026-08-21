"""Stax backend namespace.

Stax is not a second public compiler frontend.  Use ``tensorplay.compile``
and select ``backend="stax"`` once the backend has a production lowering.
The native ``tensorplay._C._stax`` module remains available for low-level IR
and optimizer development.
"""

from __future__ import annotations

from .backends.stax import stax

try:
    from . import _C

    _native = getattr(_C, "_stax", None)
except ImportError:  # pragma: no cover - source trees without a build
    _native = None


def is_available() -> bool:
    """Return whether the native Stax extension is loaded."""

    return _native is not None


__all__ = ["is_available", "stax"]

