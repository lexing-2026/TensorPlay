"""Context for distributed loss computations."""

from __future__ import annotations

import contextlib
import contextvars

__all__ = ["loss_parallel"]

_enabled = contextvars.ContextVar("tensorplay_loss_parallel", default=False)


@contextlib.contextmanager
def loss_parallel():
    token = _enabled.set(True)
    try:
        yield
    finally:
        _enabled.reset(token)


def is_loss_parallel_enabled() -> bool:
    return _enabled.get()
