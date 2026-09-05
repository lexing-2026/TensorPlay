"""Shared helpers for higher-order operators.

The entry points here mirror the contract of a traced higher-order call:
``setup_compilation_env`` prepares the capture state and yields the backend
that inner ``compile`` invocations should target.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator


@contextlib.contextmanager
def setup_compilation_env() -> Iterator[Any]:
    """
    Context manager that sets up the environment and backend for ``compile``
    invoked inside a higher-order operator or an export region.

    Yields the backend that the inner compile call should pass on.
    """
    from tensorplay.compiler import get_default_backend

    yield get_default_backend()
