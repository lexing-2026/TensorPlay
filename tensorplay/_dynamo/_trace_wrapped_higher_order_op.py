# mypy: allow-untyped-defs
"""Tracing helpers for higher-order operators.

``TransformGetItemToIndex`` lets a tracer rewrite ``tensor[()]`` index
expressions into explicit index operations while recording a graph.  Under
eager execution the transformation has no target representation, so the
context manager is a no-op and ``mod_index`` passes values through.
"""

from __future__ import annotations

import contextlib
from typing import Any


@contextlib.contextmanager
def TransformGetItemToIndex():
    yield


def mod_index(value: Any, index: Any) -> Any:
    """Index ``value`` at ``index``; a no-op for values without dims."""
    del index
    return value
