# mypy: allow-untyped-defs
"""Executors for traced primitive graphs.

An execution object may expose ``forward``, be a plain callable, or contain a
call record with a callable ``forward`` attribute.  The native executor keeps
argument handling in the graph object and adds no per-element Python loop.
"""

from collections.abc import Callable
from typing import Any, TypeVar

from tensorplay.primitives.context import TorchRefsMode

T = TypeVar("T")

__all__ = ["execute", "make_traced"]


def execute(
    gm: Any,
    *args: Any,
    executor: str = "native",
    executor_parameters: dict | None = None,
) -> Any:
    """Execute a traced graph callable through its native execution entry."""
    if executor == "native":
        forward = getattr(gm, "forward", None)
        if forward is not None:
            return forward(*args)
        if callable(gm):
            return gm(*args)
        raise TypeError(
            f"Received a graph object without a callable forward method: {type(gm)}"
        )
    raise ValueError(
        f"Received unexpected value for 'executor': {executor}. "
        f"Only the default executor value is supported."
    )


def make_traced(fn: Callable[..., T]) -> Callable[..., T]:
    """Return a callable that executes primitive operations in a scoped mode.

    The wrapper removes only the executor control keyword before calling the
    user function.  All positional and remaining keyword arguments retain
    their original identity and ordering.
    """

    def _traced(*args: Any, **kwargs: Any) -> T:
        executor = str(kwargs.pop("executor", "native"))
        with TorchRefsMode():
            return fn(*args, **kwargs)

    return _traced
