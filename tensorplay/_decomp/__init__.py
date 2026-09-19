"""Decomposition registry keyed by operator overload.

A decomposition expresses one operator overload through other operators.
Tracers consult a decomposition table (``{overload: function}``) and record
the operators the function dispatches instead of the decomposed one, so a
backend only needs kernels for the operators left over.

* :func:`register_decomposition` adds a function for overloads or whole
  overload packets (every overload of the packet).
* :func:`get_decompositions` selects the registered functions for a list of
  overloads/packets; :func:`remove_decompositions` drops entries again.
* :func:`core_decompositions` is the table that lowers everything outside
  the ``core`` operator set onto it.

Tables are split by stage: ``post_autograd`` decompositions run on graphs
that already contain the backward (they must not rely on autograd);
``pre_autograd`` ones run before autograd sees the program.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = [
    "core_decompositions",
    "decomposition_table",
    "get_decompositions",
    "global_decomposition_table",
    "pre_autograd_decomposition_table",
    "register_decomposition",
    "remove_decompositions",
]

_STAGES = ("post_autograd", "pre_autograd")

global_decomposition_table: dict[str, dict[Any, Callable[..., Any]]] = {
    stage: {} for stage in _STAGES
}
decomposition_table = global_decomposition_table["post_autograd"]
pre_autograd_decomposition_table = global_decomposition_table["pre_autograd"]


def _overloads_of(op: Any) -> list[Any]:
    from tensorplay._ops import OpOverload, OpOverloadPacket

    if isinstance(op, OpOverload):
        return [op]
    if isinstance(op, OpOverloadPacket):
        return [getattr(op, name) for name in op.overloads()]
    raise TypeError(f"expected an operator overload or packet, got {type(op).__name__}")


def _flatten_ops(ops: Any) -> Iterable[Any]:
    if isinstance(ops, (list, tuple, set)):
        for op in ops:
            yield from _flatten_ops(op)
    else:
        yield ops


def _without_out(fn: Callable[..., Any], overload: Any) -> Callable[..., Any]:
    """Serve an ``out=`` overload with a functional decomposition.

    The decomposition computes the result; the destination receives it the
    way the out-variant contract prescribes (resized to the result's shape,
    then written), and is what the call returns.
    """

    schema = overload._schema
    out_names = [a.name for a in schema.arguments if a.is_out]
    if not out_names or "out" in inspect.signature(fn).parameters:
        return fn

    def with_out(*args: Any, **kwargs: Any) -> Any:
        destinations = [kwargs.pop(name) for name in out_names if name in kwargs]
        result = fn(*args, **kwargs)
        results = result if isinstance(result, tuple) else (result,)
        for destination, value in zip(destinations, results):
            if tuple(destination.shape) != tuple(value.shape):
                destination.resize_(tuple(value.shape))
            destination.copy_(value)
        return destinations[0] if len(destinations) == 1 else tuple(destinations)

    with_out.__wrapped__ = fn  # type: ignore[attr-defined]
    return with_out


def register_decomposition(
    ops: Any,
    registry: dict[Any, Callable[..., Any]] | None = None,
    *,
    type: str = "post_autograd",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering ``fn`` as the decomposition of ``ops``.

    ``ops`` is an overload, a packet (all its overloads) or a nested list of
    them.  Registering the same overload twice is an error.
    """

    if type not in _STAGES:
        raise ValueError(f"type must be one of {_STAGES}, got {type!r}")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        table = registry if registry is not None else global_decomposition_table[type]
        for op in _flatten_ops(ops):
            for overload in _overloads_of(op):
                if overload in table:
                    raise RuntimeError(f"duplicate decomposition registered for {overload}")
                table[overload] = _without_out(fn, overload)
        return fn

    return decorator


def get_decompositions(
    ops: Sequence[Any], type: str = "post_autograd"
) -> dict[Any, Callable[..., Any]]:
    """Registered decompositions for ``ops`` (overloads or packets).

    Operators without a registered decomposition are skipped.
    """

    from tensorplay._ops import OpOverloadPacket

    if type not in _STAGES:
        raise ValueError(f"type must be one of {_STAGES}, got {type!r}")
    _load()
    table = global_decomposition_table[type]
    by_packet: dict[Any, list[Any]] = defaultdict(list)
    for overload in table:
        by_packet[overload.overloadpacket].append(overload)
    selected: dict[Any, Callable[..., Any]] = {}
    for op in _flatten_ops(list(ops)):
        if isinstance(op, OpOverloadPacket):
            for overload in by_packet.get(op, ()):
                selected[overload] = table[overload]
        elif op in table:
            selected[op] = table[op]
    return selected


def remove_decompositions(decompositions: dict[Any, Callable[..., Any]], ops: Sequence[Any]) -> None:
    """Drop the entries for ``ops`` (overloads or packets) from a table."""

    for op in _flatten_ops(list(ops)):
        for overload in _overloads_of(op):
            decompositions.pop(overload, None)


def core_decompositions() -> dict[Any, Callable[..., Any]]:
    """Decompositions lowering every non-``core`` operator that has one.

    Operators tagged ``core`` are kept: a table built from this lowers a
    program onto the core operator set.
    """

    _load()
    return {
        overload: fn
        for overload, fn in decomposition_table.items()
        if "core" not in overload.tags
    }


_loaded = False


def _load() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        from . import decompositions  # noqa: F401  (registers on import)
