"""Handing out slices of one allocation, by byte offset.

Generated code that computes several values in one place has no reason to give
each of them its own allocation: they are written once, read a known number of
times, and none of them outlives the call.  So they are carved out of one
allocation instead, each at an offset the planner has already settled, and the
shape and stride it settled are what the generated source says to carve.

Which is only sound if the carve is honest, and that is what this module is
for.  A carve is checked against the pool before it is made: the offset has to
be inside the pool, the bytes it claims have to fit, and the alignment has to be
one the element type can be addressed at.  A planner that got one of those
wrong produces a kernel that reads a neighbour's memory, which is a wrong
answer rather than a crash, so the check is here rather than trusted.
"""

from __future__ import annotations

import threading
from typing import Any

import tensorplay as tp

#: Element sizes by dtype, used to check that a carve is addressable.  A dtype
#: absent from this table cannot be carved, because the size of its element is
#: not known here and a guess would be a guess about alignment too.
_ITEMSIZE: dict[Any, int] = {
    tp.bool: 1,
    tp.uint8: 1,
    tp.int8: 1,
    tp.uint16: 2,
    tp.int16: 2,
    tp.float16: 2,
    tp.bfloat16: 2,
    tp.uint32: 4,
    tp.int32: 4,
    tp.float32: 4,
    tp.uint64: 8,
    tp.int64: 8,
    tp.float64: 8,
    tp.complex64: 8,
    tp.complex128: 16,
}

#: The alignment every carve starts at.  Generous enough for any element type
#: above and a whole cache line, so a carve never shares a line with whatever
#: the planner put before it.
_CARVE_ALIGNMENT = 64

#: Pools by name, and the lock that guards them.  A pool is filled once, by the
#: planner, and read by every call that uses it; the lock is around filling.
_pools: dict[str, Any] = {}
_pool_lock = threading.Lock()


def itemsize(dtype: Any) -> int:
    """How many bytes one element of this type takes."""

    size = _ITEMSIZE.get(dtype)
    if size is None:
        raise TypeError(
            f"an element of type {dtype} has no known size, so a block of them "
            "cannot be carved out of a shared allocation"
        )
    return size


def define_pool(name: str, nbytes: int, device: Any = None) -> None:
    """Start a pool of ``nbytes`` under ``name``.

    Called by the planner once per generated unit, before any carve is made
    from it.  Defining a pool that already exists replaces it, so a second
    compilation of the same unit starts from a clean pool rather than refusing.
    """

    with _pool_lock:
        _pools[name] = {
            "nbytes": int(nbytes),
            "device": device,
            "base": None,
        }


def pool_is_defined(name: str) -> bool:
    """Whether a pool has been started under this name."""

    return name in _pools


def _check_carve(name: str, offset: int, nbytes: int, align: int) -> None:
    """Whether a carve of ``nbytes`` at ``offset`` is inside the pool.

    Raises rather than returning a verdict, because the caller cannot do
    anything useful with a carve it is not allowed to make: the alternative is a
    tensor over memory that belongs to something else.
    """

    pool = _pools.get(name)
    if pool is None:
        raise KeyError(f"no allocation pool is defined under {name!r}")
    if offset % align:
        raise ValueError(
            f"a carve at offset {offset} is not a multiple of {align}, so its "
            "elements would not be addressable"
        )
    if offset < 0 or offset + nbytes > pool["nbytes"]:
        raise ValueError(
            f"a carve of {nbytes} bytes at offset {offset} does not fit in the "
            f"{pool['nbytes']} bytes of pool {name!r}"
        )


def alloc_from_pool(
    name: str,
    offset: int,
    dtype: Any,
    shape: Any,
    stride: Any,
) -> Any:
    """A tensor over the bytes of pool ``name`` starting at ``offset``.

    The first call for a pool makes the allocation it carves from; later calls
    return windows onto that same allocation, which is what makes the offsets
    meaningful.  ``shape`` and ``stride`` are what the planner settled, and are
    checked against the bytes claimed before anything is handed out.
    """

    element = itemsize(dtype)
    extents = [int(extent) for extent in shape]
    strides = [int(step) for step in stride]
    if len(extents) != len(strides):
        raise ValueError(
            f"a carve of {len(extents)} dimensions was given {len(strides)} strides"
        )

    nbytes = 0
    if all(extent != 0 for extent in extents):
        # The bytes a window covers are measured from the low end of the run it
        # walks, so a negative stride reaches its furthest element by going
        # backwards from the start.
        span = 1
        for extent, step in zip(extents, strides):
            span = max(span, abs(step) * extent)
        nbytes = span * element

    _check_carve(name, int(offset), nbytes, _CARVE_ALIGNMENT)

    with _pool_lock:
        pool = _pools[name]
        if pool["base"] is None:
            pool["base"] = tp.empty(
                pool["nbytes"], dtype=tp.uint8, device=pool["device"]
            )

    return tp.as_strided(
        tp.as_strided(pool["base"], (int(offset),), (1,)),
        tuple(extents),
        tuple(strides),
        0,
    )


def clear_pools() -> None:
    """Forget every pool, releasing what they held.

    The allocations go with them, which is what a caller wants between two
    compilations: the first one's tensors are dead by then and holding them
    would keep their memory for the rest of the process.
    """

    with _pool_lock:
        _pools.clear()


__all__ = [
    "alloc_from_pool",
    "clear_pools",
    "define_pool",
    "itemsize",
    "pool_is_defined",
]
