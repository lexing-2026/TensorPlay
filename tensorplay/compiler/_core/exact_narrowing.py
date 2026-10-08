"""Moving values around in the type they were widened from.

A program run in mixed precision widens a value before an operation that is
listed as wanting the wider type, and narrows it again for the next one that
does not.  Some of the operations in between only move values -- picking the
nearest neighbour, laying tensors end to end -- and a value that was exact in
the narrow type before being moved is exact in it afterwards.  Moving the
narrow values and widening the result gives the same numbers, element for
element, and leaves the wide tensor as something a reader can compute while
reading instead of something that has to be written out at twice the size.

The rules here are consulted while a region is being recorded, before it is
differentiated, so the backward that is recorded is the backward of what the
forward actually does:

* a widening cast is recorded as written, and remembered as one;
* an operation that only moves values, handed nothing but widened values and
  values already in the narrow type, moves the narrow ones and widens after;
* narrowing a value back to the type it was widened from gives the narrow
  value itself, when that value is one of the above and so one the program
  never held.

A rule that does not apply declines, and the call is recorded as it stands.
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable

import tensorplay as tp

__all__ = ["exact_narrowing_rules"]


#: Pairs (narrow, wide) where every narrow value has exactly one wide spelling
#: and comes back unchanged.
_EXACT_WIDENINGS = frozenset({
    (tp.float16, tp.float32),
    (tp.float16, tp.float64),
    (tp.bfloat16, tp.float32),
    (tp.bfloat16, tp.float64),
    (tp.float32, tp.float64),
})

#: Set while a cast is being recorded as written, so the rule that asked for
#: it does not answer for it again.
_AS_WRITTEN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "exact_narrowing_as_written", default=False
)

# What a node of the trace is remembered by.
_SOURCE_VERSION = "widened_source_version"
_NARROW_TEMPORARY = "narrow_temporary"
_HANDED_OUT = "narrow_temporary_handed_out"


def _tracer():
    from tensorplay.graph.experimental.proxy_tensor import get_proxy_mode

    mode = get_proxy_mode()
    tracer = getattr(mode, "tracer", None)
    return tracer if hasattr(tracer, "producer") else None


def _is_tensor(value: Any) -> bool:
    return isinstance(value, tp.Tensor)


def _widened_from(value: Any):
    """``(narrow tensor, its node)`` this value is an exact widening of.

    Only a cast this module recorded counts, and only while the narrow tensor
    is still what it was when the cast read it: a tensor written to in place
    since -- directly or through a view -- is a different value under the same
    name.
    """

    tracer = _tracer()
    if tracer is None or not _is_tensor(value):
        return None
    node = tracer.producer(value)
    if node is None or _SOURCE_VERSION not in node.meta:
        return None
    source_node = node.args[0]
    source = node.meta.get("widened_source")
    if source is None:
        source = getattr(source_node, "meta", {}).get("val")
    if not _is_tensor(source) or tracer.producer(source) is not source_node:
        return None
    if source._version != node.meta[_SOURCE_VERSION]:
        return None
    return source, source_node


def _widen(narrow: Any, dtype: Any) -> Any:
    """Record ``narrow`` as a value the program never held, and widen it."""

    tracer = _tracer()
    node = tracer.producer(narrow) if tracer is not None else None
    if node is not None:
        node.meta[_NARROW_TEMPORARY] = True
    return narrow.to(dtype)


def _cast_rule(func: Any, has_device: bool) -> Callable[..., Any]:
    names = (["device"] if has_device else []) + [
        "dtype", "non_blocking", "copy", "memory_format",
    ]

    def rule(x, *args, **kwargs):
        if _AS_WRITTEN.get() or not _is_tensor(x) or len(args) > len(names):
            return NotImplemented
        given = dict(zip(names, args))
        given.update(kwargs)
        dtype = given.get("dtype")
        if (
            dtype is None
            or dtype == x.dtype
            or given.get("copy")
            or given.get("memory_format") is not None
        ):
            return NotImplemented
        if has_device and tp.device(given.get("device")) != x.device:
            return NotImplemented

        found = _widened_from(x)
        if found is not None:
            source, source_node = found
            # Back to the type it came from: the narrow value itself, once,
            # and only one the program never held -- a caller is free to
            # write into what a cast returns.
            if (
                source.dtype == dtype
                and source_node.meta.get(_NARROW_TEMPORARY)
                and not source_node.meta.get(_HANDED_OUT)
            ):
                source_node.meta[_HANDED_OUT] = True
                return source
            return NotImplemented

        if (x.dtype, dtype) not in _EXACT_WIDENINGS:
            return NotImplemented
        tracer = _tracer()
        if tracer is None:
            return NotImplemented
        version = x._version
        token = _AS_WRITTEN.set(True)
        try:
            out = func(x, *args, **kwargs)
        finally:
            _AS_WRITTEN.reset(token)
        node = tracer.producer(out)
        if node is not None and out is not x:
            node.meta[_SOURCE_VERSION] = version
            node.meta["widened_source"] = x
        return out

    return rule


def _nearest_rule(ndim: int) -> Callable[..., Any]:
    def rule(x, output_size, *scales, **kwargs):
        found = _widened_from(x)
        if found is None:
            return NotImplemented
        source, _ = found
        if source.dim() <= ndim:
            return NotImplemented
        try:
            extents = [int(s) for s in source.shape[-ndim:]]
            wanted = [int(s) for s in output_size]
        except (TypeError, ValueError):
            return NotImplemented
        if len(wanted) != ndim or any(not e or w % e for w, e in zip(wanted, extents)):
            return NotImplemented
        factors = [w // e for w, e in zip(wanted, extents)]
        given = list(scales) + list(kwargs.values())
        if len(given) > ndim:
            return NotImplemented
        # A scale spelled differently from the sizes picks different
        # neighbours, which is no longer each element repeated in place.
        for scale, factor in zip(given, factors):
            if scale is not None and float(scale) != float(factor):
                return NotImplemented
        # Every element repeated along each spatial axis: an axis of one put
        # beside the axis, grown to the factor, and folded back into it.  The
        # moving is spelled with shapes alone, which no precision policy
        # reads, so the narrow values stay narrow however the graph is run.
        lead = [int(s) for s in source.shape[:-ndim]]
        split, grown = list(lead), list(lead)
        for extent, factor in zip(extents, factors):
            split += [extent, 1]
            grown += [extent, factor]
        narrow = source.reshape(split).expand(grown).reshape(lead + wanted)
        return _widen(narrow, x.dtype)

    return rule


def _cat_rule(func: Any) -> Callable[..., Any]:
    def rule(tensors, *args, **kwargs):
        if not isinstance(tensors, (list, tuple)) or not tensors:
            return NotImplemented
        if not all(_is_tensor(t) for t in tensors):
            return NotImplemented
        sources = []
        wide = None
        for tensor in tensors:
            found = _widened_from(tensor)
            if found is None:
                sources.append(tensor)
                continue
            if wide is not None and tensor.dtype != wide:
                return NotImplemented
            wide = tensor.dtype
            sources.append(found[0])
        if wide is None:
            return NotImplemented
        narrow = sources[0].dtype
        # Every operand has to be the same narrow type once the widenings are
        # looked through; anything else and the answer is not that type.
        if (narrow, wide) not in _EXACT_WIDENINGS:
            return NotImplemented
        if any(s.dtype != narrow for s in sources):
            return NotImplemented
        # Operands of one type are laid end to end in that type under any
        # precision policy, so the call needs nothing held off around it.
        joined = func(type(tensors)(sources), *args, **kwargs)
        return _widen(joined, wide)

    return rule


def exact_narrowing_rules() -> dict[Any, Callable[..., Any]]:
    """The rules, keyed by the operation each one answers for."""

    ops = tp.ops.tp
    rules: dict[Any, Callable[..., Any]] = {
        ops.to.dtype: _cast_rule(ops.to.dtype, has_device=False),
        ops.to.device: _cast_rule(ops.to.device, has_device=True),
        ops.cat.default: _cat_rule(ops.cat.default),
    }
    for name, ndim in (
        ("upsample_nearest1d", 1), ("upsample_nearest2d", 2),
        ("upsample_nearest3d", 3), ("_upsample_nearest_exact1d", 1),
        ("_upsample_nearest_exact2d", 2), ("_upsample_nearest_exact3d", 3),
    ):
        packet = getattr(ops, name, None)
        overload = getattr(packet, "default", None)
        if overload is not None:
            rules[overload] = _nearest_rule(ndim)
    return rules
