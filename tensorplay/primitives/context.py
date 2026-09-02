# mypy: allow-untyped-defs
"""Interpretation mode for the primitives layer.

The context maps public tensor operations and tensor methods to primitive
implementations.  Nested contexts preserve the active mode, while a
primitive may opt out of interception through the pass-through set.  This
keeps recursive primitive calls from being interpreted twice and gives
tracing layers one scope for the complete captured operation.
"""

from __future__ import annotations

import functools
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import tensorplay
from tensorplay import primitives
from tensorplay.primitives.common import TensorLike

# Operations in this set bypass interpretation and execute directly.
function_passthrough = frozenset({})

_R = TypeVar("_R")


@functools.cache
def tensorplay_to_refs_map() -> dict[Any, Any]:
    """Build the callable-to-primitive mapping used by the active mode.

    Public functions exported by the primitive namespace are matched by
    object identity.  Tensor methods that have no package-level function are
    registered explicitly, so method syntax and function syntax use the same
    implementation.  The result is cached because the callable identities do
    not change during a process lifetime.
    """
    r: dict[Any, Any] = {}
    for s in primitives.__all__:
        value = getattr(primitives, s, None)
        if callable(value):
            op = getattr(tensorplay, s, None)
            if op is not None:
                r[op] = value
    # Tensor methods that are primitive-shaped map back to the prim
    for method, prim_name in [
        ("copy_", "copy_to"),
        ("resize", "resize"),
    ]:
        method_obj = getattr(tensorplay.Tensor, method, None)
        prim_obj = getattr(primitives, prim_name, None)
        if method_obj is not None and prim_obj is not None:
            r[method_obj] = prim_obj
    return r


@functools.cache
def all_prims() -> set[Any]:
    """Set of all primitive callables defined by this package."""
    return {getattr(primitives, s) for s in primitives.__all__ if hasattr(primitives, s)}


class TensorPlayRefsMode:
    """Context manager that reinterprets public API calls as primitives.

    ``strict`` raises when an operation has no registered primitive.  The
    fallback callback can selectively retain eager execution for operations
    whose primitive is not appropriate for a call site.  Direct calls to a
    primitive bypass the mapping; recursive calls made by a selected
    primitive remain inside the current mode.
    """

    def __init__(
        self,
        strict: bool = False,
        should_fallback_fn: Callable[..., bool] = lambda *_: False,
        prims_mode_cls: type = nullcontext,
    ) -> None:
        self.strict = strict
        self.should_fallback_fn = should_fallback_fn
        self.prims_mode_cls = prims_mode_cls

    def __enter__(self):
        self._prev = getattr(_tensorplay_refs_mode_state, "value", None)
        _tensorplay_refs_mode_state.value = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev is None:
            del _tensorplay_refs_mode_state.value
        else:
            _tensorplay_refs_mode_state.value = self._prev
        return False

    def __tensorplay_function__(
        self,
        orig_func: Callable[..., Any],
        types: Sequence[type],
        args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        if orig_func in function_passthrough or orig_func in all_prims():
            with self.prims_mode_cls():
                return orig_func(*args, **kwargs)
        mapping = tensorplay_to_refs_map()
        func = mapping.get(orig_func, None)
        if func is not None:
            if self.should_fallback_fn(self, orig_func, func, args, kwargs):
                return orig_func(*args, **kwargs)
            with self:
                return func(*args, **kwargs)
        if self.strict:
            raise RuntimeError(f"no refs support for {getattr(orig_func, '__name__', orig_func)}")
        return orig_func(*args, **kwargs)


class _ModeState:
    value: TensorPlayRefsMode | None = None


_tensorplay_refs_mode_state = _ModeState()


__all__ = [
    "TensorPlayRefsMode",
    "all_prims",
    "tensorplay_to_refs_map",
    "function_passthrough",
]
