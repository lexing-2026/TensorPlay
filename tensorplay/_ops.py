"""

Two kinds of entries resolve here:

1. Python-registered operators (:mod:`tensorplay.library`):
   ``tensorplay.ops.mylib.add(x, y)`` returns the :class:`CustomOpDef` and
   calling it runs the normal dispatch path (autograd, capture awareness).
2. Natively loaded extension libraries: ``tensorplay.ops.load_library(path)``
   dlopens a shared object whose static registrars feed the p10 dispatcher
   (the ``TENSORPLAY_LIBRARY_IMPL`` macro family) and attaches the module
"""

from __future__ import annotations

import math
import types
from typing import Any

import tensorplay
import tensorplay._C as _C


def _lower_right_causal_mask(query: Any, key: Any) -> Any:
    """Boolean keep-mask aligned to the lower-right (L, S) corner."""
    L, S = query.size(-2), key.size(-2)
    q_idx = tensorplay.arange(L, device=query.device).view(L, 1)
    k_idx = tensorplay.arange(S, device=query.device).view(1, S)
    return q_idx >= k_idx - (S - L)


def _flash_attention_adapter(
    query: Any,
    key: Any,
    value: Any,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Any = None,
) -> tuple[Any, ...]:
    """Flash-attention composite over the fused kernels shipped in this build.

    The dispatcher contract calls for a nine-field result, with the causal
    flag aligned to the lower-right (L, S) corner.  The fused kernels align
    their causal mask to the query index (the upper-left corner), which only
    coincides for square sequence lengths; non-square causal calls therefore
    run through the math composite with an explicit lower-right mask.  The
    CPU fused kernel returns the output and the per-row logsumexp; the CUDA
    fused kernel returns the output only, and takes no scale argument, so a
    non-default scale is folded into the query — rescaling the scores by
    ``s`` equals rescaling ``q`` by ``s * sqrt(E)`` given the kernel's
    built-in ``1 / sqrt(E)`` factor.
    """
    del return_debug_mask
    if dropout_p != 0.0:
        raise NotImplementedError(
            "flash attention: dropout > 0 is not supported in this build"
        )
    empty = tensorplay.empty(0, dtype=query.dtype, device=query.device)
    rng_state = tensorplay.zeros((2,), dtype=tensorplay.uint64, device=query.device)
    max_q, max_k = query.size(-2), key.size(-2)
    if is_causal and query.size(-2) != key.size(-2):
        keep = _lower_right_causal_mask(query, key)
        fmask = tensorplay.where(
            keep,
            tensorplay.zeros((), dtype=query.dtype, device=query.device),
            tensorplay.full((), float("-inf"), dtype=query.dtype, device=query.device),
        )
        out, lse = _C._scaled_dot_product_attention_math(
            query, key, value, fmask, 0.0, False, None, scale=scale
        )
        return out, lse, empty, empty, max_q, max_k, rng_state, empty, empty
    if query.device.type == "cpu":
        out, lse = _C._scaled_dot_product_flash_attention_for_cpu(
            query, key, value, dropout_p, is_causal, attn_mask=None, scale=scale
        )
        return out, lse, empty, empty, max_q, max_k, rng_state, empty, empty
    if scale is not None:
        head_dim = query.size(-1)
        query = query * (scale * math.sqrt(head_dim))
    out = _C.scaled_dot_product_attention(query, key, value, is_causal, 1)
    return out, empty, empty, empty, max_q, max_k, rng_state, empty, empty


# Composite contracts declared in the op schema set that have no dedicated
# kernel registration in this build.  Each entry adapts over the kernels
# that do exist; anything without an entry resolves through the native
# dispatcher and raises its own "kernel not found" error.
_ATEN_FALLBACKS: dict[str, Any] = {
    "_scaled_dot_product_flash_attention": _flash_attention_adapter,
}


class _OpNamespace(types.ModuleType):
    """Attribute-access packet for one operator namespace (``ns``)."""

    def __init__(self, ns: str) -> None:
        super().__init__(f"tensorplay.ops.{ns}")
        self.ns = ns

    def __getattr__(self, opname: str) -> Any:
        # Native extension modules registered via load_library win: they are
        # real submodules placed on this namespace.
        own = self.__dict__.get(opname)
        if own is not None:
            return own
        if self.ns == "aten":
            # Composite fallbacks come first: they wrap the fused kernels of
            # this build for contracts without their own registration.
            fallback = _ATEN_FALLBACKS.get(opname)
            if fallback is not None:
                return fallback
            native = getattr(_C, opname, None)
            if native is not None:
                return native
        full_name = f"{self.ns}::{opname}"
        if tensorplay.library.has_op(full_name):
            return tensorplay.library.get_op(full_name)
        raise AttributeError(
            f"No operator {full_name!r} is registered; define it with "
            f"tensorplay.library.custom_op(\"{full_name}\") or load its "
            "extension library via tensorplay.ops.load_library"
        )


class _Ops(types.ModuleType):
    """The ``tensorplay.ops`` root namespace."""

    __file__ = "_ops.py"

    def __getattr__(self, name: str) -> _OpNamespace:
        if name.startswith("_"):
            raise AttributeError(name)
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        return namespace

    @property
    def load_library(self) -> Any:
        return _C.ops.load_library

    @property
    def loaded_libraries(self) -> Any:
        return getattr(_C.ops, "loaded_libraries")


ops = _Ops("tensorplay.ops")
