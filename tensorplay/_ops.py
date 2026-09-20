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
_NATIVE_FALLBACKS: dict[str, Any] = {
    "_scaled_dot_product_flash_attention": _flash_attention_adapter,
}


# Namespace of the operators declared in the op contract (config/); an
# interop identifier the public ``tensorplay.ops.<ns>`` surface already uses.
NATIVE_NAMESPACE = "tp"


class OpOverload:
    """One operator overload of the op contract (``add.Tensor``).

    Calling it runs exactly that overload through the dispatcher.  Dispatch
    modes receive these objects as ``func``; ``_schema`` describes the
    arguments, returns and alias annotations.
    """

    # The dunder identity attributes are written per instance; ``__dict__``
    # carries them because CPython forbids ``__name__``/``__qualname__``/
    # ``__module__`` inside ``__slots__`` (they collide with reserved
    # class-level attributes).
    __slots__ = (
        "_schema",
        "_overloadpacket",
        "_overloadname",
        "_opname",
        "_key",
        "_tags",
        "__weakref__",
        "__dict__",
    )

    def __init__(self, packet: "OpOverloadPacket", key: str, schema: Any, tags: tuple[str, ...]) -> None:
        self._schema = schema
        self._overloadpacket = packet
        self._overloadname = schema.overload_name or "default"
        self._opname = schema.name
        self._key = key
        self._tags = tags
        self.__name__ = f"{schema.name}.{self._overloadname}"
        self.__qualname__ = self.__name__
        self.__module__ = f"tensorplay.ops.{schema.namespace}"

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        return _C._call_overload(self._key, args, kwargs)

    @property
    def overloadpacket(self) -> "OpOverloadPacket":
        return self._overloadpacket

    @property
    def op(self) -> "OpOverload":
        return self

    @property
    def namespace(self) -> str:
        return self._schema.namespace

    @property
    def tags(self) -> tuple[str, ...]:
        return self._tags

    @property
    def is_view(self) -> bool:
        return self._schema._is_view_op()

    def name(self) -> str:
        return f"{self._schema.namespace}::{self._key}"

    def has_kernel_for_dispatch_key(self, key: str) -> bool:
        return bool(_C._dispatch_has_kernel_for_dispatch_key(self._key, key))

    def __repr__(self) -> str:
        return (
            f"<OpOverload(op='{self._schema.namespace}.{self._opname}', "
            f"overload='{self._overloadname}')>"
        )

    def __str__(self) -> str:
        return f"{self._schema.namespace}.{self._opname}.{self._overloadname}"

    def __reduce__(self) -> Any:
        return (_overload_for_dispatch, (self._key,))

    def __deepcopy__(self, memo: Any) -> "OpOverload":
        return self


class OpOverloadPacket:
    """All overloads of one operator name (``add``).

    Attribute access yields an overload (``.Tensor``, ``.default``); calling
    the packet resolves the overload from the arguments.
    """

    def __init__(self, namespace: str, name: str) -> None:
        self._namespace = namespace
        self._opname = name
        self._qualified_op_name = f"{namespace}::{name}"
        self.__name__ = name
        self.__qualname__ = name
        self.__module__ = f"tensorplay.ops.{namespace}"
        self._overloads: dict[str, OpOverload] = {}

    def _add(self, overload: OpOverload) -> None:
        self._overloads[overload._overloadname] = overload

    def overloads(self) -> list[str]:
        return list(self._overloads)

    def __getattr__(self, name: str) -> OpOverload:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self._overloads[name]
        except KeyError:
            raise AttributeError(
                f"'{self._qualified_op_name}' has no overload named '{name}'"
            ) from None

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        resolver = getattr(_C, self._opname, None)
        if resolver is not None:
            return resolver(*args, **kwargs)
        # No public binding: the single declared overload, or the first
        # whose arguments bind.
        errors: list[str] = []
        for overload in self._overloads.values():
            try:
                return overload(*args, **kwargs)
            except TypeError as exc:
                errors.append(f"{overload}: {exc}")
        raise TypeError(
            f"no overload of {self._qualified_op_name} accepts these arguments:\n  "
            + "\n  ".join(errors)
        )

    def __repr__(self) -> str:
        return f"<OpOverloadPacket(op='{self._namespace}.{self._opname}')>"

    def __str__(self) -> str:
        return f"{self._namespace}.{self._opname}"

    def __reduce__(self) -> Any:
        return (_packet_for, (self._opname,))


_packets: dict[str, OpOverloadPacket] | None = None
_overloads_by_key: dict[str, OpOverload] = {}


def _load_native_overloads() -> dict[str, OpOverloadPacket]:
    global _packets
    if _packets is not None:
        return _packets
    from ._function_schema import parse_schema

    packets: dict[str, OpOverloadPacket] = {}
    entries = getattr(_C, "_python_dispatch_entries", None)
    for key, schema_text, _names, _npos, tags in (entries() if entries else ()):
        schema = parse_schema(schema_text, namespace=NATIVE_NAMESPACE)
        packet = packets.get(schema.name)
        if packet is None:
            packet = packets[schema.name] = OpOverloadPacket(NATIVE_NAMESPACE, schema.name)
        overload = OpOverload(packet, key, schema, tuple(t for t in tags.split(",") if t))
        packet._add(overload)
        _overloads_by_key[key] = overload
    _packets = packets
    return packets


def _packet_for(name: str) -> OpOverloadPacket:
    return _load_native_overloads()[name]


def _overload_for_dispatch(key: str) -> OpOverload:
    """The interned overload object for a dispatcher key (``add.Tensor``)."""

    _load_native_overloads()
    return _overloads_by_key[key]


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
        if self.ns == NATIVE_NAMESPACE:
            # Composite fallbacks come first: they wrap the fused kernels of
            # this build for contracts without their own registration.
            fallback = _NATIVE_FALLBACKS.get(opname)
            if fallback is not None:
                return fallback
            packet = _load_native_overloads().get(opname)
            if packet is not None:
                setattr(self, opname, packet)
                return packet
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
