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

import types
from contextvars import ContextVar
from typing import Any

import tensorplay
import tensorplay._C as _C

_active_tracer_getter: Any = None

_symbolic_call_mode = ContextVar("tensorplay_symbolic_call_mode", default=None)

_FLOAT_WIDTH = {
    tensorplay.float16: 0,
    tensorplay.bfloat16: 0,
    tensorplay.float32: 1,
    tensorplay.float64: 2,
}
# These backward entry points take the gradient in a wider type than the
# activations they were saved with, and re-cast everything to the activation
# type themselves.  Forcing one element type here would move them off their
# native-precision kernels.
_MIXED_DTYPE_OPS = {
    "_scaled_dot_product_attention_backward_with_lse",
    "scaled_dot_product_attention_backward",
}

# Operators whose native kernels require every floating tensor operand to
# share one element type.  Pointwise operators promote their own mixed inputs
# (a half activation next to a float weight is read in float), so aligning
# those here only adds runtime casts that the kernels would widen themselves.
_NEEDS_DTYPE_ALIGN_OPS = {
    "conv2d",
    "convolution",
    "conv_transpose",
    "conv2d_grad_input",
    "conv2d_grad_weight",
    "conv2d_grad_bias",
    "matmul",
    "addmm",
    "mm",
    "bmm",
    "linear",
    "native_group_norm",
    "native_group_norm_backward",
    "layer_norm",
    "native_layer_norm",
    "native_layer_norm_backward",
    "avg_pool2d",
    "avg_pool2d_backward",
    "max_pool2d",
    "max_pool2d_backward",
    "upsample_nearest2d",
    "upsample_nearest2d_backward",
    "_scaled_dot_product_attention_forward",
}

def _align_eager_dtypes(opname: str, args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Give every tensor operand of a call one element type.

    The compiled backward can hand the dispatch a half-precision gradient
    next to a single-precision activation saved earlier in the graph.  Each
    framework kernel expects one element type, so the floating operands are
    promoted to the widest one among them before the call goes out.
    """

    if opname == "to" or opname in _MIXED_DTYPE_OPS:
        return args
    if opname not in _NEEDS_DTYPE_ALIGN_OPS:
        return args
    tensors = [a for a in args if isinstance(a, tensorplay.Tensor)]
    if len(tensors) < 2:
        return args
    target = None
    for t in tensors:
        rank = _FLOAT_WIDTH.get(t.dtype, -1)
        if rank < 0:
            return args
        if target is None or rank > _FLOAT_WIDTH[target]:
            target = t.dtype
    if all(t.dtype == target for t in tensors):
        return args
    return tuple(
        a.to(target) if isinstance(a, tensorplay.Tensor) and a.dtype != target else a
        for a in args
    )


def _capture_may_be_active() -> Any:
    """Whether a graph tracer is recording in this context.

    A proxy exists only while a tracer is active, so a call arriving with no
    active tracer can never be part of a capture.  The gate is one
    thread-local read, taken in place of scanning every argument of every
    call on paths that never record.
    """
    global _active_tracer_getter
    if _active_tracer_getter is None:
        from .graph._utils import get_active_tracer

        _active_tracer_getter = get_active_tracer
    return _active_tracer_getter() is not None


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
        # While a graph is being captured, an overload reached with a
        # symbolic argument is recorded rather than run: the arguments are
        # descriptions of values that do not exist yet, so there is nothing to
        # compute, and running it would ask the operator for the type of
        # something that has not been made.  Proxies only exist under an
        # active tracer, so the check is worth its cost only there -- the
        # steady-state call (eager op, compiled artifact, backward) is a
        # single dispatch into C.
        if _capture_may_be_active():
            from .graph import capture_call as _capture_call

            captured = _capture_call(self, args, kwargs)
            if captured is not None:
                return captured
        args = _align_eager_dtypes(self._opname, args)
        mode = _symbolic_call_mode.get()
        if mode is not None:
            return mode.call_with_symbolic_args(self, args, kwargs)
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
        return f"<OpOverload(op='{self._schema.namespace}.{self._opname}', overload='{self._overloadname}')>"

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
            raise AttributeError(f"'{self._qualified_op_name}' has no overload named '{name}'") from None

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
        raise TypeError(f"no overload of {self._qualified_op_name} accepts these arguments:\n  " + "\n  ".join(errors))

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
    for key, schema_text, _names, _npos, tags in entries() if entries else ():
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
        if self.ns == "higher_order":
            from tensorplay._higher_order_ops import __getattr__ as get_hop

            try:
                hop = get_hop(opname)
            except AttributeError:
                hop = None
            if hop is not None:
                setattr(self, opname, hop)
                return hop
        if self.ns == NATIVE_NAMESPACE:
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
            f'tensorplay.library.custom_op("{full_name}") or load its '
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
