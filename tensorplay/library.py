"""Custom operator registration and compiler integration.

Four public layers are provided:

1. :func:`custom_op` / the :class:`Library` class register opaque custom
   operators with device-specific kernels, optional fake (meta) kernels,
   an autograd formula (:meth:`CustomOpDef.register_autograd`), a vector-
   map formula (:meth:`CustomOpDef.register_vmap`) and an autocast cast
   rule (:meth:`CustomOpDef.register_autocast`).
2. :func:`triton_op` + :func:`wrap_triton` and :func:`tile_lang_op` +
   :func:`wrap_tilelang` integrate user-written Triton / TileLang kernels:
   such operators behave like any other custom operator in eager mode and
   are captured as one opaque graph node by ``tensorplay.compile`` —
   preserving the fusion boundary through compilation.
3. Registered operators compose with every compiler backend: during capture
   a call whose arguments are symbolic records a single ``call_function``
   node targeting the :class:`CustomOpDef`; backends that cannot lower it
   treat the node as a barrier and fall back to the interpreter, which
   dispatches to the registered kernel.
   :func:`define`/:func:`impl`/:func:`impl_abstract`, :func:`infer_schema`,
   :func:`get_kernel`, :meth:`CustomOpDef.set_kernel_enabled` and the
   validation harness :func:`opcheck`.

``"namespace::name"`` string.  Schemas are not modeled by a C++ dispatcher;
the optional ``schema=`` strings and :func:`infer_schema` output attach to
the operator for introspection, documentation and ``opcheck``.

``OpOverload``/``overload``/``deprecated``/``fallthrough_kernel``/
``NAMELESS_SCHEMA`` and ``get_ctx`` are absent, ``Library.fallback`` raises
``NotImplementedError``, and re-registering a kernel replaces the previous
one instead of raising (hot swaps for interactive sessions).
"""

from __future__ import annotations

import contextlib
import inspect
import threading
import typing
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Iterable

import tensorplay
from .graph import (
    GraphCaptureError,
    capture_call as _capture_call,
    capturing as _capturing,
    _iter_proxies as _walk_proxies,
)

__all__ = [
    "Library",
    "CustomOpDef",
    "custom_op",
    "triton_op",
    "tile_lang_op",
    "wrap_triton",
    "wrap_tilelang",
    "register_kernel",
    "register_fake",
    "register_autograd",
    "register_vmap",
    "register_autocast",
    "define",
    "impl",
    "impl_abstract",
    "infer_schema",
    "opcheck",
    "get_kernel",
    "get_op",
    "has_op",
]


_LOCK = threading.RLock()
# "ns::op" -> CustomOpDef; namespaces that already own a DEF Library.
_OP_REGISTRY: dict[str, "CustomOpDef"] = {}
_DEFINED_LIBRARY_NAMESPACES: set[str] = set()

# Hot-path aliases resolved once at import (this module is imported last by
# tensorplay/__init__, so every attribute below already exists).
_is_grad_enabled = tensorplay.is_grad_enabled

_COMPOSITE_KEYS = frozenset(
    {"CompositeExplicitAutograd", "CompositeImplicitAutograd"}
)
_COMPOSITE_KEYS_LOWERED = frozenset(key.lower() for key in _COMPOSITE_KEYS)

_UNSET = object()


def _contains_proxy(*values: Any) -> bool:
    for _proxy in _walk_proxies(values):
        return True
    return False


def _validate_name(name: str) -> tuple[str, str]:
    if not isinstance(name, str):
        raise TypeError(f"op name must be a str, got {type(name)!r}")
    parts = name.split("::")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f'expected a qualified op name like "mylib::my_op", got {name!r}'
        )
    namespace, opname = parts
    if not all(part.isidentifier() for part in (namespace, opname)):
        raise ValueError(
            f"namespace and op name must be identifiers, got {name!r}"
        )
    return namespace, opname


def _normalize_device_types(device_types: Any) -> list[str] | None:
    """Normalize ``device_types`` into registry keys.

    ``device_types`` is omitted); strings and iterables of strings map onto
    per-device entries keyed by ``"cpu"``/``"cuda"``/...
    """

    if device_types is None:
        return None
    items = [device_types] if isinstance(device_types, str) else list(device_types)
    keys: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(
                f"device types must be strings, got {item!r}; use None to "
                "declare one device-agnostic implementation"
            )
        key = item.lower()
        if key not in keys:
            keys.append(key)
    return keys


def _bridge_slot_key(device_types: Any) -> Any:
    """

    ``None``/empty iterables/composite spellings select the device-agnostic
    slot (``None`` key); concrete devices lowercase into their own slot.
    Used by the top-level registrations and ``Library.impl``.
    """

    if device_types is None:
        return None
    if isinstance(device_types, str):
        return None if device_type_is_composite(device_types) else device_types.lower()
    keys = list(device_types)
    if not keys:
        return None
    if len(keys) == 1:
        return _bridge_slot_key(keys[0])
    return [_bridge_slot_key(k) for k in keys]


def device_type_is_composite(device_type: str) -> bool:
    return device_type in _COMPOSITE_KEYS


def _validate_mutates_args(mutates_args: Any) -> tuple[str, ...]:
    if isinstance(mutates_args, str) or not isinstance(mutates_args, Iterable):
        raise TypeError(
            f"mutates_args must be an iterable of argument names, got "
            f"{mutates_args!r}"
        )
    mutated = tuple(mutates_args)
    if any(not isinstance(item, str) for item in mutated):
        raise TypeError(
            f"mutates_args entries must be strings, got {mutated!r}"
        )
    return mutated


def _validate_schema(schema: Any) -> str | None:
    """

    TensorPlay keeps them verbatim for introspection/opcheck, so only the
    qualified-name head is checked.
    """

    if schema is None:
        return None
    if not isinstance(schema, str):
        raise TypeError(f"schema must be a str or None, got {type(schema)!r}")
    signature = schema.split("(", 1)[0].strip()
    _validate_name(signature)
    return schema


class CustomOpDef:
    """

    Instances are callable.  Calling with symbolic (tracer proxy) arguments
    records one opaque graph node; calling with real tensors dispatches to
    the kernel registered for the first tensor argument's device, wrapped
    in the registered autograd formula when gradients are requested.
    """

    def __init__(
        self,
        name: str,
        *,
        mutates_args: Sequence[str] = (),
        device_types: Any = None,
        schema: str | None = None,
        is_triton_op: bool = False,
        is_tile_lang_op: bool = False,
    ) -> None:
        self._namespace, self._opname = _validate_name(name)
        self._name = f"{self._namespace}::{self._opname}"
        self._mutates_args = frozenset(_validate_mutates_args(mutates_args))
        self._device_keys = _normalize_device_types(device_types)
        self._schema = _validate_schema(schema)
        self._is_triton_op = bool(is_triton_op)
        self._is_tile_lang_op = bool(is_tile_lang_op)
        # ``None`` key holds the device-agnostic kernel (device_types=None).
        self._kernels: dict[str | None, Callable[..., Any]] = {}
        self._disabled_kernels: set[str] = set()
        self._fake_fn: Callable[..., Any] | None = None
        self._backward: Callable[..., Any] | None = None
        self._setup_context: Callable[..., Any] | None = None
        self._vmap_fn: Callable[..., Any] | None = None
        self._autocast_rules: dict[str, Any] = {}
        self._autograd_cls: type | None = None

    def _install_default_kernel(self, fn: Callable[..., Any]) -> None:
        """Use ``fn`` as the initial kernel (the ``@custom_op`` body).

        the advertised ``device_types`` — every device when omitted.
        """

        if not callable(fn):
            raise TypeError(f"operator body must be callable, got {type(fn)!r}")
        if self._device_keys is None:
            self._kernels[None] = fn
        else:
            for key in self._device_keys:
                self._kernels[key] = fn
        self._mirror_native(self._device_keys, fn)

    # -- introspection -----------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def opname(self) -> str:
        return self._opname

    @property
    def mutates_args(self) -> frozenset[str]:
        return self._mutates_args

    @property
    def is_triton_op(self) -> bool:
        return self._is_triton_op

    @property
    def is_tile_lang_op(self) -> bool:
        return self._is_tile_lang_op

    @property
    def schema(self) -> str | None:
        return self._schema

    # Readable node names once this object becomes a graph target
    # (the graph target formatter resolves ``target.__name__`` first).
    @property
    def __name__(self) -> str:  # type: ignore[override]
        return self._name

    def __repr__(self) -> str:
        kind = (
            "triton_op"
            if self._is_triton_op
            else "tile_lang_op"
            if self._is_tile_lang_op
            else "custom_op"
        )
        return f"<{kind} {self._name}>"

    # -- registration API --------------------------------------------------

    def register_kernel(
        self, device_types: Any = None, fn: Callable[..., Any] | None = None, /
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register the implementation for one or more device types.

        ``device_types=None`` registers a single device-agnostic kernel;
        otherwise pass a device string (``"cpu"``, ``"cuda"``, ...) or an
        iterable of them.  Usable directly
        (``op.register_kernel("cpu", my_fn)``) or as a decorator
        (``@op.register_kernel("cpu")``).  Re-registering the same device
        allows hot swaps for interactive sessions and tests).

        CPU/CUDA kernels are additionally mirrored into the native p10
        dispatcher under this operator's qualified name, so native code and
        :meth:`run_native` can invoke them through the real dispatch path.
        """

        keys = _normalize_device_types(device_types)

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(f):
                raise TypeError(f"kernel must be callable, got {type(f)!r}")
            with _LOCK:
                if keys is None:
                    self._kernels[None] = f
                else:
                    for key in keys:
                        self._kernels[key] = f
            self._mirror_native(keys, f)
            return f

        return decorator(fn) if fn is not None else decorator

    def _mirror_native(
        self, keys: list[str] | None, fn: Callable[..., Any]
    ) -> None:
        """Push a kernel into the native dispatcher bridge (best effort).

        Tensor-in/tensor-out kernels on CPU/CUDA become callable from native
        code via ``Dispatcher::findHandle("ns::op")``; other signatures keep
        working through the Python dispatch path only.
        """

        bridge = getattr(tensorplay._C, "_register_python_op_kernel", None)
        if bridge is None:
            return
        slots = ["default"] if keys is None else [k for k in keys if k in ("cpu", "cuda")]
        for slot in slots:
            try:
                bridge(self._name, slot, fn)
            except Exception:  # noqa: BLE001 - this is an optimization
                return

    def register_fake(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a meta/fake kernel computing output metadata.

        The fake kernel receives the same arguments but must not allocate
        real tensor data; it returns tensors describing shape/dtype/device
        (``tensorplay.empty_like`` style factories without data).  It is
        exercised by :func:`opcheck`'s ``test_faketensor``; capturing
        compilers never execute either kernel version during tracing.
        """

        if not callable(fn):
            raise TypeError(f"fake kernel must be callable, got {type(fn)!r}")
        self._fake_fn = fn
        return fn

    def register_autograd(
        self,
        backward: Callable[..., Any],
        /,
        *,
        setup_context: Callable[..., Any] | None = None,
    ) -> None:
        """

        ``backward(ctx, *grad_outputs)`` receives the saved context.  The
        ``setup_context`` callback may save tensors via
        ``ctx.save_for_backward``.  When no
        ``setup_context`` is given the context stays empty, so backward must
        derive its result purely from ``grad_outputs`` (or close over module
        state).
        """

        if not callable(backward):
            raise TypeError(f"backward must be callable, got {type(backward)!r}")
        if setup_context is not None and not callable(setup_context):
            raise TypeError(
                f"setup_context must be callable or None, got {setup_context!r}"
            )
        self._backward = backward
        self._setup_context = setup_context
        self._autograd_cls = self._build_autograd_class()

    def register_vmap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """

        The engine does not batch yet; the registration is stored and surfaced
        through :func:`get_kernel`-style introspection.
        """

        if not callable(fn):
            raise TypeError(f"vmap function must be callable, got {type(fn)!r}")
        self._vmap_fn = fn
        # Rebuild so the generated autograd class exposes the formula the
        # same way a hand-written Function subclass would.
        if self._backward is not None:
            self._autograd_cls = self._build_autograd_class()
        return fn

    def register_autocast(
        self, device_type: str, cast_inputs: Any
    ) -> None:
        """

        When autocast is enabled for ``device_type``, floating-point tensor
        arguments are cast to ``cast_inputs`` before any kernel runs — the
        """

        if not isinstance(device_type, str):
            raise TypeError(
                f"device_type must be a str, got {type(device_type)!r}"
            )
        if not (
            hasattr(cast_inputs, "is_floating_point")
            and cast_inputs.is_floating_point
        ):
            raise TypeError(
                f"cast_inputs must be a floating-point dtype, got "
                f"{cast_inputs!r}"
            )
        self._autocast_rules[device_type.lower()] = cast_inputs

    @contextlib.contextmanager
    def set_kernel_enabled(self, device_type: str, enabled: bool = True):
        """

        context is active the concrete kernel for ``device_type`` is skipped
        and dispatch falls back to the device-agnostic kernel (if any).
        Disabling an already-disabled (or enabling an already-enabled)
        kernel warns and is otherwise a no-op; the original state is always
        restored on exit.
        """

        if isinstance(device_type, str):
            key = device_type.lower()
        else:
            key = device_type
        originally_disabled = key in self._disabled_kernels
        has_own_kernel = key in self._kernels
        action = "enable" if enabled else "disable"
        if not has_own_kernel and None not in self._kernels:
            warnings.warn(
                f"Attempted to {action} kernel for {key!r} but no kernel was "
                "registered for this device type.",
                stacklevel=2,
            )
        if not enabled:
            if originally_disabled:
                warnings.warn(
                    f"Attempted to disable kernel for {key!r} but it was "
                    "already disabled.",
                    stacklevel=2,
                )
            else:
                self._disabled_kernels.add(key)
        else:  # enable the kernel
            if not originally_disabled:
                warnings.warn(
                    f"Attempted to enable kernel for {key!r} but it was "
                    "already enabled.",
                    stacklevel=2,
                )
            else:
                self._disabled_kernels.remove(key)
        try:
            yield
        finally:
            # restore original state
            if originally_disabled:
                self._disabled_kernels.add(key)
            else:
                self._disabled_kernels.discard(key)

    def _build_autograd_class(self) -> type:
        op_def = self

        class _CustomOpAutograd(tensorplay.autograd.Function):
            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                return op_def._run_kernel(args, kwargs)

            @staticmethod
            def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
                if op_def._setup_context is not None:
                    op_def._setup_context(ctx, inputs, output)

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:
                return op_def._backward(ctx, *grad_outputs)

        if op_def._vmap_fn is not None:
            _vmap_fn = op_def._vmap_fn

            @staticmethod
            def vmap(info: Any, in_dims: Any, *args: Any) -> Any:
                return _vmap_fn(info, in_dims, *args)

            _CustomOpAutograd.vmap = staticmethod(vmap)  # type: ignore[method-assign]

        _CustomOpAutograd.__qualname__ = f"_CustomOpAutograd[{self._name}]"
        return _CustomOpAutograd

    # -- dispatch ----------------------------------------------------------

    def _kernel_for(
        self, args: tuple[Any, ...], key: Any = _UNSET
    ) -> Callable[..., Any]:
        # Lock-free read path: dict/set membership tests are atomic under
        # the GIL and registration only ever swaps callables wholesale.
        if key is _UNSET:
            key = _first_device_key(args)
        kernels = self._kernels
        if key is not None:
            fn = kernels.get(key)
            if fn is not None and key not in self._disabled_kernels:
                return fn
            # Disabled/shadowed concrete kernels fall back to the composite
        fn = kernels.get(None)
        if fn is not None:
            return fn
        with _LOCK:  # cold error path only
            registered = sorted(str(item) for item in kernels)
        where = f" for device {key!r}" if key is not None else ""
        hint = f"; registered devices: {registered}" if registered else ""
        raise NotImplementedError(
            f"{self._name} has no kernel{where}{hint}. Register one via "
            f"{self._name}.register_kernel(...)"
        )

    def _run_kernel(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return self._kernel_for(args)(*args, **kwargs)

    def run_native(
        self, inputs: Sequence[Any], *, device_type: str | None = None
    ) -> list[tensorplay.Tensor]:
        """Invoke this operator through the native p10 dispatcher.

        Exercises ``Dispatcher::findHandle`` plus the real kernel table —
        the same path C++ code takes when calling the operator by name.
        Requires a tensor-only signature (the canonical unboxed convention
        is tensors-in/tensors-out); kernels that were never mirrored (e.g.
        non-tensor arguments) raise ``NotImplementedError``.
        """

        bridge = getattr(tensorplay._C, "_call_native_op", None)
        if bridge is None:
            raise RuntimeError(
                "the native custom-op bridge is unavailable in this build"
            )
        return bridge(self._name, list(inputs), device_type)

    def _eager_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Dispatch real tensors with full eager semantics (no capture).

        Shared by :meth:`__call__` and the native re-entry below so compiled
        graphs keep device dispatch AND ``register_autograd`` behavior.
        Hot path: one device-key scan, one dict lookup, one call.
        """

        key = _first_device_key(args)
        rule = self._autocast_rules.get(key) if key is not None else None
        if rule is not None and _autocast_enabled(key):
            args = tuple(_cast_if_floating(v, rule) for v in args)
            if kwargs:
                kwargs = {
                    k: _cast_if_floating(v, rule) for k, v in kwargs.items()
                }
        if self._autograd_cls is not None and _is_grad_enabled():
            return self._autograd_cls.apply(*args, **kwargs)
        return self._kernel_for(args, key)(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # A Proxy can only exist while a Tracer.trace() is live on this
        # thread (the graph tracing depth), so the proxy walk is skipped
        # entirely outside capture — identical to the generated functional
        # wrappers' hot path in tensorplay/functional.py.
        if _capturing():
            captured = _capture_call(self, args, kwargs)
            if captured is not None:
                return captured
        return self._eager_call(args, kwargs)


def _first_device_key(values: tuple[Any, ...]) -> str | None:
    for value in values:
        if isinstance(value, tensorplay.Tensor):
            device = value.device
            device_type = getattr(device, "type", None)
            if isinstance(device_type, str):
                return device_type
            return "cuda" if device.is_cuda() else "cpu"
    return None


def _autocast_enabled(device_key: str) -> bool:
    fn = getattr(tensorplay, "is_autocast_enabled", None)
    if fn is None:
        return False
    try:
        return bool(fn(device_key))
    except Exception:  # noqa: BLE001 - autocast is best effort
        return False


def _cast_if_floating(value: Any, dtype: Any) -> Any:
    if isinstance(value, tensorplay.Tensor):
        value_dtype = value.dtype
        if value_dtype.is_floating_point and value_dtype != dtype:
            return value.to(dtype)
    return value


def get_op(op_name: str) -> CustomOpDef:
    """Return the previously defined operator named ``"ns::op"``."""

    if not isinstance(op_name, str):
        raise TypeError(f"op name must be a str, got {type(op_name)!r}")
    with _LOCK:
        op_def = _OP_REGISTRY.get(op_name)
    if op_def is None:
        raise RuntimeError(f"unknown operator {op_name!r}")
    return op_def


def has_op(op_name: str) -> bool:
    with _LOCK:
        return op_name in _OP_REGISTRY


def _resolve_op(op: str | CustomOpDef) -> CustomOpDef:
    if isinstance(op, CustomOpDef):
        return op
    if isinstance(op, str):
        return get_op(op)
    raise TypeError(
        f"expected an operator name or CustomOpDef, got {type(op)!r}"
    )


def _register_op_def(
    op_def: CustomOpDef,
) -> CustomOpDef:
    with _LOCK:
        previous = _OP_REGISTRY.get(op_def.name)
        if previous is not None:
            raise RuntimeError(
                f"operator {op_def.name!r} is already defined; use "
                f"get_op({op_def.name!r}) or a fresh namespace"
            )
        _OP_REGISTRY[op_def.name] = op_def
    return op_def


def _define_or_get(qualname: str, schema: str | None) -> CustomOpDef:
    namespace, _opname = _validate_name(qualname)
    del namespace
    _validate_schema(schema)
    with _LOCK:
        op_def = _OP_REGISTRY.get(qualname)
    if op_def is None:
        op_def = _register_op_def(CustomOpDef(qualname))
    elif schema is not None:
        op_def._schema = _validate_schema(schema)
    return op_def


def custom_op(
    name: str,
    fn: Callable[..., Any] | None = None,
    /,
    *,
    mutates_args: Sequence[str] = (),
    device_types: Any = None,
    schema: str | None = None,
) -> Callable[[Callable[..., Any]], CustomOpDef]:
    """

    Example::

        @tensorplay.library.custom_op("mylib::weighted_sum", mutates_args=())
        def weighted_sum(x, weight):
            return (x * weight).sum()

        # Optional extra kernels per device:
        @weighted_sum.register_kernel("cuda")
        def _(x, weight): ...

    ``fn`` may also be passed positionally
    (``custom_op("mylib::op", my_fn, mutates_args=())``), matching
    operator's default kernel for the advertised ``device_types`` (every
    device when omitted).

    Args:
        name: Qualified ``"namespace::name"`` identifier.
        fn: Operator body; omit to use the return value as a decorator.
        mutates_args: Names of arguments the kernel mutates in place.
            Compile-time fusion treats these as barriers regardless of the
            value; eager execution trusts the declaration.
        device_types: Restriction advertised to users at definition time.
            Kernels are selected per call from whatever was registered via
            :meth:`CustomOpDef.register_kernel`.
        schema: Optional schema string kept for introspection and
            :func:`opcheck` (TensorPlay models no schema grammar).

    Returns:
        A decorator producing a callable :class:`CustomOpDef`.
    """

    _validate_name(name)
    _validate_mutates_args(mutates_args)
    _normalize_device_types(device_types)
    _validate_schema(schema)

    def decorator(f: Callable[..., Any]) -> CustomOpDef:
        op_def = CustomOpDef(
            name,
            mutates_args=mutates_args,
            device_types=device_types,
            schema=schema,
        )
        _register_op_def(op_def)
        op_def._install_default_kernel(f)
        return op_def

    return decorator(fn) if fn is not None else decorator


def triton_op(
    name: str,
    fn: Callable[..., Any] | None = None,
    /,
    *,
    mutates_args: Sequence[str] = (),
    device_types: Any = None,
    schema: str | None = None,
) -> Callable[[Callable[..., Any]], CustomOpDef]:
    """

    The registered kernel(s) must launch their Triton kernels through
    :func:`wrap_triton` and only mutate arguments listed in
    ``mutates_args``.  Under ``tensorplay.compile`` the whole operator is
    captured as a single opaque node — the compiler never traces into the
    Triton launches.  ``device_types`` is a
    """

    _validate_name(name)
    _validate_mutates_args(mutates_args)
    _normalize_device_types(device_types)
    _validate_schema(schema)

    def decorator(f: Callable[..., Any]) -> CustomOpDef:
        op_def = CustomOpDef(
            name,
            mutates_args=mutates_args,
            device_types=device_types,
            schema=schema,
            is_triton_op=True,
        )
        _register_op_def(op_def)
        op_def._install_default_kernel(f)
        return op_def

    return decorator(fn) if fn is not None else decorator


def tile_lang_op(
    name: str,
    fn: Callable[..., Any] | None = None,
    /,
    *,
    mutates_args: Sequence[str] = (),
    device_types: Any = None,
    schema: str | None = None,
) -> Callable[[Callable[..., Any]], CustomOpDef]:
    """Define a TileLang-backed operator (Triton contract, tile-lang kernels).

    TileLang (https://github.com/tile-ai/tilelang) compiles ``@T.prim_func``
    DSL programs into highly-tuned CUDA/Metal/CPU kernels; grid and thread
    configuration live inside the prim func's ``with T.Kernel(...)`` block,
    so unlike Triton there is no grid indexing at the launch site — a
    compiled ``JITKernel`` is called directly with tensors.

    The registered kernel body must launch its TileLang kernels through
    :func:`wrap_tilelang`.  Under ``tensorplay.compile`` the operator is
    captured as a single opaque fusion-barrier node whose body never runs
    during tracing, exactly like :func:`triton_op`.
    """

    _validate_name(name)
    _validate_mutates_args(mutates_args)
    _normalize_device_types(device_types)
    _validate_schema(schema)

    def decorator(f: Callable[..., Any]) -> CustomOpDef:
        op_def = CustomOpDef(
            name,
            mutates_args=mutates_args,
            device_types=device_types,
            schema=schema,
            is_tile_lang_op=True,
        )
        _register_op_def(op_def)
        op_def._install_default_kernel(f)
        return op_def

    return decorator(fn) if fn is not None else decorator


def _native_invoke(op_name: str, *tensors: Any) -> Any:
    """Re-entry point for compiled native graphs (Stax ``custom_op`` nodes).

    The C++ executor installed by the bindings calls this with the
    operator's qualified name and its tensor inputs; routing through the
    :class:`CustomOpDef` keeps device dispatch and autograd identical to
    eager execution instead of bypassing them with a raw kernel.
    """

    return get_op(op_name)._eager_call(tuple(tensors), {})


class TritonKernelWrapper:
    """Grid-indexable passthrough around a ``@triton.jit`` kernel.

    At eager time ``wrapped[grid](...)`` simply launches the kernel.  If a
    launch is ever captured symbolically (proxy arguments), it raises:
    Triton launches are never part of the canonical graph — they live
    inside :func:`triton_op` bodies, which the compiler captures as one
    opaque node instead.
    """

    __slots__ = ("kernel",)

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        kernel = self.kernel

        def launcher(*args: Any, **kwargs: Any) -> Any:
            if _contains_proxy(grid, args, kwargs):
                raise GraphCaptureError(
                    "a raw Triton launch cannot be captured inside "
                    "tensorplay.compile; define the launch inside "
                    "tensorplay.library.triton_op and call that operator "
                    "instead"
                )
            return kernel[grid](*args, **kwargs)

        launcher.__name__ = getattr(kernel, "__name__", "triton_kernel")
        return launcher


def wrap_triton(kernel: Any) -> TritonKernelWrapper:
    """Mark a Triton kernel as launchable from within a ``triton_op``.

    Accepts a ``triton.runtime.jit.JITFunction`` (the ``@triton.jit``
    result) or any grid-indexable launcher; idempotent on wrappers.
    """

    if isinstance(kernel, TritonKernelWrapper):
        return kernel
    is_jit_function = False
    try:
        from triton.runtime.jit import JITFunction

        is_jit_function = isinstance(kernel, JITFunction)
    except ImportError:
        is_jit_function = False
    if not is_jit_function and not hasattr(kernel, "__getitem__"):
        raise TypeError(
            "wrap_triton expects a @triton.jit kernel (JITFunction) or a "
            f"grid-indexable launcher, got {type(kernel)!r}"
        )
    return TritonKernelWrapper(kernel)


class TileLangKernelWrapper:
    """Passthrough around a compiled TileLang kernel.

    TileLang launch sites carry no grid (``with T.Kernel(...)`` lives in
    the prim func), so unlike :class:`TritonKernelWrapper` this wraps a
    directly-callable object: a ``tilelang.jit.kernel.JITKernel``, the
    ``JITImpl`` produced by ``@tilelang.jit`` (lazy mode: calling the
    factory with shape/constexpr arguments yields the compiled kernel),
    or any duck-typed adapter.  Eager calls forward untouched; a symbolic
    (proxy) launch raises, steering raw launches behind a
    :func:`tile_lang_op` boundary.
    """

    __slots__ = ("kernel",)

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel

    def compile(self, *args: Any, **kwargs: Any) -> "TileLangKernelWrapper":
        """Bind a lazy-mode ``@tilelang.jit`` factory into a ready kernel."""

        bound = self.kernel(*args, **kwargs)
        return TileLangKernelWrapper(bound)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _contains_proxy(args, kwargs):
            raise GraphCaptureError(
                "a raw TileLang launch cannot be captured inside "
                "tensorplay.compile; define the launch inside "
                "tensorplay.library.tile_lang_op and call that operator "
                "instead"
            )
        return self.kernel(*args, **kwargs)


def _looks_like_tilelang_kernel(kernel: Any) -> bool:
    """Duck-typing for TileLang objects without importing tilelang.

    Recognizes ``tilelang.jit.kernel.JITKernel`` (compiled, exposes
    factory, exposes ``get_tir``/``out_idx``) and plain adapters that opt
    in via a ``_tilelang_kernel`` marker attribute.
    """

    if hasattr(kernel, "_tilelang_kernel"):
        return bool(getattr(kernel, "_tilelang_kernel"))
    return any(
        hasattr(kernel, attr) for attr in ("adapter", "torch_function", "get_tir")
    )


def wrap_tilelang(kernel: Any) -> TileLangKernelWrapper:
    """Mark a TileLang kernel as launchable from within a ``tile_lang_op``.

    Accepts a compiled ``tilelang.jit.kernel.JITKernel``, the lazy-mode
    ``JITImpl`` factory returned by ``@tilelang.jit``, or any duck-typed
    callables are rejected so typos surface early.  Idempotent on
    wrappers.
    """

    if isinstance(kernel, TileLangKernelWrapper):
        return kernel
    recognized = False
    try:
        from tilelang.jit.kernel import JITKernel as _JITKernel

        recognized = isinstance(kernel, _JITKernel)
    except ImportError:
        recognized = False
    if not recognized:
        try:
            from tilelang.jit import JITImpl as _JITImpl

            recognized = isinstance(kernel, _JITImpl)
        except ImportError:
            recognized = False
    if not recognized:
        recognized = _looks_like_tilelang_kernel(kernel)
    if not recognized and callable(kernel):
        raise TypeError(
            "wrap_tilelang expects a tilelang JITKernel/JITImpl (see "
            "https://github.com/tile-ai/tilelang) or a duck-typed adapter "
            f"{type(kernel)!r}; plain callables are rejected — set "
            f"'_tilelang_kernel = True' on custom launchers to opt in"
        )
    if not callable(kernel):
        raise TypeError(
            "wrap_tilelang expects a callable kernel, got "
            f"{type(kernel)!r}"
        )
    return TileLangKernelWrapper(kernel)


def register_kernel(
    op: str | CustomOpDef,
    device_types: Any = None,
    func: Callable[..., Any] | None = None,
    /,
    *,
    lib: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """

    Accepts a :class:`CustomOpDef` or a qualified operator name.
    ``device_types=None`` or an empty iterable means the device-agnostic
    slot; composite spellings (``Composite…``) map there too.  Usable
    directly or as a decorator.
    """

    del lib
    op_def = _resolve_op(op)
    key: Any = _bridge_slot_key(device_types)
    return op_def.register_kernel(key) if func is None else op_def.register_kernel(key)(func)


def register_fake(
    op: str | CustomOpDef,
    func: Callable[..., Any] | None = None,
    /,
    *,
    lib: Any = None,
    allow_override: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:

    del lib, allow_override
    op_def = _resolve_op(op)
    return op_def.register_fake(func) if func is not None else op_def.register_fake


def register_autograd(
    op: str | CustomOpDef,
    backward: Callable[..., Any],
    /,
    *,
    setup_context: Callable[..., Any] | None = None,
    lib: Any = None,
) -> None:

    del lib
    op_def = _resolve_op(op)
    op_def.register_autograd(backward, setup_context=setup_context)


def register_vmap(
    op: str | CustomOpDef,
    func: Callable[..., Any] | None = None,
    /,
    *,
    lib: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:

    del lib
    op_def = _resolve_op(op)
    return op_def.register_vmap(func) if func is not None else op_def.register_vmap


def register_autocast(
    op: str | CustomOpDef,
    device_type: str,
    cast_inputs: Any,
    /,
    *,
    lib: Any = None,
) -> None:

    del lib
    _resolve_op(op).register_autocast(device_type, cast_inputs)


def define(
    qualname: str, schema: str | None = None, *, lib: Any = None, tags: Any = ()
) -> None:
    """

    Creates the :class:`CustomOpDef` if absent (kernels are then attached
    with :func:`impl` or ``Library("ns", "IMPL").impl``).  ``tags`` is
    Like ``Library.define``, a full ``"ns::op(Tensor) -> Tensor"`` string
    may be pasted as ``qualname``.
    """

    del lib, tags
    if isinstance(qualname, str) and "(" in qualname:
        if schema is None:
            schema = qualname
        qualname = qualname.split("(", 1)[0].strip()
    _define_or_get(qualname, schema)


def impl(
    qualname: str,
    types: Any,
    func: Callable[..., Any] | None = None,
    /,
    *,
    lib: Any = None,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """

    ``types`` accepts concrete devices (``"CPU"``/``"CUDA"``) or composite
    spellings (``CompositeExplicitAutograd`` → the device-agnostic slot).
    """

    del lib
    return register_kernel(qualname, types, func)


def impl_abstract(
    qualname: str,
    func: Callable[..., Any] | None = None,
    /,
    *,
    lib: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:

    return register_fake(qualname, func, lib=lib)


def get_kernel(
    op: str | CustomOpDef, dispatch_key: str
) -> Callable[..., Any]:
    """

    ``dispatch_key`` accepts ``"cpu"``/``"cuda"``/``"default"`` and the
    composite spellings.  Raises ``LookupError`` when nothing usable is
    registered (a disabled concrete kernel counts as absent, matching
    :meth:`CustomOpDef.set_kernel_enabled` visibility).
    """

    op_def = _resolve_op(op)
    if not isinstance(dispatch_key, str):
        raise TypeError(
            f"dispatch_key must be a str, got {type(dispatch_key)!r}"
        )
    lowered = dispatch_key.lower()
    key: Any = (
        None
        if lowered in _COMPOSITE_KEYS_LOWERED or lowered in ("default", "composite")
        else lowered
    )
    # does NOT satisfy a concrete-device query (use "default" for that).
    fn = op_def._kernels.get(key)
    if fn is not None and (
        key is None or key not in op_def._disabled_kernels
    ):
        return fn
    raise LookupError(
        f"No kernel registered for {op_def.name} with dispatch key "
        f"{dispatch_key!r}"
    )


# ---------------------------------------------------------------------------
# infer_schema
# ---------------------------------------------------------------------------

_SCHEMA_PRIMITIVES: dict[Any, str] = {int: "SymInt", float: "float", bool: "bool", str: "str"}


def _annotation_to_schema_str(annotation: Any) -> str:
    """Best-effort translation of Python annotations to schema atoms.

    Unannotated parameters default to ``Tensor`` (the overwhelming custom-op
    case); unsupported annotations fall back to ``Tensor`` rather than
    failing, since TensorPlay enforces no grammar.
    """

    if annotation is None or annotation is type(None):
        return "Tensor"
    if annotation is tensorplay.Tensor or annotation == "Tensor":
        return "Tensor"
    if annotation in _SCHEMA_PRIMITIVES:
        return _SCHEMA_PRIMITIVES[annotation]
    origin = typing.get_origin(annotation)
    if origin is typing.Union or (
        hasattr(typing, "UnionType") and origin is typing.UnionType
    ):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _annotation_to_schema_str(args[0]) + "?"
        return "(" + ", ".join(_annotation_to_schema_str(a) for a in args) + ")"
    if origin in (list, Sequence, tuple):
        args = typing.get_args(annotation)
        if origin is tuple and len(args) >= 2:
            inner = ", ".join(_annotation_to_schema_str(a) for a in args)
            return f"({inner})"
        atom = _annotation_to_schema_str(args[0] if args else tensorplay.Tensor)
        return f"{atom}[]"
    text = str(annotation)
    if "Tensor" in text:
        return "Tensor"
    return "Tensor"


def infer_schema(
    prototype_function: Callable[..., Any],
    /,
    *,
    mutates_args: Sequence[str],
    op_name: str | None = None,
    tags: Any = (),
) -> str:
    """

    Produces ``"ns::op(Tensor self, SymInt n, Tensor(a!) out) -> Tensor"``
    alias-annotation ``(<type>(<letter>!))`` marker.  Parameters without
    annotations are treated as tensors; ``*args``/``**kwargs`` are skipped.
    """

    del tags
    mutated = frozenset(_validate_mutates_args(mutates_args))
    if not callable(prototype_function):
        raise TypeError(
            f"prototype_function must be callable, got "
            f"{type(prototype_function)!r}"
        )
    try:
        hints = typing.get_type_hints(prototype_function)
    except Exception:  # noqa: BLE001 - unresolvable hints fall back to raw
        hints = dict(getattr(prototype_function, "__annotations__", {}) or {})
    parameters = inspect.signature(prototype_function).parameters
    alias_letters = iter("abcdefghijklmnopqrstuvwxyz")
    parts: list[str] = []
    for pname, parameter in parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        atom = _annotation_to_schema_str(hints.get(pname))
        if pname in mutated:
            atom = f"{atom}({next(alias_letters)}!)"
        parts.append(atom)
    return_atom = _annotation_to_schema_str(hints.get("return"))
    name = op_name if op_name is not None else getattr(
        prototype_function, "__name__", "op"
    )
    _validate_name(name) if "::" in name else None
    return f"{name}({', '.join(parts)}) -> {return_atom}"


# ---------------------------------------------------------------------------
# opcheck
# ---------------------------------------------------------------------------

_OPCHECK_DEFAULT_UTILS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def _flatten_tensors(value: Any) -> list[tensorplay.Tensor]:
    if isinstance(value, tensorplay.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        flat: list[tensorplay.Tensor] = []
        for item in value:
            flat.extend(_flatten_tensors(item))
        return flat
    if isinstance(value, dict):
        flat = []
        for item in value.values():
            flat.extend(_flatten_tensors(item))
        return flat
    return []


def _bind_named_tensors(
    op_def: CustomOpDef, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, tensorplay.Tensor]:
    """Map argument names to tensor values via the kernel's signature."""

    kernel = next(iter(op_def._kernels.values()), None)
    named: dict[str, tensorplay.Tensor] = {}
    if kernel is not None:
        try:
            sig = inspect.signature(kernel)
            bound = sig.bind_partial(*args, **kwargs)
        except TypeError:
            bound = None
        if bound is not None:
            for name, value in bound.arguments.items():
                if isinstance(value, tensorplay.Tensor):
                    named[name] = value
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, tensorplay.Tensor):
                            named[name] = item
                            break
            return named
    # Signature-less fallback: positional tensors get synthetic names.
    for index, value in enumerate(args):
        if isinstance(value, tensorplay.Tensor):
            named[f"arg{index}"] = value
    for name, value in kwargs.items():
        if isinstance(value, tensorplay.Tensor):
            named[name] = value
    return named


def _tensors_equal(a: tensorplay.Tensor, b: tensorplay.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    try:
        return bool(tensorplay.equal(a, b))
    except Exception:  # noqa: BLE001 - exotic dtypes: fall back to identity
        return a.data_ptr() == b.data_ptr()


def _opcheck_test_schema(
    op_def: CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    named_before = _bind_named_tensors(op_def, args, kwargs)
    snapshots = {n: t.clone() for n, t in named_before.items()}
    output = op_def(*args, **kwargs)
    # Undeclared mutation is the safety-critical direction: fusion treats
    # undeclared side effects as barriers, so catch silent in-place edits.
    for name, before in snapshots.items():
        if name in op_def.mutates_args:
            continue
        after = named_before[name]
        if not _tensors_equal(before, after):
            raise AssertionError(
                f"{op_def.name}: kernel mutated argument '{name}' which is "
                "not declared in mutates_args; declare it or stop mutating"
            )
    # Outputs must be fresh allocations, never aliases of an input storage
    input_ptrs = {t.data_ptr() for t in named_before.values()}
    for tensor in _flatten_tensors(output):
        if tensor.data_ptr() in input_ptrs:
            raise AssertionError(
                f"{op_def.name}: kernel returned a tensor aliasing an input; "
                "custom ops must return fresh outputs (mutation goes through "
                "mutates_args)"
            )


def _opcheck_test_faketensor(
    op_def: CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    fake_fn = op_def._fake_fn
    if fake_fn is None:
        raise AssertionError(
            f"{op_def.name}: no fake kernel registered; register one with "
            "register_fake so compilers can propagate metadata"
        )
    real_out = _flatten_tensors(op_def(*args, **kwargs))
    fake_out = _flatten_tensors(fake_fn(*args, **kwargs))
    if len(real_out) != len(fake_out):
        raise AssertionError(
            f"{op_def.name}: fake kernel returned {len(fake_out)} tensor(s) "
            f"but the real kernel returned {len(real_out)}"
        )
    for i, (real, fake) in enumerate(zip(real_out, fake_out)):
        if tuple(real.shape) != tuple(fake.shape):
            raise AssertionError(
                f"{op_def.name}: output {i} shape mismatch "
                f"(real {tuple(real.shape)} vs fake {tuple(fake.shape)})"
            )
        if real.dtype != fake.dtype:
            raise AssertionError(
                f"{op_def.name}: output {i} dtype mismatch "
                f"(real {real.dtype} vs fake {fake.dtype})"
            )
        if str(real.device) != str(fake.device):
            raise AssertionError(
                f"{op_def.name}: output {i} device mismatch "
                f"(real {real.device} vs fake {fake.device})"
            )


def _opcheck_test_autograd_registration(
    op_def: CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    kernel = next(iter(op_def._kernels.values()), None)
    if kernel is None:
        return
    try:
        parameters = [
            p
            for p in inspect.signature(kernel).parameters.values()
            if p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if len(parameters) < len(args):
            raise TypeError
    except (TypeError, ValueError):
        return  # cannot map inputs onto the kernel signature; skip probe

    new_args = list(args)
    new_kwargs = dict(kwargs)
    leaves: list[tuple[tensorplay.Tensor, tensorplay.Tensor]] = []
    for index, parameter in enumerate(parameters):
        source = args[index] if index < len(args) else new_kwargs.get(parameter.name)
        if (
            isinstance(source, tensorplay.Tensor)
            and source.dtype.is_floating_point
        ):
            leaf = source.detach().clone().requires_grad_(True)
            if index < len(args):
                new_args[index] = leaf
            else:
                new_kwargs[parameter.name] = leaf
            leaves.append((source, leaf))
    if not leaves:
        return  # nothing differentiable to probe

    outputs = op_def(*new_args, **new_kwargs)
    flat_outputs = _flatten_tensors(outputs)
    if not flat_outputs:
        return
    first = flat_outputs[0]
    if not first.requires_grad:
        raise AssertionError(
            f"{op_def.name}: output does not require grad although a "
            "floating input does; the kernel breaks the autograd graph"
        )
    first.backward(tensorplay.ones_like(first))
    for original, leaf in leaves:
        grad = leaf.grad
        if grad is None:
            raise AssertionError(
                f"{op_def.name}: no gradient reached input; register an "
                "autograd formula via register_autograd"
            )
        if tuple(grad.shape) != tuple(original.shape):
            raise AssertionError(
                f"{op_def.name}: gradient shape {tuple(grad.shape)} does not "
                f"match input shape {tuple(original.shape)}"
            )


def _make_fixed_arity_wrapper(
    fn: Callable[..., Any], nargs: int, kwargs: dict[str, Any]
) -> Callable[..., Any]:
    """Build a fixed-arity ``f(a0, ..., aN)`` shim (the tracer rejects
    varargs), forwarding into ``fn``."""
    if nargs == 0:
        return lambda: fn(**dict(kwargs))
    if nargs == 1:
        return lambda a0: fn(a0, **dict(kwargs))
    if nargs == 2:
        return lambda a0, a1: fn(a0, a1, **dict(kwargs))
    if nargs == 3:
        return lambda a0, a1, a2: fn(a0, a1, a2, **dict(kwargs))
    if nargs == 4:
        return lambda a0, a1, a2, a3: fn(a0, a1, a2, a3, **dict(kwargs))
    if nargs == 5:
        return lambda a0, a1, a2, a3, a4: fn(a0, a1, a2, a3, a4, **dict(kwargs))
    if nargs == 6:
        return lambda a0, a1, a2, a3, a4, a5: fn(
            a0, a1, a2, a3, a4, a5, **dict(kwargs)
        )
    raise ValueError(
        f"opcheck supports up to 6 sample tensor arguments, got {nargs}"
    )


def _opcheck_test_aot_dispatch_dynamic(
    op_def: CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    from tensorplay.graph import Tracer

    tensor_positions = [
        i for i, v in enumerate(args) if isinstance(v, tensorplay.Tensor)
    ]
    if not tensor_positions:
        return  # nothing symbolic could flow; capture adds no information
    limit = min(len(tensor_positions), 6)
    samples = [args[i].clone() for i in tensor_positions[:limit]]
    wrapped = _make_fixed_arity_wrapper(op_def, len(samples), kwargs)
    param_names = [f"a{i}" for i in range(len(samples))]
    traced = Tracer().trace(
        wrapped, sample_inputs=dict(zip(param_names, samples))
    )
    eager_out = _flatten_tensors(op_def(*samples, **kwargs))
    compiled_out = _flatten_tensors(traced(*samples))
    if len(eager_out) != len(compiled_out):
        raise AssertionError(
            f"{op_def.name}: captured graph returned {len(compiled_out)} "
            f"tensor(s), eager returned {len(eager_out)}"
        )
    for i, (want, got) in enumerate(zip(eager_out, compiled_out)):
        if not bool(tensorplay.allclose(want, got)):
            raise AssertionError(
                f"{op_def.name}: compiled output {i} diverges from eager"
            )


_OPCHECK_TESTS = {
    "test_schema": _opcheck_test_schema,
    "test_faketensor": _opcheck_test_faketensor,
    "test_autograd_registration": _opcheck_test_autograd_registration,
    "test_aot_dispatch_dynamic": _opcheck_test_aot_dispatch_dynamic,
}


def opcheck(
    op: str | CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    test_utils: str | Sequence[str] = _OPCHECK_DEFAULT_UTILS,
    raise_exception: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict[str, str]:
    """

    Runs each selected check and reports failures keyed by test name:

    - ``test_schema``: undeclared inputs are left unmutated and no output
      aliases an input storage (declared-mutation direction is trusted,
      matching TensorPlay's declaration-driven fusion barriers).
    - ``test_autograd_registration``: gradients reach every floating input
      with matching shapes.  TensorPlay composes Python kernels implicitly
      (CompositeImplicitAutograd semantics), so a missing explicit formula
      is legal — this check catches kernels that break the autograd graph
      or drop gradients.
    - ``test_faketensor``: the fake kernel reproduces the real outputs'
      metadata.
    - ``test_aot_dispatch_dynamic``: capture + execution reproduce the

    Returns the failure mapping; empty means all checks passed.
    """

    del atol, rtol  # accepted for signature compatibility; comparisons are exact
    op_def = _resolve_op(op)
    kwargs = dict(kwargs or {})
    if isinstance(test_utils, str):
        selected = (
            _OPCHECK_DEFAULT_UTILS if test_utils == "all" else (test_utils,)
        )
    else:
        selected = tuple(test_utils)
    unknown = [t for t in selected if t not in _OPCHECK_TESTS]
    if unknown:
        raise ValueError(
            f"unknown opcheck test_utils {unknown}; expected a subset of "
            f"{sorted(_OPCHECK_TESTS)} or 'all'"
        )
    failures: dict[str, str] = {}
    for name in selected:
        checker = _OPCHECK_TESTS[name]
        try:
            checker(op_def, args, kwargs)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            failures[name] = f"{type(exc).__name__}: {exc}"
    if failures and raise_exception:
        rendered = "\n".join(f"  {k}: {v}" for k, v in failures.items())
        raise RuntimeError(
            f"opcheck({op_def.name}) failed {len(failures)} check(s):\n"
            f"{rendered}"
        )
    return failures


class Library:
    """

    one DEF library per process/namespace), ``"IMPL"`` adds kernels, and
    ``"FRAGMENT"`` extends an existing namespace from multiple locations.
    """

    def __init__(self, namespace: str, kind: str = "DEF") -> None:
        if not isinstance(namespace, str) or not namespace.isidentifier():
            raise ValueError(
                f"library namespace must be an identifier, got {namespace!r}"
            )
        if kind not in {"DEF", "IMPL", "FRAGMENT"}:
            raise ValueError(f"unknown Library kind {kind!r}")
        self.ns = namespace
        self.kind = kind
        self._op_names: list[str] = []
        with _LOCK:
            if kind == "DEF":
                if namespace in _DEFINED_LIBRARY_NAMESPACES:
                    raise RuntimeError(
                        f"only a single DEF Library may exist for namespace "
                        f"{namespace!r}"
                    )
                _DEFINED_LIBRARY_NAMESPACES.add(namespace)

    def _define(self, opname: str) -> CustomOpDef:
        full_name = f"{self.ns}::{opname}"
        with _LOCK:
            existing = _OP_REGISTRY.get(full_name)
        if existing is not None:
            if self.kind != "FRAGMENT" and existing.namespace == self.ns:
                raise RuntimeError(
                    f"operator {full_name!r} already defined in namespace "
                    f"{self.ns!r}"
                )
            return existing
        return _register_op_def(CustomOpDef(full_name))

    def define(
        self, schema: str, *, alias_analysis: str = "", tags: Any = ()
    ) -> None:
        """Define an operator from a schema like ``"ns::add(Tensor, Tensor)"``.

        Only the qualified name is meaningful (TensorPlay models no schema
        schema strings can be pasted verbatim.  ``alias_analysis`` and
        ``tags`` are accepted and ignored for call-site compatibility.
        """

        del alias_analysis, tags
        if not isinstance(schema, str) or "::" not in schema:
            raise ValueError(
                f'schema must look like "ns::op(...)", got {schema!r}'
            )
        signature = schema.split("(", 1)[0].strip()
        namespace, opname = _validate_name(signature)
        if namespace != self.ns:
            raise ValueError(
                f"schema namespace {namespace!r} does not match library "
                f"namespace {self.ns!r}"
            )
        op_def = self._define(opname)
        if op_def.name not in self._op_names:
            self._op_names.append(op_def.name)

    def impl(
        self,
        op_name: str,
        fn: Callable[..., Any] | None = None,
        *,
        device_type: str = "CompositeExplicitAutograd",
        dispatch_key: str = "",
        allow_override: bool = True,
    ) -> Callable[..., Any]:
        """

        ``device_type`` accepts composite spellings (``Composite…`` → the
        device-agnostic slot) or concrete devices (``"CPU"``/``"CUDA"``);
        (non-empty wins).  May be used directly or as a decorator.
        """

        if dispatch_key:
            device_type = dispatch_key
        if not isinstance(device_type, str):
            raise TypeError(
                f"device_type must be a str, got {type(device_type)!r}"
            )
        del allow_override
        if device_type in _COMPOSITE_KEYS:
            key: Any = None
        else:
            key = device_type.lower()
        target = f"{self.ns}::{op_name}" if "::" not in op_name else op_name
        op_def = _resolve_op(target)
        if fn is not None and not callable(fn):
            raise TypeError(f"kernel must be callable, got {type(fn)!r}")

        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            op_def.register_kernel(key)(func)
            return func

        return wrapper(fn) if fn is not None else wrapper

    def fallback(self, kind: str) -> None:
        """

        TensorPlay's dispatcher has no per-key fallthrough table, so this
        of silently mis-dispatching.
        """

        raise NotImplementedError(
            "Library.fallback is not supported: TensorPlay's dispatcher has "
            "no per-dispatch-key fallthrough table (kind="
            f"{kind!r})"
        )

    def __repr__(self) -> str:
        return f"<Library ns={self.ns!r} kind={self.kind}>"
