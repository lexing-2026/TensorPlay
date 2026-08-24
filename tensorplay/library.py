"""User-defined operator registration, mirroring ``torch.library``.

Three public layers are provided:

1. :func:`custom_op` / the :class:`Library` class register opaque custom
   operators with device-specific kernels, optional fake (meta) kernels and
   an autograd formula (:meth:`CustomOpDef.register_autograd`).
2. :func:`triton_op` + :func:`wrap_triton` integrate user-written Triton
   kernels: a ``triton_op`` behaves like any other custom operator in eager
   mode and is captured as one opaque graph node by
   ``tensorplay.compile`` — exactly the fusion-boundary semantics Inductor
   gives ``torch.library.triton_op``.
3. Registered operators compose with every compiler backend: during capture
   a call whose arguments are symbolic records a single ``call_function``
   node targeting the :class:`CustomOpDef`; backends that cannot lower it
   treat the node as a barrier and fall back to the interpreter, which
   dispatches to the registered kernel.

Like ``torch.library.custom_op``, operators are identified by a qualified
``"namespace::name"`` string.  Schemas are not modeled (TensorPlay has no
C++ dispatcher); annotations are optional and never validated beyond the
checks below.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any
from typing import Iterable

import tensorplay
from .compiler.graph import (
    GraphCaptureError,
    capture_call as _capture_call,
    _iter_proxies as _walk_proxies,
)

__all__ = [
    "Library",
    "CustomOpDef",
    "custom_op",
    "triton_op",
    "wrap_triton",
    "register_kernel",
    "register_fake",
    "register_autograd",
    "get_op",
    "has_op",
]


_LOCK = threading.RLock()
# "ns::op" -> CustomOpDef; namespaces that already own a DEF Library.
_OP_REGISTRY: dict[str, "CustomOpDef"] = {}
_DEFINED_LIBRARY_NAMESPACES: set[str] = set()

# Device-type spellings accepted by Library.impl, following torch.library.
_COMPOSITE_KEYS = frozenset(
    {"CompositeExplicitAutograd", "CompositeImplicitAutograd"}
)


def _contains_proxy(*values: Any) -> bool:
    for _proxy in _walk_proxies(list(values)):
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

    ``None`` means "one kernel covers every device" (torch's default when
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


class CustomOpDef:
    """A single registered custom operator (``torch.library.CustomOpDef``).

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
        is_triton_op: bool = False,
    ) -> None:
        self._namespace, self._opname = _validate_name(name)
        self._name = f"{self._namespace}::{self._opname}"
        self._mutates_args = frozenset(_validate_mutates_args(mutates_args))
        self._device_keys = _normalize_device_types(device_types)
        self._is_triton_op = bool(is_triton_op)
        # ``None`` key holds the device-agnostic kernel (device_types=None).
        self._kernels: dict[str | None, Callable[..., Any]] = {}
        self._fake_fn: Callable[..., Any] | None = None
        self._backward: Callable[..., Any] | None = None
        self._setup_context: Callable[..., Any] | None = None
        self._autograd_cls: type | None = None

    def _install_default_kernel(self, fn: Callable[..., Any]) -> None:
        """Use ``fn`` as the initial kernel (the ``@custom_op`` body).

        Mirrors torch: the decorated function implements the operator for
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

    # Readable node names once this object becomes a graph target
    # (compiler.graph._target_name resolves ``target.__name__`` first).
    @property
    def __name__(self) -> str:  # type: ignore[override]
        return self._name

    def __repr__(self) -> str:
        kind = "triton_op" if self._is_triton_op else "custom_op"
        return f"<{kind} {self._name}>"

    # -- registration API --------------------------------------------------

    def register_kernel(
        self, device_types: Any = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register the implementation for one or more device types.

        ``device_types=None`` registers a single device-agnostic kernel;
        otherwise pass a device string (``"cpu"``, ``"cuda"``, ...) or an
        iterable of them.  Re-registering the same device replaces the
        previous kernel (torch raises instead; TensorPlay allows hot swaps
        for interactive sessions and tests).

        CPU/CUDA kernels are additionally mirrored into the native p10
        dispatcher under this operator's qualified name, so native code and
        :meth:`run_native` can invoke them through the real dispatch path.
        """

        keys = _normalize_device_types(device_types)

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(fn):
                raise TypeError(f"kernel must be callable, got {type(fn)!r}")
            with _LOCK:
                if keys is None:
                    self._kernels[None] = fn
                else:
                    for key in keys:
                        self._kernels[key] = fn
            self._mirror_native(keys, fn)
            return fn

        return decorator

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
            except Exception:  # noqa: BLE001 - mirroring is an optimization
                return

    def register_fake(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a meta/fake kernel computing output metadata.

        The fake kernel receives the same arguments but must not allocate
        real tensor data; it returns tensors describing shape/dtype/device
        (``tensorplay.empty_like`` style factories without data).  It is
        exposed for tooling and future shape propagation; capturing
        compilers never execute either kernel version during tracing.
        """

        if not callable(fn):
            raise TypeError(f"fake kernel must be callable, got {type(fn)!r}")
        self._fake_fn = fn
        return fn

    def register_autograd(
        self,
        backward: Callable[..., Any],
        *,
        setup_context: Callable[..., Any] | None = None,
    ) -> None:
        """Attach a gradient formula (``torch.library`` parity).

        ``backward(ctx, *grad_outputs)`` mirrors
        ``torch.autograd.Function.backward``; ``setup_context(ctx, inputs,
        output)`` may save tensors via ``ctx.save_for_backward``.  When no
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

        _CustomOpAutograd.__qualname__ = f"_CustomOpAutograd[{self._name}]"
        return _CustomOpAutograd

    # -- dispatch ----------------------------------------------------------

    def _kernel_for(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        key = _first_device_key(args)
        with _LOCK:
            if key is not None and key in self._kernels:
                return self._kernels[key]
            if None in self._kernels:
                return self._kernels[None]
            registered = sorted(str(item) for item in self._kernels)
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

    def _eager_call(self, tensors: tuple[Any, ...]) -> Any:
        """Dispatch real tensors with full eager semantics (no capture).

        Shared by :meth:`__call__` and the native re-entry below so compiled
        graphs keep device dispatch AND ``register_autograd`` behavior.
        """

        if self._autograd_cls is not None and tensorplay.is_grad_enabled():
            # Grad-mode disabled mirrors the dispatcher's autograd
            # fallthrough: run the kernel directly, no history, no warnings.
            return self._autograd_cls.apply(*tensors)
        return self._run_kernel(tensors, {})

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        captured = _capture_call(self, args, kwargs)
        if captured is not None:
            return captured
        return self._eager_call(args)


def _first_device_key(values: tuple[Any, ...]) -> str | None:
    for value in values:
        if isinstance(value, tensorplay.Tensor):
            try:
                return str(value.device.type)
            except AttributeError:
                return "cuda" if value.device.is_cuda() else "cpu"
    return None


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


def custom_op(
    name: str,
    *,
    mutates_args: Sequence[str] = (),
    device_types: Any = None,
) -> Callable[[Callable[..., Any]], CustomOpDef]:
    """Define a user operator; use as a decorator (torch parity).

    Example::

        @tensorplay.library.custom_op("mylib::weighted_sum", mutates_args=())
        def weighted_sum(x, weight):
            return (x * weight).sum()

        # Optional extra kernels per device:
        @weighted_sum.register_kernel("cuda")
        def _(x, weight): ...

    The decorated function becomes the operator's default kernel for the
    advertised ``device_types`` (every device when omitted).

    Args:
        name: Qualified ``"namespace::name"`` identifier.
        mutates_args: Names of arguments the kernel mutates in place.
            Compile-time fusion treats these as barriers regardless of the
            value; eager execution trusts the declaration.
        device_types: Restriction advertised to users at definition time.
            Kernels are selected per call from whatever was registered via
            :meth:`CustomOpDef.register_kernel`.

    Returns:
        A decorator producing a callable :class:`CustomOpDef`.
    """

    _validate_name(name)
    _validate_mutates_args(mutates_args)
    _normalize_device_types(device_types)

    def decorator(fn: Callable[..., Any]) -> CustomOpDef:
        op_def = CustomOpDef(
            name, mutates_args=mutates_args, device_types=device_types
        )
        _register_op_def(op_def)
        op_def._install_default_kernel(fn)
        return op_def

    return decorator


def triton_op(
    name: str,
    *,
    mutates_args: Sequence[str] = (),
    device_types: Any = None,
) -> Callable[[Callable[..., Any]], CustomOpDef]:
    """Define a Triton-backed operator (``torch.library.triton_op``).

    The registered kernel(s) must launch their Triton kernels through
    :func:`wrap_triton` and only mutate arguments listed in
    ``mutates_args``.  Under ``tensorplay.compile`` the whole operator is
    captured as a single opaque node — the compiler never traces into the
    Triton launches, matching Inductor's contract.
    """

    _validate_name(name)
    _validate_mutates_args(mutates_args)
    _normalize_device_types(device_types)

    def decorator(fn: Callable[..., Any]) -> CustomOpDef:
        op_def = CustomOpDef(
            name,
            mutates_args=mutates_args,
            device_types=device_types,
            is_triton_op=True,
        )
        _register_op_def(op_def)
        op_def._install_default_kernel(fn)
        return op_def

    return decorator


def _native_invoke(op_name: str, *tensors: Any) -> Any:
    """Re-entry point for compiled native graphs (Stax ``custom_op`` nodes).

    The C++ executor installed by the bindings calls this with the
    operator's qualified name and its tensor inputs; routing through the
    :class:`CustomOpDef` keeps device dispatch and autograd identical to
    eager execution instead of bypassing them with a raw kernel.
    """

    return get_op(op_name)._eager_call(tuple(tensors))


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


def register_kernel(
    op: str | CustomOpDef, device_types: Any = ()
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Top-level kernel registration (``torch.library.register_kernel``).

    Accepts a :class:`CustomOpDef` or a qualified operator name.  An empty
    ``device_types`` iterable means the device-agnostic slot.
    """

    op_def = _resolve_op(op)
    if device_types is None:
        key: Any = None
    elif isinstance(device_types, str):
        key = device_types
    else:
        keys = list(device_types)
        # An empty iterable means the device-agnostic slot.
        key = None if not keys else keys
    return op_def.register_kernel(key)


def register_fake(
    op: str | CustomOpDef,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Top-level fake-kernel registration (``torch.library.register_fake``)."""

    op_def = _resolve_op(op)
    return op_def.register_fake


def register_autograd(
    op: str | CustomOpDef,
    backward: Callable[..., Any],
    *,
    setup_context: Callable[..., Any] | None = None,
) -> None:
    """Top-level autograd registration (``torch.library.register_autograd``)."""

    op_def = _resolve_op(op)
    op_def.register_autograd(backward, setup_context=setup_context)


class Library:
    """Scoped registration bundle (``torch.library.Library`` subset).

    Kinds mirror torch: ``"DEF"`` defines operators in a namespace (only
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

    def define(self, schema: str, *, alias_analysis: str = "") -> None:
        """Define an operator from a schema like ``"ns::add(Tensor, Tensor)"``.

        Only the qualified name is meaningful (TensorPlay models no schema
        grammar); the parenthesized remainder is ignored so torch-style
        schema strings can be pasted verbatim.
        """

        del alias_analysis
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
    ) -> Callable[..., Any]:
        """Register a kernel, torch.library.impl-style.

        ``device_type`` accepts composite spellings (``Composite…`` → the
        device-agnostic slot) or concrete devices (``"CPU"``/``"CUDA"``).
        May be used directly or as a decorator.
        """

        if not isinstance(device_type, str):
            raise TypeError(
                f"device_type must be a str, got {type(device_type)!r}"
            )
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

    def __repr__(self) -> str:
        return f"<Library ns={self.ns!r} kind={self.kind}>"
