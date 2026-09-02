"""Python dispatch utilities for TensorPlay function and subclass hooks.

The native operator boundary calls subclass dispatch hooks when an operand
provides one.  Python-defined APIs use the same precedence rules through the
helpers in this module.  Keeping the ordering and mode stack here gives
composite Python functions one well-defined dispatch protocol without adding
another native entry point.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import inspect
import sys
import threading
import types
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, TypeVar, cast


__all__ = [
    "get_ignored_functions",
    "get_overridable_functions",
    "get_testing_overrides",
    "handle_tensorplay_function",
    "has_tensorplay_function",
    "resolve_name",
    "is_tensor_like",
    "is_tensor_method_or_property",
    "wrap_tensorplay_function",
    "enable_reentrant_dispatch",
    "redispatch_function",
]


_R = TypeVar("_R")
# Function dispatch uses the package-native protocol at every boundary.
_FUNCTION_HOOKS = ("__tensorplay_function__",)
_DISPATCH_HOOKS = ("__tensorplay_dispatch__",)
_ALL_HOOKS = _FUNCTION_HOOKS + _DISPATCH_HOOKS

_NON_DISPATCHABLE_NAMES = frozenset(
    {
        "typename",
        "is_tensor",
        "is_storage",
        "set_default_tensor_type",
        "set_default_device",
        "get_default_device",
        "set_rng_state",
        "get_rng_state",
        "manual_seed",
        "initial_seed",
        "seed",
        "thread_safe_generator",
        "save",
        "load",
        "set_printoptions",
        "get_default_dtype",
        "get_num_interop_threads",
        "get_num_threads",
        "init_num_threads",
        "set_num_interop_threads",
        "set_num_threads",
        "as_tensor",
        "from_numpy",
        "tensor",
        "default_generator",
        "has_cuda",
        "has_cudnn",
        "has_lapack",
        "device",
        "dtype",
        "finfo",
        "iinfo",
        "memory_format",
        "qscheme",
        "layout",
        "empty",
        "empty_strided",
        "eye",
        "full",
        "linspace",
        "logspace",
        "ones",
        "rand",
        "randn",
        "randint",
        "randperm",
        "range",
        "scalar_tensor",
        "sparse_coo_tensor",
        "tril_indices",
        "triu_indices",
        "vander",
        "zeros",
        "arange",
        "broadcast_shapes",
        "can_cast",
        "promote_types",
        "result_type",
    }
)


class _DispatchState(threading.local):
    def __init__(self) -> None:
        self.active: set[tuple[int, type]] = set()


_state = _DispatchState()


def _tensorplay() -> Any:
    module = sys.modules.get("tensorplay")
    if module is None:
        import importlib

        module = importlib.import_module("tensorplay")
    return module


_FUNCTION_ENABLED = 0
_SUBCLASSES_DISABLED = 1
_ALL_DISABLED = 2


def _native_method(name: str) -> Callable[..., Any]:
    native = getattr(_tensorplay(), "_C", None)
    method = getattr(native, name, None) if native is not None else None
    if not callable(method):
        raise RuntimeError(f"native dispatch API {name!r} is unavailable")
    return cast(Callable[..., Any], method)


def _native_call(name: str, *args: Any, **kwargs: Any) -> Any:
    return _native_method(name)(*args, **kwargs)


def _native_state() -> int:
    return int(_native_call("_get_tensor_function_state"))


def _set_native_state(state: int) -> None:
    _native_call("_set_tensor_function_state", int(state))


def _function_dispatch_enabled() -> bool:
    return _native_state() != _ALL_DISABLED


def _subclass_dispatch_enabled() -> bool:
    return _native_state() == _FUNCTION_ENABLED


def _consume_function_skip() -> bool:
    return bool(_native_call("_exchange_tensor_function_skip_next", False))


def _consume_subclass_skip() -> bool:
    return bool(_native_call("_exchange_tensor_subclass_skip_next", False))


def _function_mode_len() -> int:
    return int(_native_call("_len_tensor_function_mode"))


def _effective_hooks(hooks: tuple[str, ...]) -> tuple[str, ...]:
    if _native_state() == _SUBCLASSES_DISABLED:
        return tuple(name for name in hooks if name not in _DISPATCH_HOOKS)
    return hooks


def _hook_for_type(typ: type, hooks: tuple[str, ...] = _FUNCTION_HOOKS) -> Any:
    """Return the first effective hook defined by ``typ`` or its bases."""
    for name in hooks:
        hook = getattr(typ, name, None)
        if hook is not None and callable(hook):
            return hook
    return None


def _hook_for_value(value: Any, hooks: tuple[str, ...] = _FUNCTION_HOOKS) -> Any:
    """Return a hook bound to ``value`` so instance methods receive ``self``."""
    for name in hooks:
        hook = getattr(value, name, None)
        if hook is not None and callable(hook):
            return hook
    return None


def _has_hook(typ: type, hooks: tuple[str, ...] = _ALL_HOOKS) -> bool:
    return _hook_for_type(typ, hooks) is not None


def _is_base_tensor(value: Any) -> bool:
    try:
        tensor = getattr(_tensorplay(), "Tensor", None)
        return tensor is not None and type(value) is tensor
    except (AttributeError, TypeError):
        return False


def _is_dispatch_enabled() -> bool:
    return _function_dispatch_enabled()


def _iter_dispatch_values(value: Any) -> Iterable[Any]:
    """Yield hook candidates nested in the containers accepted by operators."""
    if type(value) in (tuple, list):
        for item in value:
            yield from _iter_dispatch_values(item)
        return
    if type(value) is dict:
        for item in value.values():
            yield from _iter_dispatch_values(item)
        return
    yield value


def _get_overloaded_args(
    relevant_args: Iterable[Any],
    get_type_fn: Callable[[Any], type] | None = None,
    hooks: tuple[str, ...] = _ALL_HOOKS,
    consume_skip: bool = True,
) -> list[Any]:
    """Collect distinct hook-bearing arguments in dispatch precedence order."""
    if get_type_fn is None:
        get_type_fn = type
    if consume_skip and _consume_function_skip():
        return []
    if not _is_dispatch_enabled():
        return []

    hooks = _effective_hooks(hooks)

    overloaded_types: set[type] = set()
    overloaded_args: list[Any] = []
    for arg in relevant_args:
        for arg in _iter_dispatch_values(arg):
            arg_type = get_type_fn(arg)
            if arg_type in overloaded_types or _hook_for_type(arg_type, hooks) is None:
                continue
            overloaded_types.add(arg_type)
            index = len(overloaded_args)
            for old_index, old_arg in enumerate(overloaded_args):
                old_type = get_type_fn(old_arg)
                try:
                    if issubclass(arg_type, old_type):
                        index = old_index
                        break
                except TypeError:
                    continue
            overloaded_args.insert(index, arg)
    return overloaded_args


def has_tensorplay_function(relevant_args: Iterable[Any]) -> bool:
    """Return whether any argument supplies a Python dispatch hook."""
    overloaded = _get_overloaded_args(relevant_args)
    return _is_dispatch_enabled() and (
        bool(overloaded) or _function_mode_len() > 0
    )


def has_tensorplay_function_unary(arg: Any) -> bool:
    """Fast single-argument form of :func:`has_tensorplay_function`."""
    if _consume_function_skip() or not _is_dispatch_enabled():
        return False
    hooks = _effective_hooks(_ALL_HOOKS)
    return _function_mode_len() > 0 or _hook_for_value(arg, hooks) is not None


def has_tensorplay_function_variadic(*args: Any) -> bool:
    """Variadic form without allocating an argument tuple."""
    if _consume_function_skip() or not _is_dispatch_enabled():
        return False
    hooks = _effective_hooks(_ALL_HOOKS)
    if _function_mode_len() > 0:
        return True
    return any(_hook_for_value(arg, hooks) is not None for arg in args)


def _call_function_hook(
    hook: Any,
    public_api: Callable[..., Any],
    types_: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    return hook(public_api, types_, args, kwargs)


def handle_tensorplay_function(
    public_api: Callable[..., _R],
    relevant_args: Iterable[Any],
    *args: Any,
    **kwargs: Any,
) -> _R:
    """Dispatch a composite API through function hooks and active modes.

    Distinct operand types are ordered from the most-derived type to the
    least-derived type; unrelated types retain their first-seen order.  A hook
    may return ``NotImplemented`` to pass control to the next candidate.
    """
    if _consume_function_skip():
        return cast(_R, public_api(*args, **kwargs))
    if not _is_dispatch_enabled():
        return cast(_R, public_api(*args, **kwargs))

    overloaded_args = _get_overloaded_args(relevant_args, consume_skip=False)
    types_ = tuple(type(arg) for arg in overloaded_args)

    if _function_mode_len() > 0:
        mode = _pop_mode()
        try:
            mode_hook = _hook_for_value(mode, _FUNCTION_HOOKS)
            if mode_hook is not None:
                result = _call_function_hook(
                    mode_hook, public_api, types_, args, kwargs
                )
                if result is not NotImplemented:
                    return cast(_R, result)
        finally:
            _push_mode(mode)

    for overloaded_arg in overloaded_args:
        typ = type(overloaded_arg)
        hook = _hook_for_value(overloaded_arg, _effective_hooks(_ALL_HOOKS))
        if hook is None:
            continue
        active_key = (id(public_api), typ)
        if active_key in _state.active:
            name = resolve_name(public_api) or getattr(public_api, "__name__", repr(public_api))
            raise RuntimeError(
                f"recursive function dispatch for {name!r} on {typ.__qualname__}; "
                "use redispatch_function() to reach the next implementation"
            )
        _state.active.add(active_key)
        try:
            result = _call_function_hook(hook, public_api, types_, args, kwargs)
        finally:
            _state.active.discard(active_key)
        if result is not NotImplemented:
            return cast(_R, result)

    name = resolve_name(public_api) or getattr(public_api, "__name__", repr(public_api))
    detail = ", ".join(t.__qualname__ for t in types_)
    if _function_mode_len() > 0:
        raise TypeError(
            f"no implementation found for {name!r} on types [{detail}] "
            f"or active mode {type(_get_current_function_mode()).__qualname__}"
        )
    raise TypeError(f"no implementation found for {name!r} on types [{detail}]")


def wrap_tensorplay_function(
    dispatcher: Callable[..., Iterable[Any]],
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """Decorate a composite function with hook discovery and dispatch."""

    def inner(func: Callable[..., _R]) -> Callable[..., _R]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> _R:
            relevant_args = dispatcher(*args, **kwargs)
            if has_tensorplay_function(relevant_args):
                return handle_tensorplay_function(
                    wrapped, relevant_args, *args, **kwargs
                )
            return func(*args, **kwargs)

        return cast(Callable[..., _R], wrapped)

    return inner


def is_tensor_like(inp: Any) -> bool:
    """Return whether ``inp`` is a Tensor or implements a dispatch hook."""
    return _is_base_tensor(inp) or _has_hook(type(inp))


def _namespace_entries() -> list[tuple[str, Any, Iterable[str]]]:
    tp = _tensorplay()
    entries: list[tuple[str, Any, Iterable[str]]] = [("tensorplay", tp, dir(tp))]
    optional = (
        ("functional", "functional", getattr(tp, "functional", None)),
        ("nn.functional", "nn.functional", None),
        ("nn.init", "nn.init", None),
        ("linalg", "linalg", getattr(tp, "linalg", None)),
        ("fft", "fft", getattr(tp, "fft", None)),
        ("foreach", "foreach", getattr(tp, "foreach", None)),
        ("special", "special", getattr(tp, "special", None)),
    )
    for display, dotted, module in optional:
        if module is None:
            try:
                module = __import__(f"tensorplay.{dotted}", fromlist=["*"])
            except (ImportError, AttributeError):
                continue
        entries.append((f"tensorplay.{display}", module, getattr(module, "__all__", dir(module))))
    try:
        tensor = tp.Tensor
    except AttributeError:
        tensor = None
    if tensor is not None:
        entries.append(("tensorplay.Tensor", tensor, dir(tensor)))
    return entries


def _is_public_callable_name(name: str) -> bool:
    return bool(name) and not name.startswith("_") and name[0].islower()


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _contains_callable(values: Iterable[Any], value: Any) -> bool:
    if not _is_hashable(value):
        return False
    return value in values


def _iter_functions() -> tuple[dict[Any, list[Callable[..., Any]]], dict[Any, str]]:
    overridable: dict[Any, list[Callable[..., Any]]] = collections.defaultdict(list)
    index: dict[Any, str] = {}
    ignored = get_ignored_functions()
    for namespace_name, namespace, names in _namespace_entries():
        for name in names:
            if type(name) is not str:
                continue
            if namespace_name.endswith("Tensor"):
                if name == "__weakref__" or getattr(object, name, None) is getattr(namespace, name, None):
                    continue
            elif not _is_public_callable_name(name):
                continue
            elif name.endswith("_") and namespace_name != "tensorplay.foreach":
                continue
            try:
                value = getattr(namespace, name)
            except AttributeError:
                continue
            if type(value) is types.ModuleType:
                continue
            if type(value) is property:
                descriptor = value.__get__
                if _is_hashable(descriptor):
                    index[descriptor] = f"{namespace_name}.{name}.__get__"
                if _is_hashable(descriptor) and not _contains_callable(ignored, descriptor):
                    overridable[namespace].append(descriptor)
                continue
            if callable(value):
                if not _is_hashable(value):
                    continue
                if value not in index:
                    index[value] = f"{namespace_name}.{name}"
                if not _contains_callable(ignored, value):
                    overridable[namespace].append(value)
    return dict(overridable), index


@functools.cache
def get_ignored_functions() -> set[Callable[..., Any]]:
    """Return public callables that do not receive function-hook dispatch."""
    tp = _tensorplay()
    ignored: set[Callable[..., Any]] = {
        has_tensorplay_function,
        has_tensorplay_function_unary,
        has_tensorplay_function_variadic,
        handle_tensorplay_function,
        is_tensor_like,
        wrap_tensorplay_function,
    }
    for name in (
        "Tensor",
        "Scalar",
        "DType",
        "Device",
        "device",
        "dtype",
        "get_default_dtype",
        "get_default_device",
        "set_default_dtype",
        "set_default_device",
        "is_tensor",
        "typename",
        "save",
        "load",
    ):
        value = getattr(tp, name, None)
        if callable(value):
            ignored.add(value)
    for _, namespace, names in _namespace_entries():
        for name in names:
            if type(name) is not str or name.startswith("_"):
                continue
            try:
                value = getattr(namespace, name)
            except AttributeError:
                continue
            if name in _NON_DISPATCHABLE_NAMES:
                if _is_hashable(value):
                    ignored.add(value)
                continue
            if not callable(value) and hasattr(value, "__get__"):
                if _is_hashable(value.__get__):
                    ignored.add(value.__get__)
    return ignored


@functools.cache
def get_default_nowrap_functions() -> set[Callable[..., Any]]:
    """Return field accessors whose identity should remain stable."""
    tensor = getattr(_tensorplay(), "Tensor", None)
    result: set[Callable[..., Any]] = set()
    if tensor is None:
        return result
    for name in ("_base", "grad", "_grad"):
        descriptor = getattr(tensor, name, None)
        getter = getattr(descriptor, "__get__", None)
        if getter is not None:
            result.add(getter)
    return result


def _signature_wrapper(signature: inspect.Signature | None) -> Callable[..., int]:
    def override(*args: Any, **kwargs: Any) -> int:
        del args, kwargs
        return -1

    if signature is not None:
        override.__signature__ = signature  # type: ignore[attr-defined]
    return override


@functools.cache
def get_testing_overrides() -> dict[Callable[..., Any], Callable[..., int]]:
    """Return signature-preserving sentinels for every overridable callable."""
    result: dict[Callable[..., Any], Callable[..., int]] = {}
    for funcs in get_overridable_functions().values():
        for func in funcs:
            try:
                signature = inspect.signature(func)
            except (TypeError, ValueError):
                signature = None
            result[func] = _signature_wrapper(signature)
    return result


@functools.cache
def _overridable_data() -> tuple[dict[Any, list[Callable[..., Any]]], dict[Any, str]]:
    return _iter_functions()


def get_overridable_functions() -> dict[Any, list[Callable[..., Any]]]:
    """Return namespaces and callables that participate in function hooks."""
    return _overridable_data()[0]


def resolve_name(f: Any) -> str | None:
    """Return the stable public name registered for a callable."""
    name = _overridable_data()[1].get(f) if _is_hashable(f) else None
    if name is not None:
        return name
    tensor = getattr(_tensorplay(), "Tensor", None)
    if tensor is not None:
        for method_name in dir(tensor):
            try:
                if getattr(tensor, method_name) is f:
                    return f"tensorplay.Tensor.{method_name}"
            except AttributeError:
                continue
    module = getattr(f, "__module__", None)
    qualname = getattr(f, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return None


@functools.cache
def _get_tensor_methods() -> set[Callable[..., Any]]:
    tensor = getattr(_tensorplay(), "Tensor", None)
    if tensor is None:
        return set()
    return set(get_overridable_functions().get(tensor, ()))


def is_tensor_method_or_property(func: Callable[..., Any]) -> bool:
    """Return whether ``func`` is a Tensor method or property getter."""
    if _contains_callable(_get_tensor_methods(), func):
        return True
    tensor = getattr(_tensorplay(), "Tensor", None)
    if tensor is not None:
        for name in dir(tensor):
            try:
                if getattr(tensor, name) is func:
                    return True
            except AttributeError:
                continue
    return getattr(func, "__name__", None) == "__get__"


class TensorPlayFunctionMode:
    """Context manager that handles function hooks for a dynamic scope."""

    def __enter__(self) -> "TensorPlayFunctionMode":
        _push_mode(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        current = _get_current_function_mode()
        if current is not self:
            raise RuntimeError("function mode stack is not properly nested")
        popped = _pop_mode()
        if popped is not self:
            _push_mode(popped)
            raise RuntimeError("function mode stack is not properly nested")

    def __tensorplay_function__(
        self,
        func: Callable[..., Any],
        types_: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types_
        return func(*args, **(kwargs or {}))

    @classmethod
    def push(cls, *args: Any, **kwargs: Any) -> "TensorPlayFunctionMode":
        instance = cls(*args, **kwargs)
        _push_mode(instance)
        return instance


def _get_current_function_mode() -> TensorPlayFunctionMode | None:
    length = _function_mode_len()
    if length == 0:
        return None
    return cast(
        TensorPlayFunctionMode,
        _native_call("_get_tensor_function_mode", length - 1),
    )


def _get_current_function_mode_stack() -> list[TensorPlayFunctionMode]:
    return [
        cast(
            TensorPlayFunctionMode,
            _native_call("_get_tensor_function_mode", index),
        )
        for index in range(_function_mode_len())
    ]


def _push_mode(mode: TensorPlayFunctionMode) -> None:
    _native_call("_push_tensor_function_mode", mode)


def _pop_mode() -> TensorPlayFunctionMode:
    return cast(
        TensorPlayFunctionMode, _native_call("_pop_tensor_function_mode")
    )


@contextlib.contextmanager
def _pop_mode_temporarily() -> Any:
    mode = _pop_mode()
    try:
        yield mode
    finally:
        _push_mode(mode)


class BaseTensorPlayFunctionMode(TensorPlayFunctionMode):
    """Mode that forwards to the selected callable."""

    def __tensorplay_function__(
        self,
        func: Callable[..., Any],
        types_: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types_
        return func(*args, **(kwargs or {}))


@contextlib.contextmanager
def _enable_tensorplay_function() -> Any:
    previous = _native_state()
    try:
        _set_native_state(_FUNCTION_ENABLED)
        yield
    finally:
        _set_native_state(previous)


@contextlib.contextmanager
def _disable_tensorplay_function() -> Any:
    """Temporarily suppress Python hook discovery on the current thread."""
    previous = _native_state()
    try:
        _set_native_state(_ALL_DISABLED)
        yield
    finally:
        _set_native_state(previous)


@contextlib.contextmanager
def enable_reentrant_dispatch() -> Any:
    """Enable nested hook discovery while an override is executing."""
    previous_state = _native_state()
    previous_function_skip = bool(
        _native_call("_exchange_tensor_function_skip_next", False)
    )
    previous_subclass_skip = bool(
        _native_call("_exchange_tensor_subclass_skip_next", False)
    )
    try:
        _set_native_state(_FUNCTION_ENABLED)
        yield
    finally:
        _native_call(
            "_exchange_tensor_function_skip_next", previous_function_skip
        )
        _native_call(
            "_exchange_tensor_subclass_skip_next", previous_subclass_skip
        )
        _set_native_state(previous_state)


def redispatch_function(
    func: Any,
    types_: Iterable[type],
    args: Iterable[Any] | None,
    kwargs: dict[str, Any] | None,
) -> _R:
    """Skip the current native dispatch layer and invoke ``func`` again."""
    type_list = tuple(types_)
    del type_list
    if args is None:
        call_args: tuple[Any, ...] = ()
    elif type(args) is tuple:
        call_args = cast(tuple[Any, ...], args)
    elif type(args) is list:
        call_args = tuple(args)
    else:
        raise TypeError("redispatch_function() args must be a tuple or list")
    if kwargs is None:
        call_kwargs: dict[str, Any] = {}
    elif type(kwargs) is dict:
        call_kwargs = kwargs
    else:
        raise TypeError("redispatch_function() kwargs must be a dictionary")

    clear_subclass_skip = False
    if type(func) is str:
        target = getattr(_tensorplay(), func, None)
        if not callable(target):
            native = getattr(_tensorplay(), "_C", None)
            target = getattr(native, func, None) if native is not None else None
        if not callable(target):
            raise TypeError(f"no callable tensorplay operation named {func!r}")
        if bool(_native_call("_peek_tensor_function_skip_next")) or bool(
            _native_call("_peek_tensor_subclass_skip_next")
        ):
            raise RuntimeError("cannot skip two dispatch levels")
        _native_call("_exchange_tensor_function_skip_next", True)
        _native_call("_exchange_tensor_subclass_skip_next", True)
        clear_subclass_skip = True
    elif callable(func):
        target = cast(Callable[..., _R], func)
        if bool(_native_call("_peek_tensor_function_skip_next")):
            raise RuntimeError("cannot skip two function dispatch levels")
        dispatch_layer = int(_native_call("_get_tensor_dispatch_layer"))
        if dispatch_layer == 3 and bool(
            _native_call("_peek_tensor_subclass_skip_next")
        ):
            raise RuntimeError("cannot skip two subclass dispatch levels")
        _native_call("_exchange_tensor_function_skip_next", True)
        if dispatch_layer == 3:
            _native_call("_exchange_tensor_subclass_skip_next", True)
            clear_subclass_skip = True
    else:
        raise TypeError("redispatch_function() expected a callable or operation name")

    result: Any = None
    try:
        result = target(*call_args, **call_kwargs)
    except BaseException:
        _native_call("_exchange_tensor_function_skip_next", False)
        if clear_subclass_skip:
            _native_call("_exchange_tensor_subclass_skip_next", False)
        raise
    finally:
        function_left = bool(
            _native_call("_exchange_tensor_function_skip_next", False)
        )
        subclass_left = (
            bool(_native_call("_exchange_tensor_subclass_skip_next", False))
            if clear_subclass_skip
            else False
        )
    if function_left or subclass_left:
        raise RuntimeError(
            "redispatch_function() target did not enter the native dispatch boundary"
        )
    return cast(_R, result)


_tp_namespace = sys.modules.get("tensorplay")
if _tp_namespace is not None:
    setattr(_tp_namespace, "overrides", sys.modules[__name__])
