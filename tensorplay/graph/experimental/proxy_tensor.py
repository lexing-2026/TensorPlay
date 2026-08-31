from __future__ import annotations

import contextvars
import functools
import inspect
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import Any, TypeVar

from ..graph import Graph
from ..graph_module import GraphModule
from ..interpreter import Transformer
from ..node import Node
from ..proxy import Proxy
from ..tracer import Tracer
from .._pytree import tree_flatten, tree_unflatten
from .dynamic_spec import ParamsSpec, ShapesSpec, _resolve_dynamic_shapes

__all__ = [
    "DecompositionInterpreter",
    "PythonKeyTracer",
    "decompose",
    "dispatch_trace",
    "extract_val",
    "fake_signature",
    "get_innermost_proxy_mode",
    "get_proxy_mode",
    "handle_sym_dispatch",
    "make_graph",
    "maybe_disable_thunkify",
    "maybe_enable_thunkify",
    "selective_decompose",
    "set_meta",
    "snapshot_fake",
    "track_tensor",
    "track_tensor_tree",
    "disable_proxy_modes_tracing",
    "get_dispatch_modes",
    "get_proxy_node",
    "unwrap_proxy",
    "wrap_with_proxy",
    "wrapper_and_args_for_make_graph",
    "get_isolated_graphmodule",
    "disable_autocast_cache",
]

T = TypeVar("T")
_CURRENT_MODE: contextvars.ContextVar["ProxyMode | None"] = contextvars.ContextVar(
    "tensorplay_graph_proxy_mode", default=None
)
_THUNKIFY: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tensorplay_graph_thunkify", default=False
)


def fake_signature(fn: Callable[..., T], nargs: int) -> Callable[..., T]:
    """Wrap a callable with a fixed positional signature."""

    if nargs < 0:
        raise ValueError("nargs must be non-negative")

    @functools.wraps(fn)
    def wrapped(*args: Any) -> T:
        if len(args) != nargs:
            raise TypeError(f"expected {nargs} arguments, got {len(args)}")
        return fn(*args)

    parameters = [
        inspect.Parameter(
            f"arg{index}", inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        for index in range(nargs)
    ]
    wrapped.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    return wrapped


class ProxyMode:
    """State shared by a graph trace and its decomposition helpers."""

    def __init__(self, tracer: "PythonKeyTracer") -> None:
        self.tracer = tracer
        self.decomposition_table: dict[Any, Callable[..., Any]] = {}
        self.enable_thunkify = False

    def __enter__(self) -> "ProxyMode":
        self._token = _CURRENT_MODE.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        token = getattr(self, "_token", None)
        if token is not None:
            _CURRENT_MODE.reset(token)
            self._token = None

    @contextmanager
    def enable_decompositions(
        self, table: Mapping[Any, Callable[..., Any]] | None
    ) -> Generator[Mapping[Any, Callable[..., Any]], None, None]:
        previous = self.decomposition_table
        self.decomposition_table = dict(table or {})
        try:
            yield self.decomposition_table
        finally:
            self.decomposition_table = previous


@contextmanager
def decompose(
    decomposition_table: Mapping[Any, Callable[..., Any]] | None,
) -> Generator[Mapping[Any, Callable[..., Any]], None, None]:
    mode = get_proxy_mode()
    if mode is None:
        raise RuntimeError("decompose requires an active graph trace")
    with mode.enable_decompositions(decomposition_table) as table:
        yield table


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, tuple | list):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten(item))
        return result
    return [value]


def is_sym_node(value: Any) -> bool:
    return value.__class__.__name__ in {"SymInt", "SymFloat", "SymBool", "SymNode"}


def snapshot_fake(value: Any, include_real: bool = False) -> Any:
    """Capture stable metadata from a tensor-like value."""

    if value is None:
        return None
    if include_real:
        return value
    clone = getattr(value, "clone", None)
    if callable(clone):
        try:
            return clone()
        except Exception:
            pass
    return value


def extract_val(value: T, include_real: bool = False) -> T:
    if hasattr(value, "real") and not include_real:
        try:
            return value.real  # type: ignore[return-value]
        except Exception:
            pass
    return value


def set_meta(proxy: Proxy, value: Any) -> Proxy:
    if not isinstance(proxy, Proxy):
        raise TypeError(f"expected Proxy, got {type(proxy).__name__}")
    proxy.node.meta["val"] = extract_val(value)
    proxy.node.meta["tensor_meta"] = _value_metadata(value)
    return proxy


def _value_metadata(value: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"type": type(value)}
    shape = getattr(value, "shape", None)
    if callable(shape):
        shape = shape()
    if shape is not None:
        try:
            metadata["shape"] = tuple(shape)
        except TypeError:
            pass
    for name in ("dtype", "device", "requires_grad"):
        if hasattr(value, name):
            metadata[name] = getattr(value, name)
    return metadata


def track_tensor(value: Any, proxy: Proxy, *, constant: Any = None, tracer: Any = None) -> Any:
    if not isinstance(proxy, Proxy):
        raise TypeError(f"expected Proxy, got {type(proxy).__name__}")
    set_meta(proxy, value)
    if constant is not None:
        proxy.node.meta["constant"] = constant
    owner = tracer or proxy.tracer
    tracker = getattr(owner, "tensor_tracker", None)
    if tracker is None:
        tracker = {}
        setattr(owner, "tensor_tracker", tracker)
    try:
        tracker[id(value)] = proxy
    except TypeError:
        pass
    return value


def track_tensor_tree(
    value: Any,
    proxy: Any,
    *,
    constant: Any = None,
    tracer: Any = None,
) -> Any:
    if isinstance(proxy, Proxy):
        return track_tensor(value, proxy, constant=constant, tracer=tracer)
    if isinstance(value, tuple) and isinstance(proxy, tuple):
        for left, right in zip(value, proxy):
            track_tensor_tree(left, right, constant=constant, tracer=tracer)
    elif isinstance(value, list) and isinstance(proxy, list):
        for left, right in zip(value, proxy):
            track_tensor_tree(left, right, constant=constant, tracer=tracer)
    elif isinstance(value, dict) and isinstance(proxy, dict):
        for key, left in value.items():
            if key in proxy:
                track_tensor_tree(left, proxy[key], constant=constant, tracer=tracer)
    return value


class PythonKeyTracer(Tracer):
    """Tracer used by the functional graph capture entry point."""

    def __init__(
        self,
        decomposition_table: Mapping[Any, Callable[..., Any]] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decomposition_table = dict(decomposition_table or {})
        self.tensor_tracker: dict[int, Proxy] = {}
        self.proxy_mode = ProxyMode(self)

    def create_proxy(
        self, kind: str, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        decomposition = self.proxy_mode.decomposition_table.get(target)
        if decomposition is None:
            decomposition = self.decomposition_table.get(target)
        if decomposition is not None and kind == "call_function":
            result = decomposition(*args, **kwargs)
            if isinstance(result, Proxy):
                return result
            raise TypeError(
                f"decomposition for {target!r} returned {type(result).__name__}; "
                "a traced decomposition must return a Proxy"
            )
        proxy = super().create_proxy(kind, target, args, kwargs)
        if kind == "placeholder":
            proxy.node.meta.setdefault("val", self.sample_inputs.get(str(target)))
        return proxy


def _bind_sample_inputs(fn: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(getattr(fn, "forward", fn))
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def dispatch_trace(
    root: Any,
    tracer: PythonKeyTracer,
    concrete_args: tuple[Any, ...] | dict[str, Any] | None = None,
) -> GraphModule:
    if isinstance(concrete_args, dict):
        samples = concrete_args
    else:
        samples = {}
        if concrete_args is not None:
            samples = _bind_sample_inputs(root, tuple(concrete_args), {})
    token = _CURRENT_MODE.set(tracer.proxy_mode)
    try:
        return tracer.trace(root, sample_inputs=samples)
    finally:
        _CURRENT_MODE.reset(token)


def make_graph(
    f: Any,
    decomposition_table: Mapping[Any, Callable[..., Any]] | None = None,
    tracing_mode: str = "real",
    _allow_non_fake_inputs: bool = False,
    *,
    pre_dispatch: bool = False,
    record_module_stack: bool = False,
    _allow_fake_constant: bool = False,
    _error_on_data_dependent_ops: bool = True,
    record_stack_traces: bool = False,
    proxy_module_inputs: bool = False,
    _disable_function_metadata_mode: bool = False,
    dynamic_shapes: ShapesSpec | ParamsSpec | dict[str, Any] | None = None,
) -> Callable[..., GraphModule]:
    """Return a callable that captures each invocation into a graph module."""

    del (
        _allow_non_fake_inputs,
        pre_dispatch,
        record_module_stack,
        _allow_fake_constant,
        _error_on_data_dependent_ops,
        record_stack_traces,
        proxy_module_inputs,
        _disable_function_metadata_mode,
    )
    if tracing_mode not in {"real", "fake", "symbolic"}:
        raise ValueError(f"unknown tracing mode {tracing_mode!r}")
    if tracing_mode in {"fake", "symbolic"}:
        raise NotImplementedError(
            f"{tracing_mode} tensor materialization requires symbolic shape support"
        )
    dynamic_shapes = _resolve_dynamic_shapes(f, dynamic_shapes)

    @functools.wraps(f)
    def wrapped(*args: Any, **kwargs: Any) -> GraphModule:
        samples = _bind_sample_inputs(f, args, kwargs)
        tracer = PythonKeyTracer(decomposition_table=decomposition_table, execute=False)
        tracer.dynamic_shapes = dynamic_shapes
        graph_module = dispatch_trace(f, tracer, samples)
        graph_module.meta["tracing_mode"] = tracing_mode
        return graph_module

    return wrapped


class DecompositionInterpreter(Transformer):
    """Rebuild a graph while expanding selected call targets."""

    def __init__(
        self,
        module: GraphModule,
        new_graph: Graph | None = None,
        decomposition_table: Mapping[Any, Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(module)
        if new_graph is not None:
            self.new_graph = new_graph
            self.tracer.graph = new_graph
        self.decomposition_table = dict(decomposition_table or {})

    def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        decomposition = self.decomposition_table.get(target)
        if decomposition is not None:
            value = decomposition(*args, **kwargs)
            if isinstance(value, (Proxy, tuple, list, dict)):
                return value
            raise TypeError(f"decomposition for {target!r} returned a non-symbolic value")
        return super().call_function(target, args, kwargs)


def selective_decompose(
    module: GraphModule,
    should_decompose: Callable[[Node], bool],
    decomposition_table: Mapping[Any, Callable[..., Any]] | None,
    **kwargs: Any,
) -> GraphModule:
    """Expand only the nodes selected by a predicate."""

    class Selective(DecompositionInterpreter):
        def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            node = self.last_node
            if node is not None and should_decompose(node):
                return super().call_function(target, args, kwargs)
            return Transformer.call_function(self, target, args, kwargs)

    return Selective(module, decomposition_table=decomposition_table, **kwargs).transform()


def get_proxy_mode() -> ProxyMode | None:
    return _CURRENT_MODE.get()


def get_innermost_proxy_mode() -> ProxyMode | None:
    return get_proxy_mode()


class FunctionMetadataMode(ProxyMode):
    """Record the callable used for metadata propagation during a trace."""

    def __call__(self, function: Callable[..., T], *args: Any, **kwargs: Any) -> Any:
        self.tracer.function_metadata = function
        return function(*args, **kwargs)


class PreDispatchFunctionMode(FunctionMetadataMode):
    """Metadata mode used before a callable reaches the graph dispatcher."""


class ProxyDispatchMode(ProxyMode):
    """Dispatch mode that materializes a graph operation for proxy inputs."""

    def __call__(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        return self.tracer.create_proxy("call_function", target, args, kwargs)


def get_dispatch_modes() -> list[ProxyMode]:
    mode = get_proxy_mode()
    return [] if mode is None else [mode]


@contextmanager
def disable_proxy_modes_tracing() -> Generator[ProxyMode | None, None, None]:
    previous = _CURRENT_MODE.get()
    token = _CURRENT_MODE.set(None)
    try:
        yield previous
    finally:
        _CURRENT_MODE.reset(token)


def get_proxy_node(value: Any) -> Node | None:
    if isinstance(value, Proxy):
        return value.node
    return None


def unwrap_proxy(value: Any) -> Any:
    if isinstance(value, Proxy):
        return value.node
    if isinstance(value, tuple):
        return tuple(unwrap_proxy(item) for item in value)
    if isinstance(value, list):
        return [unwrap_proxy(item) for item in value]
    if isinstance(value, dict):
        return {key: unwrap_proxy(item) for key, item in value.items()}
    return value


def wrap_with_proxy(value: Any, proxy: Any) -> Any:
    if isinstance(value, tuple) and isinstance(proxy, tuple):
        return tuple(wrap_with_proxy(left, right) for left, right in zip(value, proxy))
    if isinstance(value, list) and isinstance(proxy, list):
        return [wrap_with_proxy(left, right) for left, right in zip(value, proxy)]
    if isinstance(value, dict) and isinstance(proxy, dict):
        return {key: wrap_with_proxy(item, proxy[key]) for key, item in value.items() if key in proxy}
    return proxy


def wrapper_and_args_for_make_graph(
    function: Callable[..., T],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[Callable[[list[object]], T], list[object]]:
    flat_args, spec = tree_flatten((args, kwargs))

    @functools.wraps(function)
    def wrapped(flat_values: list[object]) -> T:
        original_args, original_kwargs = tree_unflatten(flat_values, spec)
        return function(*original_args, **original_kwargs)

    return wrapped, flat_args


def get_isolated_graphmodule(
    function: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    tracing_mode: str = "real",
    decomposition_table: Mapping[Any, Callable[..., Any]] | None = None,
) -> GraphModule:
    wrapped, flat_args = wrapper_and_args_for_make_graph(function, args, kwargs)
    with disable_proxy_modes_tracing():
        return make_graph(
            wrapped,
            decomposition_table=decomposition_table,
            tracing_mode=tracing_mode,
        )(flat_args)


@contextmanager
def disable_autocast_cache() -> Generator[None, None, None]:
    import tensorplay as tp

    previous = tp.is_autocast_cache_enabled()
    tp.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        tp.set_autocast_cache_enabled(previous)


def create_arg(value: Any) -> Any:
    return unwrap_proxy(value)


def create_node(
    tracer: PythonKeyTracer,
    kind: str,
    target: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Proxy:
    return tracer.create_proxy(kind, target, args, kwargs or {})


def handle_sym_dispatch(func: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> T:
    mode = get_proxy_mode()
    if mode is None:
        raise RuntimeError("symbolic dispatch requires an active graph trace")
    return func(*args, **kwargs)


@contextmanager
def maybe_enable_thunkify() -> Generator[None, None, None]:
    token = _THUNKIFY.set(True)
    try:
        yield
    finally:
        _THUNKIFY.reset(token)


@contextmanager
def maybe_disable_thunkify() -> Generator[None, None, None]:
    token = _THUNKIFY.set(False)
    try:
        yield
    finally:
        _THUNKIFY.reset(token)
