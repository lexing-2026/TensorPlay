from __future__ import annotations

import operator

from tensorplay.graph.node import Node

from ._dispatch_trace import (
    _ConstantHolder,
    _is_tensor,
    _tensor_meta,
    ProxyTensorDispatchMode,
)


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
        # Tokens are kept on a stack of their own rather than in one field,
        # because a context can be entered more than once over the span it
        # covers.  A single field would be overwritten by the inner entry, and
        # the exit that belongs to the outer entry would then restore the state
        # the inner one replaced -- leaving this mode installed after everything
        # that entered it has left, which is a trace that reports itself as
        # running long after it stopped.
        if not hasattr(self, "_tokens"):
            self._tokens: list[Any] = []
        self._tokens.append(_CURRENT_MODE.set(self))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        tokens = getattr(self, "_tokens", None)
        if tokens:
            _CURRENT_MODE.reset(tokens.pop())

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
        tracker[value._impl_id] = proxy
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
    # What comes back is the stand-in, not the value it stands for.  The value
    # came from running something -- from an example computed to find the shape
    # of a result -- and a graph can only name its own nodes, so handing the
    # value on would leave the graph pointing at a region it does not contain.
    # Handing back the stand-in leaves it naming the node that was made for it,
    # which is the node the rest of the graph should be built on.
    if isinstance(proxy, Proxy):
        # The value paired with a stand-in is whatever the run that made it
        # returned, and a run can return a number, a name, or nothing at all --
        # a modifier that returns a constant, for one.  Only a value with an
        # identity of its own is worth remembering a stand-in for; the rest are
        # recorded on the node as they are, which is all that is done with them.
        if not hasattr(value, "_impl_id"):
            set_meta(proxy, value)
            if constant is not None:
                proxy.node.meta["constant"] = constant
            return proxy
        track_tensor(value, proxy, constant=constant, tracer=tracer)
        return proxy
    if isinstance(value, tuple) and isinstance(proxy, tuple):
        return tuple(
            track_tensor_tree(left, right, constant=constant, tracer=tracer)
            for left, right in zip(value, proxy)
        )
    if isinstance(value, list) and isinstance(proxy, list):
        return [
            track_tensor_tree(left, right, constant=constant, tracer=tracer)
            for left, right in zip(value, proxy)
        ]
    if isinstance(value, dict) and isinstance(proxy, dict):
        return {
            key: track_tensor_tree(left, proxy[key], constant=constant, tracer=tracer)
            for key, left in value.items()
            if key in proxy
        }
    return proxy


_CURRENT_MAKE_GRAPH_TRACER: contextvars.ContextVar["MakeGraphTracer | None"] = (
    contextvars.ContextVar("_CURRENT_MAKE_GRAPH_TRACER", default=None)
)


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
        self.tensor_tracker: dict[int, Any] = {}
        self._constant_count = 0

    # -- recording operations as they are performed ------------------------
    #
    # The methods below are what lets an operation be recorded as it happens
    # rather than as the call that led to it.  A method called on a stand-in
    # is the method; an operation performed on a value is the operation, and
    # only the second names the work to be done.  So a trace that watches
    # operations needs the values themselves in the body, and these are how
    # a value that was passed in is found again when an operation consumes it.

    def track(self, tensor: Any, node: Node) -> None:
        """Say which node produced this value, so a later use finds it."""

        self.tensor_tracker[tensor._impl_id] = node
        node.meta["val"] = tensor
        node.meta["tensor_meta"] = _tensor_meta(tensor)

    def node_for(self, tensor: Any) -> Node:
        entry = self.tensor_tracker.get(tensor._impl_id)
        if entry is not None:
            return entry if isinstance(entry, Node) else entry.node
        return self._constant(tensor)

    def map_operands(self, value: Any) -> Any:
        """The value with the values a node stands for replaced by that node.

        An operation's arguments are the values it was handed, and a value
        nothing has computed is one of them whatever the graph is doing: it was
        read, it is not something to be recomputed, and a graph that tried to
        find a node for it would be looking for an operation that never
        happened.  So a value a node already stands for becomes that node, and
        any other value is left as it is.
        """

        if _is_tensor(value):
            return self.node_for(value)
        if isinstance(value, tuple):
            return tuple(self.map_operands(item) for item in value)
        if isinstance(value, list):
            return [self.map_operands(item) for item in value]
        if isinstance(value, dict):
            return {key: self.map_operands(item) for key, item in value.items()}
        return value

    def _constant(self, tensor: Any) -> Node:
        """A node holding a value nothing in the graph accounts for.

        A value the body was handed but that no placeholder stands for -- a
        closure's capture, or an argument the caller specialized -- has to be
        held somewhere, since the operation that made it is not in the graph
        and nothing else refers to it.  It is held on the root, so it is
        found again by the same name the next time.
        """

        name = f"_tensor_constant{self._constant_count}"
        self._constant_count += 1
        if self.root is None:
            self.root = _ConstantHolder()
        setattr(self.root, name, tensor)
        node = self.graph.get_attr(name)
        self.track(tensor, node)
        return node

    def create_arg(self, value: Any) -> Any:
        """An argument of a node being made, with a value a node already
        stands for replaced by that node.

        An operator that takes a whole body -- a scan, a branch -- is put in
        the graph by naming it and its arguments, and the arguments are the
        values the body was running on.  A value that came in through a
        placeholder, or that an earlier operation made, is that node's value:
        naming it as a fresh constant would write the example it happened to
        hold into the graph, and the graph would no longer depend on its
        input.  Only a value no node accounts for is held as a constant.
        """

        if _is_tensor(value):
            entry = self.tensor_tracker.get(value._impl_id)
            if entry is not None:
                return entry if isinstance(entry, Node) else entry.node
        return super().create_arg(value)

    def map_value(self, value: Any) -> Any:
        """The node for a value, or the value itself where it is not a tensor."""

        if _is_tensor(value):
            return self.node_for(value)
        if isinstance(value, tuple):
            return tuple(self.map_value(item) for item in value)
        if isinstance(value, list):
            return [self.map_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.map_value(item) for key, item in value.items()}
        return value

    def record(self, func: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], out: Any) -> Node:
        """Put an operation into the graph, and say what it produced.

        An operation that produced several values is one node, and each value
        is reached from it by position -- so a caller that takes the first
        result gets the node it came from rather than a second copy of the
        operation.
        """

        node = self.graph.create_node(
            "call_function",
            func,
            self.map_operands(tuple(args)),
            self.map_operands(dict(kwargs)),
            name=getattr(func, "_opname", None),
        )
        if _is_tensor(out):
            self.track(out, node)
        elif isinstance(out, (tuple, list)):
            node.meta["val"] = out
            for index, item in enumerate(out):
                if _is_tensor(item):
                    element = self.graph.create_node(
                        "call_function", operator.getitem, (node, index), {}
                    )
                    self.track(item, element)
                elif isinstance(item, (tuple, list)) and any(_is_tensor(v) for v in item):
                    element = self.graph.create_node(
                        "call_function", operator.getitem, (node, index), {}
                    )
                    element.meta["val"] = item
                    for inner_index, inner in enumerate(item):
                        if _is_tensor(inner):
                            leaf = self.graph.create_node(
                                "call_function", operator.getitem, (element, inner_index), {}
                            )
                            self.track(inner, leaf)
        else:
            node.meta["val"] = out
        return node

    def map_output(self, value: Any) -> Any:
        """Say what stands for each value the body returned.

        The body computed on real values, so what it returned is a value;
        the graph stores nodes, and each of these is the node of the
        operation that produced it.
        """

        return self.map_value(value)

    def body_value(self, node: Node, sample: Any) -> Any:
        """Hand the body the value itself, and remember which node it came from.

        The value is what the caller passed, and the node is what stands for
        it in the graph.  Both are needed: the value is what the body
        operates on, so the operations it performs are the ones that get
        recorded, and the node is how an operation that consumes this value is
        found to take it as an argument rather than as something the graph has
        never seen.

        A value that is not a tensor is passed on as itself too, and this is
        not a special case: a length or a scale is part of what the program
        does, and a stand-in for one would be read as a graph value by
        whatever asks about it -- a function that captures a call when one of
        its arguments is a graph value would then capture a call whose scale
        is a stand-in, and the result would be a stand-in where the caller
        expects the value.
        """

        if sample is None:
            return super().body_value(node, sample)
        if _is_tensor(sample):
            self.track(sample, node)
        return sample

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


def _positional_signature(fn: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """A callable taking any number of positional values, as one taking these.

    A graph names each of its inputs, and a callable written over ``*args``
    names none of them -- a function built on the fly, such as a gradient
    formula's, usually is.  It is traced as a callable of exactly the
    arguments it was handed.
    """

    if kwargs:
        return fn
    try:
        parameters = inspect.signature(getattr(fn, "forward", fn)).parameters.values()
    except (TypeError, ValueError):
        return fn
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return fake_signature(fn, len(args))
    return fn


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
    # The mode that watches the operations is what turns a run of the program
    # into a graph of what it did.  Underneath it the state the decomposition
    # helpers read stays the same either way, so both are entered: the one
    # that watches, and the one that is consulted.
    with tracer.proxy_mode, ProxyTensorDispatchMode(tracer, tracer.decomposition_table):
        return tracer.trace(root, sample_inputs=samples)


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
    if tracing_mode == "fake":
        raise NotImplementedError(
            "fake tracing requires storage-free tensor materialization"
        )
    dynamic_shapes = _resolve_dynamic_shapes(f, dynamic_shapes)
    session = MakeGraphTracer(decomposition_table)

    @functools.wraps(f)
    def wrapped(*args: Any, **kwargs: Any) -> GraphModule:
        token = _CURRENT_MAKE_GRAPH_TRACER.set(session)
        try:
            fn = _positional_signature(f, args, kwargs)
            samples = _bind_sample_inputs(fn, args, kwargs)
            if tracing_mode == "symbolic":
                from ._symbolic_trace import SymbolicShapeTracer

                tracer = SymbolicShapeTracer(samples, decomposition_table=decomposition_table)
            else:
                tracer = PythonKeyTracer(decomposition_table=decomposition_table, execute=False)
            tracer.dynamic_shapes = dynamic_shapes
            if tracing_mode == "symbolic":
                with tracer.proxy_mode:
                    graph_module = tracer.trace(fn, sample_inputs=samples)
            else:
                graph_module = dispatch_trace(fn, tracer, samples)
            graph_module.meta["tracing_mode"] = tracing_mode
            return graph_module
        finally:
            _CURRENT_MAKE_GRAPH_TRACER.reset(token)

    return wrapped


class MakeGraphTracer:
    """Session object behind a ``make_fx`` capture.

    Nested captures re-enter the active session through
    :func:`reenter_make_fx`, which routes here to materialize a callable into
    its own standalone subgraph while the outer capture is running.
    """

    def __init__(
        self,
        decomposition_table: Mapping[Any, Callable[..., Any]] | None = None,
    ) -> None:
        self.decomposition_table = dict(decomposition_table or {})

    def _trace_fn(
        self,
        fn: Any,
        decomposition_table: Mapping[Any, Callable[..., Any]] | None,
        args: tuple[Any, ...],
    ) -> GraphModule:
        token = _CURRENT_MAKE_GRAPH_TRACER.set(self)
        try:
            fn = _positional_signature(fn, args, {})
            samples = _bind_sample_inputs(fn, args, {})
            tracer = PythonKeyTracer(
                decomposition_table=decomposition_table, execute=False
            )
            return dispatch_trace(fn, tracer, samples)
        finally:
            _CURRENT_MAKE_GRAPH_TRACER.reset(token)

    def trace_subgraph(self, fn: Any, *args: Any) -> GraphModule:
        return self._trace_fn(fn, self.decomposition_table, args)

    def trace_subgraph_custom_decomp(
        self,
        fn: Any,
        subgraph_decomp_table: Mapping[Any, Callable[..., Any]] | None,
        *args: Any,
    ) -> GraphModule:
        return self._trace_fn(fn, subgraph_decomp_table, args)


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
    mode = _CURRENT_MODE.get()
    if mode is not None:
        return mode
    # The dispatch stack follows the autograd engine onto the thread it runs a
    # backward on; the proxy state does not.  An operator called from a
    # gradient formula there asks the same question, so the recording mode on
    # the stack answers it.  Nothing is found where recording was switched
    # off, since that clears the stack as well.
    from tensorplay._C import _get_dispatch_mode, _len_dispatch_mode

    for index in range(_len_dispatch_mode() - 1, -1, -1):
        tracer = getattr(_get_dispatch_mode(index), "tracer", None)
        if tracer is not None and hasattr(tracer, "proxy_mode"):
            return tracer.proxy_mode
    return None


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
    # Three things answer "is something being recorded right now", and a call
    # that must not be recorded has to be off all of them.  The proxy state is
    # the one an operator consults directly; the dispatch stack is the one every
    # operator is routed through, and the mode that would record it sits on top
    # of that.  Clearing only the first leaves the recording mode in place, and
    # the call it makes is recorded as though it were part of the region -- which
    # for an operator whose own body is the region means it records itself, and
    # the region never gets recorded at all.
    #
    # The third is the tracer a factory records itself on when no argument of
    # its own names one.  Left in place, a tensor made here only to stand for a
    # shape would come back as a node of the region instead of as a tensor.
    #
    # A value of a trace that the code run here reads -- one a branch or loop
    # body closes over -- is read as the tensor it stands for.
    from tensorplay.graph._utils import _active_tracer, reading_enclosing_values
    from tensorplay.utils._dispatch import _disable_current_modes

    previous = _CURRENT_MODE.get()
    token = _CURRENT_MODE.set(None)
    tracer_token = _active_tracer.set(None)
    try:
        with _disable_current_modes(), reading_enclosing_values():
            yield previous
    finally:
        _active_tracer.reset(tracer_token)
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
