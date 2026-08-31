from __future__ import annotations

import copy
import dataclasses
import dis
import enum
import inspect
import sys
import types
import operator
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from ._utils import GraphCaptureError, _capture_disabled, capturing
from .node import Node


class Scope:
    __slots__ = ("module_path", "module_type")

    def __init__(self, module_path: str, module_type: type[Any] | None) -> None:
        self.module_path = module_path
        self.module_type = module_type

    def __repr__(self) -> str:
        return f"Scope(module_path={self.module_path!r}, module_type={self.module_type!r})"


class ScopeContextManager:
    __slots__ = ("scope", "module_path", "module_type", "_old_scope")

    def __init__(
        self,
        scope: Scope,
        module_path: str,
        module_type: type[Any] | None,
    ) -> None:
        self.scope = scope
        self.module_path = module_path
        self.module_type = module_type
        self._old_scope: Scope | None = None

    def __enter__(self) -> "ScopeContextManager":
        self._old_scope = Scope(self.scope.module_path, self.scope.module_type)
        self.scope.module_path = self.module_path
        self.scope.module_type = self.module_type
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        if self._old_scope is not None:
            self.scope.module_path = self._old_scope.module_path
            self.scope.module_type = self._old_scope.module_type


class TraceError(GraphCaptureError):
    """Raised when Python evaluation requires an unavailable graph value."""


def _is_arbitrary_callable(value: Any) -> bool:
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        return False
    if inspect.ismethod(value) or inspect.isclass(value):
        return False
    return callable(value)


def _find_arbitrary_callable(value: Any) -> Any | None:
    if _is_arbitrary_callable(value):
        return value
    if isinstance(value, tuple | list):
        for item in value:
            found = _find_arbitrary_callable(item)
            if found is not None:
                return found
    elif isinstance(value, Mapping):
        for item in value.values():
            found = _find_arbitrary_callable(item)
            if found is not None:
                return found
    elif isinstance(value, slice):
        for item in (value.start, value.stop, value.step):
            found = _find_arbitrary_callable(item)
            if found is not None:
                return found
    return None


def _register_stack_trace_anchor(value: Any, frame: types.FrameType | None = None) -> None:
    del value, frame


def _iter_values(value: Any) -> Iterator[Any]:
    if isinstance(value, tuple | list):
        for item in value:
            yield from _iter_values(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_values(item)
    elif isinstance(value, slice):
        yield from _iter_values(value.start)
        yield from _iter_values(value.stop)
        yield from _iter_values(value.step)
    else:
        yield value


_PRESERVED_NODE_META_FIELDS = (
    "module_stack",
    "nn_module_stack",
    "source_fn",
    "source_fn_stack",
    "from_node",
    "custom",
    "partitioner_tag",
)


def _apply_preserved_node_meta(node: Node) -> None:
    from . import traceback as graph_traceback

    if not graph_traceback.has_preserved_node_meta():
        return
    current_meta = graph_traceback.get_current_meta()
    stack_trace = current_meta.get("stack_trace")
    if stack_trace:
        node.stack_trace = stack_trace
    for field in _PRESERVED_NODE_META_FIELDS:
        if field in current_meta:
            node.meta[field] = copy.copy(current_meta[field])
    if current_meta.get("autograd_backward", False):
        node.meta["autograd_backward"] = True


class TracerBase:
    """Base protocol for tracers that append operations to a graph."""

    def __init__(self) -> None:
        self.record_stack_traces = False
        self._record_forward_stack_traces = False
        self.check_mutable_operations = True
        self.trace_asserts = False
        self.proxy_buffer_attributes = False
        self.traced_func_name = "forward"

    def create_node(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any | None = None,
    ) -> Node:
        if self.check_mutable_operations:
            from .operator_schemas import check_for_mutable_operation

            check_for_mutable_operation(kind, target, args, kwargs)
        graph = getattr(self, "graph", None)
        if graph is None:
            raise TraceError("tracer has no graph")
        node = graph.create_node(kind, target, args, kwargs, name, type_expr)
        _apply_preserved_node_meta(node)
        scope = getattr(self, "scope", None)
        if scope is not None:
            node_scope = getattr(self, "node_name_to_scope", None)
            if isinstance(node_scope, dict):
                node_scope[node.name] = (scope.module_path, scope.module_type)
        module_stack = getattr(self, "module_stack", None)
        if module_stack:
            node.meta["module_stack"] = copy.copy(module_stack)
        if self.record_stack_traces and not node.meta.get("stack_trace"):
            frames = inspect.stack(context=1)
            try:
                node.meta["stack_trace"] = "\n".join(
                    f'File "{frame.filename}", line {frame.lineno}, in {frame.function}\n'
                    f"    {frame.code_context[0].strip() if frame.code_context else ''}"
                    for frame in frames[1:]
                )
            finally:
                del frames
        return node

    def proxy(self, node: Node) -> "Proxy":
        return Proxy(node, self)

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any | None = None,
        proxy_factory_fn: Callable[[Node], "Proxy"] | None = None,
    ) -> "Proxy":
        if _capture_disabled.get():
            raise GraphCaptureError("graph capture is disabled for this operation")
        graph_args = self.create_arg(args)
        graph_kwargs = self.create_arg(kwargs)
        if not isinstance(graph_args, tuple):
            raise TraceError(f"expected tuple arguments, got {type(graph_args).__name__}")
        if not isinstance(graph_kwargs, dict):
            raise TraceError(f"expected dict keyword arguments, got {type(graph_kwargs).__name__}")
        node = self.create_node(kind, target, graph_args, graph_kwargs, name, type_expr)
        return self.proxy(node) if proxy_factory_fn is None else proxy_factory_fn(node)

    def create_arg(self, value: Any) -> Any:
        if isinstance(value, Proxy):
            return value.node
        creator = getattr(value, "__tensorplay_create_arg__", None)
        if callable(creator):
            return creator(self)
        if isinstance(value, tuple):
            mapped = [self.create_arg(item) for item in value]
            if hasattr(value, "_fields"):
                return type(value)(*mapped)
            try:
                return type(value)(mapped)
            except TypeError:
                return tuple(mapped)
        if isinstance(value, list):
            return [self.create_arg(item) for item in value]
        if isinstance(value, Mapping):
            result: dict[Any, Any] = {}
            for key, item in value.items():
                graph_key = key if isinstance(key, str) else self.create_arg(key)
                _reject_node_keys(graph_key)
                result[graph_key] = self.create_arg(item)
            if type(value) is dict:
                return result
            try:
                return type(value)(result)
            except TypeError:
                return result
        if isinstance(value, slice):
            return slice(
                self.create_arg(value.start),
                self.create_arg(value.stop),
                self.create_arg(value.step),
            )
        if isinstance(value, range):
            return range(
                self.create_arg(value.start),
                self.create_arg(value.stop),
                self.create_arg(value.step),
            )
        if isinstance(value, enum.Enum) or value is None or value is Ellipsis:
            return value
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = {
                field.name: self.create_arg(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
            return self.create_node("call_function", type(value), (), fields)
        if isinstance(value, (str, bytes, int, float, bool, complex, type(None), type)):
            return value
        if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)) or callable(value):
            return value
        raise NotImplementedError(f"argument of type {type(value).__name__!r} is not supported")

    def to_bool(self, obj: "Proxy") -> bool:
        del obj
        raise TraceError("symbolic graph values cannot be used as Python conditions")

    def iter(self, obj: "Proxy") -> Iterator[Any]:
        del obj
        raise TraceError("symbolic graph values cannot be iterated")

    def keys(self, obj: "Proxy") -> "Proxy":
        return Attribute(obj, "keys")()


def assert_fn(value: Any) -> None:
    if not value:
        raise AssertionError


class GraphAppendingTracer(TracerBase):
    """Tracer used when a raw graph node is wrapped directly."""

    def __init__(self, graph: Any) -> None:
        super().__init__()
        self.graph = graph
        self.scope = Scope("", None)
        self.module_stack: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        self.node_name_to_scope: dict[str, tuple[str, type[Any] | None]] = {}


class Proxy:
    """Symbolic value used while the frontend captures Python operations.

    Single value domain: this class represents both tensor and scalar values
    during capture.  A scalar routed
    through :func:`gate` stays this same proxy (the 1-element tensor);
    ``symbolic_gate_nodes`` carries the UPV flag bit, ``_node_samples``
    carries ``raw_value``, ``__int__/__float__`` are the ``need_unwrap``
    exits, and a missing sample raises like ``FakeItemVariable``.
    """

    __slots__ = ("node", "tracer", "__dict__")

    def __init__(self, node: Node, tracer: Any = None) -> None:
        if tracer is None:
            tracer = GraphAppendingTracer(node.graph)
        self.node = node
        self.tracer = tracer

    def _binary(self, target: Any, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", target, (self, other), {})

    def _unary(self, target: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", target, (self,), {})

    def __add__(self, other: Any) -> "Proxy":
        return self._binary(operator.add, other)

    def __radd__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.add, (other, self), {})

    def __sub__(self, other: Any) -> "Proxy":
        return self._binary(operator.sub, other)

    def __rsub__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.sub, (other, self), {})

    def __mul__(self, other: Any) -> "Proxy":
        return self._binary(operator.mul, other)

    def __rmul__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.mul, (other, self), {})

    def __truediv__(self, other: Any) -> "Proxy":
        return self._binary(operator.truediv, other)

    def __rtruediv__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.truediv, (other, self), {})

    def __floordiv__(self, other: Any) -> "Proxy":
        return self._binary(operator.floordiv, other)

    def __rfloordiv__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.floordiv, (other, self), {})

    def __mod__(self, other: Any) -> "Proxy":
        return self._binary(operator.mod, other)

    def __rmod__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.mod, (other, self), {})

    def __pow__(self, other: Any) -> "Proxy":
        return self._binary(operator.pow, other)

    def __rpow__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.pow, (other, self), {})

    def __matmul__(self, other: Any) -> "Proxy":
        return self._binary(operator.matmul, other)

    def __rmatmul__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.matmul, (other, self), {})

    def __lt__(self, other: Any) -> "Proxy":
        return self._binary(operator.lt, other)

    def __le__(self, other: Any) -> "Proxy":
        return self._binary(operator.le, other)

    def __eq__(self, other: Any) -> "Proxy":  # type: ignore[override]
        return self._binary(operator.eq, other)

    def __ne__(self, other: Any) -> "Proxy":  # type: ignore[override]
        return self._binary(operator.ne, other)

    def __gt__(self, other: Any) -> "Proxy":
        return self._binary(operator.gt, other)

    def __ge__(self, other: Any) -> "Proxy":
        return self._binary(operator.ge, other)

    def __neg__(self) -> "Proxy":
        return self._unary(operator.neg)

    def __pos__(self) -> "Proxy":
        return self._unary(operator.pos)

    def __getitem__(self, key: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.getitem, (self, key), {})

    def __abs__(self) -> "Proxy":
        return self._unary(operator.abs)

    def _sample(self) -> Any:
        """Example value bound to this node, if the tracer got one.

        Execute-mode tracers propagate samples through every recorded node;
        symbolic tracers only know placeholder inputs.
        """

        sample = self.tracer._node_samples.get(self.node.name)
        if sample is None and self.node.op == "placeholder":
            return self.tracer._samples.get(self.node.name)
        return sample

    def _specialize(self, kind: str, value: Any) -> Any:
        """Record a data-specialization consumption and route the value.

        ``bool``/``iter`` gates decide which Python path executes, so their
        outcome stays part of the cache key (per-branch artifacts).  Numeric
        gates (``int``/``float``) instead become synthetic placeholder inputs:
        the value flows into the graph as a runtime 0-d tensor, giving ONE
        specialization across all gate values (api.py re-evaluates the
        condition subgraph per call and feeds it back).  ``index`` stays
        fully specialized — native slicing/range need real Python ints.
        """

        # Numeric gates (int/float) cannot return a Proxy: CPython enforces
        # exact int/float returns for __int__/__float__, so their values stay
        # outcome-keyed per branch.  Runtime-parametric scalars need a
        # bytecode/frame-level frontend (plan L1-D2/D3).
        self.tracer.data_specializations.append((self.node.name, kind))
        return value

    def _scalar_sample(self) -> Any:
        """Python scalar behind this node for control-flow gates.

        A 0-d/1-element tensor reduces
        through ``item()``.  Returns ``None`` when no sample is available,
        which keeps purely symbolic capture failing fast.
        """

        sample = self._sample()
        if sample is None:
            return None
        if isinstance(sample, (bool, int, float)):
            return sample
        item = getattr(sample, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:
                return None
        return None

    def _property(self, name: str) -> Any:
        """Resolve tensor metadata: concretely when a sample is available.

        Metadata (shape/dtype/device/...) is part of the compile signature,
        so specializing on it adds no new recompile conditions; data reads
        stay symbolic or raise.
        """

        sample = self._sample()
        self.tracer.metadata_touches.add((self.node.name, name))
        if sample is not None:
            return getattr(sample, name)
        return self.tracer.create_proxy("call_function", getattr, (self, name), {})

    @property
    def shape(self) -> Any:
        return self._property("shape")

    @property
    def dtype(self) -> "Proxy":
        return self._property("dtype")

    @property
    def device(self) -> "Proxy":
        return self._property("device")

    @property
    def ndim(self) -> "Proxy":
        return self._property("ndim")

    @property
    def requires_grad(self) -> "Proxy":
        return self._property("requires_grad")

    def __getattr__(self, name: str) -> Callable[..., "Proxy"]:
        return Attribute(self, name)

    def __call__(self, *args: Any, **kwargs: Any) -> "Proxy":
        return self.tracer.create_proxy(
            "call_method", "__call__", (self, *args), kwargs
        )

    def sin(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "sin", (self,), {})

    def cos(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "cos", (self,), {})

    def exp(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "exp", (self,), {})

    def sqrt(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "sqrt", (self,), {})

    def relu(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "relu", (self,), {})

    def __bool__(self) -> bool:
        scalar = self._scalar_sample()
        if scalar is not None:
            return bool(self._specialize("bool", scalar))
        tracer_to_bool = getattr(self.tracer, "to_bool", None)
        if callable(tracer_to_bool):
            return tracer_to_bool(self)
        raise TraceError(
            "symbolic graph values cannot be used as Python conditions"
        )

    def __index__(self) -> int:
        scalar = self._scalar_sample()
        if scalar is None:
            raise GraphCaptureError(
                "using a Proxy as an integer is not supported during graph capture"
            )
        return self._specialize("index", int(scalar))

    def __int__(self) -> int:
        # Protocol note: returning an int SUBCLASS here works on current
        # CPython but is deprecated ("may be removed"), so numeric gates do
        # NOT smuggle symbolic scalars through __int__ — use the explicit
        # ``tensorplay.graph.gate`` entry point instead,
        scalar = self._scalar_sample()
        if scalar is None:
            raise GraphCaptureError("int(Proxy) is not supported during graph capture")
        self._specialize("int", scalar)
        return int(scalar)

    def __float__(self) -> float:
        scalar = self._scalar_sample()
        if scalar is None:
            raise GraphCaptureError(
                "float(Proxy) is not supported during graph capture"
            )
        self._specialize("float", scalar)
        return float(scalar)

    def __len__(self) -> int:
        sample = self._sample()
        self.tracer.metadata_touches.add((self.node.name, "len"))
        if sample is not None:
            if hasattr(sample, "__len__"):
                try:
                    return int(len(sample))
                except TypeError:
                    pass
            shape = getattr(sample, "shape", None)
            if callable(shape):
                shape = shape()
            try:
                dims = list(shape)
            except TypeError:
                dims = None
            if dims:
                return int(dims[0])
        raise GraphCaptureError(
            "len(Proxy) is not supported during graph capture; provide "
            "sample inputs to specialize on tensor shapes"
        )

    def keys(self) -> "Proxy":
        tracer_keys = getattr(self.tracer, "keys", None)
        if callable(tracer_keys):
            return tracer_keys(self)
        return Attribute(self, "keys")()

    def __iter__(self) -> Iterator[Any]:
        sample = self._sample()
        if isinstance(sample, (tuple, list)):
            if sample and any(_is_tp_tensor(item) for item in sample):
                custom = self.node.meta.get("custom")
                if isinstance(custom, dict) and custom.get("nested_region_config") is not None:
                    return (self[index] for index in range(len(sample)))
                raise GraphCaptureError(
                    "iterating over a tensor-valued Proxy is not supported "
                    "during graph capture"
                )
            self.tracer.data_specializations.append((self.node.name, "iter"))
            return iter(sample)
        frame = inspect.currentframe()
        calling_frame = None if frame is None else frame.f_back
        if calling_frame is not None:
            instructions = list(dis.get_instructions(calling_frame.f_code))
            current = None
            for index, instruction in enumerate(instructions):
                if instruction.offset >= calling_frame.f_lasti:
                    current = instructions[index]
                    break
            if current is not None and current.opname == "UNPACK_SEQUENCE":
                return (self[index] for index in range(current.argval))
        tracer_iter = getattr(self.tracer, "iter", None)
        if callable(tracer_iter):
            return tracer_iter(self)
        raise TraceError("symbolic graph values cannot be iterated")

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.update(node=self.node, tracer=self.tracer)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        node = state.pop("node")
        tracer = state.pop("tracer")
        self.node = node
        self.tracer = tracer
        self.__dict__.update(state)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Proxy":
        if id(self) in memo:
            return memo[id(self)]
        state = self.__getstate__()
        copied: dict[str, Any] = {}
        for key, value in state.items():
            try:
                copied[key] = copy.deepcopy(value, memo)
            except Exception:
                copied[key] = copy.copy(value)
        result = type(self)(copied.pop("node"), copied.pop("tracer"))
        memo[id(self)] = result
        result.__dict__.update(copied)
        return result

    @classmethod
    def __tensorplay_function__(
        cls,
        function: Callable[..., Any],
        types_: tuple[type[Any], ...],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> "Proxy":
        del types_
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        tracers: list[Any] = []

        def find(value: Any) -> None:
            if isinstance(value, cls) and value.tracer not in tracers:
                tracers.append(value.tracer)

        for value in _iter_values(args):
            find(value)
        for value in _iter_values(kwargs):
            find(value)
        if not tracers:
            raise TraceError("a graph dispatch call has no proxy argument")
        if len(tracers) != 1:
            raise TraceError("a graph operation cannot combine different tracers")
        tracer = tracers[0]
        method_name = getattr(function, "__name__", None)
        if getattr(function, "__tensorplay_method__", False) and method_name:
            return tracer.create_proxy("call_method", method_name, args, kwargs)
        return tracer.create_proxy("call_function", function, args, kwargs)

    # -- state predicates (single read path over tracer tables) --------------
    # The side tables below are keyed by NODE NAME by design (they must
    # survive across proxies referencing one node); every read goes through
    # these predicates so relocating state onto the Proxy is a one-place edit.

    @property
    def sample(self) -> Any:
        """Concrete trace-time value behind this node, or None."""

        return self.tracer._node_samples.get(self.node.name)

    @property
    def is_symbolic_gate(self) -> bool:
        """Routed through graph.gate(): stays live-in-graph, never keyed."""

        return self.node.name in self.tracer.symbolic_gate_nodes

    def __repr__(self) -> str:
        return f"Proxy({self.node.name})"


class MetaProxy(Proxy):
    """Proxy carrying an optional eager metadata value."""

    __slots__ = ("fake_mode",)

    def __init__(self, node: Node, tracer: Any = None, fake_mode: Any = None) -> None:
        super().__init__(node, tracer)
        self.fake_mode = fake_mode

    @classmethod
    def __tensorplay_function__(
        cls,
        function: Callable[..., Any],
        types_: tuple[type[Any], ...],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> "MetaProxy":
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        result = super().__tensorplay_function__(function, types_, args, kwargs)
        meta_source = next((item for item in _iter_values(args) if isinstance(item, MetaProxy)), None)
        if meta_source is None:
            return cls(result.node, result.tracer)
        meta_args = tuple(
            item.node.meta.get("val") if isinstance(item, Proxy) else item
            for item in args
        )
        meta_kwargs = {
            key: item.node.meta.get("val") if isinstance(item, Proxy) else item
            for key, item in kwargs.items()
        }
        try:
            value = function(*meta_args, **meta_kwargs)
        except Exception:
            value = None
        if value is not None:
            result.node.meta["val"] = value
        return cls(result.node, result.tracer, meta_source.fake_mode)


class Attribute(Proxy):
    """Lazy attribute access that becomes a method or attribute node on use."""

    __slots__ = ("root", "attr", "_node")

    def __init__(self, root: Proxy, attr: str) -> None:
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Node | None = None

    @property
    def node(self) -> Node:
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args: Any, **kwargs: Any) -> Proxy:
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root, *args), kwargs
        )


class ParameterProxy(Proxy):
    """Proxy whose static metadata is read from a parameter object."""

    __slots__ = ("param", "parameter_name")

    def __init__(self, tracer: Any, node: Node, name: str, param: Any) -> None:
        super().__init__(node, tracer)
        self.param = param
        self.parameter_name = name

    def __repr__(self) -> str:
        return f"ParameterProxy({self.parameter_name})"

    @property
    def shape(self) -> Any:
        return self.param.shape

    def size(self, *args: Any) -> Any:
        return self.param.size(*args) if args else self.param.size()

    def dim(self) -> int:
        return int(self.param.dim())

    @property
    def ndim(self) -> int:
        return int(self.param.ndim)

    def numel(self) -> int:
        return int(self.param.numel())

    def nelement(self) -> int:
        return int(self.param.nelement())


def _define_operator(name: str, target: Any, reflected: bool = False) -> None:
    method_name = f"__r{name}__" if reflected else f"__{name}__"
    if hasattr(Proxy, method_name):
        return

    def impl(self: Proxy, other: Any = None) -> Proxy:
        if reflected:
            args = (other, self)
        elif other is None and name in {"neg", "pos", "invert", "abs"}:
            args = (self,)
        else:
            args = (self, other)
        return self.tracer.create_proxy("call_function", target, args, {})

    impl.__name__ = method_name
    impl.__qualname__ = f"Proxy.{method_name}"
    setattr(Proxy, method_name, impl)


_operator_names = {
    "lshift": operator.lshift,
    "rshift": operator.rshift,
    "and_": operator.and_,
    "or_": operator.or_,
    "xor": operator.xor,
    "invert": operator.invert,
}
for _name, _target in _operator_names.items():
    _define_operator(_name, _target)
for _name, _target in {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "floordiv": operator.floordiv,
    "truediv": operator.truediv,
    "mod": operator.mod,
    "pow": operator.pow,
    "lshift": operator.lshift,
    "rshift": operator.rshift,
    "and_": operator.and_,
    "or_": operator.or_,
    "xor": operator.xor,
    "matmul": operator.matmul,
}.items():
    _define_operator(_name, _target, reflected=True)
for _name, _target in {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "gt": operator.gt,
    "le": operator.le,
    "ge": operator.ge,
    "pos": operator.pos,
    "neg": operator.neg,
    "invert": operator.invert,
}.items():
    _define_operator(_name, _target)
for _name, _target in {
    "iadd": operator.iadd,
    "isub": operator.isub,
    "imul": operator.imul,
    "imatmul": operator.imatmul,
    "itruediv": operator.itruediv,
    "ifloordiv": operator.ifloordiv,
    "imod": operator.imod,
    "ipow": operator.ipow,
    "ilshift": operator.ilshift,
    "irshift": operator.irshift,
    "iand": operator.iand,
    "ixor": operator.ixor,
    "ior": operator.ior,
}.items():
    _define_operator(_name, _target)


def _setitem(self: Proxy, key: Any, value: Any) -> None:
    self.tracer.create_proxy("call_function", operator.setitem, (self, key, value), {})


Proxy.__setitem__ = _setitem


def _no_nodes_error(value: Any) -> Any:
    raise TraceError(f"graph dictionary keys cannot contain a node: {value!r}")


def _reject_node_keys(value: Any) -> None:
    if isinstance(value, (Proxy, Node)):
        _no_nodes_error(value)
    if isinstance(value, tuple | list):
        for item in value:
            _reject_node_keys(item)
    elif isinstance(value, Mapping):
        for key in value:
            _reject_node_keys(key)
    elif isinstance(value, slice):
        _reject_node_keys(value.start)
        _reject_node_keys(value.stop)
        _reject_node_keys(value.step)


def _is_tp_tensor(value: Any) -> bool:
    return type(value).__module__.startswith("tensorplay") and hasattr(
        value, "shape"
    )

def gate(source: Any) -> Any:
    """Mark a traced scalar as unspecialized and keep it a tensor proxy.

    A 1-element tensor proxy uses explicit conversion to a real
    Python number happens only through explicit ``int()``/``float()``
    (which specialize+bake), never implicitly.

    Inside ``tensorplay.compile`` capture::

        n = tp.graph.gate(x.sum())
        return x * n       # tensor broadcast; ONE specialization for any sum
        if n > 3: ...      # branch outcome joins the cache key

    Outside capture this raises: gates are a compile-time concept.
    """
    if not capturing():
        raise GraphCaptureError(
            "graph.gate() is only valid inside tensorplay.compile capture"
        )
    if isinstance(source, Proxy):
        sample = source._sample()
        if sample is None:
            raise GraphCaptureError(
                "graph.gate() needs an execute-mode sample for this node"
            )
        source.tracer.symbolic_gate_nodes.add(source.node.name)
        # UPV semantics: return the tensor proxy itself, unwrapped only by
        # explicit int()/float().
        return source
    raise TypeError(
        f"graph.gate() expects a traced tensor value, got {type(source)!r}"
    )

__all__ = [
    "Attribute",
    "GraphAppendingTracer",
    "MetaProxy",
    "ParameterProxy",
    "Proxy",
    "Scope",
    "ScopeContextManager",
    "TraceError",
    "TracerBase",
    "assert_fn",
    "gate",
]
