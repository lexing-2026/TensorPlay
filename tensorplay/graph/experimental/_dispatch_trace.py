"""Dispatcher-level graph capture.

A :class:`ProxyTensorDispatchMode` sits on the dispatch-mode stack while the
traced callable runs on real tensors.  Every operator overload that reaches
the dispatcher -- including the ones the autograd engine runs inside
``tensorplay.autograd.grad`` -- is executed and recorded as one
``call_function`` node whose target is the :class:`~tensorplay._ops.OpOverload`.
The result is a graph over the op contract, independent of the Python code
that produced it (control flow is resolved by the concrete run).

Tensors map to graph values by tensor identity. A trace may retain concrete
values or record only metadata with non-owning implementation references.
Tensors the traced code did not receive as inputs and did not compute (module
parameters, captured globals) become ``get_attr`` constants of the module.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any

import tensorplay
from tensorplay.utils._dispatch import TensorPlayDispatchMode

from ..graph import Graph, _PyTreeCodeGen, _PyTreeInfo
from ..graph_module import GraphModule
from ..node import Node
from .._pytree import tree_flatten, tree_unflatten

__all__ = ["DispatchTracer", "ProxyTensorDispatchMode", "dispatch_make_graph"]


def _is_tensor(value: Any) -> bool:
    return isinstance(value, tensorplay.Tensor)


class _ConstantHolder(tensorplay.nn.Module):
    """Root module carrying the tensor constants of a traced graph."""


class DispatchTracer:
    """Graph under construction plus the tensor-to-node map."""

    #: Operations run on the values they are given and are recorded alongside;
    #: an operator that would hand back a stand-in hands back its result.
    records_real_values = True

    def __init__(self, *, record_values: bool = True) -> None:
        self.graph = Graph()
        self.root = _ConstantHolder()
        self.record_values = record_values
        # impl identity -> (tensor or non-owning implementation reference, node)
        self._tracked: dict[int, tuple[Any, Node]] = {}
        self._constant_count = 0
        self._proxy_mode: Any = None
        # graph identity -> the get_attr node that reads it off the root
        self._subgraphs: dict[int, Node] = {}

    @property
    def proxy_mode(self) -> Any:
        """The state a decomposition helper reads while this trace runs.

        An operator that a trace is meant to keep whole -- one that stands for a
        region of the program rather than a single operation -- decides whether
        to record itself as a node by asking whether a proxy trace is currently
        recording.  That question is answered by what is on the dispatch stack,
        so a trace that is entered directly, without going through the entry
        point that sets this up, would otherwise leave those operators unable to
        tell that they are being traced and they would run themselves instead.

        Built on first use, so a tracer that never runs a decomposition keeps no
        extra state.  The table starts empty because the dispatcher mode holds
        the decompositions that were asked for; this one is what a helper that
        enables its own set reads and writes.
        """

        if self._proxy_mode is None:
            from .proxy_tensor import ProxyMode

            self._proxy_mode = ProxyMode(self)
        return self._proxy_mode

    # -- tensor tracking ----------------------------------------------------

    def track(self, tensor: Any, node: Node) -> None:
        reference = tensor if self.record_values else tensorplay._C._WeakTensorRef(tensor)
        self._tracked[tensor._impl_id] = (reference, node)
        node.meta["val"] = self.recorded_value(tensor)
        node.meta["tensor_meta"] = _tensor_meta(tensor)

    def recorded_value(self, value: Any) -> Any:
        if self.record_values:
            return value
        from tensorplay.compiler._core.api import _RecordedTensorMetadata
        from ..node import map_aggregate

        return map_aggregate(
            value, lambda item: _RecordedTensorMetadata(item) if _is_tensor(item) else item
        )

    def _producer(self, tensor: Any) -> Node | None:
        entry = self._tracked.get(tensor._impl_id)
        if entry is None:
            return None
        if not self.record_values and entry[0].expired():
            self._tracked.pop(tensor._impl_id, None)
            return None
        return entry[1]

    def node_for(self, tensor: Any) -> Node:
        node = self._producer(tensor)
        if node is not None:
            return node
        return self._constant(tensor)

    def producer(self, tensor: Any) -> Node | None:
        """The node this tensor currently stands for, if the trace computed it.

        Unlike :meth:`node_for`, a tensor the trace has not seen is not turned
        into a constant: asking where a value came from must not add to the
        graph.
        """

        return self._producer(tensor)

    def _subgraph(self, module: GraphModule) -> Node:
        """A graph handed to an operation, read off the module that holds it.

        An operator that takes a function -- a branch, a loop body, a score
        modifier -- is handed its traced graph.  The graph is kept on the root
        and read by name, so a lowering sees a region of its own rather than an
        object it would have to recognise, and a graph handed twice is one
        submodule.
        """

        node = self._subgraphs.get(id(module))
        if node is None:
            name = f"_subgraph{len(self._subgraphs)}"
            setattr(self.root, name, module)
            node = self.graph.get_attr(name)
            node.meta["is_subgraph"] = True
            self._subgraphs[id(module)] = node
        return node

    def _constant(self, tensor: Any) -> Node:
        name = f"_tensor_constant{self._constant_count}"
        self._constant_count += 1
        setattr(self.root, name, tensor)
        node = self.graph.get_attr(name)
        self.track(tensor, node)
        return node

    # -- arguments ----------------------------------------------------------

    def map_value(self, value: Any) -> Any:
        if _is_tensor(value):
            return self.node_for(value)
        if isinstance(value, tuple):
            return tuple(self.map_value(item) for item in value)
        if isinstance(value, list):
            return [self.map_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.map_value(item) for key, item in value.items()}
        return value

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
            node = self._producer(value)
            return node if node is not None else value
        if isinstance(value, GraphModule):
            return self._subgraph(value)
        if isinstance(value, tuple):
            return tuple(self.map_operands(item) for item in value)
        if isinstance(value, list):
            return [self.map_operands(item) for item in value]
        if isinstance(value, dict):
            return {key: self.map_operands(item) for key, item in value.items()}
        return value

    # -- operator nodes -----------------------------------------------------

    def record(self, func: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], out: Any) -> Node:
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
            node.meta["val"] = self.recorded_value(out)
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
                    element.meta["val"] = self.recorded_value(item)
                    for inner_index, inner in enumerate(item):
                        if _is_tensor(inner):
                            leaf = self.graph.create_node(
                                "call_function", operator.getitem, (element, inner_index), {}
                            )
                            self.track(inner, leaf)
        else:
            node.meta["val"] = out
        return node


def _tensor_meta(tensor: Any) -> dict[str, Any]:
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "dtype": tensor.dtype,
        "device": tensor.device,
        "requires_grad": bool(tensor.requires_grad),
    }


class ProxyTensorDispatchMode(TensorPlayDispatchMode):
    """Records every dispatched operator into a :class:`DispatchTracer`."""

    def __init__(
        self,
        tracer: DispatchTracer,
        decomposition_table: Mapping[Any, Callable[..., Any]] | None = None,
    ) -> None:
        super().__init__()
        self.tracer = tracer
        self.decomposition_table = dict(decomposition_table or {})

    @classmethod
    def is_infra_mode(cls) -> bool:
        return True

    def __enter__(self) -> "ProxyTensorDispatchMode":
        super().__enter__()
        # The dispatch stack is what an operator consults to decide whether it
        # is being recorded, and the state a decomposition helper reads is a
        # separate thing that has to be current for the same span.  Entering
        # both together is what makes the two agree; entering only the stack
        # leaves an operator that keeps itself whole unable to tell it is being
        # traced, and it runs itself instead of becoming a node.
        #
        # The enter is re-entrant because recording a decomposed call steps
        # inside this same mode to run the decomposition without recording it,
        # so the depth of the stack and the number of times the state was
        # entered have to be kept together.  Tokens are kept on a stack of
        # their own for the same reason: a context that is entered twice has to
        # be unwound in the reverse order, and the innermost token is the one
        # that belongs to the innermost exit.
        if not hasattr(self, "_proxy_mode_tokens"):
            self._proxy_mode_tokens: list[Any] = []
        self._proxy_mode_tokens.append(self.tracer.proxy_mode.__enter__())
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        tokens = getattr(self, "_proxy_mode_tokens", None)
        if tokens:
            tokens.pop()
            self.tracer.proxy_mode.__exit__(exc_type, exc_value, traceback)
        super().__exit__(exc_type, exc_value, traceback)

    def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        decomposition = self.decomposition_table.get(func)
        if decomposition is not None:
            # The decomposition's own operators are recorded instead of func,
            # unless it declines this call, which is then recorded as it is.
            with self:
                out = decomposition(*args, **kwargs)
            if out is not NotImplemented:
                return out
        out = func(*args, **kwargs)
        self.tracer.record(func, args, kwargs, out)
        return out


def dispatch_make_graph(
    f: Callable[..., Any],
    decomposition_table: Mapping[Any, Callable[..., Any]] | None = None,
) -> Callable[..., GraphModule]:
    """Return a callable that traces ``f`` at dispatcher level on real inputs.

    The traced module accepts and returns the same nested structures as
    ``f``; tensors inside the inputs become placeholders.
    """

    def wrapped(*args: Any) -> GraphModule:
        tracer = DispatchTracer()
        flat_args, in_spec = tree_flatten(args)
        names = []
        for index, value in enumerate(flat_args):
            name = f"arg{index}_1"
            names.append(name)
            node = tracer.graph.placeholder(name)
            if _is_tensor(value):
                tracer.track(value, node)
            else:
                node.meta["val"] = value
        with ProxyTensorDispatchMode(tracer, decomposition_table):
            out = f(*tree_unflatten(flat_args, in_spec))
        flat_out, out_spec = tree_flatten(out)
        tracer.graph.output(tuple(tracer.map_value(value) for value in flat_out))
        tracer.graph._codegen = _PyTreeCodeGen(_PyTreeInfo(names, in_spec, out_spec))
        graph_module = GraphModule(tracer.root, tracer.graph)
        graph_module.meta["tracing_mode"] = "real"
        return graph_module

    return wrapped
