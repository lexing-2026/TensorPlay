"""Dispatcher-level graph capture.

A :class:`ProxyTensorDispatchMode` sits on the dispatch-mode stack while the
traced callable runs on real tensors.  Every operator overload that reaches
the dispatcher -- including the ones the autograd engine runs inside
``tensorplay.autograd.grad`` -- is executed and recorded as one
``call_function`` node whose target is the :class:`~tensorplay._ops.OpOverload`.
The result is a graph over the op contract, independent of the Python code
that produced it (control flow is resolved by the concrete run).

Tensors map to graph values by tensor identity; the trace keeps every
tracked tensor alive, so an identity is never reused while it is mapped.
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

    def __init__(self) -> None:
        self.graph = Graph()
        self.root = _ConstantHolder()
        # impl identity -> (tensor kept alive, node producing it)
        self._tracked: dict[int, tuple[Any, Node]] = {}
        self._constant_count = 0

    # -- tensor tracking ----------------------------------------------------

    def track(self, tensor: Any, node: Node) -> None:
        self._tracked[tensor._impl_id] = (tensor, node)
        node.meta["val"] = tensor
        node.meta["tensor_meta"] = _tensor_meta(tensor)

    def node_for(self, tensor: Any) -> Node:
        entry = self._tracked.get(tensor._impl_id)
        if entry is not None:
            return entry[1]
        return self._constant(tensor)

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

    # -- operator nodes -----------------------------------------------------

    def record(self, func: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], out: Any) -> Node:
        node = self.graph.create_node(
            "call_function",
            func,
            self.map_value(tuple(args)),
            self.map_value(dict(kwargs)),
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

    def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        decomposition = self.decomposition_table.get(func)
        if decomposition is not None:
            # The decomposition's own operators are recorded instead of func.
            with self:
                return decomposition(*args, **kwargs)
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
