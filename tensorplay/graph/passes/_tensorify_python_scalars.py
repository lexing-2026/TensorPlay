"""Convert scalar literals used by tensor operations into tensor values."""

from __future__ import annotations

from typing import Any

from .._utils import _iter_nodes
from ..graph_module import GraphModule
from ..node import Node

__all__ = ["tensorify_python_scalars"]


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, complex))


def _tensorify_impl(value: Any, factory) -> Any:
    def convert(item: Any) -> Any:
        if _is_scalar(item):
            return factory(item)
        if isinstance(item, tuple):
            return tuple(convert(child) for child in item)
        if isinstance(item, list):
            return [convert(child) for child in item]
        if isinstance(item, dict):
            return {key: convert(child) for key, child in item.items()}
        if isinstance(item, slice):
            return slice(convert(item.start), convert(item.stop), convert(item.step))
        return item

    return _map_arg(value, convert)


def tensorify_python_scalars(
    graph_module: GraphModule,
    scalar_types: tuple[type[Any], ...] = (int, float),
) -> GraphModule:
    """Rewrite scalar arguments of tensor-producing calls in place."""

    import tensorplay

    def factory(value: Any) -> Any:
        try:
            return tensorplay.tensor(value)
        except Exception:
            return value

    for node in graph_module.graph.nodes:
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        inputs = tuple(_iter_nodes(node.args))
        if not inputs or not any(
            isinstance(item, scalar_types)
            for item in (*node.args, *node.kwargs.values())
        ):
            continue
        node.args = _tensorify_impl(node.args, factory)
        node.kwargs = _tensorify_impl(node.kwargs, factory)
    graph_module.graph.lint()
    return graph_module
