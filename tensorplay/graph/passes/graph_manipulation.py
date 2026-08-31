"""Graph size accounting and target replacement helpers."""

from __future__ import annotations

from typing import Any, NamedTuple

from .._utils import _iter_nodes, _map_arg
from ..graph import Graph
from ..graph_module import GraphModule
from ..node import Node

__all__ = [
    "get_size_of_all_nodes",
    "get_size_of_node",
    "get_tensor_meta",
    "replace_target_nodes_with",
    "size_bytes",
]


def replace_target_nodes_with(
    graph_module: GraphModule,
    old_op: str,
    old_target: Any,
    new_op: str,
    new_target: Any,
) -> None:
    """Replace matching operation/target pairs while preserving node names."""

    graph = graph_module.graph
    new_graph = Graph()
    value_map: dict[Node, Node] = {}
    for node in graph.nodes:
        args = _map_arg(node.args, lambda item: value_map[item])
        kwargs = _map_arg(node.kwargs, lambda item: value_map[item])
        if node.op == old_op and node.target is old_target:
            copied = new_graph.create_node(new_op, new_target, args, kwargs, name=node.name)
        else:
            copied = new_graph.create_node(node.op, node.target, args, kwargs, name=node.name)
        copied.meta.update(node.meta)
        copied.type = node.type
        value_map[node] = copied
    graph_module.graph = new_graph


class size_bytes(NamedTuple):
    output_size: int
    total_size: int


def get_tensor_meta(node: Node) -> Any:
    return node.meta.get("tensor_meta", node.meta.get("val"))


def _numel(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = getattr(value, "shape", ()) if value is not None else ()
    if callable(shape):
        shape = shape()
    try:
        result = 1
        for dim in shape:
            result *= int(dim)
        return result
    except (TypeError, ValueError):
        return 0


def _element_size(value: Any) -> int:
    itemsize = getattr(value, "element_size", None)
    if callable(itemsize):
        try:
            return int(itemsize())
        except (TypeError, ValueError):
            pass
    return 8


def get_size_of_node(graph_module: GraphModule, node: Node) -> size_bytes:
    output_value = node.meta.get("val")
    output_size = _numel(output_value) * _element_size(output_value)
    total_size = output_size
    if node.op == "call_module" and graph_module.root is not None:
        try:
            module = graph_module._get_attr(node.target)
        except AttributeError:
            module = None
        if module is not None:
            for parameter in getattr(module, "parameters", lambda: ())():
                total_size += _numel(parameter) * _element_size(parameter)
    return size_bytes(output_size, total_size)


def get_size_of_all_nodes(graph_module: GraphModule) -> dict[Node, size_bytes]:
    sizes = {}
    for node in graph_module.graph.nodes:
        if node.op == "output":
            break
        sizes[node] = get_size_of_node(graph_module, node)
        node.size_bytes = sizes[node]
    return sizes
