"""Common subgraph construction and structural comparison helpers."""

from __future__ import annotations

from typing import Any

from ...graph import Graph
from ...graph_module import GraphModule
from ...node import Node
from ..base import _as_graph_module

__all__ = ["HolderModule", "compare_graphs", "lift_subgraph_as_module"]


class HolderModule:
    """Attribute holder for a lifted graph component."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        for key, value in (values or {}).items():
            setattr(self, key, value)


def lift_subgraph_as_module(
    gm: GraphModule,
    subgraph: Graph,
    comp_name: str = "",
    class_name: str = "GraphModule",
) -> tuple[GraphModule, dict[str, str]]:
    del class_name
    holder = HolderModule()
    mapping: dict[str, str] = {}
    for node in subgraph.nodes:
        if node.op not in {"get_attr", "call_module"}:
            continue
        if gm.root is None:
            continue
        try:
            value = gm._get_attr(node.target)
        except AttributeError as exc:
            raise RuntimeError(f"missing graph attribute {node.target!r}") from exc
        setattr(holder, node.target.replace(".", "_"), value)
        mapping[node.target] = f"{comp_name}.{node.target}".strip(".")
    signature = getattr(gm, "signature", None)
    return GraphModule(holder, subgraph, signature), mapping


def compare_graphs(left: Graph, right: Graph) -> bool:
    if len(left.nodes) != len(right.nodes):
        return False
    mapping: dict[Node, Node] = {}
    for a, b in zip(left.nodes, right.nodes):
        if a.op != b.op or a.target != b.target:
            return False
        if a.op == "placeholder" or a.op == "output":
            continue
        a_inputs = [node.name for node in _nodes(a.args)]
        b_inputs = [node.name for node in _nodes(b.args)]
        if len(a_inputs) != len(b_inputs):
            return False
        mapping[a] = b
    return True


def _nodes(value: Any):
    if isinstance(value, Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nodes(item)
