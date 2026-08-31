"""Shared graph inspection, fusion grouping, and ordering algorithms."""

from __future__ import annotations

import collections
import heapq
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._utils import _iter_nodes
from ..graph import Graph
from ..graph_module import GraphModule
from ..node import Node

CALLABLE_NODE_OPS = {"call_module", "call_function", "call_method"}
NodeList = list[Node]
NodeSet = set[Node]
Names = list[str]
Tensors = tuple[Any, ...] | list[Any]
TensorOrTensors = Any

__all__ = [
    "CALLABLE_NODE_OPS",
    "GraphAccFusionsFinder",
    "get_acc_ops_name",
    "get_node_target",
    "is_node_output_tensor",
    "legalize_graph",
    "stable_topological_sort",
]


def get_acc_ops_name(value: str | type[Any] | Any) -> str:
    if isinstance(value, str):
        return value
    module = getattr(value, "__module__", "") or ""
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if name is None:
        name = type(value).__qualname__
        module = type(value).__module__
    return f"{module}.{name}" if module else str(name)


def get_node_target(submodules: Mapping[str, Any], node: Node) -> str:
    if node.op not in CALLABLE_NODE_OPS:
        raise AssertionError(
            f"expected a callable node, received operation {node.op!r}"
        )
    if node.op == "call_method":
        if not isinstance(node.target, str):
            raise TypeError("method targets must be strings")
        return node.target
    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise TypeError("module targets must be strings")
        return get_acc_ops_name(submodules[node.target])
    return get_acc_ops_name(node.target)


def is_node_output_tensor(node: Node) -> bool:
    metadata = node.meta.get("tensor_meta")
    if metadata is not None:
        return hasattr(metadata, "shape")
    value = node.meta.get("val")
    if value is not None:
        return hasattr(value, "shape")
    type_value = node.meta.get("type", node.type)
    if type_value is None:
        return False
    if type_value in {type(None), bool, int, float, complex, str, bytes, tuple, list, dict}:
        return False
    return bool(
        getattr(type_value, "__module__", "").startswith("tensorplay")
        and "Tensor" in getattr(type_value, "__name__", "")
    )


class GraphAccFusionsFinder:
    """Find connected callable regions that exchange non-tensor values."""

    @dataclass
    class FusionGroup:
        top_node_idx: int
        nodes: NodeSet
        inputs: NodeSet
        nodes_need_process: NodeSet

        def add_node(self, node: Node) -> None:
            if node in self.nodes:
                return
            self.nodes_need_process.add(node)
            self.nodes.add(node)
            self.inputs.discard(node)
            self.inputs.update(
                input_node
                for input_node in _iter_nodes(node.args)
                if input_node.op in CALLABLE_NODE_OPS and input_node not in self.nodes
            )
            self.inputs.update(
                input_node
                for input_node in _iter_nodes(node.kwargs)
                if input_node.op in CALLABLE_NODE_OPS and input_node not in self.nodes
            )

    def __init__(self, module: GraphModule, acc_nodes: NodeSet) -> None:
        self.module = module
        self.nodes = list(module.graph.nodes)
        self.acc_nodes = acc_nodes
        self.node_index = {node: index for index, node in enumerate(self.nodes)}

    def recursive_add_node(
        self,
        fusion_group: "GraphAccFusionsFinder.FusionGroup",
        inputs: NodeSet | NodeList,
        visited: NodeSet | None = None,
    ) -> bool:
        for arg in inputs:
            if visited is not None:
                if arg in visited:
                    continue
                visited.add(arg)
            if arg.op not in CALLABLE_NODE_OPS:
                continue
            if self.node_index[arg] < fusion_group.top_node_idx:
                continue
            if arg in fusion_group.nodes:
                return True
            upstream = [
                *(_iter_nodes(arg.args)),
                *(_iter_nodes(arg.kwargs)),
            ]
            if self.recursive_add_node(fusion_group, upstream, visited):
                fusion_group.add_node(arg)
                return True
        return False

    def __call__(self) -> dict[Node, NodeSet]:
        result: dict[Node, NodeSet] = {}
        for seed in list(self.acc_nodes):
            if seed in result or seed.op not in CALLABLE_NODE_OPS:
                continue
            if "tensor_meta" in seed.meta or seed not in self.acc_nodes:
                continue
            group = self.FusionGroup(
                top_node_idx=self.node_index[seed],
                nodes={seed},
                inputs={*(_iter_nodes(seed.args)), *(_iter_nodes(seed.kwargs))},
                nodes_need_process={seed},
            )
            while group.nodes_need_process:
                current = group.nodes_need_process.pop()
                self.recursive_add_node(group, group.inputs, visited=set())
                if "tensor_meta" not in current.meta:
                    for user in current.users:
                        if user.op not in CALLABLE_NODE_OPS or user in group.nodes:
                            continue
                        group.add_node(user)
                        self.recursive_add_node(group, group.inputs, visited=set())
                for arg in [*(_iter_nodes(current.args)), *(_iter_nodes(current.kwargs))]:
                    if arg.op not in CALLABLE_NODE_OPS:
                        continue
                    if "tensor_meta" in arg.meta or arg in group.nodes:
                        continue
                    group.add_node(arg)
                    group.top_node_idx = min(group.top_node_idx, self.node_index[arg])
                    self.recursive_add_node(group, group.inputs, visited=set())
            if set(group.nodes) <= self.acc_nodes:
                for node in group.nodes:
                    result[node] = group.nodes
            else:
                self.acc_nodes.difference_update(group.nodes)
        return result


def _rebuild_graph_module(
    graph_module: GraphModule, order: NodeList
) -> GraphModule:
    old_graph = graph_module.graph
    new_graph = Graph()
    mapping: dict[Node, Node] = {}
    for node in order:
        if node.op == "output":
            continue
        mapping[node] = new_graph.node_copy(node, lambda value: mapping[value])
    output = old_graph.output_node
    new_graph.output(_remap_value(output.args[0], mapping))
    new_graph.lint()
    graph_module.graph = new_graph
    graph_module._compiled_forward = None
    return graph_module


def _remap_value(value: Any, mapping: Mapping[Node, Node]) -> Any:
    if isinstance(value, Node):
        return mapping[value]
    if isinstance(value, tuple):
        return tuple(_remap_value(item, mapping) for item in value)
    if isinstance(value, list):
        return [_remap_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _remap_value(item, mapping) for key, item in value.items()}
    if isinstance(value, slice):
        return slice(
            _remap_value(value.start, mapping),
            _remap_value(value.stop, mapping),
            _remap_value(value.step, mapping),
        )
    return value


def _collect_order(graph: Graph, stable: bool) -> NodeList:
    result: NodeList = []
    nodes = list(graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}
    indegree = {
        node: sum(
            1
            for value in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs))
            if value in position
        )
        for node in nodes
    }
    if stable:
        ready = [(position[node], node) for node in nodes if indegree[node] == 0]
        heapq.heapify(ready)
        while ready:
            _, node = heapq.heappop(ready)
            result.append(node)
            for user in node.users:
                if user not in indegree:
                    continue
                indegree[user] -= 1
                if indegree[user] == 0:
                    heapq.heappush(ready, (position[user], user))
    else:
        ready_queue = collections.deque(node for node in nodes if indegree[node] == 0)
        while ready_queue:
            node = ready_queue.popleft()
            result.append(node)
            for user in node.users:
                if user not in indegree:
                    continue
                indegree[user] -= 1
                if indegree[user] == 0:
                    ready_queue.append(user)
    if len(result) != len(nodes):
        remaining = [node.name for node, count in indegree.items() if count]
        raise RuntimeError(f"graph contains a dependency cycle: {remaining}")
    return result


def stable_topological_sort(value: GraphModule | NodeList):
    """Stable-sort either a graph module in place or a node subset."""

    if isinstance(value, GraphModule):
        return _rebuild_graph_module(value, _collect_order(value.graph, True))
    return _collect_order(GraphFromNodes(value), True)


def legalize_graph(graph_module: GraphModule) -> GraphModule:
    """Rebuild a graph in dependency order and validate its users."""

    return _rebuild_graph_module(
        graph_module, _collect_order(graph_module.graph, False)
    )


class GraphFromNodes(Graph):
    """Read-only graph view used for the node-list ordering API."""

    def __init__(self, nodes: NodeList) -> None:
        super().__init__()
        self.nodes = list(nodes)
