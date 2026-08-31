"""Graph partition validation and native submodule fusion helpers."""

from __future__ import annotations

import heapq
import operator
from typing import Any

from ..._utils import _iter_nodes
from ...graph import Graph
from ...graph_module import GraphModule
from ...node import Node

NodeList = list[Node]
NodeSet = set[Node]

__all__ = [
    "erase_nodes",
    "fuse_as_graphmodule",
    "fuse_by_partitions",
    "insert_subgm",
    "legalize_graph",
    "topo_sort",
    "validate_partition",
]


def topo_sort(nodes: NodeList) -> NodeList:
    """Return a stable topological ordering for a node subset."""

    position = {node: index for index, node in enumerate(nodes)}
    indegree = {node: 0 for node in nodes}
    allowed = set(nodes)
    for node in nodes:
        indegree[node] = sum(
            1 for value in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs))
            if value in allowed
        )
    ready = [(position[node], node) for node in nodes if indegree[node] == 0]
    heapq.heapify(ready)
    result: NodeList = []
    while ready:
        _, node = heapq.heappop(ready)
        result.append(node)
        for user in node.users:
            if user not in indegree:
                continue
            indegree[user] -= 1
            if indegree[user] == 0:
                heapq.heappush(ready, (position[user], user))
    if len(result) != len(nodes):
        raise AssertionError("partition contains a dependency cycle")
    return result


def validate_partition(partition: NodeList) -> bool:
    """Return whether a partition can be isolated without a dependency cycle."""

    members = set(partition)
    boundary_users = [
        user
        for node in members
        for user in node.users
        if user not in members
    ]
    visited: set[Node] = set()
    queue = list(boundary_users)
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in members:
            return False
        queue.extend(user for user in current.users if user not in visited)
    return True


def _attach_submodule(root: Any, name: str, value: Any) -> str:
    if root is None:
        raise RuntimeError("cannot attach a fused module without a graph root")
    candidate = name
    suffix = 0
    while hasattr(root, candidate):
        suffix += 1
        candidate = f"{name}_{suffix}"
    setattr(root, candidate, value)
    return candidate


def _make_graph_module(root: Any, graph: Graph, signature: Any) -> GraphModule:
    return GraphModule(root, graph, signature)


def fuse_as_graphmodule(
    gm: GraphModule,
    nodes: NodeList,
    module_name: str,
    partition_lookup_table: dict[Node, int | None] | None = None,
    *,
    always_return_tuple: bool = False,
) -> tuple[GraphModule, tuple[Node, ...], tuple[Node, ...]]:
    """Copy a partition into a graph module and expose its boundaries."""

    if not nodes:
        raise ValueError("cannot fuse an empty partition")
    for node in nodes:
        if node.graph is not gm.graph:
            raise AssertionError(f"{node.name!r} does not belong to the graph module")
    if not validate_partition(nodes):
        raise AssertionError("invalid partition, found dependency cycle")
    lookup = partition_lookup_table or dict.fromkeys(nodes)
    ordered = topo_sort(nodes)
    subgraph = Graph()
    node_to_placeholder: dict[Node, Node] = {}
    node_map: dict[Node, Node] = {}

    external_inputs: list[Node] = []
    external_set: set[Node] = set()
    for node in ordered:
        for input_node in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
            if input_node not in lookup and input_node not in external_set:
                external_inputs.append(input_node)
                external_set.add(input_node)
    if external_inputs and all(node.op == "placeholder" for node in external_inputs):
        graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
        external_inputs.sort(key=graph_order.__getitem__)

    for input_node in external_inputs:
        placeholder = subgraph.placeholder(input_node.name)
        placeholder.meta = dict(input_node.meta)
        placeholder.type = input_node.type
        placeholder.tag = input_node.tag
        node_to_placeholder[input_node] = placeholder

    def remap(value: Node) -> Node:
        if value in lookup:
            return node_map[value]
        if value not in node_to_placeholder:
            raise RuntimeError(f"missing partition input {value.name!r}")
        return node_to_placeholder[value]

    for node in ordered:
        node_map[node] = subgraph.node_copy(node, remap)

    output_mapping: dict[Node, Node] = {}
    for node in ordered:
        if any(user not in lookup for user in node.users):
            output_mapping[node] = node_map[node]
    original_outputs = tuple(output_mapping)
    outputs = tuple(output_mapping.values())
    if always_return_tuple:
        subgraph.output(outputs)
    else:
        subgraph.output(outputs[0] if len(outputs) == 1 else outputs)
    subgraph.lint()
    fused = _make_graph_module(gm.root, subgraph, None)
    fused.meta["fused_name"] = module_name
    fused.meta["original_inputs"] = tuple(node_to_placeholder)
    return fused, tuple(node_to_placeholder), original_outputs


def insert_subgm(
    gm: GraphModule,
    sub_gm: GraphModule,
    orig_inputs: tuple[Node, ...],
    orig_outputs: tuple[Node, ...],
    insertion_point: Node | None = None,
) -> GraphModule:
    """Insert a fused module call and redirect all partition consumers."""

    name = _attach_submodule(gm.root, sub_gm.meta.get("fused_name", "fused"), sub_gm)
    if insertion_point is None:
        if not orig_outputs:
            raise AssertionError("an insertion point is required for an output-free partition")
        insertion_point = orig_outputs[-1]
    with gm.graph.inserting_after(insertion_point):
        module_node = gm.graph.call_module(name, args=orig_inputs)
    if len(orig_outputs) == 1 and not (
        sub_gm.graph.output_node.args
        and isinstance(sub_gm.graph.output_node.args[0], tuple)
    ):
        orig_outputs[0].replace_all_uses_with(module_node)
    else:
        replacements: list[Node] = []
        with gm.graph.inserting_after(module_node):
            for index, original in enumerate(orig_outputs):
                replacement = gm.graph.call_function(operator.getitem, (module_node, index))
                replacements.append(replacement)
                original.replace_all_uses_with(replacement)
        module_node.meta["val"] = tuple(
            original.meta.get("val") for original in orig_outputs
        )
    return gm


def erase_nodes(gm: GraphModule, nodes: NodeList) -> None:
    """Erase a partition after all external consumers have been redirected."""

    for node in reversed(nodes):
        if node.graph is not None:
            node.erase_node()


def fuse_by_partitions(
    gm: GraphModule,
    partitions: list[dict[Node, int | None] | NodeList],
    prefix: str = "fused_",
    always_return_tuple: bool = False,
) -> GraphModule:
    """Fuse each partition into a child module and update the parent graph."""

    for partition_id, partition in enumerate(partitions):
        lookup = partition if isinstance(partition, dict) else dict.fromkeys(partition)
        nodes = topo_sort(list(lookup))
        if not nodes:
            continue
        name = f"{prefix}{partition_id}"
        sub_gm, original_inputs, original_outputs = fuse_as_graphmodule(
            gm,
            nodes,
            name,
            lookup,
            always_return_tuple=always_return_tuple,
        )
        insert_subgm(gm, sub_gm, original_inputs, original_outputs, nodes[-1])
        erase_nodes(gm, nodes)
    gm.graph.lint()
    gm.recompile()
    return gm


def legalize_graph(gm: GraphModule) -> GraphModule:
    """Validate and recompile a graph after structural mutation."""

    gm.graph.lint()
    gm.recompile()
    return gm
