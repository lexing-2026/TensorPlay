from __future__ import annotations

import itertools
import operator
from typing import Any

import tensorplay as tp

from ..graph_module import GraphModule
from ..node import Node
from ..passes.tools_common import legalize_graph
from ..proxy import Proxy
from ..symbolic_trace import symbolic_trace

__all__ = ["are_nodes_independent", "may_depend_on", "merge_matmul", "split_result_tensors"]


def split_result_tensors(result: Any, inputs: list[Any]) -> tuple[Any, ...]:
    """Split a merged matrix result at the original leading dimensions."""

    if isinstance(result, Proxy):
        return result
    sizes = [_shape(value)[0] for value in inputs]
    return tuple(tp.split(result, sizes, dim=0))


def _shape(value: Any) -> tuple[Any, ...]:
    shape = getattr(value, "shape", ())
    if callable(shape):
        shape = shape()
    return tuple(shape)


def may_depend_on(a: Node, b: Node, search_depth: int = 6) -> bool:
    if a is b:
        return True
    if not a.all_input_nodes:
        return False
    if search_depth <= 0:
        return True
    return any(may_depend_on(value, b, search_depth - 1) for value in a.all_input_nodes)


def are_nodes_independent(nodes: list[Node]) -> bool:
    return all(
        not may_depend_on(left, right) and not may_depend_on(right, left)
        for left, right in itertools.combinations(nodes, 2)
    )


def merge_matmul(in_mod: Any) -> GraphModule:
    """Merge independent matrix products that share their right operand."""

    graph_module = in_mod if isinstance(in_mod, GraphModule) else symbolic_trace(in_mod)
    targets = {tp.matmul, operator.matmul}
    rhs_users: dict[Any, list[Node]] = {}
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target not in targets or len(node.args) != 2:
            continue
        lhs, rhs = node.args
        lhs_key = lhs.target if isinstance(lhs, Node) and lhs.op == "get_attr" else lhs
        rhs_key = rhs.target if isinstance(rhs, Node) and rhs.op == "get_attr" else rhs
        del lhs_key
        rhs_users.setdefault(rhs_key, []).append(node)

    output = graph_module.graph.outputs[0]
    for rhs_key, products in rhs_users.items():
        if len(products) < 2 or not are_nodes_independent(products):
            continue
        lhs_values = [node.args[0] for node in products]
        lhs = [
            graph_module.graph.get_attr(value) if isinstance(value, str) else value
            for value in lhs_values
        ]
        rhs = graph_module.graph.get_attr(rhs_key) if isinstance(rhs_key, str) else rhs_key
        with graph_module.graph.inserting_before(output):
            merged_lhs = graph_module.graph.call_function(tp.cat, (lhs,), {"dim": 0})
            merged = graph_module.graph.call_function(tp.matmul, (merged_lhs, rhs), {})
            split = graph_module.graph.call_function(split_result_tensors, (merged, lhs), {})
            replacements = [
                graph_module.graph.call_function(operator.getitem, (split, index), {})
                for index in range(len(products))
            ]
        for old, new in zip(products, replacements):
            old.replace_all_uses_with(new)
            graph_module.graph.erase_node(old)
        legalize_graph(graph_module)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module
