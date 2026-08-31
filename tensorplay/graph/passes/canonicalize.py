"""Deterministic ordering and naming for graph nodes."""

from __future__ import annotations

import heapq
import operator
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .._utils import _iter_nodes, _target_to_str
from ..graph import Graph
from ..node import Node

__all__ = ["canonicalize_graph", "rename_nodes_to_canonical"]

_IN_PLACE_OPERATORS = frozenset(
    {
        "iadd", "iand", "iconcat", "ifloordiv", "ilshift", "imatmul",
        "imod", "imul", "ior", "ipow", "irshift", "isub", "itruediv", "ixor",
    }
)


def _computation_node_key(node: Node, canonical_idx: dict[Node, int]) -> tuple[Any, ...]:
    return (
        2,
        _target_to_str(node.target),
        tuple(canonical_idx[item] for item in _iter_nodes(node.args)),
    )


def _canonical_node_key(node: Node, canonical_idx: dict[Node, int]) -> object:
    if node.op == "placeholder":
        raise AssertionError("placeholders are ordered by their input contract")
    if node.op == "get_attr":
        return 1, str(node.target)
    if node.op == "output":
        return (3,)
    return _computation_node_key(node, canonical_idx)


def _is_safe_to_reorder(node: Node) -> bool:
    if node.op == "call_method":
        return not str(node.target).endswith("_")
    if node.op == "call_function":
        name = getattr(node.target, "__name__", "")
        return not (
            str(name).endswith("_")
            or name in _IN_PLACE_OPERATORS
            or "out" in node.kwargs
        )
    return node.op not in {"call_module"}


def rename_nodes_to_canonical(
    graph: Graph, skip_ops: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Rename nodes from their operation targets and return changed names."""

    renamed: dict[str, str] = {}
    used: set[str] = set()
    for node in graph.nodes:
        old = node.name
        if node.op in skip_ops:
            used.add(old)
            continue
        base = _target_to_str(node.target)
        name = base
        index = 0
        while name in used:
            name = f"{base}_{index}"
            index += 1
        used.add(name)
        node.name = name
        if old != name:
            renamed[old] = name
    graph._live_names = used
    return renamed


def canonicalize_graph(
    graph: Graph,
    canonical_key_fn: Callable[[Node, dict[Node, int]], object] = _canonical_node_key,
    is_safe_to_reorder: Callable[[Node], bool] = _is_safe_to_reorder,
    *,
    skip_rename_ops: frozenset[str] = frozenset(),
    group_getitems: bool = False,
) -> dict[str, str]:
    """Topologically reorder safe nodes using a deterministic ready queue."""

    del group_getitems
    original = list(graph.nodes)
    positions = {node: index for index, node in enumerate(original)}
    dependencies = {
        node: set(_iter_nodes(node.args)) | set(_iter_nodes(node.kwargs))
        for node in original
    }
    ordered: list[Node] = []
    ready = [node for node in original if not dependencies[node]]
    ready.sort(key=lambda node: positions[node])
    while ready:
        if ordered and not is_safe_to_reorder(ordered[-1]):
            candidate = min(ready, key=positions.__getitem__)
        else:
            index_map = {node: index for index, node in enumerate(ordered)}

            def ready_key(node: Node) -> object:
                if node.op == "placeholder":
                    return (0, positions[node])
                if not is_safe_to_reorder(node):
                    return (0, positions[node])
                return canonical_key_fn(node, index_map)

            candidate = min(
                ready,
                key=ready_key,
            )
        ready.remove(candidate)
        ordered.append(candidate)
        for node in original:
            if candidate in dependencies[node]:
                dependencies[node].remove(candidate)
                if not dependencies[node]:
                    ready.append(node)
    if len(ordered) != len(original):
        raise RuntimeError("graph contains a dependency cycle")
    graph.nodes[:] = ordered
    graph.lint()
    return rename_nodes_to_canonical(graph, skip_rename_ops)
