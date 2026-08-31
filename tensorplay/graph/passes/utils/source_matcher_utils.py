"""Group graph nodes by source metadata and expose their boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ...graph import Graph
from ...node import Node

__all__ = ["SourcePartition", "check_subgraphs_connected", "get_source_partitions"]


@dataclass
class SourcePartition:
    nodes: list[Node]
    source: Any
    input_nodes: list[Node] = field(default_factory=list)
    output_nodes: list[Node] = field(default_factory=list)
    params: list[Node] = field(default_factory=list)


def get_source_partitions(
    graph: Graph,
    wanted_sources: list[Any],
    filter_fn: Callable[[Node], bool] | None = None,
) -> dict[Any, list[SourcePartition]]:
    groups: dict[tuple[Any, str], list[Node]] = {}
    wanted = set(wanted_sources)
    for node in graph.nodes:
        if filter_fn is not None and not filter_fn(node):
            continue
        source = node.meta.get("source_fn", node.meta.get("source"))
        if isinstance(source, tuple):
            source = source[-1]
        if source not in wanted:
            continue
        scope = str(node.meta.get("source_scope", ""))
        groups.setdefault((source, scope), []).append(node)
    result: dict[Any, list[SourcePartition]] = {}
    for (source, _scope), nodes in groups.items():
        inside = set(nodes)
        inputs = []
        outputs = []
        for node in nodes:
            for value in (*node.args, *node.kwargs.values()):
                if isinstance(value, Node) and value not in inside and value not in inputs:
                    inputs.append(value)
            if any(user not in inside for user in node.users):
                outputs.append(node)
        result.setdefault(source, []).append(SourcePartition(nodes, source, inputs, outputs))
    return result


def check_subgraphs_connected(subgraphs: list[SourcePartition]) -> bool:
    if not subgraphs:
        return True
    seen = set(subgraphs[0].nodes)
    pending = list(subgraphs[0].output_nodes)
    while pending:
        node = pending.pop()
        for user in node.users:
            if user in seen:
                continue
            if any(user in part.nodes for part in subgraphs):
                seen.add(user)
                pending.append(user)
    return all(any(node in seen for node in part.nodes) for part in subgraphs)
