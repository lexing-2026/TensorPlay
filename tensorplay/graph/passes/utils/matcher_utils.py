"""Use-def graph matcher with literal and overlap controls."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ...graph import Graph
from ...node import Node
from ...subgraph_rewriter import _find_matches, _match_value

__all__ = ["InternalMatch", "SubgraphMatcher"]


@dataclass
class InternalMatch:
    anchors: list[Node]
    nodes_map: dict[Node, Node] = field(default_factory=dict)
    placeholder_nodes: list[Node] = field(default_factory=list)
    returning_nodes: list[Node] = field(default_factory=list)
    name_node_map: dict[str, Node] = field(default_factory=dict)

    def __copy__(self) -> "InternalMatch":
        return InternalMatch(
            list(self.anchors),
            self.nodes_map.copy(),
            list(self.placeholder_nodes),
            list(self.returning_nodes),
            self.name_node_map.copy(),
        )


class SubgraphMatcher:
    """Find connected pattern subgraphs by recursively matching dependencies."""

    def __init__(
        self,
        pattern: Graph,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = True,
        ignore_literals: bool = False,
    ) -> None:
        if not pattern.nodes:
            raise ValueError("cannot match an empty pattern")
        self.pattern = pattern
        self.match_output = match_output
        self.match_placeholder = match_placeholder
        self.remove_overlapping_matches = remove_overlapping_matches
        self.ignore_literals = ignore_literals

    def _matches_with_options(self, graph: Graph, node_name_match: str = ""):
        if not self.match_output:
            return _find_matches(
                graph,
                self.pattern,
                ignore_literals=self.ignore_literals,
                node_name_match=node_name_match,
            )
        pattern_output = self.pattern.output_node
        graph_output = graph.output_node
        mapping: dict[Node, Node] = {}
        reverse: dict[Node, Node] = {}
        if not _match_value(
            pattern_output,
            graph_output,
            mapping,
            reverse,
            ignore_literals=self.ignore_literals,
            node_name_match=node_name_match,
        ):
            return []
        return [type("_Match", (), {"anchor": graph_output, "nodes_map": mapping})()]

    def match(self, graph: Graph, node_name_match: str = "") -> list[InternalMatch]:
        candidates = self._matches_with_options(graph, node_name_match)
        result: list[InternalMatch] = []
        occupied: set[Node] = set()
        pattern_output = self.pattern.output_node
        returning = [node for node in self.pattern.nodes if node in _nodes_from(pattern_output.args)]
        for candidate in candidates:
            mapping = dict(candidate.nodes_map)
            matched = {
                value for pattern_node, value in mapping.items()
                if pattern_node.op not in {"placeholder", "output"}
            }
            if self.remove_overlapping_matches and occupied & matched:
                continue
            occupied.update(matched)
            result.append(
                InternalMatch(
                    anchors=[candidate.anchor],
                    nodes_map=mapping,
                    placeholder_nodes=[
                        mapping[node]
                        for node in self.pattern.nodes
                        if node.op == "placeholder" and node in mapping
                    ],
                    returning_nodes=[mapping[node] for node in returning if node in mapping],
                )
            )
        return result


def _nodes_from(value: Any):
    if isinstance(value, Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nodes_from(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nodes_from(item)
