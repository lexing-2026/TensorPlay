"""Matcher variant that exposes matched nodes by their names."""

from __future__ import annotations

from .matcher_utils import InternalMatch, SubgraphMatcher

__all__ = ["SubgraphMatcherWithNameNodeMap"]


class SubgraphMatcherWithNameNodeMap(SubgraphMatcher):
    def match(self, graph, node_name_match: str = ""):
        matches = super().match(graph, node_name_match=node_name_match)
        for match in matches:
            match.name_node_map = {
                pattern.name: target
                for pattern, target in match.nodes_map.items()
                if pattern.op != "output"
            }
        return matches
