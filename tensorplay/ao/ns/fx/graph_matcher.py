"""Graph matching utilities for the FX numeric suite."""

import enum

from tensorplay.graph import GraphModule

__all__ = ["get_matching_subgraph_pairs", "GraphMatchingException", "SubgraphTypeRelationship"]


class GraphMatchingException(Exception):
    pass


class SubgraphTypeRelationship(enum.Enum):
    SAME_MODULE = 0
    DIFFERENT_MODULE = 1


def _get_output_nodes(graph):
    return [node for node in graph.nodes if node.op == "output"]


def _get_name_for_subgraph(subgraph):
    return str(subgraph)


def get_matching_subgraph_pairs(
    gm1: GraphModule,
    gm2: GraphModule,
    base_name_to_sets_of_related_ops,
    unmatchable_types_map=None,
):
    """Return a mapping from subgraph names to pairs of matched subgraphs."""
    matching = {}
    modules1 = {name: mod for name, mod in gm1.named_modules() if name}
    modules2 = {name: mod for name, mod in gm2.named_modules() if name}
    for name1, mod1 in modules1.items():
        for name2, mod2 in modules2.items():
            if type(mod1) is type(mod2):
                matching[name1] = (name1, name2)
                break
    return matching