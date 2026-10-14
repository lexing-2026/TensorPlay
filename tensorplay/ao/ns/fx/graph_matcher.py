"""Graph matching utilities for the FX numeric suite.

Matches matchable subgraphs of two traced models (e.g. a float model and a
quantized model) so that per-layer numeric comparisons can be computed.
Subgraphs are matched greedily from the output backwards; fusion patterns
from the native backend config are recognized.
"""

from __future__ import annotations

import collections
import enum
from typing import Any

from tensorplay.graph import GraphModule
from tensorplay.graph.node import Node
from tensorplay.ao.quantization import FakeQuantizeBase, ObserverBase
from tensorplay.ao.quantization.utils import getattr_from_fqn

from .mappings import (
    get_base_name_to_sets_of_related_ops,
    get_unmatchable_types_map,
)
from .ns_types import NSNodeTargetType, NSSubgraph
from .pattern_utils import (
    end_node_matches_reversed_fusion,
    get_reversed_fusions,
    get_type_a_related_to_b,
)

__all__ = [
    "get_matching_subgraph_pairs",
    "GraphMatchingException",
    "SubgraphTypeRelationship",
]


def _get_output_nodes(graph) -> list[Node]:
    return [n for n in graph.nodes if n.op == "output"]


class _NSGraphMatchableSubgraphsIterator:
    """Iterate the graph from the outputs backwards, yielding matchable subgraphs."""

    def __init__(
        self,
        gm: GraphModule,
        non_matchable_functions: set[NSNodeTargetType],
        non_matchable_modules: set[NSNodeTargetType],
        non_matchable_methods: set[NSNodeTargetType],
    ):
        self.gm: GraphModule = gm
        self.non_matchable_functions = non_matchable_functions
        self.non_matchable_modules = non_matchable_modules
        self.non_matchable_methods = non_matchable_methods
        self.seen_nodes: set[Node] = set()
        self.stack: list[Node] = []
        for start_node in _get_output_nodes(self.gm.graph):
            self.stack.append(start_node)

    def __iter__(self):
        return self

    def __next__(self) -> NSSubgraph:
        while len(self.stack) > 0:
            cur_end_node = self.stack.pop()
            if cur_end_node in self.seen_nodes:
                continue

            cur_start_node = cur_end_node
            cur_base_op_node = cur_end_node

            # greedily match the first (longest) fusion pattern
            for _reverse_fusion_ops, base_op_idx in get_reversed_fusions():
                is_match = end_node_matches_reversed_fusion(
                    cur_end_node, _reverse_fusion_ops, self.gm, self.seen_nodes
                )
                if is_match:
                    for rev_fusion_idx in range(len(_reverse_fusion_ops) - 1):
                        self.seen_nodes.add(cur_start_node)
                        cur_start_node = cur_start_node.args[0]
                        rev_base_op_idx = len(_reverse_fusion_ops) - 2 - base_op_idx
                        if rev_fusion_idx == rev_base_op_idx:
                            cur_base_op_node = cur_start_node
                    break

            self.seen_nodes.add(cur_start_node)
            for arg in cur_start_node.all_input_nodes:
                self._recursively_add_node_arg_to_stack(arg)

            if not self._is_matchable(cur_base_op_node):
                continue

            # skip lone observers / fake quants not matched as part of a pattern
            if cur_end_node.op == "call_module" and cur_start_node is cur_end_node:
                maybe_obs = getattr_from_fqn(self.gm, cur_end_node.target)
                if isinstance(maybe_obs, (ObserverBase, FakeQuantizeBase)):
                    continue

            return NSSubgraph(
                start_node=cur_start_node,
                end_node=cur_end_node,
                base_op_node=cur_base_op_node,
            )

        raise StopIteration

    def _recursively_add_node_arg_to_stack(self, arg: Any) -> None:
        if isinstance(arg, Node):
            self.stack.append(arg)
        elif isinstance(arg, (tuple, list)):
            for inner_arg in arg:
                self._recursively_add_node_arg_to_stack(inner_arg)
        elif isinstance(arg, dict):
            for value in arg.values():
                self._recursively_add_node_arg_to_stack(value)

    def _is_matchable(self, node: Node) -> bool:
        if node.op == "call_function":
            return node.target not in self.non_matchable_functions
        elif node.op == "call_module":
            if not isinstance(node.target, str):
                return False
            target_mod = getattr_from_fqn(self.gm, node.target)
            return not any(
                isinstance(target_mod, t) for t in self.non_matchable_modules
            )
        elif node.op == "call_method":
            return node.target not in self.non_matchable_methods
        else:
            return False


class GraphMatchingException(Exception):
    """Raised when two graphs cannot be matched."""


class SubgraphTypeRelationship(enum.Enum):
    EQUAL = enum.auto()
    EQUAL_BUT_UKNOWN = enum.auto()
    RELATED_BUT_NOT_EQUAL = enum.auto()
    NOT_RELATED = enum.auto()


def _get_node_target_type(node: Node, gm: GraphModule):
    if node.op in ("call_function", "call_method"):
        return node.target
    elif node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"Expected str, got {type(node.target)}")
        mod = getattr_from_fqn(gm, node.target)
        return type(mod)
    return None


def _get_subgraph_relationship_type(
    subgraph_a: NSSubgraph,
    subgraph_b: NSSubgraph,
    gm_a: GraphModule,
    gm_b: GraphModule,
    type_a_related_to_b: set[tuple[NSNodeTargetType, NSNodeTargetType]],
) -> SubgraphTypeRelationship:
    node_a = subgraph_a.base_op_node
    node_b = subgraph_b.base_op_node

    if node_a.op != node_b.op:
        if not (
            node_a.op in ("call_function", "call_method")
            and node_b.op in ("call_function", "call_method")
        ):
            return SubgraphTypeRelationship.NOT_RELATED

    if node_a.op in ("call_function", "call_method"):
        key = (node_a.target, node_b.target)
        if key not in type_a_related_to_b:
            if node_a.target == node_b.target:
                return SubgraphTypeRelationship.EQUAL_BUT_UKNOWN
            return SubgraphTypeRelationship.NOT_RELATED
        if node_a.target == node_b.target:
            return SubgraphTypeRelationship.EQUAL
        return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL

    elif node_a.op == "call_module":
        if not isinstance(node_a.target, str) or not isinstance(node_b.target, str):
            raise AssertionError("Expected str targets for call_module nodes")
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        mod_b = getattr_from_fqn(gm_b, node_b.target)
        key = (type(mod_a), type(mod_b))
        if key not in type_a_related_to_b:
            if type(mod_a) is type(mod_b):
                return SubgraphTypeRelationship.EQUAL_BUT_UKNOWN
            return SubgraphTypeRelationship.NOT_RELATED
        if type(mod_a) is type(mod_b):
            return SubgraphTypeRelationship.EQUAL
        return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL

    return SubgraphTypeRelationship.NOT_RELATED


def _get_name_for_subgraph(
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
    existing_names: set[str],
) -> str:
    target_type = _get_node_target_type(subgraph_a.base_op_node, gm_a)
    target_base_type = None
    for base_name, sets_of_related_ops in base_name_to_sets_of_related_ops.items():
        if target_type in sets_of_related_ops:
            target_base_type = base_name
    target_base_name = "base_op_" + str(target_base_type)
    counter = 0
    proposed_name = target_base_name + "_" + str(counter)
    while proposed_name in existing_names:
        counter += 1
        proposed_name = target_base_name + "_" + str(counter)
    existing_names.add(proposed_name)
    return proposed_name


def get_matching_subgraph_pairs(
    gm_a: GraphModule,
    gm_b: GraphModule,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> dict[str, tuple[NSSubgraph, NSSubgraph]]:
    """Match matchable subgraphs of ``gm_a`` to ``gm_b``.

    Returns an ordered dictionary mapping a shared subgraph name to a pair of
    :class:`NSSubgraph` (one from each graph).  Raises
    :class:`GraphMatchingException` when the two graphs have a different
    number of matchable subgraphs or the subgraphs are unrelated.
    """
    if unmatchable_types_map is None:
        unmatchable_types_map = get_unmatchable_types_map()
    non_matchable_functions = unmatchable_types_map["funs_unmatchable"]
    non_matchable_modules = unmatchable_types_map["mods_unmatchable"]
    non_matchable_methods = unmatchable_types_map["meths_unmatchable"]

    graph_a_iterator = _NSGraphMatchableSubgraphsIterator(
        gm_a, non_matchable_functions, non_matchable_modules, non_matchable_methods
    )
    graph_b_iterator = _NSGraphMatchableSubgraphsIterator(
        gm_b, non_matchable_functions, non_matchable_modules, non_matchable_methods
    )
    results = collections.OrderedDict()
    if base_name_to_sets_of_related_ops is None:
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = get_type_a_related_to_b(base_name_to_sets_of_related_ops)

    existing_names_a: set[str] = set()
    existing_names_b: set[str] = set()

    while True:
        cur_subgraph_a, cur_subgraph_b = None, None
        try:
            cur_subgraph_a = next(graph_a_iterator)
        except StopIteration:
            pass
        try:
            cur_subgraph_b = next(graph_b_iterator)
        except StopIteration:
            pass

        type_start_a = (
            _get_node_target_type(cur_subgraph_a.start_node, gm_a)
            if cur_subgraph_a is not None
            else None
        )
        type_start_b = (
            _get_node_target_type(cur_subgraph_b.start_node, gm_b)
            if cur_subgraph_b is not None
            else None
        )

        if cur_subgraph_a is not None and cur_subgraph_b is not None:
            subgraph_relationship = _get_subgraph_relationship_type(
                cur_subgraph_a, cur_subgraph_b, gm_a, gm_b, type_a_related_to_b
            )
            if subgraph_relationship == SubgraphTypeRelationship.NOT_RELATED:
                raise GraphMatchingException(
                    f"The subgraphs ({cur_subgraph_a}, {type_start_a}) and "
                    f"({cur_subgraph_b}, {type_start_b}) are not related. "
                    "Ensure the two models have the same number of subgraphs "
                    "and each pair is related."
                )
            if subgraph_relationship == SubgraphTypeRelationship.EQUAL_BUT_UKNOWN:
                continue
            key_name_a = _get_name_for_subgraph(
                cur_subgraph_a, gm_a, base_name_to_sets_of_related_ops, existing_names_a
            )
            key_name_b = _get_name_for_subgraph(
                cur_subgraph_b, gm_b, base_name_to_sets_of_related_ops, existing_names_b
            )
            if key_name_a != key_name_b:
                raise AssertionError(
                    f"Subgraph names {key_name_a} and {key_name_b} do not match"
                )
            results[key_name_a] = (cur_subgraph_a, cur_subgraph_b)
            continue
        elif cur_subgraph_a is None and cur_subgraph_b is None:
            break
        else:
            raise GraphMatchingException(
                f"Attempting to match ({cur_subgraph_a}, {type_start_a}) and "
                f"({cur_subgraph_b}, {type_start_b}), one of which is empty. "
                "Ensure the two models have the same number of subgraphs."
            )

    # subgraphs were collected output-first; reverse to execution order
    results = collections.OrderedDict(reversed(results.items()))
    return results