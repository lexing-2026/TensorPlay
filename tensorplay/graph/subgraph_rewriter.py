"""Use-def based graph pattern matching and replacement."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

from ._compatibility import compatibility
from ._utils import _iter_nodes, _map_arg
from .graph import Graph, dead_code_elimination
from .graph_module import GraphModule
from .node import Node
from .symbolic_trace import symbolic_trace

__all__ = [
    "Match",
    "ReplacedPatterns",
    "replace_pattern",
    "replace_pattern_with_filters",
]


@compatibility(is_backward_compatible=True)
class Match(NamedTuple):
    """A pattern anchor and the node mapping that produced the match."""

    anchor: Node
    nodes_map: dict[Node, Node]


@compatibility(is_backward_compatible=False)
@dataclass
class ReplacedPatterns:
    """Details of one pattern replacement."""

    anchor: Node
    nodes_map: dict[Node, Node]
    replacements: list[Node]


def _as_graph(value: Callable[..., Any] | Graph | GraphModule) -> Graph:
    if isinstance(value, GraphModule):
        return value.graph
    if isinstance(value, Graph):
        return value
    if callable(value):
        return symbolic_trace(value).graph
    raise TypeError(f"expected a callable, Graph, or GraphModule; got {type(value)!r}")


def _same_target(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return isinstance(result, bool) and result


def _same_literal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except Exception:
        return left is right
    return isinstance(result, bool) and result


def _match_value(
    pattern_value: Any,
    graph_value: Any,
    mapping: dict[Node, Node],
    reverse: dict[Node, Node],
    *,
    ignore_literals: bool,
    node_name_match: str,
) -> bool:
    if isinstance(pattern_value, Node):
        if not isinstance(graph_value, Node):
            return False
        existing = mapping.get(pattern_value)
        if existing is not None:
            return existing is graph_value
        if pattern_value.op == "placeholder":
            previous = reverse.get(graph_value)
            if previous is not None and previous is not pattern_value:
                return False
            mapping[pattern_value] = graph_value
            reverse[graph_value] = pattern_value
            return True
        if graph_value.op != pattern_value.op or not _same_target(
            pattern_value.target, graph_value.target
        ):
            return False
        if node_name_match and pattern_value.name != node_name_match:
            return False
        mapping[pattern_value] = graph_value
        reverse[graph_value] = pattern_value
        if not _match_value(
            pattern_value.args,
            graph_value.args,
            mapping,
            reverse,
            ignore_literals=ignore_literals,
            node_name_match=node_name_match,
        ):
            return False
        return _match_value(
            pattern_value.kwargs,
            graph_value.kwargs,
            mapping,
            reverse,
            ignore_literals=ignore_literals,
            node_name_match=node_name_match,
        )
    if isinstance(pattern_value, tuple):
        return (
            isinstance(graph_value, tuple)
            and len(pattern_value) == len(graph_value)
            and all(
                _match_value(
                    left,
                    right,
                    mapping,
                    reverse,
                    ignore_literals=ignore_literals,
                    node_name_match=node_name_match,
                )
                for left, right in zip(pattern_value, graph_value)
            )
        )
    if isinstance(pattern_value, list):
        return (
            isinstance(graph_value, list)
            and len(pattern_value) == len(graph_value)
            and all(
                _match_value(
                    left,
                    right,
                    mapping,
                    reverse,
                    ignore_literals=ignore_literals,
                    node_name_match=node_name_match,
                )
                for left, right in zip(pattern_value, graph_value)
            )
        )
    if isinstance(pattern_value, dict):
        return (
            isinstance(graph_value, dict)
            and pattern_value.keys() == graph_value.keys()
            and all(
                _match_value(
                    pattern_value[key],
                    graph_value[key],
                    mapping,
                    reverse,
                    ignore_literals=ignore_literals,
                    node_name_match=node_name_match,
                )
                for key in pattern_value
            )
        )
    return ignore_literals or _same_literal(pattern_value, graph_value)


def _find_matches(
    graph: Graph,
    pattern: Graph,
    *,
    ignore_literals: bool,
    node_name_match: str,
) -> list[Match]:
    output = pattern.output_node.args[0]
    candidates = [node for node in graph.nodes if node.op not in {"placeholder", "output"}]
    matches: list[Match] = []
    occupied: set[Node] = set()
    for candidate in candidates:
        mapping: dict[Node, Node] = {}
        reverse: dict[Node, Node] = {}
        if not _match_value(
            output,
            candidate,
            mapping,
            reverse,
            ignore_literals=ignore_literals,
            node_name_match=node_name_match,
        ):
            continue
        matched_nodes = {
            value for key, value in mapping.items() if key.op != "placeholder"
        }
        if matched_nodes & occupied:
            continue
        occupied.update(matched_nodes)
        matches.append(Match(candidate, mapping))
    return matches


def _get_path(root: Any, target: str) -> Any:
    value = root
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _set_path(root: Any, target: str, value: Any) -> None:
    parts = target.split(".")
    holder = root
    for atom in parts[:-1]:
        holder = getattr(holder, atom)
    setattr(holder, parts[-1], value)


def _replace_attributes(gm: GraphModule, replacement: Any) -> None:
    result_root = gm.root
    replacement_root = replacement.root if isinstance(replacement, GraphModule) else replacement
    if result_root is None or replacement_root is None:
        return
    for node in gm.graph.nodes:
        if node.op not in {"call_module", "get_attr"}:
            continue
        try:
            _get_path(result_root, node.target)
            continue
        except AttributeError:
            pass
        try:
            value = copy.deepcopy(_get_path(replacement_root, node.target))
        except AttributeError as exc:
            raise RuntimeError(
                f"replacement references missing attribute {node.target!r}"
            ) from exc
        _set_path(result_root, node.target, value)


def _replace_with_value(node: Node, value: Any) -> None:
    for user in list(node.users):
        user.args = _map_arg(user.args, lambda item: value if item is node else item)
        user.kwargs = _map_arg(user.kwargs, lambda item: value if item is node else item)
        node.users.discard(user)
    node.erase_node()


def _apply_replacement(
    gm: GraphModule,
    pattern: Graph,
    replacement: Graph,
    match: Match,
) -> ReplacedPatterns:
    pattern_placeholders = [node for node in pattern.nodes if node.op == "placeholder"]
    replacement_placeholders = [node for node in replacement.nodes if node.op == "placeholder"]
    if len(pattern_placeholders) != len(replacement_placeholders):
        raise AssertionError(
            f"placeholder count mismatch: {len(pattern_placeholders)} vs "
            f"{len(replacement_placeholders)}"
        )
    value_map = {
        replacement_node: match.nodes_map[pattern_node]
        for replacement_node, pattern_node in zip(
            replacement_placeholders, pattern_placeholders
        )
    }
    original_nodes = set(gm.graph.nodes)
    with gm.graph.inserting_before(match.anchor):
        result_value = gm.graph.graph_copy(replacement, value_map)
    if isinstance(result_value, tuple) and len(result_value) == 1:
        result_value = result_value[0]
    if isinstance(result_value, Node):
        match.anchor.replace_all_uses_with(result_value)
        match.anchor.erase_node()
    else:
        _replace_with_value(match.anchor, result_value)
    removed = dead_code_elimination(gm.graph)
    del removed
    replacements = [node for node in gm.graph.nodes if node not in original_nodes]
    return ReplacedPatterns(match.anchor, match.nodes_map, replacements)


@compatibility(is_backward_compatible=True)
def replace_pattern(
    gm: GraphModule,
    pattern: Callable[..., Any] | GraphModule,
    replacement: Callable[..., Any] | GraphModule,
) -> list[Match]:
    """Replace each non-overlapping use-def match in ``gm``."""

    pattern_graph = _as_graph(pattern)
    replacement_graph = _as_graph(replacement)
    matches = _find_matches(gm.graph, pattern_graph, ignore_literals=False, node_name_match="")
    replaced: list[Match] = []
    for match in matches:
        result = _apply_replacement(gm, pattern_graph, replacement_graph, match)
        replaced.append(Match(result.anchor, result.nodes_map))
    if matches:
        _replace_attributes(gm, replacement)
        gm.recompile()
    return replaced


@compatibility(is_backward_compatible=False)
def replace_pattern_with_filters(
    gm: GraphModule,
    pattern: Callable[..., Any] | Graph | GraphModule,
    replacement: Callable[..., Any] | Graph | GraphModule | None = None,
    match_filters: list[Callable[[Any, Graph, Graph], bool]] | None = None,
    ignore_literals: bool = False,
    replacement_callback: Callable[[Any, Graph, Graph], Graph] | None = None,
    node_name_match: str = "",
) -> list[ReplacedPatterns]:
    """Match and replace graphs with optional per-match filtering."""

    pattern_graph = _as_graph(pattern)
    matches = _find_matches(
        gm.graph,
        pattern_graph,
        ignore_literals=ignore_literals,
        node_name_match=node_name_match,
    )
    filters = match_filters or []
    selected = [
        match
        for match in matches
        if all(filter_fn(match, gm.graph, pattern_graph) for filter_fn in filters)
    ]
    common_replacement = _as_graph(replacement) if replacement is not None else None
    results: list[ReplacedPatterns] = []
    for match in selected:
        if replacement_callback is not None:
            replacement_graph = replacement_callback(match, gm.graph, pattern_graph)
        elif common_replacement is not None:
            replacement_graph = common_replacement
        else:
            raise AssertionError("a replacement or replacement callback is required")
        results.append(_apply_replacement(gm, pattern_graph, replacement_graph, match))
    if results:
        _replace_attributes(gm, replacement)
        gm.recompile()
    return results
