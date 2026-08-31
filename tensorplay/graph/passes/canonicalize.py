"""Deterministic ordering and naming for graph nodes."""

from __future__ import annotations

import collections
import heapq
import itertools
import operator
from collections.abc import Callable

from ..graph import Graph, _Namespace
from ..node import Node

__all__ = ["canonicalize_graph", "rename_nodes_to_canonical"]


_IN_PLACE_OPERATORS = frozenset(
    {
        "iadd",
        "iand",
        "iconcat",
        "ifloordiv",
        "ilshift",
        "imatmul",
        "imod",
        "imul",
        "ior",
        "ipow",
        "irshift",
        "isub",
        "itruediv",
        "ixor",
    }
)


def _computation_node_key(
    node: Node, canonical_idx: dict[Node, int]
) -> tuple[int, str, tuple[int, ...]]:
    return (
        2,
        node.graph._target_to_str(node.target),
        tuple(canonical_idx[input_node] for input_node in node.all_input_nodes),
    )


def _canonical_node_key(node: Node, canonical_idx: dict[Node, int]) -> object:
    if node.op == "placeholder":
        raise AssertionError("placeholder nodes require an input ordering key")
    if node.op == "get_attr":
        return (1, str(node.target))
    if node.op == "output":
        return (3,)
    return _computation_node_key(node, canonical_idx)


def _is_safe_to_reorder(node: Node) -> bool:
    if node.op == "call_method":
        return not str(node.target).endswith("_")
    if node.op == "call_module":
        return not node.is_impure()
    if node.op != "call_function":
        return True
    if node.is_impure():
        return False

    if getattr(node.target, "mutates_args", ()):
        return False

    name = getattr(node.target, "__name__", "")
    if str(name).endswith("_"):
        return False
    if (
        getattr(node.target, "__module__", "") == "_operator"
        and name in _IN_PLACE_OPERATORS
    ):
        return False
    if isinstance(node.kwargs.get("out"), Node):
        return False
    if name == "triton_kernel_wrapper_mutation":
        return False
    if not node.all_input_nodes:
        return False
    return True


def rename_nodes_to_canonical(
    graph: Graph,
    skip_ops: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Rename graph nodes from their targets and return changed names."""

    renamed: dict[str, str] = {}
    namespace = _Namespace()
    new_names: dict[Node, str] = {}

    for node in graph.nodes:
        old_name = node.name
        if node.op in skip_ops:
            new_name = namespace.create_name(old_name, node)
        else:
            new_name = namespace.create_name(graph._target_to_str(node.target), node)
        new_names[node] = new_name
        if old_name != new_name:
            renamed[old_name] = new_name

    for node, new_name in new_names.items():
        object.__setattr__(node, "name", new_name)
    graph._graph_namespace = namespace
    graph._live_names = set(new_names.values())
    return renamed


def _sink_get_attr_nodes(order: list[Node]) -> None:
    non_get_attrs = [node for node in order if node.op != "get_attr"]
    get_attrs = [node for node in order if node.op == "get_attr"]
    if not get_attrs:
        return

    positions = {node: index for index, node in enumerate(non_get_attrs)}
    insertions: dict[int, list[Node]] = collections.defaultdict(list)
    for node in get_attrs:
        if node.users:
            target = min(
                positions.get(user, len(non_get_attrs)) for user in node.users
            )
        else:
            target = (
                len(non_get_attrs) - 1
                if non_get_attrs and non_get_attrs[-1].op == "output"
                else len(non_get_attrs)
            )
        insertions[target].append(node)

    order.clear()
    for index, node in enumerate(non_get_attrs):
        order.extend(insertions.pop(index, ()))
        order.append(node)
    for remaining in insertions.values():
        order.extend(remaining)


def _group_getitem_nodes(order: list[Node]) -> None:
    children: dict[Node, list[Node]] = collections.defaultdict(list)
    getitems: set[Node] = set()
    for node in order:
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and node.args
            and isinstance(node.args[0], Node)
        ):
            children[node.args[0]].append(node)
            getitems.add(node)
    if not getitems:
        return

    def getitem_key(node: Node) -> tuple[int, int, str]:
        index = node.args[1]
        return (0, index, "") if isinstance(index, int) else (1, 0, str(index))

    for group in children.values():
        group.sort(key=getitem_key)

    grouped: list[Node] = []

    def emit(node: Node) -> None:
        grouped.append(node)
        for child in children.get(node, ()):
            emit(child)

    for node in order:
        if node not in getitems:
            emit(node)
    if len(grouped) != len(order):
        raise AssertionError(
            f"getitem grouping lost nodes: {len(grouped)} != {len(order)}"
        )
    order[:] = grouped


def canonicalize_graph(
    graph: Graph,
    canonical_key_fn: Callable[[Node, dict[Node, int]], object] = _canonical_node_key,
    is_safe_to_reorder: Callable[[Node], bool] = _is_safe_to_reorder,
    *,
    skip_rename_ops: frozenset[str] = frozenset(),
    group_getitems: bool = False,
) -> dict[str, str]:
    """Reorder graph nodes deterministically and assign canonical names."""

    original = list(graph.nodes)
    original_positions = {node: index for index, node in enumerate(original)}
    indegree: dict[Node, int] = {
        node: len(node.all_input_nodes) for node in original
    }

    extra_users: dict[Node, list[Node]] = collections.defaultdict(list)
    previous_barrier: Node | None = None
    segment_reorderable: list[Node] = []
    for node in original:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        barrier = not is_safe_to_reorder(node)
        if barrier:
            for reorderable in segment_reorderable:
                extra_users[reorderable].append(node)
                indegree[node] += 1
            segment_reorderable = []
        if previous_barrier is not None:
            extra_users[previous_barrier].append(node)
            indegree[node] += 1
        if barrier:
            previous_barrier = node
        else:
            segment_reorderable.append(node)

    canonical_idx: dict[Node, int] = {}
    counter = itertools.count()
    ready: list[tuple[object, int, Node]] = []

    def ready_key(node: Node) -> object:
        if node.op == "placeholder" and canonical_key_fn is _canonical_node_key:
            return (0, original_positions[node])
        return canonical_key_fn(node, canonical_idx)

    for node in original:
        if indegree[node] == 0:
            ready.append((ready_key(node), next(counter), node))
    heapq.heapify(ready)

    canonical_order: list[Node] = []
    while ready:
        _, _, current = heapq.heappop(ready)
        canonical_order.append(current)
        canonical_idx[current] = len(canonical_idx)
        for user in itertools.chain(
            current.users, extra_users.get(current, ())
        ):
            indegree[user] -= 1
            if indegree[user] == 0:
                heapq.heappush(ready, (ready_key(user), next(counter), user))

    if len(canonical_order) != len(original):
        remaining = [node for node in indegree if indegree[node] != 0]
        raise RuntimeError(
            f"canonicalization failed: processed {len(canonical_order)} of "
            f"{len(original)} nodes; remaining nodes: {remaining}"
        )

    _sink_get_attr_nodes(canonical_order)
    if group_getitems:
        _group_getitem_nodes(canonical_order)

    graph.nodes[:] = canonical_order
    graph.lint()
    return rename_nodes_to_canonical(graph, skip_ops=skip_rename_ops)
