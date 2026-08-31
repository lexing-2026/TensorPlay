import operator

from tensorplay.graph import Graph
from tensorplay.graph.passes.canonicalize import (
    _canonical_node_key,
    _is_safe_to_reorder,
    canonicalize_graph,
)


def _key(node, canonical_idx):
    if node.op == "placeholder":
        return (0, node.target)
    return _canonical_node_key(node, canonical_idx)


def _combine(left, right):
    return left + right


def test_canonicalize_uses_all_inputs_and_sinks_attributes():
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    weight = graph.get_attr("weight")
    independent = graph.call_function(operator.mul, (x, y))
    combined = graph.call_function(
        _combine,
        (independent,),
        {"right": weight},
    )
    graph.output(combined)

    canonicalize_graph(graph, _key, _is_safe_to_reorder)
    nodes = list(graph.nodes)
    assert nodes.index(weight) == nodes.index(combined) - 1
    assert nodes.index(independent) < nodes.index(weight)
    graph.lint()


def test_canonicalize_keeps_barrier_segments_in_source_order():
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    first = graph.call_function(operator.mul, (x, y))
    barrier = graph.call_function(operator.iadd, (x, y))
    last = graph.call_function(operator.add, (x, y))
    graph.output(last)

    assert _is_safe_to_reorder(first)
    assert not _is_safe_to_reorder(barrier)
    canonicalize_graph(graph, _key, _is_safe_to_reorder)
    positions = {node: index for index, node in enumerate(graph.nodes)}
    assert positions[first] < positions[barrier] < positions[last]


def test_canonicalize_groups_getitems_by_index():
    graph = Graph()
    x = graph.placeholder("x")
    producer = graph.call_function(lambda value: (value, value), (x,))
    unrelated = graph.call_function(operator.neg, (x,))
    item_one = graph.call_function(operator.getitem, (producer, 1))
    item_zero = graph.call_function(operator.getitem, (producer, 0))
    graph.output((item_one, item_zero, unrelated))

    canonicalize_graph(
        graph,
        _key,
        _is_safe_to_reorder,
        group_getitems=True,
    )
    nodes = list(graph.nodes)
    producer_index = nodes.index(producer)
    assert nodes[producer_index + 1] is item_zero
    assert nodes[producer_index + 2] is item_one
    graph.lint()


def test_canonicalize_rebuilds_namespace_for_future_nodes():
    graph = Graph()
    x = graph.placeholder("x")
    first = graph.call_function(operator.add, (x, x), name="custom")
    second = graph.call_function(operator.add, (x, x), name="custom")

    renamed = canonicalize_graph(graph, _key, _is_safe_to_reorder)
    assert renamed["custom"] == "add"
    assert renamed["custom_1"] == "add_1"
    assert first.name == "add"
    assert second.name == "add_1"

    future = graph.call_function(operator.add, (x, x))
    assert future.name == "add_2"


def test_default_canonical_key_preserves_placeholder_order():
    graph = Graph()
    first = graph.placeholder("first")
    second = graph.placeholder("second")
    result = graph.call_function(operator.add, (second, first))
    graph.output(result)

    canonicalize_graph(graph)
    assert list(graph.nodes)[:2] == [first, second]
    graph.lint()
