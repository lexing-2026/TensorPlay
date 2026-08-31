import operator

import tensorplay as tp
from tensorplay.graph import Graph, GraphModule
from tensorplay.graph.interpreter import Transformer
from tensorplay.graph.proxy import GraphAppendingTracer, Proxy
from tensorplay.graph._graph_pickler import (
    GraphPickler,
    Options,
    patch_pytree_map_over_slice,
)
from tensorplay.graph._lazy_graph_module import (
    _LazyGraphModule,
    _get_graph_module_cls,
    _unwrap_lazy_graph_module,
    _use_lazy_graph_module,
)


def _make_graph() -> GraphModule:
    graph = Graph()
    value = graph.placeholder("value")
    sliced = graph.call_function(operator.getitem, (value, slice(0, 2)))
    shifted = graph.call_function(operator.add, (sliced, 1))
    shifted.meta["kept"] = {"size": 2}
    shifted.meta["stack_trace"] = "process-local"
    graph.output(shifted)
    return GraphModule(None, graph)


def test_graph_pickler_roundtrip_rebuilds_topology_and_metadata() -> None:
    module = _make_graph()
    restored = GraphPickler.loads(GraphPickler.dumps(module))

    assert isinstance(restored, GraphModule)
    assert restored(tp.tensor([1.0, 2.0, 3.0])).tolist() == [2.0, 3.0]
    nodes = list(restored.graph.nodes)
    assert nodes[2].args[0] is nodes[1]
    assert nodes[2].meta == {"kept": {"size": 2}}
    assert restored.graph.output_node.args[0] is nodes[2]


def test_graph_pickler_roundtrip_supports_graph_root_and_c_targets() -> None:
    module = _make_graph()
    graph = GraphPickler.loads(GraphPickler.dumps(module.graph))

    assert isinstance(graph, Graph)
    assert graph.output_node.args[0] is list(graph.nodes)[2]

    c_graph = Graph()
    created = c_graph.call_function(tp.empty, ((2,),))
    c_graph.output(created)
    restored_c_graph = GraphPickler.loads(GraphPickler.dumps(c_graph))
    assert restored_c_graph.nodes[0].target is not None
    assert restored_c_graph.nodes[0].target.__name__ == "empty"


def test_graph_pickler_options_and_debug_path() -> None:
    graph = _make_graph().graph
    raw_node = next(iter(graph.nodes))

    try:
        GraphPickler.dumps(raw_node)
    except AssertionError as exc:
        assert "raw graph node" in str(exc)
    else:
        raise AssertionError("raw nodes must be rejected by default")

    assert GraphPickler.loads(
        GraphPickler.dumps(raw_node, Options(ignore_raw_node=True))
    ) is None
    assert GraphPickler.debug_dumps(
        {"good": 1, "bad": [lambda value: value]}, verbose=False
    ) == "root['bad'][0]"


def test_graph_pickler_slice_registration_is_scoped() -> None:
    from tensorplay.utils import _pytree

    assert slice not in _pytree.SUPPORTED_NODES
    with patch_pytree_map_over_slice():
        assert slice in _pytree.SUPPORTED_NODES
    assert slice not in _pytree.SUPPORTED_NODES


def test_lazy_graph_module_defers_and_recompiles_on_demand() -> None:
    module = _LazyGraphModule(None, _make_graph().graph)
    assert module._compiled_forward is None
    assert module._lazy_needs_recompile
    assert module(tp.tensor([1.0, 2.0, 3.0])).tolist() == [2.0, 3.0]
    assert not module._lazy_needs_recompile
    assert module._compiled_forward is not None

    plain = _unwrap_lazy_graph_module(module)
    assert type(plain) is GraphModule
    assert plain(tp.tensor([2.0, 3.0, 4.0])).tolist() == [3.0, 4.0]


def test_lazy_graph_module_selection_is_scoped() -> None:
    assert _get_graph_module_cls() is GraphModule
    with _use_lazy_graph_module(True):
        assert _get_graph_module_cls() is _LazyGraphModule
    assert _get_graph_module_cls() is GraphModule


def test_transformer_preserves_placeholder_defaults_and_node_metadata() -> None:
    graph = Graph()
    value = graph.placeholder("value", type_expr=int, default_value=3)
    value.meta["source"] = "input"
    result = graph.call_function(operator.add, (value, 1), type_expr=int)
    result.meta["source"] = "add"
    graph.output(result)

    transformed = Transformer(GraphModule(None, graph)).transform()

    placeholder = transformed.graph.placeholders[0]
    output_value = list(transformed.graph.nodes)[1]
    assert placeholder.args == (3,)
    assert placeholder.type is int
    assert placeholder.meta["source"] == "input"
    assert output_value.type is int
    assert output_value.meta["source"] == "add"
    assert transformed() == 4


def test_graph_appending_tracer_builds_native_proxy_nodes() -> None:
    graph = Graph()
    value = graph.placeholder("value")
    proxy = Proxy(value, GraphAppendingTracer(graph))
    result = (proxy + 1).node
    graph.output(result)

    assert result.op == "call_function"
    assert result.args == (value, 1)
