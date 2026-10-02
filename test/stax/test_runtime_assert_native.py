import operator

import pytest
import sympy

from tensorplay.graph import Graph, GraphModule
from tensorplay.graph.experimental.symbolic_shapes import DimDynamic, RuntimeAssert, ShapeEnv
from tensorplay.graph.passes.runtime_assert import insert_deferred_runtime_asserts


def _symbolic_placeholder():
    shape_env = ShapeEnv()
    symbol = shape_env.create_unspecified_symbol(5, "n", DimDynamic.UNBACKED)
    value = shape_env.create_symintnode(symbol, hint=5)
    graph = Graph()
    placeholder = graph.placeholder("n")
    placeholder.meta["val"] = value
    return shape_env, symbol, value, graph, placeholder


def test_materialize_symbolic_boolean_expression():
    _, symbol, value, graph, placeholder = _symbolic_placeholder()
    expression = sympy.And(sympy.Ge(symbol + 2, 7), sympy.Lt(symbol, 10))

    result = graph.materialize_symints([expression])[0]
    graph.output(result)
    graph.lint()

    assert result.op == "call_function"
    assert any(node.target is operator.ge for node in graph.nodes)
    assert any(node.target is operator.lt for node in graph.nodes)
    assert placeholder.meta["val"] is value

    graph_module = GraphModule({}, graph)
    assert graph_module(5) is True
    assert graph_module(2) is False


def test_insert_deferred_runtime_asserts_executes_native_check():
    shape_env, symbol, _, graph, placeholder = _symbolic_placeholder()
    graph.output(placeholder)
    graph_module = GraphModule({}, graph)
    expression = sympy.Ge(symbol, 3)
    shape_env.deferred_runtime_asserts[symbol].extend(
        [
            RuntimeAssert(expression, "n must be at least three"),
            RuntimeAssert(expression, "duplicate assertion"),
        ]
    )

    insert_deferred_runtime_asserts(graph_module, shape_env, "test")

    assertions = [
        node
        for node in graph_module.graph.nodes
        if node.meta.get("runtime_assert_expr") == expression
    ]
    assert len(assertions) == 1
    assert graph_module(5) == 5
    with pytest.raises(RuntimeError, match="n must be at least three"):
        graph_module(2)


def test_insert_deferred_runtime_asserts_removes_unused_range_constraints():
    from tensorplay.functional import sym_constrain_range

    shape_env, _, _, graph, placeholder = _symbolic_placeholder()
    range_node = graph.call_function(sym_constrain_range, (placeholder, 0, 10))
    graph.output(placeholder)
    graph_module = GraphModule({}, graph)

    insert_deferred_runtime_asserts(graph_module, shape_env, "test")

    assert range_node not in graph_module.graph.nodes
    graph_module.graph.lint()
