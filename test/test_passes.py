"""Tests for the compiler pass infrastructure (P0 of L2)."""

import inspect
import operator

import pytest

import tensorplay as tp
from tensorplay.graph import Graph, GraphModule, Tracer
from tensorplay.graph.passes import (
    ConstFold,
    DeadCodeElimination,
    PassManager,
    PassResult,
    ShapeProp,
)
import tensorplay.graph.passes as tgp


def _gm_from_graph(g, params=("x",)):
    signature = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in params
        ]
    )
    return GraphModule(None, g, signature)


def test_const_fold_folds_scalar_subgraphs_and_erases():
    g = Graph()
    x = g.placeholder("x")
    k1 = g.call_function(operator.add, (2, 3))
    y = g.call_function(operator.mul, (x, k1))
    k2 = g.call_function(operator.getitem, ((10, 20), 1))
    z = g.call_function(operator.add, (y, k2))
    g.output(z)

    gm = _gm_from_graph(g)
    result = PassManager([ConstFold()])(gm)

    assert result.modified is True
    names = [n.name for n in gm.graph.nodes]
    assert names == ["x", "mul", "add_0", "output"]
    assert result.graph_module.graph.nodes[-2].args == (y, 20)

    out = gm.forward(tp.tensor([1.0, 2.0]))
    assert out.tolist() == [25.0, 30.0]


def test_const_fold_keeps_tensor_ops_and_runtime_errors():
    g = Graph()
    x = g.placeholder("x")
    scaled = g.call_function(operator.mul, (2, x))
    boom = g.call_function(operator.floordiv, (1, 0))
    g.output(scaled)

    gm = _gm_from_graph(g)
    PassManager([ConstFold()])(gm)

    # Tensor-touching and runtime-invalid nodes must survive untouched.
    assert {n.op for n in gm.graph.nodes} >= {"call_function"}
    assert any(n.target is operator.floordiv for n in gm.graph.nodes)
    with pytest.raises(ZeroDivisionError):
        gm._interpret(tp.tensor([1.0]))


def test_pass_manager_runs_to_fixpoint_and_is_idempotent():
    def build():
        g = Graph()
        x = g.placeholder("x")
        dead = g.call_function(operator.neg, (x,))
        live = g.call_function(operator.abs, (x,))
        k = g.call_function(operator.add, (2, 2))
        used = g.call_function(operator.mul, (live, 4))
        g.output(live)
        return g

    gm = _gm_from_graph(build())
    pm = PassManager([DeadCodeElimination(), ConstFold()])
    first = pm(gm)
    assert first.modified is True
    assert [n.name for n in gm.graph.nodes] == ["x", "abs", "output"]

    second = pm(gm)
    assert second.modified is False


def test_pass_result_contract_and_custom_pass():
    seen = []

    class CountingPass:
        def __init__(self):
            self.calls = 0

        def __call__(self, graph_module):
            self.calls += 1
            seen.append(self.calls)
            return PassResult(graph_module, self.calls < 3)

    counting = CountingPass()
    g = Graph()
    x = g.placeholder("x")
    g.output(x)
    result = PassManager([counting])(_gm_from_graph(g))

    assert result.modified is True
    assert counting.calls == 3  # fixpoint reached on the unmodified round
    with pytest.raises(TypeError):
        PassManager()("not a graph")


def test_shape_prop_records_values_and_shapes_for_dot():
    def fn(x):
        return (x * 2 + 1).relu()

    x = tp.tensor([-1.0, 0.5])
    tracer = Tracer()
    gm = tracer.trace(fn)

    result = PassManager([ShapeProp((x,))])(gm)
    assert result.modified is False

    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        assert "val" in node.meta
    mul_node = next(n for n in gm.graph.nodes if getattr(n.target, "__name__", None) == "mul")
    add_node = next(n for n in gm.graph.nodes if getattr(n.target, "__name__", None) == "add")
    assert mul_node.meta["tensor_shape"] == tuple(x.shape)
    assert add_node.meta["val"].tolist() == (x * 2 + 1).tolist()

    dot = gm.graph.to_dot()
    assert 'tooltip="(2,)"' in dot
    assert mul_node.name in dot and add_node.name in dot


def test_graph_pass_namespace_exports_pass_infra():
    for name in ("PassManager", "PassBase", "PassResult", "ConstFold",
                 "DeadCodeElimination", "ShapeProp"):
        assert hasattr(tgp, name)


def test_graph_namespace_keeps_passes_in_the_passes_package():
    import tensorplay.graph as tpg
    assert not hasattr(tpg, "PassManager")
    assert not hasattr(tpg, "ShapeProp")
    assert tgp.PassManager is not None
    assert tgp.ShapeProp is not None
