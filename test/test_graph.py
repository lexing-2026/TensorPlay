"""Tests for the tensorplay.graph facade: capture primitives, visualization
"""

import operator

import pytest
import shutil

import tensorplay as tp
import tensorplay.graph as tpg
from tensorplay.graph import Graph, GraphCaptureError, Tracer


class TinyBlock(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = tp.nn.Linear(3, 3)
        self.act = tp.nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class Stacked(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.block_a = TinyBlock()
        self.block_b = TinyBlock()

    def forward(self, x):
        return self.block_b(self.block_a(x))


# ---------------------------------------------------------------------------
# Graph primitives: naming, output semantics, erase/replace
# ---------------------------------------------------------------------------

def test_create_node_derives_semantic_unique_names():
    g = Graph()
    x = g.placeholder("x")
    a = g.call_function(operator.add, (x, x))
    b = g.call_function(operator.add, (a, x))
    conv = g.call_module("backbone.conv1", (x,))
    w = g.get_attr("backbone.conv1.weight")
    view = g.create_node("call_method", "view", (conv,), {"shape": (-1,)})
    g.output(view)

    assert [n.name for n in g.placeholders] == ["x"]
    names = [n.name for n in g.nodes]
    assert names.count(a.name) == 1 and a.name == "add"
    assert b.name != a.name and b.name.startswith("add_")
    assert conv.name == "conv1" and w.name == "weight"
    assert view.name == "view"
    g.lint()


def test_explicit_names_are_sanitized_and_uniquified():
    g = Graph()
    x = g.placeholder("x")
    auto = g.call_function(operator.add, (x, x))
    explicit = g.create_node(
        "call_function", operator.add, (auto, x), name="add"
    )
    assert auto.name == "add"
    assert explicit.name == "add_0"
    weird = g.create_node("call_function", operator.add, (explicit, x), name="1bad name")
    assert weird.name == "_1bad_name"
    g.output(auto)
    g.lint()


def test_output_replaces_previous_output_node():
    g = Graph()
    x = g.placeholder("x")
    y = g.call_function(operator.neg, (x,))
    first = g.output(y)
    second = g.output(x)

    assert len(g.outputs) == 1
    assert second is g.output_node
    assert first.graph is None
    assert [n.name for n in g.nodes] == ["x", "neg", "output"]


def test_interpret_and_recompile_agree_on_single_output():
    def fn(x, scale=2.0):
        return x * scale

    gm = Tracer().trace(fn)
    eager = fn(tp.tensor([1.0, 2.0]))
    interpreted = gm.forward(tp.tensor([1.0, 2.0]))
    gm.recompile()
    compiled = gm.forward(tp.tensor([1.0, 2.0]))

    assert tp.allclose(interpreted, eager)
    assert tp.allclose(compiled, interpreted)


def test_erase_node_rules_and_dce_consistency():
    g = Graph()
    x = g.placeholder("x")
    dead = g.call_function(operator.neg, (x,))
    live = g.call_function(operator.abs, (x,))
    g.output(live)

    with pytest.raises(GraphCaptureError, match="user"):
        live.erase_node()

    assert dead.erase_node() is None
    assert dead.graph is None
    assert dead not in g.nodes
    with pytest.raises(GraphCaptureError, match="already been erased"):
        dead.erase_node()
    g.lint()


def test_replace_all_uses_with_and_topological_guard():
    g = Graph()
    x = g.placeholder("x")
    neg = g.call_function(operator.neg, (x,))
    abs_ = g.call_function(operator.abs, (neg,))
    mul = g.call_function(operator.mul, (abs_, abs_))
    g.output(mul)

    with pytest.raises(GraphCaptureError, match="later in the graph"):
        x.replace_all_uses_with(mul)

    rewritten = neg.replace_all_uses_with(x)
    assert rewritten == 1
    assert abs_.args[0] is x
    assert not neg.users
    g.lint()

    assert g.eliminate_dead_code() == 1
    with pytest.raises(GraphCaptureError, match="already been erased"):
        neg.erase_node()


def test_inserting_before_keeps_creation_order():
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)

    with g.inserting_before(out):
        a = g.call_function(operator.add, (x, x))
        b = g.call_function(operator.mul, (a, x))
    assert [n.name for n in g.nodes] == ["x", "add", "mul", "output"]

    # Insert point restored after the block: new nodes append again.
    c = g.call_function(operator.neg, (b,))
    assert g.nodes[-1] is c

    with g.inserting_before(None):
        first = g.get_attr("w")
    assert g.nodes[0] is first
    g.lint()


def test_inserting_after_splices_in_reverse_creation_order():
    # creation sequence appears reversed in the graph.
    g = Graph()
    x = g.placeholder("x")

    with g.inserting_after(x):
        a = g.call_function(operator.add, (x, x))
        b = g.call_function(operator.mul, (x, x))
    assert [n.name for n in g.nodes] == ["x", "mul", "add"]

    with g.inserting_after(None):
        tail = g.call_function(operator.neg, (a,))
    assert g.nodes[-1] is tail
    g.output(tail)
    g.lint()


def test_insert_point_rejects_foreign_anchor():
    g, other = Graph(), Graph()
    foreign = other.placeholder("x")
    with pytest.raises(GraphCaptureError, match="not part of this graph"):
        g.inserting_before(foreign)
    with pytest.raises(GraphCaptureError, match="not part of this graph"):
        g.inserting_after(foreign)


def test_graph_copy_remaps_through_val_map():
    src = Graph()
    px = src.placeholder("x")
    py = src.placeholder("y")
    sub = src.call_function(operator.sub, (px, py))
    sub.meta["origin"] = "replacement"
    src.output(sub)

    dst = Graph()
    x = dst.placeholder("x")
    y = dst.placeholder("y")
    out = dst.output(y)

    val_map = {px: x, py: y}
    with dst.inserting_before(out):
        copied = dst.graph_copy(src, val_map)

    assert copied.op == "call_function" and copied.target is operator.sub
    assert copied.args == (x, y)
    assert copied.meta == {"origin": "replacement"}
    assert val_map[sub] is copied
    assert [n.name for n in dst.nodes] == ["x", "y", "sub", "output"]
    dst.lint()


def test_subgraph_rewrite_addmul_to_sub():
    #   f(x, y): x = x + y; x = x * y  ->  return x - y
    def f(x, y):
        x = x + y
        x = x * y
        return x

    def replacement(x, y):
        return x - y

    sample = {"x": tp.tensor([1.0, 2.0]), "y": tp.tensor([0.5, 0.5])}
    gm = Tracer().trace(f, sample_inputs=sample)
    replacement_graph = Tracer().trace(replacement, sample_inputs=sample).graph

    graph = gm.graph
    mul = next(
        n for n in graph.nodes
        if n.op == "call_function" and n.target is operator.mul
    )
    add = mul.args[0]
    assert add.op == "call_function" and add.target is operator.add
    x_node, y_node = add.args
    assert mul.args[1] is y_node

    repl_phs = [n for n in replacement_graph.nodes if n.op == "placeholder"]
    val_map = dict(zip(repl_phs, (x_node, y_node)))
    insert_point = min(mul.users, key=graph.nodes.index)
    with graph.inserting_before(insert_point):
        copied = graph.graph_copy(replacement_graph, val_map)
    mul.replace_all_uses_with(copied)
    for node in (mul, add):
        node.erase_node()
    graph.lint()

    assert [(n.op, n.target) for n in graph.nodes] == [
        ("placeholder", "x"),
        ("placeholder", "y"),
        ("call_function", operator.sub),
        ("output", "output"),
    ]
    x, y = tp.tensor([3.0, 4.0]), tp.tensor([1.0, 2.0])
    assert tp.allclose(gm(x, y), x - y)


# ---------------------------------------------------------------------------
# Tracer: concrete_args, leaf modules, qualname map
# ---------------------------------------------------------------------------

def test_concrete_args_specialize_parameters_away():
    def fn(x, scale):
        return x * scale

    tracer = Tracer(concrete_args={"scale": 2.5})
    gm = tracer.trace(fn)
    assert [p.name for p in gm.graph.placeholders] == ["x"]
    assert list(gm.signature.parameters) == ["x"]

    result = gm.forward(tp.tensor([1.0]))
    assert tp.allclose(result, fn(tp.tensor([1.0]), 2.5))

    with pytest.raises(GraphCaptureError, match="not.*parameters"):
        Tracer(concrete_args={"missing": 1}).trace(fn)


class LeafOnlyTracer(Tracer):
    def is_leaf_module(self, module, qualified_name):
        return next(module.named_children(), None) is None


def test_default_tracer_inlines_children_like_dynamo():
    gm = Tracer().trace(TinyBlock())
    ops = {node.op for node in gm.graph.nodes}
    assert "call_module" not in ops


def test_leaf_tracer_emits_call_module_with_qualnames():
    model = Stacked()
    tracer = LeafOnlyTracer()
    gm = tracer.trace(model)

    call_modules = {
        node.target: node for node in gm.graph.nodes if node.op == "call_module"
    }
    assert set(call_modules) == {
        "block_a.conv",
        "block_a.act",
        "block_b.conv",
        "block_b.act",
    }
    recorded = sorted(tracer.node_to_qualname.values())
    assert recorded == [
        "block_a.act",
        "block_a.conv",
        "block_b.act",
        "block_b.conv",
    ]
    # Parameters below leaf modules must not dangle as get_attr inputs.
    assert all(node.op != "get_attr" for node in gm.graph.nodes)


def test_shared_leaf_module_gets_disambiguated_qualnames():
    class Shared(tp.nn.Module):
        def __init__(self):
            super().__init__()
            inner = TinyBlock()
            self.first = inner
            self.second = inner

        def forward(self, x):
            return self.second(self.first(x))

    tracer = LeafOnlyTracer()
    tracer.trace(Shared())
    quals = sorted(tracer.node_to_qualname.values())
    assert quals == [
        "first.act",
        "first.act_0",
        "first.conv",
        "first.conv_0",
    ]


# ---------------------------------------------------------------------------
# Facade surface: wrap, re-exports
# ---------------------------------------------------------------------------

def test_wrap_is_identity_and_supports_string_form():
    def fn():
        pass

    assert tpg.wrap(fn) is fn
    assert tpg.wrap("fn_name")(fn) is fn
    assert tpg.wrap()(fn) is fn


def test_graph_namespace_exports_graph_symbols():
    assert tpg.Graph is Graph
    assert tpg.Tracer is Tracer
    assert not hasattr(tpg, "PassManager")
    assert not hasattr(tpg, "ShapeProp")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_get_graph_node_names_merges_node_names_and_module_paths():
    train_names, eval_names = tpg.get_graph_node_names(Stacked())
    assert train_names == eval_names
    for expected in ("block_a.conv", "block_b.act", "x"):
        assert expected in train_names
    assert train_names == sorted(train_names)


def test_feature_extractor_returns_selected_intermediates():
    model = Stacked().eval()
    extractor = tpg.create_feature_extractor(model, ["block_a.act"]).eval()

    x = tp.randn(2, 3)
    features = extractor(x)
    reference = model.block_a.act(model.block_a.conv(x))
    assert tp.allclose(features, reference)


def test_feature_extractor_supports_dict_renaming_and_multiple_outputs():
    model = Stacked().eval()
    extractor = tpg.create_feature_extractor(
        model, {"block_a.act": "feat_a", "block_b.act": "feat_b"}
    ).eval()

    x = tp.randn(2, 3)
    feat_a, feat_b = extractor(x)
    assert tp.allclose(feat_a, model.block_a.act(model.block_a.conv(x)))
    assert tp.allclose(feat_b, model(x))
    with pytest.raises(ValueError, match="Two return nodes"):
        tpg.create_feature_extractor(
            model, {"block_a.act": "same", "block_b.act": "same"}
        )


def test_feature_extractor_state_dict_stays_compatible():
    model = Stacked()
    extractor = tpg.create_feature_extractor(model, ["block_b.conv"])

    expected_keys = {
        "block_a.conv.weight",
        "block_a.conv.bias",
        "block_b.conv.weight",
        "block_b.conv.bias",
    }
    assert set(extractor.state_dict().keys()) == expected_keys
    if hasattr(tp, "__future__"):
        extractor.load_state_dict(model.state_dict(), strict=False)

    model.eval()
    extractor.eval()
    x = tp.randn(2, 3)
    assert tp.allclose(extractor(x), model.block_b.conv(model.block_a(x)))


def test_feature_extractor_validates_requested_nodes():
    model = TinyBlock()
    with pytest.raises(ValueError, match="Available nodes"):
        tpg.create_feature_extractor(model, ["nope"])
    with pytest.raises(ValueError, match="placeholder"):
        tpg.create_feature_extractor(model, ["x"])
    with pytest.raises(ValueError, match="either return_nodes or both"):
        tpg.create_feature_extractor(
            model, train_return_nodes=["act"], eval_return_nodes=None
        )


class ModeSensitive(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = tp.nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(x)


def test_dual_graph_switches_between_train_and_eval():
    model = ModeSensitive()
    extractor = tpg.create_feature_extractor(model, ["dropout"])

    x = tp.ones(64)
    extractor.train()
    outputs = {tuple(extractor(x).tolist()) for _ in range(8)}
    assert len(outputs) > 1  # stochastic in train mode

    extractor.eval()
    eval_out = extractor(x)
    for _ in range(4):
        assert tp.allclose(extractor(x), eval_out)
    assert tp.allclose(eval_out, x)  # eval-mode dropout is identity

    assert extractor.graph is not None
    extractor.train()
    assert extractor.graph is not extractor._extractor_executors["eval"].graph


class ModeBranching(tp.nn.Module):
    def forward(self, x):
        if self.training:
            return x * 2
        return x + 1


def test_get_graph_node_names_warns_when_modes_diverge():
    with pytest.warns(UserWarning, match="differ"):
        tpg.get_graph_node_names(ModeBranching())


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _sample_graph():
    g = Graph()
    x = g.placeholder("x")
    w = g.get_attr("backbone.weight")
    h = g.call_function(operator.add, (x, w))
    s = g.call_method("relu", (h,))
    head = g.call_module("classifier.head", (s,))
    g.output(head)
    return g


def test_to_dot_contains_every_node_and_edge():
    dot = _sample_graph().to_dot()
    assert dot.startswith("digraph")
    for fragment in (
        '"x"', '"weight"', '"add"', '"relu"', '"head"',
        'operator.add', 'backbone.weight', 'classifier.head',
        '"x" -> "add";', '"head" -> "output";',
    ):
        assert fragment in dot


def test_draw_renders_png_when_graphviz_available(tmp_path):
    g = _sample_graph()
    target = tmp_path / "graph.png"
    if shutil.which("dot") is None:
        with pytest.raises(RuntimeError, match="Graphviz"):
            g.draw(str(target))
        assert (tmp_path / "graph.gv").exists()
        return
    rendered = g.draw(str(target))
    assert str(rendered).endswith(".png")
    assert target.stat().st_size > 0


def test_compile_entrypoint_unaffected_by_graph_changes():
    def fn(x):
        return x * 2 + 1

    compiled = tp.compile(fn)
    assert tp.allclose(compiled(tp.tensor([2.0])), tp.tensor([5.0]))
