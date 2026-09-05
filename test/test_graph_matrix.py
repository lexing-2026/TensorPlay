"""Standalone test matrix for the graph subsystem.

The cases are grouped along the categories requested for the graph module:
node topology, code generation, module execution, parameters and buffers,
control flow, pickling, graph rewriting, subgraph splitting, shape
propagation and error stacks.
"""

import operator
import pickle

import pytest

import tensorplay as tp
import tensorplay.graph as tpg
from tensorplay.graph import (
    Graph,
    GraphCaptureError,
    GraphModule,
    Interpreter,
    symbolic_trace,
)
from tensorplay.graph.passes import ShapeProp, split_by_tags, split_module


class TinyBlock(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = tp.nn.Linear(3, 3)
        self.act = tp.nn.ReLU()

    def forward(self, x):
        return self.act(self.linear(x))


class Stacked(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.block_a = TinyBlock()
        self.block_b = TinyBlock()

    def forward(self, x):
        return self.block_b(self.block_a(x))


def _build():
    """Return (traced Stacked, eager Stacked)."""
    module = Stacked()
    return symbolic_trace(module), module


def _samples(*shapes):
    return [tp.randn(shape) for shape in shapes]


# ---------------------------------------------------------------------------
# 1. Node topology
# ---------------------------------------------------------------------------


class TestNodeTopology:
    def test_placeholder_and_output_maintain_single_output(self):
        g = Graph()
        x = g.placeholder("x")
        y = g.call_function(operator.neg, (x,))
        out = g.output(y)
        assert [n.op for n in g.nodes] == ["placeholder", "call_function", "output"]
        assert out.args == (y,)
        assert y.users == {out}

    def test_inserting_before_and_after_keep_topo(self):
        g = Graph()
        x = g.placeholder("x")
        first = g.call_function(operator.neg, (x,))
        g.output(first)
        with g.inserting_before(first):
            mid = g.call_function(operator.abs, (x,))
        first.args = (mid,)
        order = [n.name for n in g.nodes]
        assert order.index("abs") < order.index("neg")
        g.lint()

    def test_all_input_nodes_and_users_consistency(self):
        g = Graph()
        x = g.placeholder("x")
        y = g.call_function(operator.neg, (x,))
        z = g.call_function(operator.abs, (y,))
        g.output(z)
        assert z.all_input_nodes == [y]
        assert y.all_input_nodes == [x]
        assert x.users == {y}
        assert y.users == {z}

    def test_replace_input_with_rewires_single_consumer(self):
        g = Graph()
        x = g.placeholder("x")
        b = g.call_function(operator.neg, (x,))
        a = g.call_function(operator.abs, (b,))
        g.output(a)
        a.replace_input_with(b, x)  # x precedes b in emission order
        assert a.args == (x,)
        g.lint()

    def test_replace_all_uses_with_removes_dead_node(self):
        g = Graph()
        x = g.placeholder("x")
        a = g.call_function(operator.neg, (x,))
        b = g.call_function(operator.abs, (x,))
        out = g.output(a)
        a.replace_all_uses_with(b)
        assert out.args == (b,)
        assert g.eliminate_dead_code() == 1
        assert [n.name for n in g.nodes] == ["x", "abs", "output"]

    def test_erase_node_fails_while_used(self):
        g = Graph()
        x = g.placeholder("x")
        y = g.call_function(operator.neg, (x,))
        g.output(y)
        with pytest.raises(GraphCaptureError):
            y.erase_node()

    def test_erase_removes_users_and_inputs(self):
        g = Graph()
        x = g.placeholder("x")
        y = g.call_function(operator.neg, (x,))
        g.output(y)
        y.replace_all_uses_with(x)
        g.eliminate_dead_code()
        assert [n.op for n in g.nodes] == ["placeholder", "output"]
        assert x.users == {g.outputs[-1]}

    def test_lint_detects_wrong_topological_order(self):
        g = Graph()
        x = g.placeholder("x")
        g.output(x)
        consumer = g.create_node("call_function", operator.neg, (None,))
        producer = g.create_node("call_function", operator.abs, (x,))
        consumer.args = (producer,)
        with pytest.raises(GraphCaptureError, match="topologically"):
            g.lint()

    def test_insertion_point_rejects_foreign_anchor(self):
        g1, g2 = Graph(), Graph()
        x1 = g1.placeholder("x")
        x2 = g2.placeholder("x")
        neg = g1.call_function(operator.neg, (x1,))
        g1.output(neg)
        with pytest.raises(GraphCaptureError):
            with g1.inserting_before(x2):
                pass
        with pytest.raises(GraphCaptureError):
            with g1.inserting_after(x2):
                pass
# ---------------------------------------------------------------------------
# 2. Code generation
# ---------------------------------------------------------------------------


class TestCodeGeneration:
    def test_generated_code_roundtrips_operator_targets(self):
        g = Graph()
        x = g.placeholder("x")
        h = g.call_function(operator.add, (x, 1))
        g.output(h)
        gm = GraphModule(None, g)
        assert gm(tp.tensor([1.0])) == tp.tensor([2.0])

    def test_generated_code_uses_public_operator_facade(self):
        g = Graph()
        x = g.placeholder("x")
        h = g.call_function(operator.not_, (x,))
        g.output(h)
        gm = GraphModule(None, g)
        assert "_operator" not in gm.code
        assert "operator" in gm.code
        # private accelerator module must not leak into code globals
        gv = gm.forward.__globals__
        assert "_operator" not in gv

    def test_dot_output_uses_public_facade(self):
        g = Graph()
        x = g.placeholder("x")
        h = g.call_function(operator.le, (x, 3))
        g.output(h)
        dot = g.to_dot()
        assert "_operator" not in dot
        assert "operator.le" in dot

    def test_tabular_printing_contains_all_nodes(self):
        g = Graph()
        x = g.placeholder("x")
        h = g.call_function(operator.neg, (x,))
        g.output(h)
        rows = g.print_tabular()
        assert "placeholder" in rows and "call_function" in rows

    def test_str_rendering_lists_nodes_in_order(self):
        g = Graph()
        x = g.placeholder("x")
        y = g.call_function(operator.neg, (x,))
        g.output(y)
        text = str(g)
        assert text.index("%x") < text.index("%neg") < text.index("return")

    def test_print_readable_matches_generated_code(self):
        traced, _ = _build()
        readable = traced.print_readable(print_output=False)
        assert readable.startswith("class ")
        assert "def forward(self" in readable

    def test_get_attr_and_call_module_paths_render(self):
        traced, _ = _build()
        code = traced.code
        assert "self.block_a.linear" in code or "self.block_a" in code
        assert "linear" in code


# ---------------------------------------------------------------------------
# 3. Module execution
# ---------------------------------------------------------------------------


class TestModuleExecution:
    def test_traced_module_matches_eager(self):
        traced, eager = _build()
        x = _samples((2, 3))[0]
        assert tp.allclose(traced(x), eager(x))

    def test_parameter_and_buffer_values_shared(self):
        traced, eager = _build()
        eager.block_a.linear.weight.data += 1.0
        x = _samples((2, 3))[0]
        assert tp.allclose(traced(x), eager(x))

    def test_interpreter_matches_eager(self):
        traced, eager = _build()
        x = _samples((2, 3))[0]
        result = Interpreter(traced).run(x)
        assert tp.allclose(result, eager(x))

    def test_interpreter_can_swap_operations(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return (x + 1.0).relu()

        traced = symbolic_trace(M())
        x = tp.full((2, 3), 3.0)
        expected = tp.full((2, 3), 3.0)  # add(x,1) -> mul(x,1) == x
        seen = []

        class AddToMul(Interpreter):
            def call_function(self, target, args, kwargs):
                seen.append(target)
                if target is operator.add:
                    return super().call_function(operator.mul, args, kwargs)
                return super().call_function(target, args, kwargs)

        result = AddToMul(traced).run(x)
        assert tp.allclose(result, expected)
        assert operator.add in seen


# ---------------------------------------------------------------------------
# 4. Parameters and buffers
# ---------------------------------------------------------------------------


class TestParametersAndBuffers:
    def test_get_attr_nodes_resolve_to_parameters(self):
        eager = Stacked()
        traced = symbolic_trace(eager)
        attrs = [n for n in traced.graph.nodes if n.op == "get_attr"]
        assert attrs
        interp = Interpreter(traced)
        for node in attrs:
            value = interp.fetch_attr(node.target)
            eager_value = eager
            for part in node.target.split("."):
                eager_value = getattr(eager_value, part)
            assert tp.allclose(value, eager_value)

    def test_feature_extractor_state_dict_matches_root(self):
        eager = Stacked()
        extractor = tpg.create_feature_extractor(eager, ["block_a.act"]).eval()
        expected = {"block_a.linear.weight", "block_a.linear.bias"}
        assert set(extractor.state_dict()) == expected
        x = _samples((2, 3))[0]
        reference = eager.block_a.act(eager.block_a.linear(x))
        assert tp.allclose(extractor(x), reference)

    def test_submodule_deletion_keeps_used_modules(self):
        from tensorplay.graph.tracer import Tracer

        class LeafTracer(Tracer):
            def is_leaf_module(self, mod, qualified_name):
                return isinstance(mod, TinyBlock)

        eager = Stacked()
        traced = LeafTracer().trace(eager)
        before = {name for name, _ in traced.named_modules()}
        traced.delete_all_unused_submodules()
        after = {name for name, _ in traced.named_modules()}
        assert "block_a" in after and "block_a.linear" in after
        assert "block_b" in after and "block_b.linear" in after
        assert after <= before

    def test_parameter_values_shared_with_root(self):
        # The traced module reads get_attr targets through the root module, so
        # parameters remain a single source of truth: mutating the eager module
        # must be visible through the traced computation.
        traced, eager = _build()
        eager.block_a.linear.weight.data += 1.0
        x = _samples((2, 3))[0]
        assert tp.allclose(traced(x), eager(x))

    def test_buffer_persistent_and_non_persistent_roundtrip(self):
        from tensorplay.graph.tracer import Tracer

        class WithBuffer(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", tp.randn(2, 2), persistent=True)
                self.register_buffer("scratch", tp.randn(3), persistent=False)

            def forward(self, x):
                return self.buf + x

        module = WithBuffer()
        traced = Tracer().trace(module)
        state = dict(traced.state_dict())
        assert "buf" in state
        assert "scratch" not in state
        buffers = {name for name, _ in traced.named_buffers()}
        assert {"buf", "scratch"} <= buffers


# ---------------------------------------------------------------------------
# 5. Control flow
# ---------------------------------------------------------------------------


class TestControlFlow:
    def test_concrete_bool_branch_inlined(self):
        class M(tp.nn.Module):
            def forward(self, x, flag=True):
                if flag:
                    return x.relu()
                return x.tanh()

        true_traced = symbolic_trace(M(), concrete_args={"flag": True})
        assert "relu" in true_traced.code and "tanh" not in true_traced.code
        false_traced = symbolic_trace(M(), concrete_args={"flag": False})
        assert "tanh" in false_traced.code and "relu" not in false_traced.code
        assert tp.allclose(true_traced(tp.tensor([-1.0, 2.0])), tp.tensor([0.0, 2.0]))
        assert tp.allclose(
            false_traced(tp.tensor([-1.0, 2.0])), tp.tanh(tp.tensor([-1.0, 2.0]))
        )

    def test_loop_over_concrete_counter_unrolled(self):
        class L(tp.nn.Module):
            def forward(self, x, steps=3):
                i = 0
                while i < steps:
                    x = x + 1
                    i = i + 1
                return x

        traced = symbolic_trace(L(), concrete_args={"steps": 3})
        assert traced.code.count("+ 1") == 3
        assert traced(tp.tensor([0.0])) == tp.tensor([3.0])

    def test_tensor_control_flow_rejected_at_trace_time(self):
        from tensorplay.graph.proxy import TraceError

        class Bad(tp.nn.Module):
            def forward(self, x):
                if x.numel() > 0:
                    return x.relu()
                return x.tanh()

        with pytest.raises(TraceError, match="control flow"):
            symbolic_trace(Bad())

    def test_default_arguments_become_placeholders(self):
        class M(tp.nn.Module):
            def forward(self, x, b=4, c=5):
                return x + b - c

        traced = symbolic_trace(M())
        assert traced(tp.tensor([1.0])) == tp.tensor([0.0])
        assert traced(tp.tensor([1.0]), 1, 1) == tp.tensor([1.0])


# ---------------------------------------------------------------------------
# 6. Pickling
# ---------------------------------------------------------------------------


class TestPickling:
    def test_pickle_roundtrip_graphmodule(self):
        traced, eager = _build()
        restored = pickle.loads(pickle.dumps(traced))
        assert type(restored).__name__ == "GraphModule"
        x = _samples((2, 3))[0]
        assert tp.allclose(restored(x), eager(x))

    def test_graph_pickler_roundtrip(self):
        import tensorplay.graph._graph_pickler as graph_pickler

        traced, eager = _build()
        payload = graph_pickler.GraphPickler.dumps(traced)
        restored = graph_pickler.GraphPickler.loads(payload)
        assert isinstance(restored, GraphModule)
        x = _samples((2, 3))[0]
        assert tp.allclose(restored(x), eager(x))

    def test_pickle_keeps_callable_targets(self):
        # Module classes must be referenceable at pickle time; module-level
        # helpers such as TinyBlock satisfy that requirement.
        module = TinyBlock()
        traced = symbolic_trace(module)
        restored = pickle.loads(pickle.dumps(traced))
        callable_nodes = [
            node.target
            for node in restored.graph.nodes
            if node.op == "call_function"
        ]
        assert callable_nodes
        assert all(callable(target) for target in callable_nodes)
        x = _samples((2, 3))[0]
        assert tp.allclose(restored(x), module(x))


# ---------------------------------------------------------------------------
# 7. Graph rewriting
# ---------------------------------------------------------------------------


class TestGraphRewriting:
    def test_replace_pattern_substitutes_add_with_sub(self):
        from tensorplay.graph import replace_pattern

        class M(tp.nn.Module):
            def forward(self, x):
                return x + 1

        class Pattern(tp.nn.Module):
            def forward(self, x):
                return x + 1

        class Replacement(tp.nn.Module):
            def forward(self, x):
                return x - 1

        traced = symbolic_trace(M())
        matches = replace_pattern(traced, Pattern(), Replacement())
        assert len(matches) == 1
        traced.recompile()
        assert "sub" in traced.code or "x - 1" in traced.code
        assert tp.allclose(traced(tp.tensor([3.0])), tp.tensor([2.0]))

    def test_transformer_can_swap_call_function(self):
        from tensorplay.graph import Transformer

        class M(tp.nn.Module):
            def forward(self, x):
                return x + 1

        traced = symbolic_trace(M())
        seen = []

        class AddToSub(Transformer):
            def call_function(self, target, args, kwargs):
                seen.append(target)
                if target is operator.add:
                    return self.tracer.create_proxy(
                        "call_function", operator.sub, args, kwargs
                    )
                return super().call_function(target, args, kwargs)

        rewritten = AddToSub(traced).transform()
        assert rewritten(tp.tensor([3.0])) == tp.tensor([2.0])
        assert operator.add in seen

    def test_rewriter_rename_nodes_and_canonicalize(self):
        from tensorplay.graph.passes import canonicalize_graph

        class M(tp.nn.Module):
            def forward(self, x):
                return x.neg().abs()

        traced = symbolic_trace(M())
        names_before = [n.name for n in traced.graph.nodes]
        canonicalize_graph(traced.graph)
        assert [n.name for n in traced.graph.nodes] == names_before  # stable
        traced.recompile()
        assert traced(tp.tensor([-2.0])) == tp.tensor([2.0])


# ---------------------------------------------------------------------------
# 8. Subgraph splitting
# ---------------------------------------------------------------------------


class TestSubgraphSplitting:
    def _trace_with_leaves(self, module):
        from tensorplay.graph.tracer import Tracer

        class LeafTracer(Tracer):
            def is_leaf_module(self, mod, qualified_name):
                del mod
                return qualified_name.endswith("block_a") or qualified_name.endswith(
                    "block_b"
                )

        return LeafTracer().trace(module)

    def test_split_module_creates_submodules(self):
        from tensorplay.graph.passes import split_module

        eager = Stacked()
        traced = self._trace_with_leaves(eager)

        def partition(node):
            if node.op == "call_module":
                return 0 if "block_a" in str(node.target) else 1
            return 0

        result = split_module(traced, eager, partition)
        children = [name for name, _ in result.named_children()]
        assert "submod_0" in children and "submod_1" in children
        x = _samples((2, 3))[0]
        assert tp.allclose(result(x), eager(x))

    def test_split_by_tags_returns_components(self):
        from tensorplay.graph.passes import split_by_tags

        eager = Stacked()
        traced = self._trace_with_leaves(eager)
        for node in traced.graph.nodes:
            if node.op == "call_module":
                tag = "first" if "block_a" in str(node.target) else "second"
                node.tag = tag
        result, mapping = split_by_tags(
            traced, ["first", "second"], return_fqn_mapping=True
        )
        assert sorted(dict(result.named_children())) == ["first", "second"]
        assert mapping
        x = _samples((2, 3))[0]
        assert tp.allclose(result(x), eager(x))


# ---------------------------------------------------------------------------
# 9. Shape propagation
# ---------------------------------------------------------------------------


class TestShapePropagation:
    def test_shape_prop_populates_tensor_meta(self):
        from tensorplay.graph.passes import ShapeProp

        traced, _ = _build()
        ShapeProp(traced).propagate(tp.randn(2, 3))
        metas = [node.meta.get("tensor_meta") for node in traced.graph.nodes]
        assert metas
        assert all(meta is not None for meta in metas)
        for meta in metas:
            assert isinstance(meta.shape, tuple)
        assert metas[0].shape == (2, 3)

    def test_shape_prop_tracks_dtype(self):
        from tensorplay.graph.passes import ShapeProp

        traced, _ = _build()
        ShapeProp(traced).propagate(tp.randn(2, 3))
        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target.__name__ == "linear":
                meta = node.meta["tensor_meta"]
                assert meta.shape == (2, 3)
                assert meta.dtype == tp.float32
                break
        else:
            raise AssertionError("no linear call_function node found")


# ---------------------------------------------------------------------------
# 10. Error stacks
# ---------------------------------------------------------------------------


class TestErrorStacks:
    def test_graph_execution_error_includes_generated_frame(self):
        import traceback

        class M(tp.nn.Module):
            def forward(self, x):
                return x.view(-1, 5)

        traced = symbolic_trace(M())
        formatted = ""
        try:
            traced(tp.randn(2, 3))
        except RuntimeError:
            formatted = traceback.format_exc()
        assert "tensorplay-generated" in formatted

    def test_node_stack_trace_preserved_during_trace(self):
        from tensorplay.graph.traceback import preserve_node_meta, set_stack_trace

        class M(tp.nn.Module):
            def forward(self, x):
                return x.relu()

        with preserve_node_meta():
            set_stack_trace(["user code frame A\n", "user code frame B\n"])
            traced = symbolic_trace(M())
        frames = [node.stack_trace for node in traced.graph.nodes if node.stack_trace]
        assert frames, "expected at least one node with a preserved stack trace"
        joined = "".join(frames)
        assert "user code frame B" in joined

    def test_trace_error_reports_data_dependent_control_flow(self):
        from tensorplay.graph.proxy import TraceError

        class Bad(tp.nn.Module):
            def forward(self, x):
                return x.relu() if x.item() > 0 else x.tanh()

        with pytest.raises(TraceError):
            symbolic_trace(Bad())

    def test_erase_used_node_stack_message(self):
        g = Graph()
        x = g.placeholder("x")
        y = g.call_function(operator.neg, (x,))
        g.output(y)
        with pytest.raises(GraphCaptureError, match="still has"):
            y.erase_node()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
