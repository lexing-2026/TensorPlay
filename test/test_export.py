import pytest

import tensorplay as tp
import tensorplay.export as tp_export
from tensorplay.graph import GraphCaptureError


class MLP(tp.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = tp.nn.Linear(4, 3)
        self.bn = tp.nn.BatchNorm1d(3)
        self.fc2 = tp.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.bn(self.fc1(x)))


def test_export_module_separates_parameters_buffers_and_inputs():
    model = MLP()
    x = tp.randn(8, 4)
    program = tp_export.export(model, x)

    assert program.graph_signature.user_inputs == ("x",)
    assert "fc1.weight" in program.graph_signature.parameters
    assert "fc2.bias" in program.graph_signature.parameters
    assert "bn.running_mean" in program.graph_signature.buffers
    # BatchNorm registers running stats as persistent buffers
    assert "bn.running_mean" not in program.graph_signature.non_persistent_buffers
    # the flat contract lifts state into placeholders: state first, user last,
    # and no attribute reads remain in the graph body
    assert not [n for n in program.graph.nodes if n.op == "get_attr"]
    placeholder_names = [n.name for n in program.graph.placeholders]
    spec_names = [spec.arg.name for spec in program.graph_signature.input_specs]
    assert placeholder_names == spec_names
    assert set(placeholder_names) >= {"p_fc1_weight", "b_bn_running_mean", "x"}


def test_export_non_persistent_buffer_flagged():
    class M(tp.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("stats", tp.zeros(3), persistent=False)

        def forward(self, x):
            return x + self.stats

    program = tp_export.export(M(), tp.zeros(2))
    assert "stats" in program.graph_signature.non_persistent_buffers
    assert program.graph_signature.buffers == ("stats",)


def test_exported_program_executes_and_matches_eager():
    model = MLP()
    x = tp.randn(6, 4)
    program = tp_export.export(model, x)

    eager = model(x).tolist()
    interpreted = program(x).tolist()
    standalone = program.module()(x).tolist()
    assert interpreted == eager
    assert standalone == eager


def test_export_function_with_defaults_and_kwargs():
    def fn(x, scale=2.0, *, offset=1.0):
        return x * scale + offset

    x = tp.tensor([1.0, 2.0])
    program = tp_export.export(fn, x, offset=5.0)
    assert program.example_inputs["scale"] == 2.0
    assert program.example_inputs["offset"] == 5.0
    assert program(x).tolist() == [7.0, 9.0]


def test_export_rejects_data_dependent_control_flow():
    def fn(x):
        if x.sum() > 0:
            return x * 2
        return x

    with pytest.raises(GraphCaptureError, match="control flow"):
        tp_export.export(fn, tp.ones(3))


def test_export_rejects_iteration_over_traced_value():
    def fn(x):
        total = x
        for chunk in x:
            total = total + chunk
        return total

    with pytest.raises(GraphCaptureError, match="iterat"):
        tp_export.export(fn, tp.ones(3))


def test_dynamic_shapes_validation_and_normalization():
    batch = tp_export.Dim("batch")
    x = tp.zeros(4, 8)
    program = tp_export.export(
        lambda t: t + 1,
        x,
        dynamic_shapes={"t": {0: batch, 1: 8}},
    )
    assert program.dynamic_shapes["t"][0] is batch
    assert program.dynamic_shapes["t"][1] == 8

    with pytest.raises(ValueError, match="does not match any argument"):
        tp_export.export(lambda t: t, x, dynamic_shapes={"zzz": {0: batch}})
    with pytest.raises(TypeError, match="int or Dim"):
        tp_export.export(lambda t: t, x, dynamic_shapes={"t": {0: "batch"}})


def test_print_readable_lists_signature():
    model = MLP()
    text = tp_export.export(model, tp.zeros(2, 4)).print_readable()
    assert "def forward" in text
    assert "fc1.weight" in text
    assert "user_inputs" in text


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_buffer_and_file(tmp_path):
    import io

    model = MLP()
    x = tp.randn(3, 4)
    program = tp_export.export(model, x)

    buffer = io.BytesIO()
    tp_export.save(program, buffer)
    buffer.seek(0)
    loaded = tp_export.load(buffer)
    assert loaded(x).tolist() == model(x).tolist()

    path = tmp_path / "model.pt2"
    tp_export.save(program, str(path))
    assert tp_export.load(str(path))(x).tolist() == model(x).tolist()


def test_save_load_extra_files_roundtrip():
    import io

    program = tp_export.export(MLP(), tp.randn(2, 4))
    buffer = io.BytesIO()
    tp_export.save(program, buffer, extra_files={"notes.txt": "hello"})
    buffer.seek(0)
    extra: dict = {}
    tp_export.load(buffer, extra_files=extra)
    assert extra["notes.txt"] == "hello"


def test_save_rejects_non_program():
    import io

    with pytest.raises(TypeError, match="ExportedProgram"):
        tp_export.save(object(), io.BytesIO())


# ---------------------------------------------------------------------------
# dynamic shapes: runtime assertions and constraints
# ---------------------------------------------------------------------------


def test_dynamic_shapes_shared_dim_inserts_equality_assert():
    batch = tp_export.Dim("batch")
    program = tp_export.export(
        lambda a, b: a.sum() + b.sum(),
        tp.randn(4, 3),
        tp.randn(4, 3),
        dynamic_shapes={"a": {0: batch}, "b": {0: batch}},
    )
    assert dict(program.range_constraints)["batch"] == {"min": 0, "max": None}
    assert program(tp.randn(2, 3), tp.randn(2, 3)).shape == ()
    with pytest.raises(RuntimeError, match="runtime assertion failed"):
        program(tp.randn(2, 3), tp.randn(3, 3))


def test_dynamic_shapes_range_assert_enforced():
    bounded = tp_export.Dim("bounded", min=2, max=8)
    program = tp_export.export(
        lambda t: t * 2, tp.randn(4), dynamic_shapes={"t": {0: bounded}}
    )
    assert dict(program.range_constraints)["bounded"] == {"min": 2, "max": 8}
    assert program(tp.randn(5)).shape == (5,)
    with pytest.raises(RuntimeError, match="runtime assertion failed"):
        program(tp.randn(1))


def test_dynamic_shapes_derived_dim_relation():
    base = tp_export.Dim("base", min=2)
    derived = base * 2
    program = tp_export.export(
        lambda a, b: a + b.sum(),
        tp.randn(4, 3),
        tp.randn(8, 3),
        dynamic_shapes={"a": {0: base}, "b": {0: derived}},
    )
    assert program(tp.randn(6, 3), tp.randn(12, 3)).shape == (6, 3)
    with pytest.raises(RuntimeError, match="runtime assertion failed"):
        program(tp.randn(6, 3), tp.randn(11, 3))


def test_dynamic_shapes_static_int_mismatch_raises():
    with pytest.raises(ValueError, match="static size"):
        tp_export.export(
            lambda t: t,
            tp.randn(4),
            dynamic_shapes={"t": {0: 5}},
        )


def test_dynamic_shapes_conflicting_dim_definitions_raise():
    a = tp_export.Dim("a", min=2)
    b = tp_export.Dim("a", min=4)
    with pytest.raises(ValueError, match="conflicting definitions"):
        tp_export.export(
            lambda x, y: x + y.sum(),
            tp.randn(4, 3),
            tp.randn(8, 3),
            dynamic_shapes={"x": {0: a}, "y": {0: b}},
        )


def test_refine_dynamic_shapes_from_suggested_fixes():
    spec = {"x": {0: tp_export.Dim("dx"), 1: tp_export.Dim("dy")}}
    refined = tp_export.refine_dynamic_shapes_from_suggested_fixes(
        "Suggested fixes:\n"
        "    dy = dx + 1  # dy was tied to dx\n"
        "    dx = Dim('dx', min=2)\n",
        spec,
    )
    assert refined["x"][0].min == 2
    assert isinstance(refined["x"][1], type(tp_export.Dim("d") * 2))
    static = tp_export.refine_dynamic_shapes_from_suggested_fixes(
        "Suggested fixes:\n    dx = 4\n", spec
    )
    assert static["x"][0] == 4


# ---------------------------------------------------------------------------
# state mutation recording
# ---------------------------------------------------------------------------


def test_export_records_buffer_mutation():
    class Accum(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("acc", tp.zeros(3))

        def forward(self, x):
            self.acc.add_(1.0)
            return x + self.acc

    program = tp_export.export(Accum(), tp.randn(2, 3))
    assert list(program.graph_signature.buffers_to_mutate.values()) == ["acc"]
    result = program(tp.randn(2, 3))
    assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# program API surface
# ---------------------------------------------------------------------------


def test_exported_program_state_and_parameters():
    model = MLP()
    program = tp_export.export(model, tp.randn(2, 4))
    names = {name for name, _ in program.named_parameters()}
    assert "fc1.weight" in names
    assert any("weight" in name for name in program.state_dict)
    assert isinstance(next(program.parameters()), tp.nn.Parameter)


def test_call_exported_matches_eager():
    model = MLP()
    x = tp.randn(2, 4)
    program = tp_export.export(model, x)
    runner = tp_export.ExportedProgram.call_exported(program)
    assert runner(x).tolist() == model(x).tolist()


def test_call_spec_reports_tree_contract():
    program = tp_export.export(lambda a, b=1: a + b, tp.randn(3))
    assert program.call_spec.in_spec is not None
    assert program.call_spec.out_spec is not None


def test_graph_signature_replace_all_uses_and_str():
    program = tp_export.export(MLP(), tp.randn(2, 4))
    signature = program.graph_signature
    text = str(signature)
    assert "# inputs" in text and "# outputs" in text
    signature.replace_all_uses("p_fc1_weight", "renamed_weight")
    assert any(
        spec.arg.name == "renamed_weight" for spec in signature.input_specs
    )


def test_run_decompositions_rewrites_and_preserves_numerics():
    def fn(t):
        return tp.nn.functional.silu(t) * 2

    program = tp_export.export(fn, tp.randn(3))
    decomposed = program.run_decompositions()
    body = [
        getattr(node.target, "__name__", node.target)
        for node in decomposed.graph.nodes
        if node.op in ("call_function", "call_method")
    ]
    assert "silu" not in body
    x = tp.randn(5)
    expected = fn(x).tolist()
    actual = decomposed.module()(x).tolist()
    assert actual == pytest.approx(expected, rel=1e-5)

    untouched = program.run_decompositions(tp_export.CustomDecompTable(defaults=False))
    assert untouched(x).tolist() == expected


def test_draft_export_reports_success_and_failure():
    ok = tp_export.draft_export(lambda t: t + 1, tp.randn(3))
    assert ok.success and ok.exported_program is not None

    def boom(t):
        raise ValueError("kaboom")

    bad = tp_export.draft_export(boom, tp.randn(3))
    assert not bad.success and bad.exported_program is None
    assert "kaboom" in str(bad)
    with pytest.raises(RuntimeError):
        bad.raise_on_failure()


def test_register_dataclass_flattens_through_pytree():
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class Point:
        x: object
        y: object

    tp_export.register_dataclass(Point)
    from tensorplay.graph._pytree import tree_flatten, tree_unflatten

    p = Point(tp.randn(3), tp.randn(2))
    leaves, spec = tree_flatten(p)
    assert len(leaves) == 2
    rebuilt = tree_unflatten(leaves, spec)
    assert isinstance(rebuilt, Point)


def test_unflatten_and_interpreter_module():
    from tensorplay.export.unflatten import InterpreterModule, unflatten

    model = MLP()
    x = tp.randn(2, 4)
    program = tp_export.export(model, x)
    view = unflatten(program)
    assert view.forward(x).tolist() == model(x).tolist()
    module = InterpreterModule(view.graph_module, ty="MLP")
    assert module(x).tolist() == model(x).tolist()
    assert "MLP" in module.print_readable()


def test_flat_graph_lifts_state_and_constants():
    class Block(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = tp.nn.Linear(3, 2)
            self.register_buffer("stats", tp.zeros(2), persistent=False)
            self.scale_const = tp.randn(2)

        def forward(self, x):
            return self.linear(x) * self.scale_const + self.stats

    model = Block()
    x = tp.randn(4, 3)
    program = tp_export.export(model, x)

    assert not [n for n in program.graph.nodes if n.op == "get_attr"]
    placeholder_names = [n.name for n in program.graph.placeholders]
    # lifted inputs follow module traversal order: root attrs first, then
    # children; parameters, buffers, constants within each module
    assert placeholder_names == [
        "b_stats",
        "c_scale_const",
        "p_linear_weight",
        "p_linear_bias",
        "x",
    ]
    assert "scale_const" in program.graph_signature.lifted_tensor_constants
    assert "scale_const" in program.graph_module.meta["constants"]
    assert program(x).tolist() == model(x).tolist()
    assert program.module()(x).tolist() == model(x).tolist()
    assert program.call_exported(program)(x).tolist() == model(x).tolist()


def test_lifted_mutation_updates_program_state():
    class Accum(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("acc", tp.zeros(2))

        def forward(self, x):
            self.acc.add_(1.0)
            return x + self.acc

    program = tp_export.export(Accum(), tp.randn(2, 2))
    before = program.state_dict["acc"].tolist()
    program(tp.randn(2, 2))
    after = program.state_dict["acc"].tolist()
    assert after == [item + 1.0 for item in before]
    fresh = program.module()
    out = fresh(tp.randn(2, 2))
    assert out.shape == (2, 2)


def test_save_load_lifted_state_roundtrip(tmp_path):
    model = MLP()
    x = tp.randn(3, 4)
    program = tp_export.export(model, x)
    path = tmp_path / "lifted.pt2"
    tp_export.save(program, str(path))
    loaded = tp_export.load(str(path))
    assert not [n for n in loaded.graph.nodes if n.op == "get_attr"]
    assert loaded(x).tolist() == model(x).tolist()
    assert loaded.module()(x).tolist() == model(x).tolist()


def test_unflatten_rebuilds_module_hierarchy():
    from tensorplay.export.unflatten import unflatten as _unflatten

    class Inner(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = tp.nn.Linear(4, 4)

        def forward(self, x):
            return tp.nn.functional.relu(self.proj(x))

    class Outer(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            self.head = tp.nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.inner(x) + x)

    model = Outer()
    x = tp.randn(3, 4)
    program = tp_export.export(model, x)
    view = _unflatten(program)
    assert view._hierarchical
    assert view(x).tolist() == model(x).tolist()
    # attribute access reaches the reconstructed submodules with their state
    assert view.inner.proj.weight.shape == model.inner.proj.weight.shape
    assert view.head.weight.shape == model.head.weight.shape


def test_unflatten_supports_repeated_same_args_calls():
    from tensorplay.export.unflatten import unflatten as _unflatten

    class Twice(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = tp.nn.Linear(3, 3)

        def forward(self, x):
            return self.cell(x) + self.cell(x)

    model = Twice()
    x = tp.randn(2, 3)
    view = _unflatten(tp_export.export(model, x))
    assert view(x).tolist() == model(x).tolist()


def test_unflatten_function_capture_falls_back_to_flat_view():
    from tensorplay.export.unflatten import unflatten as _unflatten

    program = tp_export.export(lambda t: t * 2, tp.randn(3))
    view = _unflatten(program)
    assert not view._hierarchical
    x = tp.randn(4)
    assert view(x).tolist() == (x * 2).tolist()


def test_load_preserves_default_argument_bindings(tmp_path):
    class Net(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = tp.nn.Linear(4, 2)

        def forward(self, x, scale=1.0):
            return self.head(x) * scale

    program = tp_export.export(Net(), tp.randn(4, 4))
    path = tmp_path / "defaults.pt2"
    tp_export.save(program, str(path))
    loaded = tp_export.load(str(path))
    assert loaded.example_inputs.get("scale") == 1.0
    assert loaded(tp.randn(2, 4)).shape == (2, 2)


def test_unflatten_hierarchical_mutation_and_multi_output():
    from tensorplay.export.unflatten import unflatten as _unflatten

    class Block(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = tp.nn.BatchNorm1d(4)

        def forward(self, x):
            return self.bn(x)

    class Net(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()
            self.head = tp.nn.Linear(4, 3)

        def forward(self, x):
            h = self.block(x)
            return self.head(h), h.sum()

    model = Net()
    x = tp.randn(5, 4)
    view = _unflatten(tp_export.export(model, x))
    out1, out2 = view(x)
    ref1, ref2 = model(x)
    assert out1.tolist() == ref1.tolist()
    assert out2.tolist() == ref2.tolist()
