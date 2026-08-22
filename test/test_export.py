import pytest

import tensorplay as tp
import tensorplay.export as tp_export
from tensorplay.compiler.graph import GraphCaptureError


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
    # every get_attr target is accounted for exactly once in the signature
    get_attrs = {n.target for n in program.graph.nodes if n.op == "get_attr"}
    signed = set(program.graph_signature.parameters) | set(
        program.graph_signature.buffers
    )
    assert get_attrs == signed


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
