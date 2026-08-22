"""P1: static control-flow specialization via concrete metadata (L1-D1)."""

import pytest

import tensorplay as tp
from tensorplay.compiler.graph import GraphCaptureError


@pytest.fixture(autouse=True)
def _clean_probe_backends():
    yield
    for name in ("p1_probe", "p1_meta_probe"):
        tp.compiler.unregister_backend(name)


class ShapeBranch(tp.nn.Module):
    def forward(self, x):
        if x.shape[0] > 2:
            return x * 2
        return x + 1


def test_shape_condition_specializes_statically():
    model = ShapeBranch()
    compiled = tp.compile(model, fullgraph=True)

    big = compiled(tp.tensor([1.0, 2.0, 3.0, 4.0]))
    assert big.tolist() == [2.0, 4.0, 6.0, 8.0]

    # Different shape -> separate specialization -> other branch.
    small = compiled(tp.tensor([1.0]))
    assert small.tolist() == [2.0]


def test_static_branch_graph_contains_only_taken_path():
    calls = {}

    @tp.compiler.register_backend(name="p1_probe")
    def probe(graph_module, example_inputs, **kwargs):
        calls["ops"] = [
            n.target.__name__ if callable(n.target) else n.target
            for n in graph_module.graph.nodes
            if n.op in ("call_function", "call_method")
        ]
        return graph_module.forward

    def fn(x):
        out = x + 1
        if x.shape[0] > 1:
            out = out * 2
        else:
            out = out - 5
        return out.relu()

    compiled = tp.compile(fn, backend="p1_probe", fullgraph=True)
    compiled(tp.tensor([1.0, 2.0]))
    assert calls["ops"] == ["add", "mul", "relu"]


def test_loop_over_metadata_unrolls():
    def fn(x):
        acc = x
        for i in range(x.ndim):
            acc = acc + i
        return acc

    compiled = tp.compile(fn, fullgraph=True)
    x = tp.zeros((3, 4))
    assert compiled(x).tolist() == (x + 0 + 1).tolist()


def test_len_of_proxy_uses_sample():
    def fn(x):
        half = len(x) // 2
        return x[:half]

    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(tp.tensor([1.0, 2.0, 3.0, 4.0]))
    assert out.tolist() == [1.0, 2.0]


def test_data_dependent_control_flow_still_rejected():
    def fn(x):
        if bool((x > 0).all()):
            return x * 2
        return x

    with pytest.raises(GraphCaptureError):
        tp.compile(fn, fullgraph=True)(tp.tensor([1.0, -1.0]))


def test_sample_inputs_recorded_on_graph_module():
    captured = {}

    @tp.compiler.register_backend(name="p1_meta_probe")
    def meta_probe(graph_module, example_inputs, **kwargs):
        captured["meta"] = getattr(graph_module, "meta", {})
        return graph_module.forward

    compiled = tp.compile(lambda a, b: a + b, backend="p1_meta_probe", fullgraph=True)
    compiled(tp.ones(2), tp.ones(2))
    samples = captured["meta"].get("sample_inputs", {})
    assert set(samples) == {"a", "b"}


def test_tracer_without_samples_keeps_symbolic_shape():
    from tensorplay.compiler.graph import Tracer

    gm = Tracer().trace(lambda t: tp.zeros(t.shape))
    targets = [
        n.target
        for n in gm.graph.nodes
        if n.op == "call_function"
    ]
    # No sample: the shape read stays a symbolic graph node instead of
    # resolving to a concrete tuple in Python.
    assert any(getattr(t, "__name__", None) == "getattr" for t in targets)
    assert "sample_inputs" not in gm.meta
