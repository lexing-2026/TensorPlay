"""P1: static control-flow specialization via concrete metadata (L1-D1)."""

import pytest

import tensorplay as tp


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


def test_data_dependent_control_flow_specializes_with_guards():
    """D1: execute-mode capture bakes the traced branch and guards its data.

    The symbolic tracer still rejects (see
    ``test_compile.py::test_symbolic_tracer_still_rejects_data_dependent_control_flow``);
    ``compile()`` captures with sample execution and promotes the feeding
    placeholders into byte-exact data guards.
    """

    def fn(x):
        if bool((x > 0).all()):
            return x * 2
        return x

    compiled = tp.compile(fn, fullgraph=True)
    mixed = tp.tensor([1.0, -1.0])
    assert compiled(mixed).tolist() == [1.0, -1.0]
    # Branch flip recompiles instead of reusing the wrong side.
    assert compiled(tp.tensor([2.0, 3.0])).tolist() == [4.0, 6.0]


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


def test_compiler_gate_keeps_scalar_symbolic():
    """UPV-native path: gate() values flow as tensor proxies; one spec."""

    def fn(x):
        n = tp.compiler.gate(x.sum())
        return x * n

    compiled = tp.compile(fn, fullgraph=True)
    results = [compiled(tp.tensor([float(i)])).tolist() for i in range(6)]
    assert results == [[float(i * i)] for i in range(6)]
    # Value recomputes inside the artifact; cache never fragments on it.
    assert len(compiled._tensorplay_cache) == 1


def test_plain_int_consumption_stays_baked_and_keyed():
    """Without gate(), numeric consumption specializes into the key."""

    def fn(x):
        return x + int(x.sum().item())

    compiled = tp.compile(fn, fullgraph=True)
    assert compiled(tp.tensor([1.0])).tolist() == [2.0]
    assert compiled(tp.tensor([5.0])).tolist() == [10.0]
    assert compiled(tp.tensor([1.0])).tolist() == [2.0]
    # Two distinct baked constants -> two entries; reuse only on same value.
    assert len(compiled._tensorplay_cache) == 2


def test_gate_outside_capture_raises():
    t = tp.tensor([1.0])
    with pytest.raises(tp.compiler.GraphCaptureError):
        tp.compiler.gate(t.sum())
