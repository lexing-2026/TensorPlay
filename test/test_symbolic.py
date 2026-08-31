"""P2: metadata-touch shape guards for dynamic-mode specialization."""

import pytest

import tensorplay as tp


@pytest.fixture()
def probe_calls(monospace=None):
    calls = []

    def backend(graph_module, example_inputs, **kwargs):
        calls.append(
            [(n.op, getattr(n.target, "__name__", n.target)) for n in graph_module.graph.nodes]
        )
        return graph_module.forward

    tp._stax.register_backend(name="p2_probe")(backend)
    yield calls
    tp._stax.unregister_backend("p2_probe")


def _branch_on_shape(x):
    if x.shape[0] > 2:
        return x * 2
    return x + 1


def test_shape_branch_respecializes_per_size_under_dynamic(probe_calls):
    compiled = tp.compile(_branch_on_shape, backend="p2_probe", dynamic=True)

    assert compiled(tp.tensor([1.0, 2.0, 3.0, 4.0])).tolist() == [2.0, 4.0, 6.0, 8.0]
    assert compiled(tp.tensor([1.0])).tolist() == [2.0]
    # Same size again -> cached specialization reused.
    assert compiled(tp.tensor([5.0, 6.0, 7.0, 8.0])).tolist() == [10.0, 12.0, 14.0, 16.0]

    assert len(probe_calls) == 2


def test_untouched_shapes_stay_wildcards_under_dynamic(probe_calls):
    def fn(x):
        return x * 2 + 1

    compiled = tp.compile(fn, backend="p2_probe", dynamic=True)
    sizes = [(3,), (5,), (7,)]
    for size in sizes:
        out = compiled(tp.zeros(size))
        assert tuple(out.shape) == size

    assert len(probe_calls) == 1


def test_guards_are_parameter_scoped(probe_calls):
    def fn(x, y):
        if y.shape[0] > 2:
            return x * 2
        return x + 1

    compiled = tp.compile(fn, backend="p2_probe", dynamic=True)

    # x varies freely while the guarded placeholder keeps its shape.
    assert compiled(tp.tensor([1.0]), tp.tensor([1.0, 2.0, 3.0])).tolist() == [2.0]
    assert compiled(tp.tensor([9.0, 9.0]), tp.tensor([1.0, 2.0, 3.0])).tolist() == [18.0, 18.0]
    assert len(probe_calls) == 1

    # Guarded placeholder changes shape -> new specialization.
    assert compiled(tp.tensor([1.0]), tp.tensor([1.0])).tolist() == [2.0]
    assert len(probe_calls) == 2


def test_metadata_touches_recorded_on_graph_module():
    captured = {}

    def spy(graph_module, example_inputs, **kwargs):
        captured["touches"] = getattr(graph_module, "meta", {}).get("metadata_touches")
        return graph_module.forward

    tp._stax.register_backend(name="p2_spy")(spy)
    try:
        # compile() resolves the backend at wrapper creation time.
        compiled = tp.compile(_branch_on_shape, backend="p2_spy", dynamic=True)
        compiled(tp.tensor([1.0]))
    finally:
        tp._stax.unregister_backend("p2_spy")

    touches = {name for name, _attr in captured["touches"]}
    assert "x" in touches


def test_static_mode_keeps_exact_signatures(probe_calls):
    compiled = tp.compile(_branch_on_shape, backend="p2_probe")
    compiled(tp.tensor([1.0, 2.0]))
    compiled(tp.tensor([1.0, 2.0, 3.0]))
    compiled(tp.tensor([1.0, 2.0]))

    assert len(probe_calls) == 2
