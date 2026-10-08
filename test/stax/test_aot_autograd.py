import copy
import gc
import weakref

import pytest

import tensorplay as tp
import tensorplay.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(6, 5)
        self.bn = nn.BatchNorm1d(5)

    def forward(self, x):
        return self.bn(self.lin(x)).relu().sum(dim=1)


def _max_diff(a, b):
    return (a - b).abs().max().item()


@pytest.mark.parametrize("backend", ["aot_eager", "aot_eager_default_partitioner"])
def test_training_matches_eager(backend):
    tp.manual_seed(0)
    eager = Net().train()
    compiled_model = copy.deepcopy(eager)
    compiled = tp.compile(compiled_model, backend=backend)

    for _ in range(2):
        x = tp.randn(8, 6)
        expected = eager(x)
        expected.sum().backward()
        got = compiled(x)
        got.sum().backward()
        assert _max_diff(expected, got) < 1e-5

    for (name, p_eager), (_, p_compiled) in zip(
        eager.named_parameters(), compiled_model.named_parameters()
    ):
        assert p_compiled.grad is not None, name
        assert _max_diff(p_eager.grad, p_compiled.grad) < 1e-4, name
    # Running statistics advance once per call, never at compile time.
    assert _max_diff(eager.bn.running_mean, compiled_model.bn.running_mean) < 1e-5
    assert _max_diff(eager.bn.running_var, compiled_model.bn.running_var) < 1e-5


def test_inference_matches_eager():
    tp.manual_seed(0)
    model = Net().eval()
    compiled = tp.compile(model, backend="aot_eager")
    x = tp.randn(4, 6)
    with tp.no_grad():
        assert _max_diff(model(x), compiled(x)) < 1e-5


def test_input_gradients():
    def fn(a, b):
        return (a * b).sin().sum()

    compiled = tp.compile(fn, backend="aot_eager")
    a = tp.randn(3, requires_grad=True)
    b = tp.randn(3, requires_grad=True)
    compiled(a, b).backward()
    ga, gb = a.grad.clone(), b.grad.clone()
    a.grad = None
    b.grad = None
    fn(a, b).backward()
    assert _max_diff(ga, a.grad) < 1e-6
    assert _max_diff(gb, b.grad) < 1e-6


def test_aot_graphs_are_operator_level():
    def fn(a):
        return (a * 2).exp()

    compiled = tp.compile(fn, backend="aot_eager")
    a = tp.randn(3, requires_grad=True)
    compiled(a).sum().backward()
    assert a.grad is not None


def test_compiled_graphs_release_recorded_tensors():
    def fn(value):
        return (value.sin().cos().exp()).sum()

    compiled = tp.compile(fn, backend="aot_eager")
    value = tp.randn(8, 16, requires_grad=True)
    result = compiled(value)

    lowering = next(iter(compiled._tensorplay_cache.values()))
    for graph in lowering._tensorplay_aot_graphs:
        assert all(
            not isinstance(node.meta.get("val"), tp.Tensor)
            for node in graph.graph.nodes
        )
    result.backward()
    for graph in lowering._tensorplay_aot_graphs:
        assert all(
            not isinstance(node.meta.get("val"), tp.Tensor)
            for node in graph.graph.nodes
        )


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_compiled_calls_do_not_keep_inputs(cache_enabled, monkeypatch):
    import tensorplay.compiler.config as config

    monkeypatch.setattr(config, "force_disable_caches", not cache_enabled)
    compiled = tp.compile(lambda value: value.sin().sum(), backend="stax")
    refs = []
    for _ in range(3):
        value = tp.randn(8, requires_grad=True)
        refs.append(weakref.ref(value))
        compiled(value).backward()
        del value
        gc.collect()
        assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("backend", ["aot_eager", "stax"])
def test_backward_retained_graph_can_be_reused(backend):
    value = tp.randn(16, requires_grad=True)
    expected = value.cos()
    compiled = tp.compile(lambda x: x.sin().sum(), backend=backend)
    result = compiled(value)
    result.backward(retain_graph=True)
    assert _max_diff(value.grad, expected) < 1e-5
    value.grad = None
    result.backward()
    assert _max_diff(value.grad, expected) < 1e-5


def test_boxed_call_consumes_the_original_input_list():
    from tensorplay.compiler._core.aot_autograd import _call

    def fn(*args):
        raise AssertionError("expected boxed entry")

    def boxed(args):
        result = args[0] + 1
        args.clear()
        return result

    fn._tensorplay_boxed_call = boxed
    inputs = [tp.ones(2)]
    result, = _call(fn, inputs)
    assert inputs == []
    assert result.tolist() == [2., 2.]
