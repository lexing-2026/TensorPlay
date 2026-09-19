import pytest

import tensorplay as tp
from tensorplay._ops import NATIVE_NAMESPACE
from tensorplay.graph.experimental._dispatch_trace import dispatch_make_graph
from tensorplay.graph.passes.functionalize import functionalize

ops = getattr(tp.ops, NATIVE_NAMESPACE)


def _mutating_targets(gm):
    found = []
    for node in gm.graph.nodes:
        schema = getattr(node.target, "_schema", None)
        if node.op == "call_function" and schema is not None and schema.is_mutable:
            found.append(node.target)
    return found


def _check(fn, *inputs, allowed_trailing_copies=0):
    reference_inputs = [x.clone() for x in inputs]
    expected = fn(*reference_inputs)
    traced = dispatch_make_graph(fn)(*[x.clone() for x in inputs])
    functional = functionalize(traced)
    mutating = _mutating_targets(functional)
    assert all(t is ops.copy_.default for t in mutating), mutating
    assert len(mutating) == allowed_trailing_copies
    run_inputs = [x.clone() for x in inputs]
    got = functional(*run_inputs)
    expected = expected if isinstance(expected, tuple) else (expected,)
    got = got if isinstance(got, tuple) else (got,)
    for e, g in zip(expected, got):
        assert (e - g).abs().max().item() < 1e-6
    # Input mutations are still visible to the caller.
    for ref, ran in zip(reference_inputs, run_inputs):
        assert (ref - ran).abs().max().item() < 1e-6
    return functional


def test_inplace_on_intermediate():
    def fn(x):
        y = x * 2
        y.add_(1)
        y.mul_(3)
        return y

    _check(fn, tp.randn(4))


def test_inplace_on_input_is_written_back():
    def fn(x):
        x.add_(1)
        return x * 2

    _check(fn, tp.randn(3), allowed_trailing_copies=1)


def test_mutation_through_a_slice_updates_the_base():
    def fn(x):
        y = x.clone()
        y[1:3].mul_(10)
        return y

    functional = _check(fn, tp.arange(5.0))
    assert any(n.target is ops.slice_scatter.default for n in functional.graph.nodes)


def test_mutation_through_select_and_transpose():
    def fn(x):
        y = x.clone()
        z = y.t()
        z.select(0, 1).fill_(7)
        return y, z

    _check(fn, tp.randn(3, 4))


def test_mutation_through_split_outputs():
    def fn(x):
        y = x.clone()
        a, b = y.split(2)
        b.zero_()
        return y

    _check(fn, tp.arange(4.0))


def test_view_read_after_base_mutation_sees_the_write():
    def fn(x):
        y = x.clone()
        v = y.view(2, 2)
        y.add_(1)
        return v * 1

    _check(fn, tp.arange(4.0))


def test_out_variant_becomes_functional():
    def fn(x):
        out = tp.empty(3)
        tp.sin(x, out=out)
        return out

    _check(fn, tp.randn(3))


@pytest.mark.parametrize("value", [0.0, 2.5])
def test_mutation_without_functional_overload_runs_on_a_copy(value):
    def fn(x):
        y = x.clone()
        y.fill_diagonal_(value)
        return y

    _check(fn, tp.randn(3, 3))
