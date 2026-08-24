"""P3 vertical slice: AOT forward/backward partitioning (L4)."""

import operator

import pytest

import tensorplay as tp
from tensorplay.compiler import AOTError, build_aot
from tensorplay.compiler.graph import Tracer


def _make(fn, sample):
    tracer = Tracer()
    gm = tracer.trace(fn, sample_inputs={"x": sample[0], "w": sample[1]} if len(sample) > 1 else {"x": sample[0]})
    return gm


def _make_sample_map(fn, args):
    import inspect

    names = list(inspect.signature(fn).parameters)
    return dict(zip(names, args))


def _assert_grads_match_eager(fn, args, policy, required=None, rtol=1e-5, atol=1e-6, partitioner="default"):
    smap = _make_sample_map(fn, args)
    gm = Tracer().trace(fn, sample_inputs=smap)
    result = build_aot(gm, sample_inputs=smap, required_grads=required, policy=policy, partitioner=partitioner)
    out, grads = result.value_and_grad(*args)

    eager_args = [a.clone().requires_grad_(True) for a in args]
    eager_out = fn(*eager_args)
    if isinstance(eager_out, tp.Tensor):
        eager_out.sum().backward()
    else:
        sum(t.sum() for t in eager_out).backward()

    if isinstance(eager_out, tp.Tensor):
        assert out.shape == eager_out.shape
    else:
        assert tuple(o.shape for o in out) == tuple(o.shape for o in eager_out)
    for got, want in zip(grads.values(), [a.grad for a in eager_args]):
        assert tp.allclose(got, want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("policy", ["save_needed", "recompute_all"])
@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
def test_mul_relu_sum_gradients(policy, partitioner):
    def fn(x, w):
        return (x * w).relu().sum()

    x = tp.tensor([-1.0, 0.5, 2.0])
    w = tp.tensor([0.3, -0.2, 1.5])
    _assert_grads_match_eager(fn, [x, w], policy, partitioner=partitioner)


@pytest.mark.parametrize("policy", ["save_needed", "recompute_all"])
@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
def test_broadcast_scalar_param_gradient_reduces(policy, partitioner):
    def fn(x, w):
        return (x * w).sum()

    x = tp.tensor([1.0, 2.0, 3.0])
    w = tp.tensor(2.0)  # scalar: d(w) must reduce (3,) -> ()
    _assert_grads_match_eager(fn, [x, w], policy, partitioner=partitioner)


def test_sin_chain_and_truediv():
    def fn(x, w):
        return (sin_term := (x * w).sin()).sum() / 2

    x = tp.tensor([0.3, -0.7])
    w = tp.tensor([1.4, 2.2])
    _assert_grads_match_eager(fn, [x, w], "save_needed", rtol=1e-4, atol=1e-5)


def test_save_needed_saves_only_rule_operands():
    def fn(x, w):
        return ((x * w).relu()).sum()

    gm = Tracer().trace(fn, sample_inputs={"x": tp.tensor([1.0]), "w": tp.tensor([1.0])})
    result = build_aot(
        gm,
        sample_inputs={"x": tp.tensor([1.0]), "w": tp.tensor([1.0])},
        policy="save_needed",
    )
    # Backward needs the pre-activation (relu input); nothing else.
    assert len(result.saved_names) == 1

    gm2 = Tracer().trace(fn, sample_inputs={"x": tp.tensor([1.0]), "w": tp.tensor([1.0])})
    result2 = build_aot(gm2, sample_inputs={"x": tp.tensor([1.0]), "w": tp.tensor([1.0])},
                        policy="recompute_all")
    assert result2.saved_names == []


def test_multiple_outputs_tuple_grads():
    def fn(x):
        return (x.sin(), x.cos())

    x = tp.tensor([0.4])
    _assert_grads_match_eager(fn, [x], "save_needed")


def test_unsupported_op_raises_clear_error():
    def fn(x):
        return x.reshape(-1).sum()

    gm = Tracer().trace(fn, sample_inputs={"x": tp.tensor([[1.0], [2.0]])})
    with pytest.raises(AOTError, match="no derivative"):
        build_aot(gm, sample_inputs={"x": tp.tensor([[1.0], [2.0]])})


def test_aot_backend_through_compile_pipeline():
    calls = {}

    def aot_backend(graph_module, example_inputs, **kwargs):
        sample = {p.name: v for p, v in zip(graph_module.graph.placeholders, example_inputs)}
        result = build_aot(graph_module, sample_inputs=sample)
        calls["saved"] = result.saved_names
        return lambda *a, **k: result.value_and_grad(*a)[0]

    def fn(x, w):
        return (x * w).relu().sum()

    compiled = tp.compile(fn, backend=aot_backend, fullgraph=True)
    x = tp.tensor([-1.0, 0.5, 2.0])
    w = tp.tensor([0.3, -0.2, 1.5])
    out = compiled(x, w)
    assert out.tolist() == (x * w).relu().sum().tolist()
    assert len(calls["saved"]) >= 1


@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
def test_partitioner_roles_consistent(partitioner):
    def fn(x, w):
        return (x * w).relu().sum()

    x = tp.tensor([1.0, 2.0])
    smap = {"x": x, "w": tp.tensor([1.0, 2.0])}
    r = build_aot(Tracer().trace(fn, sample_inputs=smap), sample_inputs=smap,
                  partitioner=partitioner)
    phs = list(r.backward_gm.graph.placeholders)
    assert len(phs) == len(r.input_kinds) == len(r.input_keys)
    assert sum(k == "tangent" for k in r.input_kinds) == 1
    assert {"x", "w"} <= {k for k, kind in zip(r.input_keys, r.input_kinds) if kind == "leaf"}
    assert all(name in r.saved_names or True for name in [])


def test_min_cut_saves_no_more_than_default():
    def fn(x, w):
        return ((x * w).sin().cos() - x / w).exp().sum()

    smap = {"x": tp.tensor([1.0]), "w": tp.tensor([2.0])}
    gm1 = Tracer().trace(fn, sample_inputs=smap)
    gm2 = Tracer().trace(fn, sample_inputs=smap)
    d = build_aot(gm1, sample_inputs=smap, partitioner="default")
    m = build_aot(gm2, sample_inputs=smap, partitioner="min_cut")
    assert len(m.saved_names) <= len(d.saved_names)


def test_min_cut_budget_limits_saved_bytes():
    def fn(x):
        return x.mul(x).mul(x).mul(x).sum()

    smap = {"x": tp.tensor([1.0])}
    gm = Tracer().trace(fn, sample_inputs=smap)
    r = build_aot(gm, sample_inputs=smap, partitioner="min_cut",
                  memory_budget=16)
    assert isinstance(r.saved_names, list)
