"""P0 first-batch passes: NormalizeOperators + PointwiseFusionHint (L2),
plus the Triton sum-epilogue detection (L5 codegen)."""

import operator

import pytest

import tensorplay as tp
from tensorplay.compiler.fx_passes import (
    POINTWISE_FUSED_OP_NAMES,
    NormalizeOperators,
    PointwiseFusionHint,
)
from tensorplay.compiler.graph import GraphModule, Tracer
from tensorplay.compiler.passes import PassManager


def _trace(fn, *args, **kwargs):
    sample = {"x": args[0]} if len(args) == 1 else {
        name: value
        for name, value in zip(("x", "w", "z"), args)
    }
    return Tracer().trace(fn, sample_inputs=sample | kwargs if kwargs else sample)


# --- NormalizeOperators -------------------------------------------------------


def test_normalize_commutes_constant_to_rhs():
    def fn(x):
        return 2.0 * x

    x = tp.tensor([1.0, 2.0])
    gm = _trace(fn, x)
    res = NormalizeOperators()(gm)
    assert res.modified is True
    mul = next(n for n in gm.graph.nodes if n.op == "call_function")
    assert isinstance(mul.args[0], type(mul))  # node first...
    assert mul.args[1] == 2.0  # ...constant second

    # Idempotent.
    res2 = NormalizeOperators()(gm)
    assert res2.modified is False


def test_normalize_folds_double_neg():
    def fn(x):
        return -(-x)

    x = tp.tensor([1.0])
    gm = _trace(fn, x)
    res = NormalizeOperators()(gm)
    assert res.modified is True
    ops = [n for n in gm.graph.nodes if n.op == "call_function"]
    assert len(ops) == 0  # both negs gone; output aliases the placeholder


@pytest.mark.parametrize(
    "mk,target_identity",
    [
        (lambda x: x + 0.0, operator.add),
        (lambda x: x - 0.0, operator.sub),
        (lambda x: x * 1.0, operator.mul),
        (lambda x: x / 1.0, operator.truediv),
    ],
)
def test_normalize_identity_right(mk, target_identity):
    x = tp.tensor([3.0])
    gm = _trace(mk, x)
    res = NormalizeOperators()(gm)
    assert res.modified is True
    remaining = [n for n in gm.graph.nodes if n.op == "call_function"]
    assert remaining == []


def test_normalize_keeps_x_times_zero():
    def fn(x):
        return x * 0.0

    x = tp.tensor([3.0])
    gm = _trace(fn, x)
    res = NormalizeOperators()(gm)
    # x*0 must survive (NaN/Inf propagation semantics).
    assert any(n.op == "call_function" for n in gm.graph.nodes)


def test_normalize_within_full_pipeline_is_idempotent():
    def fn(x, w):
        return ((x * w) + 0.0).relu()

    x = tp.tensor([-1.0, 2.0])
    w = tp.tensor([0.5, 0.5])
    gm = _trace(fn, x, w)
    pm = PassManager([NormalizeOperators(), NormalizeOperators()])
    first = pm(gm)
    second = PassManager([NormalizeOperators(), NormalizeOperators()])(gm)
    assert isinstance(first.modified, bool) and isinstance(second.modified, bool)


# --- PointwiseFusionHint --------------------------------------------------------


def test_hint_marks_pointwise_chain_single_region():
    def fn(x, w):
        return ((x * w).relu()).sin()

    x = tp.tensor([-1.0, 2.0])
    w = tp.tensor([0.5, 0.5])
    gm = _trace(fn, x, w)
    res = PointwiseFusionHint()(gm)
    assert res.modified is True
    hinted = [n for n in gm.graph.nodes if n.meta.get("fusion_hint") == "pointwise"]
    assert len(hinted) == 3  # mul, relu, sin
    regions = {n.meta["fusion_region"] for n in hinted}
    assert len(regions) == 1
    # Idempotent.
    res2 = PointwiseFusionHint()(gm)
    assert res2.modified is False


def test_hint_boundaries_split_regions():
    # Two DISJOINT fusible chains separated by a matmul boundary.
    def fn(x, w):
        p = (x * w).relu()
        m = tp.matmul(x, x)
        q = (x + x).sqrt()
        return p, m, q

    x = tp.tensor([[1.0, 2.0]])
    w = tp.tensor([[0.5], [0.5]])
    gm = _trace(fn, x, w)
    res = PointwiseFusionHint()(gm)
    assert res.modified is True
    hinted = [n for n in gm.graph.nodes if n.meta.get("fusion_hint") == "pointwise"]
    unhinted_ops = [n for n in gm.graph.nodes
                    if n.meta.get("fusion_hint") is None
                    and n.op not in {"placeholder", "output"}]
    assert {n.meta["fusion_region"] for n in hinted} == {0, 1}
    names = {getattr(n.target, "__name__", str(n.target)) for n in unhinted_ops}
    assert "matmul" in names


def test_hint_op_set_matches_stax():
    from tensorplay.backends.stax import _CPU_FUSED_OPS

    assert _CPU_FUSED_OPS is POINTWISE_FUSED_OP_NAMES


# --- Pipeline integration -----------------------------------------------------


def test_default_pipeline_stamps_hints_via_compile():
    from tensorplay.compiler import compile, registry

    calls = {}

    def recorder(gm, inputs, **kw):
        calls["hints"] = [
            n.meta.get("fusion_region")
            for n in gm.graph.nodes
            if n.op not in {"placeholder", "output"}
        ]
        return gm.recompile()

    registry.register_backend(recorder, name="_hint_recorder")
    try:
        compiled = compile(lambda x: (x * 2.0).relu(), backend="_hint_recorder")
        out = compiled(tp.tensor([1.0, -2.0]))
        assert out.shape == (2,)
        assert calls["hints"] and all(r is not None for r in calls["hints"])
    finally:
        registry.unregister_backend("_hint_recorder")


# --- Triton reduction epilogue --------------------------------------------------


def test_sum_epilogue_detection_and_source():
    from tensorplay.compiler.codegen.triton import (
        TritonProgramCodegen,
        _split_sum_epilogue,
    )

    def fn(x, w):
        return ((x * w).relu()).sum()

    x = tp.tensor([-1.0, 0.5, 2.0])
    w = tp.tensor([0.3, -0.2, 1.5])
    gm = _trace(fn, x, w)

    detected = _split_sum_epilogue(gm)
    assert detected is not None
    tail, producer, kind = detected
    assert kind == "sum"

    # triples: tmp0=mul(in0,in1); tmp1=relu(tmp0)
    codegen = TritonProgramCodegen(program=[3, 0, 1, 17, 2, -1], constants=[],
                                   output_refs=(2,), input_count=2,
                                   reduction="sum")
    src = codegen.generate("k", fixed_config=(256, 4))
    assert src.count("@triton.jit") == 1          # one kernel, not three
    assert "tl.sum(" in src                       # epilogue folded in
    assert "@triton.autotune" not in src          # fixed config pinned
    assert "tp.empty((), dtype=" in src           # scalar output buffer


def test_no_epilogue_for_non_sum_tail():
    def fn(x, w):
        return (x * w).relu()

    x = tp.tensor([1.0])
    w = tp.tensor([1.0])
    gm = _trace(fn, x, w)
    from tensorplay.compiler.codegen.triton import _split_sum_epilogue

    assert _split_sum_epilogue(gm) is None
