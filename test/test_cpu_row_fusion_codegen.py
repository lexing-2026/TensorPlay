"""Row-staged CPU fusion: planning, generated code, and numerics."""

import math

import numpy as np
import pytest

import tensorplay
from tensorplay._stax import stax as stax_mod
from tensorplay._stax.codegen import cpp as cpp_mod
from tensorplay._stax.codegen import cpp_rowfusion as row


# ---------------------------------------------------------------------------
# region shapes used throughout


def softmax(x):
    m = x.amax(dim=[1], keepdim=True)
    e = (x - m).exp()
    return e / e.sum(dim=1, keepdim=True)


def rmsnorm(x, w):
    v = (x * x).mean(dim=-1, keepdim=True)
    return (x * (v + 1e-5).rsqrt()) * w


def layernorm(x, w, b):
    mu = x.mean(dim=-1, keepdim=True)
    d = x - mu
    var = (d * d).mean(dim=-1, keepdim=True)
    return (d * (var + 1e-5).rsqrt()) * w + b


def rownorm(x):
    return (x * x).sum(dim=1).sqrt()


def scaled_rowsum(x):
    return x.sum(dim=1, keepdim=True) / 8.0


def _plan(fn, in_shape, input_shapes=None):
    from tensorplay.graph import symbolic_trace

    module = symbolic_trace(fn)
    if input_shapes is None:
        input_shapes = (tuple(in_shape),)
    return stax_mod._plan_row_fusion(module, tuple(in_shape), input_shapes)


# ---------------------------------------------------------------------------
# planning


def test_plan_stages_softmax():
    fusion = _plan(softmax, (4, 16))
    assert fusion is not None
    assert [step.op for step in fusion.steps] == ["amax", "sum"]
    assert all(step.kind == "reduce" for step in fusion.steps)
    # The maximum reduces the raw input, so its stage carries no expression.
    assert fusion.steps[0].instructions == ()
    # The sum reduces ``exp(x - m)``, which reads the first stage's row value.
    assert fusion.steps[1].instructions == (("sub", 0, 1, 3), ("exp", 3, -1, 4))
    assert fusion.output_kind == "elem"
    assert fusion.out_shape == (4, 16)
    assert fusion.reduce_extent == 16 and fusion.rows == 4


def test_plan_stages_rmsnorm_with_row_operations():
    fusion = _plan(rmsnorm, (6, 40), ((6, 40), (40,)))
    assert fusion is not None
    kinds = [(step.kind, step.op) for step in fusion.steps]
    assert kinds == [("reduce", "mean"), ("rowop", "add"), ("rowop", "rsqrt")]
    # The epsilon rides in the shared constant pool, referenced negatively.
    assert fusion.constants == (1e-05,)
    assert fusion.steps[1].rhs == -1
    assert fusion.row_slots == 3
    assert fusion.output_kind == "elem"


def test_plan_row_valued_output_keeps_the_row_shape():
    trailing = _plan(rownorm, (5, 12))
    assert trailing is not None
    assert trailing.output_kind == "row"
    assert trailing.out_shape == (5,)
    assert trailing.out_instructions == ()
    keepdim = _plan(scaled_rowsum, (5, 12))
    assert keepdim is not None and keepdim.out_shape == (5, 1)


def test_plan_rejects_regions_without_a_trailing_reduction():
    # No reduction at all, a reduction over another axis, and a full
    # reduction all belong to the pointwise or tail-reduction paths.
    assert _plan(lambda v: (v * 2).tanh(), (4, 8)) is None
    assert _plan(lambda v: v / v.amax(dim=[0], keepdim=True), (4, 8)) is None
    assert _plan(lambda v: v - v.sum(), (4, 8)) is None


def test_plan_rejects_a_reduction_broadcast_without_keepdim():
    # Without the kept axis the row value does not line up with the row it
    # would be broadcast against.
    assert _plan(lambda v: v - v.amax(dim=[1]), (4, 8)) is None


def test_plan_rejects_inputs_that_do_not_span_the_reduced_axis():
    # A per-row or scalar input is not elementwise over the row, so the
    # classification the planner relies on would not hold.
    assert _plan(softmax, (4, 8), ((4, 8),) ) is not None
    assert (
        _plan(
            lambda v, s: softmax(v * s),
            (4, 8),
            ((4, 8), (1, 1)),
        )
        is None
    )
    # Rank one has no row axis to stage over.
    assert _plan(softmax, (8,)) is None


def test_elem_dependencies_orders_producers_first_and_stops_at_row_values():
    from tensorplay.graph import symbolic_trace

    module = symbolic_trace(softmax)
    nodes = {node.name: node for node in module.graph.nodes}
    kinds = {}
    for node in module.graph.nodes:
        if node.op == "placeholder":
            kinds[node] = "elem"
        elif node.op != "output":
            kinds[node] = "row" if node.name in {"amax", "sum"} else "elem"
    order = stax_mod._elem_dependencies(nodes["truediv"], kinds)
    assert [node.name for node in order] == ["sub", "exp", "truediv"]
    # The reduction feeding the division ends the walk: it is read, not redone.
    assert nodes["sum"] not in order


# ---------------------------------------------------------------------------
# generated code


def _render(fusion, entry, shapes, strides):
    return row.render_row_fusion_source(
        fusion,
        entry,
        out_device=(0, -1),
        input_shapes=shapes,
        input_strides=strides,
    )


def test_generated_source_reads_row_values_as_broadcasts():
    fusion = _plan(softmax, (256, 1024))
    source = _render(fusion, "tp_probe_rows", ((256, 1024),), ((1024, 1),))
    # The first stage's fold lands in a broadcast the later stages read.
    assert "V s0 = V(0.0f);" in source
    assert "s0 = V(fold_);" in source
    assert "(x0_r1_0 - s0)" in source
    # Rows are independent, so the row loop itself is what gets split.
    assert "tp_parallel_for_c(0, 256L," in source
    assert "tp_make_runner" in source and "tp_direct" in source


def test_generated_source_keeps_small_regions_serial():
    fusion = _plan(softmax, (2, 8))
    source = _render(fusion, "tp_probe_small", ((2, 8),), ((8, 1),))
    assert "tp_parallel_for_c(0," not in source
    assert "tp_body(&ctx, 0, 2L);" in source


def test_generated_source_broadcasts_a_row_wide_input_within_the_row():
    fusion = _plan(rmsnorm, (8, 40), ((8, 40), (40,)))
    source = _render(fusion, "tp_probe_rowwide", ((8, 40), (40,)), ((40, 1), (1,)))
    # The row-wide weight is addressed by its position inside the row, which
    # the row loop already bounds; no lane can straddle the boundary.
    assert "in1 + ((base + q) % 40)" in source


def test_row_expression_rejects_a_shaped_tensor_operand():
    fusion = row.RowFusion(
        input_count=1,
        row_slots=2,
        constants=(),
        steps=(
            row.RowStep(kind="reduce", slot=0, op="sum", instructions=(), output_ref=0),
            row.RowStep(kind="rowop", slot=1, op="add", lhs=0, rhs=1),
        ),
        output_kind="row",
        out_instructions=(),
        out_ref=2,
        reduce_extent=8,
        rows=4,
        in_shape=(4, 8),
        out_shape=(4,),
    )
    with pytest.raises(cpp_mod._ProgramError):
        _render(fusion, "tp_probe_bad", ((4, 8),), ((8, 1),))


# ---------------------------------------------------------------------------
# end-to-end numerics


def _close(got, ref, rel=2e-5):
    got = np.asarray(got.tolist(), dtype=np.float64)
    ref = np.asarray(ref.tolist(), dtype=np.float64)
    assert got.shape == ref.shape
    scale = max(1e-6, float(np.max(np.abs(ref))) if ref.size else 1.0)
    assert float(np.max(np.abs(got - ref))) <= rel * scale


ROW_CASES = [
    ("softmax", softmax, 1),
    ("rmsnorm", rmsnorm, 2),
    ("layernorm", layernorm, 3),
    ("rownorm", rownorm, 1),
    ("scaled-rowsum", scaled_rowsum, 1),
]


@pytest.mark.parametrize(
    "name,fn,arity", ROW_CASES, ids=[case[0] for case in ROW_CASES]
)
@pytest.mark.parametrize("shape", [(6, 40), (17, 64), (3, 129)])
def test_row_fusion_matches_reference(name, fn, arity, shape):
    tensorplay.manual_seed(0)
    value = tensorplay.randn(*shape)
    args = [value] + [tensorplay.randn(shape[-1]) for _ in range(arity - 1)]
    _close(tensorplay.compile(fn)(*args), fn(*args))


def test_row_fusion_shapes_around_the_vector_tail():
    for width in (1, 3, 7, 15, 16, 17, 31, 33, 64, 65, 127):
        tensorplay.manual_seed(width)
        value = tensorplay.randn(3, width)
        _close(tensorplay.compile(softmax)(value), softmax(value), rel=5e-5)


def test_row_fusion_handles_higher_rank_rows():
    tensorplay.manual_seed(4)
    value = tensorplay.randn(2, 5, 48)
    fn = lambda v: v / v.sum(dim=-1, keepdim=True).abs()  # noqa: E731
    _close(tensorplay.compile(fn)(value), fn(value))


def test_row_fusion_propagates_nan_through_the_order_stage():
    value = tensorplay.tensor(
        [[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]], dtype=tensorplay.float32
    )
    fn = lambda v: v - v.amax(dim=[1], keepdim=True)  # noqa: E731
    got = tensorplay.compile(fn)(value).tolist()
    ref = fn(value).tolist()
    for got_row, ref_row in zip(got, ref):
        assert [math.isnan(item) for item in got_row] == [
            math.isnan(item) for item in ref_row
        ]
        for a, b in zip(got_row, ref_row):
            assert math.isnan(a) or a == b


def test_row_fusion_result_is_reusable_across_calls():
    tensorplay.manual_seed(5)
    compiled = tensorplay.compile(softmax)
    first = tensorplay.randn(6, 48)
    second = tensorplay.randn(6, 48)
    _close(compiled(first), softmax(first))
    _close(compiled(second), softmax(second))
    _close(compiled(first), softmax(first))


def test_grad_inputs_keep_the_uncompiled_route():
    value = tensorplay.randn(4, 8, requires_grad=True)
    result = tensorplay.compile(softmax)(value)
    result.sum().backward()
    assert value.grad is not None


# ---------------------------------------------------------------------------
# routing


@pytest.mark.parametrize(
    "name,fn,arity", ROW_CASES, ids=[case[0] for case in ROW_CASES]
)
def test_compile_routes_the_region_to_the_row_staged_kernel(name, fn, arity):
    tensorplay.manual_seed(6)
    value = tensorplay.randn(9, 64)
    args = [value] + [tensorplay.randn(64) for _ in range(arity - 1)]
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(*args), fn(*args))
    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen == "stax-fused-cpu-rowfuse"
    assert lowering._native_runner is not None


def test_kill_switch_disables_the_generated_row_kernel(monkeypatch):
    fusion = _plan(softmax, (4, 16))
    monkeypatch.setenv("TP_STAX_CPU_NATIVE", "0")
    assert (
        row.build_cpu_row_fusion_kernel(
            fusion,
            device=tensorplay.device("cpu"),
            input_shapes=((4, 16),),
            input_strides=((16, 1),),
        )
        is None
    )
