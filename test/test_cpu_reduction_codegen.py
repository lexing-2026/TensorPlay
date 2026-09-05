"""Fused CPU reduction code generation: planning, numerics, and lowering."""

import math

import numpy as np
import pytest

import tensorplay
from tensorplay._stax import stax as stax_mod
from tensorplay._stax.codegen import cpp_reduction as red


# ---------------------------------------------------------------------------
# loop planning


def test_plan_collapses_adjacent_runs():
    plan = red.plan_reduction((4, 8, 16), red.ReduceSpec("sum", (1, 2), False))
    assert plan is not None
    # Two adjacent reduced dimensions become one run of stride 1.
    assert plan.row_dims == ((4, 128),)
    assert plan.red_dims == ((128, 1),)
    assert plan.post == 1
    assert plan.schedule == "horizontal"
    assert plan.out_shape == (4,)


def test_plan_keeps_trailing_axis_as_post():
    plan = red.plan_reduction((32, 64), red.ReduceSpec("sum", (0,), False))
    assert plan is not None
    assert plan.row_dims == ()
    assert plan.red_dims == ((32, 64),)
    assert plan.post == 64
    assert plan.schedule == "vertical"
    assert plan.rows == 1
    assert plan.out_shape == (64,)


def test_plan_interleaved_axes_keep_declaration_order():
    plan = red.plan_reduction(
        (2, 3, 4, 5), red.ReduceSpec("sum", (0, 2), False)
    )
    assert plan is not None
    assert plan.red_dims == ((2, 60), (4, 5))
    assert plan.row_dims == ((3, 20),)
    assert plan.post == 5
    assert plan.out_shape == (3, 5)


def test_plan_keepdim_and_full_reduction():
    keep = red.plan_reduction((6, 7), red.ReduceSpec("mean", (1,), True))
    assert keep is not None and keep.out_shape == (6, 1)
    full = red.plan_reduction((6, 7), red.ReduceSpec("sum", (0, 1), False))
    assert full is not None and full.out_shape == ()
    assert full.rows == 1 and full.red == 42


def test_plan_rejects_degenerate_specs():
    assert red.plan_reduction((), red.ReduceSpec("sum", (0,), False)) is None
    assert red.plan_reduction((4, 0), red.ReduceSpec("sum", (0,), False)) is None
    # duplicate, out-of-range, and unknown operations
    assert red.plan_reduction((4, 4), red.ReduceSpec("sum", (0, 0), False)) is None
    assert red.plan_reduction((4, 4), red.ReduceSpec("sum", (2,), False)) is None
    assert red.plan_reduction((4, 4), red.ReduceSpec("median", (0,), False)) is None
    # no reduced axis at all is not a reduction
    assert red.plan_reduction((4, 4), red.ReduceSpec("sum", (), False)) is None


def test_plan_resolves_negative_axes():
    plan = red.plan_reduction((3, 5), red.ReduceSpec("sum", (-1,), False))
    assert plan is not None and plan.out_shape == (3,)


# ---------------------------------------------------------------------------
# node parsing


def _spec(fn):
    from tensorplay.graph import symbolic_trace

    module = symbolic_trace(fn)
    node = [
        value
        for output in module.graph.outputs
        for value in stax_mod._nodes(output.args)
    ][0]
    return stax_mod._parse_reduction(node, 2)


def test_parse_reduction_spellings():
    assert _spec(lambda x: x.sum()) == red.ReduceSpec("sum", (0, 1), False)
    assert _spec(lambda x: x.sum(dim=1)) == red.ReduceSpec("sum", (1,), False)
    assert _spec(lambda x: x.sum(dim=(0, 1), keepdim=True)) == red.ReduceSpec(
        "sum", (0, 1), True
    )
    assert _spec(lambda x: x.mean(dim=-1)) == red.ReduceSpec("mean", (-1,), False)
    assert _spec(lambda x: x.amax(dim=[0])) == red.ReduceSpec("amax", (0,), False)


def test_parse_reduction_rejects_value_index_forms():
    # ``max``/``min`` with an axis return a value/index pair, which is not
    # this lowering's contract; without an axis they are plain reductions.
    assert _spec(lambda x: x.max()) == red.ReduceSpec("max", (0, 1), False)
    assert _spec(lambda x: x.max(dim=1)) is None
    assert _spec(lambda x: x.argmax()) is None
    assert _spec(lambda x: x.tanh()) is None


# ---------------------------------------------------------------------------
# end-to-end numerics


def _close(got, ref, rel=2e-5):
    got = np.asarray(got.tolist(), dtype=np.float64)
    ref = np.asarray(ref.tolist(), dtype=np.float64)
    assert got.shape == ref.shape
    scale = max(1e-6, float(np.max(np.abs(ref))) if ref.size else 1.0)
    assert float(np.max(np.abs(got - ref))) <= rel * scale


REDUCTION_CASES = [
    ("full-sum", (4, 129), lambda v: v.sum()),
    ("row-sum", (4, 129), lambda v: v.sum(dim=1)),
    ("col-sum", (33, 65), lambda v: v.sum(dim=0)),
    ("row-sum-keepdim", (7, 40), lambda v: v.sum(dim=1, keepdim=True)),
    ("col-sum-keepdim", (7, 40), lambda v: v.sum(dim=0, keepdim=True)),
    ("mean-full", (13, 17), lambda v: v.mean()),
    ("mean-row", (13, 17), lambda v: v.mean(dim=1)),
    ("mean-col", (13, 17), lambda v: v.mean(dim=0)),
    ("amax-row", (9, 31), lambda v: v.amax(dim=[1])),
    ("amin-col", (9, 31), lambda v: v.amin(dim=[0])),
    ("max-full", (9, 31), lambda v: v.max()),
    ("min-full", (9, 31), lambda v: v.min()),
    ("prod-row", (5, 6), lambda v: v.prod(dim=1)),
    ("fused-chain", (16, 100), lambda v: (v * 2).tanh().sum(dim=1)),
    ("fused-square", (16, 100), lambda v: (v * v).sum()),
    ("fused-exp-col", (16, 100), lambda v: v.exp().sum(dim=0)),
    ("neg-axis", (11, 23), lambda v: v.sum(dim=-1)),
]


@pytest.mark.parametrize(
    "name,shape,fn", REDUCTION_CASES, ids=[case[0] for case in REDUCTION_CASES]
)
def test_reduction_matches_reference(name, shape, fn):
    tensorplay.manual_seed(0)
    value = tensorplay.randn(*shape)
    compiled = tensorplay.compile(fn)
    _close(compiled(value), fn(value))


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize("dims", [(0,), (1,), (-1,), (0, 2), (1, 2)])
def test_reduction_over_axis_subsets(shape, dims):
    if any(dim >= len(shape) for dim in dims):
        pytest.skip("axis outside this rank")
    tensorplay.manual_seed(1)
    value = tensorplay.randn(*shape)
    fn = lambda v: v.sum(dim=dims)  # noqa: E731
    _close(tensorplay.compile(fn)(value), fn(value))


def test_reduction_shapes_around_the_vector_tail():
    # Widths that land exactly on, just below, and just above whole vectors
    # exercise the full, single-vector, and masked-tail loops.
    for width in (1, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 65):
        value = tensorplay.randn(3, width)
        fn = lambda v: v.sum(dim=1)  # noqa: E731
        _close(tensorplay.compile(fn)(value), fn(value), rel=5e-5)


def test_order_reductions_propagate_nan():
    value = tensorplay.tensor(
        [[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]], dtype=tensorplay.float32
    )
    for fn in (
        lambda v: v.amax(dim=[1]),
        lambda v: v.amin(dim=[1]),
        lambda v: v.amax(dim=[0]),
    ):
        got = tensorplay.compile(fn)(value).tolist()
        ref = fn(value).tolist()
        assert [math.isnan(item) for item in got] == [
            math.isnan(item) for item in ref
        ]
        for a, b in zip(got, ref):
            assert math.isnan(a) or a == b


def test_broadcast_scalar_input_is_hoisted():
    tensorplay.manual_seed(2)
    value = tensorplay.randn(8, 64)
    scale = tensorplay.randn(1, 1)
    fn = lambda v, s: (v * s).sum(dim=1)  # noqa: E731
    _close(tensorplay.compile(fn)(value, scale), fn(value, scale))


def test_cascade_sum_beats_sequential_accumulation():
    # A long run of same-magnitude values is where a flat float accumulator
    # loses low-order bits; the cascade has to stay close to the exact sum.
    count = 1 << 20
    data = np.full(count, 0.1, dtype=np.float32)
    value = tensorplay.tensor(data.tolist(), dtype=tensorplay.float32)
    exact = float(np.sum(data.astype(np.float64)))
    fn = lambda v: v.sum()  # noqa: E731
    compiled = float(tensorplay.compile(fn)(value).item())
    sequential = np.float32(0.0)
    for item in data[: 1 << 14]:
        sequential = np.float32(sequential + item)
    # The compiled sum keeps a relative error far below what a flat float32
    # accumulator reaches after only a sixteenth of the same input.
    flat_error = abs(float(sequential) - float(np.sum(data[: 1 << 14], dtype=np.float64)))
    flat_error /= float(np.sum(data[: 1 << 14], dtype=np.float64))
    assert abs(compiled - exact) / exact < flat_error


def test_reduction_result_is_reusable_across_calls():
    tensorplay.manual_seed(3)
    fn = lambda v: (v * 3).sum(dim=1)  # noqa: E731
    compiled = tensorplay.compile(fn)
    first = tensorplay.randn(6, 48)
    second = tensorplay.randn(6, 48)
    _close(compiled(first), fn(first))
    _close(compiled(second), fn(second))
    _close(compiled(first), fn(first))


def test_grad_inputs_keep_the_uncompiled_route():
    value = tensorplay.randn(4, 8, requires_grad=True)
    fn = lambda v: (v * 2).sum()  # noqa: E731
    result = tensorplay.compile(fn)(value)
    result.backward()
    assert value.grad is not None
    _close(value.grad, tensorplay.ones_like(value) * 2)


def test_kill_switch_disables_the_generated_reduction(monkeypatch):
    monkeypatch.setenv("TP_STAX_CPU_NATIVE", "0")
    assert (
        red.build_cpu_reduction_kernel(
            [],
            [],
            1,
            0,
            red.ReduceSpec("sum", (0,), False),
            in_shape=(16,),
            device=tensorplay.device("cpu"),
            input_shapes=((16,),),
            input_strides=((1,),),
        )
        is None
    )


def test_generated_source_selects_a_worksharing_strategy():
    spec = red.ReduceSpec("sum", (1,), False)
    plan = red.plan_reduction((1024, 1024), spec)
    source = red.render_reduction_source(
        [],
        [],
        1,
        0,
        spec,
        plan,
        "tp_probe",
        out_device=(0, -1),
        input_shapes=((1024, 1024),),
        input_strides=((1024, 1),),
        in_shape=(1024, 1024),
        lane_count=8,
    )
    assert "tp_parallel_for_c" in source
    assert "tp_direct" in source
    assert "tp_make_runner" in source
    # Many rows: the row loop is what gets split, so no slot fold is emitted.
    assert "slot_buf" not in source


def test_generated_source_splits_a_single_row_reduction():
    spec = red.ReduceSpec("sum", (0, 1), False)
    plan = red.plan_reduction((1024, 1024), spec)
    source = red.render_reduction_source(
        [],
        [],
        1,
        0,
        spec,
        plan,
        "tp_probe_split",
        out_device=(0, -1),
        input_shapes=((1024, 1024),),
        input_strides=((1024, 1),),
        in_shape=(1024, 1024),
        lane_count=8,
    )
    # One output element cannot fill the pool from the row loop, so the
    # reduction itself is split into per-worker slots.
    assert "slot_buf" in source
    assert "tp_cascade_promote" in source
