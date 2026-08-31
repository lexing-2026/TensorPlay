"""Extended fused pointwise surface: comparisons, where, order ops, casts.

The Triton code generator shares the postfix program with the CPU fused
path; these tests exercise the program encoding and the emitted kernel
source without requiring a CUDA device.
"""

import pytest
import tensorplay as tp
from tensorplay.graph import Tracer
from tensorplay.graph.passes import (
    DecomposePass,
    DeadCodeElimination,
    NormalizeOperators,
    PassManager,
)
from tensorplay._stax.stax import (
    _CPU_FUSED_OPCODES,
    _TRITON_OPCODES,
    _build_pointwise_program,
)
from tensorplay._stax.codegen.triton import (
    TritonProgramCodegen,
    _extract_segment_view,
)


def _pipeline(fn, *inputs):
    gm = Tracer(execute=True).trace(
        fn, sample_inputs={f"arg{i}": value for i, value in enumerate(inputs)}
    )
    PassManager(
        [
            NormalizeOperators(),
            DecomposePass(),
            DeadCodeElimination(),
        ]
    )(gm)
    return gm


def _program(fn, *inputs, opcodes=None):
    gm = _pipeline(fn, *inputs)
    return gm, _build_pointwise_program(gm, opcodes=opcodes)


def _source(fn, *inputs, **gen_kwargs):
    gm, built = _program(fn, *inputs)
    assert built is not None
    external_nodes, program, constants, instructions, output_ref = built
    shape = tuple(int(dim) for dim in inputs[0].shape)
    gen = TritonProgramCodegen(
        program,
        constants,
        (output_ref,),
        len(external_nodes),
        input_shapes=tuple(shape for _ in external_nodes),
        reference_shape=shape,
        **gen_kwargs,
    )
    return gm, gen.generate("probe", fixed_config=(128, 4))


# --- program encoding --------------------------------------------------------


def test_comparison_produces_bool_then_where_selects():
    def fn(x, y):
        return tp.where(x > y, x * 2.0, tp.minimum(x, y))

    x = tp.randn(8, 8)
    gm, built = _program(fn, x, x)
    assert built is not None
    _, program, _, instructions, _ = built
    names = [op for op, *_ in instructions]
    assert names == ["gt", "mul", "minimum", "where", "where_rest"]
    # where/where_rest are adjacent and carry the same condition ref
    where_at = names.index("where")
    assert program[3 * where_at + 1] == program[3 * (where_at + 1) + 1]


def test_where_rejects_numeric_condition():
    def fn(x, y):
        return tp.where(x + y, x, y)

    x = tp.randn(4, 4)
    _, built = _program(fn, x, x)
    assert built is None


def test_casts_encode_float_dtype_ids():
    def fn(x):
        return x.half().float().double()

    x = tp.randn(4, 4)
    _, built = _program(fn, x)
    assert built is not None
    _, _, _, instructions, _ = built
    ids = [rhs for op, _lhs, rhs, _res in instructions if op == "cast"]
    assert ids == [1, 3, 4]  # float16, float32, float64


def test_cast_to_positional_and_keyword_spelling():
    def fn(x):
        return x.to(tp.float16) + x.to(dtype=tp.float64)

    x = tp.randn(4, 4)
    _, built = _program(fn, x)
    assert built is not None
    _, _, _, instructions, _ = built
    ids = sorted(rhs for op, _l, rhs, _r in instructions if op == "cast")
    assert ids == [1, 4]


def test_non_float_casts_stay_uncompiled():
    def fn(x):
        return x.to(tp.bool)

    x = tp.randn(4, 4)
    _, built = _program(fn, x)
    assert built is None


def test_bool_output_rejected():
    def fn(x, y):
        return x > y

    x = tp.randn(4, 4)
    _, built = _program(fn, x, x)
    assert built is None


def test_bool_feeds_arithmetic_through_normalized_floats():
    def fn(x, y):
        return (x > y) * x + (x <= y)

    x = tp.randn(4, 4)
    _, built = _program(fn, x, x)
    assert built is not None
    _, _, _, instructions, _ = built
    assert [op for op, *_ in instructions] == ["gt", "mul", "le", "add"]


def test_cpu_table_rejects_triton_only_opcodes():
    def fn(x, y):
        return tp.where(x > y, x, y)

    x = tp.randn(4, 4)
    _, built = _program(fn, x, x, opcodes=_CPU_FUSED_OPCODES)
    assert built is None
    _, built = _program(fn, x, x, opcodes=_TRITON_OPCODES)
    assert built is not None


def test_decomposed_select_family_lands_in_one_program():
    def fn(x):
        return (
            tp.nn.functional.leaky_relu(x, 0.1)
            + tp.nn.functional.relu6(x)
            + tp.nn.functional.hardshrink(x, 0.2)
        )

    x = tp.randn(4, 4)
    _, built = _program(fn, x)
    assert built is not None
    _, _, _, instructions, _ = built
    assert {op for op, *_ in instructions} <= {
        name for name in _TRITON_OPCODES
    }
    assert "gt" in {op for op, *_ in instructions}


# --- emitted kernel source ---------------------------------------------------


def test_source_comparison_and_where():
    def fn(x, y):
        return tp.where(x > y, x * 2.0, tp.minimum(x, y))

    x = tp.randn(8, 8)
    _, src = _source(fn, x, x)
    assert "(in0 > in1).to(tl.float32)" in src
    # the ``where`` instruction consumes a temp slot silently; the select
    # lands on the following temp
    assert "tmp4 = tl.where((tmp0) != 0.0, tmp1, tmp2)" in src
    assert "tl.minimum(in0, in1)" in src


def test_source_order_ops_and_transcendentals():
    def fn(x, y):
        return tp.maximum(x, y) + x.clamp_min(-1.0) + tp.rsqrt(x.abs()) + tp.erf(x)

    x = tp.randn(8, 8)
    _, src = _source(fn, x, x)
    assert "tl.maximum(in0, in1)" in src
    assert "tl.maximum(in0, -1.0)" in src
    assert "libdevice.rsqrt" in src
    assert "libdevice.erf" in src


def test_source_casts():
    def fn(x):
        return x.half().float()

    x = tp.randn(8, 8)
    _, src = _source(fn, x)
    assert "tmp0 = in0.to(tl.float16)" in src
    assert "tmp1 = tmp0.to(tl.float32)" in src


def test_source_nested_where():
    def fn(x):
        return tp.where(x > 0, 1.0, tp.where(x < 0, -1.0, 0.0))

    x = tp.randn(8, 8)
    _, src = _source(fn, x)
    # inner select completes (tmp3) before the outer one consumes it (tmp5)
    assert "tmp3 = tl.where((tmp1) != 0.0, -1.0, 0.0)" in src
    assert "tmp5 = tl.where((tmp0) != 0.0, 1.0, tmp3)" in src


def test_source_dims_reduction_with_where_prologue():
    def fn(x, y):
        return tp.where(x > y, x, x * x).sum(dim=1)

    x = tp.randn(4, 6)
    gm = _pipeline(fn, x, x)
    red_node = [
        n for n in gm.graph.nodes if n.op == "call_method" and n.target == "sum"
    ][0]
    from tensorplay._stax.codegen.triton import _reduction_spec_from_node

    spec = _reduction_spec_from_node(red_node)
    seg_nodes = [
        n
        for n in gm.graph.nodes
        if n.op in {"call_function", "call_method"}
        and not (
            n.op in {"call_function", "call_method"}
            and red_node in n.args
            and n is not red_node
        )
    ]
    view, mapping, _ = _extract_segment_view(gm.graph, seg_nodes, red_node)
    built = _build_pointwise_program(
        view,
        skip_node=mapping[red_node],
        output_override=mapping[red_node.args[0]],
        allow_empty=True,
    )
    assert built is not None
    external_nodes, program, constants, _, output_ref = built
    gen = TritonProgramCodegen(
        program,
        constants,
        (output_ref,),
        len(external_nodes),
        reduction=spec,
        input_shapes=tuple((4, 6) for _ in external_nodes),
        reference_shape=(4, 6),
        value_dtype=str(tp.float32),
    )
    src = gen.generate("probe", fixed_config=(32, 4, 16, 2))
    assert "tl.where((tmp0) != 0.0, in0, tmp1)" in src
    assert "chunk = tl.sum" in src


def test_source_where_epilogue_after_full_reduction():
    def fn(x):
        s = (x * x).sum()
        return tp.where(s > 1.0, s, 0.0)

    x = tp.randn(16, 16)
    gm = _pipeline(fn, x)
    red_node = [
        n for n in gm.graph.nodes if n.op == "call_method" and n.target == "sum"
    ][0]
    from tensorplay._stax.codegen.triton import _reduction_spec_from_node

    spec = _reduction_spec_from_node(red_node)
    seg_nodes = [
        n
        for n in gm.graph.nodes
        if n.op in {"call_function", "call_method"}
        and not (
            n.op in {"call_function", "call_method"} and red_node in n.args and n is not red_node
        )
    ]
    view, mapping, _ = _extract_segment_view(gm.graph, seg_nodes, red_node)
    built = _build_pointwise_program(
        view,
        skip_node=mapping[red_node],
        output_override=mapping[red_node.args[0]],
        allow_empty=True,
    )
    assert built is not None
    external_nodes, program, constants, _, output_ref = built

    epi_nodes = [
        n
        for n in gm.graph.nodes
        if n.op in {"call_function", "call_method"} and red_node in n.args
    ]
    epi_view, epi_map, _ = _extract_segment_view(
        gm.graph, epi_nodes, epi_nodes[-1]
    )
    _, eprog, econst, _, _ = _build_pointwise_program(
        epi_view, output_override=epi_map[epi_nodes[-1]]
    )
    esrc = next(
        index
        for index, node in enumerate(epi_view.graph.placeholders)
        if node.name == red_node.name
    )
    gen = TritonProgramCodegen(
        program,
        constants,
        (output_ref,),
        len(external_nodes),
        reduction=spec,
        input_shapes=tuple((16, 16) for _ in external_nodes),
        reference_shape=(16, 16),
        value_dtype=str(tp.float32),
        epilogue=(eprog, econst, esrc),
    )
    for fixed in ((256, 4), (256, 4, 4)):
        src = gen.generate("probe", fixed_config=fixed)
        assert "etmp1 = (reduced > 1.0).to(tl.float32)" in src
        assert "etmp3 = tl.where((etmp1) != 0.0, reduced, 0.0)" in src


# --- CPU end-to-end semantics (decomposed graph runs eagerly) ----------------


_SELECT_FAMILY_CASES = {
    "elu": lambda t: tp.nn.functional.elu(t),
    "selu": lambda t: tp.nn.functional.selu(t),
    "gelu": lambda t: tp.nn.functional.gelu(t),
    "leaky_relu": lambda t: tp.nn.functional.leaky_relu(t, 0.05),
    "relu6": lambda t: tp.nn.functional.relu6(t),
    "hardtanh": lambda t: tp.nn.functional.hardtanh(t, -0.5, 1.5),
    "hardsigmoid": lambda t: tp.nn.functional.hardsigmoid(t),
    "hardshrink": lambda t: tp.nn.functional.hardshrink(t, 0.3),
    "softshrink": lambda t: tp.nn.functional.softshrink(t, 0.3),
    "threshold": lambda t: tp.nn.functional.threshold(t, 0.1, -2.0),
}


@pytest.mark.parametrize("name", sorted(_SELECT_FAMILY_CASES))
def test_select_family_numerics_match_eager(name):
    x = tp.tensor([-1.5, -0.2, 0.0, 0.3, 2.0])
    call = _SELECT_FAMILY_CASES[name]

    eager = call(x)
    gm = _pipeline(lambda t: call(t), x)
    interpreted = gm.recompile()
    got = interpreted(x)
    assert [float(v) for v in got.tolist()] == pytest.approx(
        [float(v) for v in eager.tolist()], rel=1e-6, abs=1e-6
    )


def test_clamp_decomposition_bounds_and_kwargs():
    x = tp.tensor([-3.0, 0.5, 7.0])

    for clamp in (
        lambda t: tp.clamp(t, -1.0, 1.0),
        lambda t: t.clamp(min=-1.0, max=1.0),
        lambda t: t.clamp(min=-1.0),
        lambda t: t.clamp(max=1.0),
    ):
        eager = clamp(x)
        gm = _pipeline(lambda t: clamp(t), x)
        got = gm.recompile()(x)
        assert [float(v) for v in got.tolist()] == pytest.approx(
            [float(v) for v in eager.tolist()]
        )


def test_where_comparison_end_to_end_cpu():
    x = tp.tensor([-1.0, 2.0, 0.5])
    y = tp.tensor([0.0, 1.0, 2.0])

    def fn(a, b):
        return tp.where(a > b, a * 2.0, tp.minimum(a, b))

    eager = fn(x, y)
    gm, built = _program(fn, x, y)
    assert built is not None
    got = gm.recompile()(x, y)
    assert [float(v) for v in got.tolist()] == pytest.approx(
        [float(v) for v in eager.tolist()]
    )
