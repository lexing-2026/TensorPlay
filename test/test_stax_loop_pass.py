"""Loop-level optimization passes for the Stax Triton emission.

Unit tests pin the pass applicability and candidate construction; source
structure tests pin the emitted kernel forms (vectorized pointwise
addressing, unrolled and split reduction loops); GPU numeric tests pin the
semantics of every knob against eager.
"""

import pytest

import tensorplay as tp
from tensorplay._stax.codegen import triton as st
from tensorplay._stax.codegen.triton import (
    ReductionSpec,
    TritonProgramCodegen,
    _build_pointwise_program,
    _extract_segment_view,
    _reduction_spec_from_node,
)
from tensorplay._stax.codegen.loop_pass import (
    dims_loop_candidates,
    pointwise_loop_candidates,
    split_applies,
    unroll_applies,
    vectorize_applies,
)
from tensorplay._stax.codegen.triton import _DIM_REDUCTION_CANDIDATES
from tensorplay.graph import Tracer
from tensorplay.graph.passes import (
    DeadCodeElimination,
    DecomposePass,
    NormalizeOperators,
    PassManager,
)

GPU = pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")


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


def _pointwise_source(fn, *inputs, fixed_config):
    gm = _pipeline(fn, *inputs)
    built = _build_pointwise_program(gm)
    assert built is not None
    external_nodes, program, constants, _, output_ref = built
    shape = tuple(int(dim) for dim in inputs[0].shape)
    gen = TritonProgramCodegen(
        program,
        constants,
        (output_ref,),
        len(external_nodes),
        input_shapes=tuple(shape for _ in external_nodes),
        reference_shape=shape,
    )
    return gen.generate("probe", fixed_config=fixed_config)


def _dims_source(fn, *inputs, fixed_config, reduction_outputs=None, target="sum"):
    gm = _pipeline(fn, *inputs)
    red_node = [
        n
        for n in gm.graph.nodes
        if n.op == "call_method" and str(n.target) == target
    ][0]
    spec = _reduction_spec_from_node(red_node)
    seg_nodes = [
        n
        for n in gm.graph.nodes
        if n.op in {"call_function", "call_method"}
        and not (red_node in n.args and n is not red_node)
    ]
    view, mapping, externals = _extract_segment_view(
        gm.graph, seg_nodes, red_node
    )
    producer = red_node.args[0]
    producer_new = mapping.get(producer)
    if producer_new is None:
        producer_new = externals[producer]
    built = _build_pointwise_program(
        view,
        skip_node=mapping[red_node],
        output_override=producer_new,
        allow_empty=True,
    )
    assert built is not None
    external_nodes, program, constants, _, output_ref = built
    shape = tuple(int(dim) for dim in inputs[0].shape)
    gen = TritonProgramCodegen(
        program,
        constants,
        (output_ref,),
        len(external_nodes),
        reduction=spec,
        input_shapes=tuple(shape for _ in external_nodes),
        reference_shape=shape,
        value_dtype=str(tp.float32),
        reduction_outputs=reduction_outputs,
    )
    return gen.generate("probe", fixed_config=fixed_config)


# --- pass applicability --------------------------------------------------------


def test_vectorize_applies_needs_alignment_and_width():
    assert vectorize_applies(64, 4, 1)
    assert vectorize_applies(64, 4, 4)
    assert not vectorize_applies(65, 4, 4)  # element count not divisible
    assert not vectorize_applies(64, 4, 8)  # 8 fp32 = 32B > one vector
    assert vectorize_applies(64, 2, 8)  # 8 fp16 = 16B fits


def test_unroll_and_split_need_a_loop_with_remainder():
    assert unroll_applies(1024, 128, 4)
    assert not unroll_applies(128, 128, 2)  # one tile: no loop
    assert not split_applies(1024, 128)  # exact tiling: already mask-free
    assert split_applies(1000, 128)  # partial tail tile to peel
    assert not split_applies(100, 128)


def test_pointwise_loop_candidates_prune_by_geometry():
    base = ((64, 4), (128, 8))
    # 64 elements, fp32: width 4 divides, width 2 divides
    table = pointwise_loop_candidates(64, 4, base)
    assert (64, 4) in table and (64, 4, 2) in table and (64, 4, 4) in table
    assert all(entry[2] != 8 for entry in table if len(entry) > 2)
    # 65 elements: only the neutral width survives
    table65 = pointwise_loop_candidates(65, 4, base)
    assert table65 == ((64, 4), (128, 8))


def test_dims_loop_candidates_cover_six_tuples():
    table = dims_loop_candidates(3000, ((16, 4, 1024, 3), (32, 4, 3)))
    # 3-tuple geometry materializes its derived RBLOCK; both loop knobs
    # appear; split is pruned for the exact tiling (3000 % 1024 != 0 -> kept)
    assert all(len(entry) == 6 for entry in table)
    quad = next(e for e in table if e[:4] == (16, 4, 1024, 3))
    assert quad[4] in (1, 2, 4)
    assert quad[5] in (0, 1)
    # exact tiling (rnumel % rblock == 0) carries no split variants
    exact = dims_loop_candidates(2048, ((16, 4, 1024, 3),))
    assert all(entry[5] == 0 for entry in exact)


# --- emitted kernel forms ------------------------------------------------------


def test_pointwise_vectorized_addressing_emitted():
    x = tp.randn(8, 8)
    src = _pointwise_source(lambda t: t * 2.0 + 1.0, x, fixed_config=(16, 4, 4))
    assert "VEC: tl.constexpr" in src
    assert "xlane = tl.arange(0, VEC)" in src
    assert "xindex = xoffset + xrow[:, None] * VEC + xlane[None, :]" in src
    # grid covers XBLOCK*VEC elements per program (64 / (16*4) = 1)
    assert "[(1,)]" in src
    # the packed row mask is 2-D
    assert "xmask = (xoffset + xrow * 4)[:, None] < xnumel" in src


def test_pointwise_vectorize_falls_back_when_not_divisible_shape():
    x = tp.randn(9, 5)  # 45 elements
    src = _pointwise_source(lambda t: t * 2.0 + 1.0, x, fixed_config=(16, 4, 4))
    assert "VEC" not in src  # 45 % 4 != 0


def test_dims_unrolled_loop_emitted():
    x = tp.randn(4, 1000)
    src = _dims_source(
        lambda t: (t * 2.0).sum(dim=1),
        x,
        fixed_config=(16, 4, 256, 3, 4, 0),
    )
    assert "loop_unroll_factor=4" in src
    assert "num_stages=3, loop_unroll_factor=4" in src


def test_dims_split_loop_peels_masked_tail():
    x = tp.randn(4, 1000)  # 1000 % 256 != 0
    src = _dims_source(
        lambda t: (t * 2.0).sum(dim=1),
        x,
        fixed_config=(16, 4, 256, 3, 1, 1),
    )
    # main loop over the full tiles only (768 = 3 * 256)
    assert "for roffset in tl.range(0, 768, RBLOCK" in src
    # the main body carries no r-side predication
    assert "rmask = rindex < 1000" in src  # exactly once: the peeled tail
    assert src.count("rmask =") == 1


def test_dims_plain_config_stays_loop_neutral():
    x = tp.randn(4, 1000)
    src = _dims_source(
        lambda t: (t * 2.0).sum(dim=1),
        x,
        fixed_config=(16, 4, 256, 3),
    )
    assert "loop_unroll_factor" not in src
    assert "for roffset in tl.range(0, 1000, RBLOCK" in src
    assert src.count("rmask =") == 1


# --- pair emission structure ---------------------------------------------------


def test_pair_dual_stream_stores_both_ports():
    x = tp.randn(4, 1000)
    src = _dims_source(
        lambda t: t.max(dim=1),
        x,
        fixed_config=(16, 4, 256, 3),
        reduction_outputs=("values", "indices"),
        target="max",
    )
    assert "tl.argmin" not in src
    assert "cwin = tl.argmax(prio, axis=1) + roffset" in src
    assert "tl.store(out_ptr0 + xindex, acc" in src
    assert "tl.store(out_ptr1 + xindex, acci" in src


def test_pair_min_polarity_uses_argmin():
    x = tp.randn(4, 1000)
    src = _dims_source(
        lambda t: t.min(dim=1),
        x,
        fixed_config=(16, 4, 256, 3),
        reduction_outputs=("values", "indices"),
        target="min",
    )
    assert "cwin = tl.argmin(prio, axis=1) + roffset" in src
    assert "prio = tl.where(isnan_, -1.0e38" in src
    assert "tl.store(out_ptr0 + xindex, acc" in src
    assert "tl.store(out_ptr1 + xindex, acci" in src


# --- GPU numeric equivalence ----------------------------------------------------


@GPU
def test_vectorized_pointwise_matches_eager_gpu():
    device = tp.device("cuda", 0)

    def fn(t):
        return t * 2.0 + 1.0

    x = tp.randn(1024, device=device)
    built = _build_pointwise_program(_pipeline(fn, x))
    external_nodes, program, constants, _, output_ref = built
    shape = (1024,)
    gen_launch = st._compile_program(
        program,
        constants,
        (output_ref,),
        [x],
        fixed_config=(128, 4, 4),
        input_shapes=(shape,),
        reference_shape=shape,
    )
    out = gen_launch([x])
    ref = fn(x)
    assert tp.abs(out - ref).max().item() < 1e-5


@GPU
def test_unrolled_and_split_reduction_match_eager_gpu():
    device = tp.device("cuda", 0)
    x = tp.randn(64, 1000, device=device)
    for config in (
        (16, 4, 256, 3, 4, 0),
        (16, 4, 256, 3, 1, 1),
        (16, 4, 256, 3, 2, 1),
    ):
        gm = _pipeline(lambda t: (t * 2.0).sum(dim=1), x)
        red_node = [
            n
            for n in gm.graph.nodes
            if n.op == "call_method" and str(n.target) == "sum"
        ][0]
        spec = _reduction_spec_from_node(red_node)
        view, mapping, _ = _extract_segment_view(
            gm.graph, [n for n in gm.graph.nodes if n.op != "placeholder"], red_node
        )
        built = _build_pointwise_program(
            view,
            skip_node=mapping[red_node],
            output_override=mapping[red_node.args[0]],
            allow_empty=True,
        )
        external_nodes, program, constants, _, output_ref = built
        launch = st._compile_program(
            program,
            constants,
            (output_ref,),
            [x],
            fixed_config=config,
            reduction=spec,
            input_shapes=((64, 1000),),
            reference_shape=(64, 1000),
            value_dtype=str(tp.float32),
        )
        out = launch([x])
        ref = (x * 2.0).sum(dim=1)
        assert tp.abs(out - ref).max().item() < 1e-4, config


@GPU
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("form", ["values", "indices", "getitem", "keepdim"])
def test_pair_reduction_forward_matches_eager_gpu(op, form):
    device = tp.device("cuda", 0)

    if form == "values":
        fn = lambda t: getattr(t, op)(dim=1).values  # noqa: E731
    elif form == "indices":
        fn = lambda t: getattr(t, op)(dim=1).indices.to(tp.float32)  # noqa: E731
    elif form == "getitem":
        fn = lambda t: getattr(t, op)(dim=1)[0]  # noqa: E731
    else:
        fn = lambda t: getattr(t, op)(dim=1, keepdim=True).values  # noqa: E731

    x = tp.randn(37, 514, device=device)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(x)
    ref = fn(x)
    assert out.shape == ref.shape
    if form == "indices":
        assert (out.long() == ref.long()).all().item()
    else:
        assert tp.abs(out - ref).max().item() < 1e-5


@GPU
def test_pair_reduction_forward_with_nan_gpu():
    device = tp.device("cuda", 0)
    x = tp.randn(8, 9, device=device)
    x[3, 4] = float("nan")
    x[5, 8] = float("nan")
    compiled = tp.compile(
        lambda t: t.max(dim=1).values, fullgraph=True
    )
    out = compiled(x)
    ref = x.max(dim=1).values
    assert tp.equal((out != out), (ref != ref))
    finite = out == out
    assert tp.abs(out[finite] - ref[finite]).max().item() < 1e-5


@GPU
@pytest.mark.parametrize("op", ["max", "min"])
def test_bare_pair_reduction_trains_via_scatter_gpu(op):
    device = tp.device("cuda", 0)

    def fn(t):
        return getattr(t, op)(dim=1).values.sum()

    xc = tp.randn(4, 128, device=device, requires_grad=True)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(xc)
    out.backward()

    xr = xc.detach().clone().requires_grad_(True)
    ref = fn(xr)
    ref.backward()

    assert tp.abs(out - ref).max().item() < 1e-5
    assert tp.abs(xc.grad - xr.grad).max().item() < 1e-5


@GPU
def test_pair_over_extern_export_trains_gpu():
    """A bare pair reduction whose input is an eager operator's output
    (extern export) still trains through the scatter VJP."""

    def fn(t):
        return (t * 3.0 + 1.0).max(dim=1).values.sum()

    xc = tp.randn(8, 64, device=tp.device("cuda", 0), requires_grad=True)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(xc)
    out.backward()

    xr = xc.detach().clone().requires_grad_(True)
    ref = fn(xr)
    ref.backward()
    assert tp.abs(out - ref).max().item() < 1e-5
    assert tp.abs(xc.grad - xr.grad).max().item() < 1e-5


@GPU
def test_argmax_in_grad_graph_compiles_and_drops_tangent_gpu():
    """A training region containing argmax compiles: the index reduction
    closes with the no-tangent rule, its integer output flows to an extern
    consumer (index_select) whose engine VJP skips the non-float operand,
    and the argmax branch contributes no gradient — eager semantics."""

    device = tp.device("cuda", 0)
    table = tp.randn(32, 4, device=device)

    def fn(t):
        return (t * 2.0).sum() + tp.index_select(
            table, 0, t.argmax(dim=1)
        ).sum()

    # the frontend imports its own backend module instance; spy the
    # canonical entry after a warmup compile to prove the fused lowering
    # claimed the region (the argmax segment launches as a fused kernel)
    import sys

    tp.compile(lambda a: a * 2.0, fullgraph=True)(
        tp.rand(4, device=device)
    )
    canonical = sys.modules[
        "tensorplay.compiler.backends.stax.codegen.triton"
    ]
    seen = []
    original_launch = canonical._autotune_launch

    def spy(name, *args, **kwargs):
        seen.append(name)
        return original_launch(name, *args, **kwargs)

    canonical._autotune_launch = spy

    xc = tp.randn(8, 32, device=device, requires_grad=True)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(xc)
    out.backward()
    canonical._autotune_launch = original_launch

    assert any(name.startswith("fwd") for name in seen)
    xr = xc.detach().clone().requires_grad_(True)
    ref = fn(xr)
    ref.backward()
    assert tp.abs(out - ref).max().item() < 1e-4
    assert tp.abs(xc.grad - xr.grad).max().item() < 1e-5


@GPU
def test_get_attr_region_trains_gpu():
    """A module parameter lifted as get_attr trains: its gradient returns
    to the leaf through the trailing autograd input."""

    import operator

    from tensorplay.graph import Graph, GraphModule
    import tensorplay.nn as nn

    class _Holder(nn.Module):
        pass

    device = tp.device("cuda", 0)
    root = _Holder()
    root.scale = tp.randn(6, device=device)
    root.weight = tp.randn(6, device=device)
    scale = root.scale.detach().clone().requires_grad_(True)
    weight = root.weight.detach().clone().requires_grad_(True)
    root.scale = scale
    root.weight = weight

    g = Graph()
    x = g.placeholder("x")
    s = g.get_attr("scale")
    w = g.get_attr("weight")
    g.output(g.call_function(operator.add, (
        g.call_function(operator.mul, (x, s)), w)))

    x_in = tp.randn(4, 6, device=device, requires_grad=True)
    compiled = st.compile_graph_module(GraphModule(root, g), [x_in])
    assert compiled is not None
    assert compiled._tensorplay_codegen == "triton"
    assert compiled._tensorplay_backward_codegen == "triton"

    out = compiled(x_in)
    out.sum().backward()

    # eager reference on fresh clones: backward accumulates, so the same
    # leaves must not be differentiated twice
    xr = x_in.detach().clone().requires_grad_(True)
    sr = scale.detach().clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    (xr * sr + wr).sum().backward()

    assert tp.abs(out - (x_in * scale + weight)).max().item() < 1e-5
    assert tp.abs(x_in.grad - xr.grad).max().item() < 1e-5
    assert scale.grad is not None
    assert tp.abs(scale.grad - sr.grad).max().item() < 1e-5
    assert weight.grad is not None
    assert tp.abs(weight.grad - wr.grad).max().item() < 1e-5
