"""L5-M5c static fusion segmentation semantics."""

import tensorplay as tp
from tensorplay.compiler.codegen.triton import _reduction_spec_from_node
from tensorplay.compiler.fx_passes import POINTWISE_FUSED_OP_NAMES
from tensorplay.compiler.graph import Tracer
from tensorplay.compiler.scheduler import annotate, describe, segment_graph


def _trace(fn, *args):
    sample = {name: value for name, value in zip(("x", "w"), args)}
    return Tracer().trace(fn, sample_inputs=sample)


def _segments(fn, *args):
    gm = _trace(fn, *args)

    def is_pointwise(node):
        return (
            node.op in {"call_function", "call_method"}
            and not node.kwargs
            and (
                node.target.__name__
                if callable(node.target)
                and hasattr(node.target, "__name__")
                else str(node.target)
            )
            in POINTWISE_FUSED_OP_NAMES
        )

    def classify(node):
        return (
            _reduction_spec_from_node(node) if node.op == "call_method" else None
        )

    gm._preds = (is_pointwise, classify)
    return gm, segment_graph(gm, is_pointwise=is_pointwise,
                             classify_reduction=classify)


def test_pointwise_run_is_one_segment():
    x = tp.tensor([1.0, -2.0])
    gm, segs = _segments(lambda t: ((t * 2.0).relu() + 1.0).sigmoid(), x)
    assert segs is not None and len(segs) == 1
    assert segs[0].kind == "pw"
    assert len(segs[0].nodes) == 4


def test_pointwise_then_full_sum_fuses_vertically():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(lambda t: ((t * 2.0).relu()).sum(), x)
    assert segs is not None and len(segs) == 1
    assert segs[0].kind == "pw+red"
    assert segs[0].reduction.op == "sum" and segs[0].reduction.is_full
    assert segs[0].producer.op == "call_method"  # the relu node


def test_reduction_then_pointwise_fuses_into_epilogue():
    """M5e: relu(sum) runs INSIDE the reduction kernel as a store epilogue."""

    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(lambda t: ((t * 2.0).sum(dim=1)).relu(), x)
    assert segs is not None and len(segs) == 1
    assert describe(segs) == "pw+red+ep"
    assert segs[0].reduction.dims == (1,)
    assert len(segs[0].epilogue) == 1
    assert str(segs[0].epilogue[0].target) == "relu"
    assert segs[0].export_node is segs[0].epilogue[-1]


def test_epilogue_chain_requires_transitive_dependency():
    """A pw node reading only placeholders cannot join the epilogue run."""

    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])

    def fn(t):
        s = t.sum(dim=1)
        return s.relu() + (t * 2.0).sqrt()

    _, segs = _segments(fn, x)
    assert segs is not None and len(segs) == 2
    # sum+relu fuse; the placeholder-only sqrt chain starts a new kernel;
    # the cross-kernel add closes segment 2.
    assert describe(segs) == "pw+red+ep -> pw"
    assert [n.name for n in segs[0].epilogue] == ["relu"]


def test_back_to_back_reductions_split():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(lambda t: t.sum(dim=1).sum(), x)
    assert segs is not None and len(segs) == 2
    assert all(s.kind == "pw+red" for s in segs)
    # inner segment has no pointwise prologue: its producer IS a reduction
    assert segs[0].reduction.dims == (1,)
    assert segs[1].reduction.is_full
    assert segs[1].producer is not None and segs[1].producer.op == "call_method"


def test_extern_op_falls_back_whole_graph():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(
        lambda t: ((t * 2.0).reshape(4)).sum(), x
    )  # reshape is not pointwise-fusible
    assert segs is None


def test_bare_input_reduction_segment():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    gm, segs = _segments(lambda t: (t * 2.0).amax(dim=0), x)
    assert segs is not None and len(segs) == 1
    assert segs[0].kind == "pw+red"

    # reduction applied directly to the input: legal segmentation even
    # though there is no pointwise prologue to fuse
    gm2, segs2 = _segments(lambda t: t.amax(dim=0), x)
    assert segs2 is not None and segs2[0].kind == "pw+red"
    assert segs2[0].producer.op == "placeholder"


def test_annotate_records_plan_in_meta():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    gm, segs = _segments(lambda t: ((t * 2.0).sum(dim=1)).relu(), x)
    annotate(gm, segs)
    plan = gm.meta["stax_segments"]
    assert [entry["kind"] for entry in plan] == ["pw+red"]
    assert plan[0]["epilogue"] == [segs[0].epilogue[0].name]
    assert plan[0]["nodes"] == [
        n.name for n in segs[0].nodes
    ]


# --- M5c per-segment emission wiring ---------------------------------------------


def _plan_harness(fn, *args):
    """Run the real scheduler + extractor the way codegen does."""
    from types import SimpleNamespace

    from tensorplay.backends.stax import _build_pointwise_program
    from tensorplay.compiler.codegen.triton import (
        _ExternSource,
        _extract_segment_view,
    )
    from tensorplay.compiler.codegen.triton import (
        _reduction_spec_from_node as classify,
    )

    gm = _trace(fn, *args)

    def is_pointwise(node):
        return (
            node.op in {"call_function", "call_method"}
            and not node.kwargs
            and str(getattr(node.target, "__name__", node.target))
            in POINTWISE_FUSED_OP_NAMES
        )

    segments = segment_graph(
        gm, is_pointwise=is_pointwise, classify_reduction=lambda n: (
            classify(n) if n.op == "call_method" else None
        )
    )
    assert segments is not None
    plans = []
    for seg in segments:
        view, mapping, externals = _extract_segment_view(gm.graph, seg.nodes, seg.tail)
        if seg.kind == "pw+red":
            producer_new = mapping.get(seg.producer) or externals[seg.producer]
            prog = _build_pointwise_program(
                SimpleNamespace(graph=view.graph),
                skip_node=mapping[seg.tail],
                output_override=producer_new,
            )
        else:
            prog = _build_pointwise_program(SimpleNamespace(graph=view.graph))
        plans.append((seg, prog))
    return gm, plans


def test_pw_red_pw_fuses_to_single_segment_with_epilogue_program():
    """M5e: pw→red→pw lowers as ONE segment; epilogue builds its own program."""

    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])

    def fn(x):
        return ((x * 2.0).relu()).sum(dim=1) * 3.0

    gm, plans = _plan_harness(fn, x)
    assert len(plans) == 1
    seg0, prog0 = plans[0]
    assert seg0.kind == "pw+red" and len(seg0.epilogue) == 1
    # main program: pointwise chain folded with its sum tail -> one scalar
    assert prog0 is not None and len(prog0[1]) > 0

    # epilogue program follows compile_graph_module's construction
    from tensorplay.compiler.codegen.triton import _extract_segment_view
    from tensorplay.backends.stax import _build_pointwise_program
    from types import SimpleNamespace as NS

    view, epi_mapping, epi_externals = _extract_segment_view(
        gm.graph, list(seg0.epilogue), seg0.epilogue[-1]
    )
    # the reduction tail resolves to the view's single external placeholder
    assert len(epi_externals) == 1 and seg0.tail in epi_externals
    built = _build_pointwise_program(
        NS(graph=view.graph),
        output_override=epi_mapping[seg0.epilogue[-1]],
    )
    assert built is not None
    placeholders_e, eprog, econst, _, eref = built
    # single external input (the reduction result) at ref 0
    assert len(placeholders_e) == 1
    assert eref >= 1 and len(eprog) % 3 == 0


def test_scalar_intermediate_folds_into_epilogue():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    gm, plans = _plan_harness(lambda t: t.sum() * 2.0 + 1.0, x)
    assert len(plans) == 1
    seg0 = plans[0][0]
    # full reduction with a two-node epilogue chain (mul, add)
    assert seg0.reduction.is_full and len(seg0.epilogue) == 2
