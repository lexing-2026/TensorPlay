"""Static fusion segmentation semantics."""

import tensorplay as tp
from tensorplay._stax.codegen.triton import _reduction_spec_from_node
from tensorplay.graph.passes import POINTWISE_FUSED_OP_NAMES
from tensorplay.graph import Tracer
from tensorplay._stax.scheduler import annotate, describe, segment_graph


def _trace(fn, *args):
    sample = {name: value for name, value in zip(("x", "w"), args)}
    return Tracer().trace(fn, sample_inputs=sample)


def _segments(fn, *args, **kwargs):
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
                             classify_reduction=classify,
                             **kwargs)


def _name(node) -> str:
    """Target spelling shared by call_function and call_method nodes."""

    target = node.target
    return getattr(target, "__name__", str(target))


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
    """relu(sum) runs INSIDE the reduction kernel as a store epilogue."""

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


def test_extern_op_is_a_segment_barrier():
    """Unsupported operators become eager one-node segments, not a bail."""

    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(
        lambda t: ((t * 2.0).reshape(4)).sum(), x
    )  # reshape is not pointwise-fusible
    assert segs is not None
    assert describe(segs) == "pw -> extern -> pw+red"
    assert segs[1].kind == "extern"
    assert segs[1].nodes[0].op == "call_method"
    assert segs[1].export_node is segs[1].nodes[0]


def test_mixed_graph_interleaves_fused_and_eager():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(lambda t: t.softmax(dim=1).exp() + 1.0, x)
    assert segs is not None
    # the eager operator carries the single-user pointwise tail itself
    # (store-time epilogue); no separate pw kernel follows
    assert describe(segs) == "extern+ep"


def test_interior_value_across_barrier_gains_extra_export():
    """Horizontal fusion: a later segment reading an interior value
    makes the producer store it as an extra export, not a fallback."""

    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])

    def fn(t):
        g = (t * 2.0).relu()  # interior: consumed by h and by the output
        h = g * 3.0
        return t.softmax(dim=1) * h * g

    _, segs = _segments(fn, x)
    assert segs is not None and describe(segs) == "pw -> extern -> pw"
    assert segs[0].export_node is segs[0].nodes[-1]  # h, the main export
    assert segs[0].exports == (segs[0].nodes[-1], segs[0].nodes[1])  # + g
    assert len(segs[2].exports) == 1


def test_independent_chains_share_one_kernel():
    """Two sibling pointwise chains coalesce into one multi-export kernel."""

    x = tp.tensor([1.0, 2.0])
    _, segs = _segments(lambda t: (t * 2.0) + (t * 3.0).sigmoid(), x)
    assert segs is not None and len(segs) == 1
    assert segs[0].kind == "pw"
    assert len(segs[0].exports) == 1  # only the final add crosses out


def test_final_value_interior_to_last_kernel_is_promoted():
    x = tp.tensor([1.0, 2.0])

    def fn(t):
        a = t * 2.0
        b = t * 3.0
        return a  # b stays in the same run; a becomes an extra store

    _, segs = _segments(fn, x)
    assert segs is not None and len(segs) == 1
    assert segs[0].export_node is segs[0].nodes[-1]  # b
    assert segs[0].exports[-1] is segs[0].nodes[0]  # a, the graph output


def test_reduction_interior_across_segments_still_falls_back():
    """Pre-reduction values do not survive the accumulator: no extra store."""

    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])

    def fn(t):
        p = (t * 2.0).relu()  # interior of the pw+red kernel
        return p.sum() * p.mean()

    _, segs = _segments(fn, x)
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


# --- per-segment emission wiring ---------------------------------------------


def _plan_harness(fn, *args):
    """Run the real scheduler + extractor the way codegen does."""
    from types import SimpleNamespace

    from tensorplay._stax.stax import _build_pointwise_program
    from tensorplay._stax.codegen.triton import (
        _ExternSource,
        _extract_segment_view,
    )
    from tensorplay._stax.codegen.triton import (
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
    """pw→red→pw lowers as ONE segment; epilogue builds its own program."""

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
    from tensorplay._stax.codegen.triton import _extract_segment_view
    from tensorplay._stax.stax import _build_pointwise_program
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


def test_training_schedule_splits_epilogue():
    """Training scheduling keeps the store-time epilogue out of the
    reduction kernel: it becomes its own pointwise segment closed by a
    local VJP; inference keeps the join."""

    gm, segs = _segments(lambda t: t.sum() * 2.0 + 1.0)
    assert describe(segs) == "pw+red+ep"

    _, split = _segments(lambda t: t.sum() * 2.0 + 1.0, allow_epilogue=False)
    assert describe(split) == "pw+red -> pw"
    red, epi = split
    assert red.reduction.is_full and not red.epilogue
    assert len(epi.nodes) == 2
    # the split epilogue reads the reduction export, not an interior value
    assert epi.exports == (epi.nodes[-1],)


def test_pair_reduction_projections_become_exports():
    """max(dim) folds as a pair reduction; the value/index projection
    nodes are its exports (in encounter order, with per-port kinds)."""

    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])

    def fn(t):
        m = t.max(dim=1)
        return m.values * 2.0 + m.indices.to(tp.float32)

    gm, segs = _segments(fn, x)
    assert segs is not None
    assert [s.kind for s in segs] == ["pw+red", "pw"]
    pair, tail = segs
    assert pair.reduction.op == "max" and pair.reduction.is_pair
    assert pair.export_kinds == ("values", "indices")
    assert all(node.op == "call_function" for node in pair.exports)
    assert pair.exports[0].args[1] == "values"
    assert pair.exports[1].args[1] == "indices"
    # the tail pointwise segment wires against the projection ports
    assert gm is not None


def test_pair_reduction_values_only_consumption():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(lambda t: t.max(dim=1).values * 2.0, x)
    assert segs is not None and len(segs) == 2
    assert segs[0].kind == "pw+red"
    # only the values stream was consumed: one port, that kind
    assert segs[0].export_kinds == ("values",)
    assert segs[1].kind == "pw"


def test_pair_reduction_getitem_projection():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(lambda t: t.max(dim=1)[1].to(tp.float32), x)
    assert segs is not None and len(segs) == 2
    assert segs[0].export_kinds == ("indices",)


def test_pair_projection_attaches_after_segment_closed():
    """A projection arriving after the pair segment closed (another node
    intervened) still attaches to that segment as an export."""

    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])

    def fn(t):
        pair = t.max(dim=1)
        other = t * 2.0
        return pair.values + other

    _, segs = _segments(fn, x)
    assert segs is not None
    assert [seg.kind for seg in segs] == ["pw+red", "pw"]
    assert segs[0].export_kinds == ("values",)
    # the later pw run reads the projection, wired as the pair's export
    assert segs[1].exports == (segs[1].nodes[-1],)


def test_pair_tail_as_graph_output_has_no_exports():
    """The raw pair node is not a wireable value: a region returning it
    exports nothing from the pair segment, and the backend falls back."""

    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(lambda t: t.max(dim=1), x)
    assert segs is not None
    assert segs[0].kind == "pw+red" and segs[0].exports == ()


def test_min_dim_is_pair_amax_is_not():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, segs = _segments(lambda t: t.min(dim=1).values, x)
    assert segs is not None
    assert segs[0].reduction.op == "min" and segs[0].reduction.is_pair

    _, segs2 = _segments(lambda t: t.amax(dim=1), x)
    assert segs2 is not None
    assert not segs2[0].reduction.is_pair


# --- extern store-time epilogue ------------------------------------------


def test_extern_pointwise_tail_attaches_as_epilogue():
    """A single-user pointwise chain over the operator output folds into
    the extern segment; the chain tail is the segment's only export."""

    w = tp.randn(4, 6)
    _, segs = _segments(lambda t: ((t @ w).relu() + 1.0).sigmoid(), tp.randn(3, 4))
    assert segs is not None and len(segs) == 1
    assert segs[0].kind == "extern"
    assert describe(segs) == "extern+ep"
    assert [_name(n) for n in segs[0].epilogue] == [
        "relu", "add", "sigmoid",
    ]
    assert segs[0].exports == (segs[0].epilogue[-1],)


def test_extern_epilogue_requires_single_user_chain():
    """A second consumer of the raw operator output blocks the attach: the
    whole chain keeps its own pw segment so every reader stays wired."""

    w = tp.randn(4, 6)

    def fn(t):
        y = t @ w
        return y.relu() + y

    _, segs = _segments(fn, tp.randn(3, 4))
    assert segs is not None and [seg.kind for seg in segs] == ["extern", "pw"]
    assert segs[0].epilogue == ()
    assert segs[0].exports == (segs[0].nodes[-1],)


def test_extern_epilogue_stops_at_sibling_reader():
    """The chain continues while each value feeds exactly one node; a
    branching reader ends the fold at the last linear value."""

    w = tp.randn(4, 6)

    def fn(t):
        y = (t @ w).relu()
        return y * 2.0 + y * 3.0

    _, segs = _segments(fn, tp.randn(3, 4))
    assert segs is not None and [seg.kind for seg in segs] == [
        "extern", "pw",
    ]
    assert [_name(n) for n in segs[0].epilogue] == ["relu"]
    # the branching tail reads the folded chain through the export
    assert segs[0].exports == (segs[0].epilogue[-1],)


def test_extern_epilogue_needs_only_tensor_input():
    """A chain node reading a placeholder as well cannot fold (the fused
    tile would need a second input); it starts a regular pw run."""

    w = tp.randn(4, 6)
    x = tp.randn(3, 4)
    z = tp.randn(3, 6)
    _, segs = _segments(lambda t, u: (t @ w).relu() + u, x, z)
    assert segs is not None and [seg.kind for seg in segs] == [
        "extern", "pw",
    ]
    assert [str(n.target) for n in segs[0].epilogue] == ["relu"]


def test_training_schedule_carries_no_extern_epilogue():
    w = tp.randn(4, 6)
    _, segs = _segments(
        lambda t: (t @ w).relu(),
        tp.randn(3, 4),
        allow_epilogue=False,
    )
    assert segs is not None and describe(segs) == "extern -> pw"


def test_extern_epilogue_feeds_following_segment():
    """The folded chain's export wires into a later reduction segment."""

    w = tp.randn(4, 6)
    _, segs = _segments(lambda t: ((t @ w).relu()).sum(dim=1), tp.randn(3, 4))
    assert segs is not None
    assert describe(segs) == "extern+ep -> pw+red"
    assert segs[1].producer is segs[0].epilogue[-1]
