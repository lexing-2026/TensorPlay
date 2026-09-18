"""Static fusion segmentation for the Stax backend (L5-M5c/M5g).

Splits a captured graph into an ordered list of fusion segments:

* pointwise runs merge into one segment (vertical fusion, pw→pw);
* a pointwise run may end with ONE reduction epilogue (``x.sum(dim)``
  family) — the pw→red vertical fusion;
* a pure pointwise chain that transitively consumes the reduction result
  folds back INTO the same kernel as a store-time epilogue (red→pw
  vertical fusion, ``Segment.epilogue``); chains reading anything else
  start a new kernel;
* back-to-back reductions split into separate segments;
* any non-pointwise, non-reduction operator becomes a ONE-NODE ``"extern"``
  segment: the backend runs it eagerly between fused kernels, so a graph
  may interleave compiled pointwise/reduction segments with unsupported
  operators instead of falling back as a whole.

Horizontal fusion (M5g): a later segment may read a value that is interior
to an earlier kernel.  Instead of giving the region up, the read is routed
into an EXTRA export of the producing segment — a pure-pointwise kernel
gains one additional store (``Segment.exports``) and the consumer wires
against that port.  Kernels that cannot carry spare stores (reduction
segments, whose post-accumulator register space is single-valued) keep a
single export and reject the region as before.

The scheduler owns no operator knowledge itself: the backend injects the
pointwise predicate and the reduction classifier, so this module stays free
of imports from ``backends``/``codegen`` (no cycles, one source of truth for
fusibility decisions — the old ad-hoc whole-graph detectors delegate here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from tensorplay.graph import GraphModule, Node


@dataclass
class Segment:
    """One planned kernel region."""

    nodes: Tuple[Node, ...]
    #: "pw" (pure pointwise), "pw+red" (pointwise prologue + reduction tail)
    #: or "extern" (single operator the backend runs eagerly)
    kind: str
    #: classifier result for the reduction tail when ``kind == "pw+red"``
    reduction: Any = None
    #: pointwise chain computed on the reduction result INSIDE the same
    ## kernel (store-time epilogue).  Every node here transitively consumes
    #: ``nodes[-1]``; anything else stays a separate segment.
    epilogue: Tuple[Node, ...] = ()
    #: Every value this kernel leaves in a runtime buffer: the main export
    #: first (epilogue tail when present, else the run tail), then extra
    #: stores for interior values later segments read (horizontal fusion).
    #: Each export shares the kernel's reference shape, so one extra store
    #: costs one output buffer, not a second kernel.
    exports: Tuple[Node, ...] = ()

    @property
    def tail(self) -> Node | None:
        return self.nodes[-1] if self.nodes else None

    @property
    def export_node(self) -> Node | None:
        """First value this segment produces (the default consumer port)."""

        if self.exports:
            return self.exports[0]
        if self.epilogue:
            return self.epilogue[-1]
        return self.nodes[-1] if self.nodes else None

    @property
    def producer(self) -> Node | None:
        """Input of the reduction tail (the pointwise chain result)."""

        if self.kind == "pw+red" and self.tail is not None:
            value = self.tail.args[0]
            return value if isinstance(value, Node) else None
        return None


def _flatten_values(value: Any):
    """Yield every ``Node`` inside a (possibly nested) arg structure."""

    if isinstance(value, Node):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_values(item)


def _epilogue_join(
    node: Node,
    reduction_tail: Node,
    epilogue_run: Tuple[Node, ...],
) -> bool:
    """May ``node`` continue the in-kernel epilogue after a reduction?

    The node must depend on something already living in the kernel's
    register space — the reduction result or an earlier epilogue node —
    and may otherwise read only graph placeholders (or scalars).  A pure
    pointwise node over placeholders alone would need the pre-reduction
    tile, which no longer exists past the reduction.
    """

    live = {reduction_tail, *epilogue_run}
    touches_live = False
    for dep in set(_flatten_values(node.args)) | set(
        _flatten_values(node.kwargs)
    ):
        if dep in live:
            touches_live = True
        elif dep.op != "placeholder":
            return False
    return touches_live


def segment_graph(
    graph_module: GraphModule,
    *,
    is_pointwise: Callable[[Node], bool],
    classify_reduction: Callable[[Node], Any],
) -> Optional[List[Segment]]:
    """Partition ``graph_module`` into ordered fusion segments.

    Returns ``None`` when the schedule is not expressible — notably when a
    later consumer needs a value interior to a REDUCTION kernel (whose
    register space past the accumulator is single-valued); the backend then
    falls back for the whole region.
    """

    segments: List[Segment] = []
    current: List[Node] = []
    current_reduction: Any = None
    current_epilogue: List[Node] = []
    #: extra exports accumulated for the OPEN run (consumers discovered
    #: while the run is still open are folded in when it closes)
    open_extras: List[Node] = []
    #: closed-segment node -> segment index
    owner: dict = {}

    def close() -> None:
        nonlocal current, current_reduction, current_epilogue, open_extras
        if not current:
            return
        kind = "pw+red" if current_reduction is not None else "pw"
        main = current_epilogue[-1] if current_epilogue else current[-1]
        index = len(segments)
        segments.append(
            Segment(
                nodes=tuple(current),
                kind=kind,
                reduction=current_reduction,
                epilogue=tuple(current_epilogue),
                exports=(main, *open_extras),
            )
        )
        for node in (*current, *current_epilogue):
            owner[node] = index
        current = []
        current_reduction = None
        current_epilogue = []
        open_extras = []

    def route(dependencies: set, close_open: bool) -> bool:
        """Absorb cross-kernel reads into producer exports.

        A dependency on a closed segment's interior becomes an extra store
        on that segment (pure-pointwise producers only).  ``close_open``
        marks a caller that is about to close the open run, so reads of the
        run's interior also promote; callers that merely continue the run
        read those values through registers and skip them.
        """

        open_main = (
            current_epilogue[-1]
            if current_epilogue
            else (current[-1] if current else None)
        )
        for dep in dependencies:
            if dep is open_main or dep in open_extras:
                continue
            if dep in current or dep in current_epilogue:
                if not close_open:
                    continue  # in-run read, no store needed
                if current_reduction is not None:
                    # pre-reduction values do not survive the accumulator
                    return False
                open_extras.append(dep)
                continue
            index = owner.get(dep)
            if index is None:
                continue  # placeholder or scalar
            producer = segments[index]
            if dep in producer.exports:
                continue  # already a store the consumer may read
            if producer.kind != "pw":
                # reduction/extern kernels carry no spare stores (v2 rule)
                return False
            producer.exports = producer.exports + (dep,)
        return True

    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        dependencies = set(_flatten_values(node.args)) | set(
            _flatten_values(node.kwargs)
        )
        if not route(dependencies, close_open=False):
            return None
        reduction = classify_reduction(node)
        if reduction is not None:
            if current_reduction is not None:
                # reduction feeding a reduction: kernel boundary between them
                if not route(dependencies, close_open=True):
                    return None
                close()
                current = [node]
            else:
                current.append(node)
            current_reduction = reduction
            continue
        if is_pointwise(node):
            if current_reduction is not None:
                # pw after a reduction joins the SAME kernel as a store-time
                # epilogue when it lives on the reduction's registers;
                # otherwise the kernel boundary falls here (v1 rule).
                if _epilogue_join(node, current[-1], tuple(current_epilogue)):
                    current_epilogue.append(node)
                    continue
                if not route(dependencies, close_open=True):
                    return None
                close()
            current.append(node)
            continue
        # extern operator: promote any read of the open run's interior into
        # an extra store, close the run, then record the operator as its own
        # single-node segment so the backend can interleave an eager call
        # between fused kernels.
        if not route(dependencies, close_open=True):
            return None
        close()
        segments.append(
            Segment(nodes=(node,), kind="extern", exports=(node,))
        )
        continue

    # The graph output counts as a consumer: a final value interior to the
    # last kernel is promoted to an extra store rather than forcing a
    # fallback.
    for out_node in graph_module.graph.outputs:
        if not route(
            set(_flatten_values(out_node.args))
            | set(_flatten_values(out_node.kwargs)),
            close_open=True,
        ):
            return None

    close()
    return segments or None


def describe(segments: List[Segment]) -> str:
    """Compact human-readable schedule, e.g. ``pw+red+ep -> extern -> pw``."""

    parts = []
    for segment in segments:
        label = segment.kind
        if segment.epilogue:
            label += "+ep"
        parts.append(label)
    return " -> ".join(parts)


def annotate(
    graph_module: GraphModule, segments: List[Segment]
) -> None:
    """Record the plan on the GraphModule for backends/debug tooling."""

    graph_module.meta["stax_segments"] = [
        {
            "kind": segment.kind,
            "nodes": [node.name for node in segment.nodes],
            "epilogue": [node.name for node in segment.epilogue],
            "exports": [node.name for node in segment.exports],
            "reduction": getattr(segment.reduction, "__dict__", segment.reduction),
        }
        for segment in segments
    ]
