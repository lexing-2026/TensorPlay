"""Static fusion segmentation for the Stax backend (L5-M5c).

Splits a captured graph into an ordered list of fusion segments using the
Inductor vertical-fusion rule in miniature:

* pointwise runs merge into one segment;
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
  operators instead of falling back as a whole.  Whether a particular
  mixed schedule can be wired remains the backend's decision (an extern
  result still has to reach later segments as a single export value).

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
    ## kernel (store-time epilogue, Inductor's red→pw vertical fusion).
    #: Every node here transitively consumes ``nodes[-1]``; anything else
    ## stays a separate segment.
    epilogue: Tuple[Node, ...] = ()

    @property
    def tail(self) -> Node | None:
        return self.nodes[-1] if self.nodes else None

    @property
    def export_node(self) -> Node | None:
        """Last value this segment produces (epilogue tail when present)."""

        if self.epilogue:
            return self.epilogue[-1]
        return self.tail

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

    Returns ``None`` when the schedule is not expressible with single-output
    segments — notably when a later segment needs a value that is interior
    to an earlier one (multi-output kernels are not modelled yet); the
    backend then falls back for the whole region.
    """

    segments: List[Segment] = []
    current: List[Node] = []
    current_reduction: Any = None
    current_epilogue: List[Node] = []
    #: Values of already-closed segments that are NOT that segment's export.
    #: A later segment consuming one of these needs a second output port on
    #: the producing kernel (multi-output segments are not modelled yet), so
    #: any such dependency is rejected for the whole region.
    interior_values: set = set()

    def close() -> None:
        nonlocal current, current_reduction, current_epilogue
        if not current:
            return
        kind = "pw+red" if current_reduction is not None else "pw"
        segments.append(
            Segment(
                nodes=tuple(current),
                kind=kind,
                reduction=current_reduction,
                epilogue=tuple(current_epilogue),
            )
        )
        export = current_epilogue[-1] if current_epilogue else current[-1]
        interior_values.update(
            node for node in current if node is not export
        )
        interior_values.update(
            node for node in current_epilogue if node is not export
        )
        current = []
        current_reduction = None
        current_epilogue = []

    def close_and_guard(dependencies: set) -> bool:
        """Close the open run; False when ``dependencies`` need its interior.

        Closing promotes every non-export value of the open run to an
        interior value, so a node that closes the run may itself reference
        them only via the export.
        """

        close()
        return not (dependencies & interior_values)

    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        dependencies = set(_flatten_values(node.args)) | set(
            _flatten_values(node.kwargs)
        )
        if dependencies & interior_values:
            return None
        reduction = classify_reduction(node)
        if reduction is not None:
            if current_reduction is not None:
                # reduction feeding a reduction: kernel boundary between them
                if not close_and_guard(dependencies):
                    return None
                current = [node]
                current_reduction = reduction
                continue
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
                if not close_and_guard(dependencies):
                    return None
            current.append(node)
            continue
        # extern operator: close the open run, then record the operator as
        # its own single-node segment so the backend can interleave an eager
        # call between fused kernels.  The operator must not read the open
        # run's interior either — after the boundary only the export value
        # crosses into later segments.
        open_export = (
            current_epilogue[-1]
            if current_epilogue
            else (current[-1] if current else None)
        )
        open_interior = {
            value
            for value in [*current, *current_epilogue]
            if value is not open_export
        }
        if dependencies & open_interior:
            return None
        close()
        segments.append(Segment(nodes=(node,), kind="extern"))
        continue

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
            "reduction": getattr(segment.reduction, "__dict__", segment.reduction),
        }
        for segment in segments
    ]
