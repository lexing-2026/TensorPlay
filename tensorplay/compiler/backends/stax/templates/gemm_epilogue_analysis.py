"""Whether a region that consumes a product's result can ride that product's tile.

A product computes a tile of its result and writes it out.  A region that
consumes that result can sometimes be computed from the tile while the tile is
still in registers -- the write and the work that would read it never both
happen -- and whether it can is not a matter of taste.  It is a question about
which axes the region reads and which axes the tile holds:

  the region reads only the tile's own two axes, element for element, then it
  is a chain of elementwise work over the tile and costs nothing extra to carry

  the region reduces along one of those two axes, and keeps only the other, then
  the whole of what it reduces is inside the tile -- a row of the tile is a whole
  row of the result -- so the reduction is local to the tile and can be finished
  before anything is written

  the region reduces along an axis the tile does not hold, or walks an axis the
  result does not have, then the tile is a slice of something the region needs
  in full, and no amount of carrying it through the multiply makes that local

The *grouped* case is the same question with a group count attached, and it has
one more way to go wrong: a region that reads its value in groups is making a
claim about the memory, not about the values, so the claim is checked against the
extents rather than believed.  A group count that does not divide the axis it is
supposed to cut is not a layout anybody may plan on.

A region this cannot place is a region somebody else has to compute.  It is never
a region this will half-fuse: a plan built on a tile that does not hold what the
region reads is not a slower plan, it is a wrong one, and it would be wrong in a
way that only shows up on shapes nobody tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..ir import (
    ArgReduction,
    Loops,
    MultiOutputReduction,
    OnlineSoftmaxReduction,
    Pointwise,
    Reduction,
    TensorBox,
    WelfordReduction,
)
from .gemm_epilogue import (
    GemmEpilogueGraph,
    GemmReductionGeometry,
    GemmReductionPlan,
    NormalizedNode,
    normalize_gemm_epilogue_fx_node,
)

__all__ = [
    "EpilogueFusibility",
    "FEED_MAIN_BINARY_FUNCTIONS",
    "GemmLocalReduceAnalysis",
    "GemmLocalReduceMatch",
    "GemmLocalReduceStore",
    "GemmOutputLocalReducePlan",
    "GemmOutputPlan",
    "grouped_tensor_layout",
]

#: The operations whose *result* is a value another operation may read, and so
#: may be what carries the main output of a grouped product.  Everything else a
#: region does is either the value itself or a shape of it.
FEED_MAIN_BINARY_FUNCTIONS = frozenset({
    "add", "sub", "mul", "truediv", "maximum", "minimum", "pow",
    "exp", "logarithm", "sqrt", "abs",
})


def _unwrap(value: Any) -> Any:
    """The body behind a value, whether it arrives bare or boxed.

    A value reaches this module from a graph walk, where it is boxed, or from a
    test, where it is not, and the two are the same region.  Asking which of them
    is the exception is a worse question than just looking through the box.
    """

    seen = 0
    while isinstance(value, TensorBox) and seen < 8:
        value = value.data
        seen += 1
    return value


def _extents(value: Any) -> Optional[Sequence]:
    """A body's own axes, or ``None`` when it has none to report."""

    if value is None:
        return None
    for name in ("get_size", "get_pointwise_size"):
        getter = getattr(value, name, None)
        if callable(getter):
            try:
                return tuple(int(v) for v in getter())
            except Exception:  # noqa: BLE001 - a body that will not say has no shape
                return None
    return None


def _reduced(value: Any) -> Optional[Sequence]:
    """The axes a body reduces over, or ``None`` when it reduces over none."""

    if not isinstance(value, Reduction):
        return None
    try:
        return tuple(int(v) for v in value.reduction_ranges)
    except Exception:  # noqa: BLE001 - a reduction that will not say is not fusible
        return None


# ---------------------------------------------------------------- grouped
# A region that reads its value in groups is saying the value is one matrix with
# the group as an axis.  That is a claim about the memory, and each of these
# checks one part of it, because a claim that is wrong produces numbers that are
# wrong in a way that only shows up at one shape.


def _is_inferred_reshape_dim(value: Any) -> bool:
    """Whether an extent was worked out rather than read.

    An extent nobody wrote down is one this cannot check anything against, so a
    layout that depends on one is not a layout to plan on.
    """

    if value is None:
        return True
    try:
        return not int(value).free_symbols == set()
    except AttributeError:
        try:
            int(value)
            return False
        except Exception:  # noqa: BLE001 - an extent that is neither is inferred
            return True
    except Exception:  # noqa: BLE001
        return True


def _kept_dim_matches_source(kept_size: int, source_size: int) -> bool:
    """Whether the axis a region kept is the axis it read.

    A region that kept an extent the source did not have is not reading that
    source, whatever else it says it is doing.
    """

    return int(kept_size) == int(source_size)


def _group_count_matches_selected_dim(group_count: int, selected_size: int,
                                      group: int, kept_size: int) -> bool:
    """Whether the group count is a whole number of groups of the kept axis.

    The groups are read as a single axis, so the kept axis has to divide by the
    group count -- and the per-group extent is what the tile is fitted to, so it
    has to be the same on both sides of that division.
    """

    if int(group_count) < 1:
        return False
    if int(selected_size) % int(group_count):
        return False
    return (int(selected_size) // int(group_count)) == int(kept_size)


def _grouped_layout_matches_source_shape(shape: Sequence[int],
                                         source_shape: Sequence[int],
                                         layout) -> bool:
    """Whether a grouped read agrees with the shape it claims to read."""

    shape = tuple(int(v) for v in shape)
    source_shape = tuple(int(v) for v in source_shape)
    if len(shape) < 2 or len(source_shape) < 2:
        return False
    if not _kept_dim_matches_source(shape[-2], source_shape[-2]):
        return False
    if not _kept_dim_matches_source(shape[-1] if len(shape) > 1 else 1,
                                    source_shape[-1] if len(source_shape) > 1 else 1):
        return False
    return True


def _guard_grouped_reshape_group(shape: Sequence[int],
                                 source_shape: Sequence[int]) -> bool:
    """Whether a reshape's group is a whole number of the axis it cuts.

    A reshape that cuts an axis into groups of a number that does not divide it
    is not a reshape; it is a claim about a memory layout that does not hold, and
    the only way it produces numbers is by producing numbers for a different
    tensor.
    """

    shape = tuple(int(v) for v in shape)
    if len(shape) < 1:
        return False
    leading = shape[0]
    if any(_is_inferred_reshape_dim(v) for v in shape):
        return False
    return leading >= 1


def _syntactic_grouped_tensor_layout(shape: Sequence[int]) -> Optional[tuple]:
    """What a shape says about how it is read in groups, or ``None``.

    Read off the shape alone, which is the only thing available before anything
    has been traced: a leading axis that could be a group count, and the axis it
    would be cutting.
    """

    shape = tuple(int(v) for v in shape)
    if len(shape) < 2:
        return None
    if any(_is_inferred_reshape_dim(v) for v in shape):
        return None
    return (shape[0], shape[1:])


def _as_shape(value: Any) -> Optional[Sequence[int]]:
    """A shape, whether it was given as one or as the value that has it."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value
    extents = _extents(_unwrap(value))
    return None if extents is None else tuple(extents)


def grouped_tensor_layout(shape: Any,
                          source_shape: Optional[Sequence[int]] = None) \
        -> Optional[tuple]:
    """The layout a grouped region reads its value through, or ``None``.

    A region that walks its input in groups reads it as one matrix with the group
    as an axis rather than as a batch, and a reshape that says so is a claim
    about the memory rather than about the values.  So the claim is checked here
    against the extents themselves: a group count that does not divide the axis
    it is supposed to cut, or a kept axis whose extent disagrees with the axis it
    came from, is not a layout anybody may plan on.
    """

    extents = _as_shape(shape)
    if extents is None:
        return None
    layout = _syntactic_grouped_tensor_layout(extents)
    if layout is None:
        return None
    group, kept = layout
    if group < 1:
        return None
    source = _as_shape(source_shape)
    if source is not None and not _grouped_layout_matches_source_shape(
        tuple(kept) + (group,), tuple(source), None
    ):
        return None
    return (group, tuple(kept))


# ---------------------------------------------------------------- local reduce


@dataclass
class GemmLocalReduceMatch:
    """A value a reduction can be finished from, and where the reduction runs.

    A *match* is the pairing of a value with the geometry a reduction over it
    would have.  A value with no geometry cannot be reduced inside a tile,
    because nothing says which way round the tile was walked.
    """

    value_node: Any = field(repr=False, default=None)
    geometry: GemmReductionGeometry = field(default_factory=GemmReductionGeometry)
    reduction_node: Any = field(repr=False, default=None)
    reduction_type: Optional[str] = None
    tile_along_reduce: int = 0
    reduce_extent: int = 0
    keep_extent: int = 0
    multi_output: bool = False

    def __post_init__(self):
        if not isinstance(self.geometry, GemmReductionGeometry):
            raise TypeError("a match needs a geometry to say where it runs")

    @property
    def axis(self) -> str:
        """Which way round the tile the walk runs: by its rows or its columns."""

        return "row" if int(self.geometry.axis) == 0 else "column"

    @property
    def reduces_rows(self) -> bool:
        """Whether the walk runs along the rows of the tile."""

        return self.axis == "row"

    @property
    def spans(self) -> bool:
        """Whether the tile holds the whole of what is being reduced.

        A tile that does not is a tile of a longer walk: the value it produces
        is one addend of the answer rather than the answer, and the addends have
        to be combined, which is a second piece of work and not this one.
        """

        return int(self.tile_along_reduce) >= int(self.reduce_extent)

    @property
    def partials(self) -> int:
        """How many tiles' worth of partial this walk produces, if not one."""

        if self.spans:
            return 1
        return -(-int(self.reduce_extent) // max(int(self.tile_along_reduce), 1))

    @property
    def needs_physical_callbacks(self) -> bool:
        """Whether finishing this reduction means calling out to the machine."""

        return self.geometry.needs_physical_callbacks

    def to_plan(self) -> GemmReductionPlan:
        """This match as the plan a kernel would be asked to follow."""

        return GemmReductionPlan(
            reduction_output=str(self.reduction_type or "sum"),
            group=self.geometry.group,
            axis=self.geometry.axis,
            reduction_type=str(self.reduction_type or "sum"),
        )

    @classmethod
    def common(cls, matches: Sequence["GemmLocalReduceMatch"],
               mixed_match_error: Optional[str] = None
               ) -> Optional["GemmLocalReduceMatch"]:
        """The one match several values agree on, or ``None``.

        Two values reduced together have to be reduced the same way round, and
        when they are not there is no single answer to give -- so this says so
        rather than picking one of them.
        """

        found = [m for m in matches if m is not None]
        if not found:
            return None
        keys = {(m.geometry.group, m.geometry.axis, m.reduction_type)
                for m in found}
        if len(keys) > 1:
            if mixed_match_error:
                raise AssertionError(mixed_match_error)
            return None
        return found[0]

    @classmethod
    def common_value(cls, matches: Sequence["GemmLocalReduceMatch"],
                     mixed_match_error: Optional[str] = None) -> Optional[tuple]:
        """The values several matches agree on, in one order."""

        found = [m for m in matches if m is not None]
        if not found:
            return None
        common = cls.common(found, mixed_match_error)
        if common is None:
            return None
        return (common.geometry, common.reduction_type)

    def __repr__(self) -> str:
        return "GemmLocalReduceMatch(%r over %r%s)" % (
            self.reduction_type, self.geometry,
            "" if self.reduction_node is None else ", reduced",
        )


@dataclass
class GemmLocalReduceStore:
    """Where a reduction's value was written, and which of the outputs it was.

    A reduction that carries more than one value writes more than one output,
    and which of them is the result is what this says.
    """

    node: Any = field(repr=False, default=None)
    aux_index: int = 0

    def __post_init__(self):
        if int(self.aux_index) < 0:
            raise ValueError("an auxiliary output is one past the first at worst")

    @property
    def is_primary(self) -> bool:
        """Whether this store is the reduction's own value rather than a carried one."""

        return int(self.aux_index) == 0


@dataclass
class GemmOutputLocalReducePlan:
    """A reduction's match, where its value went, and what it feeds."""

    match: GemmLocalReduceMatch = field(default_factory=GemmLocalReduceMatch)
    store: Optional[GemmLocalReduceStore] = None
    feeds_main: bool = False

    def __post_init__(self):
        if not isinstance(self.match, GemmLocalReduceMatch):
            raise TypeError("a plan needs the match it was planned from")

    def needs_physical_callbacks(self) -> bool:
        """Whether finishing this reduction means calling out to the machine."""

        return self.match.needs_physical_callbacks


@dataclass
class GemmOutputPlan:
    """One output of a region, and the reduction behind it if there is one."""

    output: Any = field(repr=False, default=None)
    aux_outputs: tuple = ()
    local_reduce: Optional[GemmOutputLocalReducePlan] = None

    def __post_init__(self):
        if self.local_reduce is not None and not isinstance(
            self.local_reduce, GemmOutputLocalReducePlan
        ):
            raise TypeError("a plan needs a reduction plan or none at all")

    @property
    def reduction_plan(self) -> Optional[GemmReductionPlan]:
        """The reduction this output is, or ``None`` when there is not one."""

        return None if self.local_reduce is None else self.local_reduce.match.to_plan()


class GemmLocalReduceAnalysis:
    """Which of a region's values a tile could finish a reduction from.

    The analysis walks the region's operations and asks, of each value, whether
    the tile holds the whole of what a reduction over it would need.  A value it
    can place gets a match; a value it cannot is left alone, and a region whose
    main output has no match is a region that has to be computed on its own.
    """

    def __init__(self, graph: Optional[GemmEpilogueGraph] = None,
                 grouped_tensors: Optional[dict] = None,
                 matches: Optional[dict] = None):
        self.graph = graph if graph is not None else GemmEpilogueGraph()
        self.grouped_tensors: dict = dict(grouped_tensors or {})
        self.matches: dict = dict(matches or {})

    # -- construction ----------------------------------------------------

    @classmethod
    def from_epilogue(cls, value: Any, tile_shape: Sequence[int],
                      result_shape: Optional[Sequence[int]] = None) \
            -> "GemmLocalReduceAnalysis":
        """The analysis of one region against a tile of a result.

        This is the whole of the analysis for a single value, which is the case a
        product asks about: a product's tile is one value, and the question is
        whether that value's region can be carried by it.
        """

        analysis = cls()
        analysis.bind_region(value, tile_shape, result_shape)
        return analysis

    def bind_region(self, value: Any, tile_shape: Sequence[int],
                    result_shape: Optional[Sequence[int]] = None) -> None:
        """Place one region, recording what was found and what was not."""

        tile = tuple(int(v) for v in tile_shape)
        result = tuple(int(v) for v in result_shape) if result_shape else tile
        self.graph = GemmEpilogueGraph.from_nodes([value])
        body = _unwrap(value)
        extents = _extents(body)
        reduced = _reduced(body)
        if extents is None or not isinstance(body, Loops):
            return
        if reduced is None:
            if tuple(extents) == result:
                self.matches[id(value)] = GemmLocalReduceMatch(
                    value_node=value,
                    geometry=GemmReductionGeometry(group=1, axis=0),
                    reduction_node=None,
                    reduction_type=None,
                )
            return
        if len(extents) != 1 or len(reduced) != 1:
            return
        keep, reduce_extent = int(extents[0]), int(reduced[0])
        kept_axis = _place_axis(keep, reduce_extent, result)
        if kept_axis is None:
            return
        along = 1 - kept_axis
        if reduce_extent != result[along]:
            return
        self.matches[id(value)] = GemmLocalReduceMatch(
            value_node=value,
            geometry=GemmReductionGeometry(
                group=int(result[0] // max(int(tile[0]), 1)) or 1, axis=kept_axis
            ),
            reduction_node=value,
            reduction_type=str(getattr(body, "reduction_type", None) or "sum"),
            tile_along_reduce=tile[along],
            reduce_extent=reduce_extent,
            keep_extent=keep,
            multi_output=isinstance(body, MultiOutputReduction),
        )

    # -- the questions it answers ---------------------------------------

    def match_for(self, value: Any) -> Optional[GemmLocalReduceMatch]:
        """The match for a value, or ``None`` when the tile does not hold it."""

        return self.matches.get(id(value))

    def output_plan(self, value: Any) -> Optional[GemmOutputPlan]:
        """The plan for a value's output, or ``None`` when it cannot be carried."""

        match = self.match_for(value)
        if match is None:
            return None
        if match.reduction_node is None:
            # A value the tile holds with nothing reduced off it is carried as
            # it is: there is no reduction here for a plan to be about.
            return GemmOutputPlan(output=value, aux_outputs=(), local_reduce=None)
        return GemmOutputPlan(
            output=value,
            aux_outputs=(),
            local_reduce=GemmOutputLocalReducePlan(
                match=match,
                store=GemmLocalReduceStore(node=value),
                feeds_main=False,
            ),
        )

    def has_physical_grouped_input(self, value: Any) -> bool:
        """Whether a value is read in groups rather than as a batch.

        A grouped read is a claim about the memory, so it is asked about rather
        than assumed from the rank.
        """

        shape = _extents(_unwrap(value))
        if not shape or len(shape) < 2:
            return False
        return grouped_tensor_layout(shape) is not None


def _place_axis(keep: int, reduce_extent: int, result: Sequence[int]) \
        -> Optional[int]:
    """Which of the result's two axes a reduction keeps, or ``None``.

    The kept axis is the one whose extent the region's kept extent matches.  A
    region whose two extents are the same number matches both, and then which one
    it meant is not in anything it said -- so it is refused rather than guessed,
    because the two readings reduce along different axes of the tile and produce
    results of different shapes.
    """

    matches = [axis for axis, extent in enumerate(result)
               if int(extent) == keep and int(result[1 - axis]) == reduce_extent]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------- the answer
# What the product is told.  Kept as its own answer so that "this region cannot
# be carried" and "nobody asked" stay different, and so that the reason is
# available to whoever has to decide what to do about it.


class EpilogueFusibility:
    """What a region would take from a product's tile, or why it takes nothing."""

    def __init__(
        self,
        kind: str,
        tile_shape: Optional[Sequence] = None,
        local_reduce: Optional[GemmOutputLocalReducePlan] = None,
        blocked_by: Optional[str] = None,
        captured: Sequence = (),
    ):
        self.kind = kind
        self.tile_shape = tuple(tile_shape) if tile_shape is not None else None
        self.local_reduce = local_reduce
        self.blocked_by = blocked_by
        self.captured = tuple(captured)

    @property
    def fusible(self) -> bool:
        """Whether the region can be computed from the tile alone."""

        return self.kind in ("elementwise", "local_reduce")

    def __bool__(self) -> bool:
        return self.fusible

    def __repr__(self) -> str:
        if self.fusible:
            return "EpilogueFusibility(%s, tile=%s%s)" % (
                self.kind, self.tile_shape,
                "" if self.local_reduce is None else ", %r" % (self.local_reduce,),
            )
        return "EpilogueFusibility(not %s)" % (self.blocked_by or "fusible",)


def analyse_epilogue(value: Any, tile_shape: Sequence[int],
                     result_shape: Optional[Sequence[int]] = None) \
        -> EpilogueFusibility:
    """What a region takes from a tile, or that it takes nothing.

    ``tile_shape`` is the pair of extents the product holds in registers at the
    moment the region would run, and ``result_shape`` is what the whole product
    produces -- the same thing for a product small enough to be one tile.  Both
    are needed because the question is about the *region's* axes, and where they
    sit is a fact about the result rather than about whichever piece of it a
    tile happens to be holding.
    """

    tile = tuple(int(v) for v in tile_shape)
    if len(tile) != 2:
        return EpilogueFusibility(
            "unsupported", blocked_by="a tile that is not two-dimensional"
        )
    result = tuple(int(v) for v in result_shape) if result_shape else tile
    if len(result) != 2:
        return EpilogueFusibility(
            "unsupported", blocked_by="a result that is not two-dimensional"
        )
    body = _unwrap(value)
    if body is None:
        return EpilogueFusibility("none", blocked_by="no region")
    if not isinstance(body, Loops):
        return EpilogueFusibility(
            "unsupported", blocked_by="a region that is not a set of loops"
        )
    analysis = GemmLocalReduceAnalysis.from_epilogue(value, tile, result)
    plan = analysis.output_plan(value)
    if plan is None:
        extents = _extents(body)
        if extents is None:
            return EpilogueFusibility(
                "unsupported", blocked_by="a region that will not report its axes"
            )
        if len(extents) < 2:
            return EpilogueFusibility(
                "unsupported",
                blocked_by="a region that does not read both axes of the result",
            )
        if len(extents) == 2 and tuple(extents) == tile:
            return EpilogueFusibility(
                "unsupported",
                blocked_by="a region whose axes are the tile's own, which is "
                           "the result rather than a piece of it",
            )
        return EpilogueFusibility(
            "unsupported",
            blocked_by="a region whose axes are not the result's own",
        )
    # A plan with no reduction behind it is the tile carried as it is; one with
    # a reduction behind it is a walk the tile finishes.
    kind = "elementwise" if plan.local_reduce is None else "local_reduce"
    return EpilogueFusibility(
        kind, tile_shape=tile, local_reduce=plan.local_reduce
    )


def reduction_family(value: Any) -> str:
    """Which family of reduction a region is, named rather than matched on.

    The tree already says what a walk is by the kind of node it is, which is a
    stronger statement than a pattern over the operations that produced it: a
    body that carries the largest value with the running total is that kind of
    walk whatever it was written as.
    """

    body = _unwrap(value)
    if isinstance(body, OnlineSoftmaxReduction):
        return "online_softmax"
    if isinstance(body, WelfordReduction):
        return "welford"
    if isinstance(body, ArgReduction):
        return "arg"
    if isinstance(body, Pointwise):
        return "pointwise"
    if isinstance(body, Reduction):
        return str(getattr(body, "reduction_type", None) or "reduction")
    return "unknown"
