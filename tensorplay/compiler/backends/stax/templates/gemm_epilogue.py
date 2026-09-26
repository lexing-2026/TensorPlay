"""What a region's operations are, named, and what a product could do with them.

A region that a product's tile might carry is a short sequence of operations on
its own axes, and the question is not whether each operation is legal -- they all
are -- but whether the *pair* of them means one thing.  A region that reads
another's output and divides it by a total is a normalisation; a region that
reads two values and adds them is not.  Nothing in a single operation says which,
and the difference is the whole of what a product's store may be asked to do.

So the operations are *normalised* here: each one becomes a value saying what it
did, in terms that do not depend on how it was written.  A reshape becomes a view
of a shape, a cast becomes a view of a type, a reduction becomes a reduction of a
named kind over a named axis, and everything the product cannot be told by becomes
a node that says so rather than a node that is quietly dropped.  A dropped
operation is a product that writes something nobody asked for, and the only way to
notice is for the operation to still be there to be looked at.

The reduction objects beside them say the same about *where* a reduction runs: a
group and an axis are the two things a product's tile has to agree with before a
reduction can be finished inside it, and every question about whether it can is a
question about those two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence, Union

from ..ir import (
    ArgReduction,
    MultiOutputReduction,
    OnlineSoftmaxReduction,
    Pointwise,
    Reduction,
    TensorBox,
    WelfordReduction,
)

__all__ = [
    "FUNCTION_REDUCTION_TYPES",
    "FUNCTION_UNSUPPORTED_REDUCTIONS",
    "GemmEpilogueGraph",
    "GemmReductionArguments",
    "GemmReductionConfig",
    "GemmReductionDescriptor",
    "GemmReductionGeometry",
    "GemmReductionPlan",
    "NormalizedDtypeView",
    "NormalizedGetItem",
    "NormalizedNode",
    "NormalizedPrepareSoftmax",
    "NormalizedReduction",
    "NormalizedSelect",
    "NormalizedSplit",
    "NormalizedSqueeze",
    "NormalizedToBlocked",
    "NormalizedUnsupportedReduction",
    "NormalizedView",
    "iter_fx_node_inputs",
    "normalize_gemm_epilogue_fx_node",
]

#: The walks this names.  A walk whose name is not here is reported as one
#: rather than guessed at: a name the product does not know is a walk it will
#: finish as a plain reduction, which is always right and sometimes slower.
FUNCTION_REDUCTION_TYPES = (
    "sum", "prod", "max", "amin", "amax", "any", "all", "argmax", "argmin",
    "online_softmax_reduce", "welford_reduce", "welford_combine",
    "arg_reduce",
)

#: The walks whose value is not a reduction of the values at all, and which a
#: product's tile therefore cannot be asked to finish.
FUNCTION_UNSUPPORTED_REDUCTIONS = ("scan", "sort", "kthvalue", "median")


# ---------------------------------------------------------------- normalised
# Each of these is one operation, said in terms that do not depend on how it was
# written.  They are frozen because a normalised node is a reading of something
# that has already happened: there is nothing to change about it afterwards.


@dataclass(frozen=True)
class NormalizedView:
    """A region that read another value and gave it a different shape."""

    source: Any
    shape: tuple


@dataclass(frozen=True)
class NormalizedDtypeView:
    """A region that read another value and gave it a different type."""

    source: Any
    dtype: Any


@dataclass(frozen=True)
class NormalizedReduction:
    """A region that reduced another value over one axis.

    ``keepdim`` is kept because it changes the shape that comes out, and a
    product that is told the reduction happened but not whether the axis was kept
    would be told the wrong output shape.
    """

    source: Any
    dim: int
    keepdim: bool
    dtype: Any
    reduction_type: str


@dataclass(frozen=True)
class NormalizedPrepareSoftmax:
    """A region that reduced over an axis *in order to subtract it later*.

    This is the largest value over an axis, kept rather than consumed: a product
    that is asked for the total of the exponentials needs the largest one too, and
    the two are not separable after the fact.
    """

    source: Any
    dim: int


@dataclass(frozen=True)
class NormalizedSqueeze:
    """A region that dropped an extent of one."""

    source: Any


@dataclass(frozen=True)
class NormalizedGetItem:
    """A region that took one element of one value."""

    source: Any
    index: Any


@dataclass(frozen=True)
class NormalizedSplit:
    """A region that cut one value into pieces of a fixed size."""

    source: Any
    split_size: int
    dim: int


@dataclass(frozen=True)
class NormalizedSelect:
    """A region that kept one index of one axis."""

    source: Any
    dim: int
    index: int


@dataclass(frozen=True)
class NormalizedToBlocked:
    """A region that laid a factor out to the shape of the tile it multiplies.

    This is the operation that makes a block-shaped factor possible at all, and it
    is the reason a product must be told the factor's shape rather than inferring
    it: the expansion is what a tile rides on.
    """

    source: Any


@dataclass(frozen=True)
class NormalizedUnsupportedReduction:
    """A region that reduced in a way no product's tile can finish.

    Kept as a node rather than dropped, so that a product asked to carry it says
    why it will not rather than writing something nobody asked for.
    """

    source: Any
    target: str


#: Every normalised kind, as one name, so that a reader can say what may appear.
NormalizedNode = Union[
    NormalizedView, NormalizedDtypeView, NormalizedReduction,
    NormalizedPrepareSoftmax, NormalizedSqueeze, NormalizedGetItem,
    NormalizedSplit, NormalizedSelect, NormalizedToBlocked,
    NormalizedUnsupportedReduction,
]


def iter_fx_node_inputs(value: Any) -> Iterator:
    """The values a node was made from, in the order it was given them.

    A node's operands are its *inputs*, not its arguments: an argument that was a
    constant is not something the value depends on, and treating it as one would
    make a region that multiplies by two depend on a two.
    """

    if isinstance(value, TensorBox):
        seen = 0
        while isinstance(value, TensorBox) and seen < 8:
            value = value.data
            seen += 1
    node = getattr(value, "graph_node", None) or value
    args = getattr(node, "args", None)
    if not args:
        return iter(())
    return iter(args)


def normalize_gemm_epilogue_fx_node(node: Any) -> Optional[NormalizedNode]:
    """What one traced operation is, in terms that do not depend on how it was written.

    A node this does not recognise is returned as itself rather than dropped: a
    dropped operation is a product writing something nobody asked for, and the
    only way to notice is for the operation to still be there.
    """

    if isinstance(node, (NormalizedView, NormalizedDtypeView, NormalizedReduction,
                         NormalizedPrepareSoftmax, NormalizedSqueeze,
                         NormalizedGetItem, NormalizedSplit, NormalizedSelect,
                         NormalizedToBlocked, NormalizedUnsupportedReduction)):
        return node
    body = node
    seen = 0
    while isinstance(body, TensorBox) and seen < 8:
        body = body.data
        seen += 1
    if isinstance(body, OnlineSoftmaxReduction):
        return NormalizedPrepareSoftmax(
            source=body, dim=_reduced_dim(body)
        )
    if isinstance(body, Reduction):
        kind = str(getattr(body, "reduction_type", None) or "")
        if kind in FUNCTION_UNSUPPORTED_REDUCTIONS:
            return NormalizedUnsupportedReduction(source=body, target=kind)
        return NormalizedReduction(
            source=body,
            dim=_reduced_dim(body),
            keepdim=False,
            dtype=getattr(body, "dtype", None),
            reduction_type=kind,
        )
    if isinstance(body, Pointwise):
        return NormalizedView(source=body, shape=tuple(body.ranges))
    return None


def _reduced_dim(body: Any) -> int:
    """Which axis a reduction ran over, or ``-1`` when it does not say."""

    ranges = getattr(body, "reduction_ranges", None)
    if not ranges:
        return -1
    kept = len(getattr(body, "ranges", ()) or ())
    return kept


@dataclass
class GemmEpilogueGraph:
    """A region's operations, normalised, and what each depends on.

    Both halves are kept because they answer different questions: what a region
    *does* is read off the normalised nodes, and what may be carried at all is
    read off the dependencies -- a region that reads something nobody named is a
    region whose reads have to be reproduced wherever it is carried to.
    """

    dependencies: dict = field(default_factory=dict)
    normalized_nodes: dict = field(default_factory=dict)

    @classmethod
    def from_nodes(cls, nodes: Sequence[Any]) -> "GemmEpilogueGraph":
        """A graph of these nodes, and of what each was made from."""

        graph = cls()
        by_node = {}
        for node in nodes:
            normalized = normalize_gemm_epilogue_fx_node(node)
            if normalized is not None:
                graph.normalized_nodes[id(node)] = normalized
            by_node[id(node)] = node
        for node in nodes:
            inputs = []
            for value in iter_fx_node_inputs(node):
                key = id(value)
                if key in by_node:
                    inputs.append(by_node[key])
            graph.dependencies[id(node)] = tuple(inputs)
        return graph

    def depends_on(self, node: Any, target: Any) -> bool:
        """Whether a value is reachable from a node through what it was made from.

        Asked over the whole chain rather than one step, because what a product
        needs to know is whether carrying this carries that, and one step of that
        question is not an answer to it.
        """

        seen: set = set()
        stack = [id(node)]
        target_id = id(target)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current == target_id:
                return True
            stack.extend(id(value) for value in self.dependencies.get(current, ()))
        return False


# ---------------------------------------------------------------- reduction
# A reduction the product's tile might finish says two things: which group it runs
# over, and which axis of that group.  Everything else about it -- how wide, what
# kind, what it feeds -- is checked against the tile afterwards.


@dataclass(frozen=True)
class GemmReductionGeometry:
    """Which group, and which axis of it, a reduction runs over.

    A *group* is the number of groups a value is read in, and an *axis* is which
    of the value's two axes the walk is along.  A reduction with no group is one
    over a whole value, which is the case a tile owns outright.
    """

    group: int = 1
    axis: int = 0

    def __post_init__(self):
        if int(self.group) < 1:
            raise ValueError("a reduction is over at least one group")

    @property
    def needs_physical_callbacks(self) -> bool:
        """Whether finishing this reduction means calling out to the machine.

        A reduction whose value is finished inside the tile is arithmetic.  One
        whose value is a partial -- because the tile holds part of the axis -- has
        to be combined, and combining is a second piece of work rather than part
        of this one.
        """

        return False

    @property
    def group_size(self) -> int:
        """How many groups a reduction over a whole value stands for."""

        return max(int(self.group), 1)

    @classmethod
    def from_output_shape(cls, shape: Sequence[int], group: int = 1) \
            -> "GemmReductionGeometry":
        """The geometry a reduction whose result is this shape would have."""

        return cls(group=int(group), axis=-2 if len(tuple(shape)) >= 2 else 0)

    def reduce_dims(self, shape: Sequence[int]) -> tuple:
        """The axes of a value of this shape that this reduction would reduce."""

        shape = tuple(int(v) for v in shape)
        if len(shape) < 2:
            return (0,)
        return (1 - self.axis,) if self.axis in (0, 1) else (1,)

    def matches_reduction_dim(self, dim: int) -> bool:
        """Whether a reduction over this axis is the one this geometry describes."""

        return int(dim) in self.reduce_dims((2, 2))

    def matches_output_shape(self, shape: Sequence[int]) -> bool:
        """Whether a result of this shape is one value per row or per column.

        A reduction along the columns leaves one value per row, and a reduction
        along the rows leaves one value per column -- so the *result's* rank is
        how a product knows which way round the tile was walked, without being
        told twice.
        """

        shape = tuple(int(v) for v in shape)
        if len(shape) == 1:
            return True
        return len(shape) == 2

    def __repr__(self) -> str:
        return "GemmReductionGeometry(group=%d, axis=%d)" % (self.group, self.axis)


@dataclass(frozen=True)
class GemmReductionDescriptor:
    """A reduction's kind and its parameters, in a form that can be written down.

    A reduction is chosen by a measurement, and a measurement is persisted; so
    what was chosen has to be readable back, and a description that is only
    comparable in memory cannot be.  The parameters are therefore a mapping whose
    values are all numbers.
    """

    kind: str
    parameters: tuple = ()

    @classmethod
    def parse(cls, text: str) -> "GemmReductionDescriptor":
        """A descriptor read back from its written form."""

        kind, _, rest = str(text).partition(":")
        parameters = []
        if rest:
            for part in rest.split(","):
                if part.strip():
                    parameters.append(part.strip())
        return cls(kind=kind.strip(), parameters=tuple(parameters))

    def serialize(self) -> str:
        """A descriptor's written form, which round-trips through ``parse``."""

        if not self.parameters:
            return self.kind
        return "%s:%s" % (self.kind, ",".join(str(p) for p in self.parameters))

    def __repr__(self) -> str:
        return "GemmReductionDescriptor(%r)" % (self.serialize(),)


@dataclass(frozen=True)
class GemmReductionConfig:
    """One configuration for a reduction: what it reduces, and what feeds it."""

    output_name: str
    group: int = 1
    axis: int = 0
    reduction_type: str = "sum"
    source_type: str = "float32"

    @property
    def geometry(self) -> GemmReductionGeometry:
        """Where this reduction runs, as the one thing both the tile and it know."""

        return GemmReductionGeometry(group=self.group, axis=self.axis)

    @property
    def contract(self) -> GemmReductionDescriptor:
        """This reduction as something that can be written down and read back."""

        return GemmReductionDescriptor(
            kind=self.reduction_type,
            parameters=(self.group, self.axis, self.source_type),
        )


@dataclass(frozen=True)
class GemmReductionPlan:
    """A reduction, and everything that has to be true of the tile to finish it."""

    reduction_output: str
    group: int = 1
    axis: int = 0
    reduction_type: str = "sum"
    source_type: str = "float32"
    primary_output: str = ""
    feeds_main: bool = False
    feed_output: str = ""
    secondary_feed_output: str = ""
    secondary_feed_type: str = ""

    @property
    def geometry(self) -> GemmReductionGeometry:
        """Where this reduction runs."""

        return GemmReductionGeometry(group=self.group, axis=self.axis)

    @property
    def auxiliary_outputs(self) -> tuple:
        """The outputs a reduction carries besides the one it is named for.

        A walk that carries the largest value with its running total, or a mean
        with its spread, produces more than one value; which of them is *the*
        result is a choice, and the rest have to be written somewhere.
        """

        out = []
        if self.feed_output:
            out.append(self.feed_output)
        if self.secondary_feed_output:
            out.append(self.secondary_feed_output)
        return tuple(out)


@dataclass(frozen=True)
class GemmReductionArguments:
    """A reduction as it is handed to a kernel: which values, and which is which."""

    output: Any
    group: int = 1
    axis: int = 0
    reduction_type: str = "sum"
    source_type: str = "float32"
    feeds_main: bool = False
    feed_output: Any = None
    secondary_feed_output: Any = None
    secondary_feed_type: str = ""

    #: The fields that are part of what this is identified by.  A measurement is
    #: keyed by these and by nothing else, so two calls that differ only in the
    #: rest are one measurement rather than two.
    SPECIALIZATION_FIELDS = ("group", "axis", "reduction_type", "source_type")

    @property
    def enabled(self) -> bool:
        """Whether there is a reduction to finish at all."""

        return self.output is not None

    @property
    def primary_enabled(self) -> bool:
        """Whether the reduction's own value is one this is being asked for.

        A reduction can be carried for what it *feeds* -- the largest value a
        normalisation subtracts -- without its own value being wanted, and the
        two are separate: one of them is an output, the other is not.
        """

        return self.enabled

    @property
    def descriptor(self) -> GemmReductionDescriptor:
        """This as something that can be written down and read back."""

        return GemmReductionDescriptor(
            kind=self.reduction_type,
            parameters=(self.group, self.axis, self.source_type),
        )

    @property
    def tensors(self) -> tuple:
        """The values a kernel is handed for this reduction, in order."""

        out = [self.output]
        for value in (self.feed_output, self.secondary_feed_output):
            if value is not None:
                out.append(value)
        return tuple(out)


# ---------------------------------------------------------------- this tree's
# The classes above say what a reduction is.  What a *product* wants to know is
# narrower: whether the tile in its hands is the whole of what the region reads,
# and what the region produces.  That is the plan, and it is the same decision
# with the question asked the other way round.


class GemmEpilogueRejected(Exception):
    """A region that cannot be carried by a tile, and the axis that stopped it.

    Raised rather than returned because a caller already holding a fusible plan
    has no way to act on a second one: the only correct response is to compute
    the region on its own, which is a different code path rather than a variant
    of this one.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class GemmEpilogueGeometry:
    """The tile a plan is planned against, and the result it covers.

    Both are here: a plan that recorded only the tile would look valid for a
    product it cannot cover, and one that recorded only the result would be right
    about a product no tile is holding.
    """

    tile: tuple = ()
    result: tuple = ()

    @property
    def is_whole_result(self) -> bool:
        """Whether the tile is the entire result, so nothing is left over."""

        return tuple(self.tile) == tuple(self.result)

    def tiles(self) -> int:
        """How many tiles cover the result, which is how often the plan runs."""

        rows = -(-int(self.result[0]) // int(self.tile[0]))
        cols = -(-int(self.result[1]) // int(self.tile[1]))
        return rows * cols


@dataclass(frozen=True)
class GemmEpiloguePlan:
    """One region's ride on one product's tile."""

    geometry: GemmEpilogueGeometry
    kind: str
    local_reduce: Any = None
    family: str = "pointwise"
    captured: Sequence = ()

    @property
    def _match(self):
        return None if self.local_reduce is None else self.local_reduce.match

    @property
    def partials(self) -> int:
        """How many tiles' worth of partial this reduction produces, if not one.

        One means the tile finishes the reduction itself, which is the case that
        needs no second pass; more than one means the answer is the combination
        of what each tile produced.
        """

        match = self._match
        return 1 if match is None else match.partials

    @property
    def needs_combining(self) -> bool:
        """Whether the reduction's value has to be combined after the walk."""

        return self.partials > 1

    @property
    def result_shape(self) -> tuple:
        """The shape of what the region produces, which the store must match."""

        if self._match is None:
            return tuple(self.geometry.result)
        return ((int(self.geometry.result[0]),) if self._match.reduces_rows
                else (int(self.geometry.result[1]),))

    def describe(self) -> str:
        """One line saying what this plan does, for a measurement's label."""

        if self._match is None:
            return "elementwise on the tile"
        text = "%s over the %s of the tile, %d wide" % (
            self.family, "rows" if self._match.reduces_rows else "columns",
            self._match.reduce_extent,
        )
        if self.needs_combining:
            text += ", %d partials to combine" % self.partials
        return text


def plan_epilogue(value: Any, tile_shape: Sequence[int],
                  result_shape: Optional[Sequence[int]] = None) -> GemmEpiloguePlan:
    """The plan for carrying a region on a tile, or the reason there is none.

    ``result_shape`` defaults to the tile, which is the case of a product small
    enough to be one tile -- and the only case where a plan built against the
    tile alone is automatically right about the whole result.
    """

    from .gemm_epilogue_analysis import analyse_epilogue, reduction_family

    tile = tuple(int(v) for v in tile_shape)
    result = tuple(int(v) for v in result_shape) if result_shape else tile
    if len(result) != 2:
        raise GemmEpilogueRejected("a result that is not two-dimensional")
    analysis = analyse_epilogue(value, tile, result)
    if not analysis.fusible:
        raise GemmEpilogueRejected(analysis.blocked_by or "not fusible")
    plan = GemmEpiloguePlan(
        geometry=GemmEpilogueGeometry(tile=tile, result=result),
        kind=analysis.kind,
        local_reduce=analysis.local_reduce,
        family=reduction_family(value),
        captured=analysis.captured,
    )
    if plan.result_shape != plan.geometry.result and plan.local_reduce is None:
        raise GemmEpilogueRejected("a plan whose result is not the tile's shape")
    return plan


def construct_strides(sizes, fill_order):
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    if len(sizes) != len(fill_order):
        raise AssertionError('Length of sizes must match the length of the fill order')
    strides: list[_IntLike] = [0] * len(sizes)
    current_stride: _IntLike = 1
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]
    return strides
