"""The shape of a lowered loop body, read back off the operations it performed.

A product's tile wants to finish a reduction inside itself rather than write a
partial result and have it read back, and the only thing that can tell it whether
that is worth doing is what the body *does*: a body that reduces and then divides
by the total it reduced is one reduction with a finalizer, and a body that reduces
and then multiplies by the total is two different things.  The body is already
lowered, so none of that is written anywhere -- it is only visible as the
sequence of operations the body performed, which is what is captured here.

Capture is a matter of replaying the body with an operations handler in place of
the ones that would have written anything, so the body runs exactly as it was
written and the operations it performs become values instead of writes.  Every
expression keeps the name of its operation, its arguments in the order given, and
the loads and walks among them, so a question asked afterwards can be answered
from the expression alone -- which is what makes the answer checkable, since a
question about a product is really a question about a shape of expression.

The walks are kept beside the expression that used them rather than inside it,
because one walk can be read by several operations and a walk written twice is
still one walk.  They are identified by identity, so an expression that mentions
the same walk twice mentions it once, and a body with two walks of the same shape
over different values keeps both.

Every question in here is asked of a *shape* and answered strictly: a shape that is
nearly the one being looked for is not it, because a product told a shape is one
thing and given another writes a number nobody asked for, and there is no way to
notice from the outside.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any, Iterator, Optional

import sympy

import tensorplay as _tp

from tensorplay.graph.experimental.sympy_functions import OrderedSet
from ..ir import ComputedBuffer
from ..loops import V
from ..ops_handler import DefaultHandler
from .gemm_epilogue import GemmReductionConfig

__all__ = [
    "GemmEpilogueIRAnalysis",
    "GemmEpilogueIRExpression",
    "GemmEpilogueIRFinalizer",
    "GemmEpilogueIROutputRole",
    "GemmEpilogueIRRegion",
    "GemmEpilogueIRReduction",
    "GemmEpilogueIRStore",
]


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRExpression:
    """One operation a body performed, and what it was performed on.

    The name is the operation's own; the arguments are its own, in the order it
    was given them, because which of two operands was written first is sometimes
    the only thing that says what the operation means.  The loads and the walks
    are the two things a later question cannot find by walking the arguments: a
    load is named by a buffer rather than by a value, and a walk is a value that
    several operations share.
    """

    op: str
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...] = ()
    loads: frozenset[str] = frozenset()
    reductions: tuple["GemmEpilogueIRReduction", ...] = ()


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRReduction:
    """One walk a body performed, and what it walked.

    The type is the walk's own kind -- a sum, a largest, a running mean -- and the
    source is the value it walked.  Two of them are what a product's tile has to
    be able to hold to finish a walk early, so they are the two things recorded
    here; the rest of a walk's meaning is in the type and the source.
    """

    reduction_type: str
    source: GemmEpilogueIRExpression
    result: int | None = None
    source_type: str | None = None


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRRegion:
    """One output of a body, with the walks that produced it and its expression.

    The algorithm is the walk family's name rather than each walk's own: a body
    that walks in more than one step is walking with a particular algorithm, and
    two walks of the same family are two steps of one walk and not two walks.  A
    body's walks are named this way so that a product can be told which it is
    holding without being told the steps.
    """

    output_name: str
    reductions: tuple[GemmEpilogueIRReduction, ...]
    expression: GemmEpilogueIRExpression

    @property
    def algorithm(self) -> str:
        reduction_types = OrderedSet(
            reduction.reduction_type for reduction in self.reductions
        )
        if "online_softmax_reduce" in reduction_types:
            return "online_softmax"
        if "welford_reduce" in reduction_types:
            return "welford"
        return "generic"


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRStore:
    """One place a body wrote, and what it wrote there.

    The index is symbolic because where a body writes is a function of the
    iteration it is on, and a question about a body's output is usually a question
    about that function rather than about a number.
    """

    index: Any
    value: GemmEpilogueIRExpression


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIROutputRole:
    """Which buffers decide one output, and which of them decide it by walking.

    A buffer decides an output if the output cannot be computed without it, and a
    buffer decides it *by walking* if what it contributes arrives through a walk
    rather than directly.  The two are what tell a product whether it can fold
    this output into a tile, or whether the tile would have to be given a partial
    walk to compute at all.
    """

    transitive_inputs: frozenset[str]
    reduction_inputs: frozenset[str]


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRFinalizer:
    """The one thing a body did to a finished walk.

    The kind is a name rather than a shape so that a body which finished its walk
    and did nothing else is not treated as a body which finished its walk and
    halved it; the two compute different numbers and only one of them is a
    finalizer.
    """

    output_name: str
    source_name: str
    kind: str


def _expression_values(value: Any) -> Iterator[GemmEpilogueIRExpression]:
    """Every expression in a value, at whatever depth it is nested."""

    if isinstance(value, GemmEpilogueIRExpression):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _expression_values(item)


def _unique_reductions(
    values: Sequence[GemmEpilogueIRExpression],
) -> tuple[GemmEpilogueIRReduction, ...]:
    """The walks the given expressions performed, each one once.

    Deduplicated by identity rather than by value: two walks of the same kind over
    the same expression are two walks, and only the same object read twice is one
    walk read twice.
    """

    reductions = []
    seen: OrderedSet[int] = OrderedSet()
    for value in values:
        for reduction in value.reductions:
            if id(reduction) not in seen:
                seen.add(id(reduction))
                reductions.append(reduction)
    return tuple(reductions)


class _GemmEpilogueIRHandler(DefaultHandler):
    """Where a body's operations go while it is being replayed.

    Every operation a handler is not asked about becomes an expression of its own
    name, so a body that does something nobody thought of is still captured and can
    still be looked at.  An operation that is dropped is a body whose meaning is
    gone, and the question asked afterwards would be answered about a program that
    is not the one that ran.
    """

    def __init__(self) -> None:
        self.stores: dict[str, GemmEpilogueIRStore] = {}

    def _default(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> GemmEpilogueIRExpression:
        values = tuple(_expression_values((*args, *tuple(kwargs.values()))))
        return GemmEpilogueIRExpression(
            name,
            args,
            tuple(sorted(kwargs.items())),
            frozenset().union(*(value.loads for value in values)),
            _unique_reductions(values),
        )

    def indirect_indexing(self, x, size, check=True, wrap_neg=True):
        """A fresh name for an index computed at run time.

        The index is not known until the body runs, so it cannot be captured as a
        value; what can be captured is that there is one here, and one name per
        one, so that two of them are still two.
        """

        return sympy.Symbol(f"indirect_{len(self.stores)}", integer=True)

    def load(self, name: str, index: Any) -> GemmEpilogueIRExpression:
        """A read, and what was written to the same place earlier if anything was.

        A buffer read after the body wrote to it is not a read of the original
        value but a read of what the body computed, so the earlier value is carried
        along: a question about this expression has to be able to see what it is
        really made of.
        """

        stored = self.stores.get(name)
        if stored is None:
            return GemmEpilogueIRExpression(
                "load", (name, index, None), loads=frozenset((name,))
            )
        return GemmEpilogueIRExpression(
            "load",
            (name, index, stored.value),
            loads=frozenset((name,)) | stored.value.loads,
            reductions=stored.value.reductions,
        )

    def reduction(self, dtype, src_dtype, reduction_type, value):
        """A walk, and the walks it made of its own value.

        A walk that is computed as more than one value -- a running total, a
        running mean and a count -- produces one expression per value, and each
        names the same walk.  They differ only in which of the walk's values they
        are, which is what the position says.
        """

        args = (dtype, src_dtype, reduction_type, value)
        if reduction_type in ("online_softmax_reduce", "welford_reduce"):
            count = 2 if reduction_type == "online_softmax_reduce" else 3
            return tuple(
                GemmEpilogueIRExpression(
                    "reduction",
                    (*args, index),
                    loads=value.loads,
                    reductions=(
                        *value.reductions,
                        GemmEpilogueIRReduction(reduction_type, value, index),
                    ),
                )
                for index in range(count)
            )
        return GemmEpilogueIRExpression(
            "reduction",
            args,
            loads=value.loads,
            reductions=(
                *value.reductions,
                GemmEpilogueIRReduction(reduction_type, value),
            ),
        )

    def store(
        self,
        name: str,
        index: Any,
        value: GemmEpilogueIRExpression,
        mode=None,
    ) -> None:
        self.stores[name] = GemmEpilogueIRStore(index, value)

    def store_reduction(
        self, name: str, index: Any, value: GemmEpilogueIRExpression
    ) -> None:
        self.stores[name] = GemmEpilogueIRStore(index, value)


def _loaded_names(expr: Any) -> frozenset[str]:
    """Which buffers a value is made of, however indirectly.

    A read names a buffer; everything else is made of what it was made of.  The
    value a buffer held earlier is followed too, because a value that reads what
    the body wrote is made of that as much as of the buffer.
    """

    if not isinstance(expr, GemmEpilogueIRExpression):
        return frozenset()
    if expr.op == "load":
        name, _, stored = expr.args
        return frozenset((name,)) | _loaded_names(stored)
    return frozenset().union(*(_loaded_names(arg) for arg in expr.args))


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRAnalysis:
    """What a set of bodies did, asked of the bodies rather than of the buffers.

    The bodies are replayed once, on construction, and every question afterwards is
    a question about what was captured.  That matters because a question about a
    body cannot be answered by asking the body: a body answers by writing, and
    what it wrote is the thing being questioned.
    """

    stores: dict[str, GemmEpilogueIRStore]

    @classmethod
    def from_buffers(
        cls, buffers: Sequence[ComputedBuffer]
    ) -> "GemmEpilogueIRAnalysis":
        """Replay the bodies of some buffers and keep what they did."""

        handler = _GemmEpilogueIRHandler()
        with V.set_ops_handler(handler):
            for buffer in buffers:
                buffer.get_store_function()(*buffer.data.inner_fn_args())
        return cls(handler.stores)

    @classmethod
    def store_from_buffer(cls, buffer: ComputedBuffer) -> GemmEpilogueIRStore | None:
        """What one body's one output was, or nothing if the body wrote none."""

        return cls.from_buffers((buffer,)).store(buffer.get_name())

    def store(self, name: str) -> GemmEpilogueIRStore | None:
        return self.stores.get(name)

    def output_role(self, name: str) -> GemmEpilogueIROutputRole | None:
        store = self.store(name)
        if store is None:
            return None
        transitive_inputs = store.value.loads or _loaded_names(store.value)
        return GemmEpilogueIROutputRole(
            transitive_inputs,
            transitive_inputs & self.reduction_stores,
        )

    def grouped_reduction(
        self,
        output_name: str,
        source_name: str,
        group: int,
        axis: int,
        source_dtype: Any,
    ) -> GemmReductionConfig | None:
        """The walk an output is, if it is one walk of one value.

        The axis is asked for rather than inferred here because the caller is the
        one that knows the geometry: which of two axes a body reduces is a fact
        about the tile it belongs to, and the body alone cannot say.
        """

        store = self.store(output_name)
        classified = (
            grouped_reduction_ir(store, source_name, group, source_dtype)
            if store is not None
            else None
        )
        if classified is None:
            return None
        reduction_type, source_type = classified
        return GemmReductionConfig(
            output_name, group, axis, reduction_type, source_type
        )

    def reduction_region(
        self,
        output_name: str,
        source_name: str,
        group: int,
        source_dtype: Any,
    ) -> GemmEpilogueIRRegion | None:
        """The walks an output is made of, and the expression they made it.

        A body that wrote its walks as operations already says which walks it did;
        one that unrolled them into arithmetic says so only by its shape, so those
        are recovered from the expression.  Either way the walks have to be of the
        named value and nothing else -- a region that also walks something else is
        not one walk of one value, and folding it in would fold in a decision
        nothing asked for.
        """

        store = self.store(output_name)
        if store is None:
            return None
        reductions = store.value.reductions or _synthetic_reductions_ir(
            store.value, store.index, source_name, group, source_dtype
        )
        if not reductions:
            return None
        if any(
            (reduction.source.loads or _loaded_names(reduction.source))
            != frozenset((source_name,))
            for reduction in reductions
        ):
            return None
        return GemmEpilogueIRRegion(output_name, reductions, store.value)

    def reduction_finalizer(
        self,
        output_name: str,
        source_name: str,
        group: int | None = None,
    ) -> GemmEpilogueIRFinalizer | None:
        """The one thing done to a finished walk, or nothing if it was not one.

        A body that read a value and wrote it back is a finalizer of kind identity,
        and that is a different thing from one that read it and wrote half of it:
        the first can be folded away, the second cannot.  So the kinds are named
        apart, and a body that did anything else is not a finalizer at all.
        """

        store = self.store(output_name)
        if store is None:
            return None
        if operation_names_ir(store).issubset(
            ("load", "to_dtype", "to_dtype_bitcast", "identity")
        ):
            kind = "identity"
        elif group is not None and single_source_affine_ir(store, source_name) == (
            1.0 / group,
            0.0,
        ):
            kind = "mean"
        elif is_absmax_scale_finalizer_ir(store, source_name):
            kind = "absmax_scale"
        elif (
            store.value.loads == frozenset((source_name,))
            and not store.value.reductions
        ):
            kind = "generic"
        else:
            return None
        return GemmEpilogueIRFinalizer(output_name, source_name, kind)

    @property
    def reduction_stores(self) -> frozenset[str]:
        """Which outputs walk something, whether they said so or their shape did."""

        return frozenset(
            name
            for name, store in self.stores.items()
            if store.value.reductions or _contains_reduction(store.value)
        )


def _constant_value(expr: Any) -> Any | None:
    """The number an expression is, or nothing if it is not a number.

    A number can be written through a cast without becoming something else, so a
    cast of a number is still that number; anything else is not a number however
    it was written.
    """

    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op in ("constant", "index_expr") and expr.args:
        return expr.args[0]
    if expr.op in ("to_dtype", "to_dtype_bitcast") and expr.args:
        return _constant_value(expr.args[0])
    return None


def _strip_conversions(expr: Any) -> Any:
    """The value under any casts and no-ops, which compute it just as well.

    A cast changes how a number is stored and not what it is, so a question about
    what a value *is* has to see through one.  A no-op is the same: it is written,
    and it is nothing.
    """

    while (
        isinstance(expr, GemmEpilogueIRExpression)
        and expr.op in ("to_dtype", "to_dtype_bitcast", "identity")
        and expr.args
    ):
        expr = expr.args[0]
    return expr


def _walk(expr: Any) -> Iterator[GemmEpilogueIRExpression]:
    """Every node of an expression, the node itself first."""

    if not isinstance(expr, GemmEpilogueIRExpression):
        return
    yield expr
    for arg in expr.args:
        yield from _walk(arg)


def grouped_reduction_axis_ir(
    reduction: GemmEpilogueIRReduction, group: int, n: int
) -> int | None:
    """Which axis a walk collapses, read off the strides of what it read.

    A walk that unrolled a group of `group` values read them at indices that form
    an arithmetic progression, and the step says which axis it moved along: a step
    of one is the axis that moves fastest, and a step of n is the one that moves
    once per row.  A walk that read at no such pattern, or whose pattern fits
    both, is not one of these two and is left unsaid -- a walk given an axis it
    does not have is a walk folded along the wrong one.
    """

    indices = [
        expr.args[1]
        for expr in _walk(reduction.source)
        if expr.op == "load" and len(expr.args) > 1
    ]
    unique_indices = []
    for index in indices:
        if not any(sympy.simplify(index - other) == 0 for other in unique_indices):
            unique_indices.append(index)
    if len(unique_indices) != group:
        return None

    def is_progression(step: int) -> bool:
        return any(
            all(
                any(
                    sympy.simplify(index - (base + offset * step)) == 0
                    for index in unique_indices
                )
                for offset in range(group)
            )
            for base in unique_indices
        )

    axes = [axis for axis, step in ((1, 1), (0, n)) if is_progression(step)]
    return axes[0] if len(axes) == 1 else None


def operation_names_ir(store: GemmEpilogueIRStore) -> frozenset[str]:
    """Which operations a body performed, as a set.

    A set rather than a sequence because the question being asked is whether some
    operation is among them, and an order would suggest a dependence that is not
    there.
    """

    return frozenset(expr.op for expr in _walk(store.value))


def _source_transform(
    expr: Any,
    source_name: str,
    allowed_conversion_dtypes: Optional[frozenset[Any]] = None,
) -> Optional[str]:
    """What was done to the named value to get here, in words.

    Three answers are possible: it was read, its magnitude was taken, or it was
    multiplied by itself.  Anything else is not one of these and is left unsaid,
    because a walk that saw anything else is a walk whose result cannot be read
    off by whoever asked.

    When the dtypes that may be cast through are given, a cast to any other one is
    refused outright: it changes the value rather than its storage, and a value
    that was rounded on the way to the walk is not the value the walk was said to
    be over.
    """

    if allowed_conversion_dtypes is None:
        expr = _strip_conversions(expr)
    else:
        while isinstance(expr, GemmEpilogueIRExpression) and expr.args:
            if expr.op == "identity":
                expr = expr.args[0]
            elif expr.op == "to_dtype":
                if len(expr.args) < 2 or expr.args[1] not in allowed_conversion_dtypes:
                    return None
                expr = expr.args[0]
            elif expr.op == "to_dtype_bitcast":
                return None
            else:
                break
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op == "load":
        return "identity" if expr.args[0] == source_name else None
    if expr.op == "abs":
        return (
            "abs"
            if _source_transform(expr.args[0], source_name, allowed_conversion_dtypes)
            == "identity"
            else None
        )
    if expr.op == "mul" and expr.args[0] == expr.args[1]:
        return (
            "square"
            if _source_transform(expr.args[0], source_name, allowed_conversion_dtypes)
            == "identity"
            else None
        )
    if expr.op == "pow":
        exponent = _constant_value(expr.args[1])
        return (
            "square"
            if exponent == 2
            and _source_transform(expr.args[0], source_name, allowed_conversion_dtypes)
            == "identity"
            else None
        )
    return None


def _flatten_associative(expr: Any, op: str) -> list[Any]:
    """The terms of a chain of the same operation, however long the chain is.

    Addition and multiplication may be written in any order and any grouping
    without changing the value, so the terms are all that a question about the
    value can depend on -- the shape of the chain says nothing the terms do not.
    """

    stripped = _strip_conversions(expr)
    if isinstance(stripped, GemmEpilogueIRExpression) and stripped.op == op:
        return _flatten_associative(stripped.args[0], op) + _flatten_associative(
            stripped.args[1], op
        )
    return [expr]


def grouped_reduction_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    group: int,
    source_dtype: Any,
) -> tuple[str, str] | None:
    """Name the walk a body performed, and what it walked, or say nothing.

    Two ways of writing one are recognised, because a body may take either: as a
    walk the body performed itself, or as the arithmetic of that walk unrolled.
    The first is read from the operations; the second from the shape of the
    expression, which is why the group has to be known -- a sum of two values is a
    sum of two, and only a sum of the whole group is the walk.

    A body that walked something and also did arithmetic of its own is not a walk:
    it is a walk and something else, and answering about the walk alone would
    leave the something else unsaid.
    """

    # A cast to the 32-bit float is safe from any source: it does not lose
    # anything a walk of the source could have kept.  Any other cast rounds,
    # and a walk of a rounded value is not a walk of the value.
    allowed_conversion_dtypes = frozenset((source_dtype, _tp.float32))
    candidates = [expr for expr in _walk(store.value) if expr.op == "reduction"]
    matches = []
    for reduction in candidates:
        reduction_type = str(reduction.args[2])
        source_type = _source_transform(
            reduction.args[3], source_name, allowed_conversion_dtypes
        )
        if source_type is not None:
            matches.append((reduction_type, source_type))
    if candidates:
        return matches[0] if len(candidates) == len(matches) == 1 else None

    root = _strip_conversions(store.value)
    while isinstance(root, GemmEpilogueIRExpression) and root.op in (
        "add",
        "mul",
        "truediv",
    ):
        if root.op == "truediv" and _constant_value(root.args[1]) == group:
            terms = _flatten_associative(root.args[0], "add")
            transforms = [
                _source_transform(term, source_name, allowed_conversion_dtypes)
                for term in terms
            ]
            if (
                len(terms) == group
                and len(frozenset(transforms)) == 1
                and transforms[0]
            ):
                return "mean", transforms[0]
        terms = _flatten_associative(root, root.op)
        transforms = [
            _source_transform(term, source_name, allowed_conversion_dtypes)
            for term in terms
        ]
        if len(terms) == group and len(frozenset(transforms)) == 1 and transforms[0]:
            return ("sum" if root.op == "add" else "prod"), transforms[0]
        break
    for op, reduction_type in (("maximum", "max"), ("minimum", "min")):
        terms = _flatten_associative(root, op)
        transforms = [
            _source_transform(term, source_name, allowed_conversion_dtypes)
            for term in terms
        ]
        if len(terms) == group and len(frozenset(transforms)) == 1 and transforms[0]:
            return reduction_type, transforms[0]
    return None


def _synthetic_reductions_ir(
    expr: Any,
    index: Any,
    source_name: str,
    group: int,
    source_dtype: Any,
) -> tuple[GemmEpilogueIRReduction, ...]:
    """The walks an expression performed without saying so.

    A body that unrolled its walks wrote arithmetic where a walk would have gone.
    The arithmetic is the walk, so the walks are named from it -- the same names a
    body that said so would have used, so that a question asked of either is
    answered the same way.
    """

    if not isinstance(expr, GemmEpilogueIRExpression):
        return ()
    classified = grouped_reduction_ir(
        GemmEpilogueIRStore(index, expr), source_name, group, source_dtype
    )
    if classified is not None:
        reduction_type, source_type = classified
        return (GemmEpilogueIRReduction(reduction_type, expr, source_type=source_type),)
    reductions = []
    seen: OrderedSet[int] = OrderedSet()
    for arg in expr.args:
        for value in _expression_values(arg):
            for reduction in _synthetic_reductions_ir(
                value, index, source_name, group, source_dtype
            ):
                if id(reduction.source) not in seen:
                    seen.add(id(reduction.source))
                    reductions.append(reduction)
    return tuple(reductions)


def is_direct_bool_gt_zero_ir(store: GemmEpilogueIRStore, source_name: str) -> bool:
    """Whether a body compared one read of a buffer against zero, and nothing else.

    Compared directly, because a comparison that went through arithmetic is a
    comparison of something else that happens to be compared against zero.
    """

    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op not in ("gt", "ge"):
        return False
    lhs, rhs = map(_strip_conversions, expr.args[:2])
    return (
        isinstance(lhs, GemmEpilogueIRExpression)
        and lhs.op == "load"
        and lhs.args[0] == source_name
        and _constant_value(rhs) == 0
    )


def is_absmax_scale_finalizer_ir(store: GemmEpilogueIRStore, source_name: str) -> bool:
    """Whether a body wrote the scale a run of low-precision values needs.

    The shape is the largest magnitude of the value, kept away from zero, divided
    by the largest value the low-precision format can hold -- so that dividing by
    the result brings the run inside the format's range and multiplying by it
    brings the range back.  The three numbers are what make it this and not any
    other scale, so all three are required.
    """

    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression):
        return False
    if expr.op == "truediv" and _constant_value(expr.args[1]) == 448.0:
        clamped = expr.args[0]
    elif expr.op == "mul":
        lhs, rhs = expr.args[:2]
        if _constant_value(lhs) == 1.0 / 448.0:
            clamped = rhs
        elif _constant_value(rhs) == 1.0 / 448.0:
            clamped = lhs
        else:
            return False
    else:
        return False

    clamped = _strip_conversions(clamped)
    if not isinstance(clamped, GemmEpilogueIRExpression):
        return False
    if clamped.op in ("maximum", "clamp_min"):
        lhs, rhs = clamped.args[:2]
        return (
            _source_transform(lhs, source_name) == "identity"
            and _constant_value(rhs) == 1e-12
        ) or (
            _source_transform(rhs, source_name) == "identity"
            and _constant_value(lhs) == 1e-12
        )
    if clamped.op == "clamp" and len(clamped.args) >= 2:
        return (
            _source_transform(clamped.args[0], source_name) == "identity"
            and _constant_value(clamped.args[1]) == 1e-12
            and (len(clamped.args) < 3 or clamped.args[2] is None)
        )
    return False


def _contains_reduction(expr: Any) -> bool:
    """Whether a value is made of a walk anywhere inside it."""

    if not isinstance(expr, GemmEpilogueIRExpression):
        return False
    return expr.op == "reduction" or any(_contains_reduction(arg) for arg in expr.args)


def _affine_scale(
    value: tuple[float, float, float], scale: float
) -> tuple[float, float, float]:
    return value[0] * scale, value[1] * scale, value[2] * scale


def _affine_add(
    lhs: tuple[float, float, float], rhs: tuple[float, float, float]
) -> tuple[float, float, float]:
    return lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]


def _affine_coefficients(
    expr: Any, source_name: str, reduction_names: frozenset[str]
) -> tuple[float, float, float] | None:
    """How much of each of three things a value is, or nothing if it is not linear.

    Three numbers because three things can be mixed in one linear combination: the
    value read, the value a walk produced, and a number.  A value that multiplies
    two of them is not linear in any of them, so the reading stops there rather
    than guessing which one the product was of.
    """

    expr = _strip_conversions(expr)
    if isinstance(expr, (int, float, sympy.Number)) and not isinstance(expr, bool):
        return 0.0, 0.0, float(expr)
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op == "load":
        name, _, stored = expr.args
        if name == source_name:
            return 1.0, 0.0, 0.0
        if name in reduction_names or _contains_reduction(stored):
            return 0.0, 1.0, 0.0
        return _affine_coefficients(stored, source_name, reduction_names)
    if expr.op == "reduction":
        return 0.0, 1.0, 0.0
    constant = _constant_value(expr)
    if isinstance(constant, (int, float, sympy.Number)) and not isinstance(
        constant, bool
    ):
        return 0.0, 0.0, float(constant)
    if expr.op == "neg":
        value = _affine_coefficients(expr.args[0], source_name, reduction_names)
        return None if value is None else _affine_scale(value, -1.0)
    if expr.op in ("add", "sub"):
        lhs = _affine_coefficients(expr.args[0], source_name, reduction_names)
        rhs = _affine_coefficients(expr.args[1], source_name, reduction_names)
        if lhs is None or rhs is None:
            return None
        scale = 1.0 if expr.op == "add" else -1.0
        return _affine_add(lhs, _affine_scale(rhs, scale))
    if expr.op in ("mul", "truediv"):
        lhs = _affine_coefficients(expr.args[0], source_name, reduction_names)
        rhs = _affine_coefficients(expr.args[1], source_name, reduction_names)
        if lhs is None or rhs is None:
            return None
        if expr.op == "truediv":
            if rhs[:2] != (0.0, 0.0) or rhs[2] == 0.0:
                return None
            return _affine_scale(lhs, 1.0 / rhs[2])
        if lhs[:2] == (0.0, 0.0):
            return _affine_scale(rhs, lhs[2])
        if rhs[:2] == (0.0, 0.0):
            return _affine_scale(lhs, rhs[2])
    if expr.op == "fma":
        lhs = _affine_coefficients(expr.args[0], source_name, reduction_names)
        rhs = _affine_coefficients(expr.args[1], source_name, reduction_names)
        addend = _affine_coefficients(expr.args[2], source_name, reduction_names)
        if lhs is None or rhs is None or addend is None:
            return None
        if lhs[:2] == (0.0, 0.0):
            product = _affine_scale(rhs, lhs[2])
        elif rhs[:2] == (0.0, 0.0):
            product = _affine_scale(lhs, rhs[2])
        else:
            return None
        return _affine_add(product, addend)
    return None


def centered_mean_consumer_type_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    reduction_names: frozenset[str],
    reduction_scale: float = 1.0,
) -> str | None:
    """Name an affine combination of a value and a walk, with its numbers.

    The numbers are in the name because the combination is what the caller has to
    reproduce: two combinations with the same words and different numbers are two
    different computations, and a caller that was told only the words would have
    to guess the numbers it needs.  They are written at full precision because the
    whole point is that they can be read back exactly.
    """

    coefficients = _affine_coefficients(store.value, source_name, reduction_names)
    if coefficients is not None:
        coefficients = (
            coefficients[0],
            coefficients[1] * reduction_scale,
            coefficients[2],
        )
    if (
        coefficients is None
        or coefficients[0] == 0.0
        or coefficients[1] == 0.0
        or not all(math.isfinite(value) for value in coefficients)
    ):
        return None
    return "mean_linear:" + ":".join(format(value, ".17g") for value in coefficients)


def single_source_affine_ir(
    store: GemmEpilogueIRStore, source_name: str
) -> tuple[float, float] | None:
    """The scale and shift of a value that is one buffer's value scaled and shifted.

    Nothing else may be mixed in: a value that is also a walk's result is not an
    affine of the buffer, and reporting one for it would lose the walk.
    """

    coefficients = _affine_coefficients(store.value, source_name, frozenset())
    if coefficients is None or coefficients[1] != 0.0:
        return None
    return coefficients[0], coefficients[2]


def _affine_around(expr: Any, basis) -> tuple[float, float] | None:
    """The scale and shift of an expression, if everything in it is one of those.

    A predicate says what may be multiplied; everything else has to be a number.
    An expression that multiplied two things the predicate accepts is not an
    affine of either -- the product of two is not linear in either -- and the
    reading stops rather than picking one.
    """

    expr = _strip_conversions(expr)
    if basis(expr):
        return 1.0, 0.0
    constant = _constant_value(expr)
    if isinstance(constant, (int, float, sympy.Number)) and not isinstance(
        constant, bool
    ):
        return 0.0, float(constant)
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op == "neg":
        value = _affine_around(expr.args[0], basis)
        return None if value is None else (-value[0], -value[1])
    if expr.op in ("add", "sub"):
        lhs = _affine_around(expr.args[0], basis)
        rhs = _affine_around(expr.args[1], basis)
        if lhs is None or rhs is None:
            return None
        sign = 1.0 if expr.op == "add" else -1.0
        return lhs[0] + sign * rhs[0], lhs[1] + sign * rhs[1]
    if expr.op == "mul":
        lhs = _affine_around(expr.args[0], basis)
        rhs = _affine_around(expr.args[1], basis)
        if lhs is None or rhs is None:
            return None
        if lhs[0] == 0.0:
            return rhs[0] * lhs[1], rhs[1] * lhs[1]
        if rhs[0] == 0.0:
            return lhs[0] * rhs[1], lhs[1] * rhs[1]
    return None


def _is_source(expr: Any, source_name: str) -> bool:
    return _source_transform(expr, source_name) == "identity"


def _sum_terms(expr: Any, source_name: str, group: int) -> bool:
    """Whether a value is the whole group of values, added."""

    terms = _flatten_associative(expr, "add")
    return len(terms) == group and all(_is_source(term, source_name) for term in terms)


def _group_max(expr: Any, source_name: str, group: int) -> bool:
    """Whether a value is the whole group of values, compared for the largest."""

    terms = _flatten_associative(expr, "maximum")
    return len(terms) == group and all(_is_source(term, source_name) for term in terms)


def _stable_group_max(expr: Any, source_name: str, group: int) -> bool:
    """Whether a value is the largest of the group, guarded against an infinity.

    The guard is the same computation -- the largest of the group -- written so
    that an infinite largest becomes a finite one, because the value that follows
    is exponentiated after being shifted by it and an infinity there is an
    infinity everywhere.  It is recognised as the same largest, because it is.
    """

    expr = _strip_conversions(expr)
    if _group_max(expr, source_name, group):
        return True
    if not (
        isinstance(expr, GemmEpilogueIRExpression)
        and expr.op == "where"
        and len(expr.args) >= 3
        and _constant_value(expr.args[1]) == 0.0
    ):
        return False
    condition = _strip_conversions(expr.args[0])
    maximum = _strip_conversions(expr.args[2])
    if not (
        isinstance(condition, GemmEpilogueIRExpression)
        and condition.op == "eq"
        and _group_max(maximum, source_name, group)
    ):
        return False
    for absolute, infinity in (condition.args[:2], reversed(condition.args[:2])):
        absolute = _strip_conversions(absolute)
        if (
            isinstance(absolute, GemmEpilogueIRExpression)
            and absolute.op == "abs"
            and _strip_conversions(absolute.args[0]) == maximum
            and _constant_value(infinity) == math.inf
        ):
            return True
    return False


def _shifted_exp(expr: Any, source_name: str) -> Any | None:
    """The largest value a value was shifted by before being exponentiated, or nothing.

    Both halves are required.  An exponentiated value that was not shifted is the
    spelling that overflows, and it is not this one: recognising it here would have
    a caller promise a stability the body does not have.
    """

    expr = _strip_conversions(expr)
    if not (
        isinstance(expr, GemmEpilogueIRExpression) and expr.op == "exp" and expr.args
    ):
        return None
    shifted = _strip_conversions(expr.args[0])
    if not (
        isinstance(shifted, GemmEpilogueIRExpression)
        and shifted.op == "sub"
        and _is_source(shifted.args[0], source_name)
    ):
        return None
    return _strip_conversions(shifted.args[1])


def is_softmax_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    group: int,
    reduction_names: frozenset[str] = frozenset(),
) -> bool:
    """Whether a body turned a group of values into a distribution over them.

    All four parts are required: the largest value, the exponentials of what was
    shifted by it, the total of those, and a divide by the total.  A value with
    three of them is a quantity near one, not a distribution, and a product told it
    was a distribution would write a different number.

    A body that kept its walks instead of unrolling them reads the largest and the
    total as two reads of two named walks rather than as arithmetic, so both ways
    of writing it are recognised -- the two compute the same distribution, and
    neither is the other's mistake.
    """

    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "truediv":
        return False
    numerator, denominator = expr.args[:2]
    maximum = _shifted_exp(numerator, source_name)
    if maximum is None:
        return False
    denominator = _strip_conversions(denominator)
    if reduction_names:
        return (
            isinstance(maximum, GemmEpilogueIRExpression)
            and maximum.op == "load"
            and maximum.args[0] in reduction_names
            and isinstance(denominator, GemmEpilogueIRExpression)
            and denominator.op == "load"
            and denominator.args[0] in reduction_names
            and maximum.args[0] != denominator.args[0]
        )
    terms = _flatten_associative(denominator, "add")
    return (
        _group_max(maximum, source_name, group)
        and len(terms) == group
        and all(_shifted_exp(term, source_name) == maximum for term in terms)
    )


def is_absmax_normalize_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    scale_names: str | frozenset[str],
) -> bool:
    """Whether a body brought a group of values inside a low-precision range.

    Written as a multiply by the reciprocal of the scale rather than as a divide,
    which is what a body that had the reciprocal to hand would write.  Both compute
    the same thing, so both are recognised; the scale may be under any of the names
    given, because which buffer held it is the caller's business and not this
    question's.
    """

    if isinstance(scale_names, str):
        scale_names = frozenset((scale_names,))
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "mul":
        return False
    for source, reciprocal in (expr.args[:2], reversed(expr.args[:2])):
        reciprocal = _strip_conversions(reciprocal)
        if (
            _is_source(source, source_name)
            and isinstance(reciprocal, GemmEpilogueIRExpression)
            and reciprocal.op == "reciprocal"
            and any(
                is_absmax_scale_finalizer_ir(
                    GemmEpilogueIRStore(store.index, reciprocal.args[0]), scale_name
                )
                for scale_name in scale_names
            )
        ):
            return True
    return False


def _sum_affine_ir(
    expr: Any,
    source_name: str,
    reduction_names: frozenset[str],
    group: int,
) -> tuple[float, float] | None:
    """The scale and shift of a total, whether the total was walked or added."""

    def is_sum(candidate: Any) -> bool:
        candidate = _strip_conversions(candidate)
        return (
            isinstance(candidate, GemmEpilogueIRExpression)
            and candidate.op == "load"
            and candidate.args[0] in reduction_names
        ) or _sum_terms(candidate, source_name, group)

    return _affine_around(expr, is_sum)


def sum_normalize_consumer_type_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    reduction_names: frozenset[str],
    group: int,
) -> str | None:
    """Name a value divided by a total, and say which way round and by how much.

    Which way round is in the name because a value divided by a total is not the
    same as a total divided by a value, and the two are written the same way round
    the other time.  The numbers are in it for the same reason as elsewhere: they
    are what a caller has to reproduce.

    The whole expression may be scaled and shifted around the division, so the
    division is found inside an affine rather than demanded to be the whole thing.
    """

    parameters: tuple[str, float, float] | None = None

    def is_normalization(expr: Any) -> bool:
        nonlocal parameters
        expr = _strip_conversions(expr)
        if not isinstance(expr, GemmEpilogueIRExpression):
            return False
        if expr.op == "truediv":
            lhs, rhs = expr.args[:2]
            if _is_source(lhs, source_name):
                affine = _sum_affine_ir(rhs, source_name, reduction_names, group)
                if affine is not None:
                    parameters = "forward", *affine
                    return True
            if _is_source(rhs, source_name):
                affine = _sum_affine_ir(lhs, source_name, reduction_names, group)
                if affine is not None:
                    parameters = "reverse", *affine
                    return True
        if expr.op == "mul":
            for source, reciprocal in (expr.args[:2], reversed(expr.args[:2])):
                reciprocal = _strip_conversions(reciprocal)
                if (
                    _is_source(source, source_name)
                    and isinstance(reciprocal, GemmEpilogueIRExpression)
                    and reciprocal.op == "reciprocal"
                ):
                    affine = _sum_affine_ir(
                        reciprocal.args[0], source_name, reduction_names, group
                    )
                    if affine is not None:
                        parameters = "forward", *affine
                        return True
        return False

    affine = _affine_around(store.value, is_normalization)
    if (
        affine is None
        or parameters is None
        or affine[0] == 0.0
        or not all(math.isfinite(value) for value in (*affine, *parameters[1:]))
    ):
        return None
    kind = (
        "normalize_sum_affine"
        if parameters[0] == "forward"
        else "normalize_sum_reverse_affine"
    )
    values = (*affine, *parameters[1:])
    return kind + ":" + ":".join(format(value, ".17g") for value in values)


def sum_multiply_consumer_type_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    reduction_names: frozenset[str],
    group: int,
) -> str | None:
    """Name a value multiplied by a total, which is a different thing from divided.

    Kept apart from the division deliberately: a value multiplied by a total is not
    a distribution over the group, and treating the two the same would have a
    product fold one into a tile as if it were the other.
    """

    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "mul":
        return None
    for source, reduction in (expr.args[:2], reversed(expr.args[:2])):
        if not _is_source(source, source_name):
            continue
        affine = _sum_affine_ir(reduction, source_name, reduction_names, group)
        if affine is not None and all(math.isfinite(value) for value in affine):
            return "sum_mul_affine:" + ":".join(
                format(value, ".17g") for value in affine
            )
    return None


def variance_parameters_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    group: int
) -> tuple[float, float] | None:
    """The scale and shift a spread is reported in, when a body computed one.

    A spread is the total of the squared differences from the mean, divided by
    how many there were -- and every part of that has to be there, because a total
    of squared differences is not a spread, and a spread without its mean is a
    quantity nobody asked for.  What is reported is the affine applied to it,
    which is the only part of the answer that is not already said by the shape.
    """

    def is_variance(expr: Any) -> bool:
        expr = _strip_conversions(expr)
        if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "truediv":
            return False
        if _constant_value(expr.args[1]) != group:
            return False
        squares = _flatten_associative(expr.args[0], "add")
        if len(squares) != group:
            return False
        for square in squares:
            square = _strip_conversions(square)
            if not (
                isinstance(square, GemmEpilogueIRExpression)
                and square.op == "mul"
                and square.args[0] == square.args[1]
            ):
                return False
            centered = _strip_conversions(square.args[0])
            if not (
                isinstance(centered, GemmEpilogueIRExpression)
                and centered.op == "sub"
                and _is_source(centered.args[0], source_name)
            ):
                return False
            mean = _strip_conversions(centered.args[1])
            if not (
                isinstance(mean, GemmEpilogueIRExpression)
                and mean.op == "truediv"
                and _constant_value(mean.args[1]) == group
                and _sum_terms(mean.args[0], source_name, group)
            ):
                return False
        return True

    affine = _affine_around(store.value, is_variance)
    return affine if affine is not None and affine[0] != 0.0 else None


def centered_mean_consumer_type_unrolled_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    group: int
) -> str | None:
    """Name an affine of a value and its mean, in a body that unrolled the mean.

    Written separately from the walk-based naming because the arithmetic is
    different -- a mean written as a divide is a divide, and reading it through the
    walk's name would have to pretend the body had walked when it had not.  The
    three numbers are in the name for the reason they are everywhere.
    """

    def coefficients(expr: Any) -> tuple[float, float, float] | None:
        expr = _strip_conversions(expr)
        if isinstance(expr, GemmEpilogueIRExpression) and expr.op == "truediv":
            if _constant_value(expr.args[1]) == group and _sum_terms(
                expr.args[0], source_name, group
            ):
                return 0.0, 1.0, 0.0
        if _is_source(expr, source_name):
            return 1.0, 0.0, 0.0
        constant = _constant_value(expr)
        if isinstance(constant, (int, float, sympy.Number)) and not isinstance(
            constant, bool
        ):
            return 0.0, 0.0, float(constant)
        if not isinstance(expr, GemmEpilogueIRExpression):
            return None
        if expr.op in ("add", "sub"):
            lhs, rhs = coefficients(expr.args[0]), coefficients(expr.args[1])
            if lhs is None or rhs is None:
                return None
            return _affine_add(
                lhs, _affine_scale(rhs, 1.0 if expr.op == "add" else -1.0)
            )
        if expr.op == "mul":
            lhs, rhs = coefficients(expr.args[0]), coefficients(expr.args[1])
            if lhs is None or rhs is None:
                return None
            if lhs[:2] == (0.0, 0.0):
                return _affine_scale(rhs, lhs[2])
            if rhs[:2] == (0.0, 0.0):
                return _affine_scale(lhs, rhs[2])
        return None

    values = coefficients(store.value)
    if (
        values is None
        or values[0] == 0.0
        or values[1] == 0.0
        or not all(math.isfinite(value) for value in values)
    ):
        return None
    return "mean_linear:" + ":".join(format(value, ".17g") for value in values)


def is_logsumexp_ir(store: GemmEpilogueIRStore, source_name: str, group: int) -> bool:
    """Whether a body turned a group of values into the log of their total.

    The largest value plus the log of the total of what was shifted by it.  Both
    halves are required and the shift has to be by that same largest value: the
    two halves are only the same computation when the shift is the one the first
    half computed, and taking the shift from anywhere else is a different number
    that happens to look like one.
    """

    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "add":
        return False
    for maximum, logarithm in (expr.args[:2], reversed(expr.args[:2])):
        maximum = _strip_conversions(maximum)
        logarithm = _strip_conversions(logarithm)
        if not (
            _stable_group_max(maximum, source_name, group)
            and isinstance(logarithm, GemmEpilogueIRExpression)
            and logarithm.op == "log"
        ):
            continue
        terms = _flatten_associative(logarithm.args[0], "add")
        if len(terms) == group and all(
            _shifted_exp(term, source_name) == maximum for term in terms
        ):
            return True
    return False
