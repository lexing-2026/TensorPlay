"""Carrying a structured position through arithmetic that thinks in offsets.

A position in a tensor is usually one number: the distance from the start of
the buffer, which is what indexing arithmetic wants because a load is at an
offset.  The emitter used for these kernels wants the opposite -- the position
as one number per dimension, so it can write ``tensor[i, j]`` and be answerable
for strides itself.

Both are wanted at once, because the position is built by arithmetic that only
understands offsets and is finally handed to an emitter that only understands
dimensions.  So the position stays one expression, and the dimensions ride
along inside it: a value that is an expression to everything that touches it,
and a tuple of coordinates to the one place that asks.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import tensorplay as tp
import sympy
from tensorplay.graph.experimental.sympy_functions import FloorDiv


class HierarchicalIndex(sympy.Function):
    """One position in a tensor, held as one number per dimension.

    Nothing is done to the value it holds.  It is not simplified, not flattened
    and not reordered, because the dimensions are not a number that could be:
    they are a position, and a position that was rearranged would be a
    different one.  So evaluating it produces nothing, which is what tells the
    expression machinery to carry the node as it stands.

    A value like this is meant to be short-lived -- built where a position is
    produced and taken apart where it is consumed -- and only the emitter for
    these kernels reads it, by taking the node's arguments as the coordinates.
    """

    @classmethod
    def eval(cls, *args):
        return None


# ---------------------------------------------------------------------------
# A mask written as a range of lanes
# ---------------------------------------------------------------------------


def _simplify_expr(expr: Any) -> Any:
    """The expression as something a pattern can be read off.

    Simplified first because the pattern below reads a shape rather than a
    spelling: the same range written two ways is one range, and recognising it
    as one is the whole of what this is for.
    """

    if isinstance(expr, sympy.Expr):
        return sympy.sympify(expr).simplify()
    return expr


def render_sympy_args(
    args: Any, symbol_codes: Any = None
) -> Any:
    """Every argument as code, or nothing if any of them cannot be.

    All or nothing rather than the ones that could be: an operation with one
    argument missing is not that operation with a hole in it, and a half-written
    one would be read as a smaller number rather than as nothing.
    """

    rendered = []
    for arg in args:
        code = sympy_to_cute_index(arg, symbol_codes)
        if code is None:
            return None
        rendered.append(code)
    return rendered


def sympy_to_cute_index(expr: Any, symbol_codes: Any = None) -> Any:
    """A whole-number expression as the code that computes it.

    Only the forms these masks are made of are answered, and anything else
    answers with nothing rather than with a guess: a mask written as a shape
    that was rendered wrongly would read memory that belongs to a different
    lane, which is a wrong answer with no sign that it is one.
    """

    expr = _simplify_expr(expr)
    if isinstance(expr, sympy.Integer):
        return f"cutlass.Int32({int(expr)})"
    if isinstance(expr, sympy.Symbol):
        if symbol_codes is not None and expr in symbol_codes:
            return symbol_codes[expr]
        # The two positions a mask is written against are read as the first
        # element of their own index, because that is how they arrive.
        if expr.name == "q_idx":
            return "q_idx[0]"
        if expr.name == "kv_idx":
            return "kv_idx[0]"
        return None
    if isinstance(expr, sympy.Add) or isinstance(expr, sympy.Mul):
        args = render_sympy_args(expr.args, symbol_codes)
        if args is None:
            return None
        separator = " + " if isinstance(expr, sympy.Add) else " * "
        return "(" + separator.join(args) + ")"
    if isinstance(expr, (sympy.Min, sympy.Max)):
        args = render_sympy_args(expr.args, symbol_codes)
        if args is None:
            return None
        op = "min" if isinstance(expr, sympy.Min) else "max"
        return f"{op}(" + ", ".join(args) + ")"
    if isinstance(expr, FloorDiv):
        args = render_sympy_args(expr.args, symbol_codes)
        if args is None:
            return None
        return f"({args[0]} // {args[1]})"
    return None


@dataclasses.dataclass(frozen=True)
class PackedMaskInterval:
    """The lanes a mask keeps: from one up to but not including another.

    A mask that says which lanes of a group are kept can be evaluated one lane
    at a time, or -- when the answer is a range -- as one number describing the
    range.  The second is a whole group of answers in one value, and is what
    makes a mask over a group of lanes cost the same as a mask over one.

    A mask over a window of lanes starting at the window's own position is the
    common case: "the query's position is at or past this one's" keeps every
    lane from the start of the window up to the difference between them.
    """

    lower_lane: Any
    upper_lane_exclusive: Any
    #: The code for symbols standing in for values that are the same across a
    #: group of lanes.  Kept apart from the range because a range says which
    #: lanes and this says how to name what they are read against, and the two
    #: are not the same question.
    symbol_codes: Any = dataclasses.field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def full(cls) -> "PackedMaskInterval":
        """Every lane kept, which is the mask that keeps everything."""

        return cls(sympy.Integer(0), sympy.Integer(32))

    def is_full(self) -> bool:
        return self.lower_lane == 0 and self.upper_lane_exclusive == 32

    def with_symbol_codes(self, symbol_codes: Any) -> "PackedMaskInterval":
        return dataclasses.replace(self, symbol_codes=dict(symbol_codes))

    def render_lower(self) -> str:
        return self._render(self.lower_lane)

    def render_upper(self) -> str:
        return self._render(self.upper_lane_exclusive)

    def keep_mask_expr(self) -> str:
        """The group of lanes kept, as one number rather than as a comparison.

        A run of lanes is a run of bits, and a run of bits is two shifted
        bounds: everything above the lower bound, and everything below the
        upper one, with the two overlapping.  Both bounds are clamped into the
        group first, because a run that starts before the group or ends after
        it is the same run as one that starts at its edge or ends at its edge --
        and the group is what there is.
        """

        lower = self.render_lower()
        upper = self.render_upper()
        return (
            "(utils.shr_u32(cutlass.Uint32(0xFFFFFFFF), "
            "cutlass.Uint32(min(max(cutlass.Int32(32) - "
            f"{upper}, cutlass.Int32(0)), cutlass.Int32(32)))) & "
            "utils.shl_u32(cutlass.Uint32(0xFFFFFFFF), "
            "cutlass.Uint32(min(max("
            f"{lower}, cutlass.Int32(0)), cutlass.Int32(32)))))"
        )

    def _render(self, expr: Any) -> str:
        rendered = sympy_to_cute_index(expr, self.symbol_codes)
        if rendered is None:
            raise AssertionError(f"failed to render expr to cute index: {expr}")
        return rendered


IntervalSet = tuple
MaybeIntervalSet = Any


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FlexFlashConfig:
    """One way of writing the kernel, among the ways worth measuring.

    How many elements one pass over the kernel handles is a choice about what
    the hardware does with them rather than about what the kernel computes, and
    the two are answered separately because they are: the width a mask is
    evaluated at is a property of the mask, and the width a score is applied at
    is a property of the score.
    """

    #: How many elements one pass applies the score to.  Left unset to take the
    #: kernel's own width, which is the right answer when nothing has been
    #: measured to say otherwise.
    score_mod_vec_size: Any = None
    #: How many consecutive lanes one pass evaluates the mask over.  The same
    #: width as the mask is read at, so that a mask written as a range of lanes
    #: is one value rather than a comparison per lane.
    mask_mod_vec_size: Any = None
    #: The ranges the mask keeps, worked out ahead of the kernel rather than in
    #: it.  Only usable for a mask that is a range; unset leaves the kernel to
    #: work it out per lane.
    mask_mod_packed_intervals: Any = None


def collect_aux_scalar_symbols(*buffer_groups: Any) -> tuple:
    """The symbols every captured value is written in terms of, in one order.

    Taken across all of a graph's captures rather than per capture, and in a
    name order rather than an arrival order, because these are the arguments a
    kernel is called with: a kernel's signature has to be the same whichever
    order the values arrived in, or the same graph captured twice would produce
    two kernels.
    """

    symbols: dict = {}
    for buffers in buffer_groups:
        for buffer in buffers:
            if isinstance(buffer, sympy.Expr):
                for symbol in sorted(buffer.free_symbols, key=lambda s: s.name):
                    symbols.setdefault(symbol, None)
    return tuple(symbols)


# ---------------------------------------------------------------------------
# What a graph is made of
# ---------------------------------------------------------------------------


def input_buffers_require_grads(graph_module: Any, num_score_mod_placeholders: int) -> bool:
    """Whether any input beyond the score's own placeholders needs a gradient.

    The placeholders the score was given are the score's business; what matters
    here is whether the values the kernel was called with are ones a backward
    pass has to be able to reach.
    """

    inputs = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    if len(inputs) <= num_score_mod_placeholders:
        return False

    def requires_grad(n: Any) -> bool:
        tensor_meta = n.meta.get("tensor_meta")
        return tensor_meta.requires_grad if tensor_meta is not None else False

    return any(requires_grad(n) for n in inputs[num_score_mod_placeholders:])


def is_trivial_score_graph(graph_module: Any) -> bool:
    """Whether the score is just the score, passed through.

    Which is the case the backward pass can be written for: a score that is
    more than itself changes the gradient in a way the backward pass does not
    yet account for, and recognising it here is what keeps that from being
    discovered as a wrong gradient rather than as an unsupported score.
    """

    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    if len(output) != 1:
        raise AssertionError("Got graph w/ multiple outputs")
    output_val = output[0].args[0]
    return output_val == placeholders[0]


def is_trivial_mask_graph(graph_module: Any) -> bool:
    """Whether the mask is the one that keeps everything.

    A mask that keeps everything is not a mask, and recognising that is what
    lets the kernel take the path that has no mask in it at all.
    """

    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    if len(output) != 1:
        raise AssertionError("Got graph w/ multiple outputs")
    output_val = output[0].args[0]
    return len(placeholders) == 4 and output_val.target is tp.ops.tp.full.default


def has_unsupported_cpu_scalar_tensor_captures(
    score_mod_other_buffers: Any, mask_mod_other_buffers: Any
) -> bool:
    """Whether a captured value is a lone number held on the host.

    A kernel is handed numbers, not the values that happen to contain one, so a
    capture that is a single value on the host has to be turned into a number
    before it can be one -- and until it has been, the kernel cannot be told
    what it would have been handed.
    """

    from ...ir import TensorBox

    for buf in list(score_mod_other_buffers) + list(mask_mod_other_buffers):
        if isinstance(buf, TensorBox):
            device = buf.get_device()
            size = buf.get_size()
            if device is not None and getattr(device, "type", None) == "cpu" and len(size) == 0:
                return True
    return False
