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
import math
from typing import Any

import tensorplay as tp
import sympy
from sympy import Mod

from ..ir import (
    ExternKernel,
    FixedLayout,
    FlexibleLayout,
    InputBuffer,
    IRNode,
    TensorBox,
)
from ..loops import V, get_fill_order
from tensorplay.utils._pytree import tree_map, tree_map_only
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


# ---------------------------------------------------------------------------
# Shapes these kernels assume
# ---------------------------------------------------------------------------


def is_power_of_2(n: Any) -> bool:
    """Whether a number is a power of two.

    A power of two is one bit set, so a number that is one has exactly one bit
    that is not in the number below it.  Zero is not one however it is
    written, and is the one value where the test would otherwise say yes.
    """

    return n != 0 and ((n & (n - 1)) == 0)


def next_power_of_two(n: Any) -> Any:
    """The smallest power of two at least as large.

    A kernel that processes a number of elements at a time wants that number to
    be one it can halve, because halving is how a range of work is split.  A
    length that is not a power of two is rounded up rather than down, because
    rounding down would leave positions with nobody to process them.
    """

    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


def set_head_dim_values(
    kernel_options: Any, qk_head_dim: Any, v_head_dim: Any, graph_sizevars: Any
) -> None:
    """Record the two head sizes a kernel is written for, and the rounded forms.

    Two sizes rather than one, because the size the scores are computed at and
    the size the values are combined at are not always the same, and a kernel
    written for one of them cannot be used for the other.  Recorded rather than
    read from the graph because the kernel is written before the graph is read
    and has to agree with it.

    The rounded forms are recorded alongside: a kernel works in whole groups of
    a power of two, and a size that is not one is padded up to the next, which
    is why the padding is the kernel's business and not the caller's.
    """

    qk_head_dim_static = graph_sizevars.guard_int(qk_head_dim)
    kernel_options.setdefault("QK_HEAD_DIM", qk_head_dim_static)
    kernel_options.setdefault(
        "QK_HEAD_DIM_ROUNDED", next_power_of_two(qk_head_dim_static)
    )

    v_head_dim_static = graph_sizevars.guard_int(v_head_dim)
    kernel_options.setdefault("V_HEAD_DIM", v_head_dim_static)
    kernel_options.setdefault(
        "V_HEAD_DIM_ROUNDED", next_power_of_two(v_head_dim_static)
    )

    # Whether either size is already a whole number of groups.  Recorded
    # because a kernel can take a shorter path when both are, and a kernel
    # that has to pad is a different kernel from one that does not -- so
    # whether it does is part of what the kernel is.
    kernel_options.setdefault(
        "SAFE_HEAD_DIM",
        is_power_of_2(qk_head_dim_static) and is_power_of_2(v_head_dim_static),
    )


def can_skip_boundary_checks(seq_len: Any, sparse_block_size: Any) -> bool:
    """Whether an axis divides into whole tiles, so no tile runs off its end.

    Asked before a configuration is chosen, and therefore against the largest
    tile a candidate might use rather than against whichever one is chosen:
    a kernel that skips the check must skip it for every shape it will see, not
    for the one that happens to be measured first.
    """

    return V.graph.sizevars.statically_known_true(
        sympy.And(
            sympy.Eq(Mod(seq_len, 128), 0),
            sympy.Or(
                sympy.Eq(Mod(seq_len, sparse_block_size), 0),
                sympy.Ge(sparse_block_size, seq_len),
            ),
        )
    )


def is_tensor_ir_node(node: Any) -> bool:
    """Whether a node is a value rather than a number.

    The two are told apart here because a value and a number are both things a
    graph holds, and a list of them has to be taken apart before it can be
    passed on -- which is only possible if which is which is known.
    """

    return isinstance(node, IRNode) and node.has_tensor_output()


def contiguous_last_dim(x: Any) -> Any:
    """A value whose innermost axis has no gaps between its elements.

    Asked for by a kernel that reads along that axis one element after another,
    which is only the same thing as walking positions when the elements are
    adjacent.  Reordered rather than copied, because a copy here would be a
    copy of the whole value to fix a property of one axis of it.
    """

    strides = x.maybe_get_stride()
    if strides and strides[-1] != 1:
        contiguous_stride_order = list(reversed(range(len(x.get_size()))))
        return ExternKernel.require_stride_order(x, contiguous_stride_order)
    return x


def maybe_realize(args: Any) -> Any:
    """Write down every value in a list that has not been written down yet.

    Taken one at a time and asked of each, because what a kernel is handed is
    a list of things of different kinds -- values, and numbers that stand for
    shapes -- and only some of them are things that can be written down.
    """

    from ..op_lowerings import realize_inputs

    return tree_map(
        lambda x: (
            realize_inputs(x) if x is not None and not isinstance(x, sympy.Expr) else x
        ),
        args,
    )


def freeze_irnodes(tree: Any) -> Any:
    """Stop every value in a tree from being written anywhere else.

    A kernel is handed values and decides for itself where they live, so a
    value whose layout could still change after it was handed would be a value
    the kernel and the graph disagree about.  A value that cannot be frozen is
    one that is not a value -- a number standing for a shape -- and is left
    alone rather than refused.
    """

    if tree is None:
        return None

    def _freeze(node: Any) -> Any:
        try:
            node.freeze_layout()
        except (NotImplementedError, AttributeError):
            pass
        return node

    return tree_map_only(IRNode, _freeze, tree)


def create_placeholder(
    name: str, dtype: Any, device: Any, size: Any = None
) -> Any:
    """A value the kernel is handed that nothing has produced yet.

    A kernel is written against a signature, and the values in it are what the
    kernel reads; the ones nothing has produced are the arguments, and they
    are given a shape here so that the kernel can be written before anything
    has been computed.
    """

    input_buffer = InputBuffer(
        name=name,
        layout=FixedLayout(
            device,
            dtype,
            size if size else [],
            FlexibleLayout.contiguous_strides(size) if size else [],
        ),
    )
    return TensorBox.create(input_buffer)


def construct_strides(sizes: Any, fill_order: Any) -> Any:
    """The strides a shape has when its axes are filled in a given order.

    Filled innermost-first because that is what makes the result dense: the
    axis filled first has the smallest stride, and every axis after it moves
    further.  The order is given rather than inferred because a kernel needs
    the layout its own reads assume, and which one that is is a property of
    how it reads rather than of the value it reads.
    """

    if len(sizes) != len(fill_order):
        raise AssertionError("Length of sizes must match the length of the fill order")
    strides = [0] * len(sizes)
    current_stride: Any = 1
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride = current_stride * sizes[dim]
    return strides


def infer_dense_strides(size: Any, orig_strides: Any) -> Any:
    """Dense strides that keep the layout the value already had.

    A value is read in the order its layout says, and reordering the axes would
    make every read wrong -- so the layout is kept and only the gaps are
    removed.  The innermost axis is made adjacent whatever the layout said,
    because these kernels read along it one element after another, and that is
    only walking positions when the elements are next to each other.
    """

    fill_order = get_fill_order(orig_strides, V.graph.sizevars.shape_env)
    strides = construct_strides(size, fill_order)

    if strides[-1] != 1:
        last_dim = len(size) - 1
        fill_order = list(fill_order)
        fill_order.remove(last_dim)
        fill_order = [last_dim] + fill_order
        strides = construct_strides(size, fill_order)

    return strides


def get_fwd_subgraph_outputs(subgraph_buffer: Any, mask_graph_buffer: Any) -> Any:
    """What the forward pass produces: the score's outputs, then the mask's.

    In that order because the mask is applied to what the score produced, and a
    caller that took them the other way round would be applying a mask to
    something that has not been computed.
    """

    subgraph_buffer = (
        subgraph_buffer if isinstance(subgraph_buffer, (list, tuple)) else [subgraph_buffer]
    )
    mask_graph_buffer = (
        mask_graph_buffer if isinstance(mask_graph_buffer, (list, tuple)) else [mask_graph_buffer]
    )
    return [*subgraph_buffer, *mask_graph_buffer]


def create_indices_fake(x: Any) -> Any:
    """A stand-in for an index, for measuring a kernel with.

    Every position named, rather than the first one or none: an index that
    named only some positions would make the kernel look cheaper than it is,
    because the work of following an index is not the same as the work of
    reading a position.
    """

    size = V.graph.sizevars.optimization_hints(x.get_size())
    indices = tp.arange(0, size[-1], dtype=x.get_dtype(), device=x.get_device())
    indices = indices.expand(size).contiguous()
    return indices


def create_num_blocks_fake_generator(sparse_indices: Any) -> Any:
    """A stand-in for a count of blocks, for measuring a kernel with.

    A count has to be one the kernel would really do that much work for, or the
    measurement is of a different kernel than the one that will run.  A count
    of no blocks would measure a kernel that reads nothing; a count of every
    block would measure a kernel that takes far longer than any real one would,
    for no better answer.  So a count in between: enough that reading ahead
    would help if it were going to, few enough that measuring is quick.
    """

    def create_num_blocks_fake(x: Any) -> Any:
        num_blocks_for_autotuning = V.graph.sizevars.optimization_hint(
            sparse_indices.shape[-1]
        )
        size = V.graph.sizevars.optimization_hints(x.get_size())
        return tp.full(
            size,
            num_blocks_for_autotuning,
            dtype=x.get_dtype(),
            device=x.get_device(),
        )

    return create_num_blocks_fake
