"""Attention that reads its mask a position at a time.

Two ways of writing attention live here and in the file beside this one.  The
difference is how a mask is read.  A mask is a predicate over positions -- which
queries may see which keys -- and a device that can compare several positions at
once should be handed the ranges rather than the comparisons, which is what the
kernels next door do and what makes them fast on a device that can.  A device
that cannot is handed the comparisons themselves, one lane at a time, and that
is slower and runs anywhere.

So the two are not a fast path and a slow path of one thing.  They are the same
computation written for two different capabilities, and which one is right is a
fact about the device rather than a tuning decision.  This one does not need a
particular device; the one beside it does, and says so rather than falling back.

The mask is not the only thing a program can change about attention.  A score can
be changed too -- by adding a bias, by discounting by distance -- and both are
programs rather than numbers, so both are captured and carried into the kernel
as something to run per position.  A kernel is therefore written per pair of
captures, and the widths those captures are evaluated at are choices worth
measuring rather than fixing here.
"""

from __future__ import annotations

from typing import Any

import math

import sympy

from tensorplay.graph.experimental.sympy_functions import FloorDiv

import tensorplay as tp

from ..heuristics.template.base import SymbolicGridFn
from ....._higher_order_ops.omni_attention import (
    omni_attention as omni_attention_hop,
    omni_attention_backward as omni_attention_backward_hop,
)
from ..ir import ComputedBuffer
from ..loops import V
from ..op_lowerings import register_lowering
from ..runtime.runtime_utils import ceildiv
from ..templates.mm_common import load_kernel_template
from ..templates.select_algorithm import TritonTemplate
from .omni_flash_attention import (
    _omni_kernel_options_example,
    _omni_kernel_tuning_options,
    build_subgraph_buffer,
    create_placeholder,
    freeze_irnodes,
    has_unsupported_cpu_scalar_tensor_captures,
    realize_captures_for_cutedsl,
)

#: What capturing a score or a mask produced: a body to run, or several of them
#: because the mask is applied after the score and so is a second thing, or
#: nothing at all because the program did not change either.
#:
#: A list rather than a fixed pair, because a program may change only the score,
#: only the mask, or both, and how many bodies that comes to is a fact about the
#: program rather than something the caller gets to decide.
SubgraphResults = list[ComputedBuffer | None] | ComputedBuffer | None

#: Which of the ways of writing attention a program asked for.
#:
#: Not an enum, because the value travels as a string in the options a program
#: wrote, and turning it into something closed here would mean a program naming
#: a way this does not know about is refused rather than answered.
_BACKEND_AUTO = "AUTO"
_BACKEND_TRITON = "TRITON"
_BACKEND_FLASH = "FLASH"
_BACKEND_TRITON_DECODE = "TRITON_DECODE"


# ---------------------------------------------------------------------------
# Taking the backward pass apart
# ---------------------------------------------------------------------------


class JointOutputResult:
    """What the backward pass produced, taken apart into the parts a kernel wants.

    Four things rather than one, because the kernel wants them apart.  The
    gradient of the query is the one value the kernel computes; the gradients of
    everything the caller captured are values it reads, and are read as whatever
    they were rather than as gradients; a gradient that was computed rather than
    read is a different thing from one that was written into, because only the
    second is somewhere the kernel can add to.
    """

    grad_input: Any
    captured_grads_compute: list
    captured_grads: list
    mutated_grads: list


def process_joint_outputs(all_joint_outputs: Any, num_placeholders: int) -> Any:
    """Take the backward pass's own outputs and hand back what a kernel can read.

    A caller can capture a value in a score, and then that value has a gradient.
    Whether the backward pass computed that gradient or wrote it into something
    it already had is not something the kernel should have to know: a gradient it
    reads is a value, and one it is given as somewhere to add to is an
    accumulator.  So the two are told apart here, where the backward pass's own
    answer can still be read.

    A captured value with no gradient is not in the list at all, rather than in
    it as nothing -- because a kernel handed nothing for it would be a kernel
    with an argument it cannot use, and the caller did not ask for one.
    """

    from ..ir import ComputedBuffer, TensorBox

    # The first outputs are the ones the kernel produced; the rest are the
    # gradients of what the caller captured, one per captured value, and nothing
    # for a captured value that does not need one.
    grad_input = all_joint_outputs[0]
    if not isinstance(grad_input, ComputedBuffer):
        raise AssertionError(
            f"Expected ComputedBuffer for the query's gradient, got {type(grad_input)}"
        )
    if grad_input.name is None:
        raise AssertionError("ComputedBuffer name must not be None")

    other_grads = all_joint_outputs[num_placeholders:]

    grads_compute = [buf for buf in other_grads if buf is not None]

    def get_out(buf: Any) -> Any:
        if buf is None:
            return None
        if not isinstance(buf, ComputedBuffer):
            raise AssertionError(f"Expected ComputedBuffer, got {type(buf)}")
        if buf.name is None:
            raise AssertionError("ComputedBuffer name must not be None")
        return TensorBox.create(V.graph.get_buffer(buf.name))

    grads_out = [get_out(x) for x in other_grads]
    mutated_grads = [buf for buf in grads_out if buf is not None]

    return JointOutputResult(
        grad_input=grad_input,
        captured_grads_compute=grads_compute,
        captured_grads=grads_out,
        mutated_grads=mutated_grads,
    )


def get_bwd_subgraph_outputs(
    subgraph_buffer: SubgraphResults,
    mask_graph_buffer: SubgraphResults,
    joint_outputs: Any,
) -> list:
    """Everything the backward pass's kernel reads, in the order it reads them.

    The score's body, then the mask's -- the mask is applied to what the score
    produced, so a caller that took them the other way round would be applying a
    mask to something not yet computed.  Then the gradient it computes, then the
    gradients it reads, then the ones it adds to.  Those last two are separate
    because they are different things: one is a value the kernel reads and one is
    somewhere the kernel writes, and a list that mixed them would say neither.
    """

    from collections.abc import Sequence

    subgraph_buffer = (
        subgraph_buffer if isinstance(subgraph_buffer, Sequence) else [subgraph_buffer]
    )
    mask_graph_buffer = (
        mask_graph_buffer
        if isinstance(mask_graph_buffer, Sequence)
        else [mask_graph_buffer]
    )
    joint_output_buffers = [
        joint_outputs.grad_input,
        *joint_outputs.captured_grads_compute,
        *joint_outputs.captured_grads,
        *joint_outputs.mutated_grads,
    ]

    return [*subgraph_buffer, *mask_graph_buffer, *joint_output_buffers]


# ---------------------------------------------------------------------------
# What a program asked for
# ---------------------------------------------------------------------------


def sanitize_kernel_options_for_triton(kernel_options: Any) -> Any:
    """Take the backend out of the options, and hand back which one it was.

    The backend is not an option a kernel can be handed: it says which kernel to
    hand it to, so it is answered here and not carried the rest of the way.  The
    rest is a copy rather than the original, because a program that is asked
    twice for the same thing should get the same answer both times, and a default
    written into its own dictionary would not.
    """

    sanitized = dict(kernel_options)
    backend = sanitized.pop("BACKEND", _BACKEND_AUTO)
    return sanitized, backend


def get_float32_precision() -> str:
    """How a product of 32-bit floats is to be carried out, as a kernel says it.

    Answered as the text a kernel is written with rather than as a setting,
    because that is what the kernel takes.

    Exact unless the device has been told to trade accuracy for speed.  There are
    two ways to have told it, one of which supersedes the other, and which is
    which is not something this decides by asking which is set -- it is a fact
    about the program, not about the current state.  So the older answer is used
    when the newer one is not there, and the newer one is a plain yes or no
    about trading accuracy away, which is what it has always meant here.

    A device whose products are not carried out by the same units as everything
    else answers exact regardless, because there is nothing there to trade
    accuracy for speed with.
    """

    import tensorplay as tp

    newer = getattr(tp.backends.cuda.matmul, "fp32_precision", None)
    if newer is not None and newer != "none":
        exact = newer == "ieee"
    elif getattr(tp.backends.cuda.matmul, "allow_tf32", None) is not None:
        exact = not tp.backends.cuda.matmul.allow_tf32
    else:
        exact = tp.get_float32_matmul_precision() == "highest"

    if exact or tp.version.hip or (hasattr(tp, "mtia") and tp.mtia.is_available()):
        return "'ieee'"
    else:
        return "'tf32'"


def raise_omni_kernel_options_error(
    kernel_name: str,
    kernel_options: Any,
    option_names: Any,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> None:
    """Say which options cannot both be true, and what would be.

    A tile has to divide the block it walks, so two options that name a tile
    and a block can be given that do not.  Which two, and what the numbers were,
    is the whole of what is wrong -- so all of it is said, along with a set that
    would have worked, because a caller who cannot see the two that clash cannot
    change either of them.
    """

    option_values = ", ".join(f"{name}={kernel_options[name]}" for name in option_names)
    raise ValueError(
        f"Invalid attention {kernel_name} kernel options: Q and KV block sizes "
        f"must be divisible by the selected tile sizes. Got "
        f"SPARSE_Q_BLOCK_SIZE={sparse_q_block_size}, "
        f"SPARSE_KV_BLOCK_SIZE={sparse_kv_block_size}, and {option_values}. "
        f"Pass compatible values with kernel_options. Available {kernel_name} "
        f"tuning options are {_omni_kernel_tuning_options(kernel_name)}. For example: "
        f"{_omni_kernel_options_example(kernel_name)}. If you did not pin "
        f"these options, and the default choice errors, compiling with "
        f"mode='max-autotune-no-cudagraphs' can also fix this by trying more "
        f"attention configs."
    )


def check_flash_supported_scalar_captures(
    score_mod_other_buffers: Any,
    mask_mod_other_buffers: Any,
    *,
    backward: bool = False,
) -> None:
    """Refuse a capture the flash kernels cannot be handed, before they are built.

    Said here rather than where it is discovered because the discovering is
    inside a kernel body: by then the answer is a kernel that failed to write
    itself, and the reason is a sentence about a device rather than about the
    program that asked for something this device cannot do.
    """

    if has_unsupported_cpu_scalar_tensor_captures(
        score_mod_other_buffers, mask_mod_other_buffers
    ):
        direction = " backward" if backward else ""
        raise RuntimeError(
            f"BACKEND='FLASH' but flash attention{direction} cannot be used: "
            "NYI: score_mod or mask_mod captures a 0-dim CPU tensor scalar. "
            "Workarounds: use BACKEND='TRITON' or pass the value as a tensor "
            "on device instead of capturing a CPU scalar tensor."
        )


# ---------------------------------------------------------------------------
# Taking a call apart before a kernel is chosen
# ---------------------------------------------------------------------------


def check_embedding_is_wide_enough(query: Any, value: Any) -> None:
    """Refuse an embedding too narrow for the product the kernels do.

    A product of two very narrow rows is not a smaller version of a wider one:
    below a width the device's units have nothing to work with, and the shape of
    the answer stops being the shape of a product and becomes a sum over whatever
    fitted.  So this is refused with both widths named rather than computed
    differently from the wide case, because a program that got a different
    answer for a narrower embedding would have no way of knowing.
    """

    small_dqk = V.graph.sizevars.evaluate_expr(
        sympy.Lt(query.get_size()[-1], 16)
    )
    small_dv = V.graph.sizevars.evaluate_expr(sympy.Lt(value.get_size()[-1], 16))
    if small_dqk or small_dv:
        raise NotImplementedError(
            f"NYI: embedding dimension of the query, key, and value must be "
            f"at least 16 but got E={query.get_size()[-1]} and Ev={value.get_size()[-1]}"
        )


def unpack_block_mask(block_mask: Any) -> Any:
    """The seventeen things a mask carries, each under its own name.

    A mask is not one thing.  It says how long each axis is, which blocks of
    keys exist and which do not, which of those are full rather than partial,
    the same four things again for the other direction because the backward pass
    walks queries as blocks, an order to write in so the backward pass does not
    depend on scheduling, and the two block widths everything above is counted
    in.  Sixteen of those are values and the seventeenth is the program that
    decides which blocks are allowed.

    They are named rather than indexed because there are seventeen and a kernel
    handed the wrong one is wrong rather than broken -- and a mask that gained a
    field would otherwise be unpacked into the names of the wrong fields.
    """

    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        _,  # dq_write_order (backward-only)
        _,  # dq_write_order_full (backward-only)
        _,  # dq_kv_order (backward-only)
        _,  # dq_kv_order_spt (backward-only)
        sparse_q_block_size,
        sparse_kv_block_size,
        mask_graph,
    ) = block_mask

    return {
        "kv_num_blocks": kv_num_blocks,
        "kv_indices": kv_indices,
        "full_kv_num_blocks": full_kv_num_blocks,
        "full_kv_indices": full_kv_indices,
        "q_num_blocks": q_num_blocks,
        "q_indices": q_indices,
        "full_q_num_blocks": full_q_num_blocks,
        "full_q_indices": full_q_indices,
        "sparse_q_block_size": sparse_q_block_size,
        "sparse_kv_block_size": sparse_kv_block_size,
        "mask_graph": mask_graph,
    }


def capture_score_and_mask(
    query: Any,
    subgraph: Any,
    mask_graph: Any,
    score_mod_other_buffers: Any,
    mask_mod_other_buffers: Any,
) -> Any:
    """Turn the two programs a caller passed into two bodies a kernel can run.

    A caller changes attention by passing a function, not a number: a score can
    be biased, a mask can depend on distance, and both are programs.  A kernel
    cannot take a function, so each is captured -- recorded as the sequence of
    operations it performs -- and the sequence becomes a body the kernel runs
    per position.

    The score's body is handed the value it produces as well as the four numbers
    that say which position it is at, because a score is a function of the
    position and of nothing else.  The mask's body is handed the four numbers
    and not the score: a mask is applied to a score that has already been
    produced, and a body that could see the score would be able to change it,
    which is a different thing from masking it.
    """

    # Which of the two bodies is written at all is a fact about the program: a
    # caller who changed nothing has no body to run, and one who changed only the
    # score has one.  Building a body for a program that is the identity would be
    # a kernel paying for a comparison it was told to make against itself.
    if subgraph is not None:
        placeholder_inps = [
            create_placeholder(name, dtype, query.get_device())
            for name, dtype in [
                ("score", query.get_dtype()),
                ("b", tp.int32),
                ("h", tp.int32),
                ("m", tp.int32),
                ("n", tp.int32),
            ]
        ]
        subgraph_buffer: SubgraphResults = build_subgraph_buffer(
            placeholder_inps + list(score_mod_other_buffers), subgraph
        )
        freeze_irnodes(subgraph_buffer)
    else:
        subgraph_buffer = None

    if mask_graph is not None:
        mask_graph_placeholder_inps = [
            create_placeholder(name, dtype, query.get_device())
            for name, dtype in [
                ("b", tp.int32),
                ("h", tp.int32),
                ("m", tp.int32),
                ("n", tp.int32),
            ]
        ]
        mask_graph_buffer: SubgraphResults = build_subgraph_buffer(
            mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
        )
        freeze_irnodes(mask_graph_buffer)
    else:
        mask_graph_buffer = None

    return subgraph_buffer, mask_graph_buffer


def guard_kernel_options(kernel_options: Any) -> Any:
    """Pin the sizes a program named, and say how wide a product is to be.

    A size a program wrote as a symbol is not yet a number, and a kernel cannot
    be written against a symbol -- it is written against the number the symbol
    will be.  So each one is pinned here, and pinning is what makes the pinning
    safe: a program whose symbol turns out to be two different numbers in two
    places gets two kernels rather than one kernel written for neither.

    The width of a product of 32-bit floats is added rather than asked for,
    because a program that set it has said something and a program that did not
    has said nothing -- and nothing is not the same as the default.
    """

    guarded = {
        k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v
        for k, v in kernel_options.items()
    }
    guarded.setdefault("FLOAT32_PRECISION", get_float32_precision())
    return guarded


def heads_are_grouped(query: Any, key: Any) -> bool:
    """Whether the query has more heads than the key, so the two are shared.

    A question about the shapes and not about the values, and asked as one
    because every kernel that cares has to answer it the same way: whether the
    query's heads are in groups over the key's, or one to each.
    """

    return V.graph.sizevars.evaluate_expr(
        sympy.Ne(query.get_size()[1], key.get_size()[1]),
    )


# ---------------------------------------------------------------------------
# How the work is spread over the device
# ---------------------------------------------------------------------------


@SymbolicGridFn
def omni_attention_grid(
    batch_size: Any, q_heads: Any, num_queries: Any, d_model: Any, meta: Any, *, cdiv: Any
) -> Any:
    """One program per query tile, per head, per batch.

    The keys are not an axis of the launch: one program walks all of them, which
    is what lets the running total it accumulates carry from one to the next
    without anything having to be written out and read back between them.  The
    query tile is the only thing that has to be a separate program, because a
    query's answer does not depend on any other query's.
    """

    return (cdiv(num_queries, meta["BLOCK_M"]), batch_size, q_heads)


@SymbolicGridFn
def omni_attention_backward_grid(
    batch_size: Any,
    q_heads: Any,
    num_queries: Any,
    d_model: Any,
    kv_heads: Any,
    num_key_value: Any,
    meta: Any,
    *,
    cdiv: Any,
) -> Any:
    """One program per query tile, plus one per key tile, per head and batch.

    Two kinds of gradient come out of this pass and they are not the same shape
    of work.  A query's gradient belongs to that query alone, so a query tile is
    a program.  A key's gradient is touched by every query that may see that key,
    so it is summed from many places and cannot be finished by whichever program
    arrives last -- which is why the key axis is a separate set of programs
    rather than something folded into the query one.

    The query tiles are counted per query head and then divided by the number of
    key heads, so that a head sharing one key head's keys is not given a whole
    set of programs for the same work.
    """

    return (
        cdiv(num_queries, meta["BLOCK_M2"]) * (q_heads // kv_heads)
        + cdiv(num_key_value, meta["BLOCK_N1"]),
        batch_size,
        kv_heads,
    )


@SymbolicGridFn
def omni_decoding_grid(
    batch_size: Any, kv_heads: Any, gqa_group_size: Any, seq_len_q: Any, d_model: Any, meta: Any
) -> Any:
    """One program per query tile, per head and batch, times the split of the keys.

    Decoding asks one question at a time, so the query axis is tiny and the key
    axis is everything.  The keys are split across programs and the partial
    answers are combined afterwards, because one program walking every key for
    one query leaves the device idle for everything except that one query.
    """

    block_m = meta["BLOCK_M"]
    num_block_m = ceildiv(seq_len_q * gqa_group_size, block_m)
    return (num_block_m, batch_size * kv_heads, meta["SPLIT_KV"])


# ---------------------------------------------------------------------------
# The bodies these are written from
# ---------------------------------------------------------------------------

#: A body is a file rather than a string, and a body that shares its bookkeeping
#: with another is that other body concatenated on.  Which is which is not a
#: detail of this file: a body rendered without the one it shares its
#: bookkeeping with is a body that refers to names nothing has defined, so the
#: three sources below name what each is made of rather than reading as one name
#: each.
#:
#: Forward: the kernel, then the helpers it and the backward both use, then the
#: helpers only it uses.
omni_attention_source = (
    load_kernel_template("omni_attention")
    + load_kernel_template("omni_utilities")
    + load_kernel_template("omni_common")
)
#: Backward: the kernel, then the helpers.  The forward-only ones are not here
#: because nothing in the backward refers to them.
omni_attention_backward_source = load_kernel_template("omni_backwards") + load_kernel_template(
    "omni_utilities"
)
#: Decoding: a kernel that asks one question at a time, then the same two sets
#: of helpers the forward uses -- it is a forward with the query axis short.
omni_decoding_source = (
    load_kernel_template("omni_decode")
    + load_kernel_template("omni_utilities")
    + load_kernel_template("omni_common")
)

#: The three ways of writing attention that run anywhere.
#:
#: The layout is frozen in each case, which is what says these kernels address
#: positions by how they were laid out rather than by how they would be laid out
#: if they were free: a program that computed a stride the caller had not agreed
#: to would write its answer where nobody reads it.
OMNI_ATTENTION = TritonTemplate(
    name="omni_attention",
    grid=omni_attention_grid,
    source=omni_attention_source,
    always_freeze_layout=True,
)

OMNI_ATTENTION_BACKWARD = TritonTemplate(
    name="omni_attention_backward",
    grid=omni_attention_backward_grid,
    source=omni_attention_backward_source,
    always_freeze_layout=True,
)

OMNI_DECODING = TritonTemplate(
    name="omni_decoding",
    grid=omni_decoding_grid,
    source=omni_decoding_source,
    always_freeze_layout=True,
)


def create_omni_attention_kernel(
    query: Any,
    key: Any,
    value: Any,
    scale: float,
    kernel_options: Any,
    subgraph_buffer: SubgraphResults,
    mask_graph_buffer: SubgraphResults,
    score_mod_other_buffers: Any,
    mask_mod_other_buffers: Any,
    mask_parts: Any,
) -> Any:
    """Write the pass for a query axis of any length.

    Three values rather than one.  The answer is the first.  The second is the
    total the answer was accumulated against, and a later pass needs it to work
    out which rows came from where.  The third is the largest value any key in a
    row produced, which the answer does not need but the second does: the total
    is kept relative to a largest value so that adding it to another row's total
    does not lose the small terms, and a largest value is what it is relative
    to.  So the third is written out rather than recovered, because by the time a
    later pass runs, the running total has already been reduced against it.

    The two of the three that are not the answer are at full width whatever the
    values' own width, because they are added to and compared against each other
    across rows.
    """

    from .omni_flash_attention import (
        can_skip_boundary_checks,
        create_indices_fake,
        create_num_blocks_fake_generator,
        freeze_irnodes,
        get_fwd_subgraph_outputs,
        is_power_of_2,
        is_tensor_ir_node,
        maybe_realize,
        set_head_dim_values,
    )
    from ..op_lowerings import empty_strided
    from ..utils import can_use_tma

    kv_num_blocks = mask_parts["kv_num_blocks"]
    kv_indices = mask_parts["kv_indices"]
    full_kv_num_blocks = mask_parts["full_kv_num_blocks"]
    full_kv_indices = mask_parts["full_kv_indices"]
    sparse_q_block_size = mask_parts["sparse_q_block_size"]
    sparse_kv_block_size = mask_parts["sparse_kv_block_size"]

    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)

    bq, hq, seq_len_q, qk_head_dim = query.get_size()
    bkv, hkv, seq_len_kv, v_head_dim = value.get_size()
    if not V.graph.sizevars.evaluate_expr(sympy.Eq(bq, bkv) | sympy.Eq(bkv, 1)):
        raise AssertionError(
            f"Bq and Bkv must be broadcastable. Got Bq={bq} and Bkv={bkv}"
        )
    if not V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_q, 0)):
        raise AssertionError("Query length must be greater than 0")
    if not V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_kv, 0)):
        raise AssertionError("Key length must be greater than 0")

    b = bq

    # A row or column that is not a whole number of tiles has to be masked off on
    # the last one, and when both divide there is nothing to mask.
    seq_q_divisible = can_skip_boundary_checks(seq_len_q, sparse_q_block_size)
    seq_kv_divisible = can_skip_boundary_checks(seq_len_kv, sparse_kv_block_size)
    kernel_options.setdefault(
        "IS_DIVISIBLE", bool(seq_q_divisible and seq_kv_divisible)
    )

    # The answer is read the way the question was laid out, so its distances
    # follow the question's.  The embedding is the one axis that may differ: it
    # is the values that are being read, not the positions.
    q_strides = query.get_stride()
    out_size = [b, hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, q_strides)

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [b, hq, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in out_strides],
    )
    logsumexp_shape = [b, hq, seq_len_q]
    logsumexp = empty_strided(
        logsumexp_shape,
        None,
        dtype=tp.float32,
        device=query.get_device(),
    )
    max_scores = empty_strided(
        logsumexp_shape,
        None,
        dtype=tp.float32,
        device=query.get_device(),
    )
    kernel_options.setdefault("SM_SCALE", scale)

    gqa_shared_heads = FloorDiv(hq, hkv)
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # A block entirely inside the mask needs only the score applied to it, and
    # asking whether a position is inside the mask is a comparison per position.
    # So a mask that has both kinds of block says which are which.
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        full_kv_num_blocks, full_kv_indices = (
            empty_strided([0], None, dtype=query.get_dtype(), device=query.get_device())
            for _ in range(2)
        )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    choices: list = []

    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.guard_int(query.get_size()[-1])
    configs = V.choices.get_omni_attention_fwd_configs(
        head_dim, seq_len_q, dtype, query.get_device().type
    )

    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)
    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)

    original_kernel_options = kernel_options.copy()
    invalid_block_options: Any = None

    for conf in configs:
        cur_kernel_options = original_kernel_options.copy()
        # The prefix says which pass an option is for, and a kernel is only ever
        # given options for its own pass.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("fwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("bwd_"):
                cur_kernel_options.pop(k)
        cur_kernel_options.setdefault("num_stages", conf.num_stages)
        cur_kernel_options.setdefault("num_warps", conf.num_warps)

        cur_kernel_options.setdefault("USE_TMA", False)
        if cur_kernel_options["USE_TMA"] and not can_use_tma(query, key, value):
            cur_kernel_options["USE_TMA"] = False

        # A tile wider than the block it walks is wasted lanes and, worse, a
        # boundary mask paid for on every step.  Narrowed only when there is
        # exactly one candidate and both block sizes are whole powers of two --
        # a program that pinned the tile keeps it, and gets told if it does not
        # divide.
        block_m, block_n = conf.block_m, conf.block_n
        if len(configs) == 1 and all(
            is_power_of_2(s) and s >= 16
            for s in (sparse_q_block_size, sparse_kv_block_size)
        ):
            block_m = min(block_m, sparse_q_block_size)
            block_n = min(block_n, sparse_kv_block_size)
        cur_kernel_options.setdefault("BLOCK_M", block_m)
        cur_kernel_options.setdefault("BLOCK_N", block_n)
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", sparse_q_block_size)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size)

        if (
            cur_kernel_options["SPARSE_KV_BLOCK_SIZE"] % cur_kernel_options["BLOCK_N"]
            != 0
            or cur_kernel_options["SPARSE_Q_BLOCK_SIZE"] % cur_kernel_options["BLOCK_M"]
            != 0
        ):
            invalid_block_options = cur_kernel_options
            if len(configs) == 1:
                raise_omni_kernel_options_error(
                    "forward",
                    cur_kernel_options,
                    ("BLOCK_M", "BLOCK_N"),
                    sparse_q_block_size,
                    sparse_kv_block_size,
                )
            continue

        for attrib in ("kpack", "matrix_instr_nonkdim", "waves_per_eu"):
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)

        error = OMNI_ATTENTION.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                logsumexp,
                max_scores,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
            ],
            layout=layout,
            subgraphs=[
                subgraph_buffer,
                mask_graph_buffer,
            ],
            mutated_inputs=[
                logsumexp,
                max_scores,
            ],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )
        if error is not None and len(configs) == 1:
            raise error

    choices = V.choices.append_omni_attention_choices(
        choices,
        configs,
        [
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ],
        [subgraph_buffer, mask_graph_buffer],
        layout,
        original_kernel_options,
        sparse_q_block_size,
        sparse_kv_block_size,
    )

    if not choices and invalid_block_options is not None:
        raise_omni_kernel_options_error(
            "forward",
            invalid_block_options,
            ("BLOCK_M", "BLOCK_N"),
            sparse_q_block_size,
            sparse_kv_block_size,
        )

    inputs_for_autotuning = (
        [
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
    )
    input_gen_fns = {
        5: create_num_blocks_fake_generator(kv_indices),
        6: create_indices_fake,
        7: create_num_blocks_fake_generator(full_kv_indices),
        8: create_indices_fake,
    }

    out, _ = autotune_select_algorithm(
        "omni_attention",
        choices,
        [x for x in inputs_for_autotuning if is_tensor_ir_node(x)],
        layout,
        input_gen_fns=input_gen_fns,
    )

    out.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    out.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer
    )

    return (out, logsumexp, max_scores)


# ---------------------------------------------------------------------------
# The entry
# ---------------------------------------------------------------------------


@register_lowering(omni_attention_hop, type_promotion_kind=None)
def lower_omni_attention(
    query: Any,
    key: Any,
    value: Any,
    subgraph: Any,
    block_mask: Any,
    scale: float,
    kernel_options: Any,
    score_mod_other_buffers: Any,
    mask_mod_other_buffers: Any,
) -> Any:
    """Write attention for a call, whichever way this device can be written to.

    Four ways, and they are four rather than one with a switch because they are
    not four tunings of one thing.  A device that cannot compare several
    positions at once is handed the comparisons; a device that can is handed the
    ranges and reads them several at a time; a device with neither is walked by
    one program at a time.  Which is right is a fact about the device, not a
    preference, so a program that named none of them is given the one that suits
    it and a program that named one is told if that one cannot be had.

    The two things that are asked of a device rather than of the program are
    asked first: whether the embedding is wide enough for a product at all, and
    which backend was named.  Both are cheaper to answer than to discover later,
    and the first has to come first because a kernel that cannot be written is
    not worth choosing.
    """

    kernel_options, backend = sanitize_kernel_options_for_triton(kernel_options)

    device_type = query.get_device().type
    if device_type == "cpu":
        # A processor is handed a program rather than tiles: there is no block to
        # choose and no grid to spread, only a program it runs.  Which is why
        # this does not go through anything below.
        from .omni_cpu import lower_omni_attention_cpu

        return lower_omni_attention_cpu(
            query,
            key,
            value,
            subgraph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
    if device_type == "mps":
        # Also a shader rather than tiles, and this compiler has none for that
        # device yet.  Said here rather than falling through to the tiled path,
        # because falling through would produce a kernel that device cannot run
        # -- an answer that is wrong rather than one that is missing.
        raise NotImplementedError(
            "attention on mps needs a shader rather than tiles, and this "
            "compiler has none for that device yet. The tiled path below runs "
            "on a device that has blocks."
        )

    check_embedding_is_wide_enough(query, value)

    mask_parts = unpack_block_mask(block_mask)
    mask_graph = mask_parts["mask_graph"]

    if backend == _BACKEND_FLASH:
        check_flash_supported_scalar_captures(
            score_mod_other_buffers, mask_mod_other_buffers
        )
        score_mod_other_buffers = realize_captures_for_cutedsl(score_mod_other_buffers)
        mask_mod_other_buffers = realize_captures_for_cutedsl(mask_mod_other_buffers)

    subgraph_buffer, mask_graph_buffer = capture_score_and_mask(
        query,
        subgraph,
        mask_graph,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    kernel_options = guard_kernel_options(kernel_options)
    enable_gqa = heads_are_grouped(query, key)

    from .omni_decoding import (
        create_omni_decoding_kernel,
        use_omni_decoding,
    )
    from .omni_flash_attention import (
        create_omni_flash_attention_kernel,
        _use_omni_flash_attention,
    )

    can_use_decode = use_omni_decoding(
        query, mask_parts["kv_indices"], value, kernel_options, enable_gqa
    )
    use_decode = (backend == _BACKEND_TRITON_DECODE) or (
        backend == _BACKEND_AUTO and can_use_decode
    )

    if backend == _BACKEND_TRITON_DECODE and not can_use_decode:
        raise RuntimeError(
            "BACKEND='TRITON_DECODE' was specified but attention decoding cannot be "
            "used for this input. Decoding is only available for short sequence "
            "lengths with specific configurations."
        )

    if use_decode:
        return create_omni_decoding_kernel(
            query,
            key,
            value,
            scale,
            kernel_options,
            subgraph_buffer,
            mask_graph_buffer,
            score_mod_other_buffers,
            mask_mod_other_buffers,
            mask_parts["kv_num_blocks"],
            mask_parts["kv_indices"],
            mask_parts["full_kv_num_blocks"],
            mask_parts["full_kv_indices"],
            mask_parts["sparse_q_block_size"],
            mask_parts["sparse_kv_block_size"],
        )

    if _use_omni_flash_attention(
        subgraph,
        mask_graph,
        kernel_options,
        num_score_mod_placeholders=5,
        backend=backend,
    ):
        return create_omni_flash_attention_kernel(
            query,
            key,
            value,
            scale,
            kernel_options,
            subgraph_buffer,
            mask_graph_buffer,
            score_mod_other_buffers,
            mask_mod_other_buffers,
            mask_parts["kv_num_blocks"],
            mask_parts["kv_indices"],
            mask_parts["full_kv_num_blocks"],
            mask_parts["full_kv_indices"],
            mask_parts["sparse_q_block_size"],
            mask_parts["sparse_kv_block_size"],
            mask_graph,
            subgraph,
        )

    return create_omni_attention_kernel(
        query,
        key,
        value,
        scale,
        kernel_options,
        subgraph_buffer,
        mask_graph_buffer,
        score_mod_other_buffers,
        mask_mod_other_buffers,
        mask_parts,
    )


def create_omni_attention_backward_kernel(
    query: Any,
    key: Any,
    value: Any,
    out: Any,
    logsumexp: Any,
    grad_out: Any,
    grad_logsumexp: Any,
    scale: float,
    kernel_options: Any,
    subgraph_buffer: SubgraphResults,
    mask_graph_buffer: SubgraphResults,
    joint_outputs: Any,
    score_mod_other_buffers: Any,
    mask_mod_other_buffers: Any,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> Any:
    """Write the pass that goes back the way attention came.

    Three values rather than one, and the difference is arithmetic rather than
    bookkeeping.  The gradient of the query is computed.  The gradients of the key
    and the value are accumulated into, because each is read by every query that
    may see it -- so neither is finished by whichever program arrives last, and
    the order they are added in is not something the hardware decides.

    What makes that accumulation correct is the fourth thing computed here and
    never handed back: how much each key contributed to the row it was read in.
    The answer to a row is a weighted sum over the keys that were visible, and
    the weight of a key depends on the total of the row -- so the key's gradient
    cannot be formed until the total is known, and the total is the same for every
    key in the row.  Computing it once per row is what turns the accumulation
    from a sum of differently scaled numbers into a sum of numbers.
    """

    from ..ir import ExternKernel
    from ..op_lowerings import (
        _convert_element_type,
        empty_strided,
        lower_mul,
        lower_sub,
        lower_sum,
    )
    from ..utils import can_use_tma
    from .omni_flash_attention import (
        is_power_of_2,
        is_tensor_ir_node,
        maybe_realize,
        set_head_dim_values,
    )

    bq, hq, seq_len_q, qk_head_dim = query.get_size()
    bkv, hkv, seq_len_kv, v_head_dim = value.get_size()

    key_size = [bq, hkv, seq_len_kv, qk_head_dim]
    key_strides = infer_dense_strides(key_size, key.get_stride())

    layout_broadcasted_k = FixedLayout(
        key.get_device(),
        key.get_dtype(),
        key_size,
        stride=[sympy.sympify(s) for s in key_strides],
    )

    # What each key was worth to the row it was read in, and what the row's
    # weights came to.  The second is the same for every key in a row, which is
    # the whole reason the first can be turned into a gradient at all.
    mul_delta = lower_mul(out, grad_out)
    delta = lower_sum(mul_delta, axis=-1)
    delta = _convert_element_type(delta, tp.float32)
    if grad_logsumexp is not None:
        # A gradient of the running total says how much the total itself is worth
        # moving, and the total was held in units of doubling -- so the amount is
        # restated in the units the weight was computed in before it is taken
        # off.
        grad_lse_exp2 = lower_mul(grad_logsumexp, 1 / math.log(2))
        grad_lse_exp2 = ExternKernel.require_contiguous(grad_lse_exp2)
        delta = lower_sub(delta, grad_lse_exp2)
        delta = ExternKernel.require_contiguous(delta)
        delta, grad_lse_exp2 = maybe_realize([delta, grad_lse_exp2])
    else:
        delta = ExternKernel.require_contiguous(delta)
        (delta,) = maybe_realize([delta])

    query_size = [bq, hq, seq_len_q, qk_head_dim]
    grad_query_strides = infer_dense_strides(query_size, query.get_stride())
    grad_query = empty_strided(
        query_size,
        stride=[sympy.sympify(s) for s in grad_query_strides],
        dtype=query.get_dtype(),
        device=query.get_device(),
    )

    # The key's gradient is added into, and it is added into at the distances the
    # value had -- because it is the value's positions that say which key each
    # element belongs to.
    value_size = [bq, hkv, seq_len_kv, v_head_dim]
    value_strides = infer_dense_strides(value_size, value.get_stride())
    grad_value = empty_strided(
        value_size,
        stride=[sympy.sympify(s) for s in value_strides],
        dtype=value.get_dtype(),
        device=value.get_device(),
    )
    grad_key = empty_strided(
        key_size,
        stride=[sympy.sympify(s) for s in key_strides],
        dtype=key.get_dtype(),
        device=key.get_device(),
    )

    kernel_options.setdefault("SM_SCALE", scale)

    gqa_shared_heads = FloorDiv(hq, hkv)
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    has_full_blocks = True
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)

    choices: list = []
    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.guard_int(query.get_size()[-1])
    configs = V.choices.get_omni_attention_bwd_configs(
        head_dim, dtype, query.get_device().type
    )

    invalid_block_options: Any = None
    original_kernel_options = kernel_options.copy()

    for conf in configs:
        cur_kernel_options = original_kernel_options.copy()
        # The prefix says which pass an option is for, and a kernel is only ever
        # given options for its own pass.  The backward's own prefix is this one,
        # so it comes off rather than being handed to a kernel that has never
        # heard of it.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("bwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("fwd_"):
                cur_kernel_options.pop(k)
        cur_kernel_options.setdefault("num_warps", conf.num_warps)
        cur_kernel_options.setdefault("num_stages", conf.num_stages)

        cur_kernel_options.setdefault("USE_TMA", False)
        if cur_kernel_options["USE_TMA"] and not can_use_tma(query, key, value):
            cur_kernel_options["USE_TMA"] = False

        # A tile wider than the block it walks is wasted lanes and a boundary mask
        # paid for on every step.  Narrowed only when there is one candidate and
        # both block sizes are whole powers of two -- a program that pinned the
        # tile keeps it, and is told if it does not divide.
        block_m1, block_n1 = conf.block_m1, conf.block_n1
        block_m2, block_n2 = conf.block_m2, conf.block_n2
        if len(configs) == 1 and all(
            is_power_of_2(s) and s >= 16
            for s in (sparse_q_block_size, sparse_kv_block_size)
        ):
            block_m1 = min(block_m1, sparse_q_block_size)
            block_n1 = min(block_n1, sparse_kv_block_size)
            block_m2 = min(block_m2, sparse_q_block_size)
            block_n2 = min(block_n2, sparse_kv_block_size)
        cur_kernel_options.setdefault("BLOCK_M1", block_m1)
        cur_kernel_options.setdefault("BLOCK_N1", block_n1)
        cur_kernel_options.setdefault("BLOCK_M2", block_m2)
        cur_kernel_options.setdefault("BLOCK_N2", block_n2)
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", sparse_q_block_size)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size)

        # The two walks are paired rather than independent: one reads the keys
        # for a block of queries and the other reads the queries for a block of
        # keys, and each tile has to divide the other's -- so only half the
        # combinations satisfy both and pairing is what makes one check enough.
        if (
            cur_kernel_options["BLOCK_N1"] % cur_kernel_options["BLOCK_M1"] != 0
            or cur_kernel_options["BLOCK_M2"] % cur_kernel_options["BLOCK_N2"] != 0
        ):
            invalid_block_options = cur_kernel_options
            if len(configs) == 1:
                raise_omni_kernel_options_error(
                    "backward",
                    cur_kernel_options,
                    ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"),
                    sparse_q_block_size,
                    sparse_kv_block_size,
                )
            continue

        for attrib in ("kpack", "matrix_instr_nonkdim", "waves_per_eu"):
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)

        error = OMNI_ATTENTION_BACKWARD.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                out,
                grad_out,
                logsumexp,
                delta,
                grad_key,
                grad_value,
            ],
            layout=layout_broadcasted_k,
            subgraphs=[subgraph_buffer, mask_graph_buffer],
            mutated_inputs=[grad_key, grad_value],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )
        if error is not None and len(configs) == 1:
            raise error

    if not choices and invalid_block_options is not None:
        raise_omni_kernel_options_error(
            "backward",
            invalid_block_options,
            ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"),
            sparse_q_block_size,
            sparse_kv_block_size,
        )

    inputs_for_autotuning = [
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp,
        delta,
        grad_key,
        grad_value,
    ]
    input_gen_fns: Any = None

    grad_query, _ = autotune_select_algorithm(
        "omni_attention_backward",
        choices,
        [x for x in inputs_for_autotuning if is_tensor_ir_node(x)],
        layout_broadcasted_k,
        input_gen_fns=input_gen_fns,
    )

    grad_query.data.data.subgraph_outs = get_bwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer, joint_outputs
    )
    grad_query.data.data.delta = delta
    grad_query.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )

    return (grad_query, grad_key, grad_value)


@register_lowering(omni_attention_backward_hop, type_promotion_kind=None)
def lower_omni_attention_backward(*args: Any, **kwargs: Any) -> Any:
    """Write the pass that goes back the way attention came.

    A backward pass is a forward pass and its own arithmetic.  The caller has
    already differentiated the program it wrote, so what arrives here is that
    program run backwards: it produces the gradient of the score, which is a value
    like any other, and this turns that value into the gradients of the three
    things that went in.

    The two devices that are written as programs rather than as tiles are asked
    first, and a device that has neither is refused by name -- because the tiled
    path below would produce a kernel that device cannot run, which is an answer
    that is wrong rather than one that is missing.
    """

    from .omni_flash_attention import (
        build_subgraph_buffer,
        create_omni_flash_attention_backward_kernel,
        create_placeholder,
        freeze_irnodes,
        maybe_realize,
        is_trivial_mask_graph,
        is_trivial_score_graph,
        _use_omni_flash_attention_backward,
    )

    (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    ) = args

    if query.get_device().type in ("mps", "cpu"):
        raise NotImplementedError(
            f"The backward pass on {query.get_device().type} is not written yet. "
            "A program that only needs the forward pass should not be asking "
            "for this one."
        )

    mask_parts = unpack_block_mask(block_mask)
    mask_graph = mask_parts["mask_graph"]

    backend = kernel_options.get("BACKEND", _BACKEND_AUTO)

    # What the caller's program, run backwards, produced.  Taken apart before
    # anything else, because whether a captured gradient is read or added to is
    # something the kernel should not have to know.
    joint_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("delta", tp.float32),
            ("b", tp.int32),
            ("h", tp.int32),
            ("m", tp.int32),
            ("n", tp.int32),
        ]
    ]
    joint_subgraph_buffer = build_subgraph_buffer(
        joint_placeholder_inps + list(score_mod_other_buffers), joint_graph
    )
    freeze_irnodes(joint_subgraph_buffer)

    all_joint_outputs = joint_subgraph_buffer
    freeze_irnodes(all_joint_outputs)

    joint_outputs = process_joint_outputs(
        all_joint_outputs, len(joint_placeholder_inps)
    )

    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("b", tp.int32),
            ("h", tp.int32),
            ("m", tp.int32),
            ("n", tp.int32),
        ]
    ]
    mask_graph_buffer = build_subgraph_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
    )
    freeze_irnodes(mask_graph_buffer)

    if _use_omni_flash_attention_backward(
        fw_graph,
        mask_graph,
        backend=backend,
        joint_outputs=joint_outputs,
        score_mod_other_buffers=score_mod_other_buffers,
    ):
        needs_block_mask = not is_trivial_mask_graph(
            getattr(mask_graph, "graph_module", mask_graph)
        )
        if grad_logsumexp is not None:
            (grad_logsumexp,) = maybe_realize([grad_logsumexp])

        score_is_trivial = is_trivial_score_graph(
            getattr(fw_graph, "graph_module", fw_graph)
        )
        return create_omni_flash_attention_backward_kernel(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            scale,
            kernel_options,
            mask_parts["sparse_q_block_size"],
            mask_parts["sparse_kv_block_size"],
            fw_subgraph_buffer=None if score_is_trivial else joint_subgraph_buffer,
            joint_subgraph_buffer=None
            if score_is_trivial
            else joint_outputs.grad_input,
            score_mod_other_buffers=list(score_mod_other_buffers),
            mask_graph_buffer=mask_graph_buffer if needs_block_mask else None,
            mask_mod_other_buffers=list(mask_mod_other_buffers),
            q_num_blocks=mask_parts["q_num_blocks"] if needs_block_mask else None,
            q_indices=mask_parts["q_indices"] if needs_block_mask else None,
            full_q_num_blocks=mask_parts["full_q_num_blocks"]
            if needs_block_mask
            else None,
            full_q_indices=mask_parts["full_q_indices"] if needs_block_mask else None,
        )

    return create_omni_attention_backward_kernel(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        scale,
        kernel_options,
        joint_subgraph_buffer,
        mask_graph_buffer,
        joint_outputs,
        score_mod_other_buffers,
        mask_mod_other_buffers,
        mask_parts["sparse_q_block_size"],
        mask_parts["sparse_kv_block_size"],
    )
