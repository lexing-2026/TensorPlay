"""Attention that asks one question at a time.

Decoding is attention with a query axis of one.  Nothing about the arithmetic
changes: the same products, the same running total, the same mask.  What changes
is that there is almost nothing to do, and a kernel written for the general case
spends that almost-nothing on setting up work that has one row.

So this is the general kernel with its query axis short, and it is a separate
body only because the two spread their work differently.  The general one gives
each query tile a program and walks every key from it.  This one gives each query
tile a program per key, has each of them produce a partial answer, and combines
them -- which is only worth doing when there are far more keys than queries,
because otherwise the combining costs more than the walking it saves.

Which is why the choice is not made by asking which is faster.  It is made by
asking whether this call has the shape that makes splitting pay, and a shape
that cannot be shown to have it does not get it: the general kernel is always an
answer, so nothing here is allowed to be the reason a call cannot be compiled.
"""

from __future__ import annotations

from typing import Any

import sympy

import tensorplay as tp

from tensorplay.graph.experimental.sympy_functions import FloorDiv, Mod
from ..ir import FixedLayout
from ..loops import V
from ..op_lowerings import (
    _convert_element_type,
    lower_add,
    lower_as_strided,
    lower_div,
    lower_eq,
    lower_exp2,
    lower_log2,
    lower_max,
    lower_mul,
    lower_squeeze,
    lower_sub,
    lower_sum,
    lower_unsqueeze,
    lower_where,
)
from ..runtime.runtime_utils import ceildiv
from ..templates.mm_common import load_kernel_template
from ..templates.select_algorithm import TritonTemplate, autotune_select_algorithm
from .omni_attention import (
    _omni_kernel_options_example,
    _omni_kernel_tuning_options,
    guard_kernel_options,
    omni_decoding_grid,
)

#: The body and the helpers it is joined with, and the way the work is spread --
#: which is a copy of the forward's, because it is the same work: one program
#: per query tile, per head and batch, times the split of the keys.
#: The query axis has to be shorter than this for splitting the keys to pay.
#:
#: A number rather than a comparison against the key axis, because the key axis
#: is not known here and the query axis is short precisely because there is one
#: question.  A hundred and twenty-eight is where a query tile stops being one
#: row: past it there is enough work per query that a program walking every key
#: is not idle.
_MAX_DECODE_QUERY_LENGTH = 128

#: The name a program uses to insist on the general kernel.
#:
#: In the options rather than as a backend, because it is not a different way of
#: writing attention -- it is the same one, asked for by name.  A program that
#: sets it is saying the split is not wanted, whatever the shapes say.
_FORCE_GENERAL_OPTION = "FORCE_USE_OMNI_ATTENTION"


# ---------------------------------------------------------------------------
# Options that cannot both be true
# ---------------------------------------------------------------------------


def raise_omni_decoding_kernel_options_error(
    kernel_options: Any,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> None:
    """Say which two options cannot both be true, and what would be.

    A tile has to divide the block it walks, and the two widths that say so are
    the caller's and not this kernel's to choose, so a pair that does not divide
    is a pair the caller has to be told about -- with the numbers, because a
    caller who cannot see them cannot change either.
    """

    formated_kernel_options = ", ".join(
        f"{name}={kernel_options[name]}" for name in ("BLOCK_M", "BLOCK_N")
    )
    raise ValueError(
        "Invalid attention decode kernel options: Q and KV block sizes must "
        "be divisible by the selected tile sizes. Got "
        f"SPARSE_Q_BLOCK_SIZE={sparse_q_block_size}, "
        f"SPARSE_KV_BLOCK_SIZE={sparse_kv_block_size}, and "
        f"{formated_kernel_options}. "
        "Pass compatible values with kernel_options. Available decode tuning "
        f"options are {_omni_kernel_tuning_options('decode')}. For example: "
        f"{_omni_kernel_options_example('decode')}. If you did not pin "
        "these options, compiling with mode='max-autotune-no-cudagraphs' "
        "can also fix this by trying more attention configs."
    )


# ---------------------------------------------------------------------------
# Whether this call is one
# ---------------------------------------------------------------------------


def use_omni_decoding(
    query: Any, kv_indices: Any, value: Any, kernel_options: Any, enable_gqa: Any
) -> bool:
    """Whether this call is one where splitting the keys pays for itself.

    Every question below is asked so that not being able to answer it means no.
    That is the whole of the design: the general kernel is always an answer, so
    nothing here may become the reason a call cannot be compiled.  An unknown
    shape therefore disables the split rather than enabling it, and a shape that
    cannot be shown to be short enough is not treated as short enough.

    The batch and the head count have to be known rather than merely written,
    because the split is decided from them: how many keys each program walks is
    a count of programs, and a count of programs cannot be divided by something
    not yet known.
    """

    force_general = kernel_options.get(_FORCE_GENERAL_OPTION, False)

    # A query axis short enough that one row is a row, and long enough to be
    # worth computing at all.
    short_query_length = V.graph.sizevars.guard_or_false(
        sympy.Lt(query.get_size()[-2], _MAX_DECODE_QUERY_LENGTH)
    )
    non_zero_length = V.graph.sizevars.guard_or_false(
        sympy.Gt(query.get_size()[-2], 0)
    )

    # The split is counted from these two, so they are wanted as numbers.
    static_batch = isinstance(query.get_size()[0], (int, sympy.Integer))
    static_num_heads = isinstance(query.get_size()[1], (int, sympy.Integer))

    if enable_gqa:
        # A group of query heads over one key head is walked by the same program
        # here, so every head in the group has to be allowed the same blocks.
        # They are not, in general -- each head's mask says which blocks it may
        # see -- so this is only true when there is one head's worth of blocks
        # and therefore nothing to disagree about.
        valid_block_mask_num_heads = V.graph.sizevars.guard_or_false(
            sympy.Eq(kv_indices.get_size()[1], 1)
        )
    else:
        # Without groups, the blocks are either one head's worth or one per
        # query head, and both of those put every query head's program on the
        # same blocks.
        valid_block_mask_num_heads = V.graph.sizevars.guard_or_false(
            sympy.Or(
                sympy.Eq(kv_indices.get_size()[1], 1),
                sympy.Eq(kv_indices.get_size()[1], query.get_size()[1]),
            )
        )

    # The ratio of query heads to key heads is what one program ends up walking,
    # and it is walked as a number of whole groups: a ratio of three leaves a
    # group of one and a program that walks a whole number of groups wastes two
    # thirds of itself.
    hq = query.get_size()[1]
    hkv = value.get_size()[1]
    ratio = hq // hkv if isinstance(hq, (int, sympy.Integer)) and isinstance(
        hkv, (int, sympy.Integer)
    ) and hkv else 0

    pw_of_two = V.graph.sizevars.guard_or_false(
        sympy.And(sympy.Gt(ratio, 0), sympy.Eq(ratio & (ratio - 1), 0))
    )

    return (
        not force_general
        and not kernel_options.get("OUTPUT_MAX", False)
        and short_query_length
        and static_batch
        and static_num_heads
        and non_zero_length
        and valid_block_mask_num_heads
        and pw_of_two
    )


def get_split_k(B: int, H: int, Mk: int) -> int:
    """How many programs each query tile's keys are split across.

    Counted from what the device has rather than from what would fill it: a
    device with eighty-two processors given two programs has forty-one of them
    idle, and a split that does not account for that is a split that leaves the
    device idle on purpose.

    Two per processor, so that a processor whose share turns out to be masked
    away still has one that is not -- the split is decided before the mask is
    read, so it cannot know which shares are real.  At least one, because a
    split of zero is not a split.
    """

    import tensorplay as tp

    if hasattr(tp, "xpu") and tp.xpu.is_available():
        num_sm = tp.xpu.get_device_properties("xpu").gpu_subslice_count
    else:
        num_sm = tp.cuda.get_device_properties("cuda").multi_processor_count
    bh = max(B * H, 1)
    if not isinstance(bh, (int, sympy.Integer)):
        raise AssertionError("B and H must be concrete integers")
    split_k = num_sm // bh * 2
    split_k = max(split_k, 1)

    return split_k


# ---------------------------------------------------------------------------
# The body
# ---------------------------------------------------------------------------

#: A decoding kernel is a forward kernel with the query axis short, so it takes
#: the same two sets of helpers: the ones the forward shares with the backward,
#: and the ones only the forward uses.
omni_decoding_source = (
    load_kernel_template("omni_decode")
    + load_kernel_template("omni_utilities")
    + load_kernel_template("omni_common")
)

#: The body, and the way the work is spread -- which is the forward's, because it
#: is the same work: one program per query tile, per head and batch, times the
#: split of the keys.
OMNI_DECODING = TritonTemplate(
    name="omni_decoding",
    grid=omni_decoding_grid,
    source=omni_decoding_source,
    always_freeze_layout=True,
)


# ---------------------------------------------------------------------------
# Writing the kernel
# ---------------------------------------------------------------------------


def create_omni_decoding_kernel(
    query: Any,
    key: Any,
    value: Any,
    scale: float,
    kernel_options: Any,
    score_mod_subgraph: Any,
    mask_mod_subgraph: Any,
    score_mod_other_buffers: Any,
    mask_mod_other_buffers: Any,
    kv_num_blocks: Any,
    kv_indices: Any,
    full_kv_num_blocks: Any,
    full_kv_indices: Any,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> Any:
    """Write the pass for a query axis of one, and hand back what it produced.

    Two values rather than one, for the same reason the general forward gives
    two: the second is the running total the first was accumulated against, and
    a later pass needs it.  It is kept at a width that does not lose the running
    total to rounding, whatever width the values themselves are at.

    What is different here is that the answer comes out in pieces.  Each program
    walks some of the keys and produces a partial answer together with the
    largest value it saw and the total of the exponentials it accumulated, and
    the combining of those is arithmetic on values rather than another kernel --
    which is why those two are written out at a width that can be reduced
    against itself, and why the answer is only complete once they have been.
    """

    from ..ir import ExternKernel, FlexibleLayout
    from ..op_lowerings import empty_strided
    from ..utils import can_use_tma
    from .omni_flash_attention import (
        can_skip_boundary_checks,
        create_indices_fake,
        create_num_blocks_fake_generator,
        freeze_irnodes,
        get_fwd_subgraph_outputs,
        is_power_of_2,
        is_tensor_ir_node,
        maybe_realize,
        next_power_of_two,
        set_head_dim_values,
    )
    from .omni_attention import heads_are_grouped

    bq, hq, seq_len_q, qk_head_dim = query.get_size()
    bkv, hkv, seq_len_kv, v_head_dim = value.get_size()

    if not V.graph.sizevars.evaluate_expr(sympy.Eq(bq, bkv) | sympy.Eq(bkv, 1)):
        raise AssertionError(
            f"Bq and Bkv must broadcastable. Got Bq={bq} and Bkv={bkv}"
        )

    b = bq
    kernel_options = guard_kernel_options(kernel_options)

    # A row that is not a whole number of tiles has to be masked off on the last
    # tile, and a column likewise.  When both divide there is nothing to mask,
    # and saying so lets the tiles drop the comparisons entirely.
    seq_q_divisible = can_skip_boundary_checks(seq_len_q, sparse_q_block_size)
    seq_kv_divisible = can_skip_boundary_checks(seq_len_kv, sparse_kv_block_size)
    kernel_options.setdefault("IS_DIVISIBLE", bool(seq_q_divisible and seq_kv_divisible))

    # How many query heads share one key head.  A power of two because a program
    # walks a whole number of them and three would leave it walking two thirds
    # of itself.
    gqa_shared_heads = FloorDiv(hq, hkv)
    if not is_power_of_2(gqa_shared_heads):
        raise ValueError(
            "Number of shared query heads sharing the same KV head must be power of 2. "
        )
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # Blocks that are entirely inside the mask need only the score applied to
    # them, and asking whether a position is inside the mask is a comparison per
    # position.  So a mask that has both kinds of block says which are which.
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        # A list of nothing, of the right length, so that the kernel can be
        # handed two lists of the same shape whether or not there is anything in
        # the second one.
        full_kv_num_blocks, full_kv_indices = (
            empty_strided([0], None, dtype=query.get_dtype(), device=query.get_device())
            for _ in range(2)
        )

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
    )
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)

    choices: list = []
    dtype = key.get_dtype()
    head_dim = V.graph.sizevars.guard_int(key.get_size()[-1])
    configs = V.choices.get_omni_decode_configs(
        head_dim, dtype, query.get_device().type
    )

    kernel_options.setdefault("SM_SCALE", scale)
    kernel_options.setdefault("SPLIT_KV", get_split_k(b, hkv, seq_len_kv))
    max_split_kv = kernel_options["SPLIT_KV"]

    # The partial answers, one per program, and beside each the two numbers that
    # say what it saw: the largest value and the total of the exponentials.  Both
    # at full width regardless of the values' own width, because they are added
    # to and compared against each other across programs, and a width that lost
    # the running total would make the combined answer a different number than
    # an unsplit one.
    buf_acc_shape = [b, max_split_kv, hq, seq_len_q, v_head_dim]
    buf_ml_shape = buf_acc_shape[:-1]
    buf_m = empty_strided(
        buf_ml_shape,
        None,
        dtype=tp.float32,
        device=query.get_device(),
    )
    buf_l = empty_strided(
        buf_ml_shape,
        None,
        dtype=tp.float32,
        device=query.get_device(),
    )

    layout_acc = FixedLayout(
        query.get_device(),
        tp.float32,
        buf_acc_shape,
        FlexibleLayout.contiguous_strides(buf_acc_shape),
    )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    kernel_options.setdefault(
        "BLOCK_M",
        max(
            next_power_of_two(
                V.graph.sizevars.optimization_hint(seq_len_q) * gqa_shared_heads
            ),
            16,
        ),
    )

    query = ExternKernel.realize_input(query)
    stride_b, stride_hq, stride_seq_len_q, stride_qk_head_dim = query.get_stride()

    # A group of query heads over one key head is walked by the same program, so
    # the heads are made adjacent -- reshaped rather than copied, and with the
    # distances they already had, so nothing moves.
    gqa_query_shape = (b, hkv, gqa_shared_heads, seq_len_q, qk_head_dim)
    gqa_query_stride = (
        stride_b,
        stride_hq * gqa_shared_heads,
        stride_hq,
        stride_seq_len_q,
        stride_qk_head_dim,
    )
    query = lower_as_strided(query, gqa_query_shape, gqa_query_stride)

    kernel_options.setdefault(
        "SAFE_M_BOUNDARY",
        Mod(seq_len_q * gqa_shared_heads, kernel_options["BLOCK_M"]) == 0,
    )
    kernel_options.setdefault("SAFE_N_BOUNDARY", True)
    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)

    original_kernel_options = kernel_options.copy()

    invalid_block_options: Any = None

    for conf in configs:
        cur_kernel_options = original_kernel_options.copy()
        # The prefix says which pass an option is for, and a kernel is only ever
        # given options for its own pass -- so the prefix comes off here and the
        # other pass's options are dropped rather than handed to a kernel that
        # has no use for them.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("fwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("bwd_"):
                cur_kernel_options.pop(k)

        cur_kernel_options.setdefault("BLOCK_N", min(conf.block_n, sparse_kv_block_size))
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", sparse_q_block_size)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size)
        cur_kernel_options.setdefault("num_warps", conf.num_warps)
        cur_kernel_options.setdefault("num_stages", conf.num_stages)

        if (
            cur_kernel_options["SPARSE_Q_BLOCK_SIZE"] % cur_kernel_options["BLOCK_M"]
            != 0
            or cur_kernel_options["SPARSE_KV_BLOCK_SIZE"] % cur_kernel_options["BLOCK_N"]
            != 0
        ):
            # A tile that does not divide the block it walks would read past the
            # end of it.  One bad candidate among several is not an error -- it
            # is one fewer thing to measure -- but a caller who pinned the option
            # has to be told, because there is nothing left to measure.
            invalid_block_options = cur_kernel_options
            if len(configs) == 1:
                raise_omni_decoding_kernel_options_error(
                    cur_kernel_options,
                    sparse_q_block_size,
                    sparse_kv_block_size,
                )
            continue

        cur_kernel_options.setdefault("USE_TMA", False)
        if cur_kernel_options["USE_TMA"] and not can_use_tma(query, key, value):
            cur_kernel_options["USE_TMA"] = False

        for attrib in ("kpack", "matrix_instr_nonkdim", "waves_per_eu"):
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)

        OMNI_DECODING.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                buf_m,
                buf_l,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
            ],
            layout=layout_acc,
            subgraphs=[
                score_mod_subgraph,
                mask_mod_subgraph,
            ],
            mutated_inputs=[buf_m, buf_l],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )

    if not choices and invalid_block_options is not None:
        raise_omni_decoding_kernel_options_error(
            invalid_block_options,
            sparse_q_block_size,
            sparse_kv_block_size,
        )

    inputs_for_omni_decoding = (
        [
            query,
            key,
            value,
            buf_m,
            buf_l,
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

    buf_acc, _ = autotune_select_algorithm(
        "omni_decoding",
        choices,
        [x for x in inputs_for_omni_decoding if is_tensor_ir_node(x)],
        layout_acc,
        input_gen_fns=input_gen_fns,
    )

    buf_acc.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    buf_acc.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        score_mod_subgraph, mask_mod_subgraph
    )

    # ---- combining the partial answers ------------------------------------
    #
    # Each program produced an answer that was accumulated against a largest
    # value it saw for itself.  Adding two such answers directly would add two
    # numbers that were scaled by different things, so each is first rescaled by
    # the ratio of the two largest values -- which is what makes the exponentials
    # add rather than the values.

    g_m = lower_max(buf_m, dim=1, keepdim=True)[0]
    # A row whose keys were all masked away saw no value at all, and its largest
    # is the identity for exponentiation rather than a number.  Told apart here so
    # that the rescaling below does not turn it into a number.
    masked_rows = lower_eq(g_m, -float("inf"))
    adj_m = lower_sub(buf_m, g_m)
    adj_m = lower_where(masked_rows, 0, adj_m)
    alpha = lower_exp2(adj_m)

    buf_l = lower_mul(buf_l, alpha)
    g_l = lower_sum(buf_l, dims=1)
    masked_rows_squeezed = lower_squeeze(masked_rows, dim=1)
    g_l = lower_where(masked_rows_squeezed, 1.0, g_l)
    logsumexp = lower_log2(g_l)
    logsumexp = lower_add(logsumexp, lower_squeeze(g_m, dim=1))

    alpha_unseq = lower_unsqueeze(alpha, 4)
    buf_acc = lower_mul(buf_acc, alpha_unseq)
    output = lower_sum(buf_acc, dims=1)
    l_unseq = lower_unsqueeze(g_l, 3)
    output = lower_div(output, l_unseq)
    output = _convert_element_type(output, query.get_dtype())

    return (
        output,
        logsumexp,
    )
