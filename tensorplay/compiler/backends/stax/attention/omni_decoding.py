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

from ..loops import V
from ..runtime.runtime_utils import ceildiv
from ..templates.mm_common import load_kernel_template
from ..templates.select_algorithm import TritonTemplate
from .omni_attention import (
    _omni_kernel_options_example,
    _omni_kernel_tuning_options,
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
