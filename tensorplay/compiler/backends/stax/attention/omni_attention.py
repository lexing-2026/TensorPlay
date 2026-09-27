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

from ..runtime.runtime_utils import ceildiv
from ..templates.mm_common import load_kernel_template
from ..templates.select_algorithm import TritonTemplate

# ---------------------------------------------------------------------------
# How the work is spread over the device
# ---------------------------------------------------------------------------


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
