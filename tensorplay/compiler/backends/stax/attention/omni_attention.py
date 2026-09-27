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
from .omni_flash_attention import (
    _omni_kernel_options_example,
    _omni_kernel_tuning_options,
    has_unsupported_cpu_scalar_tensor_captures,
)

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
