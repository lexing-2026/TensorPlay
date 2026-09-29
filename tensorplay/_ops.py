"""

Two kinds of entries resolve here:

1. Python-registered operators (:mod:`tensorplay.library`):
   ``tensorplay.ops.mylib.add(x, y)`` returns the :class:`CustomOpDef` and
   calling it runs the normal dispatch path (autograd, capture awareness).
2. Natively loaded extension libraries: ``tensorplay.ops.load_library(path)``
   dlopens a shared object whose static registrars feed the p10 dispatcher
   (the ``TENSORPLAY_LIBRARY_IMPL`` macro family) and attaches the module
"""

from __future__ import annotations

import math
import types
from typing import Any

import tensorplay
import tensorplay._C as _C


def _lower_right_causal_mask(query: Any, key: Any) -> Any:
    """Boolean keep-mask aligned to the lower-right (L, S) corner."""
    L, S = query.size(-2), key.size(-2)
    q_idx = tensorplay.arange(L, device=query.device).view(L, 1)
    k_idx = tensorplay.arange(S, device=query.device).view(1, S)
    return q_idx >= k_idx - (S - L)


#: Mask alignment codes the kernel contracts pass around.  Zero leaves the
#: score unmasked, one aligns the included positions to the top-left corner of
#: the score matrix and two to the bottom-right corner.  The two alignments
#: name the same set of positions only when the query and key lengths match,
#: which is why the choice has to be carried rather than recomputed.
_NO_CUSTOM_MASK = 0
_CAUSAL_FROM_TOP_LEFT = 1
_CAUSAL_FROM_BOTTOM_RIGHT = 2


def _additive_mask(keep: Any, dtype: Any) -> Any:
    """Restate a boolean keep-mask additively, in the units scores are summed in."""
    device = keep.device
    return tensorplay.where(
        keep,
        tensorplay.zeros((), dtype=dtype, device=device),
        tensorplay.full((), float("-inf"), dtype=dtype, device=device),
    )


def _causal_keep(length_q: int, length_k: int, variant: int, device: Any) -> Any:
    q_idx = tensorplay.arange(length_q, device=device).unsqueeze(-1)
    k_idx = tensorplay.arange(length_k, device=device).unsqueeze(-2)
    if variant == _CAUSAL_FROM_BOTTOM_RIGHT:
        return q_idx >= k_idx - (length_k - length_q)
    return q_idx >= k_idx


def _window_keep(length_q: int, length_k: int, window_left: int, window_right: int, device: Any) -> Any:
    """Which keys each query may see, for the sliding-window contract.

    Both bounds are measured from the diagonal running from the top-left
    corner to the bottom-right one, so a query and a key of different lengths
    shift together: query ``r`` sees the keys from ``r + length_k - length_q -
    left`` up to but not including ``r + length_k - length_q + right + 1``.
    A negative bound leaves that side unbounded, and so does a bound at least
    as large as the key length, since it then excludes nothing.

    Two things follow that are easy to read past.  With the right bound at zero
    and the left unbounded, which is what the causal flag asks for, the visible
    keys are the ones at or before the shifted diagonal — so on this contract
    the causal flag aligns to the lower-right corner whenever the two lengths
    differ, and only coincides with the upper-left corner when they match.  And
    a query further along than there are keys can end up with an empty row,
    which the softmax has to survive rather than divide by zero.
    """
    if window_left < 0 and window_right < 0:
        return None
    q_idx = tensorplay.arange(length_q, device=device).unsqueeze(-1)
    k_idx = tensorplay.arange(length_k, device=device).unsqueeze(-2)
    shift = q_idx + (length_k - length_q)
    keep = None
    if window_left >= 0:
        keep = k_idx >= shift - window_left
    if window_right >= 0:
        nearer = k_idx < shift + window_right + 1
        keep = nearer if keep is None else keep & nearer
    return keep


def _resolve_window(is_causal: bool, window_left: Any, window_right: Any) -> tuple[int, int]:
    """Turn the causal flag and a window pair into one pair of bounds.

    The flag is not a separate case: it names the one window whose right bound
    is zero, which is also the only reading under which the visible keys sit
    at or before the diagonal rather than after it.
    """
    left = -1 if window_left is None else int(window_left)
    right = -1 if window_right is None else int(window_right)
    if is_causal:
        right = 0
    return left, right


def _expand_grouped_heads(query: Any, key: Any, value: Any) -> tuple[Any, Any]:
    """Give the keys and values as many heads as the query asks for.

    With grouped or multi-query attention the key and value tensors carry
    fewer heads than the query, and each one head feeds a fixed run of query
    heads, so the run is repeated in place of the head.
    """
    groups = query.size(-3) // key.size(-3)
    if groups == 1:
        return key, value
    if query.size(-3) % key.size(-3) != 0:
        raise ValueError(f"query heads {query.size(-3)} must be a multiple of key/value heads {key.size(-3)}")
    return (
        tensorplay.repeat_interleave(key, groups, dim=-3),
        tensorplay.repeat_interleave(value, groups, dim=-3),
    )


def _check_layout(query: Any, key: Any, value: Any, packed: bool, head_axis: int) -> None:
    """Refuse a shape the composite would read with the axes in the wrong roles.

    Each of these contracts takes its batched inputs in a fixed order, and the
    two orders differ only in which of the second and third axes is the
    sequence.  A caller that hands over the other one gets a sensible-looking
    number out of a product taken along the wrong axis, so it is worth
    catching the swap here rather than downstream.
    """
    if packed:
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.dim() != 3:
                raise ValueError(
                    f"packed attention takes {name} as (total, heads, dim), got "
                    f"{tensor.dim()} axes {tuple(tensor.shape)}"
                )
        return
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.dim() != 4:
            raise ValueError(
                f"batched attention takes {name} as (batch, "
                f"{'sequence, heads' if head_axis == 2 else 'heads, sequence'}, "
                f"dim), got {tensor.dim()} axes {tuple(tensor.shape)}"
            )


def _math_attention(query: Any, key: Any, value: Any, add_mask: Any, scale: Any) -> tuple[Any, Any]:
    """Attention composite that also reports the softmax normalizing constant.

    The output weights the values by the softmax of the scores, and the
    logsumexp of those same scores is that softmax's normalizing constant, so
    forming the scores once and reading both results off them costs one pass
    and cannot leave the two inconsistent.  Reduced-precision inputs
    accumulate in float32 and the output is cast back, which is where the
    fused kernels keep their accumulators too.  ``add_mask`` is already
    additive, or None where nothing is masked.

    A query can end up with no key it is allowed to see, which is what a
    query further along than there are keys looks like, and what the first
    rows of a sliding window look like before the window has filled.  Such a
    row has no softmax to speak of, and dividing by its zero total would make
    every output a NaN, so the total is left at one and the row keeps a zero
    output.  Its reported constant is positive infinity rather than negative:
    a backward pass that reuses the constant to reweight the scores then gets
    zeros out of ``exp(score - inf)``, which is the right derivative for a row
    that contributes nothing, where the opposite sign would hand back a NaN.

    The constant is reported in single precision whatever the inputs are, which
    is the width the contracts declare it at and the width a backward pass
    reads it back at.
    """
    dtype = query.dtype
    reduced = dtype in (tensorplay.float16, tensorplay.bfloat16)
    working = tensorplay.float32 if reduced else dtype
    q = query.to(working) if reduced else query
    k = key.to(working) if reduced else key
    v = value.to(working) if reduced else value
    k, v = _expand_grouped_heads(q, k, v)
    factor = 1.0 / math.sqrt(q.size(-1)) if scale is None else float(scale)
    scores = tensorplay.matmul(q * factor, tensorplay.transpose(k, -2, -1))
    if add_mask is not None:
        scores = scores + add_mask.to(working)
    # A row whose largest score is minus infinity has nothing to exponentiate
    # against, so its reference point is zero; the masked entries stay at
    # minus infinity and exponentiate to zero either way.
    row_max = tensorplay.amax(scores, dim=-1, keepdim=True)
    row_max = tensorplay.where(row_max == float("-inf"), tensorplay.zeros_like(row_max), row_max)
    probs = tensorplay.exp(scores - row_max)
    total = probs.sum(dim=-1, keepdim=True)
    empty = total == 0
    normalizer = tensorplay.where(empty, tensorplay.ones_like(total), total)
    lse = (row_max + tensorplay.log(total)).squeeze(-1)
    lse = tensorplay.where(empty.squeeze(-1), tensorplay.full_like(lse, float("inf")), lse)
    out = tensorplay.matmul(probs / normalizer, v)
    if reduced:
        out = out.to(dtype)
    return out, lse.to(tensorplay.float32)


def _packed_attention(
    query: Any,
    key: Any,
    value: Any,
    cu_q: Any,
    cu_k: Any,
    scale: Any,
    window_left: int,
    window_right: int,
) -> tuple[Any, Any]:
    """Attention over a packed batch of sequences, one sequence at a time.

    The packed layout holds every sequence back to back with nothing marking
    where one ends and the next begins, so the boundaries come from the
    cumulative-length table.  Visiting one sequence at a time keeps each score
    matrix the size of the sequence it belongs to instead of the size of the
    whole packed batch, which for a batch of many short sequences is the
    difference between one matrix and thousands of them.  Every sequence gets
    its own window, so a length bound that is loose for one is applied to
    each on its own terms.

    The table is read on the host, which costs a synchronization on an
    accelerator.  That is the price of knowing where the sequences begin; the
    kernels this stands in for are handed the same table and never pay it.
    """
    bounds_q = cu_q.tolist()
    bounds_k = cu_k.tolist() if cu_k is not None else bounds_q
    if len(bounds_q) != len(bounds_k):
        raise ValueError(f"query and key sequence tables disagree: {len(bounds_q)} vs {len(bounds_k)} entries")
    device = query.device
    outs: list[Any] = []
    lses: list[Any] = []
    for seq in range(len(bounds_q) - 1):
        start_q, stop_q = int(bounds_q[seq]), int(bounds_q[seq + 1])
        start_k, stop_k = int(bounds_k[seq]), int(bounds_k[seq + 1])
        length_q, length_k = stop_q - start_q, stop_k - start_k
        if length_q == 0:
            continue
        if length_k == 0:
            raise ValueError(f"sequence {seq} has {length_q} queries and no keys, so every score row would be empty")
        keep = _window_keep(length_q, length_k, window_left, window_right, device)
        add_mask = None if keep is None else _additive_mask(keep, query.dtype)
        # Head-last slices of the packed layout become the head-second-rank
        # layout the composite multiplies in.
        out_seq, lse_seq = _math_attention(
            query[start_q:stop_q].transpose(0, 1),
            key[start_k:stop_k].transpose(0, 1),
            value[start_k:stop_k].transpose(0, 1),
            add_mask,
            scale,
        )
        outs.append(out_seq.transpose(0, 1))
        lses.append(lse_seq)
    if not outs:
        heads, dim = query.size(1), value.size(-1)
        total_q = query.size(0)
        return (
            tensorplay.empty((total_q, heads, dim), dtype=query.dtype, device=device),
            tensorplay.empty((heads, total_q), dtype=tensorplay.float32, device=device),
        )
    return tensorplay.cat(outs, dim=0), tensorplay.cat(lses, dim=-1)


def _dense_or_packed(
    query: Any,
    key: Any,
    value: Any,
    cu_q: Any,
    cu_k: Any,
    keep: Any,
    scale: Any,
) -> tuple[Any, Any]:
    """Route a call to the packed or the plain composite by its shape.

    A cumulative-length table is what distinguishes the two layouts; without
    one the tensors are already batched and the score matrix can be formed in
    a single product.
    """
    if cu_q is not None:
        return _packed_attention(query, key, value, cu_q, cu_k, scale, -1, -1)
    if query.dim() == 3 and keep is not None:
        raise ValueError("an explicit mask needs the batched layout")
    add_mask = None if keep is None else _additive_mask(keep, query.dtype)
    return _math_attention(query, key, value, add_mask, scale)


def _flash_attention_adapter(
    query: Any,
    key: Any,
    value: Any,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Any = None,
) -> tuple[Any, ...]:
    """Flash-attention composite over the fused kernels shipped in this build.

    The dispatcher contract calls for a nine-field result, with the causal
    flag aligned to the lower-right (L, S) corner.  The fused kernels align
    their causal mask to the query index (the upper-left corner), which only
    coincides for square sequence lengths; non-square causal calls therefore
    run through the math composite with an explicit lower-right mask.  The
    CPU fused kernel returns the output and the per-row logsumexp; the CUDA
    fused kernel returns the output only, and takes no scale argument, so a
    non-default scale is folded into the query — rescaling the scores by
    ``s`` equals rescaling ``q`` by ``s * sqrt(E)`` given the kernel's
    built-in ``1 / sqrt(E)`` factor.
    """
    del return_debug_mask
    if dropout_p != 0.0:
        raise NotImplementedError("flash attention: dropout > 0 is not supported in this build")
    empty = tensorplay.empty(0, dtype=query.dtype, device=query.device)
    rng_state = tensorplay.zeros((2,), dtype=tensorplay.uint64, device=query.device)
    max_q, max_k = query.size(-2), key.size(-2)
    if is_causal and query.size(-2) != key.size(-2):
        keep = _causal_keep(query.size(-2), key.size(-2), _CAUSAL_FROM_BOTTOM_RIGHT, query.device)
        out, lse = _math_attention(query, key, value, _additive_mask(keep, query.dtype), scale)
        return out, lse, empty, empty, max_q, max_k, rng_state, empty, empty
    if query.device.type == "cpu":
        out, lse = _C._scaled_dot_product_flash_attention_for_cpu(
            query, key, value, dropout_p, is_causal, attn_mask=None, scale=scale
        )
        return out, lse, empty, empty, max_q, max_k, rng_state, empty, empty
    if scale is not None:
        head_dim = query.size(-1)
        query = query * (scale * math.sqrt(head_dim))
    out = _C.scaled_dot_product_attention(query, key, value, is_causal, 1)
    return out, empty, empty, empty, max_q, max_k, rng_state, empty, empty


def _flash_attention_quantized_adapter(
    query: Any,
    key: Any,
    value: Any,
    q_descale: Any = None,
    k_descale: Any = None,
    v_descale: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Any = None,
) -> tuple[Any, ...]:
    """Low-precision entry point, a contract for a kernel this build does not carry.

    The scores here are formed from eight-bit inputs, and a kernel that reads
    them in that precision keeps the rounding that precision implies.  A
    composite would rescale the inputs into float first and land somewhere
    else, so there is nothing to adapt: the answer is the refusal.
    """
    del query, key, value, q_descale, k_descale, v_descale
    del dropout_p, is_causal, return_debug_mask, scale
    raise NotImplementedError(
        "low-precision flash attention needs a low-precision attention kernel; "
        'register one with tensorplay.nn.attention.activate_flash_attention_impl("FA3")'
    )


# The low-precision entry point hangs off the full-precision one because that is
# how a caller reaches it: the two are the same contract at two precisions, so
# the name is a suffix on the same object rather than a name of its own.
_flash_attention_adapter.quantized = _flash_attention_quantized_adapter


def _batched_math_attention(query: Any, key: Any, value: Any, keep: Any, scale: Any) -> tuple[Any, Any]:
    """The composite over batched inputs in the sequence-major order.

    The head sits in the third axis there and in the second axis where the
    product is formed, so it moves across and the result moves back.  The
    logsumexp is handed back in the order it was computed in, which is the
    head-second order rather than the order the inputs arrived in.
    """
    out, lse = _math_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        None if keep is None else _additive_mask(keep, query.dtype),
        scale,
    )
    return out.transpose(1, 2), lse


def _heads_math_attention(query: Any, key: Any, value: Any, keep: Any, scale: Any) -> tuple[Any, Any]:
    """The composite over batched inputs already in the head-major order.

    Nothing moves here: this is the order the product is formed in, so the
    output and the logsumexp both come back where the inputs were.
    """
    return _math_attention(
        query,
        key,
        value,
        None if keep is None else _additive_mask(keep, query.dtype),
        scale,
    )


def _efficient_attention_forward_adapter(
    query: Any,
    key: Any,
    value: Any,
    bias: Any = None,
    cu_seqlens_q: Any = None,
    cu_seqlens_k: Any = None,
    max_seqlen_q: Any = None,
    max_seqlen_k: Any = None,
    dropout_p: float = 0.0,
    custom_mask_type: int = _NO_CUSTOM_MASK,
    compute_log_sumexp: bool = False,
    *,
    scale: Any = None,
    seqlen_k: Any = None,
    window_size: Any = None,
) -> tuple[Any, ...]:
    """Attention composite for the memory-efficient attention contract.

    Sequence-major on the way in and on the way out with the head in the third
    axis, and a six-field result whose second field is the logsumexp over the
    keys each query saw.  Asking for that field is what says a caller is about
    to train, since the backward pass needs the normalizing constant; a caller
    that only reads the output gets an empty one instead of paying for it.

    The two seed fields go unread by every caller in this tree, so they are
    reported as a fixed zero rather than sampled.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("memory-efficient attention: dropout > 0 is not supported in this build")
    if bias is not None:
        raise NotImplementedError(
            "memory-efficient attention: an explicit attention bias is not supported in this build"
        )
    if seqlen_k is not None or window_size is not None:
        raise NotImplementedError(
            "memory-efficient attention: per-batch key lengths and sliding windows are not supported in this build"
        )
    _check_layout(query, key, value, packed=cu_seqlens_q is not None, head_axis=2)
    device = query.device
    if cu_seqlens_q is None:
        keep = None
        if custom_mask_type != _NO_CUSTOM_MASK:
            keep = _causal_keep(query.size(1), key.size(1), custom_mask_type, device)
        out, lse = _batched_math_attention(query, key, value, keep, scale)
        if not compute_log_sumexp:
            lse = tensorplay.empty((0, lse.size(1), lse.size(2)), dtype=lse.dtype, device=device)
    else:
        out, lse = _packed_attention(query, key, value, cu_seqlens_q, cu_seqlens_k, scale, -1, -1)
    seed = tensorplay.zeros((), dtype=tensorplay.int64, device=device)
    length_q = query.size(1) if cu_seqlens_q is None else query.size(0)
    length_k = key.size(1) if cu_seqlens_k is None else key.size(0)
    return out, lse, seed, seed, length_q, length_k


def _flash_attention_forward_adapter(
    query: Any,
    key: Any,
    value: Any,
    cum_seq_q: Any = None,
    cum_seq_k: Any = None,
    max_q: Any = None,
    max_k: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Any = None,
    window_size_left: Any = None,
    window_size_right: Any = None,
    seqused_k: Any = None,
    alibi_slopes: Any = None,
    block_table: Any = None,
    num_splits: Any = None,
) -> tuple[Any, ...]:
    """Attention composite for the packed-and-batched flash attention contract.

    A cumulative-length table means the inputs are packed, one sequence after
    another with nothing marking the joins.  Without one they are already
    batched and are taken in the sequence-major order, head in the third axis.
    Either way the result is the output, the per-query logsumexp, the dropout
    state, an unused scalar, and the debug mask.
    """
    if _native_kernel_serves("_flash_attention_forward", query.device.type):
        return _prefer_native_kernel(
            "_flash_attention_forward",
            _flash_attention_forward_adapter,
            query,
            (query, key, value, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, return_debug_mask),
            {
                "scale": scale,
                "window_size_left": window_size_left,
                "window_size_right": window_size_right,
                "seqused_k": seqused_k,
                "alibi_slopes": alibi_slopes,
                "block_table": block_table,
                "num_splits": num_splits,
            },
        )
    if _native_kernel_serves("_efficient_attention_forward", query.device.type):
        return _prefer_native_kernel(
            "_efficient_attention_forward",
            _efficient_attention_forward_adapter,
            query,
            (
                query,
                key,
                value,
                bias,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                custom_mask_type,
                compute_log_sumexp,
            ),
            {"scale": scale, "seqlen_k": seqlen_k, "window_size": window_size},
        )
    del return_debug_mask, alibi_slopes, num_splits
    if dropout_p != 0.0:
        raise NotImplementedError("flash attention: dropout > 0 is not supported in this build")
    for name, arg in (("seqused_k", seqused_k), ("block_table", block_table)):
        if arg is not None:
            raise NotImplementedError(
                f"flash attention: {name} names a cached or paged key/value store, which is not supported in this build"
            )
    if (cum_seq_q is None) != (cum_seq_k is None):
        raise ValueError("cumulative query and key lengths must both be given or both be absent")
    _check_layout(query, key, value, packed=cum_seq_q is not None, head_axis=2)
    window_left, window_right = _resolve_window(is_causal, window_size_left, window_size_right)
    if cum_seq_q is not None:
        out, lse = _packed_attention(query, key, value, cum_seq_q, cum_seq_k, scale, window_left, window_right)
    else:
        keep = _window_keep(query.size(1), key.size(1), window_left, window_right, query.device)
        out, lse = _batched_math_attention(query, key, value, keep, scale)
    rng_state = tensorplay.zeros((2,), dtype=tensorplay.uint64, device=query.device)
    unused = tensorplay.zeros((), dtype=tensorplay.uint64, device=query.device)
    debug = tensorplay.empty(0, dtype=query.dtype, device=query.device)
    return out, lse, rng_state, unused, debug


def _flash_attention_inplace_adapter(
    out: Any,
    query: Any,
    key: Any,
    value: Any,
    cum_seq_q: Any = None,
    cum_seq_k: Any = None,
    max_q: Any = None,
    max_k: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Any = None,
    window_size_left: Any = None,
    window_size_right: Any = None,
    seqused_k: Any = None,
    alibi_slopes: Any = None,
    block_table: Any = None,
    num_splits: Any = None,
) -> Any:
    """The packed-and-batched flash contract writing into a caller's buffer.

    The same computation as :func:`_flash_attention_forward_adapter`, with the
    output landing in a tensor the caller already owns so a decoding loop can
    hand one buffer to step after step.  Only the logsumexp comes back as a
    value; the output is the buffer, which is what the caller already has.
    """
    if _native_kernel_serves("_flash_attention_forward_no_dropout_inplace", out.device.type):
        packet = _load_native_overloads().get("_flash_attention_forward_no_dropout_inplace")
        if packet is not None:
            return packet(
                out,
                query,
                key,
                value,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                dropout_p,
                is_causal,
                return_debug_mask,
                scale=scale,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                seqused_k=seqused_k,
                alibi_slopes=alibi_slopes,
                block_table=block_table,
                num_splits=num_splits,
            )
    result = _flash_attention_forward_adapter(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
        block_table=block_table,
        num_splits=num_splits,
    )
    out.copy_(result[0])
    return result[1]


def _cudnn_attention_forward_adapter(
    query: Any,
    key: Any,
    value: Any,
    attn_bias: Any = None,
    cum_seq_q: Any = None,
    cum_seq_k: Any = None,
    max_q: Any = None,
    max_k: Any = None,
    compute_logsumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Any = None,
    seqused_k: Any = None,
    block_table: Any = None,
) -> tuple[Any, ...]:
    """Attention composite for the contract shaped like the cuDNN one.

    Batched inputs arrive with the head in the second axis, which is where the
    product is formed, and a cumulative-length table means they are packed
    instead.  Either way the result is nine fields, and unlike the flash
    contract it hands the cumulative-length tables back unchanged: a caller
    that asked for the packed layout gets to see which layout it got, and the
    tables it passed in are what describe it.  The two seed fields go unread by
    every caller in this tree and are reported as a fixed zero.
    """
    if _native_kernel_serves("_cudnn_attention_forward", query.device.type):
        return _prefer_native_kernel(
            "_cudnn_attention_forward",
            _cudnn_attention_forward_adapter,
            query,
            (
                query,
                key,
                value,
                attn_bias,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                compute_logsumexp,
                dropout_p,
                is_causal,
                return_debug_mask,
            ),
            {"scale": scale, "seqused_k": seqused_k, "block_table": block_table},
        )
    del return_debug_mask
    if dropout_p != 0.0:
        raise NotImplementedError("cuDNN attention: dropout > 0 is not supported in this build")
    if attn_bias is not None:
        raise NotImplementedError("cuDNN attention: an explicit attention bias is not supported in this build")
    for name, arg in (("seqused_k", seqused_k), ("block_table", block_table)):
        if arg is not None:
            raise NotImplementedError(
                f"cuDNN attention: {name} names a cached or paged key/value store, which is not supported in this build"
            )
    _check_layout(query, key, value, packed=cum_seq_q is not None, head_axis=1)
    window_left, window_right = _resolve_window(is_causal, None, None)
    if cum_seq_q is not None:
        out, lse = _packed_attention(query, key, value, cum_seq_q, cum_seq_k, scale, window_left, window_right)
    else:
        keep = _window_keep(query.size(2), key.size(2), window_left, window_right, query.device)
        out, lse = _heads_math_attention(query, key, value, keep, scale)
    if not compute_logsumexp:
        lse = tensorplay.empty((0, *lse.shape[1:]), dtype=lse.dtype, device=lse.device)
    device = query.device
    seed = tensorplay.zeros((), dtype=tensorplay.int64, device=device)
    empty_q = tensorplay.empty(0, dtype=query.dtype, device=device)
    empty_k = tensorplay.empty(0, dtype=key.dtype, device=device)
    length_q = query.size(2) if cum_seq_q is None else query.size(0)
    length_k = key.size(2) if cum_seq_k is None else key.size(0)
    return (
        out,
        lse,
        cum_seq_q if cum_seq_q is not None else empty_q,
        cum_seq_k if cum_seq_k is not None else empty_k,
        length_q if max_q is None else max_q,
        length_k if max_k is None else max_k,
        seed,
        seed,
        empty_q,
    )


#: The dispatch key a device's kernels register under.  A composite is chosen
#: per call because a kernel may be registered for one device and not another,
#: and the answer has to be the one for the tensors in hand.
_DISPATCH_KEY_FOR_DEVICE = {"cpu": "CPU", "cuda": "CUDA"}


def _native_kernel_serves(opname: str, device_type: str) -> bool:
    """Whether a kernel of this build answers ``opname`` on ``device_type``.

    An op the contract declares but no backend implements is served by the
    composite below.  A backend that later grows one should take the call back
    without the composite having to be told, so each entry asks this first and
    only composes when the answer is no.
    """
    key = _DISPATCH_KEY_FOR_DEVICE.get(device_type)
    if key is None:
        return False
    try:
        return bool(_C._dispatch_has_kernel_for_dispatch_key(opname, key))
    except Exception:
        # An op the dispatch table has never heard of has no kernel anywhere.
        return False


def _prefer_native_kernel(opname: str, composite: Any, device_of: Any, args, kwargs):
    """Hand the call to a registered kernel, or compose it here.

    The dispatcher is asked first because a kernel is the whole point of the
    contract: it answers in one pass without forming a score matrix, and it
    answers for the device the tensors are on rather than for whichever device
    happens to be compiled in.  Only when nothing is registered does the
    composite run, and then the answer is the same either way -- which is what
    makes it safe for the composite to be the default.
    """
    probe = None
    for candidate in args:
        if isinstance(candidate, tensorplay.Tensor):
            probe = candidate
            break
    if probe is not None and _native_kernel_serves(opname, probe.device.type):
        packet = _load_native_overloads().get(opname)
        if packet is not None:
            return packet(*args, **kwargs)
    return composite(*args, **kwargs)


# Composite contracts declared in the op schema set that have no dedicated
# kernel registration in this build.  Each entry adapts over the kernels
# that do exist; anything without an entry resolves through the native
# dispatcher and raises its own "kernel not found" error.
_NATIVE_FALLBACKS: dict[str, Any] = {
    "_scaled_dot_product_flash_attention": _flash_attention_adapter,
    "_efficient_attention_forward": _efficient_attention_forward_adapter,
    "_flash_attention_forward": _flash_attention_forward_adapter,
    "_flash_attention_forward_no_dropout_inplace": _flash_attention_inplace_adapter,
    "_cudnn_attention_forward": _cudnn_attention_forward_adapter,
}


# Namespace of the operators declared in the op contract (config/); an
# interop identifier the public ``tensorplay.ops.<ns>`` surface already uses.
NATIVE_NAMESPACE = "tp"


class OpOverload:
    """One operator overload of the op contract (``add.Tensor``).

    Calling it runs exactly that overload through the dispatcher.  Dispatch
    modes receive these objects as ``func``; ``_schema`` describes the
    arguments, returns and alias annotations.
    """

    # The dunder identity attributes are written per instance; ``__dict__``
    # carries them because CPython forbids ``__name__``/``__qualname__``/
    # ``__module__`` inside ``__slots__`` (they collide with reserved
    # class-level attributes).
    __slots__ = (
        "_schema",
        "_overloadpacket",
        "_overloadname",
        "_opname",
        "_key",
        "_tags",
        "__weakref__",
        "__dict__",
    )

    def __init__(self, packet: "OpOverloadPacket", key: str, schema: Any, tags: tuple[str, ...]) -> None:
        self._schema = schema
        self._overloadpacket = packet
        self._overloadname = schema.overload_name or "default"
        self._opname = schema.name
        self._key = key
        self._tags = tags
        self.__name__ = f"{schema.name}.{self._overloadname}"
        self.__qualname__ = self.__name__
        self.__module__ = f"tensorplay.ops.{schema.namespace}"

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        # While a graph is being captured, an overload reached with a
        # symbolic argument is recorded rather than run: the arguments are
        # descriptions of values that do not exist yet, so there is nothing to
        # compute, and running it would ask the operator for the type of
        # something that has not been made.  ``capture_call`` is what already
        # decides whether an argument is symbolic, so it is what decides this
        # too -- asking it here rather than deciding again would be the same
        # question with two answers.
        from .graph import capture_call as _capture_call

        captured = _capture_call(self, args, kwargs)
        if captured is not None:
            return captured
        return _C._call_overload(self._key, args, kwargs)

    @property
    def overloadpacket(self) -> "OpOverloadPacket":
        return self._overloadpacket

    @property
    def op(self) -> "OpOverload":
        return self

    @property
    def namespace(self) -> str:
        return self._schema.namespace

    @property
    def tags(self) -> tuple[str, ...]:
        return self._tags

    @property
    def is_view(self) -> bool:
        return self._schema._is_view_op()

    def name(self) -> str:
        return f"{self._schema.namespace}::{self._key}"

    def has_kernel_for_dispatch_key(self, key: str) -> bool:
        return bool(_C._dispatch_has_kernel_for_dispatch_key(self._key, key))

    def __repr__(self) -> str:
        return f"<OpOverload(op='{self._schema.namespace}.{self._opname}', overload='{self._overloadname}')>"

    def __str__(self) -> str:
        return f"{self._schema.namespace}.{self._opname}.{self._overloadname}"

    def __reduce__(self) -> Any:
        return (_overload_for_dispatch, (self._key,))

    def __deepcopy__(self, memo: Any) -> "OpOverload":
        return self


class OpOverloadPacket:
    """All overloads of one operator name (``add``).

    Attribute access yields an overload (``.Tensor``, ``.default``); calling
    the packet resolves the overload from the arguments.
    """

    def __init__(self, namespace: str, name: str) -> None:
        self._namespace = namespace
        self._opname = name
        self._qualified_op_name = f"{namespace}::{name}"
        self.__name__ = name
        self.__qualname__ = name
        self.__module__ = f"tensorplay.ops.{namespace}"
        self._overloads: dict[str, OpOverload] = {}

    def _add(self, overload: OpOverload) -> None:
        self._overloads[overload._overloadname] = overload

    def overloads(self) -> list[str]:
        return list(self._overloads)

    def __getattr__(self, name: str) -> OpOverload:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self._overloads[name]
        except KeyError:
            raise AttributeError(f"'{self._qualified_op_name}' has no overload named '{name}'") from None

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        resolver = getattr(_C, self._opname, None)
        if resolver is not None:
            return resolver(*args, **kwargs)
        # No public binding: the single declared overload, or the first
        # whose arguments bind.
        errors: list[str] = []
        for overload in self._overloads.values():
            try:
                return overload(*args, **kwargs)
            except TypeError as exc:
                errors.append(f"{overload}: {exc}")
        raise TypeError(f"no overload of {self._qualified_op_name} accepts these arguments:\n  " + "\n  ".join(errors))

    def __repr__(self) -> str:
        return f"<OpOverloadPacket(op='{self._namespace}.{self._opname}')>"

    def __str__(self) -> str:
        return f"{self._namespace}.{self._opname}"

    def __reduce__(self) -> Any:
        return (_packet_for, (self._opname,))


_packets: dict[str, OpOverloadPacket] | None = None
_overloads_by_key: dict[str, OpOverload] = {}


def _load_native_overloads() -> dict[str, OpOverloadPacket]:
    global _packets
    if _packets is not None:
        return _packets
    from ._function_schema import parse_schema

    packets: dict[str, OpOverloadPacket] = {}
    entries = getattr(_C, "_python_dispatch_entries", None)
    for key, schema_text, _names, _npos, tags in entries() if entries else ():
        schema = parse_schema(schema_text, namespace=NATIVE_NAMESPACE)
        packet = packets.get(schema.name)
        if packet is None:
            packet = packets[schema.name] = OpOverloadPacket(NATIVE_NAMESPACE, schema.name)
        overload = OpOverload(packet, key, schema, tuple(t for t in tags.split(",") if t))
        packet._add(overload)
        _overloads_by_key[key] = overload
    _packets = packets
    return packets


def _packet_for(name: str) -> OpOverloadPacket:
    return _load_native_overloads()[name]


def _overload_for_dispatch(key: str) -> OpOverload:
    """The interned overload object for a dispatcher key (``add.Tensor``)."""

    _load_native_overloads()
    return _overloads_by_key[key]


class _OpNamespace(types.ModuleType):
    """Attribute-access packet for one operator namespace (``ns``)."""

    def __init__(self, ns: str) -> None:
        super().__init__(f"tensorplay.ops.{ns}")
        self.ns = ns

    def __getattr__(self, opname: str) -> Any:
        # Native extension modules registered via load_library win: they are
        # real submodules placed on this namespace.
        own = self.__dict__.get(opname)
        if own is not None:
            return own
        if self.ns == NATIVE_NAMESPACE:
            # Composite fallbacks come first: they wrap the fused kernels of
            # this build for contracts without their own registration.
            fallback = _NATIVE_FALLBACKS.get(opname)
            if fallback is not None:
                return fallback
            packet = _load_native_overloads().get(opname)
            if packet is not None:
                setattr(self, opname, packet)
                return packet
            native = getattr(_C, opname, None)
            if native is not None:
                return native
        full_name = f"{self.ns}::{opname}"
        if tensorplay.library.has_op(full_name):
            return tensorplay.library.get_op(full_name)
        raise AttributeError(
            f"No operator {full_name!r} is registered; define it with "
            f'tensorplay.library.custom_op("{full_name}") or load its '
            "extension library via tensorplay.ops.load_library"
        )


class _Ops(types.ModuleType):
    """The ``tensorplay.ops`` root namespace."""

    __file__ = "_ops.py"

    def __getattr__(self, name: str) -> _OpNamespace:
        if name.startswith("_"):
            raise AttributeError(name)
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        return namespace

    @property
    def load_library(self) -> Any:
        return _C.ops.load_library

    @property
    def loaded_libraries(self) -> Any:
        return getattr(_C.ops, "loaded_libraries")


ops = _Ops("tensorplay.ops")
