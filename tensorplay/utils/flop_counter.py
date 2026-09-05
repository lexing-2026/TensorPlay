# mypy: allow-untyped-defs
"""FLOP counting registry.

``flop_registry`` maps a callable (a custom op or higher-order operator) to a
formula that computes its FLOP count from the shapes of its inputs.  The
formulas below cover the variable-length attention family; the registry is
populated at import time by the modules that own the corresponding ops.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import tensorplay
from tensorplay import Tensor

__all__ = [
    "flop_registry",
    "_varlen_attn_backward_flop",
    "_varlen_attn_forward_flop",
    "_varlen_attn_out_flop",
]


flop_registry: dict[Any, Any] = {}


def bmm_flop(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> int:
    """Count flops for the bmm operation."""
    b, m, k = a_shape
    b2, k2, n = b_shape
    if b != b2:
        raise AssertionError(
            f"bmm: batch dimensions must match (b == b2), got {b} and {b2}"
        )
    if k != k2:
        raise AssertionError(
            f"bmm: inner dimensions must match (k == k2), got {k} and {k2}"
        )
    # NB: Should be 2 * k - 1 technically for FLOPs.
    flop = b * m * n * 2 * k
    return flop


def sdpa_flop_count(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
) -> int:
    """
    Count flops for self-attention.

    Supports GQA (grouped-query attention) where key/value have fewer heads
    than the query. The kernel broadcasts KV heads to match query heads.
    """
    b, h_q, s_q, d_q = query_shape
    _b2, h_kv, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    if not (b == _b2 == _b3 and h_kv == _h3 and d_q == _d2 and s_k == _s3):
        raise AssertionError(
            f"sdpa_flop_count: query/key/value shapes are incompatible: "
            f"q={query_shape}, k={key_shape}, v={value_shape}"
        )
    if h_q < h_kv or h_q % h_kv != 0:
        raise AssertionError(
            f"sdpa_flop_count: query heads ({h_q}) must be a multiple of "
            f"key/value heads ({h_kv})"
        )
    total_flops = 0
    # q: [b, h_q, s_q, d_q] @ k: [b, h_q, d_q, s_k] -> scores: [b, h_q, s_q, s_k]
    total_flops += bmm_flop((b * h_q, s_q, d_q), (b * h_q, d_q, s_k))
    # scores: [b, h_q, s_q, s_k] @ v: [b, h_q, s_k, d_v] -> out: [b, h_q, s_q, d_v]
    total_flops += bmm_flop((b * h_q, s_q, s_k), (b * h_q, s_k, d_v))
    return total_flops


def sdpa_backward_flop_count(
    grad_out_shape: tuple[int, ...],
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
) -> int:
    b, h_q, s_q, d_q = query_shape
    _b2, h_kv, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    _b4, _h4, _s4, _d4 = grad_out_shape
    if not (b == _b2 == _b3 == _b4 and h_kv == _h3 and h_q == _h4):
        raise AssertionError(
            "sdpa_backward_flop_count: batch/heads mismatch among tensors"
        )
    if h_q < h_kv or h_q % h_kv != 0:
        raise AssertionError(
            f"sdpa_backward_flop_count: query heads ({h_q}) must be a multiple of "
            f"key/value heads ({h_kv})"
        )
    if not (d_q == _d2 and d_v == _d4 and s_k == _s3 and s_q == _s4):
        raise AssertionError(
            "sdpa_backward_flop_count: grad_out/value/key/query shapes are incompatible"
        )
    total_flops = 0
    # Step 1: We recompute the scores matrix.
    # q: [b, h_q, s_q, d_q] @ k: [b, h_q, d_q, s_k] -> scores: [b, h_q, s_q, s_k]
    total_flops += bmm_flop((b * h_q, s_q, d_q), (b * h_q, d_q, s_k))

    # Step 2: We propagate the gradients through the score @ v operation.
    # gradOut: [b, h_q, s_q, d_v] @ v: [b, h_q, d_v, s_k] -> gradScores: [b, h_q, s_q, s_k]
    total_flops += bmm_flop((b * h_q, s_q, d_v), (b * h_q, d_v, s_k))
    # scores: [b, h_q, s_k, s_q] @ gradOut: [b, h_q, s_q, d_v] -> gradV: [b, h_q, s_k, d_v]
    total_flops += bmm_flop((b * h_q, s_k, s_q), (b * h_q, s_q, d_v))

    # Step 3: We propagate the gradients through the k @ v operation
    # gradScores: [b, h_q, s_q, s_k] @ k: [b, h_q, s_k, d_q] -> gradQ: [b, h_q, s_q, d_q]
    total_flops += bmm_flop((b * h_q, s_q, s_k), (b * h_q, s_k, d_q))
    # q: [b, h_q, d_q, s_q] @ gradScores: [b, h_q, s_q, s_k] -> gradK: [b, h_q, d_q, s_k]
    total_flops += bmm_flop((b * h_q, d_q, s_q), (b * h_q, s_q, s_k))
    return total_flops


def _offsets_to_lengths(offsets: Tensor, max_len: int) -> list[int]:
    """
    If the offsets tensor is symbolic, we don't know the actual lengths.
    In that case, we can just assume the worst case; each batch has max length.
    """
    from tensorplay.graph.experimental.symbolic_shapes import (
        has_symbolic_sizes_strides,
    )

    if offsets.device.type != "meta" and not has_symbolic_sizes_strides(offsets):
        return offsets.diff().tolist()
    return [max_len] * (offsets.size(0) - 1)


def _unpack_flash_attention_nested_shapes(
    *,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_out: Tensor | None = None,
    cum_seq_q: Tensor | None,
    cum_seq_k: Tensor | None,
    max_q: int,
    max_k: int,
) -> Iterator[
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]
]:
    """
    Given inputs to a flash_attention_(forward|backward) kernel, this will
    handle behavior for ragged (packed) inputs by effectively unbinding the
    ragged batch and yielding the shapes for each sequence.

    In the case that this isn't a packed kernel, then it just yields the
    original shapes.
    """
    if cum_seq_q is not None:
        # The inputs are packed: shape (sum(sequence len), heads, dimension).
        # Convert to per-sequence (1, heads, seq_len, dimension) shapes.
        if len(key.shape) != 3:
            raise AssertionError(
                "sdpa_flop_count: expected key.shape to be 3-dimensional"
            )
        if len(value.shape) != 3:
            raise AssertionError(
                "sdpa_flop_count: expected value.shape to be 3-dimensional"
            )
        if grad_out is not None and grad_out.shape != query.shape:
            raise AssertionError(
                "sdpa_flop_count: grad_out.shape must match query.shape when provided"
            )
        _, h_q, d_q = query.shape
        _, h_k, d_k = key.shape
        _, h_v, d_v = value.shape
        if cum_seq_q is None:
            raise AssertionError("sdpa_flop_count: cum_seq_q must not be None")
        if cum_seq_k is None:
            raise AssertionError("sdpa_flop_count: cum_seq_k must not be None")
        if cum_seq_q.shape != cum_seq_k.shape:
            raise AssertionError(
                "sdpa_flop_count: cum_seq_q and cum_seq_k must have the same shape"
            )
        seq_q_lengths = _offsets_to_lengths(cum_seq_q, max_q)
        seq_k_lengths = _offsets_to_lengths(cum_seq_k, max_k)
        for seq_q_len, seq_k_len in zip(seq_q_lengths, seq_k_lengths, strict=True):
            new_query_shape = (1, h_q, seq_q_len, d_q)
            new_key_shape = (1, h_k, seq_k_len, d_k)
            new_value_shape = (1, h_v, seq_k_len, d_v)
            new_grad_out_shape = new_query_shape if grad_out is not None else None
            yield new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape
        return

    yield (
        query.shape,
        key.shape,
        value.shape,
        grad_out.shape if grad_out is not None else None,
    )


def _varlen_attn_forward_flop(
    query,
    key,
    value,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    *args,
    out_val=None,
    **kwargs,
) -> int:
    """Count flops for varlen_attn forward."""
    sizes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        cum_seq_q=cu_seq_q,
        cum_seq_k=cu_seq_k if cu_seq_k is not None else cu_seq_q,
        max_q=max_q,
        max_k=max_k,
    )
    return sum(
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


def _varlen_attn_out_flop(
    out,
    query,
    key,
    value,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    *args,
    out_val=None,
    **kwargs,
) -> int:
    """Count flops for varlen_attn_out forward."""
    return _varlen_attn_forward_flop(
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
    )


def _varlen_attn_backward_flop(
    grad_out,
    query,
    key,
    value,
    out,
    lse,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    *args,
    out_val=None,
    **kwargs,
) -> int:
    """Count flops for varlen_attn backward."""
    sizes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        grad_out=grad_out,
        cum_seq_q=cu_seq_q,
        cum_seq_k=cu_seq_k,
        max_q=max_q,
        max_k=max_k,
    )
    return sum(
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in sizes
    )
