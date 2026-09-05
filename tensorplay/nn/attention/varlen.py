"""
Variable-length attention implementation using Flash Attention.

This module provides a high-level Python interface for variable-length attention
that calls into the optimized Flash Attention kernels.
"""

import logging
from functools import lru_cache
from typing import Any, NamedTuple

import tensorplay
from tensorplay._C import _SDPBackend as SDPBackend

from . import _is_sdp_priority_order_active
from ._utils import _empty_with_matching_layout


log = logging.getLogger(__name__)

__all__ = ["varlen_attn", "varlen_attn_out", "AuxRequest"]

# Custom op schemas do not support enum arguments, so pass SDPBackend values as ints.
_FLASH_ATTENTION_BACKEND = SDPBackend.FLASH_ATTENTION.value
_CUDNN_ATTENTION_BACKEND = SDPBackend.CUDNN_ATTENTION.value


def _normalize_window_size(window_size: list[int] | None) -> list[int]:
    if window_size is None:
        window_size = [-1, -1]

    if len(window_size) != 2:
        raise ValueError(f"window_size must have length 2, got {len(window_size)}")
    return window_size


def _validate_scale(scale: float | None) -> None:
    """Require scales supported by the fused varlen backends."""
    # This form also rejects NaN, unlike scale <= 0.
    if scale is not None and not scale > 0:
        raise ValueError(f"scale must be greater than 0, got {scale}")


@tensorplay.compiler.assume_constant_result
def _get_sdp_priority_order() -> list[int]:
    """Capture varlen backend priority at trace time."""
    if _is_sdp_priority_order_active():
        return tensorplay._C._get_sdp_priority_order()
    return [_CUDNN_ATTENTION_BACKEND, _FLASH_ATTENTION_BACKEND]


@lru_cache(maxsize=8)
@tensorplay.compiler.assume_constant_result
def _should_use_cudnn(device_index: int) -> bool:
    """Cache device capability check to avoid repeated calls."""
    if tensorplay.version.hip is not None:
        return False
    cudnn_version = tensorplay.backends.cudnn.version()
    if cudnn_version is None or cudnn_version < 91800:
        return False
    major_cap = tensorplay.cuda.get_device_capability(device_index)[0]
    if major_cap == 9 or major_cap == 10:
        return True
    return False


def _cudnn_rejection_reasons(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor | None,
    max_q: int,
    window_size: list[int],
    enable_gqa: bool = False,
    seqused_k: tensorplay.Tensor | None = None,
    block_table: tensorplay.Tensor | None = None,
    num_splits: int | None = None,
) -> list[str]:
    """Return the constraints preventing cuDNN varlen attention."""
    reasons = []
    if not query.is_cuda:
        reasons.append("query must be on CUDA")
    elif not _should_use_cudnn(query.device.index):
        reasons.append("cuDNN >= 9.18 on SM90 or SM100 is required")
    if max_q <= 128:
        reasons.append("max_q must be greater than 128")
    if query.dtype not in (tensorplay.float16, tensorplay.bfloat16):
        reasons.append("query dtype must be float16 or bfloat16")
    if query.shape[-1] % 8 != 0 or value.shape[-1] % 8 != 0:
        reasons.append("query and value head dimensions must be divisible by 8")
    if window_size == [-1, 0]:
        if cu_seq_q is not cu_seq_k:
            reasons.append(
                "causal attention requires the same cu_seq tensor for Q and K"
            )
        if seqused_k is not None or block_table is not None:
            reasons.append("causal attention does not support a KV cache")
    elif window_size != [-1, -1]:
        reasons.append("window_size must be (-1, -1) or causal (-1, 0)")
    if enable_gqa or query.size(-2) != key.size(-2):
        reasons.append("GQA is not supported")
    if num_splits is not None:
        reasons.append("num_splits is not supported")
    if block_table is not None and seqused_k is None:
        reasons.append("block_table requires seqused_k")
    return reasons


def _select_backend(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor | None,
    max_q: int,
    window_size: list[int],
    enable_gqa: bool = False,
    seqused_k: tensorplay.Tensor | None = None,
    block_table: tensorplay.Tensor | None = None,
    num_splits: int | None = None,
) -> int:
    """Select the first eligible varlen backend in the SDPA priority order."""
    cudnn_enabled = tensorplay._C._get_cudnn_sdp_enabled()
    flash_enabled = tensorplay._C._get_flash_sdp_enabled()
    cudnn_reasons = (
        _cudnn_rejection_reasons(
            query,
            key,
            value,
            cu_seq_q,
            cu_seq_k,
            max_q,
            window_size,
            enable_gqa,
            seqused_k,
            block_table,
            num_splits,
        )
        if cudnn_enabled
        else []
    )
    cudnn_eligible = cudnn_enabled and not cudnn_reasons
    for backend in _get_sdp_priority_order():
        if backend == _CUDNN_ATTENTION_BACKEND and cudnn_eligible:
            return backend
        if backend == _FLASH_ATTENTION_BACKEND and flash_enabled:
            return backend
    if cudnn_enabled:
        constraints = "\n  - ".join(cudnn_reasons)
        raise RuntimeError(
            "SDPBackend.CUDNN_ATTENTION was requested for varlen_attn, but its "
            f"constraints are not satisfied:\n  - {constraints}"
        )
    raise RuntimeError(
        "No viable backend for varlen_attn. Enable SDPBackend.FLASH_ATTENTION "
        "or SDPBackend.CUDNN_ATTENTION with sdpa_kernel()."
    )


class AuxRequest(NamedTuple):
    """
    Request which auxiliary outputs to compute from varlen_attn.

    Each field is a boolean indicating whether that auxiliary output should be computed.
    """

    lse: bool = False


@tensorplay.library.custom_op("tensorplay_attn::_varlen_attn", mutates_args={})
def _varlen_attn(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor | None,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    seqused_k: tensorplay.Tensor | None = None,
    block_table: tensorplay.Tensor | None = None,
    num_splits: int | None = None,
    backend: int = _FLASH_ATTENTION_BACKEND,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
    """
    Private custom op for variable-length attention.

    This is the internal implementation. Users should use the public varlen_attn function instead.
    """
    window_size = _normalize_window_size(window_size)

    if backend == _CUDNN_ATTENTION_BACKEND:
        log.info("Using cuDNN backend for varlen_attn")
        result = tensorplay.ops.aten._cudnn_attention_forward(
            query=query,
            key=key,
            value=value,
            attn_bias=None,
            cum_seq_q=cu_seq_q,
            cum_seq_k=cu_seq_k,
            max_q=max_q,
            max_k=max_k,
            compute_log_sumexp=True,
            dropout_p=0.0,  # dropout_p hardcoded to 0.0
            is_causal=is_causal,
            return_debug_mask=False,  # return_debug_mask
            scale=scale,
            seqused_k=seqused_k,
            block_table=block_table,
        )
        # cuDNN returns: (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask)
        output, softmax_lse, rng_state = result[0], result[1], result[6]
    elif backend == _FLASH_ATTENTION_BACKEND:
        log.info("Using Flash Attention backend for varlen_attn")
        output, softmax_lse, rng_state, _, _ = tensorplay.ops.aten._flash_attention_forward(
            query,
            key,
            value,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            0.0,  # dropout_p hardcoded to 0.0
            is_causal,
            return_debug_mask=False,
            scale=scale,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            seqused_k=seqused_k,
            block_table=block_table,
            num_splits=num_splits,
        )
    else:
        raise AssertionError(f"Unsupported varlen attention backend: {backend}")

    rng_state_ = tensorplay.zeros(
        (2,), dtype=tensorplay.uint64, device=query.device
    )  # hardcoded since dropout is hardcoded to 0
    return output, softmax_lse, rng_state_


@_varlen_attn.register_fake
def _varlen_attn_fake(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor | None,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    seqused_k: tensorplay.Tensor | None = None,
    block_table: tensorplay.Tensor | None = None,
    num_splits: int | None = None,
    backend: int = _FLASH_ATTENTION_BACKEND,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
    """
    Fake implementation for meta tensor computation and tracing.

    Based on the 3D varlen path:
    - query shape: (total, num_heads, head_dim)
    - logsumexp shape: (num_heads, total_q)
    """
    window_size = _normalize_window_size(window_size)

    output = _empty_with_matching_layout(query, (*query.shape[:-1], value.size(-1)))

    # For varlen path: logsumexp shape is (num_heads, total_q)
    total_q = query.size(0)
    num_heads = query.size(1)
    logsumexp = tensorplay.empty(
        (num_heads, total_q), dtype=tensorplay.float32, device=query.device
    )

    rng_state = tensorplay.empty((2,), dtype=tensorplay.uint64, device=query.device)

    return output, logsumexp, rng_state


def varlen_attn(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor | None,
    max_q: int,
    max_k: int,
    *,
    return_aux: AuxRequest | None = None,
    scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    enable_gqa: bool = False,
    seqused_k: tensorplay.Tensor | None = None,
    block_table: tensorplay.Tensor | None = None,
    num_splits: int | None = None,
) -> tensorplay.Tensor | tuple[tensorplay.Tensor, tensorplay.Tensor]:
    r"""Compute variable-length attention using Flash Attention.

    This function is similar to scaled_dot_product_attention but optimized for
    variable-length sequences using cumulative sequence position tensors.
    Backend enablement follows :func:`tensorplay.nn.attention.sdpa_kernel`. By default,
    eligible cuDNN is preferred over Flash; ``set_priority=True`` overrides this order.

    Args:
        query (Tensor): Query tensor; shape :math:`(T_q, H_q, D)`
        key (Tensor): Key tensor; shape :math:`(T_k, H_{kv}, D)`, or
            :math:`(\text{total\_pages}, \text{page\_size}, H_{kv}, D)` when ``block_table`` is provided.
        value (Tensor): Value tensor; shape :math:`(T_k, H_{kv}, D)`, or
            :math:`(\text{total\_pages}, \text{page\_size}, H_{kv}, D)` when ``block_table`` is provided.
        cu_seq_q (Tensor): Cumulative sequence positions for queries; shape :math:`(N+1,)`
        cu_seq_k (Tensor): Cumulative sequence positions for keys/values; shape :math:`(N+1,)`
        max_q (int): Maximum query sequence length in the batch.
        max_k (int): Maximum key/value sequence length in the batch.
        return_aux (Optional[AuxRequest]): If not None and ``return_aux.lse`` is True, also returns the logsumexp tensor.
        scale (float, optional): Positive scaling factor for attention scores.
        window_size (tuple[int, int], optional): Window size for sliding window attention as (left, right).
            Use (-1, -1) for full attention (default), (-1, 0) for causal attention,
            or (W, 0) for causal attention with sliding window of size W.
        enable_gqa (bool): If set to True, enables Grouped Query Attention (GQA)
            and allows key/value to have fewer heads than query.
            Each KV head is shared by a group of :math:`H_q / H_{kv}` query heads,
            so :math:`H_q` must be divisible by :math:`H_{kv}`.
            Default is False.
        seqused_k (Tensor, optional): Number of valid KV tokens per batch element; shape :math:`(N,)`.
            When set, only the first ``seqused_k[i]`` tokens in the key/value sequence for batch
            element *i* participate in attention. Useful for KV-cache decoding where the cache slot
            is larger than the actual sequence. Inference-only (not supported in backward).
        block_table (Tensor, optional): Block table for paged KV cache; shape
            :math:`(N, \text{max\_pages\_per\_seq})`, dtype ``int32``.
            Requires ``seqused_k``. Inference-only (not supported in backward).

            When ``block_table`` is provided, ``key`` and ``value`` are a "pool" of
            pages of tokens of KV data and the pages belong to any sequence/order.
            The ``block_table`` is what maps each sequence's logical chunks
            back to physical pages in this pool.

            ``seqused_k[i]`` tells the kernel how many tokens in sequence *i* are
            actually valid, since the last page is typically only partially filled.
        num_splits (int, optional): Number of splits for split-KV. Set to ``1``
            to disable split-KV which enables batch invariance. Split-KV
            parallelizes the key/value sequence dimension across multiple thread
            blocks and combines partial results. The split decision depends
            on ``max_k`` (the longest sequence in the batch), so different batch
            compositions can change the reduction order and produce different
            floating-point results for the same sequence. When this is disabled,
            bitwise identical outputs are guaranteed for a given sequence
            regardless of what other sequences are in the batch, at the
            cost of lower GPU utilization when there are few queries. When
            ``None`` (default), the kernel chooses automatically.

    Returns:
        output (Tensor): Output tensor from attention computation; shape :math:`(T_q, H_q, D)`.

        If ``return_aux`` is not None and ``return_aux.lse`` is True:
            lse (Tensor): Log-sum-exp of attention scores; shape :math:`(H_q, T_q)`.

    Shape legend:
        - :math:`N`: Batch size
        - :math:`T_q`: Total number of query tokens in the batch (sum of all query sequence lengths)
        - :math:`T_k`: Total number of key/value tokens in the batch (sum of all key/value sequence lengths)
        - :math:`H_q`: Number of query attention heads
        - :math:`H_{kv}`: Number of key/value attention heads (equal to :math:`H_q` unless GQA is enabled)
        - :math:`D`: Head dimension

    Example::

        >>> # xdoctest: +REQUIRES(env:TENSORPLAY_DOCTEST_CUDA)
        >>> batch_size, max_seq_len, embed_dim, num_heads = 2, 512, 1024, 16
        >>> head_dim = embed_dim // num_heads
        >>> seq_lengths = []
        >>> for _ in range(batch_size):
        ...     length = tensorplay.randint(1, max_seq_len // 64 + 1, (1,)).item() * 64
        ...     seq_lengths.append(min(length, max_seq_len))
        >>> seq_lengths = tensorplay.tensor(seq_lengths, device="cuda")
        >>> total_tokens = seq_lengths.sum().item()
        >>>
        >>> # Create packed query, key, value tensors
        >>> query = tensorplay.randn(
        ...     total_tokens, num_heads, head_dim, dtype=tensorplay.float16, device="cuda"
        ... )
        >>> key = tensorplay.randn(
        ...     total_tokens, num_heads, head_dim, dtype=tensorplay.float16, device="cuda"
        ... )
        >>> value = tensorplay.randn(
        ...     total_tokens, num_heads, head_dim, dtype=tensorplay.float16, device="cuda"
        ... )
        >>>
        >>> # Build cumulative sequence tensor
        >>> cu_seq = tensorplay.zeros(batch_size + 1, device="cuda", dtype=tensorplay.int32)
        >>> cu_seq[1:] = seq_lengths.cumsum(0)
        >>> max_len = seq_lengths.max().item()
        >>>
        >>> # Call varlen_attn
        >>> output = varlen_attn(
        ...     query, key, value, cu_seq, cu_seq, max_len, max_len
        ... )
    """

    num_heads_q = query.size(1)
    num_heads_k = key.size(2) if block_table is not None else key.size(1)
    if not enable_gqa and num_heads_q != num_heads_k:
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa and num_heads_q % num_heads_k != 0:
        raise ValueError(
            f"Expect number of query heads to be a multiple of kv heads for GQA "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}."
        )

    _validate_scale(scale)
    window_size_list = list(window_size)
    is_causal = window_size_list == [-1, 0]
    backend = _select_backend(
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        window_size_list,
        enable_gqa,
        seqused_k,
        block_table,
        num_splits,
    )
    out, lse, _ = tensorplay.ops.tensorplay_attn._varlen_attn(
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        scale,
        window_size_list,
        enable_gqa,
        seqused_k,
        block_table,
        num_splits,
        backend,
    )
    if return_aux is not None and return_aux.lse:
        return out, lse
    return out


@tensorplay.library.custom_op(
    "tensorplay_attn::_varlen_attn_out", mutates_args={"out"}
)
def _varlen_attn_out(
    out: tensorplay.Tensor,
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor | None,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    seqused_k: tensorplay.Tensor | None = None,
    block_table: tensorplay.Tensor | None = None,
    num_splits: int | None = None,
) -> tensorplay.Tensor:
    """
    Private custom op for variable-length attention with pre-allocated output.
    Same as _varlen_attn but writes the attention output into the provided out tensor.
    """
    window_size = _normalize_window_size(window_size)

    log.info("Using Flash Attention backend for varlen_attn_out")
    softmax_lse = tensorplay.ops.aten._flash_attention_forward_no_dropout_inplace(
        out,
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        0.0,  # dropout_p hardcoded to 0.0
        is_causal,
        False,  # return_debug_mask
        scale=scale,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        seqused_k=seqused_k,
        block_table=block_table,
        num_splits=num_splits,
    )

    return softmax_lse


@_varlen_attn_out.register_fake
def _varlen_attn_out_fake(
    out: tensorplay.Tensor,
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor | None,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    seqused_k: tensorplay.Tensor | None = None,
    block_table: tensorplay.Tensor | None = None,
    num_splits: int | None = None,
) -> tensorplay.Tensor:
    """
    Fake implementation for meta tensor computation and tracing.
    """
    total_q = query.size(0)
    num_heads = query.size(1)
    logsumexp = tensorplay.empty(
        (num_heads, total_q), dtype=tensorplay.float32, device=query.device
    )

    return logsumexp


def varlen_attn_out(
    out: tensorplay.Tensor,
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor | None,
    max_q: int,
    max_k: int,
    *,
    return_aux: AuxRequest | None = None,
    scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    enable_gqa: bool = False,
    seqused_k: tensorplay.Tensor | None = None,
    block_table: tensorplay.Tensor | None = None,
    num_splits: int | None = None,
) -> tensorplay.Tensor | tuple[tensorplay.Tensor, tensorplay.Tensor]:
    r"""Compute variable-length attention using Flash Attention with a pre-allocated output tensor.

    Same as :func:`varlen_attn` but writes the attention output into the provided ``out`` tensor
    instead of allocating a new one.

    """
    num_heads_q = query.size(1)
    num_heads_k = key.size(2) if block_table is not None else key.size(1)
    if not enable_gqa and num_heads_q != num_heads_k:
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa and num_heads_q % num_heads_k != 0:
        raise ValueError(
            f"Expect number of query heads to be a multiple of kv heads for GQA "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}."
        )

    _validate_scale(scale)
    if not tensorplay._C._get_flash_sdp_enabled():
        raise RuntimeError(
            "varlen_attn_out only supports SDPBackend.FLASH_ATTENTION; enable it "
            "with sdpa_kernel()."
        )

    is_causal = window_size == (-1, 0)
    lse = tensorplay.ops.tensorplay_attn._varlen_attn_out(
        out,
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        scale,
        list(window_size),
        enable_gqa,
        seqused_k,
        block_table,
        num_splits,
    )
    if return_aux is not None and return_aux.lse:
        return out, lse
    return out


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    (
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        scale,
        window_size,
        enable_gqa,
        seqused_k,
        block_table,
        num_splits,
        backend,
    ) = inputs
    out, lse, rng_state = output

    if seqused_k is not None:
        raise RuntimeError("seqused_k is an inference-only parameter.")
    if block_table is not None:
        raise RuntimeError("block_table is an inference-only parameter.")

    ctx.backend = backend
    ctx.mark_non_differentiable(lse, rng_state)
    ctx.save_for_backward(query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state)

    ctx.max_q = max_q
    ctx.max_k = max_k
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.window_size = window_size


@tensorplay.library.custom_op("tensorplay_attn::_varlen_attn_backward", mutates_args={})
def _varlen_attn_backward(
    grad_out: tensorplay.Tensor,
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    out: tensorplay.Tensor,
    lse: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    rng_state: tensorplay.Tensor,
    scale: float | None = None,
    window_size: list[int] | None = None,
    backend: int = _FLASH_ATTENTION_BACKEND,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
    window_size = _normalize_window_size(window_size)

    unused = tensorplay.empty(0, device=query.device)

    if backend == _CUDNN_ATTENTION_BACKEND:
        log.info("Using cuDNN backend for varlen_attn")
        dq, dk, dv = tensorplay.ops.aten._cudnn_attention_backward(
            grad_out=grad_out,
            query=query,
            key=key,
            value=value,
            out=out,
            logsumexp=lse,
            cum_seq_q=cu_seq_q,
            cum_seq_k=cu_seq_k,
            max_q=max_q,
            max_k=max_k,
            dropout_p=0.0,
            philox_seed=rng_state,
            philox_offset=rng_state,  # should be unused
            attn_bias=None,
            is_causal=is_causal,
            scale=scale,
        )
    elif backend == _FLASH_ATTENTION_BACKEND:
        log.info("Using Flash Attention backend for varlen_attn")
        dq, dk, dv = tensorplay.ops.aten._flash_attention_backward(
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
            0.0,
            is_causal,
            rng_state,
            unused,
            scale=scale,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
        )
    else:
        raise AssertionError(f"Unsupported varlen attention backend: {backend}")
    return dq, dk, dv


@_varlen_attn_backward.register_fake
def _varlen_attn_backward_fake(
    grad_out: tensorplay.Tensor,
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    out: tensorplay.Tensor,
    lse: tensorplay.Tensor,
    cu_seq_q: tensorplay.Tensor,
    cu_seq_k: tensorplay.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    rng_state: tensorplay.Tensor,
    scale: float | None = None,
    window_size: list[int] | None = None,
    backend: int = _FLASH_ATTENTION_BACKEND,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
    """
    Fake implementation for meta tensor computation and tracing.
    """
    window_size = _normalize_window_size(window_size)

    grad_query = tensorplay.empty_like(query)
    grad_key = tensorplay.empty_like(key)
    grad_value = tensorplay.empty_like(value)

    return grad_query, grad_key, grad_value


def _backward(
    ctx: Any, grad_out: tensorplay.Tensor, grad_lse: tensorplay.Tensor, grad_rng: tensorplay.Tensor
) -> tuple[tensorplay.Tensor | None, ...]:
    query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state = ctx.saved_tensors

    max_q = ctx.max_q
    max_k = ctx.max_k
    is_causal = ctx.is_causal
    scale = ctx.scale
    window_size = ctx.window_size

    dq, dk, dv = tensorplay.ops.tensorplay_attn._varlen_attn_backward(
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
        is_causal,
        rng_state,
        scale,
        window_size,
        ctx.backend,
    )
    # cu_seq_q, cu_seq_k, max_q, max_k, is_causal, scale, window_size, \
    # enable_gqa, seqused_k, block_table, num_splits, backend
    num_params = 12
    return (dq, dk, dv, *((None,) * num_params))


_varlen_attn.register_autograd(_backward, setup_context=_setup_context)

tensorplay._dynamo.disallow_in_graph(
    tensorplay.ops.aten._flash_attention_forward_no_dropout_inplace
)

from tensorplay.utils.flop_counter import (
    _varlen_attn_backward_flop,
    _varlen_attn_forward_flop,
    _varlen_attn_out_flop,
    flop_registry,
)


flop_registry[tensorplay.ops.tensorplay_attn._varlen_attn] = _varlen_attn_forward_flop
flop_registry[tensorplay.ops.tensorplay_attn._varlen_attn_out] = _varlen_attn_out_flop
flop_registry[tensorplay.ops.tensorplay_attn._varlen_attn_backward] = _varlen_attn_backward_flop
