"""Higher-order operators for omni attention.

``omni_attention`` computes scaled dot product attention with a user-defined
score modification function; ``omni_attention_backward`` is its joint
backward counterpart.  Both are registry-based higher-order operators: the
composite eager implementation is the base registration, and autograd routes
through the :class:`OmniAttentionAutogradOp` formula.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Sequence
from typing import Any

import tensorplay
from tensorplay import Tensor
from tensorplay._higher_order_ops._hop_base import (
    FakeTensorMode,
    HigherOrderOperator,
    _AutoDispatchBelowAutograd,
    _ExcludeAutocastGuard,
    disable_functional_mode,
    disable_proxy_modes_tracing,
    is_fake_tensor,
    register_fake,
    suspend_functionalization,
)
from tensorplay._higher_order_ops.utils import (
    UnsupportedAliasMutationException,
    _has_potential_branch_input_mutation,
    _maybe_reenter_make_fx,
    autograd_not_implemented,
    has_user_subclass,
    redirect_to_mode,
    save_values_for_backward,
    saved_values,
    validate_subgraph_args_types,
)


def _construct_strides(
    sizes: Sequence[int],
    fill_order: Sequence[int],
) -> Sequence[int]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    # Initialize strides
    if len(sizes) != len(fill_order):
        raise AssertionError(
            f"Length of sizes must match the length of the fill order, got {len(sizes)} vs {len(fill_order)}"
        )
    strides = [0] * len(sizes)

    # Start with stride 1 for the innermost dimension
    current_stride = 1

    # Iterate through the fill order populating strides
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]

    return strides


def _permute_strides(out: Tensor, query_strides: tuple[int, ...]) -> Tensor:
    """
    Create a new tensor with the same data and shape as the input,
    but with strides permuted based on the input tensor's stride order.

    Args:
        out (Tensor): The output tensor of attention.
        query_strides (List[int]): The stride order of the input query tensor

    Returns:
        Tensor: A new tensor with same shape and data as the input,
        but with strides permuted based on the query tensor's stride order.
    """
    fill_order = sorted(range(len(query_strides)), key=lambda i: query_strides[i])
    if out.storage_offset() != 0:
        raise AssertionError(
            f"Only support storage_offset == 0, got {out.storage_offset()}"
        )
    out_strides = list(_construct_strides(out.shape, fill_order))

    # Attention kernels require stride[-1]=1 for efficient memory access.
    # Ensure this by moving last dim to front of fill_order if needed.
    if out_strides[-1] != 1:
        last_dim = len(out.shape) - 1
        fill_order = list(fill_order)
        fill_order.remove(last_dim)
        fill_order = [last_dim] + fill_order
        out_strides = _construct_strides(out.shape, fill_order)

    new_out = tensorplay.empty_strided(
        out.shape, out_strides, dtype=out.dtype, device=out.device
    )
    new_out.copy_(out)
    return new_out


class OmniAttentionHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("omni_attention", cacheable=True)

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        score_mod: Callable,
        block_mask: tuple,
        scale: float,
        kernel_options: dict[str, Any],
        score_mod_other_buffers: tuple = (),
        mask_mod_other_buffers: tuple = (),
    ) -> tuple[Tensor, Tensor, Tensor]:
        validate_subgraph_args_types(score_mod_other_buffers + mask_mod_other_buffers)
        return super().__call__(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )


omni_attention = OmniAttentionHOP()


class OmniAttentionBackwardHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("omni_attention_backward", cacheable=True)

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        out: Tensor,
        logsumexp: Tensor,
        grad_out: Tensor | None,
        grad_logsumexp: Tensor | None,
        fw_graph: Callable,
        joint_graph: Callable,
        block_mask: tuple,
        scale: float,
        kernel_options: dict[str, Any],
        score_mod_other_buffers: tuple = (),
        mask_mod_other_buffers: tuple = (),
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor | None, ...]]:
        validate_subgraph_args_types(score_mod_other_buffers + mask_mod_other_buffers)
        return super().__call__(
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
        )


omni_attention_backward = OmniAttentionBackwardHOP()


_BLOCK_SPARSE_MAX_BLOCK_SIZE = 1 << 20


def _vmap_for_qk_block(
    fn: Callable[..., Any],
    num_buffers: int,
    prefix: tuple[int | None, ...] = (),
) -> Callable[..., Any]:
    """Vectorize a modifier over one score block's rows and columns.

    The wrapped function receives the modifier's signature with a
    two-dimensional score block (for score modifiers, ``prefix=(0,)``) or
    without one (for mask modifiers, ``prefix=()``), plus one-dimensional
    ``q_idx``/``kv_idx`` position vectors; ``b``/``h`` pass through unmapped
    so a single batch/head pair can be processed at a time.  The modifier
    sees zero-dimensional score and index values, exactly like the dense
    evaluator's per-element view.
    """
    suffix = (None,) * num_buffers
    # Innermost: iterate the columns of one row slice alongside kv_idx.
    fn = tensorplay.func.vmap(fn, in_dims=prefix + (None, None, None, 0) + suffix)
    # Outermost: iterate the block's rows alongside q_idx.
    fn = tensorplay.func.vmap(fn, in_dims=prefix + (None, None, 0, None) + suffix)
    return fn


def _use_block_sparse_math(query: Tensor, key: Tensor, block_mask: tuple) -> bool:
    """Whether eager execution may iterate present KV blocks only.

    Needs a genuine block-sparse mask (the catch-all mask uses a giant block
    covering the whole sequence, where the dense evaluator is cheaper), a
    layout matching this call's shapes exactly, and no active graph capture
    (the block walk reads data-dependent lengths).
    """
    if len(block_mask) < 17:
        return False
    kv_num_blocks = block_mask[2]
    q_block_size, kv_block_size = block_mask[14], block_mask[15]
    if kv_num_blocks is None or not (0 < int(q_block_size) < _BLOCK_SPARSE_MAX_BLOCK_SIZE):
        return False
    from tensorplay.graph.experimental.proxy_tensor import get_proxy_mode

    if get_proxy_mode() is not None:
        return False
    B, H, M, _ = query.shape
    N = key.shape[2]
    if M % int(q_block_size) or N % int(kv_block_size):
        return False
    if M != int(block_mask[0]) or N != int(block_mask[1]):
        return False
    if tuple(kv_num_blocks.shape[:2]) != (B, H):
        return False
    return True


def _block_sparse_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor]:
    """Block-sparse eager evaluation of the same math the dense path computes.

    Only KV blocks listed in ``kv_indices`` (partially unmasked) and
    ``full_kv_indices`` (fully unmasked) are visited, and query blocks absent
    from ``q_indices`` are skipped outright; each visited block feeds a
    running max / sum-of-exponentials accumulator (online softmax), so the
    full ``[B, H, M, N]`` score matrix is never materialized.  Returns the
    same triple as the dense evaluator under the log2-space statistics
    contract.
    """
    from tensorplay._higher_order_ops.utils import TransformGetItemToIndex

    kv_num_blocks, kv_indices = block_mask[2], block_mask[3]
    full_kv_num_blocks, full_kv_indices = block_mask[4], block_mask[5]
    q_block_size = int(block_mask[14])
    kv_block_size = int(block_mask[15])

    G = query.shape[1] // key.shape[1]
    value = tensorplay.repeat_interleave(value, G, dim=1)
    key = tensorplay.repeat_interleave(key, G, dim=1)

    Bq, Bkv = query.shape[0], key.shape[0]
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")
    key = key.expand((Bq, *key.shape[1:]))
    value = value.expand((Bq, *value.shape[1:]))

    B, H, M, _ = query.shape
    N, Dv = value.shape[2], value.shape[3]

    working = tensorplay.float64 if query.dtype == tensorplay.float64 else tensorplay.float32
    q = query.to(working)
    k = key.to(working)
    v = value.to(working)

    vmapped_score_mod = _vmap_for_qk_block(
        score_mod, len(score_mod_other_buffers), prefix=(0,)
    )
    vmapped_mask_mod = _vmap_for_qk_block(
        block_mask[-1], len(mask_mod_other_buffers), prefix=()
    )

    out = tensorplay.zeros((B, H, M, Dv), dtype=working, device=query.device)
    running_max = tensorplay.full((B, H, M), -float("inf"), dtype=working, device=query.device)
    running_sum = tensorplay.zeros((B, H, M), dtype=working, device=query.device)
    neg_inf = tensorplay.tensor(-float("inf"), dtype=working, device=query.device)

    num_q_blocks = M // q_block_size

    for b in range(B):
        for h in range(H):
            for qi in range(num_q_blocks):
                n_partial = int(kv_num_blocks[b, h, qi].item())
                n_full = (
                    int(full_kv_num_blocks[b, h, qi].item())
                    if full_kv_num_blocks is not None
                    else 0
                )
                if n_partial + n_full == 0:
                    # no unmasked key position anywhere in this query block:
                    # every row stays at the zero/-inf initial state
                    continue
                q_blk = q[b, h, qi * q_block_size:(qi + 1) * q_block_size]
                q_idx = tensorplay.arange(
                    qi * q_block_size, (qi + 1) * q_block_size, device=query.device
                )
                m_run = tensorplay.full(
                    (q_block_size,), -float("inf"), dtype=working, device=query.device
                )
                l_run = tensorplay.zeros(
                    (q_block_size,), dtype=working, device=query.device
                )
                acc = tensorplay.zeros(
                    (q_block_size, Dv), dtype=working, device=query.device
                )

                block_ids: list[tuple[int, bool]] = [
                    (int(kv_indices[b, h, qi, t].item()), False)
                    for t in range(n_partial)
                ]
                block_ids.extend(
                    (int(full_kv_indices[b, h, qi, t].item()), True)
                    for t in range(n_full)
                )

                b_scalar = tensorplay.tensor(b, device=query.device)
                h_scalar = tensorplay.tensor(h, device=query.device)

                for kv_blk, is_full in block_ids:
                    k_blk = k[b, h, kv_blk * kv_block_size:(kv_blk + 1) * kv_block_size]
                    v_blk = v[b, h, kv_blk * kv_block_size:(kv_blk + 1) * kv_block_size]
                    scores = (q_blk @ k_blk.transpose(-2, -1)) * scale
                    kv_idx = tensorplay.arange(
                        kv_blk * kv_block_size,
                        (kv_blk + 1) * kv_block_size,
                        device=query.device,
                    )
                    with TransformGetItemToIndex():
                        smod = vmapped_score_mod(
                            scores, b_scalar, h_scalar, q_idx, kv_idx,
                            *score_mod_other_buffers,
                        )
                        if is_full:
                            post = smod
                        else:
                            keep = vmapped_mask_mod(
                                b_scalar, h_scalar, q_idx, kv_idx,
                                *mask_mod_other_buffers,
                            )
                            post = tensorplay.where(keep, smod, neg_inf)

                    block_max = tensorplay.max(post, dim=-1)[0]
                    m_new = tensorplay.maximum(m_run, block_max)
                    finite = m_new != -float("inf")
                    m_safe = tensorplay.where(
                        finite, m_new, tensorplay.zeros_like(m_new)
                    )
                    alpha = tensorplay.exp(m_run - m_safe)
                    probs = tensorplay.exp(post - m_safe[:, None])
                    l_run = alpha * l_run + tensorplay.sum(probs, dim=-1)
                    acc = alpha[:, None] * acc + probs @ v_blk
                    m_run = m_new

                lo = qi * q_block_size
                hi = lo + q_block_size
                nonzero = l_run > 0
                safe_l = tensorplay.where(nonzero, l_run, tensorplay.ones_like(l_run))
                block_out = tensorplay.where(
                    nonzero[:, None], acc / safe_l[:, None], tensorplay.zeros_like(acc)
                )
                out[b, h, lo:hi] = block_out
                running_max[b, h, lo:hi] = m_run
                running_sum[b, h, lo:hi] = l_run

    nonzero = running_sum > 0
    safe_l = tensorplay.where(nonzero, running_sum, tensorplay.ones_like(running_sum))
    lse = tensorplay.where(
        nonzero,
        running_max + tensorplay.log(safe_l),
        tensorplay.full_like(running_max, -float("inf")),
    )

    return (
        out.to(query.dtype),
        lse / math.log(2),
        running_max / math.log(2),
    )


def _math_attention_inner(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor]:
    from tensorplay._higher_order_ops.utils import TransformGetItemToIndex
    from tensorplay.nn.attention.omni_attention import _vmap_for_bhqkv

    working_precision = tensorplay.float64 if query.dtype == tensorplay.float64 else tensorplay.float32

    scores = query.to(working_precision) @ key.to(working_precision).transpose(-2, -1)

    b = tensorplay.arange(0, scores.size(0), device=scores.device)
    h = tensorplay.arange(0, scores.size(1), device=scores.device)
    m = tensorplay.arange(0, scores.size(2), device=scores.device)
    n = tensorplay.arange(0, scores.size(3), device=scores.device)

    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)

    # first input is score
    score_mod = _vmap_for_bhqkv(score_mod, prefix=(0,), suffix=captured_buffers_in_dim)

    mask_mod = block_mask[-1]
    mask_mod_in_dim_buffers = (None,) * len(mask_mod_other_buffers)
    mask_mod = _vmap_for_bhqkv(mask_mod, prefix=(), suffix=mask_mod_in_dim_buffers)

    with TransformGetItemToIndex():
        scores = (scores * scale).to(working_precision)
        post_mod_scores = tensorplay.where(
            mask_mod(b, h, m, n, *mask_mod_other_buffers),
            score_mod(scores, b, h, m, n, *score_mod_other_buffers),
            tensorplay.tensor(-float("inf"), dtype=working_precision, device=scores.device),
        )

    return scores, post_mod_scores


def math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor]:
    """Eager implementation

    This implementation uses vmap to vectorize the score_mod function over the batch, head, m, and n dimensions.
    We then apply the vectorized score_mod function to the scores matrix. Each wrap of vmap applies one of the
    batch, head, m, or n dimensions. We need to apply vmap 4 times to vectorized over all 4 dimensions.

    Args:
        query: The query tensor
        key: The key tensor
        value: The value tensor
        score_mod: The score_mod function
        other_buffers: Other buffers that are passed to the score_mod function

    Notes:
        Query and Keys are dtype cast up to float64 (if query.dtype is float64) and float32 otherwise.
        Scores and Values are dtype cast to input query.dtype at the end.
    """
    # broadcast query & key along head dim for GQA
    # ``.shape`` reads resolve concretely under capture (metadata is part of
    # the compile signature); the ``.size(i)`` method form stays symbolic and
    # cannot feed Python arithmetic or unpacking.
    G = query.shape[1] // key.shape[1]
    value = tensorplay.repeat_interleave(value, G, dim=1)
    key = tensorplay.repeat_interleave(key, G, dim=1)

    Bq, Bkv = query.shape[0], key.shape[0]
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.shape[1:]))
    value = value.expand((Bq, *value.shape[1:]))

    _, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )

    # Set fully masked rows' sumexp to 0.0
    logsumexp = post_mod_scores.logsumexp(dim=-1)
    masked_rows = tensorplay.all(post_mod_scores == -float("inf"), dim=-1)
    logsumexp = tensorplay.where(masked_rows, -float("inf"), logsumexp)

    # working precision will be used so no need to cast to fp32
    max_scores = tensorplay.max(post_mod_scores, dim=-1)[0]

    post_mod_scores = tensorplay._safe_softmax(post_mod_scores, dim=-1)

    # NB: kernel computes in ln2 space, we always convert back at the top level op, so
    # for math impl we divide by log(2) because we will multiply by log(2)

    return (
        post_mod_scores.to(query.dtype) @ value.to(query.dtype),
        logsumexp / math.log(2),
        max_scores / math.log(2),
    )


def _omni_attention_autocast_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple,
    mask_mod_other_buffers: tuple,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Forward-only autocast shim: cast Q/K/V to the active autocast dtype, then
    redispatch with autocast routing excluded so we hit the normal implementation.
    """
    from tensorplay.amp.autocast_mode import _cast as _autocast_cast

    device_type = query.device.type
    autocast_dtype = tensorplay.get_autocast_dtype(device_type)

    query = _autocast_cast(query, device_type, autocast_dtype)
    key = _autocast_cast(key, device_type, autocast_dtype)
    value = _autocast_cast(value, device_type, autocast_dtype)

    with _ExcludeAutocastGuard():
        return omni_attention(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )


@omni_attention.py_impl("AutocastCUDA")
def omni_attention_autocast_cuda(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor]:
    return _omni_attention_autocast_impl(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )


@omni_attention.py_impl("AutocastCPU")
def omni_attention_autocast_cpu(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor]:
    return _omni_attention_autocast_impl(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )


@omni_attention.py_impl("CompositeExplicitAutograd")
def sdpa_dense(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor]:
    if _use_block_sparse_math(query, key, block_mask):
        out, lse, max_scores = _block_sparse_attention(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
    else:
        out, lse, max_scores = math_attention(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
    out = _permute_strides(out, query.stride())
    return out, lse, max_scores


def trace_omni_attention(
    proxy_mode: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor]:
    """Traces the omni_attention operator with the given score_mod function and other_buffers.

    Trace SDPA will call make_fx with "fake" example vals and then trace the score_mod function
    This will produce a GraphModule that will be stored on the root tracer as "sdpa_score". We
    access this graph module in the compiler to inline the score_mod function to the template.
    """
    from contextlib import nullcontext

    from tensorplay._higher_order_ops.utils import TransformGetItemToIndex
    from tensorplay.graph.experimental.proxy_tensor import (
        track_tensor_tree,
        unwrap_proxy,
    )

    # The mode must not intercept its own implementation body: evaluate the
    # example output with the capture suspended, then trace the subgraphs.
    with disable_proxy_modes_tracing():
        example_out = omni_attention(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
    example_vals = [query.new_zeros((), requires_grad=query.requires_grad)] + [
        query.new_zeros((), dtype=tensorplay.int64) for _ in range(4)
    ]
    mask_example_vals = [query.new_zeros((), dtype=tensorplay.int64) for _ in range(4)]
    mask_mod = block_mask[-1]
    with TransformGetItemToIndex():
        score_graph = reenter_make_fx(score_mod)(
            *example_vals, *score_mod_other_buffers
        )
        mask_graph = reenter_make_fx(mask_mod)(
            *mask_example_vals, *mask_mod_other_buffers
        )
    block_mask = block_mask[:-1] + (mask_graph,)
    node_args = (
        query,
        key,
        value,
        score_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    proxy_args = unwrap_proxy(node_args)
    set_original_aten_op = nullcontext
    with set_original_aten_op():
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", omni_attention, proxy_args, {}
        )
    return track_tensor_tree(
        example_out,
        out_proxy,
        constant=None,
        tracer=proxy_mode.tracer,
    )


@omni_attention.py_impl("ProxyDispatchMode")
def omni_attention_proxy_dispatch_mode(
    mode: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor]:
    if mode is None:
        raise AssertionError("Mode should always be enabled for python fallback key")
    return trace_omni_attention(
        mode,
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )


class _IdentityFunctionalizeCtx:
    """Default functionalization context.

    Tensor wrappers do not exist in this build, so unwrapping and wrapping are
    identities and redispatch continues at the same operator.
    """

    mode = None

    def unwrap_tensors(self, x: Any) -> Any:
        return x

    def wrap_tensors(self, x: Any) -> Any:
        return x

    def redispatch_to_next(self):
        import contextlib

        return contextlib.nullcontext()

    def functionalize(self, fn: Callable) -> Callable:
        return fn


@omni_attention.py_functionalize_impl
def omni_attention_functionalize(
    ctx: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor]:
    """Defines the functionalization rules for the omni_attention operator.

    Right now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations in the score_mod function, to the other_buffers since those
    are free variables.
    """
    from tensorplay._higher_order_ops.utils import TransformGetItemToIndex

    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    block_mask_unwrapped = ctx.unwrap_tensors(block_mask)
    score_mod_other_buffers_unwrapped = ctx.unwrap_tensors(score_mod_other_buffers)
    mask_mod_other_buffers_unwrapped = ctx.unwrap_tensors(mask_mod_other_buffers)

    # Appease the mypy overlords
    if not isinstance(query_unwrapped, Tensor):
        raise AssertionError(
            f"expected query_unwrapped to be Tensor, got {type(query_unwrapped)}"
        )
    if not isinstance(key_unwrapped, Tensor):
        raise AssertionError(
            f"expected key_unwrapped to be Tensor, got {type(key_unwrapped)}"
        )
    if not isinstance(value_unwrapped, Tensor):
        raise AssertionError(
            f"expected value_unwrapped to be Tensor, got {type(value_unwrapped)}"
        )
    if not isinstance(block_mask_unwrapped, tuple):
        raise AssertionError(
            f"expected block_mask_unwrapped to be tuple, got {type(block_mask_unwrapped)}"
        )
    if not isinstance(score_mod_other_buffers_unwrapped, tuple):
        raise AssertionError(
            f"expected score_mod_other_buffers_unwrapped to be tuple, got {type(score_mod_other_buffers_unwrapped)}"
        )
    if not isinstance(mask_mod_other_buffers_unwrapped, tuple):
        raise AssertionError(
            f"expected mask_mod_other_buffers_unwrapped to be tuple, got {type(mask_mod_other_buffers_unwrapped)}"
        )

    example_vals = (
        [query_unwrapped.new_zeros(())]
        + [query_unwrapped.new_zeros((), dtype=tensorplay.int64) for _ in range(4)]
        + list(score_mod_other_buffers_unwrapped)
    )
    with ctx.redispatch_to_next():
        functional_score_mod = ctx.functionalize(score_mod)
        pre_dispatch = getattr(ctx, "mode", None) is not None and ctx.mode.pre_dispatch
        with TransformGetItemToIndex():
            # TODO: So far only the input mutations are checked
            # In the other HOPs, also aliases are checked which is
            # omitted here
            mutates = _has_potential_branch_input_mutation(
                score_mod, example_vals, pre_dispatch
            )
        # We only care about mutations of existing buffers since we can't replay these.
        # However, we can just error if anything is detected
        if mutates:
            raise UnsupportedAliasMutationException("Mutations detected in score_mod")

        out = omni_attention(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            functional_score_mod,
            block_mask_unwrapped,
            scale,
            kernel_options,
            score_mod_other_buffers_unwrapped,
            mask_mod_other_buffers_unwrapped,
        )
    return ctx.wrap_tensors(out)


def create_fw_bw_graph(
    score_mod: Callable,
    index_values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    other_buffers: tuple[Tensor, ...],
) -> tuple[Callable, Callable]:
    """Build the forward graph and the joint gradient graph for score_mod.

    The forward graph is the score_mod itself.  The joint graph is derived
    eagerly: one autograd pass of score_mod on detached stand-in inputs
    verifies the output depends on score, and the returned joint evaluates
    the vector-Jacobian product for a given cotangent.
    """

    def _from_fun(t: Tensor | int) -> Tensor | int:
        if isinstance(t, Tensor):
            stand_in = tensorplay.empty_strided(
                t.size(),
                t.stride(),
                device=t.device,
                dtype=t.dtype,
            )
            # The joint verification below evaluates the score modifier
            # eagerly on these stand-ins, so index stand-ins must hold
            # in-range values (user code indexes buffers with them);
            # zero is always in range for a non-empty tensor.
            with tensorplay.no_grad():
                stand_in.zero_()
            if t.requires_grad:
                stand_in.requires_grad_(True)
            return stand_in
        return t

    unwrapped_score_mod_indexes = tuple(_from_fun(t) for t in index_values)
    unwrapped_other_buffers = tuple(_from_fun(t) for t in other_buffers)

    with tensorplay.enable_grad():
        diff_tensors = [
            t
            for t in unwrapped_score_mod_indexes + unwrapped_other_buffers
            if isinstance(t, Tensor) and t.requires_grad
        ]
        if not diff_tensors:
            raise RuntimeError(
                "omni_attention backward requires at least one differentiable input "
                "to score_mod (the score or a captured buffer)."
            )
        example_flat_out = score_mod(
            *unwrapped_score_mod_indexes, *unwrapped_other_buffers
        )
        if not isinstance(example_flat_out, Tensor):
            raise RuntimeError(
                "Expected output of score_mod to be a tensor."
                f"Got type {type(example_flat_out)}."
            )
        example_grad = _from_fun(example_flat_out)
        joint_grads = tensorplay.autograd.grad(
            example_flat_out,
            diff_tensors,
            allow_unused=True,
        )
        score_stand_in = unwrapped_score_mod_indexes[0]
        if (
            isinstance(score_stand_in, Tensor)
            and score_stand_in.requires_grad
            and score_stand_in is diff_tensors[0]
            and joint_grads[0] is None
        ):
            raise RuntimeError(
                "omni_attention backward requires the output of score_mod to "
                "depend on score. Got a score_mod whose output does not "
                "require gradients with respect to score."
            )

    def joint_f(
        score: Tensor,
        b: Tensor,
        h: Tensor,
        m: Tensor,
        n: Tensor,
        cotangent: Tensor,
        *other_buffers: tuple[Tensor, ...],
    ) -> list[Tensor | None]:
        args = [score, b, h, m, n, *other_buffers]
        diff_tensors = [
            (i, t) for i, t in enumerate(args) if isinstance(t, Tensor) and t.requires_grad
        ]
        if not diff_tensors:
            raise RuntimeError(
                "omni_attention backward requires the output of score_mod to "
                "depend on score. Got a score_mod whose output does not "
                "require gradients with respect to score."
            )
        with tensorplay.enable_grad():
            fw_out = score_mod(*args[:5], *args[5:])
        grads = tensorplay.autograd.grad(
            fw_out,
            [t for _, t in diff_tensors],
            grad_outputs=cotangent if cotangent.requires_grad or cotangent.numel() else None,
            allow_unused=True,
        )
        grad_by_index = {i: g for (i, _), g in zip(diff_tensors, grads)}
        if grad_by_index.get(0) is None:
            raise RuntimeError(
                "omni_attention backward requires the output of score_mod to "
                "depend on score. Got a score_mod whose output does not "
                "require gradients with respect to score."
            )
        return [grad_by_index.get(i) for i in range(len(args))]

    return score_mod, joint_f


class OmniAttentionAutogradOp(tensorplay.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        fw_graph: Callable,
        joint_graph: Callable,
        block_mask: tuple[Any, ...],
        scale: float,
        kernel_options: dict[str, Any],
        mask_mod_other_buffers: tuple[Any, ...],
        *score_mod_other_buffers: tuple[Any, ...],
    ) -> tuple[Tensor, Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        from tensorplay.graph.traceback import current_meta

        # Capture sparsity_hint from tracing metadata so backward can re-annotate
        ctx._sparsity_hint = current_meta.get("custom", {}).get("sparsity_hint", 0.0)
        any_buffer_requires_grad = any(
            buffer.requires_grad
            for buffer in mask_mod_other_buffers
            if isinstance(buffer, Tensor)
        )
        if any_buffer_requires_grad:
            raise AssertionError(
                "Captured buffers from mask mod that require grad are not supported."
            )
        ctx._fw_graph = fw_graph
        ctx._joint_graph = joint_graph
        ctx._mask_graph = block_mask[-1]
        ctx.scale = scale
        ctx.kernel_options = kernel_options
        ctx._score_mod_other_buffers_len = len(score_mod_other_buffers)
        with _AutoDispatchBelowAutograd():
            out, logsumexp, max_scores = omni_attention(
                query,
                key,
                value,
                fw_graph,
                block_mask,
                scale,
                kernel_options,
                score_mod_other_buffers,
                mask_mod_other_buffers,
            )
        # no grads for you sir
        ctx.mark_non_differentiable(max_scores)
        save_values_for_backward(
            ctx,
            (
                query,
                key,
                value,
                out,
                logsumexp,
                max_scores,
                *block_mask[:-1],
                *score_mod_other_buffers,
                *mask_mod_other_buffers,
            ),
        )
        return out, logsumexp, max_scores

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_out: Tensor,
        grad_logsumexp: Tensor,
        grad_max_scores: Tensor,
    ) -> tuple[Tensor | None, ...]:
        fw_args = saved_values(ctx)
        (
            query,
            key,
            value,
            out,
            logsumexp,
            max_scores,
            query_lengths,
            kv_lengths,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            dq_write_order,
            dq_write_order_full,
            dq_kv_order,
            dq_kv_order_spt,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            *other_buffers,
        ) = fw_args
        fw_graph = ctx._fw_graph
        joint_graph = ctx._joint_graph
        mask_graph = ctx._mask_graph
        scale = ctx.scale
        kernel_options = ctx.kernel_options
        score_mod_other_buffers = tuple(
            other_buffers[: ctx._score_mod_other_buffers_len]
        )
        mask_mod_other_buffers = tuple(
            other_buffers[ctx._score_mod_other_buffers_len :]
        )
        # We have asserted that mask_mod_other_buffers do not require grad,
        # but score_mod_other_buffers can require grad.
        none_grads = [None] * 6
        from tensorplay.graph.traceback import annotate

        _sparsity_ctx = (
            annotate({"sparsity_hint": ctx._sparsity_hint})
            if ctx._sparsity_hint > 0
            else contextlib.nullcontext()
        )
        with _sparsity_ctx:
            (
                grad_query,
                grad_key,
                grad_value,
                grad_score_mod_captured,
            ) = omni_attention_backward(
                query,
                key,
                value,
                out,
                logsumexp,
                grad_out,
                grad_logsumexp,
                fw_graph,
                joint_graph,
                (
                    query_lengths,
                    kv_lengths,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                    q_num_blocks,
                    q_indices,
                    full_q_num_blocks,
                    full_q_indices,
                    dq_write_order,
                    dq_write_order_full,
                    dq_kv_order,
                    dq_kv_order_spt,
                    Q_BLOCK_SIZE,
                    KV_BLOCK_SIZE,
                    mask_graph,
                ),
                scale,
                kernel_options,
                score_mod_other_buffers,
                mask_mod_other_buffers,
            )
        # The engine tracks one gradient slot per flattened input position
        # (the block-mask tuple and the kernel-options mapping expand into
        # their elements), so pad with None up to that count; the
        # captured-buffer gradients occupy the trailing slots because the
        # captured buffers are the trailing arguments of the forward call.
        n_in = len(ctx.needs_input_grad)
        n_tail = len(grad_score_mod_captured)
        n_none = max(n_in - 3 - n_tail, 0)
        return (
            grad_query,
            grad_key,
            grad_value,
            *((None,) * n_none),
            *grad_score_mod_captured,
        )


@omni_attention.py_impl("Autograd")
def omni_attention_autograd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple[Tensor, ...] = (),
    mask_mod_other_buffers: tuple[Tensor, ...] = (),
) -> tuple[Tensor, Tensor, Tensor]:
    from tensorplay._higher_order_ops.utils import TransformGetItemToIndex

    with TransformGetItemToIndex():
        input_requires_grad = any(
            isinstance(t, Tensor) and t.requires_grad
            for t in (query, key, value, *score_mod_other_buffers)
        )
        if tensorplay.is_grad_enabled() and input_requires_grad:
            if block_mask[7] is None:
                raise RuntimeError(
                    "BlockMask q_indices is None. Backward pass requires q_indices to be computed. "
                    "Please create the BlockMask with compute_q_blocks=True"
                )
            example_vals = (
                query.new_zeros((), requires_grad=input_requires_grad),
                query.new_zeros((), dtype=tensorplay.int64),
                query.new_zeros((), dtype=tensorplay.int64),
                query.new_zeros((), dtype=tensorplay.int64),
                query.new_zeros((), dtype=tensorplay.int64),
            )
            fw_graph, bw_graph = create_fw_bw_graph(
                score_mod, example_vals, score_mod_other_buffers
            )
        else:
            fw_graph, bw_graph = score_mod, None
        out, logsumexp, max_scores = OmniAttentionAutogradOp.apply(
            query,
            key,
            value,
            fw_graph,
            bw_graph,
            block_mask,
            scale,
            kernel_options,
            mask_mod_other_buffers,
            *score_mod_other_buffers,
        )
    return out, logsumexp, max_scores


def _captured_grad_buffer_mask(
    joint_graph: Callable, num_captures: int
) -> tuple[bool, ...]:
    """Return which score_mod captures need concrete grad buffers.

    A capture can require grad but still be dead in score_mod, for example when
    the function reads a closed-over tensor but returns the original score. A
    traced joint graph represents that as a None grad output, and that None
    must stay a None instead of being copied into a materialized buffer.

    Conversely, this is independent of the saved capture's requires_grad flag:
    compiled backward may save an intermediate whose grad is needed to propagate
    to earlier user inputs.
    """
    from tensorplay.graph.graph_module import GraphModule

    if not isinstance(joint_graph, GraphModule):
        # The @register_fake path for direct omni_attention_backward HOP calls
        # can see callable subgraphs before make_fx materializes them. In that
        # metadata-only phase, no capture is provably dead.
        return (True,) * num_captures

    output_node = next(node for node in joint_graph.graph.nodes if node.op == "output")
    score_mod_arg_grads = 5
    captured_grad_outputs = output_node.args[0][score_mod_arg_grads:]
    if len(captured_grad_outputs) != num_captures:
        raise AssertionError(
            f"Expected {num_captures} captured grad outputs, "
            f"got {len(captured_grad_outputs)}"
        )
    return tuple(grad is not None for grad in captured_grad_outputs)


def _empty_like_contiguous(buffer: Tensor) -> Tensor:
    """Allocate a contiguous-buffer-shaped tensor.

    The lowering returns captured grads as contiguous buffers, so the
    allocation sites match that layout explicitly.
    """
    strides = []
    acc = 1
    for size in reversed(tuple(int(s) for s in buffer.shape)):
        strides.append(acc)
        acc *= size
    strides.reverse()
    return tensorplay.empty_strided(
        buffer.shape, strides, dtype=buffer.dtype, device=buffer.device
    )


def _new_captured_grad_buffers(
    score_mod_other_buffers: tuple, joint_graph: Callable
) -> tuple[Tensor | None, ...]:
    """Allocate only the captured grad buffers that the joint graph can produce."""
    needs_grad_buffer = _captured_grad_buffer_mask(
        joint_graph, len(score_mod_other_buffers)
    )
    # The lowering returns captured grads as contiguous buffers, so match that here.
    return tuple(
        _empty_like_contiguous(buffer)
        if isinstance(buffer, Tensor) and needs_buffer
        else None
        for buffer, needs_buffer in zip(score_mod_other_buffers, needs_grad_buffer)
    )


@omni_attention_backward.py_impl("CompositeExplicitAutograd")
def sdpa_dense_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    grad_out: Tensor | None,
    grad_logsumexp: Tensor | None,
    fw_graph: Callable,
    joint_graph: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple,
    mask_mod_other_buffers: tuple,
) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor | None, ...]]:
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Backward pass with mixed query, key, and value dtype is not supported, "
            f"got query.dtype={query.dtype}, key.dtype={key.dtype}, "
            f"and value.dtype={value.dtype}"
        )
    if joint_graph is None:
        example_vals = (
            query.new_zeros((), requires_grad=True),
            query.new_zeros((), dtype=tensorplay.int64),
            query.new_zeros((), dtype=tensorplay.int64),
            query.new_zeros((), dtype=tensorplay.int64),
            query.new_zeros((), dtype=tensorplay.int64),
        )
        _, joint_graph = create_fw_bw_graph(
            fw_graph, example_vals, score_mod_other_buffers
        )
    from tensorplay._higher_order_ops.utils import TransformGetItemToIndex
    from tensorplay.nn.attention.omni_attention import _vmap_for_bhqkv

    Bq, Hq, seq_len_q, qk_head_dim = query.shape
    Bkv, Hkv, seq_len_kv, v_head_dim = value.shape

    # Get outputs before calling repeat interleave and permute to input stride orders
    actual_grad_query = query.new_empty((Bq, Hq, seq_len_q, qk_head_dim))
    actual_grad_query = _permute_strides(actual_grad_query, query.stride())

    actual_grad_key = key.new_empty((Bq, Hkv, seq_len_kv, qk_head_dim))
    actual_grad_key = _permute_strides(actual_grad_key, key.stride())

    actual_grad_value = value.new_empty((Bq, Hkv, seq_len_kv, v_head_dim))
    actual_grad_value = _permute_strides(actual_grad_value, value.stride())

    actual_grad_score_mod_captured = _new_captured_grad_buffers(
        score_mod_other_buffers, joint_graph
    )

    Bq, Bkv = query.shape[0], key.shape[0]
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.shape[1:]))
    value = value.expand((Bq, *value.shape[1:]))

    G = query.shape[1] // key.shape[1]
    key = tensorplay.repeat_interleave(key, G, dim=1)
    value = tensorplay.repeat_interleave(value, G, dim=1)

    if grad_out is None:
        grad_out = tensorplay.zeros_like(out)
    if grad_logsumexp is None:
        grad_logsumexp = tensorplay.zeros_like(logsumexp)

    # logsumexp is expected in log2 scale (as returned by the forward HOP).
    # The public omni_attention API converts lse to natural log before returning,
    # so callers using the public API must not pass that value here directly.
    logsumexp = logsumexp * math.log(2)
    # The backwards formula for the log -> log2 change of base in the forwards
    grad_logsumexp = grad_logsumexp / math.log(2)
    scores, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        fw_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    masked_out_rows = logsumexp == -float("inf")
    softmax_scores = tensorplay.exp(post_mod_scores - logsumexp.unsqueeze(-1))
    softmax_scores = tensorplay.where(masked_out_rows.unsqueeze(-1), 0, softmax_scores)

    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out

    grad_softmax_scores = grad_out.to(dtype=softmax_scores.dtype) @ value.to(
        dtype=softmax_scores.dtype
    ).transpose(-2, -1)

    sum_scores = tensorplay.sum(
        out.to(dtype=softmax_scores.dtype) * grad_out.to(dtype=softmax_scores.dtype),
        -1,
        keepdim=True,
    )
    grad_score_mod = softmax_scores * (
        grad_softmax_scores - sum_scores + grad_logsumexp.unsqueeze(-1)
    )

    b = tensorplay.arange(0, scores.size(0), device=scores.device)
    h = tensorplay.arange(0, scores.size(1), device=scores.device)
    m = tensorplay.arange(0, scores.size(2), device=scores.device)
    n = tensorplay.arange(0, scores.size(3), device=scores.device)

    mask_graph = block_mask[-1]
    # Gradient of the inline score_mod function, with respect to the scores.
    # The joint is evaluated with one autograd pass over the recomputed
    # post-modification scores: the chain mask -> score_mod gives the masked,
    # scaled gradient with respect to the raw scores in a single call.
    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    with tensorplay.enable_grad():
        scores_leaf = scores.detach().requires_grad_(True)
        buffer_leaves = tuple(
            buffer.detach().requires_grad_(buffer.requires_grad)
            if isinstance(buffer, Tensor) and buffer.requires_grad
            else buffer
            for buffer in score_mod_other_buffers
        )
        score_mod_leaf = _vmap_for_bhqkv(
            fw_graph, prefix=(0,), suffix=captured_buffers_in_dim
        )
        mask_mod = _vmap_for_bhqkv(
            mask_graph, prefix=(), suffix=(None,) * len(mask_mod_other_buffers)
        )
        with TransformGetItemToIndex():
            mask_scores = mask_mod(b, h, m, n, *mask_mod_other_buffers)
            post_leaf = tensorplay.where(
                mask_scores,
                score_mod_leaf(scores_leaf, b, h, m, n, *buffer_leaves),
                tensorplay.tensor(
                    -float("inf"), dtype=scores.dtype, device=scores.device
                ),
            )
        diff_tensors = [scores_leaf] + [
            buffer for buffer in buffer_leaves if isinstance(buffer, Tensor) and buffer.requires_grad
        ]
        joint_grads = tensorplay.autograd.grad(
            post_leaf,
            diff_tensors,
            grad_outputs=grad_score_mod,
            allow_unused=True,
        )
    grad_by_index = {id(t): g for t, g in zip(diff_tensors, joint_grads)}
    grad_scores = grad_by_index.get(id(scores_leaf))
    if grad_scores is None:
        raise RuntimeError(
            "omni_attention backward requires the output of score_mod to "
            "depend on score. Got a score_mod whose output does not "
            "require gradients with respect to score."
        )
    grad_scores = grad_scores.to(query.dtype)
    # The joint graph differentiates the post-modification scores; the
    # derivative of the scaled scores with respect to the raw query-key
    # products contributes one factor of scale.
    grad_scores = grad_scores * scale

    grad_query = grad_scores @ key
    grad_key = grad_scores.transpose(-2, -1) @ query

    # Reduce DK, DV along broadcasted heads.
    grad_key = grad_key.view(
        grad_key.size(0), -1, G, grad_key.size(-2), grad_key.size(-1)
    )
    grad_value = grad_value.view(
        grad_value.size(0), -1, G, grad_value.size(-2), grad_value.size(-1)
    )

    grad_key = tensorplay.sum(grad_key, 2, keepdim=False)
    grad_value = tensorplay.sum(grad_value, 2, keepdim=False)

    # Fill to correctly strided outputs
    actual_grad_query.copy_(grad_query)
    actual_grad_key.copy_(grad_key)
    actual_grad_value.copy_(grad_value)

    if Bq != Bkv:
        if not (Bq > 1 and Bkv == 1):
            raise AssertionError(
                f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}"
            )

        actual_grad_key = tensorplay.sum(actual_grad_key, 0, keepdim=True)
        actual_grad_value = tensorplay.sum(actual_grad_value, 0, keepdim=True)

    score_mod_other_buffer_grads = []
    for actual_grad, buffer in zip(
        actual_grad_score_mod_captured, buffer_leaves
    ):
        grad = grad_by_index.get(id(buffer)) if isinstance(buffer, Tensor) else None
        if not isinstance(grad, Tensor):
            score_mod_other_buffer_grads.append(None)
            continue
        if actual_grad is None:
            raise AssertionError("Expected a captured grad buffer for a tensor grad")
        score_mod_other_buffer_grads.append(actual_grad.copy_(grad))

    return (
        actual_grad_query,
        actual_grad_key,
        actual_grad_value,
        tuple(score_mod_other_buffer_grads),
    )


def trace_omni_attention_backward(
    proxy_mode: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    grad_out: Tensor,
    grad_logsumexp: Tensor,
    fw_graph: Callable,
    joint_graph: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor | None, ...]]:
    """We already have the forward graph and joint graph from the forward pass, so we create a proxy attach both graphs"""
    from contextlib import nullcontext

    from tensorplay._higher_order_ops.utils import TransformGetItemToIndex
    from tensorplay.graph.experimental.proxy_tensor import (
        track_tensor_tree,
        unwrap_proxy,
    )

    requires_grad = any(
        x.requires_grad for x in (query, key) if isinstance(x, Tensor)
    )
    fw_example_vals = [query.new_zeros((), requires_grad=requires_grad)] + [
        query.new_zeros((), dtype=tensorplay.int64) for _ in range(4)
    ]
    bw_example_vals = fw_example_vals + [query.new_zeros(())]
    mask_example_vals = [query.new_zeros((), dtype=tensorplay.int64) for _ in range(4)]
    mask_graph = block_mask[-1]
    with disable_proxy_modes_tracing():
        with TransformGetItemToIndex():
            # There's no active make_fx during the compiled autograd graph's initial capture
            fw_graph = _maybe_reenter_make_fx(fw_graph)(
                *fw_example_vals, *score_mod_other_buffers
            )
            joint_graph = _maybe_reenter_make_fx(joint_graph)(
                *bw_example_vals, *score_mod_other_buffers
            )
            mask_graph = _maybe_reenter_make_fx(mask_graph)(
                *mask_example_vals, *mask_mod_other_buffers
            )
    block_mask = block_mask[:-1] + (mask_graph,)
    with disable_proxy_modes_tracing():
        example_out = omni_attention_backward(
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
        )

    node_args = (
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
    )
    proxy_args = unwrap_proxy(node_args)
    set_original_aten_op = nullcontext
    with set_original_aten_op():
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function",
            omni_attention_backward,
            proxy_args,
            {},
            name="omni_attention_backward",
        )
    return track_tensor_tree(
        example_out,
        out_proxy,
        constant=None,
        tracer=proxy_mode.tracer,
    )


@omni_attention_backward.py_impl("ProxyDispatchMode")
def omni_attention_backward_proxy_dispatch_mode(
    mode: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    grad_out: Tensor,
    grad_logsumexp: Tensor,
    fw_graph: Callable,
    joint_graph: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor | None, ...]]:
    if mode is None:
        raise AssertionError("Mode should always be enabled for python fallback key")
    return trace_omni_attention_backward(
        mode,
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
    )


@omni_attention_backward.py_functionalize_impl
def omni_attention_backward_functionalize(
    ctx: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    grad_out: Tensor,
    grad_logsumexp: Tensor,
    fw_graph: Callable,
    joint_graph: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor | None, ...]]:
    """Defines the functionalization rules for the omni_attention operator.

    Right now we are unwrapping each tensor and then redispatching to the next,
    since we know that the forward score mod function is assured to be free of mutations
    to the other_buffers, we skip that mutate check and go straight to redispatching.
    """

    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    out_unwrapped = ctx.unwrap_tensors(out)
    logsumexp_unwrapped = ctx.unwrap_tensors(logsumexp)
    grad_out_unwrapped = ctx.unwrap_tensors(grad_out)
    grad_logsumexp_unwrapped = ctx.unwrap_tensors(grad_logsumexp)
    block_mask_unwrapped = ctx.unwrap_tensors(block_mask)
    score_mod_other_buffers_unwrapped = ctx.unwrap_tensors(score_mod_other_buffers)
    mask_mod_other_buffers_unwrapped = ctx.unwrap_tensors(mask_mod_other_buffers)

    # Appease the mypy overlords
    if not isinstance(query_unwrapped, Tensor):
        raise AssertionError(
            f"expected query_unwrapped to be Tensor, got {type(query_unwrapped)}"
        )
    if not isinstance(key_unwrapped, Tensor):
        raise AssertionError(
            f"expected key_unwrapped to be Tensor, got {type(key_unwrapped)}"
        )
    if not isinstance(value_unwrapped, Tensor):
        raise AssertionError(
            f"expected value_unwrapped to be Tensor, got {type(value_unwrapped)}"
        )
    if not isinstance(out_unwrapped, Tensor):
        raise AssertionError(
            f"expected out_unwrapped to be Tensor, got {type(out_unwrapped)}"
        )
    if not isinstance(logsumexp_unwrapped, Tensor):
        raise AssertionError(
            f"expected logsumexp_unwrapped to be Tensor, got {type(logsumexp_unwrapped)}"
        )
    if grad_out_unwrapped is not None and not isinstance(
        grad_out_unwrapped, Tensor
    ):
        raise AssertionError(
            f"expected grad_out_unwrapped to be Tensor or None, got {type(grad_out_unwrapped)}"
        )
    if grad_logsumexp_unwrapped is not None and not isinstance(
        grad_logsumexp_unwrapped, Tensor
    ):
        raise AssertionError(
            f"expected grad_logsumexp_unwrapped to be Tensor or None, got {type(grad_logsumexp_unwrapped)}"
        )
    if not isinstance(block_mask_unwrapped, tuple):
        raise AssertionError(
            f"expected block_mask_unwrapped to be tuple, got {type(block_mask_unwrapped)}"
        )
    if not isinstance(score_mod_other_buffers_unwrapped, tuple):
        raise AssertionError(
            f"expected score_mod_other_buffers_unwrapped to be tuple, got {type(score_mod_other_buffers_unwrapped)}"
        )
    if not isinstance(mask_mod_other_buffers_unwrapped, tuple):
        raise AssertionError(
            f"expected mask_mod_other_buffers_unwrapped to be tuple, got {type(mask_mod_other_buffers_unwrapped)}"
        )

    with ctx.redispatch_to_next():
        functional_fw_graph = ctx.functionalize(fw_graph)
        functional_joint_graph = ctx.functionalize(joint_graph)

        (
            grad_query,
            grad_key,
            grad_value,
            grad_score_mod_captured,
        ) = omni_attention_backward(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            out_unwrapped,
            logsumexp_unwrapped,
            grad_out_unwrapped,
            grad_logsumexp_unwrapped,
            functional_fw_graph,
            functional_joint_graph,
            block_mask_unwrapped,
            scale,
            kernel_options,
            score_mod_other_buffers_unwrapped,
            mask_mod_other_buffers_unwrapped,
        )

    return ctx.wrap_tensors((grad_query, grad_key, grad_value, grad_score_mod_captured))


omni_attention_backward.py_autograd_impl(
    autograd_not_implemented(omni_attention_backward, deferred_error=True)
)


@register_fake(omni_attention)
def omni_attention_fake_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor]:
    if has_user_subclass(
        (
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        ),
        allowed_subclasses=(),
    ):
        return NotImplemented

    v_head_dim = value.size(-1)
    batch_size, num_heads, seq_len_q, _q_head_dim = query.shape
    logsumexp = query.new_empty(batch_size, num_heads, seq_len_q, dtype=tensorplay.float32)
    max_scores = query.new_empty(batch_size, num_heads, seq_len_q, dtype=tensorplay.float32)
    out_shape = (batch_size, num_heads, seq_len_q, v_head_dim)
    out = query.new_empty(out_shape)
    out = _permute_strides(out, query.stride())
    return out, logsumexp, max_scores


@register_fake(omni_attention_backward)
def omni_attention_backward_fake_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    grad_out: Tensor,
    grad_logsumexp: Tensor,
    fw_graph: Callable,
    joint_graph: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor | None, ...]]:
    if has_user_subclass(
        (
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        ),
        allowed_subclasses=(),
    ):
        return NotImplemented
    Bq, _, _, qk_head_dim = query.shape
    Bkv, Hkv, seq_len_kv, v_head_dim = value.shape

    grad_query = query.new_empty(query.shape)
    grad_query = _permute_strides(grad_query, query.stride())

    grad_score_mod_captured = _new_captured_grad_buffers(
        score_mod_other_buffers, joint_graph
    )

    broadcasted_grad_key = key.new_empty((Bq, Hkv, seq_len_kv, qk_head_dim))
    broadcasted_grad_key = _permute_strides(broadcasted_grad_key, key.stride())

    broadcasted_grad_value = value.new_empty((Bq, Hkv, seq_len_kv, v_head_dim))
    broadcasted_grad_value = _permute_strides(broadcasted_grad_value, value.stride())

    from tensorplay.graph.experimental.symbolic_shapes import guard_or_false, sym_and

    # This branch chooses fake tensor metadata, including strides. Guarding is
    # valid for backed symbolic sizes, while unbacked/data-dependent predicates
    # conservatively fall through to the non-broadcast contract below.
    if guard_or_false(sym_and(Bkv == 1, Bq > 1)):
        grad_key = tensorplay.sum(broadcasted_grad_key, dim=0, keepdim=True)
        grad_value = tensorplay.sum(broadcasted_grad_value, dim=0, keepdim=True)
    else:
        tensorplay._check(
            Bq == Bkv,
            "grad_key/grad_value batch must match key/value batch.",
        )
        grad_key = broadcasted_grad_key
        grad_value = broadcasted_grad_value

    return grad_query, grad_key, grad_value, grad_score_mod_captured
