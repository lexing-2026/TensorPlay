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
    disable_functional_mode,
    disable_proxy_modes_tracing,
    is_fake_tensor,
    register_fake,
    suspend_functionalization,
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
    """Strides for the output tensor to match the memory layout of the query."""
    fill_order = sorted(
        range(len(query_strides)), key=lambda i: query_strides[i], reverse=True
    )
    expected_strides = _construct_strides(out.shape, fill_order)

    out = out.as_strided(out.shape, expected_strides)

    return out


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
    from tensorplay._dynamo._trace_wrapped_higher_order_op import (
        TransformGetItemToIndex,
    )
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
    G = query.size(1) // key.size(1)
    value = tensorplay.repeat_interleave(value, G, dim=1)
    key = tensorplay.repeat_interleave(key, G, dim=1)

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

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
            return tensorplay.empty_strided(
                t.size(),
                t.stride(),
                device=t.device,
                dtype=t.dtype,
                requires_grad=t.requires_grad,
            )
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
            materialize_grads=False,
        )
        score_grad = joint_grads[0]
        if isinstance(diff_tensors[0][1], Tensor) and diff_tensors[0][0] == 0 and score_grad is None:
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
            materialize_grads=False,
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
        ctx.save_for_backward(
            query,
            key,
            value,
            out,
            logsumexp,
            max_scores,
            *block_mask[:-1],
            *score_mod_other_buffers,
            *mask_mod_other_buffers,
        )
        return out, logsumexp, max_scores

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_out: Tensor,
        grad_logsumexp: Tensor,
        grad_max_scores: Tensor,
    ) -> tuple[Tensor | None, ...]:
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
        ) = ctx.saved_tensors
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
        return grad_query, grad_key, grad_value, *none_grads, *grad_score_mod_captured


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
    from tensorplay._dynamo._trace_wrapped_higher_order_op import (
        TransformGetItemToIndex,
    )

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


def _new_captured_grad_buffers(
    score_mod_other_buffers: tuple,
    joint_graph: Callable,
) -> tuple[Tensor | None, ...]:
    """Allocate one grad buffer per captured tensor buffer.

    The joint graph is evaluated eagerly, so the buffer shapes are read
    straight from the buffers themselves.
    """
    return tuple(
        tensorplay.empty_like(buffer) if isinstance(buffer, Tensor) else None
        for buffer in score_mod_other_buffers
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
    from tensorplay._dynamo._trace_wrapped_higher_order_op import (
        TransformGetItemToIndex,
    )
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

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

    G = query.size(1) // key.size(1)
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
            materialize_grads=False,
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


@omni_attention_backward.py_impl("Autograd")
def omni_attention_backward_autograd(
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
    # The backward runs below autograd: its inputs are already gradients, and
    # the composite formula records no further history.
    with tensorplay.no_grad():
        return sdpa_dense_backward(
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
    B, Hq, seq_len_q, _ = query.shape
    seq_len_kv = value.shape[-2]

    out = tensorplay.empty(
        (B, Hq, seq_len_q, value.shape[-1]),
        dtype=query.dtype,
        device=query.device,
    )
    lse = tensorplay.empty(
        (B, Hq, seq_len_q), dtype=tensorplay.float32, device=query.device
    )
    max_scores = tensorplay.empty(
        (B, Hq, seq_len_q), dtype=tensorplay.float32, device=query.device
    )
    return out, lse, max_scores


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

    if Bkv == 1 and Bq > 1:
        grad_key = tensorplay.sum(broadcasted_grad_key, dim=0, keepdim=True)
        grad_value = tensorplay.sum(broadcasted_grad_value, dim=0, keepdim=True)
    else:
        if Bq != Bkv:
            raise RuntimeError(
                "grad_key/grad_value batch must match key/value batch."
            )
        grad_key = broadcasted_grad_key
        grad_value = broadcasted_grad_value

    return grad_query, grad_key, grad_value, grad_score_mod_captured
